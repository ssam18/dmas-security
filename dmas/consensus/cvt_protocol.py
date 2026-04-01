"""
Consensus-based Threat Validation (CVT) Protocol.

Implements the 4-phase CVT protocol described in the paper:
  Phase 1 — Threat Detection & Alert  (VOTE_REQUEST broadcast)
  Phase 2 — Peer Evaluation           (weighted vote computation)
  Phase 3 — Vote Aggregation          (Θ_agg computation, Equation 2)
  Phase 4 — Consensus Decision        (threshold check, Equation 3)

Paper reference — Section IV-B and Theorems 1 & 2.

Key equations:
    v_j = ρ_j · θ_j · d_ij                          (peer vote)
    d_ij = exp(−α · dist(a_i, a_j))                  (proximity decay)
    Θ_agg = Σ v_j / Σ (ρ_j · d_ij)                  (Eq. 2)
    consensus iff Θ_agg > τ_consensus                 (Eq. 3)
              and |R| ≥ ⌈(n + f + 1) / 2⌉
"""

from __future__ import annotations
import math
import time
import uuid
import logging
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

from dmas.consensus.reputation import ReputationTracker

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Message types (match the paper's protocol message taxonomy)
# ---------------------------------------------------------------------------

@dataclass
class VoteRequest:
    """Phase 1: broadcast from the agent detecting the anomaly."""
    agent_id: str                  # a_i
    threat_id: str                 # T_id
    local_score: float             # θ_i
    feature_vector: List[float]    # φ_i (anomaly feature description)
    detect_time: float             # t_detect
    agent_position: Tuple[float, float] = (0.0, 0.0)  # (x, y) topology coords


@dataclass
class VoteResponse:
    """Phase 2: response from a peer agent."""
    voter_id: str                  # a_j
    threat_id: str
    vote_weight: float             # v_j = ρ_j · θ_j · d_ij
    raw_score: float               # θ_j (peer's own detection score)
    reputation: float              # ρ_j at time of vote
    distance_factor: float         # d_ij


@dataclass
class ConsensusResult:
    """Phase 4 output: final consensus decision."""
    threat_id: str
    agent_id: str
    theta_agg: float
    consensus_reached: bool
    action: str                    # "QUARANTINE" | "MONITOR" | "CLEAR"
    votes_received: int
    quorum_required: int
    elapsed_ms: float
    timestamp: float = field(default_factory=time.time)


# ---------------------------------------------------------------------------
# CVT Protocol
# ---------------------------------------------------------------------------

class CVTProtocol:
    """
    Implementation of the Consensus-based Threat Validation protocol.

    This class handles both sides of the protocol:
      - Initiator: sends VOTE_REQUEST, collects responses, decides.
      - Peer: receives VOTE_REQUEST, evaluates threat, sends VOTE_RESPONSE.

    Parameters
    ----------
    agent_id : str
    n_agents : int
        Total number of agents (n in paper).
    f_tolerance : int
        Max Byzantine agents tolerated: f ≤ ⌊(n−1)/3⌋.
    tau_alert : float
        Local score threshold for initiating CVT (paper: 0.45).
    tau_consensus : float
        Aggregate vote threshold for quarantine (paper: 0.75).
    delta_t_ms : float
        Vote collection timeout in ms (paper: 0.5).
    alpha : float
        Distance-decay rate (paper: 0.1).
    reputation_tracker : ReputationTracker, optional
    peer_score_fn : callable, optional
        f(feature_vector) -> float.  Called when evaluating a peer's
        VOTE_REQUEST using the local detection model.
    agent_position : (x, y), optional
        Logical network coordinates for proximity weighting.
    """

    def __init__(
        self,
        agent_id: str,
        n_agents: int,
        f_tolerance: Optional[int] = None,
        tau_alert: float = 0.45,
        tau_consensus: float = 0.75,
        delta_t_ms: float = 0.5,
        alpha: float = 0.1,
        reputation_tracker: Optional[ReputationTracker] = None,
        peer_score_fn: Optional[Callable[[List[float]], float]] = None,
        agent_position: Tuple[float, float] = (0.0, 0.0),
    ) -> None:
        self.agent_id = agent_id
        self.n_agents = n_agents
        self.f_tolerance = f_tolerance if f_tolerance is not None \
            else max(0, (n_agents - 1) // 3)
        self.tau_alert = tau_alert
        self.tau_consensus = tau_consensus
        self.delta_t_ms = delta_t_ms
        self.alpha = alpha
        self.reputation = reputation_tracker or ReputationTracker()
        self.peer_score_fn = peer_score_fn or (lambda fv: 0.5)
        self.agent_position = agent_position

        # Register self in reputation tracker
        self.reputation.register(agent_id)

        # In-flight vote collections: threat_id -> list of VoteResponse
        self._pending: Dict[str, Tuple[float, List[VoteResponse]]] = {}

    # ------------------------------------------------------------------
    # Phase 1 — Initiate
    # ------------------------------------------------------------------

    def build_vote_request(
        self,
        threat_id: Optional[str],
        local_score: float,
        feature_vector: List[float],
    ) -> Optional[VoteRequest]:
        """
        Returns a VoteRequest if local_score exceeds tau_alert,
        otherwise returns None (no coordination needed).
        """
        if local_score < self.tau_alert:
            return None
        tid = threat_id or str(uuid.uuid4())
        req = VoteRequest(
            agent_id=self.agent_id,
            threat_id=tid,
            local_score=local_score,
            feature_vector=feature_vector,
            detect_time=time.time(),
            agent_position=self.agent_position,
        )
        # Start tracking responses for this threat
        self._pending[tid] = (time.time(), [])
        logger.debug("[%s] CVT Phase 1: initiating vote for threat %s (score=%.3f)",
                     self.agent_id, tid, local_score)
        return req

    # ------------------------------------------------------------------
    # Phase 2 — Peer evaluation (called on the receiving agent side)
    # ------------------------------------------------------------------

    def handle_vote_request(self, req: VoteRequest) -> VoteResponse:
        """
        Evaluate a peer's vote request and return a weighted vote.

        Called on the *peer* side (a_j receiving from a_i).
        """
        # Evaluate threat using local detection model
        theta_j = self.peer_score_fn(req.feature_vector)

        # Get effective reputation of self (as voter)
        rho_j = self.reputation.effective_reputation(self.agent_id)

        # Compute proximity weight d_ij
        d_ij = self._distance_factor(req.agent_position, self.agent_position)

        # Weighted vote v_j = ρ_j · θ_j · d_ij
        v_j = rho_j * theta_j * d_ij

        logger.debug("[%s] CVT Phase 2: vote for %s | θ_j=%.3f ρ_j=%.3f d=%.3f v=%.3f",
                     self.agent_id, req.threat_id, theta_j, rho_j, d_ij, v_j)

        return VoteResponse(
            voter_id=self.agent_id,
            threat_id=req.threat_id,
            vote_weight=v_j,
            raw_score=theta_j,
            reputation=rho_j,
            distance_factor=d_ij,
        )

    # ------------------------------------------------------------------
    # Phase 3 — Vote aggregation
    # ------------------------------------------------------------------

    def receive_vote(self, response: VoteResponse) -> None:
        """Record an incoming vote response (called on the initiator)."""
        if response.threat_id in self._pending:
            _, votes = self._pending[response.threat_id]
            votes.append(response)

    def aggregate_votes(
        self, threat_id: str, votes: Optional[List[VoteResponse]] = None
    ) -> float:
        """
        Compute Θ_agg per Equation (2):
            Θ_agg = Σ v_j / Σ (ρ_j · d_ij)

        If `votes` is None, uses internally accumulated votes.
        """
        if votes is None:
            if threat_id not in self._pending:
                return 0.0
            _, votes = self._pending[threat_id]

        if not votes:
            return 0.0

        numerator = sum(v.vote_weight for v in votes)
        denominator = sum(v.reputation * v.distance_factor for v in votes)

        if denominator < 1e-9:
            return 0.0

        theta_agg = numerator / denominator
        logger.debug("[%s] CVT Phase 3: Θ_agg=%.3f from %d votes",
                     self.agent_id, theta_agg, len(votes))
        return theta_agg

    # ------------------------------------------------------------------
    # Phase 4 — Consensus decision
    # ------------------------------------------------------------------

    def decide(
        self,
        threat_id: str,
        theta_agg: float,
        n_responses: int,
        elapsed_ms: float,
    ) -> ConsensusResult:
        """
        Check the two consensus conditions (Equation 3):
            1. Θ_agg > τ_consensus
            2. |R| ≥ ⌈(n + f + 1) / 2⌉

        Returns a ConsensusResult with action = "QUARANTINE" | "MONITOR".
        """
        quorum = math.ceil((self.n_agents + self.f_tolerance + 1) / 2)
        condition1 = theta_agg > self.tau_consensus
        condition2 = n_responses >= quorum
        consensus_reached = condition1 and condition2

        action = "QUARANTINE" if consensus_reached else "MONITOR"

        result = ConsensusResult(
            threat_id=threat_id,
            agent_id=self.agent_id,
            theta_agg=theta_agg,
            consensus_reached=consensus_reached,
            action=action,
            votes_received=n_responses,
            quorum_required=quorum,
            elapsed_ms=elapsed_ms,
        )

        logger.info(
            "[%s] CVT Phase 4: %s | Θ_agg=%.3f (>%.2f? %s) "
            "votes=%d (≥%d? %s) elapsed=%.2fms",
            self.agent_id, action, theta_agg, self.tau_consensus,
            condition1, n_responses, quorum, condition2, elapsed_ms,
        )

        # Clean up pending state
        self._pending.pop(threat_id, None)
        return result

    # ------------------------------------------------------------------
    # Convenience: run full CVT round synchronously (simulation use)
    # ------------------------------------------------------------------

    def run_synchronous(
        self,
        local_score: float,
        feature_vector: List[float],
        peer_responses: List[Tuple[str, float, Tuple[float, float]]],
    ) -> Optional[ConsensusResult]:
        """
        Run a complete CVT round in one call.  Useful for simulation
        where all peer responses are available immediately.

        peer_responses: list of (peer_id, peer_theta_j, peer_position)
        """
        threat_id = str(uuid.uuid4())
        req = self.build_vote_request(threat_id, local_score, feature_vector)
        if req is None:
            return None

        t_start = time.perf_counter()
        votes: List[VoteResponse] = []
        for peer_id, peer_theta, peer_pos in peer_responses:
            rho_j = self.reputation.effective_reputation(peer_id)
            d_ij = self._distance_factor(self.agent_position, peer_pos)
            v_j = rho_j * peer_theta * d_ij
            votes.append(VoteResponse(
                voter_id=peer_id,
                threat_id=threat_id,
                vote_weight=v_j,
                raw_score=peer_theta,
                reputation=rho_j,
                distance_factor=d_ij,
            ))

        theta_agg = self.aggregate_votes(threat_id, votes)
        elapsed_ms = (time.perf_counter() - t_start) * 1000
        return self.decide(threat_id, theta_agg, len(votes), elapsed_ms)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _distance_factor(
        self,
        pos_i: Tuple[float, float],
        pos_j: Tuple[float, float],
    ) -> float:
        """d_ij = exp(−α · Euclidean distance)"""
        dist = math.sqrt((pos_i[0] - pos_j[0]) ** 2 + (pos_i[1] - pos_j[1]) ** 2)
        return math.exp(-self.alpha * dist)

    @property
    def quorum_size(self) -> int:
        return math.ceil((self.n_agents + self.f_tolerance + 1) / 2)

    def __repr__(self) -> str:
        return (f"CVTProtocol(agent={self.agent_id}, n={self.n_agents}, "
                f"f={self.f_tolerance}, τ_c={self.tau_consensus})")
