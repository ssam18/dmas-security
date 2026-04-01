"""
DMASAgent — Top-level orchestrator for one edge-gateway agent.

Wires together:
  MonitoringEngine  →  threat score θ
  CVTProtocol       →  consensus-based validation
  P2PProtocol       →  peer communication
  ResponseExecutor  →  quarantine / alert / forensics
  ReputationTracker →  per-agent reputation management

Paper reference — Section III (System Architecture).
"""

from __future__ import annotations
import logging
import time
import uuid
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from dmas.monitoring.monitoring_engine import MonitoringEngine, DeviceObservation
from dmas.consensus.cvt_protocol import CVTProtocol, VoteRequest, VoteResponse
from dmas.consensus.reputation import ReputationTracker
from dmas.communication.p2p_protocol import P2PProtocol, DMASMessage, MessageType
from dmas.response.response_executor import ResponseExecutor

logger = logging.getLogger(__name__)


@dataclass
class AgentStats:
    agent_id: str
    n_observations: int = 0
    n_threats_detected: int = 0
    n_votes_initiated: int = 0
    n_consensus_quarantines: int = 0
    n_votes_cast: int = 0
    uptime_start: float = field(default_factory=time.time)

    @property
    def uptime_seconds(self) -> float:
        return time.time() - self.uptime_start


class DMASAgent:
    """
    A single DMAS edge-gateway security agent.

    Parameters
    ----------
    agent_id : str
        Unique identifier for this agent.
    n_agents : int
        Total number of agents in the swarm.
    position : (x, y)
        Logical network coordinates for proximity weighting.
    simulation_mode : bool
        If True, response actions are logged only (no real firewall calls).
    log_dir : str
        Directory for alert and forensics logs.
    **kwargs
        Passed through to sub-components (see config/default_config.yaml).
    """

    def __init__(
        self,
        agent_id: str,
        n_agents: int = 5,
        position: Tuple[float, float] = (0.0, 0.0),
        simulation_mode: bool = True,
        log_dir: str = "logs",
        # CVT parameters
        tau_alert: float = 0.45,
        tau_consensus: float = 0.75,
        delta_t_ms: float = 0.5,
        alpha: float = 0.1,
        beta: float = 0.9,
        # Monitoring parameters
        w_s: float = 0.35,
        w_b: float = 0.40,
        w_m: float = 0.25,
        ewma_lambda: float = 0.05,
        sigma_threshold: float = 3.0,
        input_features: int = 8,
        window_size: int = 20,
        seed: int = 42,
    ) -> None:
        self.agent_id = agent_id
        self.n_agents = n_agents
        self.position = position
        self.simulation_mode = simulation_mode

        # --- Sub-components ---
        self.reputation = ReputationTracker(
            beta=beta,
            probationary_seconds=0 if simulation_mode else 86_400,
        )
        self.reputation.register(agent_id)

        self.monitoring = MonitoringEngine(
            w_s=w_s, w_b=w_b, w_m=w_m,
            ewma_lambda=ewma_lambda,
            sigma_threshold=sigma_threshold,
            input_features=input_features,
            window_size=window_size,
            seed=seed,
        )

        self.cvt = CVTProtocol(
            agent_id=agent_id,
            n_agents=n_agents,
            tau_alert=tau_alert,
            tau_consensus=tau_consensus,
            delta_t_ms=delta_t_ms,
            alpha=alpha,
            reputation_tracker=self.reputation,
            peer_score_fn=self._local_score_from_features,
            agent_position=position,
        )

        self.comm = P2PProtocol(
            agent_id=agent_id,
            on_message=self._on_message,
            use_multicast=False,   # simulation mode by default
        )

        self.response = ResponseExecutor(
            agent_id=agent_id,
            alert_log_path=f"{log_dir}/alerts_{agent_id}.jsonl",
            forensics_log_path=f"{log_dir}/forensics_{agent_id}.jsonl",
            simulation_mode=simulation_mode,
        )

        self.stats = AgentStats(agent_id=agent_id)

        # In-flight vote tracking: threat_id -> list[VoteResponse]
        self._pending_votes: Dict[str, List[VoteResponse]] = {}

        logger.info("[%s] Agent initialised | pos=%s | n_agents=%d",
                    agent_id, position, n_agents)

    # ------------------------------------------------------------------
    # Main processing loop entry point
    # ------------------------------------------------------------------

    def process_observation(self, obs: DeviceObservation) -> Optional[str]:
        """
        Process a single device telemetry observation.

        Returns the action taken ("QUARANTINE", "MONITOR", or None if
        the score was below tau_alert).
        """
        self.stats.n_observations += 1
        assessment = self.monitoring.observe(obs)

        logger.debug("[%s] device=%s θ=%.3f (s=%.3f b=%.3f m=%.3f)",
                     self.agent_id, obs.device_id,
                     assessment.score_composite,
                     assessment.score_statistical,
                     assessment.score_behavioral,
                     assessment.score_signature)

        if not assessment.is_alert:
            return None

        self.stats.n_threats_detected += 1

        # Build feature vector for CVT (use stat_features + behavioral score)
        fv = self._build_feature_vector(obs, assessment)

        # Phase 1: build vote request
        req = self.cvt.build_vote_request(
            threat_id=None,
            local_score=assessment.score_composite,
            feature_vector=fv,
        )
        if req is None:
            return None

        self.stats.n_votes_initiated += 1

        # Broadcast to peers via P2P layer
        msg = DMASMessage.vote_request(
            self.agent_id, req.threat_id,
            assessment.score_composite, fv,
        )
        self.comm.send(msg)

        return req.threat_id  # caller can await responses and call finalize_vote()

    def finalize_vote(
        self,
        threat_id: str,
        device_id: str,
        evidence: Optional[Dict] = None,
    ) -> Optional[str]:
        """
        Phase 3+4: aggregate collected votes and execute response.
        Call this after the delta_t timeout has elapsed.
        """
        votes = self._pending_votes.pop(threat_id, [])
        theta_agg = self.cvt.aggregate_votes(threat_id, votes)
        elapsed_ms = self.cvt.delta_t_ms   # in real async use perf_counter diff

        result = self.cvt.decide(threat_id, theta_agg, len(votes), elapsed_ms)

        if result.consensus_reached:
            self.stats.n_consensus_quarantines += 1

        self.response.execute(
            action=result.action,
            device_id=device_id,
            threat_id=threat_id,
            theta_agg=result.theta_agg,
            n_votes=result.votes_received,
            evidence=evidence or {},
        )

        # Broadcast outcome
        msg = DMASMessage.consensus_achieved(
            self.agent_id, threat_id, result.theta_agg, result.action
        )
        self.comm.send(msg)

        return result.action

    # ------------------------------------------------------------------
    # Incoming message handler (P2P receive callback)
    # ------------------------------------------------------------------

    def _on_message(self, msg: DMASMessage) -> None:
        if msg.msg_type == MessageType.VOTE_REQUEST:
            self._handle_vote_request(msg)
        elif msg.msg_type == MessageType.VOTE_RESPONSE:
            self._handle_vote_response(msg)
        elif msg.msg_type == MessageType.CONSENSUS_ACHIEVED:
            self._handle_consensus_achieved(msg)
        elif msg.msg_type == MessageType.HEARTBEAT:
            self.reputation.register(msg.sender_id)

    def _handle_vote_request(self, msg: DMASMessage) -> None:
        """Phase 2: evaluate peer's vote request and respond."""
        self.stats.n_votes_cast += 1
        p = msg.payload
        threat_id = p["tid"]
        feature_vector = p.get("fv", [])

        # Build a pseudo VoteRequest for the CVT protocol
        from dmas.consensus.cvt_protocol import VoteRequest
        req = VoteRequest(
            agent_id=msg.sender_id,
            threat_id=threat_id,
            local_score=p["score"],
            feature_vector=feature_vector,
            detect_time=msg.timestamp,
        )
        response = self.cvt.handle_vote_request(req)

        # Send vote back to requester
        resp_msg = DMASMessage.vote_response(
            self.agent_id, threat_id,
            response.vote_weight, response.raw_score,
        )
        self.comm.sim_unicast(msg.sender_id, resp_msg)
        logger.debug("[%s] voted on %s: v=%.3f", self.agent_id, threat_id, response.vote_weight)

    def _handle_vote_response(self, msg: DMASMessage) -> None:
        """Collect an incoming vote into the pending bucket."""
        p = msg.payload
        threat_id = p["tid"]
        rho_j = self.reputation.effective_reputation(msg.sender_id)
        vote = VoteResponse(
            voter_id=msg.sender_id,
            threat_id=threat_id,
            vote_weight=p["vw"],
            raw_score=p["rs"],
            reputation=rho_j,
            distance_factor=1.0,  # distance already baked in by peer
        )
        self._pending_votes.setdefault(threat_id, []).append(vote)

    def _handle_consensus_achieved(self, msg: DMASMessage) -> None:
        p = msg.payload
        logger.info("[%s] Received CONSENSUS_ACHIEVED from %s: action=%s θ=%.3f",
                    self.agent_id, msg.sender_id, p["act"], p["agg"])

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _local_score_from_features(self, feature_vector: List[float]) -> float:
        """
        Quick local re-evaluation of a peer's feature vector using the
        EWMA detector (no behavioral model for peer evaluation — paper
        uses f_detect(φ_i, D_j)).
        """
        if not feature_vector:
            return 0.5
        fdict = {f"f{i}": v for i, v in enumerate(feature_vector)}
        return self.monitoring.ewma.score(fdict)

    def _build_feature_vector(self, obs: DeviceObservation, assessment) -> List[float]:
        fv = list(obs.stat_features.values())[:6]
        fv += [assessment.score_statistical, assessment.score_behavioral]
        return fv

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    def summary(self) -> Dict:
        return {
            "agent_id": self.agent_id,
            "position": self.position,
            "uptime_s": round(self.stats.uptime_seconds, 1),
            "n_observations": self.stats.n_observations,
            "n_threats_detected": self.stats.n_threats_detected,
            "n_quarantines": self.stats.n_consensus_quarantines,
            "active_quarantines": list(self.response.quarantined_devices),
            "reputation_self": round(
                self.reputation.get_reputation(self.agent_id), 3),
        }

    def __repr__(self) -> str:
        return (f"DMASAgent(id={self.agent_id}, pos={self.position}, "
                f"n_agents={self.n_agents})")
