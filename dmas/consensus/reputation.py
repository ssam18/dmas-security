"""
Agent Reputation Tracker.

Maintains an EWMA-based reputation score ρ_j ∈ [0, 1] for each peer
agent.  Reputation is updated based on whether an agent's past votes
agreed with the post-hoc ground truth.

Paper reference — Equation (5) and Section IV-D:
    ρ_j^(t+1) = β · ρ_j^(t) + (1 − β) · acc_j^(t)
    where acc_j^(t) ∈ {0, 1}

New agents enter a 24-hour probationary period during which their
effective vote weight is reduced to 50%.
"""

from __future__ import annotations
import time
from dataclasses import dataclass, field
from typing import Dict


@dataclass
class _AgentRecord:
    agent_id: str
    reputation: float = 0.5
    join_time: float = field(default_factory=time.time)
    n_votes: int = 0
    n_correct: int = 0
    is_probationary: bool = True


class ReputationTracker:
    """
    Tracks and updates reputation scores for a set of peer agents.

    Parameters
    ----------
    beta : float
        EWMA smoothing factor (paper value: 0.9).
    initial_reputation : float
        Starting reputation for new agents (paper: 0.5).
    probationary_seconds : float
        Duration of probationary period in seconds (paper: 24 h).
    probationary_weight : float
        Vote weight multiplier during probation (paper: 0.5).
    """

    def __init__(
        self,
        beta: float = 0.9,
        initial_reputation: float = 0.5,
        probationary_seconds: float = 86_400,   # 24 hours
        probationary_weight: float = 0.5,
    ) -> None:
        self.beta = beta
        self.initial_reputation = initial_reputation
        self.probationary_seconds = probationary_seconds
        self.probationary_weight = probationary_weight
        self._records: Dict[str, _AgentRecord] = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def register(self, agent_id: str) -> None:
        """Register a new agent (starts in probationary mode)."""
        if agent_id not in self._records:
            self._records[agent_id] = _AgentRecord(
                agent_id=agent_id,
                reputation=self.initial_reputation,
            )

    def effective_reputation(self, agent_id: str) -> float:
        """
        Return the effective reputation weight for voting.

        Applies a 50% discount during the probationary period.
        """
        rec = self._get_or_register(agent_id)
        self._maybe_lift_probation(rec)
        rho = rec.reputation
        if rec.is_probationary:
            rho *= self.probationary_weight
        return rho

    def update(self, agent_id: str, was_correct: bool) -> float:
        """
        Apply one EWMA reputation update step.

        Parameters
        ----------
        agent_id : str
        was_correct : bool
            Whether the agent's vote matched post-hoc ground truth.

        Returns the updated reputation.
        """
        rec = self._get_or_register(agent_id)
        acc = 1.0 if was_correct else 0.0
        rec.reputation = self.beta * rec.reputation + (1 - self.beta) * acc
        rec.reputation = max(0.0, min(1.0, rec.reputation))
        rec.n_votes += 1
        if was_correct:
            rec.n_correct += 1
        return rec.reputation

    def get_reputation(self, agent_id: str) -> float:
        """Raw reputation (without probation discount)."""
        return self._get_or_register(agent_id).reputation

    def is_probationary(self, agent_id: str) -> bool:
        rec = self._get_or_register(agent_id)
        self._maybe_lift_probation(rec)
        return rec.is_probationary

    def all_reputations(self) -> Dict[str, float]:
        return {aid: rec.reputation for aid, rec in self._records.items()}

    def summary(self, agent_id: str) -> Dict:
        rec = self._get_or_register(agent_id)
        accuracy = rec.n_correct / rec.n_votes if rec.n_votes else None
        return {
            "agent_id": agent_id,
            "reputation": rec.reputation,
            "effective": self.effective_reputation(agent_id),
            "is_probationary": rec.is_probationary,
            "n_votes": rec.n_votes,
            "historical_accuracy": accuracy,
        }

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _get_or_register(self, agent_id: str) -> _AgentRecord:
        if agent_id not in self._records:
            self.register(agent_id)
        return self._records[agent_id]

    def _maybe_lift_probation(self, rec: _AgentRecord) -> None:
        if rec.is_probationary:
            age = time.time() - rec.join_time
            if age >= self.probationary_seconds:
                rec.is_probationary = False

    def __repr__(self) -> str:
        return (f"ReputationTracker(beta={self.beta}, "
                f"n_agents={len(self._records)})")
