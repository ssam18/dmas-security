"""
Statistical EWMA Anomaly Detector.

Uses Exponentially Weighted Moving Average to maintain running statistics
for each monitored feature.  A feature value that deviates more than
`sigma_threshold` standard deviations from its EWMA mean is considered
anomalous.

Paper reference — Section III-B-1:
  'Uses EWMA with smoothing factor λ=0.05 (tuned on validation data)
   to track running statistics over a sliding window of 60 seconds.
   Deviations beyond 3σ from the learned normal baseline trigger an
   anomaly flag.'
"""

from __future__ import annotations
import math
from dataclasses import dataclass, field
from typing import Dict, Optional


@dataclass
class _FeatureState:
    """Internal EWMA state for one feature."""
    mean: float = 0.0
    var: float = 1.0          # start with unit variance to avoid div-by-zero
    initialized: bool = False


class EWMADetector:
    """
    Lightweight statistical anomaly detector using EWMA.

    Parameters
    ----------
    lambda_ : float
        EWMA smoothing factor (paper value: 0.05).
    sigma_threshold : float
        How many standard deviations constitute an anomaly (paper: 3.0).
    """

    def __init__(self, lambda_: float = 0.05, sigma_threshold: float = 3.0) -> None:
        if not 0 < lambda_ <= 1:
            raise ValueError("lambda_ must be in (0, 1]")
        self.lambda_ = lambda_
        self.sigma_threshold = sigma_threshold
        self._state: Dict[str, _FeatureState] = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def update(self, features: Dict[str, float]) -> None:
        """Update EWMA state with a new observation (no scoring)."""
        for name, value in features.items():
            s = self._state.setdefault(name, _FeatureState())
            if not s.initialized:
                s.mean = value
                s.var = 1.0
                s.initialized = True
            else:
                diff = value - s.mean
                s.mean = (1 - self.lambda_) * s.mean + self.lambda_ * value
                s.var = (1 - self.lambda_) * s.var + self.lambda_ * diff ** 2

    def score(self, features: Dict[str, float]) -> float:
        """
        Compute an anomaly score in [0, 1] for the given feature dict.

        Returns 0.0 for perfectly normal traffic and approaches 1.0 as
        the maximum z-score across features grows large.
        """
        if not features:
            return 0.0

        max_z = 0.0
        known_count = 0
        for name, value in features.items():
            s = self._state.get(name)
            if s is None or not s.initialized:
                continue
            known_count += 1
            std = math.sqrt(max(s.var, 1e-9))
            z = abs(value - s.mean) / std
            max_z = max(max_z, z)

        # Return 0.0 if no features have been seen yet (no baseline)
        if known_count == 0:
            return 0.0

        # Sigmoid-like mapping: score=0.5 at z=sigma_threshold
        return 1.0 / (1.0 + math.exp(-(max_z - self.sigma_threshold)))

    def update_and_score(self, features: Dict[str, float]) -> float:
        """Convenience: update state then return score."""
        self.update(features)
        return self.score(features)

    def reset(self) -> None:
        """Clear all learned state (e.g. on device reconnect)."""
        self._state.clear()

    def feature_names(self):
        return list(self._state.keys())

    def __repr__(self) -> str:
        return (f"EWMADetector(lambda={self.lambda_}, "
                f"sigma_threshold={self.sigma_threshold}, "
                f"n_features={len(self._state)})")
