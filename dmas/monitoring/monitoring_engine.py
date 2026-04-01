"""
Monitoring Engine — weighted ensemble of three detectors.

Combines EWMA statistical detector, GRU behavioral model, and signature
matcher into a single threat score θ ∈ [0, 1].

Paper reference — Equation (1):
    θ = w_s · θ_s + w_b · θ_b + w_m · θ_m
    where w_s + w_b + w_m = 1
"""

from __future__ import annotations
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional
import numpy as np

from dmas.monitoring.ewma_detector import EWMADetector
from dmas.monitoring.behavioral_model import BehavioralModel
from dmas.monitoring.signature_matcher import SignatureMatcher


@dataclass
class DeviceObservation:
    """A single telemetry observation from one IIoT device."""
    device_id: str
    timestamp: float = field(default_factory=time.time)
    # Statistical features: packet rate, payload size, protocol dist, etc.
    stat_features: Dict[str, float] = field(default_factory=dict)
    # Time-series feature vector (length == behavioral_model.input_features)
    feature_vector: Optional[np.ndarray] = None
    # Raw payload bytes for signature matching
    payload: bytes = b""


@dataclass
class ThreatAssessment:
    """Composite output from the monitoring engine for one observation."""
    device_id: str
    timestamp: float
    score_statistical: float      # θ_s
    score_behavioral: float       # θ_b
    score_signature: float        # θ_m
    score_composite: float        # θ  (weighted ensemble)
    matched_signatures: List[Dict] = field(default_factory=list)

    @property
    def is_alert(self) -> bool:
        return self.score_composite > 0.45  # tau_alert default


class MonitoringEngine:
    """
    Ensemble monitoring engine for a single DMAS agent.

    Parameters
    ----------
    w_s, w_b, w_m : float
        Weights for statistical, behavioral, and signature detectors.
        Must sum to 1.0.
    ewma_lambda : float
        Smoothing factor for the EWMA detector.
    sigma_threshold : float
        σ threshold for the EWMA detector.
    input_features : int
        Dimensionality of feature vectors fed to the behavioral model.
    window_size : int
        Temporal window fed to the behavioral model (time steps).
    """

    def __init__(
        self,
        w_s: float = 0.35,
        w_b: float = 0.40,
        w_m: float = 0.25,
        ewma_lambda: float = 0.05,
        sigma_threshold: float = 3.0,
        input_features: int = 8,
        window_size: int = 20,
        seed: int = 42,
    ) -> None:
        if abs(w_s + w_b + w_m - 1.0) > 1e-6:
            raise ValueError(f"Weights must sum to 1.0, got {w_s + w_b + w_m}")
        self.w_s = w_s
        self.w_b = w_b
        self.w_m = w_m

        self.ewma = EWMADetector(lambda_=ewma_lambda, sigma_threshold=sigma_threshold)
        self.behavioral = BehavioralModel(
            input_features=input_features,
            window_size=window_size,
            seed=seed,
        )
        self.signatures = SignatureMatcher()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def observe(self, obs: DeviceObservation) -> ThreatAssessment:
        """
        Process one observation and return a ThreatAssessment.

        All three sub-detectors are queried; their scores are blended
        according to the learned weights (w_s, w_b, w_m).
        """
        # 1. Statistical detector
        theta_s = self.ewma.update_and_score(obs.stat_features) \
            if obs.stat_features else 0.0

        # 2. Behavioral detector
        if obs.feature_vector is not None:
            self.behavioral.push(obs.feature_vector)
        theta_b = self.behavioral.score()

        # 3. Signature detector
        matched = self.signatures.match_details(obs.payload)
        theta_m = max((s["severity"] for s in matched), default=0.0)

        # 4. Weighted composite (Equation 1)
        theta = self.w_s * theta_s + self.w_b * theta_b + self.w_m * theta_m

        return ThreatAssessment(
            device_id=obs.device_id,
            timestamp=obs.timestamp,
            score_statistical=theta_s,
            score_behavioral=theta_b,
            score_signature=theta_m,
            score_composite=theta,
            matched_signatures=matched,
        )

    def calibrate_behavioral(self, normal_sequences: List[np.ndarray]) -> None:
        """Pass a list of normal sequences to the behavioral model for calibration."""
        self.behavioral.calibrate(normal_sequences)

    def add_signature(self, sig: Dict) -> None:
        """Add a new signature to the signature database at runtime."""
        self.signatures.add_signature(sig)

    def __repr__(self) -> str:
        return (f"MonitoringEngine("
                f"w=[{self.w_s},{self.w_b},{self.w_m}], "
                f"ewma={self.ewma}, "
                f"behavioral={self.behavioral}, "
                f"sigs={self.signatures})")
