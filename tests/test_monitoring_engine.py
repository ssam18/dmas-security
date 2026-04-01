"""Tests for MonitoringEngine ensemble."""
import numpy as np
import pytest
from dmas.monitoring.monitoring_engine import MonitoringEngine, DeviceObservation


def make_engine(**kw) -> MonitoringEngine:
    return MonitoringEngine(**kw)


def normal_obs(device_id="dev_001", rate=10.0, payload_size=64.0) -> DeviceObservation:
    return DeviceObservation(
        device_id=device_id,
        stat_features={"packet_rate": rate, "payload_size": payload_size,
                       "conn_count": 3.0, "protocol_entropy": 0.8},
        feature_vector=np.array([rate, payload_size, 3.0, 0.8, 0.0, 0.0, 0.0, 0.0],
                                 dtype=np.float32),
        payload=b"\x00" * int(payload_size),
    )


def attack_obs(device_id="dev_001") -> DeviceObservation:
    """DDoS-like observation with signature match."""
    return DeviceObservation(
        device_id=device_id,
        stat_features={"packet_rate": 5000.0, "payload_size": 64.0,
                       "conn_count": 200.0, "protocol_entropy": 0.2},
        feature_vector=np.array([5000.0, 64.0, 200.0, 0.2, 0.0, 0.0, 0.0, 0.0],
                                 dtype=np.float32),
        payload=b"\x02\x04\x05\xb4\x01\x01\x08\x0a" * 4,
    )


def test_weight_validation():
    with pytest.raises(ValueError):
        MonitoringEngine(w_s=0.5, w_b=0.5, w_m=0.5)


def test_normal_obs_returns_assessment():
    eng = make_engine()
    obs = normal_obs()
    a = eng.observe(obs)
    assert a.device_id == "dev_001"
    assert 0.0 <= a.score_composite <= 1.0


def test_attack_obs_scores_higher():
    eng = make_engine()
    # Warm up EWMA with normal traffic
    for _ in range(100):
        eng.observe(normal_obs(rate=10.0 + np.random.randn() * 0.5))
    normal_score = eng.observe(normal_obs()).score_composite
    attack_score = eng.observe(attack_obs()).score_composite
    assert attack_score > normal_score


def test_signature_match_elevates_score():
    eng = make_engine(w_m=0.8, w_s=0.1, w_b=0.1)
    obs = DeviceObservation(
        device_id="d1",
        stat_features={"packet_rate": 10.0},
        feature_vector=np.zeros(8, dtype=np.float32),
        payload=b"\xff\x53\x4d\x42\x72\x00\x00\x00",  # EternalBlue signature
    )
    a = eng.observe(obs)
    assert a.score_signature > 0.5
    assert len(a.matched_signatures) >= 1


def test_is_alert_property():
    eng = make_engine()
    for _ in range(100):
        eng.observe(normal_obs())
    # After EWMA warm-up, a normal observation should not be an alert
    a = eng.observe(normal_obs())
    assert not a.is_alert, f"Normal traffic triggered alert: score={a.score_composite:.3f}"


def test_composite_is_weighted_sum():
    eng = make_engine(w_s=0.35, w_b=0.40, w_m=0.25)
    obs = DeviceObservation(
        device_id="d1",
        stat_features={"x": 9999.0},
        feature_vector=np.zeros(8, dtype=np.float32),
        payload=b"' OR '1'='1",
    )
    a = eng.observe(obs)
    expected = 0.35 * a.score_statistical + 0.40 * a.score_behavioral + 0.25 * a.score_signature
    assert a.score_composite == pytest.approx(expected, abs=1e-6)


def test_add_custom_signature():
    eng = make_engine()
    eng.add_signature({
        "id": "CUSTOM-001",
        "name": "test pattern",
        "pattern": b"CUSTOM_ATTACK_MARKER",
        "severity": 1.0,
    })
    obs = DeviceObservation(
        device_id="d1",
        stat_features={},
        feature_vector=np.zeros(8, dtype=np.float32),
        payload=b"prefix CUSTOM_ATTACK_MARKER suffix",
    )
    a = eng.observe(obs)
    assert a.score_signature == pytest.approx(1.0, abs=1e-6)
