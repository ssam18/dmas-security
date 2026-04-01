"""Tests for EWMADetector."""
import math
import pytest
from dmas.monitoring.ewma_detector import EWMADetector


def test_init_valid():
    d = EWMADetector(lambda_=0.05, sigma_threshold=3.0)
    assert d.lambda_ == 0.05
    assert d.sigma_threshold == 3.0


def test_init_invalid_lambda():
    with pytest.raises(ValueError):
        EWMADetector(lambda_=0.0)
    with pytest.raises(ValueError):
        EWMADetector(lambda_=1.5)


def test_score_zero_before_update():
    d = EWMADetector()
    score = d.score({"rate": 10.0})
    assert score == 0.0, "Unknown features should return 0"


def test_normal_traffic_low_score():
    d = EWMADetector(lambda_=0.1, sigma_threshold=3.0)
    # Feed 100 normal samples
    for _ in range(100):
        d.update({"rate": 10.0 + 0.1 * (hash(str(_)) % 10 - 5)})
    # Normal observation near the mean should yield low score
    score = d.score({"rate": 10.5})
    assert score < 0.35, f"Expected low score for normal traffic, got {score:.3f}"


def test_anomalous_traffic_high_score():
    d = EWMADetector(lambda_=0.05, sigma_threshold=3.0)
    for i in range(200):
        d.update({"rate": 10.0})
    # Massive spike
    score = d.score({"rate": 1000.0})
    assert score > 0.8, f"Expected high score for DDoS spike, got {score:.3f}"


def test_update_and_score():
    d = EWMADetector()
    score = d.update_and_score({"x": 5.0})
    # First call initializes; no deviation, should be low
    assert isinstance(score, float)
    assert 0.0 <= score <= 1.0


def test_reset_clears_state():
    d = EWMADetector()
    d.update({"rate": 10.0})
    assert "rate" in d.feature_names()
    d.reset()
    assert d.feature_names() == []


def test_score_in_range():
    d = EWMADetector()
    for v in [1, 5, 10, 100, 1000, -50]:
        d.update({"x": float(v)})
        s = d.score({"x": float(v)})
        assert 0.0 <= s <= 1.0, f"Score out of range: {s}"


def test_multiple_features():
    d = EWMADetector(sigma_threshold=2.0)
    for _ in range(50):
        d.update({"pkt_rate": 10.0, "payload": 64.0, "conns": 3.0})
    # All normal
    s_normal = d.score({"pkt_rate": 10.2, "payload": 65.0, "conns": 3.1})
    # One anomalous feature
    s_anomalous = d.score({"pkt_rate": 5000.0, "payload": 64.0, "conns": 3.0})
    assert s_anomalous > s_normal
