"""Tests for ReputationTracker."""
import time
import pytest
from dmas.consensus.reputation import ReputationTracker


def make_tracker(**kwargs) -> ReputationTracker:
    return ReputationTracker(probationary_seconds=0, **kwargs)


def test_register_and_get():
    rt = make_tracker()
    rt.register("a1")
    assert rt.get_reputation("a1") == 0.5


def test_auto_register_on_get():
    rt = make_tracker()
    rep = rt.get_reputation("unknown_agent")
    assert rep == 0.5


def test_reputation_increases_on_correct_votes():
    rt = make_tracker(beta=0.5)
    rt.register("a1")
    for _ in range(20):
        rt.update("a1", was_correct=True)
    assert rt.get_reputation("a1") > 0.9


def test_reputation_decreases_on_wrong_votes():
    rt = make_tracker(beta=0.5)
    rt.register("a1")
    for _ in range(20):
        rt.update("a1", was_correct=False)
    assert rt.get_reputation("a1") < 0.1


def test_reputation_bounded():
    rt = make_tracker(beta=0.9)
    rt.register("a1")
    for _ in range(1000):
        rt.update("a1", was_correct=True)
    assert rt.get_reputation("a1") <= 1.0
    for _ in range(1000):
        rt.update("a1", was_correct=False)
    assert rt.get_reputation("a1") >= 0.0


def test_probationary_weight_applied():
    rt = ReputationTracker(probationary_seconds=9999)   # never expires in test
    rt.register("new_agent")
    eff = rt.effective_reputation("new_agent")
    raw = rt.get_reputation("new_agent")
    assert eff == pytest.approx(raw * 0.5, abs=1e-6)


def test_probation_lifted_after_time(monkeypatch):
    rt = ReputationTracker(probationary_seconds=0.001)
    rt.register("new_agent")
    time.sleep(0.01)
    assert not rt.is_probationary("new_agent")


def test_all_reputations():
    rt = make_tracker()
    for i in range(5):
        rt.register(f"a{i}")
    reps = rt.all_reputations()
    assert len(reps) == 5
    assert all(v == 0.5 for v in reps.values())


def test_summary_structure():
    rt = make_tracker()
    rt.register("a1")
    rt.update("a1", True)
    rt.update("a1", False)
    s = rt.summary("a1")
    assert s["n_votes"] == 2
    assert s["historical_accuracy"] == pytest.approx(0.5)
    assert "reputation" in s
    assert "effective" in s


def test_ewma_formula():
    """Verify ρ^(t+1) = β·ρ^t + (1-β)·acc^t exactly."""
    beta = 0.9
    rt = make_tracker(beta=beta)
    rt.register("a1")
    rho = 0.5  # initial
    for correct in [True, False, True, True, False]:
        acc = 1.0 if correct else 0.0
        expected = beta * rho + (1 - beta) * acc
        rho = rt.update("a1", correct)
        assert rho == pytest.approx(expected, abs=1e-9)
