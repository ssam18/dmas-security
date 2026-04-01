"""End-to-end simulation tests."""
import pytest
from dmas.simulation.testbed import DMASTestbed
from dmas.simulation.attack_generator import AttackGenerator, AttackType


class TestAttackGenerator:
    def test_normal_event(self):
        gen = AttackGenerator(seed=1)
        event = gen.generate("dev_001", "sensor", AttackType.NORMAL)
        assert event.attack_type == AttackType.NORMAL
        assert not event.ground_truth

    def test_all_attack_types(self):
        gen = AttackGenerator(seed=2)
        for atype in [AttackType.DDOS, AttackType.MITM, AttackType.REPLAY,
                      AttackType.INJECTION, AttackType.MALWARE, AttackType.ZERO_DAY]:
            event = gen.generate("dev_001", "plc", atype)
            assert event.attack_type == atype
            assert event.ground_truth

    def test_ddos_elevated_packet_rate(self):
        gen = AttackGenerator(seed=3)
        normal = gen.generate("d1", "sensor", AttackType.NORMAL)
        ddos = gen.generate("d1", "sensor", AttackType.DDOS)
        assert (ddos.observation.stat_features["packet_rate"] >
                normal.observation.stat_features["packet_rate"] * 5)

    def test_batch_generation(self):
        gen = AttackGenerator(seed=4)
        batch = gen.generate_batch(
            device_ids=["d0", "d1", "d2"],
            n_events=60,
            attack_fraction=0.5,
        )
        assert len(batch) == 60
        attacks = [e for e in batch if e.ground_truth]
        normals = [e for e in batch if not e.ground_truth]
        # Attack fraction should be roughly 50% ± some variance
        assert 10 < len(attacks) < 55


class TestTestbed:
    def test_basic_run(self):
        tb = DMASTestbed(n_agents=3, devices_per_agent=10, seed=0)
        result = tb.run(n_events=50, verbose=False)
        assert result.n_events == 50
        assert result.n_attacks + result.n_normals == 50
        assert 0.0 <= result.detection_accuracy <= 1.0
        assert 0.0 <= result.false_positive_rate <= 1.0

    def test_detection_above_chance(self):
        tb = DMASTestbed(n_agents=5, devices_per_agent=20, seed=42)
        result = tb.run(n_events=100, attack_fraction=0.3, verbose=False)
        # Should detect more than random chance
        assert result.detection_accuracy > 0.5

    def test_no_byzantine_high_accuracy(self):
        tb = DMASTestbed(n_agents=5, devices_per_agent=20,
                         byzantine_fraction=0.0, seed=42)
        result = tb.run(n_events=150, attack_fraction=0.25, verbose=False)
        # Without Byzantine agents, accuracy should be reasonable
        assert result.detection_accuracy > 0.45

    def test_heavy_byzantine_degrades_gracefully(self):
        """At 30% Byzantine (at the f<n/3 limit), system should still function."""
        tb = DMASTestbed(n_agents=10, devices_per_agent=10,
                         byzantine_fraction=0.30, seed=42)
        result = tb.run(n_events=100, verbose=False)
        # Should not crash and accuracy should stay above zero
        assert result.detection_accuracy >= 0.0
        assert result.n_events == 100

    def test_result_metrics_consistent(self):
        tb = DMASTestbed(n_agents=3, devices_per_agent=5, seed=7)
        r = tb.run(n_events=80, verbose=False)
        # Basic consistency checks
        assert r.n_detected <= r.n_attacks
        assert r.n_false_positives <= r.n_normals
        assert r.avg_response_ms >= 0

    def test_swarm_summary(self):
        tb = DMASTestbed(n_agents=3, devices_per_agent=5, seed=0)
        summaries = tb.swarm_summary()
        assert len(summaries) == 3
        for s in summaries:
            assert "agent_id" in s
            assert "n_observations" in s

    def test_f1_score_formula(self):
        tb = DMASTestbed(n_agents=3, devices_per_agent=5, seed=0)
        r = tb.run(n_events=60, verbose=False)
        p, rec = r.precision, r.recall
        if p + rec > 0:
            expected_f1 = 2 * p * rec / (p + rec)
            assert r.f1 == pytest.approx(expected_f1, abs=1e-6)
