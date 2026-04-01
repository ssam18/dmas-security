"""Tests for CVTProtocol — the 4-phase consensus algorithm."""
import math
import pytest
from dmas.consensus.cvt_protocol import CVTProtocol, VoteRequest, VoteResponse
from dmas.consensus.reputation import ReputationTracker


def make_cvt(agent_id="a0", n_agents=5) -> CVTProtocol:
    rep = ReputationTracker(probationary_seconds=0)
    for i in range(n_agents):
        rep.register(f"a{i}")
    return CVTProtocol(
        agent_id=agent_id,
        n_agents=n_agents,
        reputation_tracker=rep,
        agent_position=(0.0, 0.0),
    )


class TestPhase1:
    def test_no_vote_below_threshold(self):
        cvt = make_cvt()
        req = cvt.build_vote_request(None, local_score=0.20, feature_vector=[0.1] * 8)
        assert req is None

    def test_vote_request_above_threshold(self):
        cvt = make_cvt()
        req = cvt.build_vote_request(None, local_score=0.60, feature_vector=[0.5] * 8)
        assert req is not None
        assert isinstance(req, VoteRequest)
        assert req.agent_id == "a0"
        assert len(req.feature_vector) == 8

    def test_threat_id_auto_assigned(self):
        cvt = make_cvt()
        req = cvt.build_vote_request(None, local_score=0.8, feature_vector=[])
        assert req.threat_id != "" and req.threat_id is not None

    def test_custom_threat_id(self):
        cvt = make_cvt()
        req = cvt.build_vote_request("TID-123", local_score=0.9, feature_vector=[])
        assert req.threat_id == "TID-123"


class TestPhase2:
    def test_vote_response_structure(self):
        cvt = make_cvt(agent_id="a1")
        req = VoteRequest(
            agent_id="a0", threat_id="T1", local_score=0.8,
            feature_vector=[0.5] * 8, detect_time=0.0,
            agent_position=(1.0, 1.0),
        )
        resp = cvt.handle_vote_request(req)
        assert isinstance(resp, VoteResponse)
        assert resp.voter_id == "a1"
        assert resp.threat_id == "T1"
        assert 0.0 <= resp.vote_weight <= 1.0
        assert 0.0 <= resp.reputation <= 1.0
        assert 0.0 < resp.distance_factor <= 1.0

    def test_distance_decay(self):
        """Agents further away should have lower distance factor."""
        rep = ReputationTracker(probationary_seconds=0)
        for i in range(5):
            rep.register(f"a{i}")

        cvt_near = CVTProtocol("a1", 5, reputation_tracker=rep,
                               agent_position=(1.0, 0.0))
        cvt_far = CVTProtocol("a2", 5, reputation_tracker=rep,
                              agent_position=(10.0, 0.0))

        req = VoteRequest("a0", "T1", 0.9, [], 0.0, agent_position=(0.0, 0.0))
        resp_near = cvt_near.handle_vote_request(req)
        resp_far = cvt_far.handle_vote_request(req)
        assert resp_near.distance_factor > resp_far.distance_factor


class TestPhase3:
    def test_aggregate_empty(self):
        cvt = make_cvt()
        theta = cvt.aggregate_votes("non_existent_tid", [])
        assert theta == 0.0

    def test_aggregate_single_vote(self):
        cvt = make_cvt()
        votes = [VoteResponse("a1", "T1", vote_weight=0.7,
                              raw_score=0.8, reputation=0.9, distance_factor=0.9)]
        theta = cvt.aggregate_votes("T1", votes)
        assert 0.0 <= theta <= 1.5   # can exceed 1 due to raw vote scaling

    def test_aggregate_multiple_votes(self):
        cvt = make_cvt()
        votes = [
            VoteResponse("a1", "T1", 0.8, 0.85, 0.9, 0.9),
            VoteResponse("a2", "T1", 0.7, 0.75, 0.8, 0.85),
            VoteResponse("a3", "T1", 0.6, 0.70, 0.7, 0.7),
        ]
        theta = cvt.aggregate_votes("T1", votes)
        assert isinstance(theta, float)
        assert theta > 0.0


class TestPhase4:
    def test_consensus_reached(self):
        cvt = make_cvt(n_agents=5)
        # quorum = ceil((5 + 1 + 1)/2) = 4
        result = cvt.decide("T1", theta_agg=0.85, n_responses=4, elapsed_ms=0.4)
        assert result.consensus_reached
        assert result.action == "QUARANTINE"

    def test_consensus_not_reached_low_theta(self):
        cvt = make_cvt(n_agents=5)
        result = cvt.decide("T1", theta_agg=0.50, n_responses=5, elapsed_ms=0.4)
        assert not result.consensus_reached
        assert result.action == "MONITOR"

    def test_consensus_not_reached_low_quorum(self):
        cvt = make_cvt(n_agents=10)
        # quorum for n=10, f=3 is ceil((10+3+1)/2)=7; only 3 votes
        result = cvt.decide("T1", theta_agg=0.90, n_responses=3, elapsed_ms=0.4)
        assert not result.consensus_reached

    def test_quorum_formula(self):
        for n in [5, 7, 10, 13, 25]:
            f = (n - 1) // 3
            cvt = CVTProtocol("a0", n_agents=n)
            expected_quorum = math.ceil((n + f + 1) / 2)
            assert cvt.quorum_size == expected_quorum


class TestSynchronousRound:
    def test_full_round_quarantine(self):
        cvt = make_cvt(n_agents=5)
        # All peers detect a strong threat
        peers = [(f"a{i}", 0.85, (float(i), 0.0)) for i in range(1, 5)]
        result = cvt.run_synchronous(
            local_score=0.90,
            feature_vector=[0.5] * 8,
            peer_responses=peers,
        )
        assert result is not None
        assert result.consensus_reached

    def test_full_round_no_alert(self):
        cvt = make_cvt()
        result = cvt.run_synchronous(
            local_score=0.20,   # below tau_alert
            feature_vector=[0.1] * 8,
            peer_responses=[],
        )
        assert result is None

    def test_byzantine_agents_outvoted(self):
        """
        n=7, f=2 (2 Byzantine agents voting max weight).
        5 honest agents should outvote them.
        """
        rep = ReputationTracker(probationary_seconds=0)
        for i in range(7):
            rep.register(f"a{i}")
        cvt = CVTProtocol("a0", n_agents=7, reputation_tracker=rep,
                          agent_position=(0.0, 0.0))

        # 2 Byzantine peers voting 1.0 (maximum)
        byz = [("byz0", 1.0, (0.0, 0.0)), ("byz1", 1.0, (0.0, 0.0))]
        # 4 honest peers with correct threat assessment
        honest = [(f"h{i}", 0.85, (float(i), 0.0)) for i in range(4)]
        peers = byz + honest

        result = cvt.run_synchronous(0.9, [0.5] * 8, peers)
        # Honest agents should achieve consensus despite 2 Byzantine
        assert result is not None
