"""
In-process Multi-Agent Testbed.

Simulates a swarm of DMAS agents sharing a common P2P message bus.
Agents are arranged on a 2D grid topology to exercise the
distance-weighted voting mechanism.

Paper reference — Section V (Experimental Methodology):
  '25 Intel NUC mini PCs used as Edge Gateways, each managing
   approximately 80 virtualized devices.'
"""

from __future__ import annotations
import logging
import math
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from dmas.agent import DMASAgent
from dmas.communication.p2p_protocol import DMASMessage, MessageType
from dmas.simulation.attack_generator import AttackEvent, AttackGenerator, AttackType

logger = logging.getLogger(__name__)


@dataclass
class SimResult:
    """Aggregated results from one simulation run."""
    n_events: int = 0
    n_attacks: int = 0
    n_normals: int = 0
    n_detected: int = 0
    n_quarantined: int = 0
    n_false_positives: int = 0
    n_false_negatives: int = 0
    total_elapsed_ms: float = 0.0
    results_by_attack_type: Dict[str, Dict] = field(default_factory=dict)

    @property
    def detection_accuracy(self) -> float:
        if self.n_events == 0:
            return 0.0
        tp = self.n_detected
        tn = self.n_normals - self.n_false_positives
        return (tp + tn) / self.n_events

    @property
    def false_positive_rate(self) -> float:
        if self.n_normals == 0:
            return 0.0
        return self.n_false_positives / self.n_normals

    @property
    def precision(self) -> float:
        denom = self.n_detected + self.n_false_positives
        return self.n_detected / denom if denom > 0 else 0.0

    @property
    def recall(self) -> float:
        denom = self.n_attacks
        return self.n_detected / denom if denom > 0 else 0.0

    @property
    def f1(self) -> float:
        p, r = self.precision, self.recall
        return 2 * p * r / (p + r) if (p + r) > 0 else 0.0

    @property
    def avg_response_ms(self) -> float:
        return self.total_elapsed_ms / max(1, self.n_events)

    def print_summary(self) -> None:
        print("\n" + "=" * 60)
        print("  DMAS Simulation Results")
        print("=" * 60)
        print(f"  Total events        : {self.n_events}")
        print(f"  Attacks             : {self.n_attacks}")
        print(f"  Normal              : {self.n_normals}")
        print(f"  Detected (TP)       : {self.n_detected}")
        print(f"  Quarantined         : {self.n_quarantined}")
        print(f"  False positives     : {self.n_false_positives}")
        print(f"  False negatives     : {self.n_attacks - self.n_detected}")
        print(f"  Detection accuracy  : {self.detection_accuracy:.1%}")
        print(f"  False positive rate : {self.false_positive_rate:.1%}")
        print(f"  Precision           : {self.precision:.1%}")
        print(f"  Recall              : {self.recall:.1%}")
        print(f"  F1-Score            : {self.f1:.3f}")
        print(f"  Avg response time   : {self.avg_response_ms:.3f} ms")
        if self.results_by_attack_type:
            print("\n  Detection by attack type:")
            for atype, stats in self.results_by_attack_type.items():
                dr = stats["detected"] / stats["total"] if stats["total"] > 0 else 0.0
                print(f"    {atype:<12} {stats['detected']:>3}/{stats['total']:<3}  ({dr:.0%})")
        print("=" * 60)


class DMASTestbed:
    """
    Simulates a swarm of DMAS agents on a shared in-process message bus.

    Agents are placed on a 2D grid so that proximity-weighted voting
    naturally favours agents nearest the threat source.

    Parameters
    ----------
    n_agents : int
        Number of edge-gateway agents (paper uses 25).
    devices_per_agent : int
        IIoT devices monitored by each agent (paper: ~80).
    byzantine_fraction : float
        Fraction of agents to mark as Byzantine (randomly voting).
    seed : int
    """

    def __init__(
        self,
        n_agents: int = 5,
        devices_per_agent: int = 20,
        byzantine_fraction: float = 0.0,
        seed: int = 42,
        log_dir: str = "logs",
    ) -> None:
        self.n_agents = n_agents
        self.devices_per_agent = devices_per_agent
        self.byzantine_fraction = byzantine_fraction
        self.seed = seed
        self.log_dir = log_dir

        self._rng = np.random.default_rng(seed)
        self._sim_bus: Dict[str, "P2PProtocol"] = {}   # shared message bus

        # Build agents on a 2D grid
        self.agents: Dict[str, DMASAgent] = {}
        self._device_to_agent: Dict[str, str] = {}
        self._byzantine_agents: set = set()

        self._setup_agents()
        self._setup_devices()

        # Attack generator
        self.attack_gen = AttackGenerator(
            seed=seed, attack_rate_per_min=5.0
        )

        logger.info("Testbed ready: %d agents, %d devices total, %.0f%% Byzantine",
                    n_agents, n_agents * devices_per_agent, byzantine_fraction * 100)

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------

    def _setup_agents(self) -> None:
        n_byz = max(0, int(self.n_agents * self.byzantine_fraction))
        byz_indices = set(self._rng.choice(
            self.n_agents, size=n_byz, replace=False
        ).tolist()) if n_byz > 0 else set()

        grid_side = math.ceil(math.sqrt(self.n_agents))

        for i in range(self.n_agents):
            agent_id = f"agent_{i:02d}"
            row, col = divmod(i, grid_side)
            position = (float(col), float(row))
            is_byz = i in byz_indices

            if is_byz:
                self._byzantine_agents.add(agent_id)

            agent = DMASAgent(
                agent_id=agent_id,
                n_agents=self.n_agents,
                position=position,
                simulation_mode=True,
                log_dir=self.log_dir,
                seed=self.seed + i,
            )

            # If Byzantine, monkey-patch the peer_score_fn to return random votes
            if is_byz:
                rng = self._rng
                agent.cvt.peer_score_fn = lambda fv, _r=rng: float(_r.uniform(0, 1))
                logger.debug("Agent %s configured as Byzantine", agent_id)

            # Register on shared simulation bus
            agent.comm.attach_sim_bus(self._sim_bus)
            self.agents[agent_id] = agent

    def _setup_devices(self) -> None:
        device_classes = ["plc", "scada", "camera", "sensor"]
        idx = 0
        for agent_id in self.agents:
            for j in range(self.devices_per_agent):
                device_id = f"dev_{idx:04d}"
                self._device_to_agent[device_id] = agent_id
                idx += 1

    # ------------------------------------------------------------------
    # Simulation runner
    # ------------------------------------------------------------------

    def run(
        self,
        n_events: int = 200,
        attack_fraction: float = 0.20,
        verbose: bool = True,
    ) -> SimResult:
        """
        Run the simulation for `n_events` device observations.

        Returns a SimResult with accuracy, FPR, F1, and response-time stats.
        """
        device_ids = list(self._device_to_agent.keys())
        device_classes = self._assign_device_classes(device_ids)

        events = self.attack_gen.generate_batch(
            device_ids=device_ids,
            device_classes=device_classes,
            n_events=n_events,
            attack_fraction=attack_fraction,
        )

        result = SimResult()
        type_stats: Dict[str, Dict] = {}

        for event in events:
            t0 = time.perf_counter()
            action = self._process_event(event)
            elapsed_ms = (time.perf_counter() - t0) * 1000

            result.n_events += 1
            result.total_elapsed_ms += elapsed_ms

            atype = event.attack_type.value
            if atype not in type_stats:
                type_stats[atype] = {"total": 0, "detected": 0}
            type_stats[atype]["total"] += 1

            if event.ground_truth:
                result.n_attacks += 1
                if action in ("QUARANTINE", "MONITOR"):
                    result.n_detected += 1
                    type_stats[atype]["detected"] += 1
                    if action == "QUARANTINE":
                        result.n_quarantined += 1
                else:
                    result.n_false_negatives += 1
            else:
                result.n_normals += 1
                if action == "QUARANTINE":
                    result.n_false_positives += 1

        result.results_by_attack_type = {
            k: v for k, v in type_stats.items() if k != "normal"
        }

        if verbose:
            result.print_summary()

        return result

    def _process_event(self, event: AttackEvent) -> Optional[str]:
        """Route one event to the responsible agent and run CVT if needed."""
        agent_id = self._device_to_agent.get(event.device_id)
        if agent_id is None:
            return None

        agent = self.agents[agent_id]
        threat_id = agent.process_observation(event.observation)

        if threat_id is None:
            return None

        # Allow brief time for peers to respond (synchronous sim: immediate)
        # In real async mode this would be an asyncio.sleep(delta_t_ms / 1000)
        action = agent.finalize_vote(
            threat_id=threat_id,
            device_id=event.device_id,
            evidence={
                "attack_type": event.attack_type.value,
                "matched_sigs": len(event.observation.payload),
            },
        )
        return action

    # ------------------------------------------------------------------
    # Byzantine stress test
    # ------------------------------------------------------------------

    def run_byzantine_sweep(
        self, byzantine_fractions: Optional[List[float]] = None, n_events: int = 300
    ) -> List[Dict]:
        """
        Sweep over increasing Byzantine fractions and collect accuracy metrics.
        Reproduces Table II from the paper.
        """
        if byzantine_fractions is None:
            byzantine_fractions = [0.0, 0.10, 0.20, 0.30, 0.40]

        rows = []
        print("\n  Byzantine Fault Tolerance Sweep")
        print(f"  {'Byz%':>6}  {'Accuracy':>9}  {'FPR':>7}  {'F1':>6}")
        print("  " + "-" * 35)

        for frac in byzantine_fractions:
            tb = DMASTestbed(
                n_agents=self.n_agents,
                devices_per_agent=self.devices_per_agent,
                byzantine_fraction=frac,
                seed=self.seed,
                log_dir=self.log_dir,
            )
            r = tb.run(n_events=n_events, verbose=False)
            row = {
                "byzantine_pct": int(frac * 100),
                "accuracy": round(r.detection_accuracy, 4),
                "fpr": round(r.false_positive_rate, 4),
                "f1": round(r.f1, 4),
                "avg_response_ms": round(r.avg_response_ms, 3),
            }
            rows.append(row)
            print(f"  {row['byzantine_pct']:>5}%  "
                  f"{row['accuracy']:>8.1%}  "
                  f"{row['fpr']:>6.1%}  "
                  f"{row['f1']:>6.3f}")

        return rows

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _assign_device_classes(self, device_ids: List[str]) -> List[str]:
        classes = ["plc", "scada", "camera", "sensor"]
        return [classes[i % len(classes)] for i in range(len(device_ids))]

    def get_agent(self, agent_id: str) -> Optional[DMASAgent]:
        return self.agents.get(agent_id)

    def swarm_summary(self) -> List[Dict]:
        return [agent.summary() for agent in self.agents.values()]

    def __repr__(self) -> str:
        return (f"DMASTestbed(n_agents={self.n_agents}, "
                f"n_devices={self.n_agents * self.devices_per_agent}, "
                f"byzantine={len(self._byzantine_agents)})")
