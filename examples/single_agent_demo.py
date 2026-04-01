#!/usr/bin/env python3
"""
single_agent_demo.py
====================
Demonstrates one DMAS agent processing a stream of device telemetry
observations — normal traffic interspersed with six attack types.

Run:
    python examples/single_agent_demo.py
"""

import logging
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
from dmas.agent import DMASAgent
from dmas.simulation.attack_generator import AttackGenerator, AttackType

logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
)

DEVICE_IDS = [f"plc_{i:02d}" for i in range(5)] + \
             [f"sensor_{i:02d}" for i in range(5)]

ATTACK_SEQUENCE = [
    (AttackType.NORMAL,    "PLC normal heartbeat"),
    (AttackType.NORMAL,    "Sensor normal reading"),
    (AttackType.DDOS,      "DDoS flood on PLC"),
    (AttackType.NORMAL,    "Normal traffic resumes"),
    (AttackType.MITM,      "Man-in-the-Middle interception"),
    (AttackType.NORMAL,    "Normal traffic"),
    (AttackType.REPLAY,    "Replay attack on Modbus"),
    (AttackType.INJECTION, "SQL/command injection"),
    (AttackType.NORMAL,    "Normal traffic"),
    (AttackType.MALWARE,   "EternalBlue malware propagation"),
    (AttackType.ZERO_DAY,  "Novel zero-day exploit"),
    (AttackType.NORMAL,    "Normal traffic"),
    (AttackType.NORMAL,    "Normal traffic"),
]


def main():
    print("=" * 65)
    print("  DMAS Single-Agent Demo")
    print("  Paper: 'Decentralized Multi-Agent Swarms for Grid Security'")
    print("=" * 65)

    gen = AttackGenerator(seed=42)

    # Single agent — no peer coordination in this demo
    agent = DMASAgent(
        agent_id="gateway_00",
        n_agents=1,
        position=(0.0, 0.0),
        simulation_mode=True,
        log_dir="logs",
        tau_alert=0.45,
    )

    # Warm up EWMA with 50 normal samples before the demo
    print("\n  [Warming up EWMA baseline with 50 normal observations...]")
    for i in range(50):
        device_id = DEVICE_IDS[i % len(DEVICE_IDS)]
        event = gen.generate(device_id, "sensor", AttackType.NORMAL)
        agent.monitoring.observe(event.observation)
    print("  [Baseline established]\n")

    print(f"  {'#':>3}  {'Description':<38}  {'θ_s':>5}  {'θ_b':>5}  "
          f"{'θ_m':>5}  {'θ':>5}  {'Alert':>7}")
    print("  " + "-" * 72)

    for idx, (attack_type, description) in enumerate(ATTACK_SEQUENCE):
        device_id = DEVICE_IDS[idx % len(DEVICE_IDS)]
        event = gen.generate(device_id, "plc", attack_type)
        assessment = agent.monitoring.observe(event.observation)

        alert_str = "⚠ ALERT" if assessment.is_alert else "  ok   "
        print(f"  {idx+1:>3}  {description:<38}  "
              f"{assessment.score_statistical:>5.3f}  "
              f"{assessment.score_behavioral:>5.3f}  "
              f"{assessment.score_signature:>5.3f}  "
              f"{assessment.score_composite:>5.3f}  {alert_str}")

    print("\n  Agent summary:")
    summary = agent.summary()
    for k, v in summary.items():
        print(f"    {k:<30} {v}")


if __name__ == "__main__":
    main()
