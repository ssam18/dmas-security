#!/usr/bin/env python3
"""
multi_agent_demo.py
===================
Runs a full multi-agent DMAS swarm simulation, printing detection
accuracy, false-positive rate, F1, and per-attack-type breakdown —
directly comparable to the paper's Table I results.

Run:
    python examples/multi_agent_demo.py
"""

import logging
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

logging.basicConfig(level=logging.WARNING,
                    format="%(asctime)s  %(levelname)-7s  %(message)s")

from dmas.simulation.testbed import DMASTestbed


def main():
    print("=" * 65)
    print("  DMAS Multi-Agent Swarm Simulation")
    print("  5 agents · 100 devices · 400 events · 20% attack rate")
    print("=" * 65)

    tb = DMASTestbed(
        n_agents=5,
        devices_per_agent=20,
        byzantine_fraction=0.0,
        seed=42,
        log_dir="logs",
    )

    tb.run(n_events=400, attack_fraction=0.20, verbose=True)

    print("\n  Swarm Agent Summaries:")
    print(f"  {'Agent':<12}  {'Observations':>13}  {'Detections':>10}  "
          f"{'Quarantines':>12}  {'Reputation':>10}")
    print("  " + "-" * 65)
    for s in tb.swarm_summary():
        print(f"  {s['agent_id']:<12}  {s['n_observations']:>13}  "
              f"{s['n_threats_detected']:>10}  "
              f"{s['n_quarantines']:>12}  "
              f"{s['reputation_self']:>10.3f}")

    print("\n  Logs written to: logs/")
    print(f"  Total alert log entries: "
          f"{sum(1 for f in os.listdir('logs') if 'alerts' in f)} files")


if __name__ == "__main__":
    main()
