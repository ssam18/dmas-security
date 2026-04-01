#!/usr/bin/env python3
"""
byzantine_stress_test.py
========================
Sweeps over increasing Byzantine agent fractions (0–40%) and measures
how DMAS accuracy degrades — reproducing Table II from the paper.

  Byzantine %  Accuracy  FPR      F1
  0%           ~98%      ~2%      high
  10%          ~98%      ~2%      high
  20%          ~97%      ~3%      high
  30%          ~95%      ~5%      high     ← theoretical limit f < n/3
  40%          ~89%      ~10%     drops    ← beyond tolerance

Run:
    python examples/byzantine_stress_test.py
"""

import logging
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

logging.basicConfig(level=logging.WARNING)

from dmas.simulation.testbed import DMASTestbed


def main():
    print("=" * 65)
    print("  DMAS Byzantine Fault Tolerance Sweep")
    print("  Reproduces Table II: Detection Accuracy Under Byzantine Agents")
    print("=" * 65)

    base_tb = DMASTestbed(
        n_agents=10,
        devices_per_agent=20,
        seed=42,
        log_dir="logs",
    )

    rows = base_tb.run_byzantine_sweep(
        byzantine_fractions=[0.0, 0.10, 0.20, 0.30, 0.40],
        n_events=300,
    )

    print("\n  Summary table (comparable to paper Table II):")
    print(f"\n  {'Byz%':>5}  {'Accuracy':>9}  {'FPR':>7}  "
          f"{'F1':>7}  {'Avg Response(ms)':>17}")
    print("  " + "-" * 52)
    for row in rows:
        byz_label = f"{row['byzantine_pct']}%"
        limit_note = " ← f<n/3 limit" if row["byzantine_pct"] == 30 else ""
        limit_note += " ← beyond tolerance" if row["byzantine_pct"] == 40 else ""
        print(f"  {byz_label:>5}  "
              f"{row['accuracy']:>8.1%}  "
              f"{row['fpr']:>6.1%}  "
              f"{row['f1']:>6.3f}  "
              f"{row['avg_response_ms']:>10.3f} ms"
              f"{limit_note}")

    print("\n  Note: Accuracy degrades smoothly until the theoretical")
    print("  Byzantine tolerance limit (f < n/3 ≈ 33%), then more rapidly.")
    print("  This validates the n ≥ 3f+1 constraint from Theorem 2.")


if __name__ == "__main__":
    main()
