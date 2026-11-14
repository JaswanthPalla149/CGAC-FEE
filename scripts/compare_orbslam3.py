#!/usr/bin/env python3
"""
Compare Adaptive SLAM benchmark runs against the ORB-SLAM3 baseline.
Produces results/orbslam3_baseline/comparison.csv and a formatted markdown report
highlighting trade-offs in accuracy, latency, memory footprint, and architectural scope.
"""

import argparse
import csv
from pathlib import Path
import sys

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--adaptive-summary", required=True, type=Path, help="Summary CSV from adaptive_slam_benchmark")
    parser.add_argument("--orbslam3-summary", required=True, type=Path, help="Summary CSV from run_orbslam3_baseline.py")
    parser.add_argument("--output", type=Path, default=Path("results/orbslam3_baseline/comparison.csv"))
    args = parser.parse_args()

    args.output.parent.mkdir(parents=True, exist_ok=True)

    with args.orbslam3_summary.open(newline="", encoding="utf-8") as f:
        orb_rows = {row["sequence"]: row for row in csv.DictReader(f)}

    with args.adaptive_summary.open(newline="", encoding="utf-8-sig") as f:
        adapt_rows = [row for row in csv.DictReader(f) if row.get("configuration") == "adaptive"]

    comparisons = []
    print("\n=========================================================================================")
    print("                      ADAPTIVE SLAM vs. ORB-SLAM3 BASELINE COMPARISON                    ")
    print("=========================================================================================")
    print(f"{'Sequence':<15} | {'Degradation':<10} | {'Adaptive ATE':<12} | {'ORB3 ATE':<10} | {'Adaptive Lat':<12} | {'ORB3 Lat':<10} | {'Memory Ratio':<12}")
    print("-" * 89)

    for row in adapt_rows:
        seq = row["sequence"]
        deg = row["degradation"]
        if seq in orb_rows:
            orb = orb_rows[seq]
            ate_adapt = float(row.get("ate_rmse_similarity_aligned", "nan"))
            ate_orb = float(orb.get("ate_rmse_similarity_aligned", "nan"))
            lat_adapt = float(row.get("mean_latency_ms", "nan"))
            lat_orb = float(orb.get("mean_latency_ms", "nan"))
            rss_adapt = float(row.get("peak_rss_bytes", "0")) / (1024 * 1024)
            rss_orb = float(orb.get("peak_rss_bytes", "0")) / (1024 * 1024)
            rss_ratio = f"{rss_orb / max(1.0, rss_adapt):.1f}x lower"

            comp_row = {
                "sequence": seq,
                "degradation": deg,
                "adaptive_ate_m": f"{ate_adapt:.4f}",
                "orbslam3_ate_m": f"{ate_orb:.4f}",
                "delta_ate_m": f"{ate_adapt - ate_orb:.4f}",
                "adaptive_latency_ms": f"{lat_adapt:.2f}",
                "orbslam3_latency_ms": f"{lat_orb:.2f}",
                "delta_latency_ms": f"{lat_adapt - lat_orb:.2f}",
                "adaptive_peak_rss_mb": f"{rss_adapt:.1f}",
                "orbslam3_peak_rss_mb": f"{rss_orb:.1f}",
                "memory_reduction_factor": f"{rss_orb / max(1.0, rss_adapt):.1f}x",
                "notes": "Adaptive SLAM is 1-thread lightweight VO; ORB-SLAM3 uses 3 threads + global BA + DBoW2"
            }
            comparisons.append(comp_row)
            print(f"{seq:<15} | {deg:<10} | {ate_adapt:<12.4f} | {ate_orb:<10.4f} | {lat_adapt:<12.2f} | {lat_orb:<10.2f} | {rss_ratio:<12}")

    with args.output.open("w", newline="", encoding="utf-8") as f:
        if comparisons:
            writer = csv.DictWriter(f, fieldnames=list(comparisons[0].keys()))
            writer.writeheader()
            writer.writerows(comparisons)

    print("-" * 89)
    print(f"Wrote comparison to {args.output}\n")

if __name__ == "__main__":
    main()

