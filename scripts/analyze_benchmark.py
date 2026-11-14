#!/usr/bin/env python3
"""Compare adaptive runs with fixed baselines from one or more summary CSVs."""

import argparse
import csv
import math
from pathlib import Path


METRICS = (
    "ate_rmse_similarity_aligned",
    "rpe_translation_rmse_similarity_aligned",
    "tracking_failure_rate",
    "mean_latency_ms",
    "p95_latency_ms",
    "peak_rss_bytes",
)
BASELINES = ("fixed_low", "fixed_high")


def number(row, field):
    try:
        value = float(row[field])
    except (KeyError, TypeError, ValueError):
        return None
    return value if math.isfinite(value) and value >= 0 else None


def load_runs(paths):
    runs = {}
    for path in paths:
        with path.open(newline="", encoding="utf-8-sig") as source:
            for row in csv.DictReader(source):
                key = (row["sequence"], row["degradation"], row["configuration"])
                if key in runs:
                    raise ValueError(f"duplicate run in input summaries: {key}")
                runs[key] = row
    return runs


def compare_runs(runs):
    comparisons = []
    scenarios = sorted({key[:2] for key in runs if key[2] == "adaptive"})
    for sequence, degradation in scenarios:
        adaptive = runs[(sequence, degradation, "adaptive")]
        for baseline_name in BASELINES:
            baseline = runs.get((sequence, degradation, baseline_name))
            if baseline is None:
                continue
            row = {
                "sequence": sequence,
                "degradation": degradation,
                "baseline": baseline_name,
            }
            better = False
            worse = False
            available = []
            for metric in METRICS:
                adaptive_value = number(adaptive, metric)
                baseline_value = number(baseline, metric)
                row[f"adaptive_{metric}"] = adaptive_value
                row[f"baseline_{metric}"] = baseline_value
                delta = None
                percent = None
                if adaptive_value is not None and baseline_value is not None:
                    available.append(metric)
                    delta = adaptive_value - baseline_value
                    if math.isclose(adaptive_value, baseline_value, rel_tol=1e-9, abs_tol=1e-12):
                        continue
                    if adaptive_value < baseline_value:
                        better = True
                    else:
                        worse = True
                    if baseline_value != 0:
                        percent = 100.0 * delta / abs(baseline_value)
                row[f"delta_{metric}"] = delta
                row[f"change_percent_{metric}"] = percent
            row["metrics_compared"] = ";".join(available)
            if not available:
                row["result"] = "not_comparable"
            elif better and not worse:
                row["result"] = "adaptive_dominates"
            elif worse and not better:
                row["result"] = "adaptive_is_dominated"
            elif better and worse:
                row["result"] = "tradeoff"
            else:
                row["result"] = "no_change"
            comparisons.append(row)
    return comparisons


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", nargs="+", required=True, type=Path,
                        help="one or more benchmark summary.csv files")
    parser.add_argument("--output", required=True, type=Path,
                        help="comparison CSV to write")
    args = parser.parse_args()

    comparisons = compare_runs(load_runs(args.input))
    if not comparisons:
        raise SystemExit("no adaptive runs with matching fixed baselines were found")
    fields = ["sequence", "degradation", "baseline", "metrics_compared", "result"]
    for metric in METRICS:
        fields.extend((
            f"adaptive_{metric}",
            f"baseline_{metric}",
            f"delta_{metric}",
            f"change_percent_{metric}",
        ))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as destination:
        writer = csv.DictWriter(destination, fieldnames=fields)
        writer.writeheader()
        writer.writerows(comparisons)
    print(f"Wrote {len(comparisons)} paired comparisons to {args.output}")


if __name__ == "__main__":
    main()
