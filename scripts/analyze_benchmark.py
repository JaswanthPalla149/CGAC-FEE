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
    "energy_joules",
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
                key = (
                    row["sequence"],
                    row["degradation"],
                    row.get("repeat", "1"),
                    row["configuration"],
                )
                if key in runs:
                    raise ValueError(f"duplicate run in input summaries: {key}")
                runs[key] = row
    return runs


def compare_runs(runs):
    comparisons = []
    scenarios = sorted({key[:3] for key in runs if key[3] == "adaptive"})
    for sequence, degradation, repeat in scenarios:
        adaptive = runs[(sequence, degradation, repeat, "adaptive")]
        for baseline_name in BASELINES:
            baseline = runs.get((sequence, degradation, repeat, baseline_name))
            if baseline is None:
                continue
            row = {
                "sequence": sequence,
                "degradation": degradation,
                "repeat": repeat,
                "baseline": baseline_name,
            }
            better = False
            worse = False
            available = []
            for metric in METRICS:
                if metric == "energy_joules":
                    comparable_energy = (
                        adaptive.get("energy_status") == "measured"
                        and baseline.get("energy_status") == "measured"
                        and adaptive.get("energy_method") == baseline.get("energy_method")
                        and adaptive.get("energy_scope") == baseline.get("energy_scope")
                    )
                    adaptive_value = (
                        number(adaptive, metric) if comparable_energy else None
                    )
                    baseline_value = (
                        number(baseline, metric) if comparable_energy else None
                    )
                    row["energy_method"] = (
                        adaptive.get("energy_method", "not_comparable")
                        if comparable_energy else "not_comparable"
                    )
                    row["energy_scope"] = (
                        adaptive.get("energy_scope", "not_comparable")
                        if comparable_energy else "not_comparable"
                    )
                else:
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
    fields = [
        "sequence", "degradation", "repeat", "baseline",
        "metrics_compared", "result",
        "energy_method", "energy_scope",
    ]
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
