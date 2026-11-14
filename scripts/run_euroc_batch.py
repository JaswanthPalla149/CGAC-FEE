#!/usr/bin/env python3
"""Run the native benchmark across EuRoC sequences found under a dataset root."""

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import subprocess


DEGRADATIONS = ("none", "blur", "low_light", "noise")


def find_sequences(root):
    root = root.resolve()
    candidates = [root]
    candidates.extend(sorted(path for path in root.iterdir() if path.is_dir()))
    return [
        path for path in candidates
        if (path / "mav0" / "cam0" / "sensor.yaml").is_file()
    ]


def parse_degradations(value):
    degradations = [part.strip() for part in value.split(",") if part.strip()]
    invalid = sorted(set(degradations) - set(DEGRADATIONS))
    if not degradations or invalid:
        raise argparse.ArgumentTypeError(
            "expected a comma-separated subset of " + ",".join(DEGRADATIONS)
        )
    return degradations


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark", required=True, type=Path,
                        help="path to adaptive_slam_benchmark executable")
    parser.add_argument("--dataset-root", required=True, type=Path,
                        help="EuRoC sequence directory or directory of sequences")
    parser.add_argument("--output", required=True, type=Path,
                        help="campaign output directory")
    parser.add_argument("--degradations", type=parse_degradations,
                        default=list(DEGRADATIONS))
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=1,
                        help="number of complete benchmark repetitions")
    parser.add_argument("--enter-recovery-below", type=float, default=0.48)
    parser.add_argument("--exit-recovery-above", type=float, default=0.60)
    args = parser.parse_args()

    benchmark = args.benchmark.resolve()
    dataset_root = args.dataset_root.resolve()
    output = args.output.resolve()
    if not benchmark.is_file():
        parser.error(f"benchmark executable does not exist: {benchmark}")
    if args.threads < 1:
        parser.error("--threads must be a positive integer")
    if args.repeats < 1:
        parser.error("--repeats must be a positive integer")
    if not (0 <= args.enter_recovery_below < args.exit_recovery_above <= 1):
        parser.error("controller thresholds must satisfy 0 <= enter < exit <= 1")
    if not dataset_root.is_dir():
        parser.error(f"dataset root does not exist: {dataset_root}")
    sequences = find_sequences(dataset_root)
    if not sequences:
        parser.error(f"no EuRoC sequences found under {dataset_root}")
    names = [path.name for path in sequences]
    if len(set(names)) != len(names):
        parser.error("sequence directory names must be unique for campaign output")

    output.mkdir(parents=True, exist_ok=True)
    all_rows = []
    for sequence in sequences:
        sequence_output = output / sequence.name
        for repeat in range(1, args.repeats + 1):
            repeat_output = (
                sequence_output if args.repeats == 1
                else sequence_output / f"repeat_{repeat:02d}"
            )
            command = [
                str(benchmark),
                "--sequence", str(sequence),
                "--output", str(repeat_output),
                "--degradations", ",".join(args.degradations),
                "--threads", str(args.threads),
                "--enter-recovery-below", str(args.enter_recovery_below),
                "--exit-recovery-above", str(args.exit_recovery_above),
            ]
            print(
                f"Running {sequence.name} repeat {repeat}/{args.repeats}: "
                f"{' '.join(command)}",
                flush=True,
            )
            subprocess.run(command, check=True)
            with (repeat_output / "summary.csv").open(
                newline="", encoding="utf-8-sig"
            ) as source:
                for row in csv.DictReader(source):
                    row["repeat"] = str(repeat)
                    all_rows.append(row)

    summary_path = output / "summary.csv"
    if all_rows:
        with summary_path.open("w", newline="", encoding="utf-8") as destination:
            writer = csv.DictWriter(destination, fieldnames=list(all_rows[0]))
            writer.writeheader()
            writer.writerows(all_rows)
    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "platform": platform.platform(),
        "benchmark": str(benchmark),
        "benchmark_sha256": sha256_file(benchmark),
        "dataset_root": str(dataset_root),
        "sequences": [str(path) for path in sequences],
        "degradations": args.degradations,
        "opencv_threads": args.threads,
        "repeats": args.repeats,
        "controller_enter_threshold": args.enter_recovery_below,
        "controller_exit_threshold": args.exit_recovery_above,
        "degradation_seed": 17,
        "summary_csv": str(summary_path),
    }
    with (output / "campaign.json").open("w", encoding="utf-8") as destination:
        json.dump(manifest, destination, indent=2)
        destination.write("\n")
    print(f"Wrote {len(all_rows)} run summaries to {summary_path}")


if __name__ == "__main__":
    main()
