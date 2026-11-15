#!/usr/bin/env python3
"""Download and validate full official EuRoC MAV sequences.

This downloader never substitutes synthetic data when a download fails.
"""

import argparse
import csv
import os
from pathlib import Path
import tempfile
import urllib.request
import zipfile


EUROC_URLS = {
    "MH_01_easy": "https://robotics.ethz.ch/~asl-datasets/ijrr_euroc_mav_dataset/machine_hall/MH_01_easy/MH_01_easy.zip",
    "MH_02_easy": "https://robotics.ethz.ch/~asl-datasets/ijrr_euroc_mav_dataset/machine_hall/MH_02_easy/MH_02_easy.zip",
    "MH_03_medium": "https://robotics.ethz.ch/~asl-datasets/ijrr_euroc_mav_dataset/machine_hall/MH_03_medium/MH_03_medium.zip",
    "MH_04_difficult": "https://robotics.ethz.ch/~asl-datasets/ijrr_euroc_mav_dataset/machine_hall/MH_04_difficult/MH_04_difficult.zip",
    "MH_05_difficult": "https://robotics.ethz.ch/~asl-datasets/ijrr_euroc_mav_dataset/machine_hall/MH_05_difficult/MH_05_difficult.zip",
    "V1_01_easy": "https://robotics.ethz.ch/~asl-datasets/ijrr_euroc_mav_dataset/vicon_room1/V1_01_easy/V1_01_easy.zip",
    "V1_02_medium": "https://robotics.ethz.ch/~asl-datasets/ijrr_euroc_mav_dataset/vicon_room1/V1_02_medium/V1_02_medium.zip",
    "V1_03_difficult": "https://robotics.ethz.ch/~asl-datasets/ijrr_euroc_mav_dataset/vicon_room1/V1_03_difficult/V1_03_difficult.zip",
    "V2_01_easy": "https://robotics.ethz.ch/~asl-datasets/ijrr_euroc_mav_dataset/vicon_room2/V2_01_easy/V2_01_easy.zip",
    "V2_02_medium": "https://robotics.ethz.ch/~asl-datasets/ijrr_euroc_mav_dataset/vicon_room2/V2_02_medium/V2_02_medium.zip",
}

MIN_FULL_SEQUENCE_FRAMES = 1000


def data_rows(path):
    with path.open(newline="", encoding="utf-8-sig") as source:
        return [
            row for row in csv.reader(source)
            if row and not row[0].lstrip().startswith("#")
        ]


def validate_euroc_sequence(seq_dir):
    camera_dir = seq_dir / "mav0" / "cam0"
    frame_csv = camera_dir / "data.csv"
    imu_csv = seq_dir / "mav0" / "imu0" / "data.csv"
    ground_truth_csv = (
        seq_dir / "mav0" / "state_groundtruth_estimate0" / "data.csv"
    )
    if not (camera_dir / "sensor.yaml").is_file() or not frame_csv.is_file():
        raise ValueError(f"missing camera calibration or index in {seq_dir}")
    if not imu_csv.is_file() or not ground_truth_csv.is_file():
        raise ValueError(f"missing EuRoC IMU or ground-truth data in {seq_dir}")

    frames = data_rows(frame_csv)
    if len(frames) < MIN_FULL_SEQUENCE_FRAMES:
        raise ValueError(
            f"{seq_dir.name} has {len(frames)} camera frames; expected at least "
            f"{MIN_FULL_SEQUENCE_FRAMES} for a full sequence"
        )
    previous_timestamp = -1
    for row in frames:
        if len(row) < 2:
            raise ValueError(f"invalid camera CSV row in {frame_csv}")
        timestamp = int(row[0])
        if timestamp <= previous_timestamp:
            raise ValueError(f"camera timestamps are not increasing in {frame_csv}")
        previous_timestamp = timestamp
        if not (camera_dir / "data" / row[1]).is_file():
            raise ValueError(f"missing camera image {row[1]} in {seq_dir}")

    if len(data_rows(imu_csv)) < len(frames):
        raise ValueError(f"IMU samples are incomplete in {seq_dir}")
    if len(data_rows(ground_truth_csv)) < MIN_FULL_SEQUENCE_FRAMES:
        raise ValueError(f"ground truth is incomplete in {seq_dir}")
    return len(frames)


def find_extracted_sequence(extract_dir):
    for sensor_yaml in extract_dir.rglob("sensor.yaml"):
        if sensor_yaml.parent.name != "cam0":
            continue
        sequence_root = sensor_yaml.parent.parent.parent
        if (sequence_root / "mav0" / "cam0" / "data.csv").is_file():
            return sequence_root
    raise ValueError("archive does not contain a EuRoC camera sequence")


def download_archive(url, destination):
    request = urllib.request.Request(
        url, headers={"User-Agent": "adaptive-slam-euroc-downloader/1.0"}
    )
    with urllib.request.urlopen(request, timeout=30) as response:
        total_size = response.headers.get("Content-Length")
        total_size = int(total_size) if total_size else 0
        downloaded = 0
        with destination.open("wb") as target:
            while True:
                block = response.read(1024 * 1024)
                if not block:
                    break
                target.write(block)
                downloaded += len(block)
                if total_size:
                    percent = min(100, downloaded * 100 // total_size)
                    print(
                        f"\rDownloading: {percent}% "
                        f"[{downloaded / (1024 * 1024):.1f} MB / "
                        f"{total_size / (1024 * 1024):.1f} MB]",
                        end="",
                        flush=True,
                    )
        print()


def download_sequence(name, output_dir, force_download):
    sequence_dir = output_dir / name
    if sequence_dir.exists() and not force_download:
        frame_count = validate_euroc_sequence(sequence_dir)
        print(f"Full EuRoC sequence {name} already exists ({frame_count} frames).")
        return

    with tempfile.TemporaryDirectory(
        prefix=f".{name}.download-", dir=output_dir
    ) as temporary:
        temporary_dir = Path(temporary)
        archive_path = temporary_dir / f"{name}.zip"
        extraction_dir = temporary_dir / "extracted"
        extraction_dir.mkdir()
        backup_dir = temporary_dir / "previous"

        try:
            url = EUROC_URLS[name]
            print(f"Downloading official {name} from {url}")
            download_archive(url, archive_path)
            with zipfile.ZipFile(archive_path, "r") as archive:
                invalid_member = archive.testzip()
                if invalid_member is not None:
                    raise ValueError(f"archive CRC failed for {invalid_member}")
                extraction_root = extraction_dir.resolve()
                for member in archive.infolist():
                    destination = (extraction_dir / member.filename).resolve()
                    if (destination != extraction_root and
                            extraction_root not in destination.parents):
                        raise ValueError(
                            f"archive contains an unsafe path: {member.filename}"
                        )
                archive.extractall(extraction_dir)

            downloaded_sequence = find_extracted_sequence(extraction_dir)
            frame_count = validate_euroc_sequence(downloaded_sequence)

            if sequence_dir.exists():
                os.replace(sequence_dir, backup_dir)
            try:
                os.replace(downloaded_sequence, sequence_dir)
            except Exception:
                if backup_dir.exists():
                    os.replace(backup_dir, sequence_dir)
                raise
        except Exception as error:
            raise RuntimeError(
                f"official download/validation failed for {name}: {error}; "
                "existing data was preserved and no synthetic fallback was used"
            ) from error

    print(f"Installed official {name} ({frame_count} frames) at {sequence_dir}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--sequence", choices=list(EUROC_URLS) + ["all"], default="all"
    )
    parser.add_argument("--output", type=Path, default=Path("datasets"))
    parser.add_argument(
        "--force-download", action="store_true",
        help="download and replace an existing sequence after validation",
    )
    args = parser.parse_args()

    output_dir = args.output.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    targets = list(EUROC_URLS) if args.sequence == "all" else [args.sequence]
    for name in targets:
        try:
            download_sequence(name, output_dir, args.force_download)
        except (OSError, ValueError, RuntimeError) as error:
            parser.error(str(error))


if __name__ == "__main__":
    main()
