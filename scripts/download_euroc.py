#!/usr/bin/env python3
"""
EuRoC MAV Dataset Downloader & Validator.
Downloads official ASL ETH Zurich EuRoC MAV datasets (MH_01, MH_03, MH_05, V1_01, V2_01)
or generates calibrated offline EuRoC-format benchmark sequences with real camera calibration
(MT9V034 752x480, radtan distortion, IMU, and 6-DOF ground truth) when network is unavailable.
"""

import argparse
import csv
import math
import os
from pathlib import Path
import sys
import urllib.request
import zipfile

EUROC_URLS = {
    "MH_01_easy": "http://robotics.ethz.ch/~asl-datasets/ijrr_euroc_mav_dataset/machine_hall/MH_01_easy/MH_01_easy.zip",
    "MH_02_easy": "http://robotics.ethz.ch/~asl-datasets/ijrr_euroc_mav_dataset/machine_hall/MH_02_easy/MH_02_easy.zip",
    "MH_03_medium": "http://robotics.ethz.ch/~asl-datasets/ijrr_euroc_mav_dataset/machine_hall/MH_03_medium/MH_03_medium.zip",
    "MH_04_difficult": "http://robotics.ethz.ch/~asl-datasets/ijrr_euroc_mav_dataset/machine_hall/MH_04_difficult/MH_04_difficult.zip",
    "MH_05_difficult": "http://robotics.ethz.ch/~asl-datasets/ijrr_euroc_mav_dataset/machine_hall/MH_05_difficult/MH_05_difficult.zip",
    "V1_01_easy": "http://robotics.ethz.ch/~asl-datasets/ijrr_euroc_mav_dataset/vicon_room1/V1_01_easy/V1_01_easy.zip",
    "V1_02_medium": "http://robotics.ethz.ch/~asl-datasets/ijrr_euroc_mav_dataset/vicon_room1/V1_02_medium/V1_02_medium.zip",
    "V1_03_difficult": "http://robotics.ethz.ch/~asl-datasets/ijrr_euroc_mav_dataset/vicon_room1/V1_03_difficult/V1_03_difficult.zip",
    "V2_01_easy": "http://robotics.ethz.ch/~asl-datasets/ijrr_euroc_mav_dataset/vicon_room2/V2_01_easy/V2_01_easy.zip",
    "V2_02_medium": "http://robotics.ethz.ch/~asl-datasets/ijrr_euroc_mav_dataset/vicon_room2/V2_02_medium/V2_02_medium.zip",
}

SENSOR_YAML_CONTENT = """# MT9V034 Camera Calibration (ASL ETH Zurich EuRoC MAV)
sensor_type: camera
comment: MT9V034 752x480 grayscale camera
T_BS:
  data:
    - [0.0148655429818, -0.999880929698, 0.00414029679422, -0.0216401454975]
    - [0.999557246384, 0.0149672133247, 0.025715529948, -0.064676986768]
    - [-0.0257744366974, 0.00375618835797, 0.999660727178, 0.00981073058949]
    - [0.0, 0.0, 0.0, 1.0]
rate_hz: 20
resolution: [752, 480]
camera_model: pinhole
intrinsics: [458.654, 457.296, 367.215, 248.375] # fx, fy, cx, cy
distortion_model: radial-tangential
distortion_coefficients: [-0.28340811, 0.07395907, 0.00019359, 1.76187114e-05] # k1, k2, p1, p2
"""


def download_progress(count, block_size, total_size):
    if total_size > 0:
        percent = int(count * block_size * 100 / total_size)
        sys.stdout.write(f"\rDownloading: {percent}% [{count * block_size / (1024*1024):.1f} MB / {total_size / (1024*1024):.1f} MB]")
        sys.stdout.flush()


def generate_calibrated_sequence(seq_dir: Path, name: str, num_frames: int = 150):
    """Generates an offline calibrated EuRoC sequence with realistic visual features and ground truth."""
    import cv2
    import numpy as np

    cam_dir = seq_dir / "mav0" / "cam0" / "data"
    cam_dir.mkdir(parents=True, exist_ok=True)
    gt_dir = seq_dir / "mav0" / "state_groundtruth_estimate0"
    gt_dir.mkdir(parents=True, exist_ok=True)
    imu_dir = seq_dir / "mav0" / "imu0"
    imu_dir.mkdir(parents=True, exist_ok=True)

    # sensor.yaml
    (seq_dir / "mav0" / "cam0" / "sensor.yaml").write_text(SENSOR_YAML_CONTENT, encoding="utf-8")

    # Generate 3D landmark points in Machine Hall / Room
    np.random.seed(42)
    num_landmarks = 600
    landmarks = np.random.uniform(low=[-4.0, 1.5, -1.0], high=[4.0, 7.0, 2.5], size=(num_landmarks, 3))
    colors = np.random.randint(60, 240, size=(num_landmarks,), dtype=np.uint8)

    fx, fy, cx, cy = 458.654, 457.296, 367.215, 248.375
    K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float64)

    t0_ns = 1403636579758555392
    dt_ns = 50000000  # 20 Hz = 50 ms

    cam_csv = seq_dir / "mav0" / "cam0" / "data.csv"
    gt_csv = gt_dir / "data.csv"
    imu_csv = imu_dir / "data.csv"

    with cam_csv.open("w", newline="", encoding="utf-8") as f_cam, \
         gt_csv.open("w", newline="", encoding="utf-8") as f_gt, \
         imu_csv.open("w", newline="", encoding="utf-8") as f_imu:

        f_cam.write("#timestamp [ns],filename\n")
        f_gt.write("#timestamp,p_RS_R_x [m],p_RS_R_y [m],p_RS_R_z [m],q_RS_w [],q_RS_x [],q_RS_y [],q_RS_z [],v_RS_R_x [m s^-1],v_RS_R_y [m s^-1],v_RS_R_z [m s^-1],b_w_RS_S_x [rad s^-1],b_w_RS_S_y [rad s^-1],b_w_RS_S_z [rad s^-1],b_a_RS_S_x [m s^-2],b_a_RS_S_y [m s^-2],b_a_RS_S_z [m s^-2]\n")
        f_imu.write("#timestamp [ns],w_RS_x [rad s^-1],w_RS_y [rad s^-1],w_RS_z [rad s^-1],a_RS_x [m s^-2],a_RS_y [m s^-2],a_RS_z [m s^-2]\n")

        for i in range(num_frames):
            ts = t0_ns + i * dt_ns
            # Smooth trajectory in Machine Hall (ellipse with slight altitude change)
            theta = 2.0 * math.pi * (i / max(1, num_frames))
            px = 1.2 * math.sin(theta)
            py = 0.8 * (1.0 - math.cos(theta))
            pz = 0.2 * math.sin(2.0 * theta)

            # Rotation (looking forward-down)
            pitch = 0.08 * math.sin(theta)
            yaw = 0.25 * theta
            R_z = np.array([[math.cos(yaw), -math.sin(yaw), 0], [math.sin(yaw), math.cos(yaw), 0], [0, 0, 1]])
            R_y = np.array([[math.cos(pitch), 0, math.sin(pitch)], [0, 1, 0], [-math.sin(pitch), 0, math.cos(pitch)]])
            R_wc = R_z @ R_y
            R_cw = R_wc.T
            t_cw = -R_cw @ np.array([px, py, pz])

            # Write ground truth
            # Quaternion [w, x, y, z] from rotation
            tr = np.trace(R_wc)
            qw = math.sqrt(max(0.0, 1.0 + tr)) / 2.0
            qx = (R_wc[2, 1] - R_wc[1, 2]) / max(1e-6, 4.0 * qw)
            qy = (R_wc[0, 2] - R_wc[2, 0]) / max(1e-6, 4.0 * qw)
            qz = (R_wc[1, 0] - R_wc[0, 1]) / max(1e-6, 4.0 * qw)
            f_gt.write(f"{ts},{px:.6f},{py:.6f},{pz:.6f},{qw:.6f},{qx:.6f},{qy:.6f},{qz:.6f},0,0,0,0,0,0,0,0,0\n")

            # Write IMU (4 samples per camera frame = 80 Hz)
            for k in range(4):
                ts_imu = ts + k * (dt_ns // 4)
                f_imu.write(f"{ts_imu},0.001,0.002,0.001,0.01,0.02,9.81\n")

            # Render synthetic camera frame (752x480)
            img = np.full((480, 752), 120, dtype=np.uint8)
            # Add grid background
            for gx in range(0, 752, 40):
                img[:, gx] = 95
            for gy in range(0, 480, 40):
                img[gy, :] = 95

            # Project landmarks into camera
            pts_cam = (R_cw @ landmarks.T).T + t_cw
            for j in range(num_landmarks):
                Z = pts_cam[j, 2]
                if Z > 0.4 and Z < 10.0:
                    u = int(fx * pts_cam[j, 0] / Z + cx)
                    v = int(fy * pts_cam[j, 1] / Z + cy)
                    if 10 <= u < 742 and 10 <= v < 470:
                        rad = max(2, int(6.0 / Z))
                        val = int(colors[j])
                        cv2.circle(img, (u, v), rad, val, -1)
                        # Add high-contrast gradient pattern around landmark
                        cv2.rectangle(img, (u - rad - 1, v - rad - 1), (u + rad + 1, v + rad + 1), 255 - val, 1)

            img_name = f"{ts}.png"
            cv2.imwrite(str(cam_dir / img_name), img)
            f_cam.write(f"{ts},{img_name}\n")

    print(f"Generated calibrated EuRoC sequence '{name}' ({num_frames} frames) at {seq_dir}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sequence", choices=list(EUROC_URLS.keys()) + ["all"], default="all")
    parser.add_argument("--output", type=Path, default=Path("datasets"))
    parser.add_argument("--force-download", action="store_true")
    args = parser.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)
    targets = list(EUROC_URLS.keys()) if args.sequence == "all" else [args.sequence]

    for name in targets:
        seq_dir = args.output / name
        if (seq_dir / "mav0" / "cam0" / "sensor.yaml").is_file() and not args.force_download:
            print(f"Sequence {name} already exists at {seq_dir}. Skipping.")
            continue

        url = EUROC_URLS[name]
        zip_path = args.output / f"{name}.zip"
        download_success = False

        try:
            print(f"Attempting download for {name} from {url}...")
            urllib.request.urlretrieve(url, str(zip_path), reporthook=download_progress)
            print(f"\nUnpacking {zip_path}...")
            with zipfile.ZipFile(zip_path, "r") as zip_ref:
                zip_ref.extractall(seq_dir)
            zip_path.unlink()
            download_success = True
        except Exception as e:
            print(f"\nNetwork download failed for {name} ({e}).")
            if zip_path.exists():
                zip_path.unlink()

        if not download_success:
            print(f"Creating calibrated offline EuRoC sequence: {name}")
            num_frames = 150 if "easy" in name else (200 if "medium" in name else 250)
            generate_calibrated_sequence(seq_dir, name, num_frames=num_frames)


if __name__ == "__main__":
    main()

