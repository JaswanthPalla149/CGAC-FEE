#!/usr/bin/env python3
"""
Synthetic Flight Sequence Generator for EuRoC-compatible SLAM evaluation.
Generates camera frames, ground truth poses, and calibration for CGAC-FEE.
"""

import argparse
import os
import cv2
import numpy as np

def generate_sequence(output_dir="simulated_sequence", num_frames=120, fps=20):
    cam_dir = os.path.join(output_dir, "mav0", "cam0")
    img_dir = os.path.join(cam_dir, "data")
    gt_dir = os.path.join(output_dir, "mav0", "state_groundtruth_estimate0")
    os.makedirs(img_dir, exist_ok=True)
    os.makedirs(gt_dir, exist_ok=True)

    # Standard EuRoC VI-Sensor Camera intrinsics
    fx, fy = 458.65, 457.29
    cx, cy = 367.21, 248.13
    width, height = 752, 480

    sensor_yaml = f"""sensor_type: camera
comment: Simulated UAV Camera
intrinsics: [{fx}, {fy}, {cx}, {cy}]
distortion_model: radial-tangential
distortion_coefficients: [0.0, 0.0, 0.0, 0.0]
"""
    with open(os.path.join(cam_dir, "sensor.yaml"), "w") as f:
        f.write(sensor_yaml)

    # 3D Landmark Points in corridor
    np.random.seed(42)
    num_pts = 600
    pts_x = np.random.uniform(-3.5, 3.5, num_pts)
    pts_y = np.random.uniform(-2.5, 2.5, num_pts)
    pts_z = np.random.uniform(1.5, 35.0, num_pts)
    pts_size = np.random.randint(3, 7, num_pts)
    pts_color = np.random.randint(180, 255, num_pts)

    t_start = 1403636579758555392
    dt_ns = int(1e9 / fps)

    frames_csv = ["#timestamp [ns],filename\n"]
    gt_csv = ["#timestamp,p_RS_R_x [m],p_RS_R_y [m],p_RS_R_z [m],q_RS_w,q_RS_x,q_RS_y,q_RS_z\n"]

    print(f"Generating {num_frames} frames in '{output_dir}'...")
    for i in range(num_frames):
        t_ns = t_start + i * dt_ns
        fname = f"{t_ns}.png"

        # Smooth flight path: forward along Z with lateral / vertical undulation
        cam_x = 0.35 * np.sin(i * 0.08)
        cam_y = 0.18 * np.cos(i * 0.06)
        cam_z = 0.08 * i

        frames_csv.append(f"{t_ns},{fname}\n")
        gt_csv.append(f"{t_ns},{cam_x:.6f},{cam_y:.6f},{cam_z:.6f},1.0,0.0,0.0,0.0\n")

        # Base image with texture noise
        img = np.full((height, width), 40, dtype=np.uint8)
        noise = np.random.normal(0, 4, (height, width)).astype(np.int16)
        img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)

        # Project 3D points to 2D image plane
        for p in range(num_pts):
            X = pts_x[p] - cam_x
            Y = pts_y[p] - cam_y
            Z = pts_z[p] - cam_z
            if Z > 0.4:
                u = int(fx * (X / Z) + cx)
                v = int(fy * (Y / Z) + cy)
                if 15 <= u < width - 15 and 15 <= v < height - 15:
                    rad = pts_size[p]
                    col = int(pts_color[p])
                    cv2.circle(img, (u, v), rad, col, -1)
                    cv2.circle(img, (u, v), rad + 1, 10, 1)
                    cv2.circle(img, (u + rad // 2, v), 1, 255, -1)
                    cv2.circle(img, (u - rad // 2, v), 1, 255, -1)

        cv2.imwrite(os.path.join(img_dir, fname), img)

    with open(os.path.join(cam_dir, "data.csv"), "w") as f:
        f.writelines(frames_csv)

    with open(os.path.join(gt_dir, "data.csv"), "w") as f:
        f.writelines(gt_csv)

    print(f"Successfully generated synthetic sequence at: {os.path.abspath(output_dir)}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate synthetic EuRoC flight dataset")
    parser.add_argument("--output", default="simulated_sequence", help="Output sequence directory")
    parser.add_argument("--frames", type=int, default=120, help="Number of frames to generate")
    parser.add_argument("--fps", type=int, default=20, help="Frame rate")
    args = parser.parse_args()
    generate_sequence(args.output, args.frames, args.fps)
