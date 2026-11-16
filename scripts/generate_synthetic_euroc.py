#!/usr/bin/env python3

import argparse
import math
import secrets
from pathlib import Path
from typing import Dict, Optional, Tuple

import cv2
import numpy as np


def create_trajectory_parameters(
    seed: Optional[int] = None,
    closed_loop: bool = True,
    total_frames: int = 300,
) -> Tuple[int, Dict[str, float]]:
    actual_seed = secrets.randbits(64) if seed is None else seed
    if actual_seed < 0:
        raise ValueError("seed must be non-negative")

    rng = np.random.default_rng(actual_seed)
    if closed_loop:
        omega = 2.0 * math.pi / max(1, total_frames)
        parameters = {
            "closed_loop": 1.0,
            "total_frames": float(total_frames),
            "x_amplitude": float(rng.uniform(1.8, 2.4)),
            "x_frequency": omega,
            "x_phase": 0.0,
            "y_amplitude": float(rng.uniform(0.6, 1.0)),
            "y_frequency": omega,
            "y_phase": 0.0,
            "y_offset": float(rng.uniform(-2.4, -1.8)),
            "z_base": float(rng.uniform(0.5, 0.7)),
            "z_amplitude": float(rng.uniform(0.04, 0.08)),
            "z_frequency": 2.0 * omega,
            "z_phase": 0.0,
        }
    else:
        parameters = {
            "closed_loop": 0.0,
            "total_frames": float(total_frames),
            "x_amplitude": float(rng.uniform(1.2, 3.0)),
            "x_frequency": float(rng.uniform(0.02, 0.06)),
            "x_phase": float(rng.uniform(0.0, 2.0 * math.pi)),
            "y_amplitude": float(rng.uniform(1.2, 3.0)),
            "y_frequency": float(rng.uniform(0.02, 0.06)),
            "y_phase": float(rng.uniform(0.0, 2.0 * math.pi)),
            "y_offset": float(rng.uniform(-3.0, -1.5)),
            "z_base": float(rng.uniform(0.4, 0.8)),
            "z_amplitude": float(rng.uniform(0.02, 0.1)),
            "z_frequency": float(rng.uniform(0.05, 0.15)),
            "z_phase": float(rng.uniform(0.0, 2.0 * math.pi)),
        }
    return actual_seed, parameters


def trajectory_position(
    index: int, parameters: Dict[str, float]
) -> Tuple[float, float, float]:
    x = parameters["x_amplitude"] * math.sin(
        index * parameters["x_frequency"] + parameters["x_phase"]
    )
    y = (
        parameters["y_amplitude"]
        * math.cos(index * parameters["y_frequency"] + parameters["y_phase"])
        + parameters["y_offset"]
    )
    z = parameters["z_base"] + parameters["z_amplitude"] * math.sin(
        index * parameters["z_frequency"] + parameters["z_phase"]
    )
    return x, y, z


def create_retail_street_texture(tex_w: int = 1600, tex_h: int = 1000) -> np.ndarray:
    """Create a realistic Retail Street storefront texture (Burger King + Nayuki)."""
    tex = np.full((tex_h, tex_w), 210, dtype=np.uint8)

    # 1. Brick / panel pattern on upper wall
    for y in range(0, 420, 28):
        cv2.line(tex, (0, y), (tex_w, y), 160, 1)
        shift = 35 if (y // 28) % 2 == 1 else 0
        for x in range(shift, tex_w, 70):
            cv2.line(tex, (x, y), (x, y + 28), 160, 1)

    # 2. Storefront fascia banner
    cv2.rectangle(tex, (40, 120), (1560, 360), 235, -1)
    cv2.rectangle(tex, (40, 120), (1560, 360), 50, 4)

    # Burger King section (Left)
    cv2.rectangle(tex, (60, 140), (880, 340), 220, -1)
    cv2.rectangle(tex, (60, 140), (880, 340), 70, 3)

    # Burger King Logo (Circles)
    cv2.circle(tex, (180, 240), 65, 40, 4)
    cv2.ellipse(tex, (180, 215), (55, 25), 0, 180, 360, 60, -1)
    cv2.ellipse(tex, (180, 265), (55, 25), 0, 0, 180, 60, -1)
    cv2.rectangle(tex, (130, 235), (230, 245), 30, -1)
    cv2.putText(tex, "BURGER", (270, 220), cv2.FONT_HERSHEY_DUPLEX, 2.2, 30, 5)
    cv2.putText(tex, "KING", (300, 310), cv2.FONT_HERSHEY_DUPLEX, 2.4, 30, 5)

    # Chinese storefront characters
    cv2.putText(tex, "BK STREET", (620, 240), cv2.FONT_HERSHEY_SIMPLEX, 1.4, 60, 3)
    cv2.putText(tex, "EST. 1954", (630, 300), cv2.FONT_HERSHEY_SIMPLEX, 0.9, 80, 2)

    # Nayuki Tea section (Right)
    cv2.rectangle(tex, (920, 140), (1540, 340), 245, -1)
    cv2.rectangle(tex, (920, 140), (1540, 340), 70, 3)
    cv2.circle(tex, (1080, 240), 65, 50, 4)
    cv2.circle(tex, (1080, 240), 45, 90, 2)
    cv2.putText(tex, "NAYUKI", (1170, 245), cv2.FONT_HERSHEY_DUPLEX, 1.8, 40, 4)
    cv2.putText(tex, "FRESH TEA", (1180, 305), cv2.FONT_HERSHEY_SIMPLEX, 0.9, 70, 2)

    # 3. Store windows & glass entrance doors (Middle)
    cv2.rectangle(tex, (80, 420), (1520, 860), 60, -1)  # Dark glass interior
    for door_x in range(120, 1500, 280):
        # Window / Door frame
        cv2.rectangle(tex, (door_x, 430), (door_x + 240, 850), 180, 4)
        cv2.line(tex, (door_x + 120, 430), (door_x + 120, 850), 180, 3)
        # Door handle
        cv2.rectangle(tex, (door_x + 110, 600), (door_x + 130, 680), 220, -1)
        # Table / posters visible through glass
        cv2.rectangle(tex, (door_x + 30, 700), (door_x + 90, 780), 120, -1)
        cv2.rectangle(tex, (door_x + 150, 700), (door_x + 210, 780), 120, -1)

    # 4. Pillars
    cv2.rectangle(tex, (0, 0), (60, 860), 130, -1)
    cv2.rectangle(tex, (1540, 0), (1600, 860), 130, -1)
    for y in range(0, 860, 40):
        cv2.line(tex, (0, y), (60, y), 90, 2)
        cv2.line(tex, (1540, y), (1600, y), 90, 2)

    # 5. Sidewalk / Pavement tiles (Bottom)
    cv2.rectangle(tex, (0, 860), (tex_w, tex_h), 170, -1)
    cv2.line(tex, (0, 860), (tex_w, 860), 40, 4)  # Curb
    for x in range(0, tex_w, 80):
        cv2.line(tex, (x, 860), (x, tex_h), 120, 2)
    for y in range(860, tex_h, 40):
        cv2.line(tex, (0, y), (tex_w, y), 120, 2)

    return tex


def render_projected_frame(
    texture: np.ndarray,
    width: int,
    height: int,
    uav_x: float,
    uav_y: float,
    uav_z: float,
) -> np.ndarray:
    """Project the 3D storefront texture onto the camera sensor using pinhole geometry."""
    tex_h, tex_w = texture.shape

    # Virtual camera intrinsics
    fx = width * 0.92
    fy = width * 0.92
    cx = width * 0.5
    cy = height * 0.5

    # 3D coordinates of the 4 storefront corners in world frame
    # Storefront is a plane at Y_world = 3.8 meters
    # Extent: X from -4.5 to +4.5 m, Z from 0.0 to 2.8 m
    store_y = 3.8
    corners_world = [
        np.array([-4.5, store_y, 2.8]),  # Top-left
        np.array([4.5, store_y, 2.8]),   # Top-right
        np.array([4.5, store_y, 0.0]),   # Bottom-right
        np.array([-4.5, store_y, 0.0]),  # Bottom-left
    ]

    # Camera position in world frame
    cam_pos = np.array([uav_x, uav_y, uav_z])

    # Project corners into camera image plane
    proj_pts = []
    for pt_w in corners_world:
        # Camera looks in +Y direction (X right, Z up)
        dx = pt_w[0] - cam_pos[0]
        dy = pt_w[1] - cam_pos[1]  # Depth along optical axis
        dz = pt_w[2] - cam_pos[2]

        depth = max(0.5, dy)
        u = fx * (dx / depth) + cx
        v = -fy * (dz / depth) + cy
        proj_pts.append([u, v])

    src_pts = np.float32([[0, 0], [tex_w, 0], [tex_w, tex_h], [0, tex_h]])
    dst_pts = np.float32(proj_pts)

    # Compute perspective transform matrix
    H = cv2.getPerspectiveTransform(src_pts, dst_pts)
    frame = cv2.warpPerspective(
        texture,
        H,
        (width, height),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=40,
    )

    return frame


def write_pgm(path: Path, image: np.ndarray) -> None:
    h, w = image.shape
    with path.open("wb") as output:
        output.write(f"P5\n{w} {h}\n255\n".encode("ascii"))
        output.write(image.tobytes())


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate a synthetic Retail Street EuRoC-style sequence")
    parser.add_argument("--output", default="synthetic/ecolivo-wsl")
    parser.add_argument("--frames", type=int, default=300)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="random seed for reproducible trajectory generation (default: random)",
    )
    parser.add_argument(
        "--closed-loop",
        action="store_true",
        default=True,
        help="generate closed-loop trajectory that revisits initial location (default: True)",
    )
    parser.add_argument(
        "--open-loop",
        action="store_false",
        dest="closed_loop",
        help="generate open-loop randomized trajectory without revisit",
    )
    args = parser.parse_args()

    if args.seed is not None and args.seed < 0:
        parser.error("--seed must be non-negative")
    trajectory_seed, trajectory_parameters = create_trajectory_parameters(
        args.seed, closed_loop=args.closed_loop, total_frames=args.frames
    )
    print(
        f"Trajectory seed: {trajectory_seed} "
        f"(reproduce with --seed {trajectory_seed})"
    )

    sequence = Path(args.output)
    image_dir = sequence / "mav0" / "cam0" / "data"
    truth_dir = sequence / "mav0" / "state_groundtruth_estimate0"
    lidar_dir = sequence / "mav0" / "lidar0"
    image_dir.mkdir(parents=True, exist_ok=True)
    truth_dir.mkdir(parents=True, exist_ok=True)
    lidar_dir.mkdir(parents=True, exist_ok=True)

    camera_dir = sequence / "mav0" / "cam0"
    (camera_dir / "sensor.yaml").write_text(
        "camera_model: pinhole\n"
        f"intrinsics: [{args.width * 0.92:.1f}, {args.width * 0.92:.1f}, "
        f"{args.width * 0.5:.1f}, {args.height * 0.5:.1f}]\n",
        encoding="ascii",
    )

    # Generate high-resolution storefront texture
    texture = create_retail_street_texture()

    frames_csv = camera_dir / "data.csv"
    truth_csv = truth_dir / "data.csv"
    lidar_csv = lidar_dir / "data.csv"

    with frames_csv.open("w", encoding="ascii") as frames_f, \
         truth_csv.open("w", encoding="ascii") as truth_f, \
         lidar_csv.open("w", encoding="ascii") as lidar_f:

        frames_f.write("#timestamp [ns],filename\n")
        truth_f.write("#timestamp [ns],p_RS_R_x [m],p_RS_R_y [m],p_RS_R_z [m],q_W_RS_x,q_W_RS_y,q_W_RS_z,q_W_RS_w\n")
        lidar_f.write("#timestamp [ns],filename\n")

        for index in range(args.frames):
            timestamp = index * 20_000_000

            x, y, z = trajectory_position(index, trajectory_parameters)

            truth_f.write(f"{timestamp},{x:.6f},{y:.6f},{z:.6f},0,0,0,1\n")

            # Project texture to camera
            frame = render_projected_frame(texture, args.width, args.height, x, y, z)
            filename = f"{index:06d}.pgm"
            write_pgm(image_dir / filename, frame)
            frames_f.write(f"{timestamp},{filename}\n")
            lidar_f.write(f"{timestamp},{filename}\n")

    bag_path = sequence / "EcoLIVO_Retail_Street.bag"
    with bag_path.open("w", encoding="ascii") as bag:
        bag.write("# EcoLIVO synthetic rosbag manifest\n")
        bag.write("format=ecolivo-bag/1\n")
        bag.write(f"seed={trajectory_seed}\n")
        bag.write(f"sequence={sequence.as_posix()}\n")
        bag.write("camera_topic=/camera/image_raw\n")
        bag.write("lidar_topic=/lidar/points\n")
        bag.write("imu_topic=/imu/data\n")
        bag.write("frame_count=" + str(args.frames) + "\n")
        bag.write("[messages]\n")
        bag.write("timestamp_ns,frame_path,topic\n")
        for index in range(args.frames):
            timestamp = index * 20_000_000
            bag.write(f"{timestamp},mav0/cam0/data/{index:06d}.pgm,/camera/image_raw\n")

    print(f"Generated {args.frames} photometrically & geometrically consistent frames at {sequence}")


if __name__ == "__main__":
    main()
