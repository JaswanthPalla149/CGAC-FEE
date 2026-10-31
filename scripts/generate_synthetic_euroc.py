#!/usr/bin/env python3

import argparse
import math
import random
import struct
from pathlib import Path


def write_pgm(path: Path, width: int, height: int, pixels: bytearray) -> None:
    with path.open("wb") as output:
        output.write(f"P5\n{width} {height}\n255\n".encode("ascii"))
        output.write(pixels)


def generate_frame(width: int, height: int, index: int, seed: int) -> bytearray:
    rng = random.Random(seed + index)
    pixels = bytearray([18] * (width * height))
    displacement = int(2.5 * math.sin(index * 0.08))
    vertical = int(1.5 * math.cos(index * 0.06))

    for y in range(8, height, 24):
        for x in range(8, width, 24):
            color = 230 if ((x // 24 + y // 24) % 2 == 0) else 75
            x0 = max(0, min(width, x + displacement))
            y0 = max(0, min(height, y + vertical))
            size = 7 + rng.randint(0, 4)
            for py in range(y0, min(height, y0 + size)):
                for px in range(x0, min(width, x0 + size)):
                    pixels[py * width + px] = color

    center_x = width // 2 + int(35 * math.sin(index * 0.035))
    center_y = height // 2 + int(20 * math.cos(index * 0.05))
    radius = 38
    for y in range(max(0, center_y - radius), min(height, center_y + radius)):
        for x in range(max(0, center_x - radius), min(width, center_x + radius)):
            dx = x - center_x
            dy = y - center_y
            if dx * dx + dy * dy < radius * radius:
                pixels[y * width + x] = 190

    return pixels


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate a synthetic EuRoC-style image sequence")
    parser.add_argument("--output", default="synthetic/euroc")
    parser.add_argument("--frames", type=int, default=300)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--seed", type=int, default=17)
    args = parser.parse_args()

    sequence = Path(args.output)
    image_dir = sequence / "mav0" / "cam0" / "data"
    truth_dir = sequence / "mav0" / "state_groundtruth_estimate0"
    image_dir.mkdir(parents=True, exist_ok=True)
    truth_dir.mkdir(parents=True, exist_ok=True)

    camera_dir = sequence / "mav0" / "cam0"
    (camera_dir / "sensor.yaml").write_text(
        "camera_model: pinhole\n"
        f"intrinsics: [{args.width * 0.92:.1f}, {args.width * 0.92:.1f}, "
        f"{args.width * 0.5:.1f}, {args.height * 0.5:.1f}]\n",
        encoding="ascii",
    )

    frames_csv = camera_dir / "data.csv"
    truth_csv = truth_dir / "data.csv"
    with frames_csv.open("w", encoding="ascii") as frames, truth_csv.open(
        "w", encoding="ascii"
    ) as truth:
        frames.write("#timestamp [ns],filename\n")
        truth.write("#timestamp [ns],p_RS_R_x [m],p_RS_R_y [m],p_RS_R_z [m],q_W_RS_x,q_W_RS_y,q_W_RS_z,q_W_RS_w\n")
        for index in range(args.frames):
            timestamp = index * 20_000_000
            filename = f"{index:06d}.pgm"
            write_pgm(
                image_dir / filename,
                args.width,
                args.height,
                generate_frame(args.width, args.height, index, args.seed),
            )
            frames.write(f"{timestamp},{filename}\n")
            x = 2.0 * math.sin(index * 0.035)
            y = 2.0 * math.cos(index * 0.035) - 2.0
            z = 0.5 + 0.05 * math.sin(index * 0.08)
            truth.write(f"{timestamp},{x:.6f},{y:.6f},{z:.6f},0,0,0,1\n")

    print(f"Generated {args.frames} frames at {sequence}")


if __name__ == "__main__":
    main()
