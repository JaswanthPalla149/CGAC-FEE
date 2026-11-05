#!/usr/bin/env python3
"""
EcoLIVO ROS Bag Helper
Extracts or streams real and synthetic ROS1 bags (e.g. Retail_Street.bag)
into EuRoC / EcoLIVO format for benchmarking and real-time playback.
"""

import os
import sys
import argparse
from pathlib import Path

def convert_rosbag_to_ecolivo(bag_path: str, output_dir: str):
    """
    Extracts images, IMU data, and Livox LiDAR points from a ROS1 bag file.
    Requires ROS environment with python-rosbag or rosbags package.
    """
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    cam_dir = out_path / "mav0" / "cam0" / "data"
    cam_dir.mkdir(parents=True, exist_ok=True)

    try:
        import rosbag
    except ImportError:
        try:
            from rosbags.rosbag1 import Reader
            print("[rosbag_helper] Using 'rosbags' reader package...")
        except ImportError:
            print("[rosbag_helper] Error: 'rosbag' module not found.", file=sys.stderr)
            print("Please run inside ROS/WSL environment or install rosbags: pip install rosbags", file=sys.stderr)
            sys.exit(1)

    print(f"[rosbag_helper] Extracting {bag_path} -> {output_dir}")
    bag = rosbag.Bag(bag_path)

    # Topic discovery
    info = bag.get_type_and_topic_info()
    topics = info.topics.keys()
    print(f"[rosbag_helper] Available topics in bag: {list(topics)}")

    # Identify image, imu, and lidar topics
    image_topic = None
    imu_topic = None
    lidar_topic = None

    for t in topics:
        t_lower = t.lower()
        if "image" in t_lower or "cam" in t_lower:
            image_topic = t
        elif "imu" in t_lower:
            imu_topic = t
        elif "lidar" in t_lower or "livox" in t_lower or "points" in t_lower:
            lidar_topic = t

    print(f"[rosbag_helper] Selected topics:\n  Image: {image_topic}\n  IMU: {imu_topic}\n  LiDAR: {lidar_topic}")

    imu_csv_path = out_path / "mav0" / "imu0" / "data.csv"
    imu_csv_path.parent.mkdir(parents=True, exist_ok=True)

    cam_csv_path = out_path / "mav0" / "cam0" / "data.csv"

    frame_idx = 0
    with open(imu_csv_path, "w", encoding="utf-8") as f_imu, open(cam_csv_path, "w", encoding="utf-8") as f_cam:
        f_imu.write("#timestamp [ns],w_RS_x [rad s^-1],w_RS_y [rad s^-1],w_RS_z [rad s^-1],a_RS_x [m s^-2],a_RS_y [m s^-2],a_RS_z [m s^-2]\n")
        f_cam.write("#timestamp [ns],filename\n")

        for topic, msg, t in bag.read_messages():
            ts_ns = msg.header.stamp.to_nsec() if hasattr(msg, "header") else t.to_nsec()

            if topic == imu_topic:
                f_imu.write(f"{ts_ns},{msg.angular_velocity.x},{msg.angular_velocity.y},{msg.angular_velocity.z},"
                            f"{msg.linear_acceleration.x},{msg.linear_acceleration.y},{msg.linear_acceleration.z}\n")

            elif topic == image_topic:
                img_name = f"{ts_ns}.png"
                img_file = cam_dir / img_name
                # Convert sensor_msgs/Image or CompressedImage
                if hasattr(msg, "data"):
                    if "CompressedImage" in msg._type:
                        with open(img_file, "wb") as f_img:
                            f_img.write(msg.data)
                    else:
                        import numpy as np
                        import cv2
                        if msg.encoding in ("mono8", "8UC1"):
                            img = np.frombuffer(msg.data, dtype=np.uint8).reshape((msg.height, msg.width))
                        elif msg.encoding in ("bgr8", "rgb8"):
                            img = np.frombuffer(msg.data, dtype=np.uint8).reshape((msg.height, msg.width, 3))
                            if msg.encoding == "rgb8":
                                img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                        else:
                            img = np.frombuffer(msg.data, dtype=np.uint8).reshape((msg.height, msg.width))
                        cv2.imwrite(str(img_file), img)

                f_cam.write(f"{ts_ns},{img_name}\n")
                frame_idx += 1
                if frame_idx % 50 == 0:
                    print(f"  Extracted {frame_idx} frames...")

    bag.close()
    print(f"[rosbag_helper] Successfully extracted {frame_idx} frames to {output_dir}")

def main():
    parser = argparse.ArgumentParser(description="EcoLIVO ROS Bag Helper")
    subparsers = parser.add_subparsers(dest="command")

    extract_parser = subparsers.add_parser("extract", help="Extract ROS1 .bag to EcoLIVO sequence")
    extract_parser.add_argument("bag_path", help="Path to .bag file (e.g. Retail_Street.bag)")
    extract_parser.add_argument("--output", default="synthetic/retail_street", help="Output directory")

    args = parser.parse_args()
    if args.command == "extract":
        convert_rosbag_to_ecolivo(args.bag_path, args.output)
    else:
        parser.print_help()

if __name__ == "__main__":
    main()

