# Linux/WSL setup and BTP presentation guide

This note records how to build and visualize the current project on Linux, what its present
benchmark results do and do not establish, and a defensible way to discuss FPGA acceleration.
It is a companion to the project README; it does not change the implementation.

## Linux and WSL prerequisites

The native benchmark requires CMake 3.18+, a C/C++17 compiler, and OpenCV 4 development files.
Python, OpenCV-Python, and NumPy are only needed to generate the optional synthetic sequence.
ROS 2 is not required to build or run the benchmark.

For a new ROS setup, prefer a supported ROS 2 release on its supported Ubuntu version. For example,
ROS 2 Jazzy binaries target Ubuntu 24.04 (Noble). This project was previously exercised in Ubuntu
20.04 WSL, where ROS 2 Foxy can be installed, but Foxy is end-of-life; it is not the recommended
starting point for a new ROS installation. Check the
[official ROS installation matrix](https://docs.ros.org/en/jazzy/Installation.html) before choosing
a distribution/Ubuntu pairing.

### Build this project in Ubuntu/WSL

From the repository root in an Ubuntu terminal:

```bash
sudo apt update
sudo apt install -y build-essential cmake libopencv-dev pkg-config
cmake --version
pkg-config --modversion opencv4

cmake -S . -B build-wsl -DCMAKE_BUILD_TYPE=Release
cmake --build build-wsl --parallel
ctest --test-dir build-wsl --output-on-failure
```

Generate and run a small synthetic EuRoC-format smoke sequence:

```bash
sudo apt install -y python3 python3-numpy python3-opencv
python3 scripts/generate_synthetic_euroc.py \
  --output synthetic/btp-smoke --frames 60 --width 640 --height 480
./build-wsl/cpp/adaptive_slam_benchmark \
  --sequence synthetic/btp-smoke --output results/btp-smoke --threads 1
```

The generated scene is for checking that the loader, executable, and CSV output work. It is not a
replacement for EuRoC ground-truth evaluation.

### ROS 2 and WSLg visualization

Install the ROS desktop package only if ROS tools/RViz are needed. Follow the [official Jazzy
Ubuntu instructions](https://docs.ros.org/en/jazzy/Installation/Ubuntu-Install-Debs.html) on a
supported Ubuntu version. After installing, source the environment in each ROS shell:

```bash
source /opt/ros/jazzy/setup.bash
ros2 --help
rviz2
```

For WSL GUI applications, enable WSLg and update WSL from Windows (`wsl --update`). Microsoft
documents the requirements and troubleshooting in its
[WSL GUI apps guide](https://learn.microsoft.com/en-us/windows/wsl/tutorials/gui-apps).

Validate ROS middleware independently of this SLAM project:

```bash
# Terminal 1
source /opt/ros/jazzy/setup.bash
ros2 run demo_nodes_cpp talker

# Terminal 2
source /opt/ros/jazzy/setup.bash
ros2 run demo_nodes_py listener
```

**Important:** installing ROS and starting RViz does not connect RViz to this benchmark. The current
program does not publish ROS messages or implement a ROS node. Its `scripts/roslaunch` and
`scripts/rosbag` files are project-specific shell wrappers, not the ROS commands. The hybrid demo's
LiDAR and IMU are simulated, and the visible panel/snapshot is an OpenCV visualization. To show this
map in genuine RViz, a ROS 2 adapter must publish at least `nav_msgs/Odometry` (and the transform
tree) and `sensor_msgs/PointCloud2`, with a valid frame ID and timestamps.

### Power measurement considerations

The benchmark's UAV-demo energy number is a model based on assumed MAC energy and a fixed power
term; do not call it measured power. The summary CSV marks energy `not_measured`. Linux RAPL can
measure CPU-package energy if the host exposes readable counters, but WSL often does not. RAPL is
whole-package energy, not process-only energy. For whole-system energy, use a calibrated external
power meter and state whether it measures the laptop input, the accelerator board, or another scope.

The WSL host used for the current smoke run was an AMD Ryzen 9 8945HS, Ubuntu 20.04 under WSL2. No
RAPL energy counter or `perf` executable was available there. Battery charge percentage alone does
not reveal workload power. Do not infer watts or joules from frame latency, CPU time, or the
displayed model-only energy.

## Results generated in this workspace

The CSV at [`../results/btp-smoke/summary.csv`](../results/btp-smoke/summary.csv) is a 60-frame
synthetic, no-degradation smoke run. It is useful for proving the executable ran and emitted
artifacts, not for claiming general SLAM quality.

| Configuration | Similarity-aligned ATE (m) | Failure frames / 59 | Mean latency (ms) | Recovery fraction | Energy |
|---|---:|---:|---:|---:|---|
| Fixed low | 0.2258 | 1 (1.69%) | 10.94 | 0% | Not measured |
| Fixed high | 0.2386 | 2 (3.39%) | 21.87 | 100% | Not measured |
| Adaptive | 0.2258 | 1 (1.69%) | 11.12 | 0% | Not measured |

On this input, adaptive never left low-cost mode: it matched fixed-low rather than demonstrating a
benefit over it. Fixed-high had about twice the mean latency but slightly worse ATE and one more
failure on this synthetic sample. The per-run similarity alignment and short generated trajectory
limit what can be inferred. There is no measured energy comparison in these results.

The hybrid demo snapshot is
[`../results/btp-hybrid-demo/fast_livo2_simulation.jpg`](../results/btp-hybrid-demo/fast_livo2_simulation.jpg).
It visualizes synthetic geometry; it is not an RViz screenshot or a result from real LiDAR/IMU.

## Fair comparison with established systems

Describe related systems by their methods and sensor assumptions. Do not make a cross-paper
leaderboard from numbers measured on different hardware, datasets, and protocols.

| System | What its published implementation contributes | How it differs from this prototype |
|---|---|---|
| [DROID-SLAM (2021)](https://arxiv.org/abs/2108.10869) | Learned dense visual SLAM with recurrent updates and differentiable dense bundle adjustment; supports monocular, stereo, and RGB-D. The official implementation uses CUDA/PyTorch and documents a GPU memory requirement. | This project is sparse OpenCV feature tracking and geometric estimation, not a neural dense tracker. Its potential low-cost advantage must be measured, not assumed. |
| [ORB-SLAM3 (2021)](https://doi.org/10.1109/TRO.2021.3075644) | Mature ORB feature-based visual, visual-inertial, and multi-map SLAM; supports monocular, stereo, RGB-D, pinhole, and fisheye cameras. | This project lacks ORB-SLAM3's full map-management, relocalization, loop-closing and optimization capabilities. Compare only on shared input and evaluation protocol, and label the scope difference. |
| [FAST-LIO2 (2021)](https://arxiv.org/abs/2107.06829) | Tightly coupled LiDAR-IMU odometry with direct raw-point registration and an incremental ikd-Tree. | This project's demo synthesizes scans and uses simplified point matching. It is not a FAST-LIO2-equivalent estimator. |
| [FAST-LIVO2 (2024)](https://arxiv.org/abs/2408.14035) | Tightly integrated direct LiDAR-inertial-visual odometry and mapping, designed for real sensor streams and UAV mapping use cases. | This project's UAV path does not consume synchronized real sensor streams or implement equivalent tightly coupled estimation. Refer to it as a long-term comparison target, not a baseline result. |
| [FAST-LIVO2 on Resource-Constrained Platforms (2025 preprint)](https://arxiv.org/abs/2501.13876) | A particularly relevant adaptive-computation comparison: degeneration-aware visual frame selection in an ESIKF plus memory-aware map design; evaluates x86 and ARM platforms. | This is closer to the project's motivation than a generic SLAM leaderboard. Cite it as related work and distinguish its real sensor/system evaluation from this project's current synthetic smoke test. |

The repository's own measured comparison is currently only fixed-low versus fixed-high versus
adaptive on one synthetic sequence. To make an empirical claim against ORB-SLAM3, DROID-SLAM or
UAV 3D systems, run the actual systems on shared recorded datasets with documented calibration,
preprocessing, compute device, and evaluation scripts. Report per-sequence ATE/RPE, tracking loss,
latency, memory, and measured energy; list systems that could not be run as related work rather
than inventing results.

## FPGA acceleration: a plausible research direction, not a measured result

### What is already in this repository

- `cpp/include/adaptive_slam/hbst_matcher.hpp` contains software code for 256-bit binary descriptors
  and Hamming distance using XOR plus CPU popcount.
- `ORB_SLAM2_Matching/hbst_matcher.cpp` is a separate HLS-oriented C++ file. It describes an
  `ap_uint<256>` descriptor, an XOR/reduction Hamming kernel, pipeline/unroll pragmas, and AXI
  interfaces.
- The main CMake build does not compile the HLS file. There is no synthesis report, FPGA bitstream,
  timing closure, resource utilization, or board power result in this repository.
- The current UAV path extracts descriptors and builds an index but does not use that index for
  its optical-flow tracking association. Therefore, the separate HLS matcher is not currently an
  accelerated stage in the measured SLAM path.

These are good grounds to say **“FPGA-amenable kernel / HLS prototype source exists.”** They are not
grounds to say **“the SLAM system is FPGA-accelerated,” “hardware power is lower,” or “this is one
of the best power-aware SLAM systems.”**

### Candidate FPGA datapath for a future implementation

1. **Host/SoC to accelerator:** move grayscale image, keypoint coordinates, and descriptor/map
   buffers using AXI DMA or a shared-memory interface. Keep buffers resident where possible to
   avoid transfer overhead dominating small kernels.
2. **Binary descriptor generation:** parallel BRIEF-style intensity-pair tests around keypoints.
   This is a fixed, integer/bitwise workload and a more direct candidate than porting the whole
   OpenCV pipeline at once.
3. **Candidate matching:** for each query/candidate descriptor, compute 256-bit XOR then a
   population count, accumulate Hamming distance, and maintain best/second-best candidates.
   Pipeline over candidate descriptors and cache descriptor words in BRAM/URAM where capacity
   permits. The current tree's data-dependent branch traversal and memory access may limit
   throughput; compare it to a flat tiled/parallel matcher rather than assuming HBST is optimal.
4. **Confidence/control:** compute tracked-feature ratio, inlier ratio and error summaries, then
   return a confidence value or low-cost/recovery recommendation. Keep geometric estimation on the
   CPU initially, and move only after profiling shows it is a useful accelerator target.
5. **CPU mapping stage:** retain RANSAC/PnP, map/keyframe logic and trajectory output on the CPU for
   the first co-design milestone. A later FPGA version can target selected, profiled kernels.

The FPGA-versus-CPU/GPU decision must be empirical. FPGA can exploit fixed-width datapaths, bit
parallelism, and local memory, but it can lose on clock rate, memory bandwidth, transfer cost,
development effort, and power. Prior work demonstrates ORB-based SLAM acceleration on an SoC FPGA
([Vemulapati & Chen, 2022](https://arxiv.org/abs/2207.08405)) and selected LiDAR/point-cloud
acceleration ([LiDAR SLAM accelerator](https://arxiv.org/abs/2103.09523);
[FPGA point-cloud registration](https://arxiv.org/abs/2203.05763)). These prove feasibility of
specific kernels, not that this implementation is faster or more energy-efficient. A 2025
FPGA-vs-GPU feature-detector study reports that the winner depends on detector and platform: its
FAST/Harris GPU configurations were more efficient in the tested setup, while FPGA SuperPoint
showed advantages in some measures. Use it to motivate a measured backend comparison, not a blanket
“FPGA uses less power” claim
([Ye & Luján, 2025 preprint](https://arxiv.org/abs/2510.13546)).

### Evidence to collect before claiming acceleration or power awareness

- **Correctness:** descriptor distance and selected matches agree with a CPU reference over random
  and real descriptors; measure match recall and downstream pose/trajectory impact.
- **Hardware implementation:** synthesis and implementation reports for LUT/CLB, FF, DSP, BRAM,
  achieved clock, initiation interval, latency, throughput, and timing closure on a named board.
- **End-to-end cost:** include host-to-FPGA transfers, setup, recovery/low-cost mode switching, and
  CPU work. Report joules/frame, average watts, and energy per successfully tracked frame.
- **Fair baselines:** same sequence, image size, confidence/failure criteria and run repetitions
  across CPU-only, FPGA-enabled, and (where available) GPU runs. Measure with the same external
  meter or clearly scoped board power counters.
- **Trade-off:** plot energy against ATE/RPE and tracking failures. A kernel speedup alone is not a
  system-level energy win, and a design that loses tracking is not an efficient SLAM solution.

### Suggested presentation wording

> **Contribution:** A confidence-guided adaptive visual-SLAM prototype and a study of its
> accuracy–compute trade-off. The current implementation is CPU/OpenCV software. Binary Hamming
> matching is a plausible FPGA kernel and a separate HLS-oriented matcher source is present, but
> FPGA synthesis, end-to-end integration, and direct power validation remain future work.

> **Current evidence:** On a 60-frame synthetic smoke test, adaptive remained in low-cost mode and
> matched fixed-low behavior. This verifies the run path; it does not establish a general accuracy
> advantage or measured energy saving.

Avoid “best,” “hardware accelerated,” “zero energy,” or measured power claims unless the corresponding
comparative experiment and hardware evidence has actually been completed.

## References

- ROS 2 Jazzy installation and platform support: [ROS documentation](https://docs.ros.org/en/jazzy/Installation/Ubuntu-Install-Debs.html).
- WSL GUI requirements: [Microsoft Learn](https://learn.microsoft.com/en-us/windows/wsl/tutorials/gui-apps).
- Teed & Deng, DROID-SLAM, NeurIPS 2021: [paper](https://arxiv.org/abs/2108.10869), [official implementation](https://github.com/princeton-vl/DROID-SLAM).
- Campos et al., ORB-SLAM3, IEEE T-RO 2021: [paper](https://doi.org/10.1109/TRO.2021.3075644), [official implementation](https://github.com/UZ-SLAMLab/ORB_SLAM3).
- Xu et al., FAST-LIO2: [paper](https://arxiv.org/abs/2107.06829), [official implementation](https://github.com/hku-mars/FAST_LIO).
- Zheng et al., FAST-LIVO2: [paper](https://arxiv.org/abs/2408.14035), [official implementation](https://github.com/hku-mars/FAST-LIVO2).
- Zhou et al., FAST-LIVO2 on Resource-Constrained Platforms (2025 preprint): [paper](https://arxiv.org/abs/2501.13876).
- Vemulapati & Chen, ORB-based SLAM accelerator on SoC FPGA (2022): [paper](https://arxiv.org/abs/2207.08405).
- Ye & Luján, FPGA vs GPU feature detectors for Visual SLAM (2025 preprint): [paper](https://arxiv.org/abs/2510.13546).
- Sugiura & Matsutani, low-cost FPGA LiDAR SLAM accelerator (2021): [paper](https://arxiv.org/abs/2103.09523).
- Sugiura & Matsutani, FPGA point-cloud registration accelerator (2022): [paper](https://arxiv.org/abs/2203.05763).
