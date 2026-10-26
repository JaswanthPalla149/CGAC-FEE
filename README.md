# Confidence-Guided Adaptive Monocular SLAM

The native C/C++ implementation is built with CMake and OpenCV. It provides an adaptive monocular feature-SLAM benchmark for EuRoC MAV sequences:

- **C controller API:** confidence-based low-cost/recovery selection with hysteresis.
- **C++ front end:** Shi-Tomasi features, pyramidal Lucas-Kanade tracking, essential-matrix RANSAC bootstrap/fallback, two-view triangulation, and map-based PnP RANSAC.
- **C++ evaluator:** EuRoC input, fixed-low/fixed-high/adaptive runs, synthetic degradations, similarity-aligned ATE, and CSV trajectory/frame/map/summary outputs.

The native C/C++ project is the only maintained runtime; it does not require Python.

## Build

Requirements:

- CMake 3.18 or newer.
- A C11/C++17 compiler (Visual Studio 2022, GCC, or Clang).
- OpenCV 4 development package with `core`, `imgproc`, `imgcodecs`, `video`, and `calib3d`.

Configure and build from the workspace root:

```powershell
cmake -S . -B build -DOpenCV_DIR="C:\path\to\opencv\build"
cmake --build build --config Release --parallel
ctest --test-dir build -C Release --output-on-failure
```

If CMake can already find `OpenCVConfig.cmake`, omit `-DOpenCV_DIR`. The executable is `build\cpp\Release\adaptive_slam_benchmark.exe` with Visual Studio and `build\cpp\adaptive_slam_benchmark` with single-configuration generators.

## Run

```powershell
.\build\cpp\Release\adaptive_slam_benchmark.exe --sequence C:\datasets\MH_01_easy --output results
```

The sequence should contain `mav0\cam0\data.csv`, `mav0\cam0\data\`, and `mav0\cam0\sensor.yaml`. EuRoC ground truth at `mav0\state_groundtruth_estimate0\data.csv` is optional; it is needed for ATE.

Run all four synthetic visual conditions:

```powershell
.\build\cpp\Release\adaptive_slam_benchmark.exe --sequence C:\datasets\MH_01_easy --output results --degradations none,blur,low_light,noise --threads 1
```

`--threads` controls OpenCV's worker count and defaults to one to avoid uncontrolled CPU parallelism in power comparisons. Run each configuration under the same thread setting and machine conditions.

Each run writes frame and trajectory CSVs, a map CSV with landmarks/keyframes, and `summary.csv`. Summary fields include similarity-aligned ATE, failure count/rate, latency, CPU/wall time, RSS snapshots, map sizes, and recovery-mode fraction. Energy is marked `not_measured`; latency is not a substitute for power measurement.

## Optimization choices and scope

The attached accelerator material motivates data reuse, locality, and controlled parallelism. The native implementation uses OpenCV's optimized SIMD kernels, a reusable CLAHE instance for recovery frames, pre-reserved feature/map/keyframe storage, in-place vector updates where practical, and one-thread-by-default execution. Most importantly, the controller avoids the recovery configuration on frames where tracking confidence is already sufficient.

**Winograd is not used intentionally.** Winograd transforms reduce multiply counts for dense small-kernel CNN convolutions. This SLAM front end is dominated by sparse feature tracking, geometric estimation, and image operations already provided by OpenCV; it has no dense convolution layer where adding Winograd would improve the workload. The supplied HLS matcher in `ORB_SLAM2_Matching\` remains a separate optional experiment and is not part of this CPU executable.

This is a research baseline, not an ORB-SLAM3 equivalent. Monocular translation scale is arbitrary. There is no loop closure, relocalization after tracking loss, bundle adjustment, place recognition, or stereo/inertial fusion. The map is incrementally triangulated and is not globally optimized, so drift and bad map points can accumulate.

ATE uses timestamp-interpolated ground truth and per-run similarity alignment; this is not metric-scale ATE and can hide scale drift. Memory is reported as process RSS snapshots, not peak use. Blur, low-light, and seeded additive-noise conditions are synthetic approximations. For energy claims, use a calibrated external meter or a verified platform energy counter and document the measurement method.

## C controller API

`cpp\include\adaptive_slam\controller.h` exposes the hysteresis controller to C programs. It returns status codes for invalid inputs and has no OpenCV dependency. The C++ visual-SLAM API is declared in `cpp\include\adaptive_slam\slam.hpp`.

## 3D UAV Hybrid LiDAR-Visual-Inertial SLAM (Low-MAC Extension)

To support power-constrained 3D UAV platforms and provide resilience to sudden illumination collapse (e.g. 0-lux pitch darkness, tunnel entry, or solar glare), the framework includes a multi-modal hybrid extension (`adaptive_slam/uav_hybrid_slam.hpp`):

- **Hardware-Accelerated HBST Matcher (`adaptive_slam/hbst_matcher.hpp`):** Integrates the Eytzinger binary search tree matching from `ORB_SLAM2_Matching/` using 256-bit binary descriptors and bitwise XOR + Popcount (**0 floating-point MACs**).
- **Integer-Only CLAHE (`adaptive_slam/low_mac_features.hpp`):** Fixed-point histogram clipping and redistribution with zero floating-point operations.
- **Multi-Sensor 3-Tier Controller:**
  - `LowCost`: Nominal feature budget (450 features), 2 pyramid levels, 5x downsampled LiDAR.
  - `Recovery`: Activated when visual confidence drops ($0.20 \le C_v < 0.48$). Runs integer CLAHE, 1200 features, 4 pyramid levels.
  - `VisualBypassLIO`: Triggered when light collapses completely ($C_v < 0.20$, e.g. 0-lux blackout or blinding solar flare). Bypasses image processing entirely (**0 visual MACs**), preserving 3D metric navigation purely via LiDAR and IMU until lighting is restored.
- **Direct Metric Scale:** Leverages 3D LiDAR point clouds and high-rate IMU pre-integration ($400\text{ Hz}$) on $SO(3) \times \mathbb{R}^3$, eliminating monocular scale ambiguity.

### Run UAV Hybrid SLAM Benchmark

```powershell
.\build\cpp\Release\adaptive_slam_benchmark.exe --sequence C:\datasets\MH_01_easy --output results --uav-hybrid
```

### Test Sudden Illumination Shock Resilience

Simulate sudden optical blackout or blinding solar flare:

```powershell
.\build\cpp\Release\adaptive_slam_benchmark.exe --sequence C:\datasets\MH_01_easy --output results --uav-hybrid --test-shock blackout
```

