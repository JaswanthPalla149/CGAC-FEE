# Confidence-Guided Adaptive Monocular SLAM

The native C/C++ implementation is built with CMake and OpenCV. It provides an adaptive monocular feature-SLAM benchmark for EuRoC-format MAV sequences:

- **C controller API:** confidence-based low-cost/recovery selection with hysteresis.
- **C++ front end:** Shi-Tomasi features, pyramidal Lucas-Kanade tracking, essential-matrix RANSAC bootstrap/fallback, two-view triangulation, and map-based PnP RANSAC.
- **C++ evaluator:** EuRoC-format input, fixed-low/fixed-high/adaptive runs, synthetic degradations, similarity-aligned ATE, translation RPE, and CSV outputs.

The benchmark runtime is C/C++ and OpenCV. Optional batch and CSV-analysis scripts use only Python's standard library.

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

Each run writes frame and trajectory CSVs, a map CSV with landmarks/keyframes, and `summary.csv`. A tracking failure is a non-initial frame where neither map-based PnP nor essential-matrix motion estimation produced a pose with at least the configured minimum inliers; the failure rate is over non-initial frames. ATE is position RMSE after one global Umeyama similarity alignment. Translation RPE is the RMSE of adjacent-frame translation increments after applying that same global alignment; it is not a full six-degree-of-freedom pose RPE. Ground truth is interpolated at image timestamps, and frames outside its timestamp range are excluded.

On Windows, each configuration samples the PDH `Energy Meter(RAPL_Package0_PKG)\Energy` counter before and after the frame loop and converts the pWh delta to joules. `energy_scope=cpu_package_shared` means CPU-package energy, including idle consumption and other processes—not process-only or wall-socket energy. The summary records the counter method and status. If the counter cannot be read, energy stays `not_measured` with an explicit status; platforms without this counter do not get model-based numbers inserted. The UAV simulator's estimated energy is not used in the benchmark energy column.

Memory columns distinguish RSS before/after the run from `peak_rss_bytes`, the maximum process RSS sampled at frame boundaries during that run. Brief within-frame peaks may be missed; this is not a per-module allocation measurement.

The monocular controller defaults to entering recovery below `0.48` and returning to low-cost above `0.60`. The exit threshold is a provisional adjustment from the observed synthetic confidence trace (which peaked at `0.621` under the old `0.68` exit threshold), not a calibrated research result. Override thresholds with `--enter-recovery-below` and `--exit-recovery-above`; frame CSVs retain confidence and mode so switches can be inspected.

To compare the adaptive run with both fixed baselines from one or more summaries:

```powershell
python scripts\analyze_benchmark.py --input results\MH_01_easy\summary.csv results\MH_02_easy\summary.csv --output results\comparison.csv
```

The comparison CSV reports per-sequence/degradation deltas and marks whether adaptive dominates, is dominated, or presents a trade-off across available ATE, RPE, failure-rate, latency, peak-RSS, and comparable measured-energy metrics. Energy is compared only when both rows have measured values with the same method and scope; shared package energy is still affected by unrelated system activity. Missing metrics are omitted rather than treated as zero. Each input must contain unique runs for its sequence/degradation/configuration combination.

To run a real EuRoC evaluation, provide the dataset locally and run the benchmark on each sequence under identical machine, thread-count, and degradation settings. The repository currently contains only a synthetic EuRoC-format sequence, not the real EuRoC dataset; its results are pipeline checks, not evidence on EuRoC.

The batch runner discovers a sequence passed directly or sequences one directory below the dataset root, runs every selected degradation for all three configurations, and writes per-sequence outputs plus a combined summary and campaign manifest. `--repeats 5` runs five complete campaigns, stores each in a separate repeat directory, and tags each row in the combined summary with its repeat number. The comparison tool pairs runs with the same repeat number and keeps repetitions separate rather than averaging them:

```powershell
python scripts\run_euroc_batch.py --benchmark build\cpp\Release\adaptive_slam_benchmark.exe --dataset-root C:\datasets\EuRoC --output results\euroc_campaign --degradations none,blur,low_light,noise --threads 1 --repeats 5 --enter-recovery-below 0.48 --exit-recovery-above 0.60
python scripts\analyze_benchmark.py --input results\euroc_campaign\summary.csv --output results\euroc_campaign\comparisons.csv
```

## Core tracker scope

The native implementation uses OpenCV's optimized kernels, reuses the CLAHE operator, caches the previous prepared image when the processing mode is unchanged, and defaults to one OpenCV thread for repeatable comparisons. A mode change requires preparing the prior frame in the new mode.

This is a research baseline, not an ORB-SLAM3 equivalent. Monocular translation scale is arbitrary. There is no loop closure, relocalization after tracking loss, bundle adjustment, place recognition, or stereo/inertial fusion. The map is incrementally triangulated and is not globally optimized, so drift and bad map points can accumulate.

ATE and RPE use timestamp-interpolated ground truth and per-run similarity alignment; these are not metric-scale errors and can hide monocular scale drift. Blur, low-light, and seeded additive-noise conditions are synthetic approximations, not calibrated camera or environment models. The available synthetic sequence has limited scene/motion diversity. Repeat timing and package-energy trials, keep the same machine conditions, and report the shared CPU-package scope when presenting results.

There is no ORB-SLAM3 integration in this repository. A defensible comparison requires a separately obtained ORB-SLAM3 checkout and its dependencies, compatible camera settings, and a documented evaluation protocol. This prototype does not implement ORB-SLAM3's loop closure, relocalization, bundle adjustment, or visual-inertial pipeline, so it must not be described as equivalent.

## C controller API

`cpp\include\adaptive_slam\controller.h` exposes the hysteresis controller to C programs. It returns status codes for invalid inputs and has no OpenCV dependency. The C++ visual-SLAM API is declared in `cpp\include\adaptive_slam\slam.hpp`.

## Appendix: Simulated UAV, LiDAR/IMU, and hardware-oriented prototypes

The material below is a software simulation/architecture exploration, not a validated LiDAR-inertial SLAM system or hardware implementation. The benchmark synthesizes LiDAR scans and IMU inputs; it does not ingest physical sensors. The software HBST and integer-CLAHE code is not FPGA-accelerated, and operation counts do not establish measured energy savings.

### UAV hybrid simulation

`adaptive_slam/uav_hybrid_slam.hpp` contains an experimental three-mode software path. It can skip visual processing for blackout frames in the simulation. The generated point cloud is not a Livox sensor capture; the IMU stream is generated in `main.cpp` at camera-frame cadence, not at a validated 400 Hz. `processImu` now propagates orientation and integrates accelerometer specific force with configured gravity using a simple discrete kinematic update. It does not perform bias estimation, covariance propagation, or factor-graph IMU pre-integration.

The UAV demo's `estimatedEnergyJoules` is a rough MAC/time model only. It is not a measurement and is not written into the benchmark's `energy_joules` field. Benchmark RAPL readings are separate shared CPU-package deltas for each visual configuration.

#### Run the simulation

```powershell
.\build\cpp\Release\adaptive_slam_benchmark.exe --sequence synthetic\ecolivo-wsl --output results\ecolivo_sim --uav-hybrid
```

#### Inject a synthetic illumination shock

```powershell
.\build\cpp\Release\adaptive_slam_benchmark.exe --sequence synthetic\ecolivo-wsl --output results\blackout_sim --uav-hybrid --test-shock blackout
```

### WSL / Linux simulated terminal workflow

EcoLIVO includes a complete laptop-only ROS-style simulation. It does not require a ROS installation,
ROS master, real LiDAR, real IMU, or real camera. The simulation provides the same four steps used in
the FAST-LIVO workflow:

1. Start an EcoLIVO mapping process.
2. Record or generate a synthetic playback bag.
3. Play the bag frame-by-frame through the mapping process.
4. Attach Linux `perf` and generate a flamegraph.

The synthetic bag contains camera frame timestamps plus marker topics for `/camera/image_raw`,
`/lidar/points`, and `/imu/data`. The benchmark currently synthesizes the LiDAR scan and does not
read real sensor messages; this is a deterministic software benchmark, not a hardware sensor
integration.

#### Install

Ubuntu 20.04 ships CMake 3.16, while this project requires CMake 3.18 or newer:

```bash
sudo apt update
sudo apt install cmake build-essential libopencv-dev linux-tools-generic \
  linux-tools-common git python3 python3-pip
python3 -m pip install --user cmake
export PATH="$HOME/.local/bin:$PATH"
```

Build once:

```bash
chmod +x scripts/*.sh scripts/*.py scripts/roslaunch scripts/rosbag
cmake -S . -B build-wsl -DCMAKE_BUILD_TYPE=Release
cmake --build build-wsl --parallel
ctest --test-dir build-wsl --output-on-failure
```

Rebuild whenever C++ or wrapper source changes. This is important because WSL may otherwise select an
existing executable from `/tmp/ecolivo-build` or an older `build-wsl` directory.

Install FlameGraph tools for SVG generation:

```bash
git clone https://github.com/brendangregg/FlameGraph.git "$HOME/FlameGraph"
```

#### T1: start the synthetic mapping process

Before starting, keep the shell wrappers normalized to LF and run them directly from the repository
root. Do not paste Windows `\r` text or add a literal backslash as a command argument. The wrappers
provide the top-level FAST-LIVO workflow while running the native C++ benchmark.

Generate a synthetic EuRoC sequence and bag:

```bash
python3 scripts/generate_synthetic_euroc.py \
  --output synthetic/ecolivo-wsl \
  --frames 300 \
  --width 640 \
  --height 480
```

Start the EcoLIVO node. ROS launch arguments use `name:=value` syntax:

```bash
./scripts/roslaunch ecolivo mapping_avia.launch \
  sequence:=synthetic/ecolivo-wsl \
  output:=results/ecolivo \
  uav_hybrid:=true \
  fps:=1 \
  threads:=1
```

T1 remains attached to the benchmark process and writes live terminal output to
`.eco_mapping.log`. Keep this terminal running while T2 is played from another terminal.

The launch displays a ROS-style startup summary and an RViz-equivalent 3D terminal visualization
for synthetic LiDAR points and the estimated trajectory. This is a live terminal 3D display;
EcoLIVO does not create video recordings.

This is the simulator equivalent of:

```text
roslaunch fast_livo mapping_avia.launch
fast_livo mapping_avia.launch
```

The mapping process writes PID 1371 style information to `.eco_mapping.pid` and output to
`.eco_mapping.log`.

#### T2: play the synthetic rosbag

For the FAST-LIVO-style command shown below, WSL automatically maps
`./bags/Retail_Street.bag` to the generated EcoLIVO synthetic bag and enables the RViz-style GUI:

```bash
./scripts/rosbag play ./bags/Retail_Street.bag --fps 1
```

The generated-bag form is also supported:

```bash
./scripts/rosbag play ./synthetic/ecolivo-wsl/EcoLIVO_Retail_Street.bag
```

This is the simulator equivalent of:

```bash
rosbag play ./bags/Retail_Street.bag
```

To replay with the same bag but an illumination shock:

```bash
./scripts/run_wsl_simulation.sh launch \
  --sequence synthetic/ecolivo-wsl \
  --output results/blackout \
  --uav-hybrid \
  --test-shock blackout \
  --fps 1 \
  --threads 1
```

Inspect bag metadata before playback:

```bash
./scripts/run_wsl_simulation.sh bag info \
  ./synthetic/ecolivo-wsl/EcoLIVO_Retail_Street.bag
```

#### T3: inspect processes and profile the simulated process

```bash
ps aux | grep adaptive_slam_benchmark
pgrep adaptive_slam_benchmark
cat .eco_mapping.pid
```

Profile PID 1371 for 30 seconds:

```bash
PID=$(pgrep adaptive_slam_benchmark | head -n 1)
sudo perf record -F 99 -p "$PID" -g -- sleep 30
sudo chown "$(id -u):$(id -g)" perf.data

perf script -i perf.data |
  "$HOME/FlameGraph/stackcollapse-perf.pl" |
  "$HOME/FlameGraph/flamegraph.pl" > ecolivo_profile.svg
```

Or use the repository wrapper:

```bash
./scripts/run_wsl_simulation.sh profile 30
```

The wrapper writes:

```text
results/perf/ecolivo_perf.data
results/perf/ecolivo_profile.svg
```

#### Stop and rerun

```bash
./scripts/run_wsl_simulation.sh stop
tail -f .eco_mapping.log
```

At `--fps 5`, 300 frames take about 60 seconds. At `--fps 1`, they take about 300 seconds and leave
enough time for a 30-second profile. Use `--fps 0` for full-speed replay.
