#!/usr/bin/env bash
set -e

# Project root directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKSPACE_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${WORKSPACE_ROOT}"

FLAMEGRAPH_DIR="${HOME}/FlameGraph"
OUTPUT_SVG="${WORKSPACE_ROOT}/cgac_slam_simulation_flamegraph.svg"
SEQUENCE_DIR="${WORKSPACE_ROOT}/simulated_sequence"

echo "=========================================================="
echo " CGAC-FEE: Profiling Simulation & Generating Flame Graph   "
echo "=========================================================="

# 1. Ensure FlameGraph scripts exist
if [ ! -d "${FLAMEGRAPH_DIR}" ]; then
    echo "Cloning FlameGraph toolkit..."
    git clone https://github.com/brendangregg/FlameGraph.git "${FLAMEGRAPH_DIR}"
fi

# 2. Build binaries with frame pointers and debug symbols
echo "[1/4] Configuring & Building CGAC-FEE..."
cmake -B build -DCMAKE_BUILD_TYPE=RelWithDebInfo -DCMAKE_CXX_FLAGS="-fno-omit-frame-pointer -g" -DCMAKE_C_FLAGS="-fno-omit-frame-pointer -g"
cmake --build build -j$(nproc)

# 3. Ensure simulation sequence dataset exists
if [ ! -d "${SEQUENCE_DIR}/mav0/cam0/data" ] || [ -z "$(ls -A "${SEQUENCE_DIR}/mav0/cam0/data" 2>/dev/null)" ]; then
    echo "[2/4] Generating synthetic flight dataset (120 frames)..."
    python3 scripts/generate_simulation.py --output "${SEQUENCE_DIR}" --frames 120
else
    echo "[2/4] Found existing synthetic sequence in ${SEQUENCE_DIR}"
fi

# 4. Profile the simulation using perf record
echo "[3/4] Recording performance profile with Linux perf..."
perf record -F 99 -g -o perf.data -- \
    ./build/cpp/adaptive_slam_benchmark \
    --sequence "${SEQUENCE_DIR}" \
    --output results \
    --uav-hybrid \
    --test-shock blackout \
    --degradations none,blur,low_light,noise

# 5. Collapse stacks and render Flame Graph SVG
echo "[4/4] Generating Flame Graph SVG..."
perf script -i perf.data | \
    "${FLAMEGRAPH_DIR}/stackcollapse-perf.pl" | \
    "${FLAMEGRAPH_DIR}/flamegraph.pl" --title "CGAC-FEE Adaptive SLAM and EcoLIVO Simulation Profile" \
    > "${OUTPUT_SVG}"

cp "${OUTPUT_SVG}" "${HOME}/cgac_slam_simulation_flamegraph.svg"

echo "=========================================================="
echo " Flame Graph successfully generated!"
echo " Saved to: ${OUTPUT_SVG}"
echo " Also mirrored to: ${HOME}/cgac_slam_simulation_flamegraph.svg"
echo "=========================================================="
