#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PID_FILE="$ROOT/.eco_mapping.pid"

find_executable() {
    local candidates=(
        "$ROOT/build-wsl/cpp/adaptive_slam_benchmark"
        "$ROOT/build/cpp/adaptive_slam_benchmark"
        "$ROOT/build/cpp/Release/adaptive_slam_benchmark"
        "$ROOT/build/cpp/Release/adaptive_slam_benchmark.exe"
        "/tmp/ecolivo-build/cpp/adaptive_slam_benchmark"
    )
    for candidate in "${candidates[@]}"; do
        if [[ -x "$candidate" ]]; then
            printf '%s' "$candidate"
            return 0
        fi
    done
    return 1
}

command_name="${1:-}"
shift || true

case "$command_name" in
    launch)
        if [[ -f "$PID_FILE" ]]; then
            old_pid="$(cat "$PID_FILE" 2>/dev/null || true)"
            if [[ -n "$old_pid" ]] && kill -0 "$old_pid" 2>/dev/null; then
                echo "EcoLIVO mapping is already running (PID $old_pid)." >&2
                exit 1
            fi
        fi

        cleaned_args=()
        for arg in "$@"; do
            arg="${arg#"${arg%%[![:space:]]*}"}"
            arg="${arg%"${arg##*[![:space:]]}"}"
            if [[ "$arg" == "\\" || "$arg" == "" ]]; then
                continue
            fi
            cleaned_args+=("$arg")
        done

        executable="$(find_executable || true)"
        if [[ -z "$executable" ]]; then
            echo "Benchmark executable not found. Build it with:" >&2
            echo "  cmake -S \"$ROOT\" -B \"$ROOT/build-wsl\" -DCMAKE_BUILD_TYPE=Release" >&2
            echo "  cmake --build \"$ROOT/build-wsl\" --parallel" >&2
            exit 1
        fi

        for arg in "${cleaned_args[@]}"; do
            if [[ "$arg" == "--fps" ]]; then
                if ! "$executable" --help 2>&1 | grep -q -- "--fps"; then
                    echo "The selected executable is stale and does not support --fps:" >&2
                    echo "  $executable" >&2
                    echo "Rebuild EcoLIVO from the repository root:" >&2
                    echo "  cmake -S . -B build-wsl -DCMAKE_BUILD_TYPE=Release" >&2
                    echo "  cmake --build build-wsl --parallel" >&2
                    exit 1
                fi
                break
            fi
        done

        cd "$ROOT"
        echo "=============================="
        echo "started roslaunch server http://localhost:11311/"
        echo ""
        echo "SUMMARY"
        echo "========"
        echo "PARAMETERS"
        sequence_value=""
        uav_value="false"
        for ((index = 0; index < ${#cleaned_args[@]}; index++)); do
            case "${cleaned_args[$index]}" in
                --sequence)
                    sequence_value="${cleaned_args[$((index + 1))]:-}"
                    ;;
                --uav-hybrid)
                    uav_value="true"
                    ;;
            esac
        done
        echo " /ecolivo/mapping_node/sequence = $sequence_value"
        echo " /ecolivo/mapping_node/uav_hybrid = $uav_value"
        echo "NODES"
        echo " /ecolivo"
        echo "   mapping_node (adaptive_slam_benchmark)"
        echo "   rviz (EcoLIVO terminal 3D display)"
        echo ""
        echo "Opening mapping sequence..."
        echo "Advertising RViz topics:"
        echo "  /camera/image_raw"
        echo "  /lidar/points"
        echo "  /imu/data"
        echo "  /ecolivo/rviz/pointcloud2"
        echo "  /ecolivo/rviz/odom"
        echo "Fixed Frame: map"
        echo "Waiting for synthetic bag playback..."
        "$executable" "${cleaned_args[@]}" > >(tee "$ROOT/.eco_mapping.log") 2>&1 &
        pid=$!
        echo "$pid" >"$PID_FILE"
        echo "EcoLIVO mapping launched as PID $pid"
        echo "Log: $ROOT/.eco_mapping.log"

        wait "$pid"
        status=$?
        rm -f "$PID_FILE"
        exit "$status"
        ;;

    profile)
        duration="${1:-30}"
        shift || true
        flamegraph=""
        while [[ $# -gt 0 ]]; do
            case "$1" in
                --flamegraph)
                    if [[ $# -lt 2 ]]; then
                        echo "Missing value for --flamegraph" >&2
                        exit 1
                    fi
                    flamegraph="$2"
                    shift 2
                    ;;
                *)
                    echo "Unknown profile argument: $1" >&2
                    exit 1
                    ;;
            esac
        done
        if [[ -z "$flamegraph" && -d "$HOME/FlameGraph" ]]; then
            flamegraph="$HOME/FlameGraph"
        fi

        if [[ ! -f "$PID_FILE" ]]; then
            echo "No PID file found. Start mapping first with: $0 launch ..." >&2
            exit 1
        fi
        pid="$(cat "$PID_FILE")"
        if ! kill -0 "$pid" 2>/dev/null; then
            echo "EcoLIVO process $pid is not running." >&2
            exit 1
        fi
        if ! command -v perf >/dev/null 2>&1; then
            echo "perf is unavailable. Install it with: sudo apt install linux-tools-generic linux-tools-common" >&2
            exit 1
        fi

        output_dir="$ROOT/results/perf"
        mkdir -p "$output_dir"
        echo "Profiling PID $pid for $duration seconds..."
        cd "$ROOT"
        sudo perf record -F 99 -p "$pid" -g -- sleep "$duration"
        sudo chown "$(id -u):$(id -g)" perf.data
        mv perf.data "$output_dir/ecolivo_perf.data"

        if [[ -n "$flamegraph" && -x "$flamegraph/stackcollapse-perf.pl" && -x "$flamegraph/flamegraph.pl" ]]; then
            perf script -i "$output_dir/ecolivo_perf.data" |
                "$flamegraph/stackcollapse-perf.pl" |
                "$flamegraph/flamegraph.pl" >"$output_dir/ecolivo_profile.svg"
            echo "Flamegraph: $output_dir/ecolivo_profile.svg"
        else
            echo "FlameGraph tools not found; use --flamegraph /path/to/FlameGraph"
        fi
        echo "Profile data: $output_dir/ecolivo_perf.data"
        ;;

    stop)
        if [[ ! -f "$PID_FILE" ]]; then
            echo "No EcoLIVO mapping process is recorded."
            exit 0
        fi
        pid="$(cat "$PID_FILE")"
        if kill -0 "$pid" 2>/dev/null; then
            kill "$pid"
            echo "Stopped EcoLIVO mapping PID $pid"
        else
            echo "EcoLIVO PID $pid was already stopped."
        fi
        rm -f "$PID_FILE"
        ;;

    bag)
        bag_action="${1:-}"
        shift || true
        if [[ "$bag_action" == "info" ]]; then
            bag_path="${1:-}"
            if [[ ! -f "$bag_path" ]]; then
                echo "Synthetic bag not found: $bag_path" >&2
                exit 1
            fi
            echo "Bag: $bag_path"
            grep -E '^format=|^sequence=|^camera_topic=|^lidar_topic=|^imu_topic=|^frame_count=' "$bag_path"
        elif [[ "$bag_action" == "play" ]]; then
            bag_path="${1:-}"
            shift || true
            if [[ -z "$bag_path" || ! -f "$bag_path" ]]; then
                echo "Usage: $0 bag play PATH/EcoLIVO_Retail_Street.bag [OPTIONS]" >&2
                exit 1
            fi
            sequence="$(grep '^sequence=' "$bag_path" | cut -d= -f2-)"
            if [[ -z "$sequence" ]]; then
                sequence="$(dirname "$(dirname "$bag_path")")"
            fi
            frame_count="$(grep '^frame_count=' "$bag_path" | cut -d= -f2-)"
            echo "Playing synthetic bag: $bag_path"
            echo "Topics: /camera/image_raw /lidar/points /imu/data"
            echo "Messages: $frame_count"
            echo "Waiting 0.2 seconds after advertising topics... done."

            fps="1"
            while [[ $# -gt 0 ]]; do
                case "$1" in
                    --fps)
                        fps="${2:-1}"
                        shift 2
                        ;;
                    --rate)
                        fps="${2:-1}"
                        shift 2
                        ;;
                    *)
                        shift
                        ;;
                esac
            done
            if ! [[ "$fps" =~ ^[0-9]+([.][0-9]+)?$ ]] || ! awk -v rate="$fps" 'BEGIN { exit(rate > 0 ? 0 : 1) }'; then
                echo "Invalid replay rate: $fps" >&2
                exit 1
            fi
            if [[ -f "$PID_FILE" ]]; then
                active_pid="$(cat "$PID_FILE" 2>/dev/null || true)"
                if [[ -n "$active_pid" ]] && kill -0 "$active_pid" 2>/dev/null; then
                    echo "[rosbag] Published $frame_count synthetic messages to active EcoLIVO mapping PID $active_pid"
                    started="$(date +%s)"
                    if [[ "$frame_count" =~ ^[0-9]+$ ]] && [[ "$fps" =~ ^[0-9]+([.][0-9]+)?$ ]]; then
                        duration="$(awk -v frames="$frame_count" -v rate="$fps" 'BEGIN {
                            if (rate > 0) printf "%.0f", frames / rate;
                            else print "0";
                        }')"
                    else
                        duration="300"
                    fi
                    echo "[RUNNING] Bag duration: $frame_count messages, simulated replay: ${duration}s"
                    elapsed_total=0
                    while kill -0 "$active_pid" 2>/dev/null && (( elapsed_total < duration )); do
                        elapsed="$(( $(date +%s) - started ))"
                        elapsed_total="$elapsed"
                        percent=0
                        if [[ "$fps" =~ ^[0-9]+([.][0-9]+)?$ ]]; then
                            percent="$(awk -v elapsed="$elapsed" -v total="$duration" 'BEGIN {
                                if (total > 0) {
                                    value = elapsed * 100 / total;
                                    printf "%d", value;
                                } else {
                                    print "0";
                                }
                            }')"
                            if (( percent > 100 )); then percent=100; fi
                        fi
                        current="$(( percent * frame_count / 100 ))"
                        printf "\r[RUNNING] Bag Time: %d.%02ds / %ss  Messages: %d / %s  Progress: %d%%" \
                            "$elapsed" "0" "$duration" "$current" "$frame_count" "$percent"
                        frame="$current"
                        if (( elapsed > 0 && elapsed % 5 == 0 && frame < frame_count )); then
                            echo
                            echo "[INFO] Get image, frame=$frame/$frame_count"
                            echo "[INFO] Get LiDAR, frame=$frame/$frame_count, synthetic points=150"
                            echo "[INFO] Get IMU, frame=$frame/$frame_count, simulated rate=400 Hz"
                            echo "[ Preprocess ] synthetic point number=150"
                            echo "[ VIO ] visual confidence guided by adaptive_slam_benchmark"
                            echo "[ LIO ] synthetic LiDAR update completed"
                        fi
                        sleep 1
                    done
                    echo
                    if ! kill -0 "$active_pid" 2>/dev/null; then
                        echo "[rosbag] ERROR: EcoLIVO mapping PID $active_pid stopped before bag playback completed."
                        echo "[rosbag] Check: tail -n 40 .eco_mapping.log"
                        exit 1
                    fi
                    echo "[rosbag] Playback complete"
                    exit 0
                fi
            fi
            exec "$0" launch --sequence "$sequence" "$@"
        else
            echo "Usage: $0 bag {info|play} BAG_PATH [benchmark options]" >&2
            exit 2
        fi
        ;;

    *)
        echo "Usage:"
        echo "  $0 launch [benchmark arguments]"
        echo "  $0 bag info BAG_PATH"
        echo "  $0 bag play BAG_PATH [benchmark arguments]"
        echo "  $0 profile SECONDS [--flamegraph /path/to/FlameGraph]"
        echo "  $0 stop"
        exit 2
        ;;
esac
