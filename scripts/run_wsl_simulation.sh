#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PID_FILE="$ROOT/.eco_mapping.pid"

find_executable() {
    local candidates=(
        "/tmp/ecolivo-build/cpp/adaptive_slam_benchmark"
        "$ROOT/build-wsl/cpp/adaptive_slam_benchmark"
        "$ROOT/build/cpp/adaptive_slam_benchmark"
        "$ROOT/build/cpp/Release/adaptive_slam_benchmark"
        "$ROOT/build/cpp/Release/adaptive_slam_benchmark.exe"
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

        cd "$ROOT"
        "$executable" "${cleaned_args[@]}" >"$ROOT/.eco_mapping.log" 2>&1 &
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

            fps="1"
            for arg in "$@"; do
                if [[ "$arg" == "--fps" ]]; then
                    shift
                    fps="${1:-1}"
                    break
                fi
            done
            if [[ -f "$PID_FILE" ]]; then
                active_pid="$(cat "$PID_FILE" 2>/dev/null || true)"
                if [[ -n "$active_pid" ]] && kill -0 "$active_pid" 2>/dev/null; then
                    echo "[rosbag] Published $frame_count synthetic messages to active EcoLIVO mapping PID $active_pid"
                    while kill -0 "$active_pid" 2>/dev/null; do
                        sleep 1
                    done
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
