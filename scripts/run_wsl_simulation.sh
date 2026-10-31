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

        executable="$(find_executable || true)"
        if [[ -z "$executable" ]]; then
            echo "Benchmark executable not found. Build it with:" >&2
            echo "  cmake -S \"$ROOT\" -B \"$ROOT/build-wsl\" -DCMAKE_BUILD_TYPE=Release" >&2
            echo "  cmake --build \"$ROOT/build-wsl\" --parallel" >&2
            exit 1
        fi

        cd "$ROOT"
        "$executable" "$@" >"$ROOT/.eco_mapping.log" 2>&1 &
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

    *)
        echo "Usage:"
        echo "  $0 launch [benchmark arguments]"
        echo "  $0 profile SECONDS [--flamegraph /path/to/FlameGraph]"
        echo "  $0 stop"
        exit 2
        ;;
esac
