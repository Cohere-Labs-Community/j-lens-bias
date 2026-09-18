#!/usr/bin/env bash
# Independent real-time safety watchdog for ONE exact Week 3 inference PID.
#
# Usage: week3_watchdog.sh <week3_pid> <stage_label>
#
# Monitors swap/memory pressure, duplicate Week 3/Qwen processes, and
# reappearance of the primary stage4r experiment. On any trigger, gracefully
# (then forcefully if needed) stops ONLY the exact Week 3 PID it was given.
# NEVER touches stage4r or any other process. NEVER uses broad kill patterns.
set -uo pipefail

WEEK3_PID="${1:?usage: week3_watchdog.sh <week3_pid> <stage_label>}"
STAGE_LABEL="${2:-unknown}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WEEK3_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
RESULTS_DIR="$WEEK3_DIR/results"
LOG_FILE="$RESULTS_DIR/overnight_run.log"
STOP_REASON_FILE="$RESULTS_DIR/watchdog_stop_reason_${STAGE_LABEL}.txt"

mkdir -p "$RESULTS_DIR"

POLL_INTERVAL=25
# Historical/sticky swap present before Week 3 must not trigger a stop; only NEW swap growth
# relative to BASELINE_SWAP_MB (written by the runner immediately before model load) does.
BASELINE_FILE="$RESULTS_DIR/week3_baseline_swap_mb.txt"
SWAP_DELTA_STOP_MB=1500
FREE_PCT_CRIT=30
FREE_PCT_CONSEC_REQUIRED=1
PRESSURE_CRITICAL_LEVEL=4
SWAP_DRIFT_MB=750
SWAP_DRIFT_WINDOW_SEC=60
MAX_MISSING_TELEMETRY=3
GRACEFUL_WAIT_SEC=10

log() {
    printf '%s [watchdog:%s] %s\n' "$(date '+%Y-%m-%dT%H:%M:%S%z')" "$STAGE_LABEL" "$1" >> "$LOG_FILE"
}

pid_alive() {
    kill -0 "$1" 2>/dev/null
}

get_swap_mb() {
    sysctl vm.swapusage 2>/dev/null | grep -oE 'used = [0-9.]+M' | grep -oE '[0-9.]+' | head -1
}

get_pressure_level() {
    sysctl -n kern.memorystatus_vm_pressure_level 2>/dev/null | head -1
}

get_free_pct() {
    memory_pressure 2>/dev/null | grep -oE 'System-wide memory free percentage: [0-9]+%' | grep -oE '[0-9]+' | head -1
}

stop_week3() {
    local reason="$1"
    log "STOP_TRIGGERED: $reason"
    printf '%s\n' "$reason" > "$STOP_REASON_FILE"
    if pid_alive "$WEEK3_PID"; then
        kill -TERM "$WEEK3_PID" 2>/dev/null || true
        local waited=0
        while pid_alive "$WEEK3_PID" && [ "$waited" -lt "$GRACEFUL_WAIT_SEC" ]; do
            sleep 1
            waited=$((waited + 1))
        done
        if pid_alive "$WEEK3_PID"; then
            kill -KILL "$WEEK3_PID" 2>/dev/null || true
            log "FORCE_KILLED_WEEK3_PID $WEEK3_PID (graceful term did not exit within ${GRACEFUL_WAIT_SEC}s)"
        fi
    fi
    log "WEEK3_STOPPED_BY_WATCHDOG: $reason"
}

log "WATCHDOG_START pid=$WEEK3_PID"

prev_swap=""
prev_swap_time=""
free_pct_low_streak=0
missing_telemetry_streak=0

while pid_alive "$WEEK3_PID"; do
    swap_mb="$(get_swap_mb || true)"
    free_pct="$(get_free_pct || true)"
    pressure_level="$(get_pressure_level || true)"
    now=$(date +%s)

    if [ -z "$swap_mb" ] || [ -z "$free_pct" ] || [ -z "$pressure_level" ]; then
        missing_telemetry_streak=$((missing_telemetry_streak + 1))
        log "TELEMETRY_UNAVAILABLE streak=$missing_telemetry_streak swap='$swap_mb' free='$free_pct'"
        if [ "$missing_telemetry_streak" -ge "$MAX_MISSING_TELEMETRY" ]; then
            stop_week3 "telemetry unavailable for $missing_telemetry_streak consecutive checks -- fail closed"
            break
        fi
        sleep "$POLL_INTERVAL"
        continue
    fi
    missing_telemetry_streak=0

    dup_pids="$(pgrep -f "run_week3_experiment.py" 2>/dev/null | grep -vx "$WEEK3_PID" || true)"
    stage4r_pids="$(pgrep -f "stage4r_run.py --algorithm gradient" 2>/dev/null || true)"

    # BASELINE_SWAP_MB: runner-recorded value once it exists; until then the first sample.
    if [ -s "$BASELINE_FILE" ]; then
        baseline_swap="$(head -1 "$BASELINE_FILE")"
    elif [ -z "${baseline_swap:-}" ]; then
        baseline_swap="$swap_mb"
    fi

    log "sample swap=${swap_mb}MB baseline=${baseline_swap}MB free=${free_pct}% pressure_level=${pressure_level} dup_week3='${dup_pids//$'\n'/,}' stage4r_present=$([ -n "$stage4r_pids" ] && echo yes || echo no)"

    if [ -n "$stage4r_pids" ]; then
        stop_week3 "primary stage4r experiment reappeared during Week 3 stage $STAGE_LABEL (pids: ${stage4r_pids//$'\n'/,})"
        log "WEEK3_STOPPED_PRIMARY_EXPERIMENT_RETURNED"
        break
    fi

    if [ -n "$dup_pids" ]; then
        stop_week3 "duplicate Week 3 inference process detected (pids: ${dup_pids//$'\n'/,})"
        break
    fi

    swap_delta="$(awk -v a="$swap_mb" -v b="$baseline_swap" 'BEGIN{printf "%.0f", a-b}')"
    if [ "$swap_delta" -ge "$SWAP_DELTA_STOP_MB" ]; then
        stop_week3 "swap delta +${swap_delta}MB vs BASELINE_SWAP_MB=${baseline_swap} (swap_used ${swap_mb}MB) >= ${SWAP_DELTA_STOP_MB}MB"
        break
    fi

    if [ "$pressure_level" -ge "$PRESSURE_CRITICAL_LEVEL" ]; then
        stop_week3 "memory pressure level ${pressure_level} (critical)"
        break
    fi

    if [ "$free_pct" -lt "$FREE_PCT_CRIT" ]; then
        free_pct_low_streak=$((free_pct_low_streak + 1))
        log "FREE_PCT_LOW streak=$free_pct_low_streak (${free_pct}%)"
        if [ "$free_pct_low_streak" -ge "$FREE_PCT_CONSEC_REQUIRED" ]; then
            stop_week3 "free memory percentage < ${FREE_PCT_CRIT}% on $FREE_PCT_CONSEC_REQUIRED consecutive samples (${free_pct}%)"
            break
        fi
    else
        free_pct_low_streak=0
    fi

    if [ -n "$prev_swap" ] && [ -n "$prev_swap_time" ]; then
        elapsed=$((now - prev_swap_time))
        if [ "$elapsed" -le "$((SWAP_DRIFT_WINDOW_SEC + 15))" ]; then
            drift="$(awk -v a="$swap_mb" -v b="$prev_swap" 'BEGIN{printf "%.0f", a-b}')"
            if [ "$drift" -ge "$SWAP_DRIFT_MB" ]; then
                stop_week3 "swap increased ${drift}MB within ~${elapsed}s (>= ${SWAP_DRIFT_MB}MB ceiling)"
                break
            fi
        fi
    fi
    prev_swap="$swap_mb"
    prev_swap_time="$now"

    sleep "$POLL_INTERVAL"
done

log "WATCHDOG_EXIT pid=$WEEK3_PID"
