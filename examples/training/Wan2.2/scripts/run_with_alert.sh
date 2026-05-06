#!/usr/bin/env bash
# Robust runner: invokes a child script and emits structured ALERT lines
# that a Monitor can grep for. Designed so an overnight pipeline never
# exits silently — every termination path produces a clear status marker.
#
# Markers (always one of these is emitted at end):
#   [ALERT:START]    label=<L> cmd=<...> at <iso>
#   [ALERT:OK]       label=<L> elapsed=<S>s
#   [ALERT:FAIL]     label=<L> elapsed=<S>s exit=<N> — see <log>; last error lines:
#                    <up to 5 grep'd error lines from the log>
#   [ALERT:TIMEOUT]  label=<L> elapsed=<S>s — child exceeded MAX_SECS
#   [ALERT:SHORT]    label=<L> elapsed=<S>s — child exited in less than MIN_OK_SECS
#                    (likely an init-time failure even if exit=0)
#
# Env hooks:
#   MIN_OK_SECS  — if the child exits cleanly faster than this, treat as suspicious
#                  and emit [ALERT:SHORT]. Default: 30. Set to 0 to disable.
#   MAX_SECS     — kill the child after this many seconds. Default: unlimited (0).
#
# Usage:
#   scripts/run_with_alert.sh <LABEL> <LOGFILE> -- <cmd> [args...]

set -u

if [ "$#" -lt 4 ]; then
  echo "[ALERT:FAIL] label=runner_misuse exit=2 — usage: run_with_alert.sh LABEL LOG -- CMD [ARGS...]"
  exit 2
fi

LABEL="$1"; shift
LOGFILE="$1"; shift
[ "$1" = "--" ] || { echo "[ALERT:FAIL] label=${LABEL} exit=2 — missing -- separator"; exit 2; }
shift

MIN_OK_SECS=${MIN_OK_SECS:-30}
MAX_SECS=${MAX_SECS:-0}

mkdir -p "$(dirname "$LOGFILE")"

T0=$(date +%s)
echo "[ALERT:START] label=${LABEL} cmd=$* log=${LOGFILE} at $(date -u +%FT%TZ)"

if [ "$MAX_SECS" -gt 0 ]; then
  timeout --preserve-status -k 30 "$MAX_SECS" "$@" > "$LOGFILE" 2>&1
else
  "$@" > "$LOGFILE" 2>&1
fi
EXIT=$?

T1=$(date +%s)
DUR=$((T1 - T0))

if [ "$EXIT" -eq 124 ] || [ "$EXIT" -eq 137 ]; then
  echo "[ALERT:TIMEOUT] label=${LABEL} elapsed=${DUR}s — child killed after MAX_SECS=${MAX_SECS}"
  exit "$EXIT"
fi

if [ "$EXIT" -ne 0 ]; then
  ERR_LINES=$(grep -nE "Error|Traceback|RuntimeError|ImportError|FAILED|OOM|HBM|nrt_|dmem_|CUDA error|out of memory" "$LOGFILE" 2>/dev/null | tail -5 | tr '\n' '|')
  echo "[ALERT:FAIL] label=${LABEL} elapsed=${DUR}s exit=${EXIT} log=${LOGFILE} last_errs=${ERR_LINES:-none}"
  exit "$EXIT"
fi

if [ "$MIN_OK_SECS" -gt 0 ] && [ "$DUR" -lt "$MIN_OK_SECS" ]; then
  ERR_LINES=$(grep -nE "Error|Traceback|RuntimeError|FAILED|OOM|HBM|nrt_|dmem_" "$LOGFILE" 2>/dev/null | tail -5 | tr '\n' '|')
  echo "[ALERT:SHORT] label=${LABEL} elapsed=${DUR}s — finished suspiciously fast (<${MIN_OK_SECS}s); log=${LOGFILE}; last_errs=${ERR_LINES:-none}"
  exit 0
fi

echo "[ALERT:OK] label=${LABEL} elapsed=${DUR}s log=${LOGFILE}"
exit 0
