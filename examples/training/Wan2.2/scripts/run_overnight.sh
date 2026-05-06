#!/usr/bin/env bash
# Overnight pipeline: chains the remaining Phase-1 + Phase-2 work behind
# scripts/run_with_alert.sh wrappers. Every stage emits structured
# [ALERT:*] markers so a Monitor can react to failures without polling.
#
# Stages (sequential — share device):
#   1. NKI kernel benches (RMSNorm, RoPE, attention)   ~15 min
#   2. Step E long-video artifact (SP=8, f=81, n=3, s=24, full VAE)  ~5 hr
#
# Final marker:
#   [ALERT:PIPELINE_OK]    if every stage emitted [ALERT:OK]
#   [ALERT:PIPELINE_FAIL]  with a stage-by-stage summary otherwise

set -u

ROOT="/home/ubuntu/sp26-optimum-neuron/examples/training/Wan2.2"
cd "$ROOT"
source ~/trn_workspace/native_venv/bin/activate

mkdir -p logs/overnight

PIPELINE_LOG="logs/overnight/pipeline_summary.log"
echo "=== overnight pipeline start $(date -u +%FT%TZ) ===" > "$PIPELINE_LOG"

declare -A STAGE_STATUS
ANY_FAIL=0

run_stage() {
  local label="$1"
  local logfile="$2"
  shift 2
  local min_ok="${MIN_OK_SECS:-30}"
  echo "--- stage: ${label} ---" | tee -a "$PIPELINE_LOG"
  # capture the runner's emitted marker line
  MIN_OK_SECS="$min_ok" bash scripts/run_with_alert.sh "$label" "$logfile" -- "$@" 2>&1 | tee -a "$PIPELINE_LOG"
  local status="$?"
  if [ "$status" -eq 0 ]; then
    STAGE_STATUS[$label]="ok"
  else
    STAGE_STATUS[$label]="fail(${status})"
    ANY_FAIL=1
  fi
  echo "" >> "$PIPELINE_LOG"
}

# --- Stage 1a: RMSNorm bench ---
MIN_OK_SECS=20 run_stage rmsnorm_bench logs/overnight/rmsnorm.log \
  env NEURON_KERNEL_TEST=1 RUN_KERNEL_BENCH=1 \
  python -m pytest -xvs tests/kernels/test_rmsnorm_nki.py

# --- Stage 1b: RoPE bench ---
MIN_OK_SECS=20 run_stage rope_bench logs/overnight/rope.log \
  env NEURON_KERNEL_TEST=1 RUN_KERNEL_BENCH=1 \
  python -m pytest -xvs tests/kernels/test_rope_nki.py

# --- Stage 1c: Attention bench (SDPA reference + NKI candidate) ---
MIN_OK_SECS=20 run_stage attention_bench logs/overnight/attention.log \
  env NEURON_KERNEL_TEST=1 RUN_KERNEL_BENCH=1 \
  python -m pytest -xvs tests/kernels/test_attention_nki.py

# --- Stage 2: Step E artifact (SP=8, f=81, n=3, s=24) ---
# This is the long one (~5 hr including VAE decode); timeout 7 hr.
MAX_SECS=25200 MIN_OK_SECS=600 run_stage step_e logs/overnight/step_e.log \
  bash scripts/run_step_e.sh 8 81 3 24

# --- Final summary ---
echo "=== pipeline summary $(date -u +%FT%TZ) ===" | tee -a "$PIPELINE_LOG"
for k in "${!STAGE_STATUS[@]}"; do
  echo "  ${k}: ${STAGE_STATUS[$k]}" | tee -a "$PIPELINE_LOG"
done

if [ "$ANY_FAIL" -eq 0 ]; then
  echo "[ALERT:PIPELINE_OK] all stages succeeded" | tee -a "$PIPELINE_LOG"
  exit 0
else
  echo "[ALERT:PIPELINE_FAIL] at least one stage failed — see $PIPELINE_LOG" | tee -a "$PIPELINE_LOG"
  exit 1
fi
