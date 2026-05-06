#!/usr/bin/env bash
# Launches a single distributed-rolling configuration and writes a structured
# log + an OOM/wall-clock summary line to outputs/sweep_summary.csv.
#
# Usage:
#   scripts/run_sweep.sh <SP> <FRAME_NUM> <SAMPLE_STEPS> <N_CHUNKS> <TAG> [<SKIP_DECODE>]
# where:
#   SP           - ulysses_size (1 = single-device, no torchrun)
#   FRAME_NUM    - chunk_frame_num (must be 4n+1: 17, 33, 49, 81, 121, 161, ...)
#   SAMPLE_STEPS - denoising steps per chunk
#   N_CHUNKS     - rolling chunks (1 for sweeps, 3+ for the long-video artifact)
#   TAG          - free-form label embedded in the log/output filenames
#   SKIP_DECODE  - "skip" bypasses the rank-0 VAE decode + mp4 save. Used for
#                  OOM ceiling sweeps where CPU VAE dominates wall-clock and
#                  isn't relevant to whether denoising fits in HBM.

set -u

SP=${1:?missing SP}
FRAME=${2:?missing FRAME}
STEPS=${3:?missing STEPS}
NCHUNKS=${4:?missing NCHUNKS}
TAG=${5:?missing TAG}
SKIP=${6:-}

ROOT="/home/ubuntu/sp26-optimum-neuron/examples/training/Wan2.2"
cd "$ROOT"
source ~/trn_workspace/native_venv/bin/activate

mkdir -p logs outputs

LOGFILE="$ROOT/logs/sweep_sp${SP}_f${FRAME}_s${STEPS}_n${NCHUNKS}_${TAG}.log"
OUTMP4="$ROOT/outputs/sweep_sp${SP}_f${FRAME}_s${STEPS}_n${NCHUNKS}_${TAG}.mp4"
SUMMARY="$ROOT/outputs/sweep_summary.csv"

# initialize summary header if absent
if [ ! -f "$SUMMARY" ]; then
  echo "tag,sp,frame_num,sample_steps,n_chunks,wall_seconds,status,artifact" > "$SUMMARY"
fi

# build common args
COMMON_ARGS=(
  --device neuron
  --ckpt_dir ckpts/Wan2.2-TI2V-5B
  --image examples/i2v_input.JPG
  --prompt "A cinematic shot of waves on a rocky coast at sunset"
  --n_chunks "$NCHUNKS"
  --chunk_frame_num "$FRAME"
  --sample_steps "$STEPS"
  --seed 42
  --verbose
  --t5_cpu
  --save_file "$OUTMP4"
)

if [ "$SKIP" = "skip" ]; then
  COMMON_ARGS+=(--skip_decode)
fi

T0=$(date +%s)

if [ "$SP" -eq 1 ]; then
  python generate_rolling.py "${COMMON_ARGS[@]}" > "$LOGFILE" 2>&1
  EXIT=$?
else
  torchrun --nproc_per_node "$SP" generate_rolling.py \
    --ulysses_size "$SP" "${COMMON_ARGS[@]}" \
    > "$LOGFILE" 2>&1
  EXIT=$?
fi

T1=$(date +%s)
DURATION=$((T1 - T0))

if [ "$SKIP" = "skip" ]; then
  # success criterion when we skipped VAE decode: denoising completed and the
  # process exited cleanly. Verbose logging prints "[skip_decode]" once per
  # successful run.
  if [ "$EXIT" -eq 0 ] && grep -q "\[skip_decode\] denoising done" "$LOGFILE" 2>/dev/null; then
    STATUS="ok_no_decode"
  elif grep -qE "OutOfMemory|out of memory|OOM|HBM_OOM|dmem_alloc_internal|Failed to allocate aligned|MLA DRAM|nrt_tensor_allocate.*failed" "$LOGFILE" 2>/dev/null; then
    STATUS="oom"
  elif [ "$EXIT" -ne 0 ]; then
    STATUS="error_${EXIT}"
  else
    STATUS="failed_no_marker"
  fi
else
  if [ "$EXIT" -eq 0 ] && [ -f "$OUTMP4" ]; then
    STATUS="ok"
  elif grep -qE "OutOfMemory|out of memory|OOM|HBM_OOM|dmem_alloc_internal|Failed to allocate aligned|MLA DRAM|nrt_tensor_allocate.*failed" "$LOGFILE" 2>/dev/null; then
    STATUS="oom"
  elif [ "$EXIT" -ne 0 ]; then
    STATUS="error_${EXIT}"
  else
    STATUS="missing_artifact"
  fi
fi

echo "${TAG},${SP},${FRAME},${STEPS},${NCHUNKS},${DURATION},${STATUS},${OUTMP4}" >> "$SUMMARY"
echo "[run_sweep] tag=${TAG} sp=${SP} frame=${FRAME} steps=${STEPS} n=${NCHUNKS} → ${STATUS} in ${DURATION}s; log=${LOGFILE}"
exit "$EXIT"
