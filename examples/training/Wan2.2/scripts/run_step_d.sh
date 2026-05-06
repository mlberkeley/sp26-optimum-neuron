#!/usr/bin/env bash
# Step D: SP=16 / SP=32 sweep at f=81/121/161, sample_steps=8, --skip_decode.
# Runs configurations sequentially (single device); on each completion the row
# is appended to outputs/sweep_summary.csv by run_sweep.sh.
set -u

ROOT="/home/ubuntu/sp26-optimum-neuron/examples/training/Wan2.2"
cd "$ROOT"

CONFIGS=(
  "16 81  D_sp16_f81"
  "16 121 D_sp16_f121"
  "16 161 D_sp16_f161"
  "32 81  D_sp32_f81"
  "32 121 D_sp32_f121"
  "32 161 D_sp32_f161"
)

for cfg in "${CONFIGS[@]}"; do
  read SP FRAME TAG <<<"$cfg"
  echo "=== [step_d] launching SP=$SP frame=$FRAME tag=$TAG at $(date -u +%FT%TZ) ==="
  bash scripts/run_sweep.sh "$SP" "$FRAME" 8 1 "$TAG" skip
  echo "=== [step_d] finished SP=$SP frame=$FRAME tag=$TAG at $(date -u +%FT%TZ) ==="
  # brief pause to let driver settle between launches
  sleep 5
done

echo "=== [step_d] ALL DONE at $(date -u +%FT%TZ) ==="
