#!/usr/bin/env bash
# Step D pivot: legal SP values for num_heads=24 are {1,2,3,4,6,8,12,24}.
# SP=16/32 fail with all_to_all chunk-mismatch (24/16=1.5 → torch.chunk yields
# 12 chunks not 16). Re-sweep at SP=12 and SP=24 instead.
#
# Frame sizes scaled up to bracket the new ceiling: SP=12 ~1.5x of SP=8 (which
# OOM'd between 81 and 121); SP=24 ~3x.
set -u

ROOT="/home/ubuntu/sp26-optimum-neuron/examples/training/Wan2.2"
cd "$ROOT"

CONFIGS=(
  "12 81  D2_sp12_f81"
  "12 121 D2_sp12_f121"
  "12 161 D2_sp12_f161"
  "24 121 D2_sp24_f121"
  "24 161 D2_sp24_f161"
  "24 241 D2_sp24_f241"
)

for cfg in "${CONFIGS[@]}"; do
  read SP FRAME TAG <<<"$cfg"
  echo "=== [step_d2] launching SP=$SP frame=$FRAME tag=$TAG at $(date -u +%FT%TZ) ==="
  bash scripts/run_sweep.sh "$SP" "$FRAME" 8 1 "$TAG" skip
  echo "=== [step_d2] finished SP=$SP frame=$FRAME tag=$TAG at $(date -u +%FT%TZ) ==="
  sleep 5
done

echo "=== [step_d2] ALL DONE at $(date -u +%FT%TZ) ==="
