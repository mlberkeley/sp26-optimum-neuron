#!/usr/bin/env bash
# One-shot conversion of T5 + VAE .pth checkpoints to .safetensors siblings
# in the same directory. Safe — does not modify the original .pth files; only
# writes new files alongside them. Cuts cold-cache load time from ~12 min
# (pickle deserialize) to ~1 min (mmap).
#
# Idempotent: re-running with the originals already converted is a no-op
# unless --overwrite is passed.
set -u

ROOT="/home/ubuntu/sp26-optimum-neuron/examples/training/Wan2.2"
cd "$ROOT"
source ~/trn_workspace/native_venv/bin/activate

LOG="logs/overnight/safetensors_conversion.log"

T0=$(date +%s)
python tools/convert_pth_to_safetensors.py > "$LOG" 2>&1
EXIT=$?
T1=$(date +%s)
DUR=$((T1 - T0))

# Parse the converter's per-file summary (one line per converted file)
CONVERTED=$(grep -cE "^converted|^skipped" "$LOG" 2>/dev/null || echo 0)

if [ "$EXIT" -eq 0 ]; then
  echo "[ALERT:SAFETENSORS_OK] elapsed=${DUR}s files=${CONVERTED} log=${LOG}"
else
  ERRS=$(grep -nE "Error|Traceback|RuntimeError" "$LOG" 2>/dev/null | tail -3 | tr '\n' '|')
  echo "[ALERT:SAFETENSORS_FAIL] exit=${EXIT} elapsed=${DUR}s log=${LOG} last_errs=${ERRS:-none}"
fi
