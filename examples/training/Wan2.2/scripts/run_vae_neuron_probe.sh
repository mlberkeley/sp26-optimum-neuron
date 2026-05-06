#!/usr/bin/env bash
# Bash wrapper around scripts/probe_vae_neuron.py. Single-rank probe (no
# torchrun); writes a structured [PROBE:VAE_NEURON_*] marker to stdout that
# the overnight Monitor can grep for.
set -u

ROOT="/home/ubuntu/sp26-optimum-neuron/examples/training/Wan2.2"
cd "$ROOT"
source ~/trn_workspace/native_venv/bin/activate

LOG="logs/overnight/vae_neuron_probe.log"

T0=$(date +%s)
python scripts/probe_vae_neuron.py > "$LOG" 2>&1
EXIT=$?
T1=$(date +%s)
DUR=$((T1 - T0))

# Surface the marker to overnight.log for the Monitor to pick up.
RES=$(grep -E "\[PROBE:VAE_NEURON_OK\]|\[PROBE:VAE_NEURON_BLOCKED\]" "$LOG" | tail -1)
if [ -z "$RES" ]; then
  RES="[PROBE:VAE_NEURON_BLOCKED] op=\"probe script exited unexpectedly (exit=${EXIT})\""
fi

echo ""
echo "=== vae-neuron probe summary $(date -u +%FT%TZ) ==="
echo "  exit=${EXIT} duration=${DUR}s"
echo "  $RES"
echo "  full log: $LOG"

# Translate the probe outcome into an [ALERT] marker.
case "$RES" in
  *VAE_NEURON_OK*)
    echo "[ALERT:VAE_PROBE_OK] vae moves to neuron and decodes a small latent"
    ;;
  *)
    OP=$(echo "$RES" | sed -nE 's/.*op="([^"]*).*/\1/p' | head -c 200)
    echo "[ALERT:VAE_PROBE_BLOCKED] reason=\"${OP}\""
    ;;
esac
