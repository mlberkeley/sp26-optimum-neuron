#!/usr/bin/env bash
# Step E launcher: produces the long-video artifact at the chosen
# (SP, chunk_frame_num) once Step C/D sweeps identify the OOM ceiling.
# Wraps run_sweep.sh with the production-quality settings (no skip_decode,
# higher sample_steps, multi-chunk rolling) so the artifact is full-fidelity.
#
# Usage:
#   scripts/run_step_e.sh <SP> <FRAME_NUM> [<N_CHUNKS=3>] [<SAMPLE_STEPS=24>]
#
# Output:
#   outputs/sweep_sp${SP}_f${FRAME}_s${STEPS}_n${N_CHUNKS}_step_e.mp4
#   logs/sweep_sp${SP}_f${FRAME}_s${STEPS}_n${N_CHUNKS}_step_e.log
#   outputs/sweep_summary.csv  (appended row)

set -u

SP=${1:?missing SP}
FRAME=${2:?missing FRAME (must be 4n+1)}
NCHUNKS=${3:-3}
STEPS=${4:-24}

ROOT="/home/ubuntu/sp26-optimum-neuron/examples/training/Wan2.2"
exec bash "$ROOT/scripts/run_sweep.sh" "$SP" "$FRAME" "$STEPS" "$NCHUNKS" step_e
