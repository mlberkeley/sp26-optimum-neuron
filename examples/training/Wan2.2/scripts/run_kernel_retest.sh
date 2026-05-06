#!/usr/bin/env bash
# Re-runs the three NKI kernel benches after the agent's compile-error fixes
# (4cd071fb RMSNorm, 45439ca4 RoPE). Wraps each in run_with_alert.sh so the
# Monitor can react to FAIL/SHORT/OK.
set -u

ROOT="/home/ubuntu/sp26-optimum-neuron/examples/training/Wan2.2"
cd "$ROOT"
source ~/trn_workspace/native_venv/bin/activate

mkdir -p logs/overnight

declare -A R
ANY_FAIL=0

run_one() {
  local label="$1"; local file="$2"
  echo "--- retest stage: ${label} ---"
  MIN_OK_SECS=20 bash scripts/run_with_alert.sh "${label}_retest" "logs/overnight/${label}_retest.log" -- \
    env NEURON_KERNEL_TEST=1 RUN_KERNEL_BENCH=1 \
    python -m pytest -xvs "tests/kernels/${file}"
  local s=$?
  R[$label]=$s
  [ "$s" -ne 0 ] && ANY_FAIL=1
  echo ""
}

run_one rmsnorm test_rmsnorm_nki.py
run_one rope     test_rope_nki.py
run_one attention test_attention_nki.py
# layernorm is opportunistic — only runs if the writer-agent landed it
if [ -f "tests/kernels/test_layernorm_nki.py" ]; then
  run_one layernorm test_layernorm_nki.py
else
  echo "(layernorm test file not present; skipping)"
fi

echo "=== retest summary $(date -u +%FT%TZ) ==="
for k in "${!R[@]}"; do
  status="ok"; [ "${R[$k]}" -ne 0 ] && status="fail(${R[$k]})"
  echo "  ${k}: ${status}"
done

if [ "$ANY_FAIL" -eq 0 ]; then
  echo "[ALERT:RETEST_OK] all 3 kernels passed"
else
  echo "[ALERT:RETEST_FAIL] at least one kernel still broken"
fi
