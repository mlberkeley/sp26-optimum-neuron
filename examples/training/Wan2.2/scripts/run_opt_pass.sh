#!/usr/bin/env bash
# Phase 3 opt pass: parses kernel retest outputs for speedup numbers,
# decides integration toggles per the ≥1.2× gate, then runs ONE e2e
# validation (SP=8 f=81 s=8 n=1) with all qualifying kernels integrated
# behind WAN_USE_NKI_KERNELS=1 + per-kernel toggles.
#
# Designed to be the final stage in the overnight chain:
#   step_e → kernel retest (with bench) → THIS → final report
#
# Outputs:
#   docs/kernel_inventory_profiled.md  — gate verdict per kernel
#   logs/overnight/opt_e2e.log         — e2e validation log
#   logs/overnight/opt_pass.log        — this script's structured log
#   sweep_summary.csv                  — appended row "kernel_e2e_<flags>"
#
# Final marker: [ALERT:OPT_PASS_OK] | [ALERT:OPT_PASS_PARTIAL N/M]
#               (M = total candidates, N = e2e successes among integrations)

set -u

ROOT="/home/ubuntu/sp26-optimum-neuron/examples/training/Wan2.2"
cd "$ROOT"
source ~/trn_workspace/native_venv/bin/activate

mkdir -p logs/overnight docs

LOG="logs/overnight/opt_pass.log"
INVENTORY="docs/kernel_inventory_profiled.md"

GATE_THRESHOLD="1.2"  # speedup multiplier required to integrate

# ---- Helper: extract speedup= from a bench line in a retest log ----
extract_speedup() {
  local logfile="$1"
  local label="$2"
  # Bench line format from _bench.BenchResult.line():
  #   [<label>] shape=... dtype=... ref=Xus kernel=Yus speedup=Zx ...
  grep -E "\[$label\]" "$logfile" 2>/dev/null \
    | grep -oE "speedup=[0-9]+\.[0-9]+x" \
    | head -1 \
    | sed 's/speedup=//;s/x//'
}

# ---- Phase A: parse retest logs ----
echo "=== opt_pass start $(date -u +%FT%TZ) ===" | tee "$LOG"
echo "" | tee -a "$LOG"

declare -A SPEEDUP
declare -A VERDICT  # "INTEGRATE" | "SKIP_LOW_SPEEDUP" | "SKIP_NO_DATA" | "SKIP_FAILED"

for kernel in rmsnorm rope attention; do
  # Most recent retest log for this kernel
  RETEST_LOG="logs/overnight/${kernel}_retest.log"
  if [ ! -f "$RETEST_LOG" ]; then
    VERDICT[$kernel]="SKIP_NO_DATA"
    SPEEDUP[$kernel]="?"
    continue
  fi
  # Was the retest itself successful?
  if grep -qE "^FAILED|^ERROR|^E +RuntimeError|^E +AssertionError" "$RETEST_LOG"; then
    VERDICT[$kernel]="SKIP_FAILED"
    SPEEDUP[$kernel]="?"
    continue
  fi
  # Extract the first speedup line per kernel using kernel-specific labels.
  case "$kernel" in
    rmsnorm)   LABEL_PATTERNS=("single" "sp8_rank") ;;
    rope)      LABEL_PATTERNS=("rope_sp8_rank0_bf16") ;;
    attention) LABEL_PATTERNS=("self_attn_full_5B" "self_attn_sp_post_a2a" "cross_attn") ;;
  esac
  BEST_SPEEDUP=""
  for lbl in "${LABEL_PATTERNS[@]}"; do
    s=$(extract_speedup "$RETEST_LOG" "$lbl")
    if [ -n "$s" ]; then
      if [ -z "$BEST_SPEEDUP" ] || awk "BEGIN{exit !($s > $BEST_SPEEDUP)}"; then
        BEST_SPEEDUP=$s
      fi
    fi
  done
  if [ -z "$BEST_SPEEDUP" ]; then
    VERDICT[$kernel]="SKIP_NO_DATA"
    SPEEDUP[$kernel]="?"
    continue
  fi
  SPEEDUP[$kernel]=$BEST_SPEEDUP
  if awk "BEGIN{exit !($BEST_SPEEDUP >= $GATE_THRESHOLD)}"; then
    VERDICT[$kernel]="INTEGRATE"
  else
    VERDICT[$kernel]="SKIP_LOW_SPEEDUP"
  fi
done

# ---- Phase B: write inventory ----
{
  echo "# NKI Kernel Inventory — opt-pass verdicts"
  echo ""
  echo "Generated $(date -u +%FT%TZ) by scripts/run_opt_pass.sh"
  echo ""
  echo "Gate threshold: speedup ≥ ${GATE_THRESHOLD}× (best across benched shapes)."
  echo ""
  echo "| Kernel | Best Speedup | Verdict | Notes |"
  echo "|---|---|---|---|"
  for k in rmsnorm rope attention; do
    v="${VERDICT[$k]}"
    s="${SPEEDUP[$k]}"
    case "$v" in
      INTEGRATE)         note="enabled in e2e validation (WAN_NKI_${k^^}=1)" ;;
      SKIP_LOW_SPEEDUP)  note="below ${GATE_THRESHOLD}× gate" ;;
      SKIP_FAILED)       note="retest failed; see logs/overnight/${k}_retest.log" ;;
      SKIP_NO_DATA)      note="no bench output found; bench wiring may be missing" ;;
    esac
    echo "| ${k} | ${s}× | ${v} | ${note} |"
  done
} > "$INVENTORY"

cat "$INVENTORY" | tee -a "$LOG"
echo "" | tee -a "$LOG"

# ---- Phase C: build per-kernel env toggles for the e2e run ----
INTEG_ENV=()
INTEG_LIST=()
for k in rmsnorm rope attention; do
  if [ "${VERDICT[$k]}" = "INTEGRATE" ]; then
    case "$k" in
      rmsnorm)   INTEG_ENV+=("WAN_NKI_RMSNORM=1") ;;
      rope)      INTEG_ENV+=("WAN_NKI_ROPE=1") ;;
      attention) INTEG_ENV+=("WAN_NKI_ATTENTION=1") ;;
    esac
    INTEG_LIST+=("$k")
  else
    case "$k" in
      rmsnorm)   INTEG_ENV+=("WAN_NKI_RMSNORM=0") ;;
      rope)      INTEG_ENV+=("WAN_NKI_ROPE=0") ;;
      attention) INTEG_ENV+=("WAN_NKI_ATTENTION=0") ;;
    esac
  fi
done

NUM_INTEGRATIONS=${#INTEG_LIST[@]}
echo "Integrations to validate: ${NUM_INTEGRATIONS} (${INTEG_LIST[*]:-none})" | tee -a "$LOG"
echo "" | tee -a "$LOG"

# ---- Phase D: e2e validation ----
if [ "$NUM_INTEGRATIONS" -eq 0 ]; then
  echo "[ALERT:OPT_PASS_PARTIAL] 0/3 — no kernel cleared the ${GATE_THRESHOLD}× gate; skipping e2e validation"
  exit 0
fi

E2E_TAG="kernel_e2e_$(printf %s_ "${INTEG_LIST[@]}" | sed 's/_$//')"
E2E_LOG="logs/overnight/opt_e2e.log"
E2E_MP4="$ROOT/outputs/sweep_sp8_f81_s8_n1_${E2E_TAG}.mp4"
E2E_DEDICATED_LOG="$ROOT/logs/sweep_sp8_f81_s8_n1_${E2E_TAG}.log"

echo "--- e2e validation: SP=8 f=81 s=8 n=1, ${INTEG_LIST[*]} ON ---" | tee -a "$LOG"
echo "tag=${E2E_TAG} log=${E2E_DEDICATED_LOG}" | tee -a "$LOG"

T0=$(date +%s)
env "${INTEG_ENV[@]}" WAN_USE_NKI_KERNELS=1 \
  torchrun --nproc_per_node 8 generate_rolling.py \
    --device neuron --ulysses_size 8 \
    --ckpt_dir ckpts/Wan2.2-TI2V-5B \
    --image examples/i2v_input.JPG \
    --prompt "A cinematic shot of waves on a rocky coast at sunset" \
    --n_chunks 1 --chunk_frame_num 81 --sample_steps 8 --seed 42 \
    --t5_cpu --verbose --skip_decode \
    --save_file "$E2E_MP4" \
    > "$E2E_DEDICATED_LOG" 2>&1
EXIT=$?
T1=$(date +%s)
DUR=$((T1 - T0))

# Capture last|x| if present (rank-0 prints it via verbose logging)
LAST_X=$(grep -oE "last\|x\| ?= ?[0-9]+\.[0-9]+" "$E2E_DEDICATED_LOG" | tail -1 | sed -E 's/.*= *//')

# Append to canonical sweep CSV
SUMMARY="$ROOT/outputs/sweep_summary.csv"
if [ "$EXIT" -eq 0 ] && grep -q "\[skip_decode\] denoising done" "$E2E_DEDICATED_LOG"; then
  STATUS="ok_no_decode"
elif grep -qE "OutOfMemory|out of memory|OOM|HBM_OOM|dmem_alloc_internal|Failed to allocate aligned|MLA DRAM|nrt_tensor_allocate.*failed" "$E2E_DEDICATED_LOG"; then
  STATUS="oom"
else
  STATUS="error_${EXIT}"
fi
echo "${E2E_TAG},8,81,8,1,${DUR},${STATUS},${E2E_MP4}" >> "$SUMMARY"

echo "" | tee -a "$LOG"
echo "=== opt_pass e2e summary $(date -u +%FT%TZ) ===" | tee -a "$LOG"
echo "  exit=${EXIT} duration=${DUR}s status=${STATUS} last|x|=${LAST_X:-?}" | tee -a "$LOG"

# ---- Phase E: emit final alert ----
if [ "$EXIT" -eq 0 ] && [ "$STATUS" = "ok_no_decode" ]; then
  # Reference last|x| at SP=8 f=81 s=8 n=1 (no NKI patches) is 0.7043 from PROGRESS.md.
  # Tolerance: ±0.05 absolute (bf16 numerical drift across kernels is acceptable).
  if [ -n "$LAST_X" ] && awk "BEGIN{exit !(($LAST_X > 0.65) && ($LAST_X < 0.75))}"; then
    echo "[ALERT:OPT_PASS_OK] ${NUM_INTEGRATIONS}/3 kernels integrated (${INTEG_LIST[*]}); e2e last|x|=${LAST_X} within ±0.05 of baseline 0.7043"
  else
    echo "[ALERT:OPT_PASS_PARTIAL] ${NUM_INTEGRATIONS}/3 integrated, e2e ran but last|x|=${LAST_X:-?} outside tolerance — numerical regression"
  fi
else
  echo "[ALERT:OPT_PASS_PARTIAL] ${NUM_INTEGRATIONS}/3 integrated, e2e failed (status=${STATUS}, exit=${EXIT})"
fi
