# NKI Kernel Inventory — opt-pass verdicts

Generated 2026-05-06T10:05:46Z by scripts/run_opt_pass.sh

Gate threshold: speedup ≥ 1.2× (best across benched shapes).

| Kernel | Best Speedup | Verdict | Notes |
|---|---|---|---|
| rmsnorm | ?× | SKIP_FAILED | retest failed; see logs/overnight/rmsnorm_retest.log |
| rope | 1.97× | INTEGRATE | enabled in e2e validation (WAN_NKI_ROPE=1) |
| attention | 3.27× | INTEGRATE | enabled in e2e validation (WAN_NKI_ATTENTION=1) |
| layernorm | ?× | SKIP_FAILED | retest failed; see logs/overnight/layernorm_retest.log |

## Combined E2E (SP=8 f=81 s=8 n=1, --skip_decode)

Integrations ON: rope attention
Result: status=ok_no_decode duration=200s last|x|=0.7657
Tolerance: ±0.05 absolute around baseline 0.7043 (no-NKI ground truth).
