# Wan2.2 NKI Kernel Port Inventory

Phase: research only. No kernels written, no benchmarks run. Numbers below assume TI2V-5B (dim=3072, num_heads=24, head_dim=128, num_layers=30 — confirmed in `wan/configs/wan_ti2v_5B.py:21-25`), 17-frame chunk at 704x1280, `seq_len=4400`, `B=1`, `text_len=512`, SP=8 → per-rank `s=550`.

## 1. Forward-pass hot path inventory

| # | Op | File:Line | Signature / What it does | Shapes (TI2V-5B, bf16 unless noted) | @trace | Existing NKI? |
|---|---|---|---|---|---|---|
| A | `flash_attention` (SDPA fallback) | `wan/modules/attention.py:225` | `F.scaled_dot_product_attention(q,k,v)` — full softmax(QKᵀ/√d)·V | Q,K,V: `[1,24,4400,128]` (self-attn); cross: `[1,24,4400,128]`×`[1,24,512,128]` | indirect via `self_attn`/`cross_attn` | YES — `nkilib.core.attention_cte` (head_dim≤128 OK), `neuronx_distributed.kernels.flash_attn.nki_flash_attn_func` |
| B | `WanRMSNorm` | `model.py:202-219` | `x·rsqrt(mean(x²)+eps)·γ`, fp32 mean | `[1,4400,3072]` per layer; called 2×/block (norm_q, norm_k) → 60×/forward | NO | YES — `nkilib.core.subkernels.rmsnorm_tkg.rmsnorm_tkg` |
| C | `WanLayerNorm` | `model.py:222-232` | `nn.LayerNorm(dim,eps,elementwise_affine=False)` | `[1,4400,3072]`; norm1+norm2/block, +norm3 if cross_attn_norm → 90×/forward | NO | Partial — `nkilib.core.subkernels.layernorm_tkg` |
| D | `rope_apply` (single-device) | `model.py:97-197` | 3D RoPE: build cos/sin from f/h/w freqs, complex-multiply, masked interleave | x: `[1,4400,24,128]` (fp32 compute, bf16 in/out); freqs: `[1024,64]` | YES (`rope_apply`, with sub-regions) | NO direct equivalent; could fuse cos/sin + multiply-add |
| E | `rope_apply` (SP variant) | `distributed/sequence_parallel.py:23-93` | Same as D but slices angles to per-rank `s=550` | x: `[1,550,24,128]` | NO | NO |
| F | `distributed_attention` | `distributed/ulysses.py:9-47` | all-to-all → attention → all-to-all (Ulysses) | per-rank `[1,550,24,128]` → gathered `[1,4400,3,128]` | NO | Inner attention can use NKI; all-to-all stays in collectives |
| G | QKV/output projections | `model.py:254-257` (Linear) called in `WanSelfAttention.forward:281-297` | 4× `Linear(3072→3072)` | input `[1,4400,3072]` | wrapped via `self_attn` trace | Native matmul; NKI fused QKV may help but lower priority |
| H | FFN GELU(tanh) | `model.py:356-358` (used in `WanAttentionBlock.forward`) | `Linear(3072→14336)→GELU(tanh)→Linear(14336→3072)` | `[1,4400,3072]` ↔ `[1,4400,14336]` | via `wan_attn_block` | NO direct fused kernel; could fuse Linear+GELU |
| I | `sinusoidal_embedding_1d` | `model.py:15-27` | fp64 outer product + cos/sin/concat | t: `[B*seq_len]=4400`; out `[4400,256]` | NO | Tiny; not a port target |
| J | Modulation broadcast/affine | `model.py:382-397` | `x*(1+e[i])+e[j]`, `chunk(6,dim=2)` | `[1,4400,3072]` × scalar-per-token vectors | wrapped in `wan_attn_block` | Trivial; fuse with adjacent norm if porting LN |

## 2. Confirmed NKI APIs (from `/neuron-nki-docs`)

- **`nkilib.core.attention_cte.attention_cte`** (`references/reference/library/attention-cte.md`):
  - Signature: `attention_cte(q, k, v, scale=1.0, causal_mask=True, sliding_window=None, tp_q=True, tp_k=False, tp_out=False, cache_softmax=False, softmax_dtype=nl.float32, ...)`
  - Shapes: `q:[B, S_q, d]`, `k:[B_kv, S_kv, d]`, `v:[B_kv, S_kv, d]`. **Constraint: `d ≤ 128`** — Wan head_dim=128 fits exactly.
  - Optimized for prefill / long sequences (>256). LNC2 sharding, flash-attention pipelining for KV>10K.
  - Has `cache_softmax` to return max & sum-recip for backward — usable for training path.
- **`nkilib.core.subkernels.rmsnorm_tkg.rmsnorm_tkg`** (`references/nkilib/core/subkernels/rmsnorm_tkg.py`): production RMSNorm. Reference (`rmsnorm_torch.py`) matches Wan's formula.
- **`nkilib.core.subkernels.layernorm_tkg`**: LayerNorm equivalent.
- **`torch.compile(backend="neuron")`**: NKI documentation contains no evidence of an automatic Torch→NKI lowering backend. The PyTorch Native Neuron path is eager-dispatch; SDPA falls through to the device's default kernel (not NKI). All five candidates above need hand-written / library NKI kernels — automatic generation is **not** an option today.

## 3. Top-3 ranked candidates (compute × frequency × kernel-readiness)

### #1 — Self/cross attention via `attention_cte` (item A)
- **Leverage**: Per layer per call: ~`2·L²·d·H` FLOPs for QKᵀ + softmax·V. Self-attn (`L=4400`): ~5.3 GFLOPs/layer. Cross-attn (`L_q=4400, L_kv=512`): ~0.6 GFLOPs/layer. ×30 layers ≈ 178 GFLOPs/forward in attention alone, plus the materialized `[4400,4400]` softmax matrix that dominates HBM bandwidth on the SDPA fallback.
- **Status**: SDPA fallback at `attention.py:225` is the single biggest hotspot — it produces a 73 MB attention matrix (4400² × bf16) that doesn't fit in SBUF, so it spills heavily.
- **What's needed**:
  - Sample tensors: Q/K/V `[1, 24, 4400, 128]` bf16 self-attn, plus `[1, 24, 512, 128]` K/V for cross-attn.
  - API: wrap `attention_cte` in a `torch.autograd.Function` (sketch already at `attention.py:13-50`). `causal_mask=False` (Wan is non-causal video DiT). Use `cache_softmax=True` to enable backward.
  - Layout: input is `[B, L, H, D]`; need transpose to `[B*H, L, D]` (kernel B-dim absorbs heads) before invocation. After SP all-to-all the head dim shrinks to 3 — kernel still applies per-head.
  - Complexity: **medium-high**. Forward + custom backward (recompute QKᵀ, softmax via cached stats, ∂V/∂K/∂Q). The kernel exists; the wrapping is the work.

### #2 — RMSNorm via `rmsnorm_tkg` (item B)
- **Leverage**: 60 calls/forward × `[1, 4400, 3072]` bf16 (with fp32 reduction). Memory-bound; today the `.float()` upcast doubles HBM traffic. Modest absolute FLOPs but very high call frequency means launch overhead matters.
- **Status**: Drop-in library kernel exists. Wan's formula matches exactly: `x · rsqrt(mean(x²) + eps) · γ`.
- **What's needed**:
  - Sample tensors: x `[1, 4400, 3072]` bf16, γ `[3072]` fp32, eps=1e-6. Match Wan's fp32-internal/bf16-IO behavior.
  - API: call `rmsnorm_tkg(input, gamma, output, eps=1e-6)` directly. Wrap in `torch.autograd.Function` for training.
  - Complexity: **low**. Library does the work; main effort is the autograd backward (well-known formula).

### #3 — Fused RoPE (item D / E)
- **Leverage**: 2 calls/block (Q, K) × 30 blocks = 60 RoPE calls/forward on `[1, 4400, 24, 128]` fp32. Today's implementation has 6 ops per token (cos, sin, two muls, two adds), plus `repeat_interleave` and a Python `for i in grid_sizes.tolist()` loop that breaks compilation cleanliness. The `.float()` cast at `model.py:110` doubles activation memory transiently.
- **Status**: No off-the-shelf NKI 3D-RoPE kernel. Custom kernel needed. Already wrapped in granular `region(...)` profiling markers (`model.py:98-166`) — easy to baseline.
- **What's needed**:
  - Sample tensors: x `[1, 4400, 24, 128]` bf16, freqs `[1024, 64]` fp32, grid `(F=17, H=22, W=40)` → seq=14960 patches but padded to 4400 valid tokens.
  - API: write a kernel that loads precomputed cos/sin tables (computed once on host or in a setup kernel), then performs the masked interleave-multiply-add along the head_dim=128 axis, partitioned across the 24 heads. Use `nl.affine_range` over the seq tile dimension; head_dim=128 fits exactly in P. Could fuse with the upstream RMSNorm output (Q/K both come from a normalized projection).
  - Complexity: **medium**. Standalone kernel is small; wiring the 3D angle-table construction (currently host-side per-batch) is the larger design question — likely keep host-side and pass cos/sin as kernel inputs.

## 4. Lower-priority candidates noted but deferred

- **Item C (LayerNorm)**: comparable to RMSNorm but Wan uses `elementwise_affine=False` for norm1/norm2 (no learned γ/β), so it's purely the variance/mean compute. Worth porting after RMSNorm to keep the norm path consistent.
- **Item H (FFN GELU-tanh)**: `Linear(3072→14336)→GELU(tanh)→Linear(14336→3072)` — biggest single matmul pair in the model. Native Neuron matmuls are likely already strong; revisit only after profiling shows it as the new bottleneck post-attention port.
- **Item G (fused QKV)**: combine three `Linear(3072→3072)` into one `Linear(3072→9216)`. Pure PyTorch refactor first; NKI port only if profiling shows a gap.

## 5. Open questions to resolve before authoring

1. Confirm head_dim=128 holds across **all** Wan2.2 variants we care about (T2V/I2V-A14B `dim=5120, num_heads=40` → head_dim=128 too — verify in `wan_t2v_A14B.py`).
2. After SP all-to-all, attention runs with `B=1, H=3, L=4400, D=128`. `attention_cte` requires `B_kv ≤ B`; confirm it tolerates B=3 (heads-as-batch trick) at this small head count.
3. Backward path: training loop calls these. Either use `cache_softmax=True` + manual backward, or write a paired `attention_cte_bwd` kernel.
4. Decide whether RoPE precompute (cos/sin tables) is per-step host-cached or refreshed per chunk — affects kernel signature.

## Sources

- `wan/modules/model.py:97,202,222,236,301,327` — RoPE, RMSNorm, LayerNorm, attention classes.
- `wan/modules/attention.py:74,183,225` — flash/attention wrappers, SDPA fallback.
- `wan/distributed/sequence_parallel.py:23,96,179` — SP rope_apply, sp_dit_forward, sp_attn_forward.
- `wan/distributed/ulysses.py:9` — distributed_attention.
- `wan/configs/wan_ti2v_5B.py:21-25` — dim/num_heads/num_layers.
- `/home/ubuntu/.claude/skills/neuron-nki-docs/references/reference/library/attention-cte.md` — `attention_cte` API.
- `/home/ubuntu/.claude/skills/neuron-nki-writing/references/nkilib/core/subkernels/rmsnorm_tkg.py` — RMSNorm kernel.
