"""NKI RMSNorm kernel for Wan2.2 (TI2V-5B / A14B variants).

Mathematical contract (matches `wan.modules.model.WanRMSNorm`):
    y = (x.float() * rsqrt(mean(x.float()^2) + eps)).to(x.dtype) * weight
where the reduction and rsqrt run in fp32, weight is fp32, and the
intermediate `x_norm` is cast back to the input dtype (bf16) BEFORE the
final multiply by `weight`. PyTorch then promotes that bf16*fp32 multiply
back to fp32 internally and returns a tensor whose dtype follows
`x.float() * fp32_weight` -> fp32, but `WanRMSNorm.forward` returns this
fp32 tensor as-is (the parent block in `model.py` casts it back to bf16
when needed). We mirror this exactly: kernel returns fp32.

Shape contract:
    x      : [B, L, H]  bf16     (B*L flattened to a single token axis)
    weight : [H]        fp32
    eps    : float (1e-6 in production; PyTorch class default is 1e-5)
    output : [B, L, H]  fp32     (matches WanRMSNorm.forward dtype)

Why a custom kernel instead of `nkilib.core.subkernels.rmsnorm_tkg.rmsnorm_tkg`?
    The production library kernel produces a transposed `[128, B*L, H//128]`
    output tailored for a downstream sharded matmul. Wan's RMSNorm flows
    directly into the next Linear in eager PyTorch, so we want a plain
    `[B, L, H]` output. We mirror the same compute layout (P-partition over
    tokens, F-dim = H, fp32 internals) but keep the output dense and
    transpose-free.

Tiling strategy
---------------
The token dimension N = B*L is partitioned across the 128-partition SBUF axis.
N is split into tiles of size <=128 (TILE_N = nl.tile_size.pmax = 128). For
TI2V-5B on a 17-frame chunk:
    - single device:  N=4400 -> 35 tiles (last tile = 80)
    - SP=8 per-rank:  N= 550 ->  5 tiles (last tile = 38)

The hidden dim H=3072 sits entirely in the free dimension — fp32 working set
is 128 * 3072 * 4B = 1.5 MiB, well under the 24 MiB SBUF budget — so we do
NOT tile along H. This keeps the reduction (sum over the H axis) a single
`nisa.tensor_reduce` per tile.

Beta-2 / NKI 0.3.0 APIs used
----------------------------
- @nki.jit                                 : kernel entry decorator
- nl.ndarray(...)                          : SBUF / shared_hbm allocation
- nisa.dma_copy(dst, src)                  : HBM <-> SBUF transfers
- nisa.tensor_copy                         : bf16 -> fp32 promotion in SBUF
- nisa.activation(op=nl.square, ...)       : fp32 square via activation engine
- nisa.tensor_reduce(op=nl.add, axis=(1,)) : sum across H, fp32
- nisa.activation(op=nl.rsqrt, scale=1/H, bias=eps)  : fused (mean+eps) -> rsqrt
                                              using the ScalarE pre-activation
                                              scale+bias to fold mean and eps.
- nisa.tensor_tensor(op=nl.multiply)       : x * inv_rms (broadcast over F)
- nisa.tensor_scalar(op0=nl.multiply, operand0=weight_sb)
                                            : multiply per-H weight broadcast
                                              over partition axis.

Note: weight is fp32 [H], loaded once into a [1, H] SBUF tile and broadcast
across the partition axis using `nisa.tensor_scalar` with a vector operand.
This avoids materializing a full [128, H] gamma tile.
"""

from __future__ import annotations

import torch

import nki
import nki.isa as nisa
import nki.language as nl


# ----- inline helpers (avoid runtime nkilib install dependency) ---------
def kernel_assert(cond, msg):
    """Structured kernel-side assertion (matches nkilib pattern)."""
    if not cond:
        raise AssertionError(f"[NCC_INKI016] Kernel validation exception: {msg}")


def div_ceil(n: int, d: int) -> int:
    return (n + d - 1) // d


# ----- the NKI kernel ---------------------------------------------------
@nki.jit
def _rmsnorm_kernel(x_hbm, weight_hbm, eps_hbm):
    """RMSNorm forward kernel.

    Args:
        x_hbm:      [N, H]  bf16    where N = B*L, H % 128 == 0 (Wan H=3072)
        weight_hbm: [1, H]  fp32
        eps_hbm:    [1, 1]  fp32    epsilon (broadcast as activation bias)

    Returns:
        y_hbm: [N, H] fp32 (intentional: matches WanRMSNorm.forward dtype)
    """
    N, H = x_hbm.shape
    TILE_N = nl.tile_size.pmax  # 128

    # Output is fp32 to mirror PyTorch's `(bf16 * fp32_weight) -> fp32`
    # promotion behavior in WanRMSNorm.forward.
    y_hbm = nl.ndarray((N, H), dtype=nl.float32, buffer=nl.shared_hbm)

    # Load weight ONCE into SBUF as [1, H] fp32. tensor_scalar will
    # broadcast it across the partition axis as a vector operand.
    weight_sb = nl.ndarray((1, H), dtype=nl.float32, buffer=nl.sbuf)
    nisa.dma_copy(dst=weight_sb, src=weight_hbm[0:1, 0:H])

    # Load eps as a [1, 1] fp32 buffer to use as activation bias.
    eps_sb = nl.ndarray((1, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.dma_copy(dst=eps_sb, src=eps_hbm[0:1, 0:1])

    num_tiles = div_ceil(N, TILE_N)
    inv_h = 1.0 / float(H)

    for tile_idx in nl.affine_range(num_tiles):
        n_start = tile_idx * TILE_N
        n_end = min(n_start + TILE_N, N)
        n_sz = n_end - n_start

        # Load x tile in source dtype, then promote to fp32 in SBUF.
        x_tile_in = nl.ndarray((n_sz, H), dtype=x_hbm.dtype, buffer=nl.sbuf)
        nisa.dma_copy(dst=x_tile_in, src=x_hbm[n_start:n_end, 0:H])

        x_tile = nl.ndarray((n_sz, H), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_copy(dst=x_tile, src=x_tile_in)

        # x^2 in fp32
        x_sq = nl.ndarray((n_sz, H), dtype=nl.float32, buffer=nl.sbuf)
        nisa.activation(dst=x_sq, data=x_tile, op=nl.square)

        # Reduce along H -> [n_sz, 1] = sum(x^2)
        sum_sq = nl.ndarray((n_sz, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_reduce(dst=sum_sq, data=x_sq, op=nl.add, axis=(1,))

        # inv_rms = rsqrt(sum_sq * (1/H) + eps)
        # Fused via activation engine's ScalarE pre-activation scale+bias.
        inv_rms = nl.ndarray((n_sz, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.activation(
            dst=inv_rms,
            data=sum_sq,
            op=nl.rsqrt,
            scale=inv_h,
            bias=eps_sb,
        )

        # x_norm_fp32 = x_fp32 * inv_rms (inv_rms broadcast across F-axis)
        x_norm_fp32 = nl.ndarray((n_sz, H), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_tensor(dst=x_norm_fp32, data1=x_tile, data2=inv_rms, op=nl.multiply)

        # WanRMSNorm casts back to input dtype here, then multiplies by
        # weight. Mirror that round-trip so the numeric profile matches.
        x_norm_lo = nl.ndarray((n_sz, H), dtype=x_hbm.dtype, buffer=nl.sbuf)
        nisa.tensor_copy(dst=x_norm_lo, src=x_norm_fp32)
        x_norm_back = nl.ndarray((n_sz, H), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_copy(dst=x_norm_back, src=x_norm_lo)

        # y = x_norm_fp32 * weight (weight broadcast over partition axis).
        y_fp32 = nl.ndarray((n_sz, H), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_scalar(
            dst=y_fp32,
            data=x_norm_back,
            op0=nl.multiply,
            operand0=weight_sb,
        )

        # Store fp32 result.
        nisa.dma_copy(dst=y_hbm[n_start:n_end, 0:H], src=y_fp32)

    return y_hbm


# ----- Python entrypoint ------------------------------------------------
def _is_neuron_tensor(t: torch.Tensor) -> bool:
    """Return True iff `t` lives on a Neuron / XLA device."""
    if t is None:
        return False
    return t.device.type in ("xla", "neuron")


def _rmsnorm_torch_ref(
    x: torch.Tensor, weight: torch.Tensor, eps: float
) -> torch.Tensor:
    """PyTorch reference matching `WanRMSNorm.forward` exactly:
        ((x.float() * rsqrt(mean(x.float()^2) + eps)).type_as(x)) * weight
    Note that the result dtype is whatever `bf16 * fp32_weight` yields,
    which is fp32 in PyTorch's standard promotion rules.
    """
    in_dtype = x.dtype
    xf = x.float()
    rms = xf.pow(2).mean(dim=-1, keepdim=True).add(eps).rsqrt()
    return (xf * rms).to(in_dtype) * weight


def nki_rmsnorm(
    x: torch.Tensor,
    weight: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Drop-in NKI RMSNorm forward.

    Args:
        x:      [B, L, H]  any float dtype (bf16 in production)
        weight: [H]        fp32 (production); other floats are accepted
        eps:    float      1e-6 in production; class default is 1e-5

    Returns:
        y: [B, L, H] fp32 (matches WanRMSNorm.forward output dtype, which
           PyTorch promotes from `bf16 * fp32_weight` -> fp32). The caller
           normally casts back to bf16 with the next op (cf. WanSelfAttention).

    Notes:
        - Reduction and rsqrt run in fp32 internally.
        - On non-Neuron tensors (CPU), this falls back to the PyTorch
          reference, so the function is safely importable in test envs.
        - Forward-only (no_grad-safe). Wan's modules call this inside
          `with torch.no_grad()` paths during inference; for training,
          the parent block currently uses the PyTorch RMSNorm.
    """
    if x.dim() != 3:
        raise ValueError(f"nki_rmsnorm expects [B, L, H], got {tuple(x.shape)}")
    B, L, H = x.shape
    if H % 128 != 0:
        raise ValueError(f"nki_rmsnorm requires H % 128 == 0, got H={H}")
    if weight.shape != (H,):
        raise ValueError(
            f"nki_rmsnorm weight must be [H={H}], got {tuple(weight.shape)}"
        )

    # CPU / non-neuron fallback uses the PyTorch reference verbatim.
    if not _is_neuron_tensor(x):
        return _rmsnorm_torch_ref(x, weight, eps)

    # Flatten B*L for the kernel; reshape back at the end.
    x_flat = x.reshape(B * L, H).contiguous()
    w_row = weight.to(torch.float32).reshape(1, H).contiguous()
    eps_buf = torch.tensor(
        [[float(eps)]], dtype=torch.float32, device=x.device
    )

    y_flat = _rmsnorm_kernel(x_flat, w_row, eps_buf)
    return y_flat.reshape(B, L, H)
