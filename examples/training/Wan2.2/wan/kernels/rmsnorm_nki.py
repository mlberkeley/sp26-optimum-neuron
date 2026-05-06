"""NKI RMSNorm kernel for Wan2.2 (TI2V-5B / A14B variants).

Mathematical contract (matches `wan.modules.model.WanRMSNorm`):
    y = (x.float() * rsqrt(mean(x.float()^2) + eps)).to(x.dtype) * weight
The kernel handles only the normalize half (reduction + rsqrt + multiply
by inv_rms, then cast back to input dtype). The final `* weight` happens
in torch on the kernel output — this is required because
`nisa.tensor_scalar` requires `operand0.free == 1`, so a `[1, H]` weight
broadcast across the partition axis is not legal (MLIR verifier rejects:
"'operand0' free dimensions total elements must be 1, got 3072"). The
alternative (broadcasting weight to `[128, H]` with `nl.broadcast_to`)
costs an extra SBUF buffer per tile and a redundant copy, so doing the
multiply in torch — which is fused with the next op by the eager runtime
anyway — is cheaper.

Shape contract:
    x      : [B, L, H]  bf16     (B*L flattened to a single token axis)
    weight : [H]        fp32
    eps    : float (1e-6 in production; PyTorch class default is 1e-5)
    output : [B, L, H]  bf16     (kernel output dtype = input dtype;
                                  the python wrapper then multiplies by
                                  weight, producing fp32 by torch
                                  promotion to match WanRMSNorm.forward)

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
                                              `bias` is passed as a Python
                                              float (compile-time constant) —
                                              on NeuronCore-v3+ (trn2/trn3) the
                                              activation engine accepts a scalar
                                              bias directly; on v2 the NKI API
                                              auto-broadcasts the scalar into a
                                              [P, 1] vector before lowering.
                                              An explicit [1, 1] SBUF eps buffer
                                              is rejected by MLIR verification
                                              ("'bias' partition total elements
                                              1 != 'dst' partition total
                                              elements 128").
- nisa.tensor_scalar(op0=nl.multiply, operand0=inv_rms)
                                            : x * inv_rms with the [P, 1]
                                              inv_rms operand broadcast across
                                              the free dim. `tensor_tensor`
                                              cannot do this broadcast — the
                                              MLIR verifier rejects
                                              lhs.free != rhs.free.
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
def _rmsnorm_kernel(x_hbm, eps):
    """RMSNorm forward kernel — normalize half only (no weight multiply).

    Args:
        x_hbm:  [N, H]  any float dtype (bf16 in production); H % 128 == 0.
                N = B*L flattened token axis.
        eps:    Python float — compile-time constant used as the
                `nisa.activation` bias.

    Returns:
        y_hbm: [N, H] same dtype as x_hbm. The python wrapper handles the
        `* weight` multiply afterwards (see module docstring for why).
    """
    N, H = x_hbm.shape
    TILE_N = nl.tile_size.pmax  # 128

    # Output dtype = input dtype. The wrapper multiplies by the fp32
    # weight in torch, which promotes to fp32 to match WanRMSNorm.forward.
    y_hbm = nl.ndarray((N, H), dtype=x_hbm.dtype, buffer=nl.shared_hbm)

    num_tiles = div_ceil(N, TILE_N)
    inv_h = 1.0 / float(H)
    eps_f = float(eps)

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

        # inv_rms = rsqrt(sum_sq * (1/H) + eps), fused via ScalarE.
        inv_rms = nl.ndarray((n_sz, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.activation(
            dst=inv_rms,
            data=sum_sq,
            op=nl.rsqrt,
            scale=inv_h,
            bias=eps_f,
        )

        # x_norm_fp32 = x_fp32 * inv_rms (inv_rms is [P, 1], broadcast
        # across F-axis via tensor_scalar — operand0.free=1 is the legal
        # case; the H-vector broadcast case fails MLIR verify).
        x_norm_fp32 = nl.ndarray((n_sz, H), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_scalar(
            dst=x_norm_fp32,
            data=x_tile,
            op0=nl.multiply,
            operand0=inv_rms,
        )

        # Cast back to input dtype to match WanRMSNorm's `(...).type_as(x)`
        # round-trip before the weight multiply.
        y_lo = nl.ndarray((n_sz, H), dtype=x_hbm.dtype, buffer=nl.sbuf)
        nisa.tensor_copy(dst=y_lo, src=x_norm_fp32)
        nisa.dma_copy(dst=y_hbm[n_start:n_end, 0:H], src=y_lo)

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

    # Kernel returns x_norm in input dtype (bf16). The final `* weight`
    # is done here in torch — see module docstring on why we can't fuse
    # the weight multiply inside the kernel (`nisa.tensor_scalar` rejects
    # operand0.free=H>1). Torch promotes `bf16 * fp32_weight -> fp32`,
    # matching WanRMSNorm.forward's output dtype.
    x_norm = _rmsnorm_kernel(x_flat, float(eps)).reshape(B, L, H)
    return x_norm * weight.to(torch.float32)
