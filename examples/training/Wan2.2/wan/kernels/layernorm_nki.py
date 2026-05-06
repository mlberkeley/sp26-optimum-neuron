"""NKI LayerNorm kernel for Wan2.2 (TI2V-5B / A14B variants).

Mathematical contract (matches `torch.nn.LayerNorm` over the last dim,
which is what `wan.modules.model.WanLayerNorm` reduces to):

    mu  = mean(x, dim=-1)                   # [B, L, 1]
    var = mean((x - mu)**2, dim=-1)          # [B, L, 1]
    x_hat = (x - mu) / sqrt(var + eps)       # [B, L, H]
    y = x_hat * gamma + beta                 # if elementwise_affine
    y = x_hat                                # otherwise

Reductions and rsqrt run in fp32 to match PyTorch's internal precision for
LayerNorm with bf16 inputs. Output is cast back to the input dtype, mirroring
`nn.LayerNorm`'s contract (which returns same dtype as input regardless of
whether weight/bias are fp32).

Shape contract:
    x      : [B, L, H]  bf16/fp32     (B*L flattened to a single token axis)
    weight : [H]        fp32          (optional — None when elementwise_affine=False)
    bias   : [H]        fp32          (optional — None when elementwise_affine=False)
    eps    : float                    (Wan default 1e-6 in DiT block; class default 1e-5)
    output : [B, L, H]  same dtype as x  (NOT promoted to fp32, unlike RMSNorm)

Two kernel variants
-------------------
- `_layernorm_no_affine_kernel(x, eps)`           — used when elementwise_affine=False
                                                    (norm1, norm2, head.norm)
- `_layernorm_affine_kernel(x, weight, bias, eps)` — used when elementwise_affine=True
                                                     (norm3 cross_attn norm)

Splitting avoids passing dummy weight/bias tensors and the associated extra
multiply/add when affine isn't used (which is 4 of the 5 DiT-block calls).

Tiling strategy
---------------
Mirrors `rmsnorm_nki.py`:

The token dimension N = B*L is partitioned across the 128-partition SBUF axis.
N is split into tiles of size <=128 (TILE_N = nl.tile_size.pmax = 128). For
TI2V-5B on a 17-frame chunk:
    - single device:  N=4400 -> 35 tiles (last tile = 80)
    - SP=8 per-rank:  N= 550 ->  5 tiles (last tile = 38)

The hidden dim H=3072 sits entirely in the free dimension — fp32 working set
is 128 * 3072 * 4B = 1.5 MiB per tensor, well under the 24 MiB SBUF budget —
so we do NOT tile along H. Each pass over H is a single `tensor_reduce`.

Beta-2 / NKI 0.3.0 APIs used
----------------------------
- @nki.jit                                 : kernel entry decorator
- nl.ndarray(...)                          : SBUF / shared_hbm allocation
- nisa.dma_copy(dst, src)                  : HBM <-> SBUF transfers
- nisa.tensor_copy                         : dtype promotion / demotion
- nisa.tensor_reduce(op=nl.add, axis=(1,)) : sum across H, fp32
- nisa.tensor_scalar(op0=nl.multiply, ...) : scale by 1/H to compute mean,
                                              broadcast weight across partition axis
- nisa.tensor_scalar(op0=nl.subtract,
                     operand0=mean)        : x - mean. Must use tensor_scalar
                                              (not tensor_tensor) so the [P, 1]
                                              mean broadcasts across the free
                                              axis — tensor_tensor requires
                                              lhs.free == rhs.free and is
                                              rejected by the MLIR verifier
                                              with "'dst' free total elements
                                              H != 'rhs' free total elements 1".
- nisa.activation(op=nl.square, ...)       : (x - mean)^2 via activation engine
- nisa.activation(op=nl.rsqrt, scale=1/H, bias=eps_f)
                                            : fused (mean(diff^2) + eps) -> rsqrt
                                              `bias` MUST be a Python float
                                              (compile-time constant). On
                                              NeuronCore-v3+ the activation
                                              engine consumes scalar bias
                                              directly; on v2 the NKI API
                                              auto-broadcasts to a [P, 1]
                                              vector. An explicit [1, 1] SBUF
                                              eps buffer is rejected by MLIR
                                              verification ("'bias' partition
                                              total elements 1 != 'dst'
                                              partition total elements 128").
                                              See commit 4cd071fb (RMSNorm fix).
- nisa.tensor_scalar(op0=nl.multiply,
                     operand0=inv_std)     : (x - mu) * inv_std with [P, 1]
                                              inv_std broadcast across the
                                              free axis (same constraint as
                                              the subtract above).
- nisa.tensor_scalar(op0=nl.multiply,
                     operand0=weight_sb,
                     op1=nl.add,
                     operand1=bias_sb)     : fused gamma * x + beta in one ScalarE op,
                                              with weight/bias broadcast across the
                                              partition axis as vector operands.

Compile gotchas avoided
-----------------------
1. Scalar `eps` is passed as Python float to nisa.activation(bias=...).
   Passing it as a [1,1] SBUF tile triggers MLIR partition-dim mismatch.
   (See commit 4cd071fb.)
2. No `raise` statements inside @nki.jit bodies — host-side validation only.
   (See commit 45439ca4.)
3. Two kernel variants instead of conditional Python if/else inside the
   kernel — the NKI parser is strict about compile-time control flow,
   and tracing is cleaner when affine vs no-affine choose different
   compiled NEFFs.
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


# ----- the NKI kernels --------------------------------------------------
@nki.jit
def _layernorm_no_affine_kernel(x_hbm, eps):
    """LayerNorm forward without learnable affine (elementwise_affine=False).

    Args:
        x_hbm: [N, H]  any float dtype (bf16 in production); H % 128 == 0.
               N = B*L (flattened token axis).
        eps:   Python float — compile-time constant, used as nisa.activation
               bias for the rsqrt step.

    Returns:
        y_hbm: [N, H] same dtype as x_hbm (matches torch.nn.LayerNorm
        contract — output dtype equals input dtype).
    """
    N, H = x_hbm.shape
    TILE_N = nl.tile_size.pmax  # 128

    y_hbm = nl.ndarray((N, H), dtype=x_hbm.dtype, buffer=nl.shared_hbm)

    num_tiles = div_ceil(N, TILE_N)
    inv_h = 1.0 / float(H)
    eps_f = float(eps)

    for tile_idx in nl.affine_range(num_tiles):
        n_start = tile_idx * TILE_N
        n_end = min(n_start + TILE_N, N)
        n_sz = n_end - n_start

        # Load x tile in source dtype, promote to fp32.
        x_tile_in = nl.ndarray((n_sz, H), dtype=x_hbm.dtype, buffer=nl.sbuf)
        nisa.dma_copy(dst=x_tile_in, src=x_hbm[n_start:n_end, 0:H])

        x_f32 = nl.ndarray((n_sz, H), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_copy(dst=x_f32, src=x_tile_in)

        # ---- Pass 1: compute mean. ---------------------------------------
        # sum over H, then multiply by 1/H to get mean.
        sum_x = nl.ndarray((n_sz, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_reduce(dst=sum_x, data=x_f32, op=nl.add, axis=(1,))

        mean = nl.ndarray((n_sz, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_scalar(
            dst=mean, data=sum_x, op0=nl.multiply, operand0=inv_h
        )

        # ---- Pass 2: x - mean, then (x - mean)^2 reduction. --------------
        # mean is [n_sz, 1]. `nisa.tensor_tensor` requires lhs and rhs to
        # share the same free dimension; passing rhs=[P, 1] vs dst=[P, H]
        # is rejected by the MLIR verifier with "'dst' free total elements
        # H != 'rhs' free total elements 1". `nisa.tensor_scalar` accepts
        # a `(P, 1)` operand and broadcasts across the free axis natively.
        # Computes `data - operand0` = `x_f32 - mean` (default reverse0=False).
        diff = nl.ndarray((n_sz, H), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_scalar(
            dst=diff,
            data=x_f32,
            op0=nl.subtract,
            operand0=mean,
        )

        diff_sq = nl.ndarray((n_sz, H), dtype=nl.float32, buffer=nl.sbuf)
        nisa.activation(dst=diff_sq, data=diff, op=nl.square)

        sum_sq = nl.ndarray((n_sz, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_reduce(dst=sum_sq, data=diff_sq, op=nl.add, axis=(1,))

        # inv_std = rsqrt(sum_sq * (1/H) + eps).
        # Fused via the activation engine's ScalarE pre-activation scale+bias.
        # `bias=eps_f` MUST be a Python scalar; an [1, 1] SBUF tile fails
        # MLIR verification (see commit 4cd071fb).
        inv_std = nl.ndarray((n_sz, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.activation(
            dst=inv_std,
            data=sum_sq,
            op=nl.rsqrt,
            scale=inv_h,
            bias=eps_f,
        )

        # x_hat = diff * inv_std (inv_std broadcast across F-axis). Same
        # `tensor_tensor`-vs-`tensor_scalar` constraint as the subtract above.
        x_hat = nl.ndarray((n_sz, H), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_scalar(
            dst=x_hat,
            data=diff,
            op0=nl.multiply,
            operand0=inv_std,
        )

        # Cast back to input dtype to match torch.nn.LayerNorm output dtype.
        y_lo = nl.ndarray((n_sz, H), dtype=x_hbm.dtype, buffer=nl.sbuf)
        nisa.tensor_copy(dst=y_lo, src=x_hat)
        nisa.dma_copy(dst=y_hbm[n_start:n_end, 0:H], src=y_lo)

    return y_hbm


@nki.jit
def _layernorm_affine_kernel(x_hbm, weight_hbm, bias_hbm, eps):
    """LayerNorm forward with learnable affine (elementwise_affine=True).

    Args:
        x_hbm:      [N, H]  any float dtype (bf16 in production); H % 128 == 0.
        weight_hbm: [1, H]  fp32 (gamma).
        bias_hbm:   [1, H]  fp32 (beta).
        eps:        Python float.

    Returns:
        y_hbm: [N, H] same dtype as x_hbm.
    """
    N, H = x_hbm.shape
    TILE_N = nl.tile_size.pmax  # 128

    y_hbm = nl.ndarray((N, H), dtype=x_hbm.dtype, buffer=nl.shared_hbm)

    # Load weight + bias ONCE into SBUF as [1, H] fp32. tensor_scalar will
    # broadcast them across the partition axis as vector operands.
    weight_sb = nl.ndarray((1, H), dtype=nl.float32, buffer=nl.sbuf)
    bias_sb = nl.ndarray((1, H), dtype=nl.float32, buffer=nl.sbuf)
    nisa.dma_copy(dst=weight_sb, src=weight_hbm[0:1, 0:H])
    nisa.dma_copy(dst=bias_sb, src=bias_hbm[0:1, 0:H])

    num_tiles = div_ceil(N, TILE_N)
    inv_h = 1.0 / float(H)
    eps_f = float(eps)

    for tile_idx in nl.affine_range(num_tiles):
        n_start = tile_idx * TILE_N
        n_end = min(n_start + TILE_N, N)
        n_sz = n_end - n_start

        x_tile_in = nl.ndarray((n_sz, H), dtype=x_hbm.dtype, buffer=nl.sbuf)
        nisa.dma_copy(dst=x_tile_in, src=x_hbm[n_start:n_end, 0:H])

        x_f32 = nl.ndarray((n_sz, H), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_copy(dst=x_f32, src=x_tile_in)

        sum_x = nl.ndarray((n_sz, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_reduce(dst=sum_x, data=x_f32, op=nl.add, axis=(1,))

        mean = nl.ndarray((n_sz, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_scalar(
            dst=mean, data=sum_x, op0=nl.multiply, operand0=inv_h
        )

        # `tensor_scalar` over `tensor_tensor` for the [P, H] - [P, 1]
        # broadcast — see the no-affine kernel above for the rationale.
        diff = nl.ndarray((n_sz, H), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_scalar(
            dst=diff,
            data=x_f32,
            op0=nl.subtract,
            operand0=mean,
        )

        diff_sq = nl.ndarray((n_sz, H), dtype=nl.float32, buffer=nl.sbuf)
        nisa.activation(dst=diff_sq, data=diff, op=nl.square)

        sum_sq = nl.ndarray((n_sz, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_reduce(dst=sum_sq, data=diff_sq, op=nl.add, axis=(1,))

        inv_std = nl.ndarray((n_sz, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.activation(
            dst=inv_std,
            data=sum_sq,
            op=nl.rsqrt,
            scale=inv_h,
            bias=eps_f,
        )

        x_hat = nl.ndarray((n_sz, H), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_scalar(
            dst=x_hat,
            data=diff,
            op0=nl.multiply,
            operand0=inv_std,
        )

        # y = x_hat * weight + bias, fused via tensor_scalar's pre-activation
        # multiply+add. weight_sb / bias_sb are [1, H] vectors broadcast over
        # the partition axis.
        y_f32 = nl.ndarray((n_sz, H), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_scalar(
            dst=y_f32,
            data=x_hat,
            op0=nl.multiply,
            operand0=weight_sb,
            op1=nl.add,
            operand1=bias_sb,
        )

        # Cast back to input dtype.
        y_lo = nl.ndarray((n_sz, H), dtype=x_hbm.dtype, buffer=nl.sbuf)
        nisa.tensor_copy(dst=y_lo, src=y_f32)
        nisa.dma_copy(dst=y_hbm[n_start:n_end, 0:H], src=y_lo)

    return y_hbm


# ----- Python entrypoint ------------------------------------------------
def _is_neuron_tensor(t: torch.Tensor) -> bool:
    """Return True iff `t` lives on a Neuron / XLA device."""
    if t is None:
        return False
    return t.device.type in ("xla", "neuron")


def _layernorm_torch_ref(
    x: torch.Tensor,
    weight: torch.Tensor | None,
    bias: torch.Tensor | None,
    eps: float,
) -> torch.Tensor:
    """PyTorch reference matching `torch.nn.LayerNorm` over the last dim.

    Returns the output in the same dtype as `x` (LayerNorm casts internally
    to fp32 for the reduction and back to x.dtype on output).
    """
    in_dtype = x.dtype
    H = x.shape[-1]
    xf = x.float()
    mu = xf.mean(dim=-1, keepdim=True)
    var = (xf - mu).pow(2).mean(dim=-1, keepdim=True)
    x_hat = (xf - mu) * torch.rsqrt(var + eps)
    if weight is not None:
        x_hat = x_hat * weight.float()
    if bias is not None:
        x_hat = x_hat + bias.float()
    return x_hat.to(in_dtype)


def nki_layernorm(
    x: torch.Tensor,
    weight: torch.Tensor | None,
    bias: torch.Tensor | None,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Drop-in NKI LayerNorm forward over the last dim.

    Args:
        x:      [B, L, H]  any float dtype (bf16 in production)
        weight: [H]        fp32, or None for elementwise_affine=False
        bias:   [H]        fp32, or None for elementwise_affine=False
        eps:    float      Wan uses 1e-6; class default in WanLayerNorm is 1e-6.

    Returns:
        y: [B, L, H] same dtype as x. Matches `torch.nn.LayerNorm` output dtype.

    Notes:
        - Reduction and rsqrt run in fp32 internally.
        - On non-Neuron tensors (CPU), this falls back to the PyTorch
          reference, so the function is safely importable in test envs.
        - Forward-only (no_grad-safe). Wan's modules call this inside
          `with torch.no_grad()` paths during inference.
        - weight is None XOR bias is None is forbidden by torch.nn.LayerNorm
          (both are bound by `elementwise_affine`), so we enforce the same.
    """
    # Host-side validation (raise here, not inside @nki.jit — see commit 45439ca4).
    if x.dim() != 3:
        raise ValueError(f"nki_layernorm expects [B, L, H], got {tuple(x.shape)}")
    B, L, H = x.shape
    if H % 128 != 0:
        raise ValueError(f"nki_layernorm requires H % 128 == 0, got H={H}")

    # weight/bias are tied: either both None (no affine) or both [H].
    if (weight is None) != (bias is None):
        raise ValueError(
            "nki_layernorm: weight and bias must both be None or both be tensors "
            f"(got weight={weight is not None}, bias={bias is not None})"
        )
    if weight is not None:
        if weight.shape != (H,):
            raise ValueError(
                f"nki_layernorm weight must be [H={H}], got {tuple(weight.shape)}"
            )
        if bias.shape != (H,):
            raise ValueError(
                f"nki_layernorm bias must be [H={H}], got {tuple(bias.shape)}"
            )

    # CPU / non-neuron fallback uses the PyTorch reference verbatim.
    if not _is_neuron_tensor(x):
        return _layernorm_torch_ref(x, weight, bias, eps)

    # Flatten B*L for the kernel; reshape back at the end.
    x_flat = x.reshape(B * L, H).contiguous()

    if weight is None:
        y_flat = _layernorm_no_affine_kernel(x_flat, float(eps))
    else:
        w_row = weight.to(torch.float32).reshape(1, H).contiguous()
        b_row = bias.to(torch.float32).reshape(1, H).contiguous()
        y_flat = _layernorm_affine_kernel(x_flat, w_row, b_row, float(eps))

    return y_flat.reshape(B, L, H)
