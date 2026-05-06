"""Drop-in NKI replacement for `wan.modules.model.WanLayerNorm`.

Same `__init__` signature and parameter semantics as the original module:
    WanLayerNormNKI(dim, eps=1e-6, elementwise_affine=False)

`WanLayerNorm` is a thin subclass of `torch.nn.LayerNorm` whose forward is
just `super().forward(x)`. That means the canonical math is exactly:

    y = (x - mean(x, dim=-1)) / sqrt(var(x, dim=-1) + eps) * gamma + beta

with `gamma` and `bias` only present when `elementwise_affine=True`. In
`wan.modules.model`:

    norm1, norm2, head.norm  -> elementwise_affine=False  (no weight/bias)
    norm3 (cross_attn_norm)  -> elementwise_affine=True

so both branches must be supported.

On Neuron tensors, the forward dispatches to `nki_layernorm`. On CPU/CUDA,
the forward falls through to the same math via PyTorch so this module can
be unit-tested off-device.

Notes
-----
- Output dtype follows `torch.nn.LayerNorm` exactly: same dtype as the input.
  This differs from `WanRMSNormNKI`, which intentionally returns fp32 to mirror
  PyTorch's `bf16 * fp32_weight -> fp32` promotion. LayerNorm doesn't promote
  in the same way because the standard nn.LayerNorm contract is dtype-preserving.
- The CPU fallback uses the same compute path as `nn.LayerNorm` (delegating
  to `torch.nn.functional.layer_norm`) so off-device numerics match bit-for-bit.
- This is forward-only. PyTorch autograd cannot trace through @nki.jit.
"""

from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F

from .layernorm_nki import _is_neuron_tensor, nki_layernorm


class WanLayerNormNKI(nn.Module):
    """Drop-in replacement for `WanLayerNorm` using a NKI kernel on Neuron.

    Mirrors `torch.nn.LayerNorm` over the last dim, with `elementwise_affine`
    controlling whether `weight` (gamma) and `bias` (beta) are learnable
    parameters. On non-Neuron devices, falls back to `F.layer_norm` for
    bit-identical numerics.
    """

    def __init__(
        self,
        dim: int,
        eps: float = 1e-6,
        elementwise_affine: bool = False,
    ):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.elementwise_affine = elementwise_affine
        self.normalized_shape = (dim,)
        if elementwise_affine:
            self.weight = nn.Parameter(torch.ones(dim))
            self.bias = nn.Parameter(torch.zeros(dim))
        else:
            # Register as None so `model.to(device)` is a no-op and the
            # attribute exists for the forward dispatch.
            self.register_parameter("weight", None)
            self.register_parameter("bias", None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Args:
            x: [B, L, C] tensor; C must equal `self.dim`.

        Returns:
            normalized tensor with shape [B, L, C], same dtype as x.
        """
        if x.shape[-1] != self.dim:
            raise ValueError(
                f"WanLayerNormNKI: trailing dim={x.shape[-1]} != self.dim={self.dim}"
            )

        if _is_neuron_tensor(x):
            return nki_layernorm(x, self.weight, self.bias, eps=self.eps)

        # CPU / non-neuron fallback: defer to torch.nn.functional for
        # bit-identical numerics with the original WanLayerNorm.
        return F.layer_norm(
            x,
            self.normalized_shape,
            weight=self.weight,
            bias=self.bias,
            eps=self.eps,
        )
