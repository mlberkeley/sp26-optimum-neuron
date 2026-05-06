"""Drop-in NKI replacement for `wan.modules.model.WanRMSNorm`.

Same `__init__` signature and parameter semantics as the original module:
    WanRMSNormNKI(dim, eps=1e-5)
The runtime override `eps=1e-6` (used by Wan's self-attention norm_q/norm_k)
is passed via the constructor at instantiation time, identical to the
PyTorch class.

On Neuron tensors, the forward dispatches to the NKI kernel in
`rmsnorm_nki.nki_rmsnorm`. On any non-Neuron device (CPU, CUDA), the
forward falls through to the same math as the PyTorch original so this
module can be unit-tested off-device and used in mixed environments.

Notes
-----
- Output dtype follows `WanRMSNorm.forward` exactly: it is the dtype of
  `(x.to(in_dtype) * fp32_weight)`, which PyTorch promotes to fp32. Code
  paths in `wan/modules/model.py` that consume this RMSNorm's output
  (WanSelfAttention.forward at lines 281-297) cast back to bf16 implicitly
  via the next Linear; behavior is unchanged.
- The original `_norm` helper is kept as a CPU fallback so `.to('cpu')`
  smoke tests reproduce bit-for-bit PyTorch numerics.
- This is forward-only. Wan's WanRMSNorm currently runs without a custom
  backward, and PyTorch autograd cannot trace a Neuron @nki.jit kernel
  back through itself; if training enablement is needed later, wrap
  `nki_rmsnorm` in a `torch.autograd.Function` whose backward is the
  PyTorch reference.
"""

from __future__ import annotations

import torch
from torch import nn

from .rmsnorm_nki import _is_neuron_tensor, nki_rmsnorm


class WanRMSNormNKI(nn.Module):
    """Drop-in replacement for `WanRMSNorm` using a NKI kernel on Neuron."""

    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Args:
            x: [B, L, C] tensor; C must equal `self.dim`.

        Returns:
            normalized tensor with shape [B, L, C].
        """
        if x.shape[-1] != self.dim:
            raise ValueError(
                f"WanRMSNormNKI: trailing dim={x.shape[-1]} != self.dim={self.dim}"
            )

        if _is_neuron_tensor(x):
            # The NKI kernel internally upcasts to fp32 and matches the
            # numerics of WanRMSNorm._norm on its bf16 input.
            return nki_rmsnorm(x, self.weight, eps=self.eps)

        # CPU / non-neuron fallback: reproduce WanRMSNorm.forward verbatim.
        return self._norm(x.float()).type_as(x) * self.weight

    def _norm(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
