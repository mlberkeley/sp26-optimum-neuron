"""Runtime monkey-patches that swap Wan2.2 torch ops for NKI equivalents.

Activated by setting environment variable ``WAN_USE_NKI_KERNELS=1`` *before*
importing ``wan``. Triggered from ``wan.kernels.__init__`` so that
``import wan`` (which is what ``generate_rolling.py`` does) is sufficient to
apply the patches.

Per-kernel toggles (default on when the master flag is set):
    WAN_NKI_ATTENTION   — swap ``wan.modules.attention.attention()`` for the
                          NKI ``attention_cte`` path on Neuron.
    WAN_NKI_RMSNORM     — swap ``WanRMSNorm.forward`` for the NKI kernel.
    WAN_NKI_ROPE        — swap ``rope_apply`` (single-device + SP variants)
                          for the fused NKI kernel.

Each toggle is independent; missing/0/false disables it. The master flag
turns *all* on by default.
"""
from __future__ import annotations

import logging
import os

logger = logging.getLogger(__name__)


def _enabled(name: str, default: bool) -> bool:
    """``WAN_NKI_<NAME>`` env-var lookup with a default."""
    val = os.environ.get(name)
    if val is None:
        return default
    return val.strip().lower() in ("1", "true", "yes", "on")


def _patch_attention() -> None:
    from wan.modules import attention as wan_attn
    from wan.kernels.wan_attention_nki import attention_nki

    _orig = wan_attn.attention

    def _patched_attention(q, k, v, *args, **kwargs):
        # Only swap on Neuron device; CUDA/CPU paths fall through unchanged.
        if q.device.type == "neuron":
            return attention_nki(q, k, v,
                                 dropout_p=kwargs.get("dropout_p", 0.0),
                                 causal=kwargs.get("causal", False),
                                 dtype=kwargs.get("dtype", q.dtype))
        return _orig(q, k, v, *args, **kwargs)

    wan_attn.attention = _patched_attention
    logger.info("[wan.kernels] swapped wan.modules.attention.attention -> NKI")


def _patch_rmsnorm() -> None:
    from wan.modules import model as wan_model
    from wan.kernels.wan_rms_norm_nki import WanRMSNormNKI

    _OrigRMSNorm = wan_model.WanRMSNorm

    class _PatchedRMSNorm(_OrigRMSNorm):
        def forward(self, x):
            if x.device.type == "neuron":
                # Use NKI path; build a one-shot NKI module that shares this
                # instance's weight tensor.
                if not hasattr(self, "_nki_impl"):
                    nki_impl = WanRMSNormNKI(dim=self.weight.shape[0], eps=self.eps)
                    nki_impl.weight = self.weight  # share buffer
                    self._nki_impl = nki_impl
                return self._nki_impl(x)
            return super().forward(x)

    wan_model.WanRMSNorm = _PatchedRMSNorm
    logger.info("[wan.kernels] swapped wan.modules.model.WanRMSNorm -> NKI")


def _patch_layernorm() -> None:
    from wan.modules import model as wan_model
    from wan.kernels.wan_layer_norm_nki import WanLayerNormNKI

    _OrigLayerNorm = wan_model.WanLayerNorm

    class _PatchedLayerNorm(_OrigLayerNorm):
        def forward(self, x):
            if x.device.type == "neuron":
                if not hasattr(self, "_nki_impl"):
                    nki_impl = WanLayerNormNKI(
                        normalized_shape=self.normalized_shape,
                        eps=self.eps,
                        elementwise_affine=self.elementwise_affine,
                    )
                    if self.elementwise_affine:
                        nki_impl.weight = self.weight
                        nki_impl.bias = self.bias
                    self._nki_impl = nki_impl
                return self._nki_impl(x)
            return super().forward(x)

    wan_model.WanLayerNorm = _PatchedLayerNorm
    logger.info("[wan.kernels] swapped wan.modules.model.WanLayerNorm -> NKI")


def _patch_rope() -> None:
    from wan.modules import model as wan_model
    from wan.distributed import sequence_parallel as sp_mod
    from wan.kernels.wan_rope_nki import rope_apply_nki

    _orig_single = wan_model.rope_apply
    _orig_sp = sp_mod.rope_apply

    def _patched_single(x, grid_sizes, freqs, even_mask, odd_mask):
        if x.device.type == "neuron":
            return rope_apply_nki(x, grid_sizes, freqs, even_mask, odd_mask,
                                  sp_rank=0, sp_size=1)
        return _orig_single(x, grid_sizes, freqs, even_mask, odd_mask)

    def _patched_sp(x, grid_sizes, freqs, even_mask, odd_mask):
        if x.device.type == "neuron":
            from wan.distributed.util import get_rank, get_world_size
            return rope_apply_nki(x, grid_sizes, freqs, even_mask, odd_mask,
                                  sp_rank=get_rank(), sp_size=get_world_size())
        return _orig_sp(x, grid_sizes, freqs, even_mask, odd_mask)

    wan_model.rope_apply = _patched_single
    sp_mod.rope_apply = _patched_sp
    logger.info("[wan.kernels] swapped rope_apply (single + SP) -> NKI")


def apply() -> None:
    """Apply all enabled NKI patches. Safe to call multiple times."""
    enabled = []
    if _enabled("WAN_NKI_ATTENTION", default=True):
        try:
            _patch_attention()
            enabled.append("attention")
        except Exception as e:
            logger.warning("[wan.kernels] attention patch failed: %r", e)
    if _enabled("WAN_NKI_RMSNORM", default=True):
        try:
            _patch_rmsnorm()
            enabled.append("rmsnorm")
        except Exception as e:
            logger.warning("[wan.kernels] rmsnorm patch failed: %r", e)
    if _enabled("WAN_NKI_LAYERNORM", default=True):
        try:
            _patch_layernorm()
            enabled.append("layernorm")
        except Exception as e:
            logger.warning("[wan.kernels] layernorm patch failed: %r", e)
    if _enabled("WAN_NKI_ROPE", default=True):
        try:
            _patch_rope()
            enabled.append("rope")
        except Exception as e:
            logger.warning("[wan.kernels] rope patch failed: %r", e)
    if enabled:
        print(f"[wan.kernels] integration ON: {','.join(enabled)}")
