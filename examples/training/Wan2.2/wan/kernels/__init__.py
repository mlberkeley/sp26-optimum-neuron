"""NKI kernels for Wan2.2 on Trainium.

Each module exposes a Python entrypoint that returns a torch.Tensor on the
same device as its primary input. Kernels are inference-forward-only unless
otherwise noted; backwards passes use the reference PyTorch implementations.

Usage:
    from wan.kernels.attention_nki import nki_attention      # raw entrypoint
    from wan.kernels.wan_attention_nki import attention_nki  # SDPA-shaped wrapper
"""

# Eagerly import the submodules so `wan.kernels.attention_nki` and
# `wan.kernels.wan_attention_nki` are reachable without a separate import,
# but DO NOT bind their public names at the package level — the module
# `attention_nki.py` and the function `attention_nki` in
# `wan_attention_nki.py` would otherwise shadow each other.
from . import attention_nki  # noqa: F401
from . import wan_attention_nki  # noqa: F401

# Activate runtime monkey-patches when WAN_USE_NKI_KERNELS=1. This is the
# integration toggle for the inference path: with the flag set, importing
# wan.kernels (which `import wan` does) replaces wan.modules.attention.attention,
# WanRMSNorm.forward, and rope_apply with NKI-backed equivalents on Neuron.
# Default OFF — must be explicitly opted into.
import os as _os
if _os.environ.get("WAN_USE_NKI_KERNELS", "").strip().lower() in ("1", "true", "yes", "on"):
    from . import _enable_kernels as _enable_kernels  # noqa: F401
    _enable_kernels.apply()
