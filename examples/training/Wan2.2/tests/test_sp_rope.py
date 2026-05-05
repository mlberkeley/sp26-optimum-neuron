"""
CPU-only equivalence test for the rewritten SP rope_apply.

Compares the new real-valued, masked-write `rope_apply` (which avoids
`view_as_complex`) against a from-scratch reference that mimics the
original complex-number implementation. Each rank sees only its s_per_rank
shard; this test verifies that gathering all rank outputs reconstructs
the same full-sequence rope as the single-device path.
"""
import math
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def make_freqs(max_seq_len, dim, theta=10000.0):
    """Mimics wan/modules/model.py:rope_params (complex-form)."""
    assert dim % 2 == 0
    freqs = torch.outer(
        torch.arange(max_seq_len),
        1.0 / theta**(torch.arange(0, dim, 2).float() / dim),
    )
    # Original returns complex angles via polar; here we return the angle table
    # directly because the new rope_apply takes angles, not complex multipliers.
    return freqs  # [max_seq_len, dim/2]


def reference_full_rope(x_full, grid_sizes, freqs):
    """Reference: applies the original complex-number RoPE to the full sequence."""
    B, L, N, D = x_full.shape
    c = D // 2
    split_sizes = [c - 2 * (c // 3), c // 3, c // 3]
    out = []
    for i, (f, h, w) in enumerate(grid_sizes.tolist()):
        token_len = f * h * w
        # complex view of x
        x_i_complex = torch.view_as_complex(
            x_full[i, :token_len].to(torch.float64).reshape(token_len, N, c, 2)
        )
        # build phases as complex unit vectors via polar
        angles = torch.cat([
            freqs[:f, :split_sizes[0]].view(f, 1, 1, -1).expand(f, h, w, -1),
            freqs[:h, split_sizes[0]:split_sizes[0] + split_sizes[1]].view(1, h, 1, -1).expand(f, h, w, -1),
            freqs[:w, split_sizes[0] + split_sizes[1]:].view(1, 1, w, -1).expand(f, h, w, -1),
        ], dim=-1).reshape(token_len, 1, c)
        phases = torch.polar(torch.ones_like(angles, dtype=torch.float64), angles.to(torch.float64))
        x_rot_complex = x_i_complex * phases
        x_rot = torch.view_as_real(x_rot_complex).flatten(2)  # [token_len, N, D]
        # re-attach the padding tail unchanged
        x_i = torch.cat([x_rot, x_full[i, token_len:].to(torch.float64)], dim=0)
        out.append(x_i)
    return torch.stack(out, dim=0).to(dtype=x_full.dtype)


def gather_sp_rope(x_full, grid_sizes, freqs, sp_size, even_mask, odd_mask):
    """Run the new SP rope_apply once per rank, then concatenate the per-rank
    outputs along the sequence dim, to compare against the full reference."""
    import wan.distributed.sequence_parallel as sp_mod
    import wan.distributed.util as sp_util

    B, L, N, D = x_full.shape
    s = L // sp_size

    # patch get_world_size / get_rank to return our test values
    orig_ws = sp_util.get_world_size
    orig_r = sp_util.get_rank
    pieces = []
    try:
        for r in range(sp_size):
            sp_util.get_world_size = lambda: sp_size
            sp_util.get_rank = lambda r=r: r
            sp_mod.get_world_size = sp_util.get_world_size
            sp_mod.get_rank = sp_util.get_rank
            x_shard = x_full[:, r * s:(r + 1) * s].contiguous()
            y_shard = sp_mod.rope_apply(x_shard, grid_sizes, freqs, even_mask, odd_mask)
            pieces.append(y_shard)
    finally:
        sp_util.get_world_size = orig_ws
        sp_util.get_rank = orig_r
        sp_mod.get_world_size = orig_ws
        sp_mod.get_rank = orig_r

    return torch.cat(pieces, dim=1)


def make_masks(D):
    even_mask = torch.zeros(1, 1, D, dtype=torch.float32)
    even_mask[..., 0::2] = 1.0
    odd_mask = 1.0 - even_mask
    return even_mask, odd_mask


def test_case(label, B, F, H, W, N, head_dim, sp_size, dtype=torch.float32):
    print(f"\n[{label}] B={B} F={F} H={H} W={W} N={N} D={head_dim} sp={sp_size} dtype={dtype}")
    torch.manual_seed(0)

    D = head_dim
    token_len = F * H * W
    L = math.ceil(token_len / sp_size) * sp_size  # padded full length

    # build x with random values; padding region is anything but should be ignored
    x_full = torch.randn(B, L, N, D, dtype=dtype)
    # pad-region values don't matter to the rope output beyond identity rotation,
    # but we keep them random to mirror the runtime.

    # freqs table large enough
    freqs = make_freqs(max_seq_len=max(F, H, W) + 8, dim=D)

    grid_sizes = torch.tensor([[F, H, W]] * B)
    even_mask, odd_mask = make_masks(D)

    ref = reference_full_rope(x_full, grid_sizes, freqs)
    sp = gather_sp_rope(x_full, grid_sizes, freqs, sp_size, even_mask, odd_mask)

    # Compare only the token_len region (first token_len positions); padded tail
    # is identity rotation in the new path and the ref keeps it unchanged in
    # source dtype, so they're the same up to float precision.
    ref_tok = ref[:, :token_len].float()
    sp_tok = sp[:, :token_len].float()

    abs_err = (ref_tok - sp_tok).abs()
    print(f"  max_abs_err = {abs_err.max().item():.3e}, mean_abs_err = {abs_err.mean().item():.3e}")
    rtol = 1e-4 if dtype == torch.float32 else 1e-2
    atol = 1e-4 if dtype == torch.float32 else 1e-2
    assert torch.allclose(ref_tok, sp_tok, rtol=rtol, atol=atol), \
        f"{label}: SP rope_apply diverges from reference"
    print(f"  OK (rtol={rtol}, atol={atol})")


if __name__ == "__main__":
    # Small synthetic case
    test_case("tiny", B=1, F=3, H=4, W=5, N=2, head_dim=16, sp_size=4)
    # SP=8 with 17-frame chunk geometry (T=5, H=22, W=40 after patch)
    test_case("17f_sp8", B=1, F=5, H=22, W=40, N=24, head_dim=128, sp_size=8)
    # SP=16
    test_case("17f_sp16", B=1, F=5, H=22, W=40, N=24, head_dim=128, sp_size=16)
    # Padding case: token_len not divisible by sp_size
    test_case("pad_sp7", B=1, F=3, H=4, W=5, N=2, head_dim=16, sp_size=7)
    print("\nAll SP rope_apply tests passed.")
