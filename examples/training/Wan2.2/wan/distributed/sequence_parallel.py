# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
import torch

from ..modules.model import sinusoidal_embedding_1d
from .ulysses import distributed_attention
from .util import gather_forward, get_rank, get_world_size


def pad_freqs(original_tensor, target_len):
    seq_len, s1, s2 = original_tensor.shape
    pad_size = target_len - seq_len
    padding_tensor = torch.ones(
        pad_size,
        s1,
        s2,
        dtype=original_tensor.dtype,
        device=original_tensor.device)
    padded_tensor = torch.cat([original_tensor, padding_tensor], dim=0)
    return padded_tensor


@torch.amp.autocast('cpu', enabled=False)
def rope_apply(x, grid_sizes, freqs, even_mask, odd_mask):
    """
    Real-valued 3D RoPE for the sequence-parallel path. Mirrors the optimized
    rope in `wan/modules/model.py` but operates on this rank's per-rank shard:
    the full f*h*w angle table is built once, padded to s*sp_size with zero
    angles (cos=1, sin=0 → identity rotation on padding), and only this rank's
    [sp_rank*s : (sp_rank+1)*s] slice is materialized as cos/sin.

    Avoids `torch.view_as_complex` / `torch.view_as_real` so it lowers cleanly
    on the PyTorch Native Neuron device.

    x:          [B, s_per_rank, N, D].   s_per_rank = seq_len // sp_size
    grid_sizes: [B, 3].
    freqs:      [M, D // 2]   (angle table from rope_params).
    even_mask:  [1, 1, D] with 1s at even channels, 0s at odd channels.
    odd_mask:   1 - even_mask.
    """
    B, s, N, D = x.shape
    c = D // 2

    sp_size = get_world_size()
    sp_rank = get_rank()
    full_s = s * sp_size

    split_sizes = [c - 2 * (c // 3), c // 3, c // 3]

    x_compute = x.float()
    freqs_compute = freqs.float()
    freqs_f = freqs_compute[:, :split_sizes[0]]
    freqs_h = freqs_compute[:, split_sizes[0]:split_sizes[0] + split_sizes[1]]
    freqs_w = freqs_compute[:, split_sizes[0] + split_sizes[1]:]

    out = []
    for i, (f, h, w) in enumerate(grid_sizes.tolist()):
        token_len = f * h * w

        # angles for the full (un-sharded) sequence
        angles_full = torch.cat(
            [
                freqs_f[:f].view(f, 1, 1, -1).expand(f, h, w, -1),
                freqs_h[:h].view(1, h, 1, -1).expand(f, h, w, -1),
                freqs_w[:w].view(1, 1, w, -1).expand(f, h, w, -1),
            ],
            dim=-1,
        ).reshape(token_len, 1, c)

        # pad to s*sp_size with zero angles → identity rotation on tail
        if full_s > token_len:
            pad = angles_full.new_zeros(full_s - token_len, 1, c)
            angles_full = torch.cat([angles_full, pad], dim=0)

        # this rank's slice
        angles = angles_full[sp_rank * s:(sp_rank + 1) * s]   # [s, 1, c]
        cos = torch.cos(angles)                                # [s, 1, c]
        sin = torch.sin(angles)                                # [s, 1, c]

        x_head = x_compute[i].reshape(s, N, c, 2)             # [s, N, c, 2]
        x0 = x_head[:, :, :, 0]                               # [s, N, c]
        x1 = x_head[:, :, :, 1]                               # [s, N, c]

        y0 = x0 * cos - x1 * sin
        y1 = x0 * sin + x1 * cos

        # interleave + masked write to avoid view_as_complex / stack-flatten
        y0e = y0.repeat_interleave(2, dim=-1)                 # [s, N, D]
        y1e = y1.repeat_interleave(2, dim=-1)                 # [s, N, D]
        x_rot = y0e * even_mask + y1e * odd_mask              # [s, N, D]

        out.append(x_rot)

    return torch.stack(out, dim=0).to(dtype=x.dtype)


def sp_dit_forward(
    self,
    x,
    t,
    context,
    seq_len,
    y=None,
):
    """
    x:              A list of videos each with shape [C, T, H, W].
    t:              [B].
    context:        A list of text embeddings each with shape [L, C].
    """
    if self.model_type == 'i2v':
        assert y is not None
    # params
    device = self.patch_embedding.weight.device
    if self.freqs.device != device:
        self.freqs = self.freqs.to(device)

    if y is not None:
        x = [torch.cat([u, v], dim=0) for u, v in zip(x, y)]

    # embeddings
    x = [self.patch_embedding(u.unsqueeze(0)) for u in x]
    grid_sizes = torch.stack(
        [torch.tensor(u.shape[2:], dtype=torch.long) for u in x])
    x = [u.flatten(2).transpose(1, 2) for u in x]
    seq_lens = torch.tensor([u.size(1) for u in x], dtype=torch.long)
    assert seq_lens.max() <= seq_len
    x = torch.cat([
        torch.cat([u, u.new_zeros(1, seq_len - u.size(1), u.size(2))], dim=1)
        for u in x
    ])

    # time embeddings — explicit fp32 (was a CUDA-typed autocast block; cast directly
    # so the path works on the Native Neuron device too)
    if t.dim() == 1:
        t = t.expand(t.size(0), seq_len)
    bt = t.size(0)
    t = t.flatten()
    e = self.time_embedding(
        sinusoidal_embedding_1d(self.freq_dim,
                                t).unflatten(0, (bt, seq_len)).float()).float()
    e0 = self.time_projection(e).unflatten(2, (6, self.dim)).float()
    assert e.dtype == torch.float32 and e0.dtype == torch.float32

    # context
    context_lens = None
    context = self.text_embedding(
        torch.stack([
            torch.cat([u, u.new_zeros(self.text_len - u.size(0), u.size(1))])
            for u in context
        ]))

    # Context Parallel
    x = torch.chunk(x, get_world_size(), dim=1)[get_rank()]
    e = torch.chunk(e, get_world_size(), dim=1)[get_rank()]
    e0 = torch.chunk(e0, get_world_size(), dim=1)[get_rank()]

    # arguments
    kwargs = dict(
        e=e0,
        seq_lens=seq_lens,
        grid_sizes=grid_sizes,
        freqs=self.freqs,
        context=context,
        context_lens=context_lens)

    for block in self.blocks:
        x = block(x, **kwargs)

    # head
    x = self.head(x, e)

    # Context Parallel
    x = gather_forward(x, dim=1)

    # unpatchify
    x = self.unpatchify(x, grid_sizes)
    return [u.float() for u in x]


def sp_attn_forward(self, x, seq_lens, grid_sizes, freqs, dtype=torch.bfloat16):
    b, s, n, d = *x.shape[:2], self.num_heads, self.head_dim
    half_dtypes = (torch.float16, torch.bfloat16)

    def half(x):
        return x if x.dtype in half_dtypes else x.to(dtype)

    # query, key, value function
    def qkv_fn(x):
        q = self.norm_q(self.q(x)).view(b, s, n, d)
        k = self.norm_k(self.k(x)).view(b, s, n, d)
        v = self.v(x).view(b, s, n, d)
        return q, k, v

    q, k, v = qkv_fn(x)
    q = rope_apply(q, grid_sizes, freqs, self.even_mask, self.odd_mask)
    k = rope_apply(k, grid_sizes, freqs, self.even_mask, self.odd_mask)

    x = distributed_attention(
        half(q),
        half(k),
        half(v),
        seq_lens,
        window_size=self.window_size,
    )

    # output
    x = x.flatten(2)
    x = self.o(x)
    return x
