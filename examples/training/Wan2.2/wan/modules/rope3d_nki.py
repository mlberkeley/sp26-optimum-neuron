# wan/modules/rope3d_nki.py

import torch

import nki
import nki.language as nl
import nki.isa as nisa


@nki.jit
def rope3d_forward_nki(
    x,
    cos,
    sin,
    f: int,
    h: int,
    w: int,
):

    # Assumptions:
    # - B == 1
    # - x: [1, L, N, D], contiguous
    # - D even
    # - cos/sin: [f*h*w, D//2], already dense + contiguous
    # - first f*h*w tokens are rotated
    # - no tail tokens, that is: L == f * h * w
    # - BM = 128

    B = x.shape[0]
    L = x.shape[1]
    N = x.shape[2]
    D = x.shape[3]
    C = D // 2
    SEQ = f * h * w
    BM = 128

    out = nl.ndarray(x.shape, dtype=x.dtype, buffer=nl.shared_hbm)
    b = 0

    # positions outer — load cos/sin ONCE per tile
    for p0 in range(0, SEQ, BM):
        rows = BM if p0 + BM <= SEQ else SEQ - p0

        co = nl.load(cos[p0:p0 + rows, 0:C])
        si = nl.load(sin[p0:p0 + rows, 0:C])

        # heads inner — pipelined, reuses co/si already in SBUF
        for n in nl.affine_range(N):
            x_tile = nl.load(x[b, p0:p0 + rows, n, 0:D])
            x_view = x_tile.reshape((rows, C, 2))
            x0 = x_view[:, :, 0]
            x1 = x_view[:, :, 1]

            y0 = nl.subtract(
                nl.multiply(x0, co),
                nl.multiply(x1, si),
            )
            y1 = nl.add(
                nl.multiply(x0, si),
                nl.multiply(x1, co),
            )

            y_tile = nl.ndarray((rows, D), dtype=x.dtype, buffer=nl.sbuf)
            y_view = y_tile.reshape((rows, C, 2))
            nisa.dma_copy(dst=y_view[:, :, 0], src=y0)
            nisa.dma_copy(dst=y_view[:, :, 1], src=y1)

            nl.store(out[b, p0:p0 + rows, n, 0:D], y_tile)

    return out



def build_dense_rope_tables(freqs: torch.Tensor, D: int, f: int, h: int, w: int):
    """
    Returns dense cos/sin tables:
      cos, sin: [f*h*w, D//2]

    This intentionally moves the f/h/w indexing logic out of the NKI kernel.
    """

    assert D % 2 == 0

    C = D // 2
    c_f = C - 2 * (C // 3)
    c_h = C // 3
    c_w = C // 3

    freqs = freqs.float()

    freqs_f = freqs[:, :c_f]
    freqs_h = freqs[:, c_f:c_f + c_h]
    freqs_w = freqs[:, c_f + c_h:c_f + c_h + c_w]

    angles = torch.cat(
        [
            freqs_f[:f].view(f, 1, 1, c_f).expand(f, h, w, c_f),
            freqs_h[:h].view(1, h, 1, c_h).expand(f, h, w, c_h),
            freqs_w[:w].view(1, 1, w, c_w).expand(f, h, w, c_w),
        ],
        dim=-1,
    ).reshape(f * h * w, C)

    return torch.cos(angles).contiguous(), torch.sin(angles).contiguous()

