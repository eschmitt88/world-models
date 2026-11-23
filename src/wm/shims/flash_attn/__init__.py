"""Minimal SDPA stand-in for the `flash_attn` API used by Matrix-Game.

No flash-attn wheel targets torch 2.9 + sm_120 here; PyTorch SDPA (which
dispatches to its own flash/efficient kernels) is numerically equivalent for
the dense, unpadded calls Matrix-Game makes. Shapes follow flash-attn:
q,k,v [B, L, H, D].
"""
import torch
import torch.nn.functional as F

__version__ = "0.0-sdpa-shim"


def _sdpa(q, k, v, causal, softmax_scale, window_size=(-1, -1)):
    if window_size not in (None, (-1, -1), [-1, -1]):
        raise NotImplementedError("sliding-window attention not supported by the shim")
    if k.shape[2] != q.shape[2]:  # GQA
        rep = q.shape[2] // k.shape[2]
        k = k.repeat_interleave(rep, 2)
        v = v.repeat_interleave(rep, 2)
    o = F.scaled_dot_product_attention(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2),
                                       is_causal=causal, scale=softmax_scale)
    return o.transpose(1, 2)


def flash_attn_func(q, k, v, dropout_p=0.0, softmax_scale=None, causal=False,
                    window_size=(-1, -1), deterministic=False, **kw):
    return _sdpa(q, k, v, causal, softmax_scale, window_size)


def flash_attn_varlen_func(q, k, v, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k,
                           dropout_p=0.0, softmax_scale=None, causal=False,
                           window_size=(-1, -1), deterministic=False, **kw):
    cq = cu_seqlens_q.tolist(); ck = cu_seqlens_k.tolist()
    outs = []
    for i in range(len(cq) - 1):
        qi = q[cq[i]:cq[i + 1]][None]; ki = k[ck[i]:ck[i + 1]][None]; vi = v[ck[i]:ck[i + 1]][None]
        outs.append(_sdpa(qi, ki, vi, causal, softmax_scale, window_size)[0])
    return torch.cat(outs, 0)
