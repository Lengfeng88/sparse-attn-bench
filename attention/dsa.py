"""
Dense-Sparse Attention (DSA) — benchmarking reference implementation.

Design:
  - Dense part:  local sliding window of size `window_size` (causal).
  - Sparse part: uniformly-sampled global tokens NOT already in the local window
                 (stub for Lightning Indexer top-k output).

Key implementation detail:
  Global tokens that fall inside the current local window are EXCLUDED from
  the sparse set to avoid counting the same key twice in the softmax, which
  would alter attention weights and produce incorrect results within the window.

Simplification boundary (vs. production hyper-parallel DSA):
  Production DSA uses a trainable Lightning Indexer network to predict top-k
  token indices dynamically. This version stubs the indexer with uniform
  sampling to isolate the memory/throughput benefits of the sparse access
  pattern, independent of indexer training dynamics.

Complexity: O(n·w + n·g)  vs. Dense O(n²)
  w = window_size, g = global_tokens (non-overlapping with window)
"""

import math
import torch


def dsa_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    window_size: int = 128,
    global_tokens: int = 16,
    causal: bool = True,
) -> torch.Tensor:
    """DSA forward pass.

    Args:
        q, k, v      : (B, H, S, D)
        window_size  : local dense window size (causal: only past tokens)
        global_tokens: number of uniformly-sampled global sparse tokens
        causal       : apply causal masking

    Returns:
        out : (B, H, S, D)
    """
    B, H, S, D = q.shape
    scale = math.sqrt(D)
    out = torch.zeros_like(q)

    # Global sparse token indices (uniform stub for Lightning Indexer)
    stride = max(1, S // global_tokens)
    global_idx = torch.arange(0, S, stride, device=q.device)[:global_tokens]
    global_set = set(global_idx.tolist())

    CHUNK = 64  # process this many query positions at once
    for start in range(0, S, CHUNK):
        end = min(start + CHUNK, S)
        qc = q[:, :, start:end, :]  # (B, H, chunk, D)

        # Local window key range
        kw_lo = max(0, start - window_size) if causal else max(0, start - window_size // 2)
        kw_hi = end if causal else min(S, end + window_size // 2)
        window_set = set(range(kw_lo, kw_hi))

        kw = k[:, :, kw_lo:kw_hi, :]
        vw = v[:, :, kw_lo:kw_hi, :]
        scores_w = torch.matmul(qc, kw.transpose(-2, -1)) / scale  # (B,H,chunk,win)

        if causal:
            q_pos = torch.arange(start, end, device=q.device).unsqueeze(1)
            k_pos = torch.arange(kw_lo, kw_hi, device=q.device).unsqueeze(0)
            scores_w = scores_w.masked_fill(
                k_pos > q_pos, float("-inf")
            )

        # Extra global tokens: only those NOT already inside the local window
        # (avoids double-counting the same key in softmax)
        extra_global = sorted(global_set - window_set)

        if extra_global:
            eg = torch.tensor(extra_global, device=q.device)
            kg = k[:, :, eg, :]
            vg = v[:, :, eg, :]
            scores_g = torch.matmul(qc, kg.transpose(-2, -1)) / scale

            if causal:
                q_pos = torch.arange(start, end, device=q.device).unsqueeze(1)
                scores_g = scores_g.masked_fill(
                    eg.unsqueeze(0) > q_pos, float("-inf")
                )

            scores_all = torch.cat([scores_w, scores_g], dim=-1)
            kv_all     = torch.cat([vw, vg], dim=2)
        else:
            scores_all = scores_w
            kv_all     = vw

        attn = torch.softmax(scores_all.float(), dim=-1).to(q.dtype)
        out[:, :, start:end, :] = torch.matmul(attn, kv_all)

    return out