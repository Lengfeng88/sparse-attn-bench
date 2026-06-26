"""
Cross Sparse Attention (CSA) — benchmarking reference implementation.

Design:
  - Intra-segment: full causal attention within each segment (local dense).
  - Cross-segment: each segment attends to a fixed set of "representative tokens"
    from every other segment (one token per segment, e.g. the segment mean or
    the first token). This simulates the all-gather + sparse-select pattern used
    in production CSA, where a lightweight router picks cross-segment top-k tokens.

Simplification boundary (vs. production hyper-parallel CSA):
  - Production CSA uses a learned router or Lightning-Indexer-style scoring to pick
    the most relevant cross-segment tokens dynamically.
  - This version uses deterministic representative tokens (segment mean pooling)
    to isolate the memory/throughput benefits of the cross-segment sparse pattern,
    independent of router training dynamics.

Complexity:
  - Intra: O(n * s)  where s = segment_size
  - Cross: O(n * k)  where k = num_segments (one rep token per segment)
  - Total: O(n * (s + k))  vs. Dense O(n^2)
"""

import math
import torch
import torch.nn.functional as F


def _segment_representative(k: torch.Tensor, v: torch.Tensor,
                             seg_start: int, seg_end: int,
                             rep_mode: str = "mean") -> tuple:
    """Return a single (k_rep, v_rep) vector for segment [seg_start, seg_end).

    Args:
        k, v : (B, H, S, D)
        rep_mode: "mean"  — average pooling over the segment
                  "first" — first token of the segment
    Returns:
        k_rep, v_rep : (B, H, 1, D)
    """
    seg_k = k[:, :, seg_start:seg_end, :]
    seg_v = v[:, :, seg_start:seg_end, :]
    if rep_mode == "first":
        return seg_k[:, :, :1, :], seg_v[:, :, :1, :]
    # default: mean
    return seg_k.mean(dim=2, keepdim=True), seg_v.mean(dim=2, keepdim=True)


def csa_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    num_segments: int = 4,
    causal: bool = True,
    rep_mode: str = "mean",
    cross_weight: float = 0.5,
) -> torch.Tensor:
    """Cross Sparse Attention forward pass.

    Args:
        q, k, v      : (B, H, S, D)  — query, key, value tensors
        num_segments  : number of segments to split the sequence into
        causal        : apply causal mask within each segment
        rep_mode      : how to compute the cross-segment representative token
                        ("mean" or "first")
        cross_weight  : blending weight for cross-segment output [0, 1]
                        output = intra_out + cross_weight * cross_out

    Returns:
        out : (B, H, S, D)
    """
    B, H, S, D = q.shape
    scale = math.sqrt(D)

    seg_size = (S + num_segments - 1) // num_segments
    segments = []
    for i in range(num_segments):
        lo = i * seg_size
        hi = min(lo + seg_size, S)
        segments.append((lo, hi))

    # ── Stage 1: intra-segment full attention ──────────────────────────────
    out_intra = torch.zeros_like(q)
    for lo, hi in segments:
        qc = q[:, :, lo:hi, :]   # (B, H, seg_len, D)
        kc = k[:, :, lo:hi, :]
        vc = v[:, :, lo:hi, :]
        scores = torch.matmul(qc, kc.transpose(-2, -1)) / scale  # (B,H,seg,seg)
        if causal:
            seg_len = hi - lo
            mask = torch.triu(
                torch.ones(seg_len, seg_len, device=q.device, dtype=torch.bool),
                diagonal=1
            )
            scores = scores.masked_fill(mask, float('-inf'))
        attn = torch.softmax(scores, dim=-1)
        out_intra[:, :, lo:hi, :] = torch.matmul(attn, vc)

    # ── Stage 2: cross-segment sparse attention ────────────────────────────
    # Build representative token table: one (k_rep, v_rep) per segment
    # Shape of each: (B, H, 1, D)
    reps_k = []
    reps_v = []
    for lo, hi in segments:
        rk, rv = _segment_representative(k, v, lo, hi, rep_mode)
        reps_k.append(rk)
        reps_v.append(rv)

    # Cross-segment KV: (B, H, num_segments, D)
    cross_k = torch.cat(reps_k, dim=2)
    cross_v = torch.cat(reps_v, dim=2)

    out_cross = torch.zeros_like(q)
    for seg_idx, (lo, hi) in enumerate(segments):
        qc = q[:, :, lo:hi, :]                         # (B, H, seg_len, D)

        # Each query segment attends to ALL segment reps EXCEPT its own
        # (attending to own rep is already covered by intra-segment attention)
        other_mask = [i for i in range(num_segments) if i != seg_idx]
        if not other_mask:
            continue

        ck = cross_k[:, :, other_mask, :]              # (B, H, k_other, D)
        cv = cross_v[:, :, other_mask, :]

        scores = torch.matmul(qc, ck.transpose(-2, -1)) / scale  # (B,H,seg,k_other)

        # Causal cross-segment masking:
        # A query token at absolute position (lo + i) can only attend to a
        # cross-segment representative whose segment ends BEFORE that position.
        if causal:
            seg_len = hi - lo
            n_other = len(other_mask)
            cross_causal = torch.zeros(seg_len, n_other,
                                       device=q.device, dtype=torch.bool)
            for j, other_seg_idx in enumerate(other_mask):
                other_lo, other_hi = segments[other_seg_idx]
                rep_pos = (other_lo + other_hi) // 2   # representative position
                # query position lo+i can attend only if rep_pos <= lo+i
                for i in range(seg_len):
                    if rep_pos > lo + i:
                        cross_causal[i, j] = True      # mask out future reps
            scores = scores.masked_fill(cross_causal.unsqueeze(0).unsqueeze(0),
                                        float('-inf'))

        attn = torch.softmax(scores, dim=-1)
        # Zero out any NaN from all-inf rows (query can't attend to anything)
        attn = torch.nan_to_num(attn, nan=0.0)
        out_cross[:, :, lo:hi, :] = torch.matmul(attn, cv)

    # ── Combine intra + cross ──────────────────────────────────────────────
    return out_intra + cross_weight * out_cross