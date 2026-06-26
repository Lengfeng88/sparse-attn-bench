"""
Hierarchical Chunk Attention (HCA) — benchmarking reference implementation.

L0: chunk-local causal full attention          O(n·c)
L1: per-token Q vs mean-pooled chunk K/V       O(n · n/c)
Output = L0_out + alpha * L1_out
"""
import math
import torch


def hca_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    chunk_size: int = 64,
    alpha: float = 0.5,
    causal: bool = True,
) -> torch.Tensor:
    B, H, S, D = q.shape
    scale = math.sqrt(D)
    n_chunks = (S + chunk_size - 1) // chunk_size
    chunks = [(i * chunk_size, min((i + 1) * chunk_size, S)) for i in range(n_chunks)]

    # L0: chunk-local causal attention
    out_l0 = torch.zeros_like(q)
    for lo, hi in chunks:
        qc, kc, vc = q[:, :, lo:hi, :], k[:, :, lo:hi, :], v[:, :, lo:hi, :]
        sc = torch.matmul(qc, kc.transpose(-2, -1)) / scale
        if causal:
            clen = hi - lo
            sc = sc.masked_fill(
                torch.triu(torch.ones(clen, clen, device=q.device, dtype=torch.bool), diagonal=1),
                float("-inf"),
            )
        out_l0[:, :, lo:hi, :] = torch.matmul(
            torch.softmax(sc.float(), dim=-1).to(q.dtype), vc
        )

    if alpha == 0.0:
        return out_l0

    # L1: per-token Q attends to mean-pooled chunk K/V summaries
    k_chunks = torch.stack([k[:, :, lo:hi, :].mean(2) for lo, hi in chunks], dim=2)
    v_chunks = torch.stack([v[:, :, lo:hi, :].mean(2) for lo, hi in chunks], dim=2)

    scores_l1 = torch.matmul(q, k_chunks.transpose(-2, -1)) / scale  # (B,H,S,C)

    if causal:
        token_pos  = torch.arange(S, device=q.device)
        chunk_ends = torch.tensor([hi for _, hi in chunks], device=q.device)
        future_mask = chunk_ends.unsqueeze(0) > token_pos.unsqueeze(1)  # (S,C)
        scores_l1 = scores_l1.masked_fill(
            future_mask.unsqueeze(0).unsqueeze(0), float("-inf")
        )

    attn_l1 = torch.softmax(scores_l1.float(), dim=-1).to(q.dtype)
    attn_l1 = torch.nan_to_num(attn_l1, nan=0.0)
    out_l1  = torch.matmul(attn_l1, v_chunks)

    return out_l0 + alpha * out_l1
