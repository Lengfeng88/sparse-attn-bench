# attention/dense.py
import torch, math

def dense_attention(q, k, v, causal=True):
    """标准 scaled dot-product attention，手写版（不用 F.sdpa）。
    q/k/v: (B, H, S, D)
    """
    scale = math.sqrt(q.shape[-1])
    scores = torch.matmul(q, k.transpose(-2, -1)) / scale  # (B,H,S,S)
    if causal:
        mask = torch.triu(torch.ones(scores.shape[-2], scores.shape[-1],
                          device=q.device), diagonal=1).bool()
        scores = scores.masked_fill(mask, float('-inf'))
    attn = torch.softmax(scores, dim=-1)
    return torch.matmul(attn, v)