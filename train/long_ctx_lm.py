# train/long_ctx_lm.py
import torch, torch.nn as nn, argparse, sys
sys.path.insert(0, '..')
from attention.dense import dense_attention
from attention.dsa   import dsa_attention
from attention.hca   import hca_attention

ATTN_FN = {
    'dense': dense_attention,
    'dsa':   lambda q,k,v: dsa_attention(q,k,v,window_size=256,global_tokens=32),
    'hca':   lambda q,k,v: hca_attention(q,k,v,chunk_size=64),
}

class SparseAttentionLayer(nn.Module):
    def __init__(self, d_model, n_heads, attn_type):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.out = nn.Linear(d_model, d_model, bias=False)
        self.attn_fn = ATTN_FN[attn_type]

    def forward(self, x):
        B, S, C = x.shape
        qkv = self.qkv(x).reshape(B, S, 3, self.n_heads, self.head_dim)
        q, k, v = qkv.unbind(2)
        q, k, v = [t.transpose(1, 2) for t in (q, k, v)]  # (B,H,S,D)
        out = self.attn_fn(q, k, v)
        out = out.transpose(1, 2).reshape(B, S, C)
        return self.out(out)

class MiniGPT(nn.Module):
    def __init__(self, vocab=256, d_model=256, n_heads=4, n_layers=2, attn_type='dense'):
        super().__init__()
        self.embed = nn.Embedding(vocab, d_model)
        self.layers = nn.ModuleList([
            nn.Sequential(
                SparseAttentionLayer(d_model, n_heads, attn_type),
                nn.LayerNorm(d_model),
                nn.Linear(d_model, d_model * 4),
                nn.GELU(),
                nn.Linear(d_model * 4, d_model),
                nn.LayerNorm(d_model),
            ) for _ in range(n_layers)
        ])
        self.head = nn.Linear(d_model, vocab, bias=False)

    def forward(self, x):
        h = self.embed(x)
        for layer in self.layers:
            h = layer(h)
        return self.head(h)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--attn_type', default='dense', choices=ATTN_FN.keys())
    parser.add_argument('--seq_len',   type=int, default=4096)
    parser.add_argument('--steps',     type=int, default=100)
    parser.add_argument('--bf16',      action='store_true')
    args = parser.parse_args()

    device = 'cuda'
    dtype  = torch.bfloat16 if args.bf16 else torch.float32
    model  = MiniGPT(attn_type=args.attn_type).to(device)

    # gradient checkpointing
    for layer in model.layers:
        layer[0] = torch.utils.checkpoint.checkpoint_wrapper(layer[0])

    opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
    print(f"attn={args.attn_type}  seq={args.seq_len}  bf16={args.bf16}")

    for step in range(args.steps):
        x = torch.randint(0, 256, (2, args.seq_len), device=device)
        with torch.cuda.amp.autocast(enabled=args.bf16, dtype=dtype):
            logits = model(x)
            loss = nn.functional.cross_entropy(
                logits[:, :-1].reshape(-1, 256),
                x[:, 1:].reshape(-1)
            )
        loss.backward()
        opt.step(); opt.zero_grad()
        if step % 10 == 0:
            mem = torch.cuda.max_memory_allocated() / 1024**3
            print(f"  step={step:3d}  loss={loss.item():.4f}  mem={mem:.2f}GB")

    print("Done.")