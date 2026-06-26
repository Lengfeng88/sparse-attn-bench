# benchmark/memory.py
import torch, gc, sys, argparse
sys.path.insert(0, '.')
from attention.dense import dense_attention
from attention.dsa   import dsa_attention
from attention.hca   import hca_attention
from attention.csa   import csa_attention

def measure(name, fn, seq_len, dtype=torch.bfloat16):
    B, H, D = 1, 8, 64
    torch.cuda.empty_cache(); gc.collect()
    torch.cuda.reset_peak_memory_stats()
    try:
        q = torch.randn(B, H, seq_len, D, dtype=dtype, device='cuda', requires_grad=True)
        k = torch.randn(B, H, seq_len, D, dtype=dtype, device='cuda', requires_grad=True)
        v = torch.randn(B, H, seq_len, D, dtype=dtype, device='cuda', requires_grad=True)
        with torch.amp.autocast('cuda', dtype=dtype):
            out = fn(q, k, v)
        out.sum().backward()
        mem = torch.cuda.max_memory_allocated() / 1024**3
        print(f"  {name:8s}  seq={seq_len:6d}  mem={mem:.2f} GB")
    except torch.cuda.OutOfMemoryError:
        print(f"  {name:8s}  seq={seq_len:6d}  OOM")
    finally:
        del q, k, v
        torch.cuda.empty_cache()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dtype', choices=['fp16','bf16','fp32'], default='bf16')
    args = parser.parse_args()
    dtype_map = {'fp16': torch.float16, 'bf16': torch.bfloat16, 'fp32': torch.float32}
    dtype = dtype_map[args.dtype]

    for seq_len in [2048, 4096, 8192, 16384, 32768]:
        print(f"\n--- seq_len={seq_len} ---")
        measure("Dense", dense_attention, seq_len, dtype)
        measure("DSA",   lambda q,k,v: dsa_attention(q,k,v,window_size=256,global_tokens=32), seq_len, dtype)
        measure("CSA",   lambda q,k,v: csa_attention(q,k,v,num_segments=8), seq_len, dtype)
        measure("HCA",   lambda q,k,v: hca_attention(q,k,v,chunk_size=64), seq_len, dtype)
