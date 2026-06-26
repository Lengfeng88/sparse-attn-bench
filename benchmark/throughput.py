# benchmark/throughput.py
import torch, sys, argparse
sys.path.insert(0, '.')
from attention.dense import dense_attention
from attention.dsa   import dsa_attention
from attention.hca   import hca_attention
from attention.csa   import csa_attention

WARMUP = 5
ITERS  = 20

def measure(name, fn, seq_len, dtype):
    B, H, D = 1, 8, 64
    torch.cuda.empty_cache()
    try:
        q = torch.randn(B, H, seq_len, D, dtype=dtype, device='cuda')
        k = torch.randn_like(q)
        v = torch.randn_like(q)
        for _ in range(WARMUP):
            fn(q, k, v)
        torch.cuda.synchronize()
        t0 = torch.cuda.Event(enable_timing=True)
        t1 = torch.cuda.Event(enable_timing=True)
        t0.record()
        for _ in range(ITERS):
            fn(q, k, v)
        t1.record()
        torch.cuda.synchronize()
        ms_per_iter = t0.elapsed_time(t1) / ITERS
        toks_per_sec = int(B * seq_len / (ms_per_iter / 1000))
        print(f"  {name:8s}  seq={seq_len:6d}  {ms_per_iter:7.2f} ms/iter  {toks_per_sec:>10,} tok/s")
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
