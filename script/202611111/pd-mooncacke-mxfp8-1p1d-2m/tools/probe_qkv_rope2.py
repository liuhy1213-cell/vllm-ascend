import torch
import torch_npu

dev = "npu:0"
B, HQ, HKV, D = 8, 48, 8, 128
q_size, kv_size = HQ * D, HKV * D
torch.manual_seed(0)

qkv = torch.randn(B, q_size + 2 * kv_size, dtype=torch.bfloat16, device=dev)
qw = torch.randn(D, dtype=torch.float32, device=dev)
kw = torch.randn(D, dtype=torch.float32, device=dev)
cos = torch.randn(B, D, dtype=torch.float32, device=dev)
sin = torch.randn(B, D, dtype=torch.float32, device=dev)
slots = torch.arange(B, dtype=torch.int64, device=dev)
q_out = torch.empty(B, q_size, dtype=torch.bfloat16, device=dev)
cache_k = torch.randn(2048, HKV, D, dtype=torch.bfloat16, device=dev)
cache_v = torch.randn(2048, HKV, D, dtype=torch.bfloat16, device=dev)

op = torch_npu.npu_qkv_rms_norm_rope_cache
trials = [
    ("qkv_size=[B,HKV,D,3] head=[HQ,HKV,D]", [B, HKV, D, 3], [HQ, HKV, D]),
    ("qkv_size=[B,1,HKV,D] head=[HQ,HKV,D]", [B, 1, HKV, D], [HQ, HKV, D]),
    ("qkv_size=[q,kv,kv,D] head=[HQ,HKV,D]", [q_size, kv_size, kv_size, D], [HQ, HKV, D]),
    ("qkv_size=[B,N,S,D] head=[HQ,HKV,0]", [B, 1, HKV, D], [HQ, HKV, 0]),
]
for label, qs, hn in trials:
    try:
        out = op(qkv, qw, kw, cos, sin, slots, q_out, cache_k, cache_v,
                 qkv_size=qs, head_nums=hn, epsilon=1e-6, cache_mode="PA")
        print("OK", label, [tuple(o.shape) if hasattr(o, "shape") else o for o in out] if isinstance(out, (list, tuple)) else out)
        break
    except Exception as e:
        print("ERR", label, "::", str(e).splitlines()[0][:280])
