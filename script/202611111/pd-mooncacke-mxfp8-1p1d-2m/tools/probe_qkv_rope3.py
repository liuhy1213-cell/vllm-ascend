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
q_out = torch.zeros(B, q_size, dtype=torch.bfloat16, device=dev)
cache_k = torch.randn(2048, HKV, D, dtype=torch.bfloat16, device=dev)
cache_v = torch.randn(2048, HKV, D, dtype=torch.bfloat16, device=dev)
snapshot = cache_k[:B].clone()

op = torch_npu.npu_qkv_rms_norm_rope_cache
out = op(qkv, qw, kw, cos, sin, slots, q_out, cache_k, cache_v,
         qkv_size=[B, HKV, D, 3], head_nums=[HQ, HKV, D],
         epsilon=1e-6, cache_mode="PA")
torch.npu.synchronize()
print("call returned:", out)
print("q_out finite:", torch.isfinite(q_out).all().item(),
      "q_out nonzero:", q_out.abs().sum().item() > 0)
print("cache rows changed:", not torch.equal(cache_k[:B], snapshot))

# ---- reference: split -> per-head rmsnorm -> NeoX rope ----
def rms_norm_head(x, g):  # x [B, H, D], g [D]
    xf = x.float()
    var = xf.pow(2).mean(-1, keepdim=True)
    return (xf * torch.rsqrt(var + 1e-6) * g.float()).to(x.dtype)

def rope_neox(x, cos, sin):  # x [B, H, D], cos/sin [B, D]
    x1, x2 = x[..., : D // 2], x[..., D // 2:]
    c1, c2 = cos[:, : D // 2].unsqueeze(1), cos[:, D // 2:].unsqueeze(1)
    s1, s2 = sin[:, : D // 2].unsqueeze(1), sin[:, D // 2:].unsqueeze(1)
    return torch.cat([x1 * c1 - x2 * s1, x1 * s2 + x2 * c2], dim=-1)

q_r, k_r, v_r = qkv.split([q_size, kv_size, kv_size], dim=-1)
q_h = rms_norm_head(q_r.view(B, HQ, D), qw)
k_h = rms_norm_head(k_r.view(B, HKV, D), kw)
q_ref = rope_neox(q_h, cos, sin).reshape(B, q_size)
k_ref = rope_neox(k_h, cos, sin).reshape(B, kv_size)

q_err = (q_out.float() - q_ref.float()).abs().max().item()
k_err = (cache_k[:B].float() - k_ref.view(B, HKV, D).float()).abs().max().item()
print(f"q maxerr {q_err:.4e}  k(cache) maxerr {k_err:.4e}")
print("v written:", bool((cache_v[:B] != 0).any().item()))
