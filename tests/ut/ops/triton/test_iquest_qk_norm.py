import unittest

import torch

from tests.ut.base import TestBase


def _npu_available() -> bool:
    try:
        import torch_npu  # noqa: F401

        return torch.npu.is_available()
    except Exception:
        return False


def _reference(qkv, q_gamma, k_gamma, q_heads, kv_heads, head_dim, eps):
    """Per-head RMSNorm on the [q | k | v] row layout, fp32 accumulation."""
    t = qkv.shape[0]
    q_size = q_heads * head_dim
    kv_size = kv_heads * head_dim
    q, k, _ = qkv.split([q_size, kv_size, kv_size], dim=-1)

    def norm(x, g):
        xf = x.float().view(t, -1, head_dim)
        var = xf.pow(2).mean(-1, keepdim=True)
        return (xf * torch.rsqrt(var + eps) * g.float()).to(x.dtype)

    return norm(q, q_gamma).reshape(t, q_size), norm(k, k_gamma).reshape(t, kv_size)


@unittest.skipUnless(_npu_available(), "requires Ascend NPU (Triton-on-Ascend kernel)")
class TestStridedQKNorm(TestBase):

    def test_matches_reference(self):
        from vllm_ascend.ops.triton.iquest_qk_norm import strided_qk_norm

        torch.manual_seed(0)
        for t, q_heads, kv_heads, dtype in [
            (8, 6, 1, torch.bfloat16),
            (96, 6, 1, torch.bfloat16),
            (77, 3, 1, torch.float16),
        ]:
            with self.subTest(t=t, q_heads=q_heads, kv_heads=kv_heads, dtype=dtype):
                head_dim = 128
                row = q_heads * head_dim + 2 * kv_heads * head_dim
                qkv = torch.randn(t, row, dtype=dtype, device="npu:0")
                qg = torch.randn(head_dim, dtype=dtype, device="npu:0")
                kg = torch.randn(head_dim, dtype=dtype, device="npu:0")
                eps = 1e-6

                q_out, k_out = strided_qk_norm(qkv, qg, kg, q_heads, kv_heads,
                                               head_dim, eps)
                torch.npu.synchronize()
                q_ref, k_ref = _reference(qkv, qg, kg, q_heads, kv_heads,
                                          head_dim, eps)
                # fp32-accumulated math, one output rounding: < 1 bf16 quantum
                self.assertLess((q_out.float() - q_ref.float()).abs().max().item(),
                                0.02)
                self.assertLess((k_out.float() - k_ref.float()).abs().max().item(),
                                0.02)
                self.assertTrue(q_out.is_contiguous())
                self.assertTrue(k_out.is_contiguous())

    def test_rejects_narrow_row(self):
        from vllm_ascend.ops.triton.iquest_qk_norm import strided_qk_norm

        qkv = torch.zeros(4, 100, dtype=torch.bfloat16, device="npu:0")
        g = torch.zeros(128, dtype=torch.bfloat16, device="npu:0")
        with self.assertRaises(AssertionError):
            strided_qk_norm(qkv, g, g, 6, 1, 128, 1e-6)


if __name__ == "__main__":
    unittest.main()
