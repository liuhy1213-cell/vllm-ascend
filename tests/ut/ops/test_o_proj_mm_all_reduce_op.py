import unittest
from unittest import mock
from unittest.mock import MagicMock, patch

import torch

from tests.ut.base import TestBase
from vllm_ascend.ops.linear_op import OProjMMAllReduceOp

# The op checks scheme membership by class name (import-cycle avoidance),
# so the fake quant method must carry the real class name.
class AscendW8A8DynamicLinearMethod:

    act_quant_type = torch.int8

    def __init__(self):
        self.apply = MagicMock()


def make_valid_layer():
    layer = MagicMock()
    layer.custom_op = None
    quant_method = AscendW8A8DynamicLinearMethod()
    layer.quant_method = quant_method
    layer.bias = None
    weight = MagicMock()
    weight.ndim = 2
    weight.dtype = torch.int8
    weight.shape = (384, 3072)
    layer.weight = weight
    layer.weight_scale_fp32 = MagicMock()
    layer.input_is_parallel = True
    layer.reduce_results = True
    layer.skip_bias_add = False
    return layer


class TestOProjMMAllReduceOp(TestBase):
    def setUp(self):
        self.mock_group = mock.MagicMock()
        self.mock_group.world_size = 8
        self.mock_group.rank_in_group = 0
        self.patches = [
            patch("vllm_ascend.ops.linear_op.get_tp_group", return_value=self.mock_group),
            # Ensure the op-availability gate passes even when torch_npu loads
            # without its device backend (e.g. CPU-only UT environments).
            patch("torch_npu.npu_mm_all_reduce_base", MagicMock(), create=True),
        ]
        for p in self.patches:
            p.start()

    def tearDown(self):
        for p in self.patches:
            p.stop()

    def test_unsupported_reason_accepts_valid_layer(self):
        layer = make_valid_layer()
        self.assertIsNone(OProjMMAllReduceOp.unsupported_reason(layer))

    def test_unsupported_reason_rejects_other_scheme(self):
        layer = make_valid_layer()
        layer.quant_method = type("SomeOtherMethod", (), {"act_quant_type": torch.int8})()
        reason = OProjMMAllReduceOp.unsupported_reason(layer)
        self.assertIn("W8A8_DYNAMIC", reason)

    def test_unsupported_reason_rejects_bias(self):
        layer = make_valid_layer()
        layer.bias = MagicMock()
        reason = OProjMMAllReduceOp.unsupported_reason(layer)
        self.assertIn("bias", reason)

    def test_unsupported_reason_rejects_bad_tp_size(self):
        self.mock_group.world_size = 16
        layer = make_valid_layer()
        reason = OProjMMAllReduceOp.unsupported_reason(layer)
        self.assertIn("TP size", reason)

    def test_unsupported_reason_rejects_non_int8_weight(self):
        layer = make_valid_layer()
        layer.weight.dtype = torch.bfloat16
        reason = OProjMMAllReduceOp.unsupported_reason(layer)
        self.assertIn("int8", reason)

    def test_unsupported_reason_rejects_existing_custom_op(self):
        layer = make_valid_layer()
        layer.custom_op = MagicMock()
        reason = OProjMMAllReduceOp.unsupported_reason(layer)
        self.assertIn("original TP group", reason)

    def test_init_builds_hcom(self):
        layer = make_valid_layer()
        op = OProjMMAllReduceOp(layer)
        self.assertIsNotNone(op.hcom)

    def test_apply_impl_fused_path(self):
        layer = make_valid_layer()
        op = OProjMMAllReduceOp(layer)
        x = torch.randn(8, 384)
        quantized = torch.zeros(8, 384, dtype=torch.int8)
        scale = torch.ones(8)
        fused_out = torch.randn(8, 3072)

        with (
            mock.patch("torch_npu.npu_dynamic_quant", return_value=(quantized, scale)) as dq,
            mock.patch("torch_npu.npu_mm_all_reduce_base", return_value=fused_out) as mm_ar,
        ):
            out, bias = op.apply_impl(x)

        self.assertIs(out, fused_out)
        self.assertIsNone(bias)
        dq.assert_called_once()
        layer.quant_method.apply.assert_not_called()
        mm_ar.assert_called_once_with(
            quantized,
            layer.weight,
            op.hcom,
            dequant_scale=layer.weight_scale_fp32,
            pertoken_scale=scale,
        )

    def test_apply_impl_falls_back_on_3d_input(self):
        layer = make_valid_layer()
        op = OProjMMAllReduceOp(layer)
        x = torch.randn(2, 4, 384)
        unfused_out = torch.randn(2, 4, 3072)
        layer.quant_method.apply.return_value = unfused_out

        with (
            mock.patch("torch_npu.npu_dynamic_quant") as dq,
            mock.patch("torch_npu.npu_mm_all_reduce_base") as mm_ar,
        ):
            out, bias = op.apply_impl(x)

        self.assertIs(out, self.mock_group.all_reduce.return_value)
        self.assertIsNone(bias)
        dq.assert_not_called()
        mm_ar.assert_not_called()
        layer.quant_method.apply.assert_called_once()
        self.mock_group.all_reduce.assert_called_once_with(unfused_out)

    def test_apply_impl_falls_back_on_k_mismatch(self):
        layer = make_valid_layer()
        op = OProjMMAllReduceOp(layer)
        x = torch.randn(8, 128)  # K != weight.shape[0]
        unfused_out = torch.randn(8, 3072)
        layer.quant_method.apply.return_value = unfused_out

        with (
            mock.patch("torch_npu.npu_dynamic_quant") as dq,
            mock.patch("torch_npu.npu_mm_all_reduce_base") as mm_ar,
        ):
            out, _ = op.apply_impl(x)

        self.assertIs(out, self.mock_group.all_reduce.return_value)
        dq.assert_not_called()
        mm_ar.assert_not_called()


if __name__ == "__main__":
    unittest.main()
