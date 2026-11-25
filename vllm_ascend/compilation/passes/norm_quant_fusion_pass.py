#
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
# This file is a part of the vllm-ascend project.
#
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
import operator

import torch
from torch._inductor.pattern_matcher import (
    MULTIPLE,
    CallFunction,
    KeywordArg,
    MultiOutputPattern,
    PatternMatcherPass,
)
from vllm.compilation.passes.vllm_inductor_pass import VllmInductorPass
from vllm.config import VllmConfig
from vllm.config.compilation import Range
from vllm.logger import logger

from vllm_ascend.compilation.passes.base_pattern import BasePattern
from vllm_ascend.device.hardware_profile import HardwareCapability, get_current_hardware_profile
from vllm_ascend.utils import enable_custom_op


class AddRMSNormQuantPattern(BasePattern):
    def __init__(self, vllm_config: VllmConfig, eps: float = 1e-6):
        super().__init__(vllm_config, eps)

    def get_inputs(self):
        """
        Generate example inputs for the AddRMSNormQuant fusion pattern.
        """
        rms_norm_input = torch.randn(2, 4, device="npu", dtype=self.dtype)
        residual = torch.randn(2, 4, device="npu", dtype=self.dtype)
        rms_norm_weight = torch.randn(4, device="npu", dtype=self.dtype)
        scale = torch.ones(4, device="npu", dtype=self.dtype)
        scale_reciprocal = torch.ones(4, device="npu", dtype=self.dtype)
        offset = torch.zeros(4, device="npu", dtype=self.dtype)
        return [rms_norm_input, residual, rms_norm_weight, scale, scale_reciprocal, offset]

    def get_pattern(self):
        def pattern(
            rms_norm_input: torch.Tensor,
            residual: torch.Tensor,
            rms_norm_weight: torch.Tensor,
            scale: torch.Tensor,
            scale_reciprocal: torch.Tensor,
            offset: torch.Tensor,
        ):
            """
            Pattern for AddRMSNormQuant fusion.
            """
            output = torch.ops._C_ascend.npu_add_rms_norm_bias(
                rms_norm_input, residual, rms_norm_weight, None, self.eps
            )
            out0 = output[0]
            out1 = output[2]
            quantized_output = torch.ops.vllm.quantize(out0, scale, scale_reciprocal, offset)
            return quantized_output, out1

        return pattern

    def get_replacement(self):
        def replacement(
            rms_norm_input: torch.Tensor,
            residual: torch.Tensor,
            rms_norm_weight: torch.Tensor,
            scale: torch.Tensor,
            scale_reciprocal: torch.Tensor,
            offset: torch.Tensor,
        ):
            """
            Replacement for the AddRMSNormQuant fusion.
            """
            output = torch.ops.npu.npu_add_rms_norm_quant(
                rms_norm_input, residual, rms_norm_weight, scale, offset, epsilon=self.eps
            )
            quantized_output = output[0]
            out1 = output[2]
            return quantized_output, out1

        return replacement


class AddRMSNormQuantPatternWithBias(BasePattern):
    def __init__(self, vllm_config: VllmConfig, eps: float = 1e-6):
        super().__init__(vllm_config, eps)

    def get_inputs(self):
        """
        Generate example inputs for the AddRMSNormQuant fusion pattern.
        """
        rms_norm_input = torch.randn(2, 4, device="npu", dtype=self.dtype)
        residual = torch.randn(2, 4, device="npu", dtype=self.dtype)
        rms_norm_weight = torch.randn(4, device="npu", dtype=self.dtype)
        rmsnorm_bias = torch.randn(4, device="npu", dtype=self.dtype)
        scale = torch.ones(4, device="npu", dtype=self.dtype)
        scale_reciprocal = torch.ones(4, device="npu", dtype=self.dtype)
        offset = torch.zeros(4, device="npu", dtype=self.dtype)
        return [rms_norm_input, residual, rms_norm_weight, scale, scale_reciprocal, offset, rmsnorm_bias]

    def get_pattern(self):
        def pattern(
            rms_norm_input: torch.Tensor,
            residual: torch.Tensor,
            rms_norm_weight: torch.Tensor,
            scale: torch.Tensor,
            scale_reciprocal: torch.Tensor,
            offset: torch.Tensor,
            bias: torch.Tensor,
        ):
            """
            Pattern for AddRMSNormQuant fusion.
            """
            output = torch.ops._C_ascend.npu_add_rms_norm_bias(
                rms_norm_input, residual, rms_norm_weight, bias, self.eps
            )
            out0 = output[0]
            out1 = output[2]
            quantized_output = torch.ops.vllm.quantize(out0, scale, scale_reciprocal, offset)
            return quantized_output, out1

        return pattern

    def get_replacement(self):
        def replacement(
            rms_norm_input: torch.Tensor,
            residual: torch.Tensor,
            rms_norm_weight: torch.Tensor,
            scale: torch.Tensor,
            scale_reciprocal: torch.Tensor,
            offset: torch.Tensor,
            bias: torch.Tensor,
        ):
            """
            Replacement for the AddRMSNormQuant fusion.
            """
            output = torch.ops.npu.npu_add_rms_norm_quant(
                rms_norm_input, residual, rms_norm_weight, scale, offset, epsilon=self.eps, beta=bias
            )
            quantized_output = output[0]
            out1 = output[2]
            return quantized_output, out1

        return replacement


class AddRMSNormDynamicQuantPattern(BasePattern):
    def __init__(self, vllm_config: VllmConfig, eps: float = 1e-6):
        super().__init__(vllm_config, eps)

    def get_inputs(self):
        """
        Generate example inputs for the AddRMSNormQuant fusion pattern.
        """
        rms_norm_input = torch.randn(2, 4, device="npu", dtype=self.dtype)
        residual = torch.randn(2, 4, device="npu", dtype=self.dtype)
        rms_norm_weight = torch.randn(4, device="npu", dtype=self.dtype)
        return [rms_norm_input, residual, rms_norm_weight]

    def get_pattern(self):
        def pattern(rms_norm_input: torch.Tensor, residual: torch.Tensor, rms_norm_weight: torch.Tensor):
            """
            Pattern for AddRMSNormQuant fusion.
            """
            output = torch.ops.npu.npu_add_rms_norm(rms_norm_input, residual, rms_norm_weight, self.eps)
            out0 = output[0]
            out1 = output[2]
            # Pin the rule to INT8: npu_dynamic_quant is shared with the INT4
            # (torch.quint4x2) and FP8 paths, while the fused op below produces
            # INT8. Inductor's matcher ignores a keyword the pattern does not
            # declare, so without this the rule would also match those graphs
            # and replace them with the INT8 fused op.
            quantized_output = torch.ops.npu.npu_dynamic_quant(out0, dst_type=torch.int8)
            return quantized_output[0], quantized_output[1], out1

        return pattern

    def get_replacement(self):
        def replacement(rms_norm_input: torch.Tensor, residual: torch.Tensor, rms_norm_weight: torch.Tensor):
            """
            Replacement for the AddRMSNormQuant fusion.
            """
            output = torch.ops.npu.npu_add_rms_norm_dynamic_quant(
                rms_norm_input, residual, rms_norm_weight, epsilon=self.eps, output_mask=[True, False]
            )
            return (
                output[0],
                output[3],
                output[2],
            )

        return replacement


class AddRMSNormDynamicQuantPatternNoBias(BasePattern):
    """add_rms_norm_bias(x, r, w, None, eps) + npu_dynamic_quant -> fused op.

    Qwen3.5/Qwen3-Next RMSNorm has no beta, so ops/layernorm.py forward_oot takes
    the ``beta=None`` branch of ``_C_ascend.npu_add_rms_norm_bias``. The existing
    AddRMSNormDynamicQuantPattern matches ``torch.ops.npu.npu_add_rms_norm``
    instead, and the WithBias variant requires a tensor bias; neither covers this
    call, so the fusion never applied to these models.
    """

    def __init__(self, vllm_config: VllmConfig, eps: float = 1e-6):
        super().__init__(vllm_config, eps)

    def get_inputs(self):
        rms_norm_input = torch.randn(2, 4, device="npu", dtype=self.dtype)
        residual = torch.randn(2, 4, device="npu", dtype=self.dtype)
        rms_norm_weight = torch.randn(4, device="npu", dtype=self.dtype)
        return [rms_norm_input, residual, rms_norm_weight]

    def get_pattern(self):
        def pattern(rms_norm_input: torch.Tensor, residual: torch.Tensor, rms_norm_weight: torch.Tensor):
            output = torch.ops._C_ascend.npu_add_rms_norm_bias(
                rms_norm_input, residual, rms_norm_weight, None, self.eps
            )
            out0 = output[0]
            out1 = output[2]
            # Pin the rule to INT8: npu_dynamic_quant is shared with the INT4
            # (torch.quint4x2) and FP8 paths, while the fused op below produces
            # INT8. Inductor's matcher ignores a keyword the pattern does not
            # declare, so without this the rule would also match those graphs
            # and replace them with the INT8 fused op.
            quantized_output = torch.ops.npu.npu_dynamic_quant(out0, dst_type=torch.int8)
            return quantized_output[0], quantized_output[1], out1

        return pattern

    def get_replacement(self):
        def replacement(rms_norm_input: torch.Tensor, residual: torch.Tensor, rms_norm_weight: torch.Tensor):
            output = torch.ops.npu.npu_add_rms_norm_dynamic_quant(
                rms_norm_input,
                residual,
                rms_norm_weight,
                beta=None,
                epsilon=self.eps,
                output_mask=[True, False],
            )
            return (
                output[0],
                output[3],
                output[2],
            )

        return replacement


class AddRMSNormDynamicQuantPatternWithBias(BasePattern):
    def __init__(self, vllm_config: VllmConfig, eps: float = 1e-6):
        super().__init__(vllm_config, eps)

    def get_inputs(self):
        """
        Generate example inputs for the AddRMSNormQuant fusion pattern.
        """
        rms_norm_input = torch.randn(2, 4, device="npu", dtype=self.dtype)
        residual = torch.randn(2, 4, device="npu", dtype=self.dtype)
        rms_norm_weight = torch.randn(4, device="npu", dtype=self.dtype)
        rmsnorm_bias = torch.randn(4, device="npu", dtype=self.dtype)
        return [rms_norm_input, residual, rms_norm_weight, rmsnorm_bias]

    def get_pattern(self):
        def pattern(
            rms_norm_input: torch.Tensor,
            residual: torch.Tensor,
            rms_norm_weight: torch.Tensor,
            bias: torch.Tensor,
        ):
            """
            Pattern for AddRMSNormQuant fusion.
            """
            output = torch.ops._C_ascend.npu_add_rms_norm_bias(
                rms_norm_input, residual, rms_norm_weight, bias, self.eps
            )
            out0 = output[0]
            out1 = output[2]
            # Pin the rule to INT8: npu_dynamic_quant is shared with the INT4
            # (torch.quint4x2) and FP8 paths, while the fused op below produces
            # INT8. Inductor's matcher ignores a keyword the pattern does not
            # declare, so without this the rule would also match those graphs
            # and replace them with the INT8 fused op.
            quantized_output = torch.ops.npu.npu_dynamic_quant(out0, dst_type=torch.int8)
            return quantized_output[0], quantized_output[1], out1

        return pattern

    def get_replacement(self):
        def replacement(
            rms_norm_input: torch.Tensor,
            residual: torch.Tensor,
            rms_norm_weight: torch.Tensor,
            bias: torch.Tensor,
        ):
            """
            Replacement for the AddRMSNormQuant fusion.
            """
            output = torch.ops.npu.npu_add_rms_norm_dynamic_quant(
                rms_norm_input, residual, rms_norm_weight, epsilon=self.eps, output_mask=[True, False], beta=bias
            )
            return (
                output[0],
                output[3],
                output[2],
            )

        return replacement


class AddRMSNormDynamicMXQuantPattern(BasePattern):
    def __init__(self, vllm_config: VllmConfig, eps: float = 1e-6):
        super().__init__(vllm_config, eps)

    def get_inputs(self):
        """
        Generate example inputs for the AddRMSNormDynamicMXQuant fusion pattern.
        """
        rms_norm_input = torch.randn(2, 64, device="npu", dtype=self.dtype)
        residual = torch.randn(2, 64, device="npu", dtype=self.dtype)
        rms_norm_weight = torch.randn(64, device="npu", dtype=self.dtype)
        return [rms_norm_input, residual, rms_norm_weight]

    def get_pattern(self):
        def pattern(rms_norm_input: torch.Tensor, residual: torch.Tensor, rms_norm_weight: torch.Tensor):
            """
            Pattern for AddRMSNormDynamicMXQuant fusion.
            """
            output = torch.ops.npu.npu_add_rms_norm(rms_norm_input, residual, rms_norm_weight, self.eps)
            out0 = output[0]
            out1 = output[2]
            quantized_output = torch.ops.npu.npu_dynamic_mx_quant(out0, dst_type=torch.float8_e4m3fn)
            return quantized_output[0], quantized_output[1], out1

        return pattern

    def get_replacement(self):
        def replacement(rms_norm_input: torch.Tensor, residual: torch.Tensor, rms_norm_weight: torch.Tensor):
            """
            Replacement for the AddRMSNormDynamicMXQuant fusion.
            """
            output = torch.ops.npu.npu_add_rms_norm_dynamic_mx_quant(
                rms_norm_input,
                residual,
                rms_norm_weight,
                epsilon=self.eps,
                dst_type=torch.float8_e4m3fn,
            )
            return (
                output[0],
                output[2],
                output[1],
            )

        return replacement


class RMSNormDynamicMXQuantPattern(BasePattern):
    def __init__(self, vllm_config: VllmConfig, eps: float = 1e-6):
        super().__init__(vllm_config, eps)

    def get_inputs(self):
        """
        Generate example inputs for the RMSNormDynamicMXQuant fusion pattern.
        """
        rms_norm_input = torch.randn(2, 64, device="npu", dtype=self.dtype)
        rms_norm_weight = torch.randn(64, device="npu", dtype=self.dtype)
        return [rms_norm_input, rms_norm_weight]

    def get_pattern(self):
        def pattern(rms_norm_input: torch.Tensor, rms_norm_weight: torch.Tensor):
            """
            Pattern for RMSNormDynamicMXQuant fusion.
            """
            output = torch.ops.npu.npu_rms_norm(rms_norm_input, rms_norm_weight, self.eps)
            out0 = output[0]
            quantized_output = torch.ops.npu.npu_dynamic_mx_quant(out0, dst_type=torch.float8_e4m3fn)
            return quantized_output[0], quantized_output[1]

        return pattern

    def get_replacement(self):
        def replacement(rms_norm_input: torch.Tensor, rms_norm_weight: torch.Tensor):
            """
            Replacement for the RMSNormDynamicMXQuant fusion.
            """
            output = torch.ops.npu.npu_rms_norm_dynamic_mx_quant(
                rms_norm_input,
                rms_norm_weight,
                epsilon=self.eps,
                dst_type=torch.float8_e4m3fn,
            )
            return output[0], output[1]

        return replacement


# ``npu::npu_rms_norm``'s schema default. ``make_fx`` drops an explicit
# epsilon argument when it equals this default, so the pattern without the
# epsilon must still be accepted for the eps=1e-6 registration.
_NPU_RMS_NORM_DEFAULT_EPS = 1e-6


def _add_matches_fused_contract(node: torch.fx.Node) -> bool:
    """Whether the add operands fit the fused op's ``x1``/``x2`` contract."""
    x1, x2 = node.args[:2]
    if not isinstance(x1, torch.fx.Node) or not isinstance(x2, torch.fx.Node):
        return False
    value1 = x1.meta.get("val")
    value2 = x2.meta.get("val")
    if value1 is None or value2 is None:
        return False
    return value1.shape == value2.shape and value1.dtype in (torch.float16, torch.bfloat16)


class AddResidualRMSNormDynamicMXQuantPattern(BasePattern):
    """``aten.add`` + ``npu_rms_norm`` + ``npu_dynamic_mx_quant`` -> one kernel.

    IQuest-Q1 applies RMSNorm directly to the residual stream, so the residual
    add at the end of a decoder layer reaches the next layer's RMSNorm as a
    plain ``aten.add`` rather than the ``npu_add_rms_norm`` emitted by the
    generic Ascend RMSNorm. When the normalized output is consumed only by the
    MX activation quant of the following projection, all three ops collapse
    into ``npu_add_rms_norm_dynamic_mx_quant``.

    ``RMSNormDynamicMXQuantPattern`` must not claim this site first: replacing
    the norm+quant pair on its own would strip the RMSNorm that the residual
    add feeds and leave the 3-op fusion unreachable. The fused op returns the
    residual sum, so every other user of the add stays valid; the normalized
    value must have exactly one user (the quant), which is also what allows
    the matched RMSNorm node itself to be erased with the rest of the chain.
    """

    def __init__(self, vllm_config: VllmConfig, eps: float = 1e-6):
        super().__init__(vllm_config, eps)

    def get_inputs(self):
        """
        Generate example inputs for the add+RMSNorm+MX quant fusion pattern.
        """
        residual = torch.randn(2, 64, device="npu", dtype=self.dtype)
        branch_output = torch.randn(2, 64, device="npu", dtype=self.dtype)
        rms_norm_weight = torch.randn(64, device="npu", dtype=self.dtype)
        return [residual, branch_output, rms_norm_weight]

    def get_pattern(self):
        def pattern(residual: torch.Tensor, branch_output: torch.Tensor, rms_norm_weight: torch.Tensor):
            """
            Pattern for the residual-add + RMSNorm + MX quant chain.
            """
            added = torch.ops.aten.add.Tensor(residual, branch_output, alpha=1)
            norm_output = torch.ops.npu.npu_rms_norm(added, rms_norm_weight, self.eps)
            quantized_output = torch.ops.npu.npu_dynamic_mx_quant(
                norm_output[0], dst_type=torch.float8_e4m3fn, scale_alg=0
            )
            return quantized_output[0], quantized_output[1], added

        return pattern

    def get_replacement(self):
        def replacement(residual: torch.Tensor, branch_output: torch.Tensor, rms_norm_weight: torch.Tensor):
            """
            Replacement for the residual-add + RMSNorm + MX quant chain.
            """
            output = torch.ops.npu.npu_add_rms_norm_dynamic_mx_quant(
                residual,
                branch_output,
                rms_norm_weight,
                epsilon=self.eps,
                scale_alg=0,
                dst_type=torch.float8_e4m3fn,
            )
            # (quantized y, mxscale, residual sum) in pattern output order.
            return output[0], output[2], output[1]

        return replacement

    def get_extra_check(self):
        """Validate the input contract of the fused operator.

        The matcher compares operators and argument structure, not shapes,
        dtypes or the integer encoding of enum-like kwargs, while the fused op
        requires ``x1``/``x2`` to have identical fp16/bf16 shapes and always
        emits FP8-E4M3 with the OCP MX scale algorithm. Anything else (a
        broadcast add, another destination dtype, a non-default quant axis)
        must stay on the unfused path.

        ``dst_type`` is deliberately not pinned in the pattern: Dynamo records
        it as a ``torch.dtype`` while ``make_fx`` records it as the integer
        scalar-type id, and only the value check can accept both encodings.
        """
        quant_targets = (
            torch.ops.npu.npu_dynamic_mx_quant,
            torch.ops.npu.npu_dynamic_mx_quant.default,
        )
        norm_targets = (
            torch.ops.npu.npu_rms_norm,
            torch.ops.npu.npu_rms_norm.default,
        )
        fp8_e4m3_ids = (24, 292)  # torch ScalarType id / aclnn dtype code

        def _check_match(match) -> bool:
            add_ok = False
            norm_ok = False
            quant_ok = False
            for node in match.nodes:
                if node.op != "call_function":
                    continue
                if node.target == torch.ops.aten.add.Tensor:
                    if not _add_matches_fused_contract(node):
                        return False
                    add_ok = True
                elif node.target in norm_targets:
                    # make_fx drops an explicit epsilon equal to the schema
                    # default, so the third argument may be absent.
                    node_eps = node.args[2] if len(node.args) > 2 else _NPU_RMS_NORM_DEFAULT_EPS
                    if node_eps != self.eps:
                        return False
                    norm_ok = True
                elif node.target in quant_targets:
                    dst_type = node.kwargs.get("dst_type")
                    if dst_type is None:
                        return False
                    if not (
                        dst_type == torch.float8_e4m3fn
                        or (isinstance(dst_type, int) and dst_type in fp8_e4m3_ids)
                    ):
                        return False
                    if node.kwargs.get("scale_alg", 0) not in (0, None):
                        return False
                    if node.kwargs.get("round_mode", "rint") != "rint":
                        return False
                    if node.kwargs.get("axis", -1) != -1:
                        return False
                    if node.kwargs.get("block_size", 32) != 32:
                        return False
                    quant_ok = True
            return add_ok and norm_ok and quant_ok

        return _check_match

    def _uses_explicit_epsilon(self) -> bool:
        """Whether the searched graph keeps the optional epsilon argument."""
        return True

    def get_search_fn_pattern(self):
        """Build the pattern by hand instead of tracing it with ``make_fx``.

        Two graph conventions must both match:

        * Dynamo records custom-op calls as ``torch.ops.npu.<op>`` packets and
          only unpacks the ``[0]`` element of a tuple return, so the RMSNorm
          node has a single user;
        * ``make_fx`` (AOTAutograd) records the resolved ``<op>.default``
          overload and materializes a ``getitem`` for every tuple element, so
          the RMSNorm node has two users.

        Passing the packet as the target keeps the packet and all of its
        overloads in the matcher's target set, and ``_users=MULTIPLE`` on the
        quantized output tolerates graphs that skip the scale unpack. The
        RMSNorm output stays strict: ``getitem(norm, 0)`` requires its default
        single user, so a second consumer of the normalized value (an
        unquantized router input, or a saved ``rstd`` in a joint graph) keeps
        the site unfused rather than leaving a live RMSNorm next to the fused
        kernel.

        ``dst_type``/``scale_alg``/``epsilon`` are checked on the matched node
        rather than pinned here; see ``get_extra_check`` and
        ``_uses_explicit_epsilon``.
        """
        residual = KeywordArg("residual")
        branch_output = KeywordArg("branch_output")
        rms_norm_weight = KeywordArg("rms_norm_weight")

        added = CallFunction(torch.ops.aten.add.Tensor, residual, branch_output, alpha=1)
        norm_args = (added, rms_norm_weight, self.eps) if self._uses_explicit_epsilon() else (added, rms_norm_weight)
        norm_output = CallFunction(torch.ops.npu.npu_rms_norm, *norm_args)
        norm_value = CallFunction(operator.getitem, norm_output, 0)
        quantized_output = CallFunction(torch.ops.npu.npu_dynamic_mx_quant, norm_value, _users=MULTIPLE)
        quantized_value = CallFunction(operator.getitem, quantized_output, 0)
        mx_scale = CallFunction(operator.getitem, quantized_output, 1)
        return MultiOutputPattern([quantized_value, mx_scale, added])


class AddResidualRMSNormDynamicMXQuantImplicitEpsPattern(AddResidualRMSNormDynamicMXQuantPattern):
    """The same fusion for graphs where ``make_fx`` dropped the default eps.

    ``make_fx`` omits a positional argument equal to the op schema default, so
    an eps=1e-6 RMSNorm arrives with two arguments there while Dynamo keeps
    three. The matcher compares argument counts, hence one variant per form.
    Registering this variant is only valid for the schema default.
    """

    def _uses_explicit_epsilon(self) -> bool:
        return False


def _model_uses_w4a4_quant(vllm_config: VllmConfig | None) -> bool:
    """Check whether the model uses W4A4 int4 quantization for any layer.

    W4A4 int4 schemes (e.g. W4A4_DYNAMIC, W4A4_FLATQUANT_DYNAMIC)
    are incompatible with the fuse_norm_quant optimization,
    so callers use this to disable that fusion.
    """
    if vllm_config is None:
        return False
    quant_config = getattr(vllm_config, "quant_config", None)
    if quant_config is None:
        return False
    quant_description = getattr(quant_config, "quant_description", None)
    if not quant_description:
        return False
    w4a4_int4_schemes = ["W4A4_DYNAMIC", "W4A4_FLATQUANT_DYNAMIC"]
    return any(
        isinstance(quant_type, str) and quant_type in w4a4_int4_schemes for quant_type in quant_description.values()
    )


class AddRMSNormQuantFusionPass(VllmInductorPass):
    """Fuse RMSNorm with the dynamic activation quantize on Ascend.

    Two families are registered:

    * INT8 patterns replace ``npu_add_rms_norm``/``npu_add_rms_norm_bias``
      followed by an int8 dynamic quantize (the W8A8 and W4A8 activation
      paths). ``_model_uses_w4a4_quant()`` disables them for W4A4 models, and
      other activation dtypes (fp8, int4) do not match, so those models keep
      the unfused path instead of being fused incorrectly;
    * on hardware with ``DYNAMIC_MX_QUANT_FUSION`` (A5), MXFP8 patterns fold
      ``npu_dynamic_mx_quant`` (FP8-E4M3, 32-element blocks) into
      ``npu_rms_norm_dynamic_mx_quant``, including the residual-add chain
      where the add reaches the RMSNorm as a plain ``aten.add``.
    """

    def __init__(self, vllm_config: VllmConfig, fuse_add_rms_norm_mx_quant: bool = True):
        super().__init__(vllm_config)
        self.pattern_match_passes: PatternMatcherPass = PatternMatcherPass(pass_name="rmsnorm_quant_fusion_pass")

        dtype = vllm_config.model_config.dtype
        if dtype not in (torch.bfloat16, torch.float16):
            logger.debug("Quant fusion not enabled: unsupported dtype %s", dtype)
            return

        if _model_uses_w4a4_quant(vllm_config):
            logger.debug(
                "Quant fusion not enabled: the model contains "
                "W4A4 quantized weights, which are incompatible with the "
                "norm-quant fusion pass."
            )
            return

        common_epsilons = [1e-5, 1e-6]
        profile = get_current_hardware_profile()

        for eps in common_epsilons:
            AddRMSNormDynamicQuantPattern(vllm_config, eps=eps).register(self.pattern_match_passes)
            if profile.supports(HardwareCapability.DYNAMIC_MX_QUANT_FUSION):
                # The three-op rules must be registered before the two-op
                # RMSNorm+quant rule, otherwise the latter strips the RMSNorm
                # and the residual add cannot be folded anymore. The dedicated
                # switch lets users keep the two-op MX fusion but skip the
                # residual-add variant.
                if fuse_add_rms_norm_mx_quant:
                    AddResidualRMSNormDynamicMXQuantPattern(vllm_config, eps=eps).register(
                        self.pattern_match_passes
                    )
                    if eps == _NPU_RMS_NORM_DEFAULT_EPS:
                        AddResidualRMSNormDynamicMXQuantImplicitEpsPattern(
                            vllm_config, eps=eps
                        ).register(self.pattern_match_passes)
                AddRMSNormDynamicMXQuantPattern(vllm_config, eps=eps).register(self.pattern_match_passes)
                RMSNormDynamicMXQuantPattern(vllm_config, eps=eps).register(self.pattern_match_passes)
            if enable_custom_op():
                AddRMSNormQuantPattern(vllm_config, eps=eps).register(self.pattern_match_passes)
                AddRMSNormQuantPatternWithBias(vllm_config, eps=eps).register(self.pattern_match_passes)
                AddRMSNormDynamicQuantPatternWithBias(vllm_config, eps=eps).register(self.pattern_match_passes)
                AddRMSNormDynamicQuantPatternNoBias(vllm_config, eps=eps).register(self.pattern_match_passes)

    def __call__(self, graph: torch.fx.Graph):
        self.begin()
        self.matched_count = self.pattern_match_passes.apply(graph)
        logger.debug("Replaced %s patterns", self.matched_count)
        self.end_and_log()

    def is_applicable_for_range(self, compile_range: Range) -> bool:
        """
        Check if the pass is applicable for the current configuration.
        """
        return True
