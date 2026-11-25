#
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
# This file is a part of the vllm-ascend project.
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
"""Fuse the residual add and RMSNorm into ``npu_add_rms_norm``.

Models that keep the residual stream explicit in their layer forward (IQuest-Q1
and other Ascend-specific families) emit a plain ``aten.add`` followed by
``npu_rms_norm`` for every "residual + norm" pair. The fused Ascend op returns
the normalized value, the rstd (unused by these models) and the rounded
residual sum, so the replacement covers every consumer of both the add and the
norm outputs - including graph outputs saved by AOTAutograd. That is why this
fusion is safe where an add+norm+quant rule cannot fire: the fused op still
materializes the norm value.

The pass must run before ``muls_add``: the dominant graph shape is
``x + branch * scale`` -> ``aten.add`` -> ``npu_rms_norm``, and muls_add would
fold the mul and add first, removing the add this pass needs.
"""

import torch
from torch._inductor.pattern_matcher import PatternMatcherPass
from vllm.compilation.passes.vllm_inductor_pass import VllmInductorPass
from vllm.config import VllmConfig
from vllm.config.compilation import Range
from vllm.logger import logger

from vllm_ascend.compilation.passes.base_pattern import BasePattern

_NPU_RMS_NORM_DEFAULT_EPS = 1e-6
_FUSED_DTYPES = (torch.bfloat16, torch.float16)


def _node_value(node: torch.fx.Node):
    """Shape/dtype metadata carried by a graph node.

    AOTAutograd fills ``meta["val"]``; a plain make_fx trace (used by tests and
    offline probes) fills ``meta["example_value"]`` instead.
    """
    return node.meta.get("val", node.meta.get("example_value"))


def _add_matches_fused_contract(node: torch.fx.Node) -> bool:
    """Whether the add operands satisfy npu_add_rms_norm's ``x1``/``x2`` contract."""
    x1, x2 = node.args[:2]
    if not isinstance(x1, torch.fx.Node) or not isinstance(x2, torch.fx.Node):
        return False
    value1 = _node_value(x1)
    value2 = _node_value(x2)
    if value1 is None or value2 is None:
        return False
    return value1.shape == value2.shape and value1.dtype in _FUSED_DTYPES


class AddRMSNormPattern(BasePattern):
    def __init__(self, vllm_config: VllmConfig, eps: float = 1e-6, implicit_eps: bool = False):
        super().__init__(vllm_config, eps)
        # make_fx drops an epsilon equal to the schema default, so the
        # default-epsilon graph keeps only two arguments.
        self.implicit_eps = implicit_eps

    def pattern_key(self) -> str:
        return f"{super().pattern_key()}_implicit{self.implicit_eps}"

    def get_inputs(self):
        residual = torch.randn(2, 64, device="npu", dtype=self.dtype)
        branch = torch.randn(2, 64, device="npu", dtype=self.dtype)
        norm_weight = torch.randn(64, device="npu", dtype=self.dtype)
        return [residual, branch, norm_weight]

    def get_pattern(self):
        def pattern(residual: torch.Tensor, branch: torch.Tensor, norm_weight: torch.Tensor):
            added = residual + branch
            if self.implicit_eps:
                norm = torch.ops.npu.npu_rms_norm(added, norm_weight)
            else:
                norm = torch.ops.npu.npu_rms_norm(added, norm_weight, self.eps)
            return norm[0], added

        return pattern

    def get_replacement(self):
        def replacement(residual: torch.Tensor, branch: torch.Tensor, norm_weight: torch.Tensor):
            y, _, added = torch.ops.npu.npu_add_rms_norm(residual, branch, norm_weight, self.eps)
            return y, added

        return replacement

    def get_extra_check(self):
        """Reject matches the fused op cannot represent.

        The matcher compares operators and argument structure only; it does not
        check shapes/dtypes. ``npu_add_rms_norm`` requires ``x1``/``x2`` to have
        the same shape and dtype, and gamma to share that dtype, so a broadcast
        add or a fp32 gamma must stay on the unfused path.
        """
        norm_targets = (torch.ops.npu.npu_rms_norm, torch.ops.npu.npu_rms_norm.default)

        def _check_match(match) -> bool:
            add_ok = False
            norm_ok = False
            add_dtype = None
            gamma_dtype = None
            for node in match.nodes:
                if node.op != "call_function":
                    continue
                if node.target == torch.ops.aten.add.Tensor:
                    if not _add_matches_fused_contract(node):
                        return False
                    add_dtype = _node_value(node.args[0]).dtype
                    add_ok = True
                elif node.target in norm_targets:
                    node_eps = node.args[2] if len(node.args) > 2 else _NPU_RMS_NORM_DEFAULT_EPS
                    if node_eps != self.eps:
                        return False
                    weight = node.args[1]
                    if not isinstance(weight, torch.fx.Node):
                        return False
                    weight_value = _node_value(weight)
                    if weight_value is None or weight_value.dtype not in _FUSED_DTYPES:
                        return False
                    gamma_dtype = weight_value.dtype
                    norm_ok = True
            # npu_add_rms_norm requires x1/x2/gamma to share one dtype.
            return add_ok and norm_ok and gamma_dtype == add_dtype

        return _check_match


class AddRMSNormFusionPass(VllmInductorPass):
    """Replace ``aten.add + npu_rms_norm`` with ``npu_add_rms_norm``."""

    def __init__(self, vllm_config: VllmConfig):
        super().__init__(vllm_config)
        self.pattern_match_passes = PatternMatcherPass(pass_name="add_rms_norm_fusion_pass")

        dtype = vllm_config.model_config.dtype
        if dtype not in _FUSED_DTYPES:
            logger.debug("AddRMSNorm fusion not enabled: unsupported dtype %s", dtype)
            return

        # 1e-5 keeps an explicit epsilon in the AOT graph; the schema-default
        # 1e-6 is materialized without the argument.
        AddRMSNormPattern(vllm_config, eps=1e-5).register(self.pattern_match_passes)
        AddRMSNormPattern(vllm_config, eps=1e-6, implicit_eps=True).register(self.pattern_match_passes)

    def __call__(self, graph: torch.fx.Graph):
        self.begin()
        self.matched_count = self.pattern_match_passes.apply(graph)
        logger.debug("Replaced %s AddRMSNorm patterns", self.matched_count)
        self.end_and_log()

    def is_applicable_for_range(self, compile_range: Range) -> bool:
        return True
