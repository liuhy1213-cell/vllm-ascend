#
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
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
# This file is a part of the vllm-ascend project.
#

import importlib
import importlib.util
import sys
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import torch

from vllm_ascend.device.hardware import AscendDeviceType
from vllm_ascend.device.hardware_profile import get_hardware_profile

MX_PATTERN_NAMES = {
    "AddResidualRMSNormDynamicMXQuantPattern",
    "AddResidualRMSNormDynamicMXQuantImplicitEpsPattern",
    "AddRMSNormDynamicMXQuantPattern",
    "RMSNormDynamicMXQuantPattern",
}


@pytest.fixture(scope="module")
def pass_module():
    # base_pattern imports npugraph_ex (or torchair) at module level. Building
    # the pass does not need it, so stand in for it when neither is installed.
    stubbed = importlib.util.find_spec("npugraph_ex") is None and importlib.util.find_spec("torchair") is None
    if stubbed:
        sys.modules["npugraph_ex"] = MagicMock()
    try:
        return importlib.import_module("vllm_ascend.compilation.passes.norm_quant_fusion_pass")
    finally:
        if stubbed:
            del sys.modules["npugraph_ex"]


def make_vllm_config():
    vllm_config = MagicMock()
    vllm_config.model_config.dtype = torch.bfloat16
    vllm_config.compilation_config.splitting_ops = []
    vllm_config.compilation_config.use_inductor_graph_partition = False
    vllm_config.quant_config = None
    return vllm_config


def build_pass(pass_module, device_type, fuse_add_rms_norm_mx_quant=True):
    """Instantiate the pass and return the (class name, eps) registration order."""
    calls = []

    def record(self, pm_pass):
        calls.append((type(self).__name__, self.eps))

    with (
        patch.object(
            pass_module,
            "get_current_hardware_profile",
            return_value=get_hardware_profile(device_type),
        ),
        patch.object(pass_module, "enable_custom_op", return_value=False),
        patch.object(pass_module.BasePattern, "register", record),
    ):
        pass_module.AddRMSNormQuantFusionPass(
            make_vllm_config(), fuse_add_rms_norm_mx_quant=fuse_add_rms_norm_mx_quant
        )
    return calls


def test_three_op_mx_pattern_registered_before_two_op_pattern(pass_module):
    """The add+norm+quant rule must win over norm+quant at the same site.

    PatternMatcherPass applies rules in registration order; if the two-op rule
    matched first it would strip the RMSNorm and leave the residual add
    behind, making the three-op fusion unreachable.
    """
    calls = build_pass(pass_module, AscendDeviceType.A5)

    for eps in (1e-5, 1e-6):
        names = [name for name, registered_eps in calls if registered_eps == eps]
        assert "AddResidualRMSNormDynamicMXQuantPattern" in names
        assert "RMSNormDynamicMXQuantPattern" in names
        assert names.index("AddResidualRMSNormDynamicMXQuantPattern") < names.index(
            "RMSNormDynamicMXQuantPattern"
        )

    # The epsilon-less form is only valid for the schema default (1e-6).
    for eps in (1e-5, 1e-6):
        implicit = [name for name, registered_eps in calls if registered_eps == eps]
        assert ("AddResidualRMSNormDynamicMXQuantImplicitEpsPattern" in implicit) == (eps == 1e-6)


def test_residual_three_op_patterns_can_be_disabled(pass_module):
    """The dedicated switch keeps the two-op MX fusion but skips add+norm+quant."""
    calls = build_pass(pass_module, AscendDeviceType.A5, fuse_add_rms_norm_mx_quant=False)

    names = {name for name, _ in calls}
    assert "AddResidualRMSNormDynamicMXQuantPattern" not in names
    assert "AddResidualRMSNormDynamicMXQuantImplicitEpsPattern" not in names
    assert "AddRMSNormDynamicMXQuantPattern" in names
    assert "RMSNormDynamicMXQuantPattern" in names


def test_mx_patterns_skipped_without_dynamic_mx_quant_capability(pass_module):
    calls = build_pass(pass_module, AscendDeviceType.A2)

    names = {name for name, _ in calls}
    assert "AddRMSNormDynamicQuantPattern" in names
    assert not names & MX_PATTERN_NAMES


def _add_node(shape1, shape2, dtype=torch.bfloat16):
    graph = torch.fx.Graph()
    x1 = graph.placeholder("x1")
    x2 = graph.placeholder("x2")
    added = graph.call_function(torch.ops.aten.add.Tensor, (x1, x2))
    x1.meta["val"] = torch.empty(shape1, dtype=dtype)
    x2.meta["val"] = torch.empty(shape2, dtype=dtype)
    return added


def test_add_contract_requires_same_shape_activation_tensors(pass_module):
    assert pass_module._add_matches_fused_contract(_add_node((4, 8), (4, 8)))
    assert not pass_module._add_matches_fused_contract(_add_node((4, 8), (8,)))
    assert not pass_module._add_matches_fused_contract(_add_node((4, 8), (4, 8), dtype=torch.float32))


def test_extra_check_rejects_partial_chain(pass_module):
    """A match without the RMSNorm/quant nodes must not be fused."""
    match = SimpleNamespace(nodes=[_add_node((4, 8), (4, 8))])
    pattern = pass_module.AddResidualRMSNormDynamicMXQuantPattern(make_vllm_config(), eps=1e-6)
    assert not pattern.get_extra_check()(match)
