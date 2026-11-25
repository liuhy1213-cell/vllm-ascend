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
"""Registration and match-contract tests for the AddRMSNorm fusion pass."""

import importlib
import importlib.util
import sys
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import torch
import torch_npu  # noqa: F401

EPSILONS = (1e-5, 1e-6)


@pytest.fixture(scope="module")
def pass_module():
    # base_pattern imports npugraph_ex (or torchair) at module level. Building
    # the pass does not need it, so stand in for it when neither is installed.
    stubbed = importlib.util.find_spec("npugraph_ex") is None and importlib.util.find_spec("torchair") is None
    if stubbed:
        sys.modules["npugraph_ex"] = MagicMock()
    try:
        return importlib.import_module("vllm_ascend.compilation.passes.add_rms_norm_fusion_pass")
    finally:
        if stubbed:
            del sys.modules["npugraph_ex"]


def make_vllm_config(dtype=torch.bfloat16):
    vllm_config = MagicMock()
    vllm_config.model_config.dtype = dtype
    vllm_config.compilation_config.splitting_ops = []
    vllm_config.compilation_config.use_inductor_graph_partition = False
    return vllm_config


def build_pass(pass_module, dtype=torch.bfloat16):
    """Instantiate the pass and return the (eps, implicit) registration order."""
    calls = []

    def record(self, pm_pass):
        calls.append((self.eps, self.implicit_eps))

    with patch.object(pass_module.BasePattern, "register", record):
        pass_module.AddRMSNormFusionPass(make_vllm_config(dtype))
    return calls


def test_default_epsilon_registers_the_two_arg_pattern(pass_module):
    calls = build_pass(pass_module)

    assert (1e-6, True) in calls
    assert (1e-5, False) in calls
    # Only the schema default may use the epsilon-less form.
    assert (1e-6, False) not in calls
    assert (1e-5, True) not in calls


def test_fp32_models_skip_the_pass(pass_module):
    assert build_pass(pass_module, dtype=torch.float32) == []


def _add_node(shape1, shape2, dtype=torch.bfloat16):
    graph = torch.fx.Graph()
    x1 = graph.placeholder("x1")
    x2 = graph.placeholder("x2")
    added = graph.call_function(torch.ops.aten.add.Tensor, (x1, x2))
    x1.meta["val"] = torch.empty(shape1, dtype=dtype)
    x2.meta["val"] = torch.empty(shape2, dtype=dtype)
    return added


def _norm_node(added, weight_dtype=torch.bfloat16, eps_arg=None):
    graph = torch.fx.Graph()
    added = graph.placeholder("added")
    weight = graph.get_attr("weight")
    weight.meta["val"] = torch.empty(64, dtype=weight_dtype)
    args = (added, weight) if eps_arg is None else (added, weight, eps_arg)
    return graph.call_function(torch.ops.npu.npu_rms_norm, args)


def test_add_contract_requires_same_shape_activation_tensors(pass_module):
    assert pass_module._add_matches_fused_contract(_add_node((4, 8), (4, 8)))
    assert not pass_module._add_matches_fused_contract(_add_node((4, 8), (8,)))
    assert not pass_module._add_matches_fused_contract(_add_node((4, 8), (4, 8), dtype=torch.float32))


def test_extra_check_accepts_complete_chain(pass_module):
    added = _add_node((4, 64), (4, 64))
    match = SimpleNamespace(nodes=[added, _norm_node(added)])
    pattern = pass_module.AddRMSNormPattern(make_vllm_config(), eps=1e-6, implicit_eps=True)
    assert pattern.get_extra_check()(match)


def test_extra_check_rejects_partial_chain(pass_module):
    match = SimpleNamespace(nodes=[_add_node((4, 64), (4, 64))])
    pattern = pass_module.AddRMSNormPattern(make_vllm_config(), eps=1e-6, implicit_eps=True)
    assert not pattern.get_extra_check()(match)


def test_extra_check_rejects_eps_or_gamma_mismatch(pass_module):
    added = _add_node((4, 64), (4, 64))
    pattern = pass_module.AddRMSNormPattern(make_vllm_config(), eps=1e-6, implicit_eps=True)

    wrong_eps = SimpleNamespace(nodes=[added, _norm_node(added, eps_arg=1e-5)])
    assert not pattern.get_extra_check()(wrong_eps)

    fp32_gamma = SimpleNamespace(nodes=[added, _norm_node(added, weight_dtype=torch.float32)])
    assert not pattern.get_extra_check()(fp32_gamma)
