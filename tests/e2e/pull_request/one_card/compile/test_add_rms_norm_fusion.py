#
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
# Copyright 2023 The vLLM team.
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
"""End-to-end check for the residual-add + RMSNorm fusion pass.

The model mirrors the IQuest-Q1 decoder-layer shape: the add result feeds both
the norm and the residual stream, and the norm value has an extra consumer.
"""

import json

import pytest
import torch
import torch.nn as nn
import vllm.config
from vllm.compilation.passes.fx_utils import OpOverload
from vllm.config import ModelConfig, VllmConfig
from vllm.distributed import (
    ensure_model_parallel_initialized,
    init_distributed_environment,
)
from vllm.utils.system_utils import update_environment_variables

import vllm_ascend.ops.register_custom_ops  # noqa
from vllm_ascend.ascend_forward_context import set_ascend_forward_context
from vllm_ascend.compilation.passes.add_rms_norm_fusion_pass import AddRMSNormFusionPass

from .backend import TestBackend

_backend_cache = None

# A local, dependency-free config keeps ModelConfig construction offline; the
# default model id would need a Hugging Face fetch, which CI has and the
# operator test hosts do not.
_TINY_CONFIG = {
    "architectures": ["Qwen2ForCausalLM"],
    "model_type": "qwen2",
    "hidden_size": 64,
    "intermediate_size": 128,
    "num_hidden_layers": 1,
    "num_attention_heads": 4,
    "num_key_value_heads": 2,
    "head_dim": 16,
    "vocab_size": 100,
    "max_position_embeddings": 128,
    "rms_norm_eps": 1e-6,
    "torch_dtype": "bfloat16",
    "tie_word_embeddings": False,
}


@pytest.fixture(scope="module")
def tiny_model_dir(tmp_path_factory):
    path = tmp_path_factory.mktemp("tiny-qwen")
    (path / "config.json").write_text(json.dumps(_TINY_CONFIG), encoding="utf-8")
    return str(path)


def get_or_create_backend(vllm_config):
    global _backend_cache
    if _backend_cache is None:
        _backend_cache = TestBackend(custom_passes=[AddRMSNormFusionPass(vllm_config=vllm_config)])
    return _backend_cache


class TestModel(nn.Module):
    def __init__(self, hidden_size: int, dtype: torch.dtype, eps: float, device="npu"):
        super().__init__()
        self.eps = eps
        self.norm_weight = nn.Parameter(torch.randn(hidden_size, dtype=dtype, device=device))

    def forward(self, x: torch.Tensor, branch: torch.Tensor):
        added = x + branch
        normed = torch.ops.npu.npu_rms_norm(added, self.norm_weight, self.eps)[0]
        return normed, added

    def ops_in_model_before(self) -> list[OpOverload]:
        # aten.add.Tensor is already an OpOverload (no .default packet).
        return [torch.ops.aten.add.Tensor, torch.ops.npu.npu_rms_norm.default]

    def ops_in_model_after(self) -> list[OpOverload]:
        return [torch.ops.npu.npu_add_rms_norm.default]


@pytest.mark.parametrize("dtype", [torch.bfloat16])
@pytest.mark.parametrize("hidden_size", [64])
@pytest.mark.parametrize("num_tokens", [257])
@pytest.mark.parametrize("eps", [1e-5, 1e-6])
def test_add_rms_norm_fusion(dtype, hidden_size, num_tokens, eps, tiny_model_dir):
    torch.set_default_dtype(dtype)
    torch.manual_seed(1)

    vllm_config = VllmConfig(
        model_config=ModelConfig(model=tiny_model_dir, dtype=dtype)
    )

    with vllm.config.set_current_vllm_config(vllm_config):
        update_environment_variables(
            {
                "RANK": "0",
                "LOCAL_RANK": "0",
                "WORLD_SIZE": "1",
                "MASTER_ADDR": "localhost",
                "MASTER_PORT": "12345",
            }
        )
        init_distributed_environment()
        ensure_model_parallel_initialized(1, 1)

    with vllm.config.set_current_vllm_config(vllm_config), set_ascend_forward_context(None, vllm_config):
        backend = get_or_create_backend(vllm_config)
        model = TestModel(hidden_size, dtype, eps, device="npu").to("npu")
        x = torch.rand(num_tokens, hidden_size, device="npu", dtype=dtype)
        branch = torch.rand(num_tokens, hidden_size, device="npu", dtype=dtype)

        # Inference mode matters: otherwise joint-AOT saves the rstd output,
        # the RMSNorm node gets a second user, and the production (inference)
        # graph shape is not what this test exercises.
        with torch.inference_mode():
            normed_ref, added_ref = model(x, branch)
            model_fused = torch.compile(model, backend=backend)
            normed, added = model_fused(x, branch)

        torch.testing.assert_close(normed.float(), normed_ref.float(), atol=5e-2, rtol=5e-3)
        assert torch.equal(added, added_ref), "residual sum must stay bitwise exact"

        backend.check_before_ops(model.ops_in_model_before(), fully_replaced=True)
        backend.check_after_ops(model.ops_in_model_after())
