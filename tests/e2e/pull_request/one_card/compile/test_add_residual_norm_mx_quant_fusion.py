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
#

import pytest
import torch
import torch.nn as nn
import torch_npu  # noqa: F401
import vllm.config
from vllm.config import ModelConfig, VllmConfig
from vllm.distributed import ensure_model_parallel_initialized, init_distributed_environment
from vllm.utils.system_utils import update_environment_variables

from vllm_ascend.ascend_forward_context import set_ascend_forward_context
from vllm_ascend.compilation.passes.norm_quant_fusion_pass import AddRMSNormQuantFusionPass
from vllm_ascend.device.hardware_profile import HardwareCapability, get_current_hardware_profile

from .backend import TestBackend


class TestModelResidualAddNormMxQuant(nn.Module):
    """Mimics an IQuest-Q1 layer boundary:

        residual = x + branch
        norm = RMSNorm(residual)
        quantized, mx_scale = DynamicMxQuant(norm, fp8)

    The add result feeds nothing else, and the normalized value is consumed
    only by the quantization.
    """

    def __init__(self, hidden_size: int, eps: float = 1e-6, device="npu"):
        super().__init__()
        self.eps = eps
        self.rms_norm_weight = nn.Parameter(torch.randn(hidden_size, device=device))

    def forward(self, x, branch):
        residual = torch.ops.aten.add.Tensor(x, branch, alpha=1)
        norm_output = torch_npu.npu_rms_norm(residual, self.rms_norm_weight, self.eps)[0]
        return torch.ops.npu.npu_dynamic_mx_quant(norm_output, dst_type=torch.float8_e4m3fn)

    def ops_in_model_before(self):
        return [
            torch.ops.aten.add.Tensor,
            torch.ops.npu.npu_rms_norm.default,
            torch.ops.npu.npu_dynamic_mx_quant.default,
        ]

    def ops_in_model_after(self):
        return [torch.ops.npu.npu_add_rms_norm_dynamic_mx_quant.default]


@pytest.mark.parametrize("dtype", [torch.bfloat16])
@pytest.mark.parametrize("hidden_size", [64])
@pytest.mark.parametrize("num_tokens", [257])
@pytest.mark.parametrize("eps", [1e-6])
def test_residual_add_norm_mx_quant_fusion(dtype: torch.dtype, hidden_size: int, num_tokens: int, eps: float):
    if not get_current_hardware_profile().supports(HardwareCapability.DYNAMIC_MX_QUANT_FUSION):
        pytest.skip("Dynamic MX quant fusion is not supported on this hardware profile")
    if not hasattr(torch.ops.npu, "npu_add_rms_norm_dynamic_mx_quant"):
        pytest.skip("Fused operator npu_add_rms_norm_dynamic_mx_quant is not available")

    torch.set_default_dtype(dtype)
    torch.manual_seed(1)

    vllm_config = VllmConfig(model_config=ModelConfig(dtype=dtype))
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
        backend = TestBackend(custom_passes=[AddRMSNormQuantFusionPass(vllm_config=vllm_config)])
        model = TestModelResidualAddNormMxQuant(hidden_size, eps=eps).to("npu")

        x = torch.randn(num_tokens, hidden_size, device="npu", dtype=dtype, requires_grad=False)
        branch = torch.randn_like(x)

        compiled_model = torch.compile(model, backend=backend)
        # vLLM serves under inference_mode, where AOTAutograd builds the
        # inference graph (no saved rstd tensor) this pass is designed for.
        with torch.inference_mode():
            quantized, mx_scale = compiled_model(x, branch)
        assert quantized.shape == (num_tokens, hidden_size)
        assert mx_scale.shape == (num_tokens, 1, 2)

        backend.check_before_ops(model.ops_in_model_before(), fully_replaced=True)
        backend.check_after_ops(model.ops_in_model_after())
