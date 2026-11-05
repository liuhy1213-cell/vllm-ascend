# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for the IQuest-Q1 sink-attention ACL-graph capture path.

The capture path registers the FIA v2 call as an updatable-graph task; these
tests verify the registered task contract (operation, kwargs, param provider)
and that the sink correction math is applied to the task's output buffers.
"""

from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from vllm_ascend.attention.attention_v1 import AscendAttentionBackend, AscendAttentionState, FIAV2ParamProvider
from vllm_ascend.models.iquest_q1.attention import (
    SWA_INT_MAX,
    IQuestQ1AscendAttentionBackend,
    IQuestQ1AscendAttentionImpl,
)

NUM_HEADS = 4
NUM_KV_HEADS = 2
HEAD_SIZE = 8
SCALE = 0.25
NUM_TOKENS = 4
ATTENTION_MODULE = "vllm_ascend.models.iquest_q1.attention"


def _make_impl(sliding_window=None):
    impl = IQuestQ1AscendAttentionImpl.__new__(IQuestQ1AscendAttentionImpl)
    impl.num_heads = NUM_HEADS
    impl.num_kv_heads = NUM_KV_HEADS
    impl.head_size = HEAD_SIZE
    impl.scale = SCALE
    impl.attn_type = "DECODER"
    impl.use_bnsd_kv_cache = False
    impl.key_cache = torch.randn(4, 16, NUM_KV_HEADS, HEAD_SIZE)
    impl.value_cache = torch.randn(4, 16, NUM_KV_HEADS, HEAD_SIZE)
    impl.sliding_window = sliding_window
    impl._use_max_workspace_for_fia_graph = False
    impl._layer_name = "model.layers.0.self_attn.attn"
    impl._sink_key = torch.randn(NUM_KV_HEADS, HEAD_SIZE)
    return impl


def _make_metadata(state=AscendAttentionState.DecodeOnly):
    return SimpleNamespace(
        attn_state=state,
        actual_seq_lengths_q=[1, 2, 3, 4],
        seq_lens_list=[5, 7, 9, 11],
        block_tables=torch.zeros(4, 2, dtype=torch.int32),
        num_actual_tokens=NUM_TOKENS,
        attn_mask=None,
        causal=True,
    )


def _expected_output(query, sink_key, raw_output, lse):
    group = NUM_HEADS // NUM_KV_HEADS
    q = query.reshape(NUM_TOKENS, NUM_KV_HEADS, group, HEAD_SIZE)
    sink_logit = (q.float() * sink_key.float().unsqueeze(1)).sum(-1) * SCALE
    sink_logit = sink_logit.reshape(NUM_TOKENS, NUM_HEADS)
    factor = torch.sigmoid(-(sink_logit.unsqueeze(-1) - lse))
    return raw_output.float() * factor


@pytest.fixture
def recorded_task():
    """Simulate the op executing inside the task group, then record the task.

    The real capture writes the FIA results into the out buffers before the
    captured correction ops run, so the fake fills them with known values.
    """
    recorded = {}
    raw_output = torch.randn(NUM_TOKENS, NUM_HEADS, HEAD_SIZE)
    lse = torch.randn(NUM_TOKENS, NUM_HEADS, 1) * 3.0

    def fake_register_task(operation, kwargs, provider):
        recorded["operation"] = operation
        recorded["kwargs"] = kwargs
        recorded["provider"] = provider
        out_buffer, lse_buffer = kwargs["out"]
        out_buffer.copy_(raw_output)
        lse_buffer.copy_(lse)

    def fake_get_capture_resource(key, factory, use_max_workspace=False):
        recorded["workspace_key"] = key
        recorded["use_max_workspace"] = use_max_workspace
        factory()
        return torch.empty(1)

    return recorded, raw_output, lse, fake_register_task, fake_get_capture_resource


@pytest.mark.parametrize("sliding_window,expected_sparse,expected_pre", [(None, 3, SWA_INT_MAX), (4096, 4, 4096)])
def test_capture_registers_fia_task_and_applies_sink(recorded_task, sliding_window, expected_sparse, expected_pre):
    recorded, raw_output, lse, fake_register_task, fake_get_capture_resource = recorded_task
    impl = _make_impl(sliding_window=sliding_window)
    metadata = _make_metadata()
    query = torch.randn(NUM_TOKENS, NUM_HEADS, HEAD_SIZE)
    key = torch.randn(NUM_TOKENS, NUM_KV_HEADS, HEAD_SIZE)
    value = torch.randn(NUM_TOKENS, NUM_KV_HEADS, HEAD_SIZE)
    kv_cache = (impl.key_cache, impl.value_cache)
    output = torch.full((NUM_TOKENS, NUM_HEADS, HEAD_SIZE), 7.0)

    with (
        patch(f"{ATTENTION_MODULE}.torch_npu") as mock_torch_npu,
        patch(f"{ATTENTION_MODULE}._EXTRA_CTX", SimpleNamespace(capturing=True)),
        patch(f"{ATTENTION_MODULE}.register_task", side_effect=fake_register_task),
        patch(f"{ATTENTION_MODULE}.get_capture_resource", side_effect=fake_get_capture_resource),
    ):
        result = impl.forward_fused_infer_attention(query, key, value, metadata, output, kv_cache)

    assert result is output
    op = mock_torch_npu.npu_fused_infer_attention_score_v2.out
    assert recorded["operation"] is op
    kwargs = recorded["kwargs"]
    assert kwargs["return_softmax_lse"] is True
    assert kwargs["input_layout"] == "TND"
    assert kwargs["sparse_mode"] == expected_sparse
    assert kwargs["pre_tokens"] == expected_pre
    assert kwargs["next_tokens"] == 0
    assert kwargs["softmax_scale"] == SCALE
    assert kwargs["actual_seq_qlen"] == metadata.actual_seq_lengths_q
    assert kwargs["actual_seq_kvlen"] == metadata.seq_lens_list
    assert kwargs["num_query_heads"] == NUM_HEADS
    assert kwargs["num_key_value_heads"] == NUM_KV_HEADS
    assert kwargs["atten_mask"] is None
    out_buffer, lse_buffer = kwargs["out"]
    assert out_buffer.shape == (NUM_TOKENS, NUM_HEADS, HEAD_SIZE)
    assert lse_buffer.shape == (NUM_TOKENS, NUM_HEADS, 1)
    assert lse_buffer.dtype == torch.float32
    assert isinstance(recorded["provider"], FIAV2ParamProvider)
    assert recorded["provider"].layer_name == impl._layer_name
    # The workspace query must see the same op parameters.
    workspace_call = mock_torch_npu._npu_fused_infer_attention_score_v2_get_max_workspace.call_args
    assert workspace_call.kwargs["return_softmax_lse"] is True
    assert workspace_call.kwargs["actual_seq_qlen"] == metadata.actual_seq_lengths_q
    assert recorded["use_max_workspace"] is False

    expected = _expected_output(query, impl._sink_key, raw_output, lse)
    torch.testing.assert_close(output, expected.to(output.dtype), rtol=1e-5, atol=1e-6)


def test_capture_uses_kv_cache_view_for_decode(recorded_task):
    """Decode capture reads K/V through the paged cache, not the current K/V."""
    recorded, *_, fake_register_task, fake_get_capture_resource = recorded_task
    impl = _make_impl()
    metadata = _make_metadata()
    query = torch.randn(NUM_TOKENS, NUM_HEADS, HEAD_SIZE)
    output = torch.zeros(NUM_TOKENS, NUM_HEADS, HEAD_SIZE)

    with (
        patch(f"{ATTENTION_MODULE}.torch_npu") as _,
        patch(f"{ATTENTION_MODULE}._EXTRA_CTX", SimpleNamespace(capturing=True)),
        patch(f"{ATTENTION_MODULE}.register_task", side_effect=fake_register_task),
        patch(f"{ATTENTION_MODULE}.get_capture_resource", side_effect=fake_get_capture_resource),
    ):
        impl.forward_fused_infer_attention(query, None, None, metadata, output, (impl.key_cache, impl.value_cache))

    kwargs = recorded["kwargs"]
    num_blocks, block_size, kv_heads, head_dim = impl.key_cache.shape
    assert kwargs["key"].shape == (num_blocks, block_size, kv_heads * head_dim)
    assert kwargs["key"].data_ptr() == impl.key_cache.data_ptr()
    assert kwargs["block_table"] is metadata.block_tables


@pytest.mark.parametrize("backend,expected", [(IQuestQ1AscendAttentionBackend, True), (AscendAttentionBackend, False)])
def test_uses_fia_v2_contract(backend, expected):
    """The draft replay path keys its param dicts off this declaration.

    The iquest sink path registers FIA v2 updatable tasks (TND key names);
    the stock backend registers v1 tasks. A wrong declaration makes the
    draft replay feed v1 key names into the v2 op (or vice versa).
    """
    assert backend.uses_fia_v2() is expected
