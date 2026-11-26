from types import SimpleNamespace

import torch

from vllm_ascend.models.iquest_q1 import model as iquest_model


def test_o_proj_reduce_scatter_only_for_active_prefill(monkeypatch):
    calls = []
    monkeypatch.setattr(
        iquest_model,
        "sp_reduce_scatter",
        lambda value: calls.append("rs") or value[:2],
    )
    monkeypatch.setattr(
        iquest_model,
        "tensor_model_parallel_all_reduce",
        lambda value: calls.append("ar") or value,
    )
    attention = SimpleNamespace(use_sequence_parallel=True)
    partial = torch.arange(8).view(4, 2)

    prefill = iquest_model.IQuestQ1Attention._reduce_o_proj_output(
        attention,
        partial,
        True,
    )
    assert prefill.shape == (2, 2)
    assert calls == ["rs"]

    calls.clear()
    decode = iquest_model.IQuestQ1Attention._reduce_o_proj_output(
        attention,
        partial,
        False,
    )
    assert decode is partial
    assert calls == ["ar"]


def test_layer_zero_keeps_full_tokens_before_sharding(monkeypatch):
    monkeypatch.setattr(
        iquest_model,
        "get_pp_group",
        lambda: SimpleNamespace(is_first_rank=True, is_last_rank=True),
    )
    monkeypatch.setattr(iquest_model, "_is_prefill_forward", lambda: True)

    shard_inputs = []

    def shard(value):
        shard_inputs.append(value.clone())
        return value[: value.shape[0] // 2]

    monkeypatch.setattr(iquest_model, "sp_shard", shard)
    monkeypatch.setattr(
        iquest_model,
        "sp_all_gather",
        lambda value: torch.cat((value, value), dim=0),
    )

    class Layer:
        def __init__(self, layer_idx):
            self.layer_idx = layer_idx
            self.inputs = []
            self.sp_active = []

        def __call__(self, positions, hidden_states, sp_active):
            self.inputs.append(hidden_states.clone())
            self.sp_active.append(sp_active)
            return hidden_states + 1

    layer0 = Layer(0)
    layer1 = Layer(1)
    fake_model = SimpleNamespace(
        use_sequence_parallel=True,
        start_layer=0,
        end_layer=2,
        layers=[layer0, layer1],
        embed_input_ids=lambda input_ids, oe_input_ids=None: input_ids.float(),
        norm=lambda value: value,
    )
    input_ids = torch.arange(8).view(4, 2)
    positions = torch.arange(4)

    output = iquest_model.IQuestQ1Model.forward(
        fake_model,
        input_ids,
        positions,
    )

    assert layer0.inputs[0].shape == (4, 2)
    assert layer0.sp_active == [False]
    assert layer1.inputs[0].shape == (2, 2)
    assert layer1.sp_active == [True]
    assert len(shard_inputs) == 1
    assert output.shape == (4, 2)
