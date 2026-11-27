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


def test_decoder_skips_bf16_gather_when_qkv_agmm_is_ready(monkeypatch):
    monkeypatch.setattr(
        iquest_model,
        "sp_all_gather",
        lambda value: (_ for _ in ()).throw(
            AssertionError("BF16 AllGather must be fused into QKV")
        ),
    )

    class FusedAttention:
        def __init__(self):
            self.calls = []

        def _fused_qkv_agmm_ready(self):
            return True

        def __call__(
            self,
            *,
            positions,
            hidden_states,
            sp_active,
            qkv_input_is_sharded,
        ):
            self.calls.append(
                (positions.shape[0], hidden_states.shape[0], sp_active, qkv_input_is_sharded)
            )
            return torch.zeros_like(hidden_states)

    attention = FusedAttention()
    layer = SimpleNamespace(
        use_sequence_parallel=True,
        use_sandwich_norm=False,
        attention_norm=lambda value: value,
        self_attn=attention,
        attn_out_norm=lambda value: value,
        attn_out_scale=1.0,
        feed_forward_norm=lambda value: value,
        _forward_ffn=lambda value, full_tokens, sp: torch.zeros_like(value),
        ffn_out_scale=1.0,
    )
    local_hidden = torch.randn(2, 8)
    positions = torch.arange(4)

    output = iquest_model.IQuestQ1DecoderLayer.forward(
        layer,
        positions,
        local_hidden,
        sp_active=True,
    )

    assert attention.calls == [(4, 2, True, True)]
    assert output.shape == local_hidden.shape


def test_qkv_agmm_quantizes_local_shard_and_uses_nd_weight(monkeypatch):
    quantized = torch.empty(2, 8, dtype=torch.float8_e4m3fn)
    x_scale = torch.empty(2, 1, 2, dtype=torch.uint8)

    class Scheme:
        def _quantize_input(self, layer, hidden_states):
            assert hidden_states.shape == (2, 8)
            return quantized, x_scale

    weight_nd = torch.empty(6, 8, dtype=torch.float8_e4m3fn)
    weight_scale_nd = torch.empty(6, 1, 2, dtype=torch.uint8)
    qkv_proj = SimpleNamespace(
        quant_method=SimpleNamespace(quant_method=Scheme()),
        _mxfp8_agmm_weight_nd_buf=weight_nd,
        _mxfp8_agmm_scale_nd_buf=weight_scale_nd,
    )
    calls = []

    def agmm(*args, **kwargs):
        calls.append((args, kwargs))
        return torch.empty(8, 6), torch.empty(0), torch.empty(0)

    attention = SimpleNamespace(
        qkv_proj=qkv_proj,
        _qkv_agmm_op=agmm,
        _fused_qkv_agmm_ready=lambda: True,
    )
    monkeypatch.setattr(
        iquest_model,
        "get_tp_group",
        lambda: SimpleNamespace(device_group="tp-process-group"),
    )

    output = iquest_model.IQuestQ1Attention._project_qkv(
        attention,
        torch.arange(7),
        torch.empty(2, 8),
        input_is_sharded=True,
    )

    assert output.shape == (7, 6)
    args, kwargs = calls[0]
    assert args[0] is quantized
    assert args[1].shape == (8, 6)
    assert not args[1].is_contiguous()
    assert args[2] == "tp-process-group"
    assert kwargs["x1_scale"] is x_scale
    assert kwargs["x2_scale"] is weight_scale_nd
    assert kwargs["group_sizes"] == [1, 1, 32]
    assert kwargs["comm_mode"] == "aiv_urma"
