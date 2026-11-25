# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Eligibility rules for the IQuest-Q1 fused split+RMSNorm+RoPE path."""

from types import SimpleNamespace

import pytest

from vllm_ascend.models.iquest_q1.model import _can_use_split_qkv_rmsnorm_rope


def _rotary(cache_width=32, rotary_dim=32, is_neox_style=True):
    return SimpleNamespace(
        is_neox_style=is_neox_style,
        rotary_dim=rotary_dim,
        cos_sin_cache=_Cache(4096, cache_width),
    )


class _Cache:
    def __init__(self, rows, width):
        self._shape = (rows, width)

    def dim(self):
        return len(self._shape)

    @property
    def shape(self):
        return self._shape


def test_iqest_shape_is_supported():
    assert _can_use_split_qkv_rmsnorm_rope(
        enabled=True,
        k_norm=object(),
        rotary_emb=_rotary(),
        head_dim=128,
    )


@pytest.mark.parametrize(
    "kwargs",
    [
        dict(enabled=False, k_norm=object(), rotary_emb=_rotary(), head_dim=128),
        dict(enabled=True, k_norm=None, rotary_emb=_rotary(), head_dim=128),
        dict(enabled=True, k_norm=object(), rotary_emb=None, head_dim=128),
        # non-power-of-two head dim cannot be tiled by tl.arange
        dict(enabled=True, k_norm=object(), rotary_emb=_rotary(), head_dim=96),
        # interleaved (GPT-J) RoPE is not what the kernel implements
        dict(
            enabled=True,
            k_norm=object(),
            rotary_emb=_rotary(is_neox_style=False),
            head_dim=128,
        ),
        # cache width must be the rotary dim, not the head dim
        dict(
            enabled=True,
            k_norm=object(),
            rotary_emb=_rotary(cache_width=128),
            head_dim=128,
        ),
    ],
)
def test_ineligible_layers_stay_on_reference_path(kwargs):
    assert not _can_use_split_qkv_rmsnorm_rope(**kwargs)
