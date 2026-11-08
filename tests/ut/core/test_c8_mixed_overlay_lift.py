# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
"""C8 mixed-page overlay: block lift and eligibility gating (S2 path A).

Exercises the real planner helpers, not mirrors (review P2-1):

* ``_c8_mixed_overlay_lift_block_sizes`` lifts heterogeneous kernel block
  sizes to their LCM in place while keeping true per-group pages. Without it
  the sliding specs keep their native 128-token block, per-request sliding
  demand quadruples (cdiv(20480, 128) + 1 = 161 pages vs 41), and the
  measured max concurrency stays at 18.01x instead of the 1.238x design
  value (node6 s2-c8 engine log, 2026-11-08).
* ``_c8_mixed_overlay_true_page_specs`` and
  ``_c8_mixed_overlay_groups_eligible`` must apply identical conditions on
  the grouping side and the config side (review P0-2/P1): a grouping
  fallback must never coexist with block-outer overlay descriptors.
"""

import dataclasses
import importlib
from types import SimpleNamespace

import pytest
import torch
from vllm.v1.kv_cache_interface import FullAttentionSpec, KVCacheGroupSpec, SlidingWindowSpec

PATCH_MODULE = "vllm_ascend.patch.platform.patch_kv_cache_utils"

# Module-level hooks the patch module installs into vllm at import time; the
# fixture snapshots and restores them so the rest of the UT process keeps
# upstream behavior.
_PATCHED_KCU_ATTRS = (
    "resolve_kv_cache_block_sizes",
    "_get_packed_kv_cache_groups",
    "_get_kv_cache_groups_uniform_page_size",
    "get_kv_cache_groups",
    "get_kv_cache_config_from_groups",
    "_max_memory_usage_bytes_from_groups",
    "_pool_bytes_per_block",
)


@pytest.fixture(scope="module")
def patch_module():
    import vllm.v1.core.kv_cache_utils as kcu
    import vllm.v1.engine.core as engine_core
    from vllm.v1.kv_cache_interface import KVCacheConfig

    saved = {name: getattr(kcu, name) for name in _PATCHED_KCU_ATTRS}
    saved_engine = engine_core.resolve_kv_cache_block_sizes
    saved_prop = KVCacheConfig.__dict__.get("has_mamba_layers")
    mod = importlib.import_module(PATCH_MODULE)
    yield mod
    for name, value in saved.items():
        setattr(kcu, name, value)
    engine_core.resolve_kv_cache_block_sizes = saved_engine
    if saved_prop is None:
        if "has_mamba_layers" in KVCacheConfig.__dict__:
            del KVCacheConfig.has_mamba_layers
    else:
        setattr(KVCacheConfig, "has_mamba_layers", saved_prop)


def _full_c8_spec(block: int = 512) -> FullAttentionSpec:
    # Plain fp8 page (2 x 128 x 1 B/token): the 264 B/token C8 packet layout
    # does not matter for lift/gate tests, only page DIVERSITY does.
    return FullAttentionSpec(
        block_size=block, num_kv_heads=1, head_size=128, dtype=torch.float8_e4m3fn
    )


def _sliding_spec(block: int = 128, window: int = 4096) -> SlidingWindowSpec:
    return SlidingWindowSpec(
        block_size=block, num_kv_heads=1, head_size=128, dtype=torch.bfloat16, sliding_window=window
    )


def _mixed_specs() -> dict[str, object]:
    return {
        "l0": _full_c8_spec(512),
        "s0": _sliding_spec(128),
        "s1": _sliding_spec(128),
        "s2": _sliding_spec(128),
    }


def _config(layout_compact: bool = True) -> SimpleNamespace:
    layout = SimpleNamespace(is_layer_compact=layout_compact, is_block_compact=layout_compact)
    return SimpleNamespace(cache_config=SimpleNamespace(get_resolved_kv_cache_layout=lambda: layout))


@pytest.fixture
def overlay_on(monkeypatch):
    import vllm_ascend.utils as ascend_utils

    monkeypatch.setattr(ascend_utils, "is_c8_mixed_kv_overlay_enabled", lambda cfg: True)


def test_lift_is_in_place_to_lcm(patch_module):
    specs = _mixed_specs()
    patch_module._c8_mixed_overlay_lift_block_sizes(specs)
    assert all(s.block_size == 512 for s in specs.values())
    # page_size_bytes is a derived property: replace(block_size=...) recomputes it.
    assert specs["s0"].page_size_bytes == 262_144
    assert specs["l0"].page_size_bytes == 131_072  # full page keeps its true size
    assert getattr(specs["s0"], "page_size_padded", None) is None  # no-padding contract
    assert type(specs["s0"]) is SlidingWindowSpec and specs["s0"].sliding_window == 4096


def test_lift_window_guard_skips_entirely(patch_module):
    specs = {"l0": _full_c8_spec(512), "s0": _sliding_spec(128, window=256)}
    patch_module._c8_mixed_overlay_lift_block_sizes(specs)
    assert specs["s0"].block_size == 128 and specs["s0"].page_size_bytes == 65_536


def test_lift_uniform_dict_is_noop(patch_module):
    specs = {"a": _full_c8_spec(128)}
    patch_module._c8_mixed_overlay_lift_block_sizes(specs)
    assert specs["a"].block_size == 128


def test_specs_gate_accepts_mixed_true_pages(patch_module, overlay_on):
    assert patch_module._c8_mixed_overlay_true_page_specs(_config(), _mixed_specs())


def test_specs_gate_rejects_uniform_pages(patch_module, overlay_on):
    specs = {"a": _full_c8_spec(128), "b": _full_c8_spec(128)}
    assert not patch_module._c8_mixed_overlay_true_page_specs(_config(), specs)


def test_specs_gate_rejects_padded_specs(patch_module, overlay_on):
    specs = _mixed_specs()
    specs["s0"] = dataclasses.replace(specs["s0"], page_size_padded=262_144)
    assert not patch_module._c8_mixed_overlay_true_page_specs(_config(), specs)


def test_specs_gate_rejects_non_compact_layout(patch_module, overlay_on):
    assert not patch_module._c8_mixed_overlay_true_page_specs(_config(layout_compact=False), _mixed_specs())


def test_specs_gate_off_without_env(patch_module):
    # Real gate: the env flag is unset in the UT process, so the overlay must
    # stay off regardless of geometry.
    assert not patch_module._c8_mixed_overlay_true_page_specs(_config(), _mixed_specs())


def _overlay_groups() -> list[KVCacheGroupSpec]:
    return [
        KVCacheGroupSpec(layer_names=["l0"], kv_cache_spec=_full_c8_spec(512)),
        KVCacheGroupSpec(layer_names=["s0", "s1", "s2"], kv_cache_spec=_sliding_spec(128)),
    ]


def test_groups_eligible_mirrors_specs_gate(patch_module, overlay_on):
    assert patch_module._c8_mixed_overlay_groups_eligible(_config(), _overlay_groups())


def test_groups_eligible_rejects_uniform_pages(patch_module, overlay_on):
    groups = [
        KVCacheGroupSpec(layer_names=["a"], kv_cache_spec=_full_c8_spec(128)),
        KVCacheGroupSpec(layer_names=["b"], kv_cache_spec=_full_c8_spec(128)),
    ]
    assert not patch_module._c8_mixed_overlay_groups_eligible(_config(), groups)


def test_groups_eligible_rejects_padded_groups(patch_module, overlay_on):
    groups = _overlay_groups()
    groups[1].kv_cache_spec = dataclasses.replace(groups[1].kv_cache_spec, page_size_padded=262_144)
    assert not patch_module._c8_mixed_overlay_groups_eligible(_config(), groups)


def test_groups_eligible_rejects_non_compact_layout(patch_module, overlay_on):
    assert not patch_module._c8_mixed_overlay_groups_eligible(_config(layout_compact=False), _overlay_groups())


def test_groups_eligible_off_without_env(patch_module):
    assert not patch_module._c8_mixed_overlay_groups_eligible(_config(), _overlay_groups())
