# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Ascend project
import math
from collections import defaultdict
from dataclasses import replace

import torch
import vllm.v1.core.kv_cache_utils
from vllm.config import VllmConfig
from vllm.logger import logger
from vllm.model_executor.models.utils import extract_layer_index
from vllm.utils.math_utils import cdiv, round_up
from vllm.v1.core.kv_cache_utils import (
    _approximate_gcd,
    _get_per_layer_spec,
    may_override_num_blocks,
)
from vllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    KVCacheConfig,
    KVCacheGroupSpec,
    KVCacheSpec,
    KVCacheSpecKind,
    KVCacheTensor,
    MambaSpec,
    MLAAttentionSpec,
    SlidingWindowMLASpec,
    SlidingWindowSpec,
    UniformTypeKVCacheSpecs,
    get_kv_cache_spec_kind,
)

from vllm_ascend.core.kv_cache_interface import is_prefix_cacheable
from vllm_ascend.models.deepseek_v41.cache_config import (
    get_deepseek_v41_kv_cache_config,
    get_deepseek_v41_pool_bytes_per_block,
    group_cache_specs,
    is_deepseek_v41_cache,
    make_cache_groups,
)
from vllm_ascend.models.glm5next.cache_config import (
    _get_glm5_next_cache_layout,
    get_glm5_next_kv_cache_config,
    get_glm5_next_kv_cache_groups,
    get_glm5_next_max_memory_usage,
    get_glm5_next_pool_bytes_per_block,
)
from vllm_ascend.models.glm5next.kv_cache import is_glm5_next_cache_spec
from vllm_ascend.quantization.methods.kv_cache.turboquant import TURBOQUANT_CACHE_DTYPE
from vllm_ascend.quantization.methods.kv_cache.turboquant import cache as tq_cache

_KIMI_K3_TARGET_LAYER_PREFIX = "language_model.model.layers."
_KIMI_K3_DRAFT_LAYER_PREFIX = "model.layers."
_orig_resolve_kv_cache_block_sizes = vllm.v1.core.kv_cache_utils.resolve_kv_cache_block_sizes
_orig_get_kv_cache_groups_uniform_page_size = vllm.v1.core.kv_cache_utils._get_kv_cache_groups_uniform_page_size
_orig_get_kv_cache_groups = vllm.v1.core.kv_cache_utils.get_kv_cache_groups
_orig_get_packed_kv_cache_groups = vllm.v1.core.kv_cache_utils._get_packed_kv_cache_groups
_orig_get_kv_cache_config_from_groups = vllm.v1.core.kv_cache_utils.get_kv_cache_config_from_groups
_orig_max_memory_usage_bytes_from_groups = vllm.v1.core.kv_cache_utils._max_memory_usage_bytes_from_groups
_orig_pool_bytes_per_block = vllm.v1.core.kv_cache_utils._pool_bytes_per_block
if UniformTypeKVCacheSpecs.max_num_blocks_per_req is KVCacheSpec.max_num_blocks_per_req:

    def _uniform_type_max_num_blocks_per_req(
        self: UniformTypeKVCacheSpecs,
        vllm_config: VllmConfig,
        max_len: int,
    ) -> int:
        """Preserve the inner spec's block-table width."""
        widths = {spec.max_num_blocks_per_req(vllm_config, max_len) for spec in self.kv_cache_specs.values()}
        assert len(widths) == 1, (
            "All layers in the same KV cache group must need the same number "
            f"of block table entries, got {sorted(widths)}."
        )
        return next(iter(widths))

    UniformTypeKVCacheSpecs.max_num_blocks_per_req = (  # type: ignore[method-assign]
        _uniform_type_max_num_blocks_per_req
    )


def _page_sizes(spec: UniformTypeKVCacheSpecs) -> set[int]:
    """Distinct page sizes across a group's cache specs.

    vLLM #51718 removed ``UniformTypeKVCacheSpecs.get_page_sizes()`` on main;
    the underlying ``kv_cache_specs``/``page_size_bytes`` are unchanged on both
    lanes, so compute it inline.
    """
    return {s.page_size_bytes for s in spec.kv_cache_specs.values()}


def _ascend_resolve_kv_cache_block_sizes(
    kv_cache_config: KVCacheConfig,
    vllm_config: VllmConfig,
) -> tuple[int, int]:
    """Ascend-compatible resolve_kv_cache_block_sizes.

    vLLM PR #40860 added a restriction that hybrid KV cache groups with
    multiple block sizes do not support DCP.
    This restriction is correct for CUDA but not for Ascend, which implements
    context parallelism for MLA and SWA-MLA layers independently.

    For multiple KV cache groups with DCP, compute scheduler_block_size as
    lcm(group_block_sizes) * dcp to maintain alignment.
    """
    cache_config = vllm_config.cache_config
    dcp = vllm_config.parallel_config.decode_context_parallel_size
    groups = kv_cache_config.kv_cache_groups
    cacheable_groups = [group for group in groups if is_prefix_cacheable(group.kv_cache_spec)]
    filtered_private_groups = bool(cacheable_groups) and len(cacheable_groups) != len(groups)
    if filtered_private_groups and not is_deepseek_v41_cache(groups):
        # A fixed tail block is not a token-page scheduling or hashing unit.
        # Pool alignment is enforced by the GLM planner and prefix coordinator.
        kv_cache_config = replace(kv_cache_config, kv_cache_groups=cacheable_groups)
        groups = cacheable_groups

    cacheable_groups = [g for g in groups if is_prefix_cacheable(g.kv_cache_spec)]
    if len(cacheable_groups) != len(groups):
        scheduler_block_size = math.lcm(*(g.kv_cache_spec.block_size for g in groups)) * dcp
        if not cache_config.enable_prefix_caching or not cacheable_groups:
            return scheduler_block_size, scheduler_block_size
        filtered = replace(kv_cache_config, kv_cache_groups=cacheable_groups)
        if dcp == 1:
            _, hash_block_size = _orig_resolve_kv_cache_block_sizes(filtered, vllm_config)
        else:
            hash_block_size = math.gcd(*(g.kv_cache_spec.block_size for g in cacheable_groups))
        return scheduler_block_size, hash_block_size

    if len(groups) <= 1:
        # EngineCore's global size is the minimum across all original groups.
        # After filtering a private tail, it can therefore be smaller than the
        # sole cacheable group's token-page size.
        block_size = groups[0].kv_cache_spec.block_size if filtered_private_groups else cache_config.block_size
        bs = block_size * dcp
        return bs, bs

    group_block_sizes = [group.kv_cache_spec.block_size for group in groups]
    if dcp != 1:
        # Ascend supports CP with multiple KV cache groups; compute
        # scheduler_block_size using the LCM of all group block sizes
        # multiplied by DCP for proper alignment.
        scheduler_block_size = math.lcm(*group_block_sizes) * dcp
        if not cache_config.enable_prefix_caching:
            return scheduler_block_size, scheduler_block_size
        hash_block_size = math.gcd(*group_block_sizes)
        return scheduler_block_size, hash_block_size

    return _orig_resolve_kv_cache_block_sizes(kv_cache_config, vllm_config)


def _get_kimi_k3_dspark_mixed_kv_cache_groups(
    kv_cache_spec: dict[str, KVCacheSpec],
) -> list[KVCacheGroupSpec] | None:
    """Build topology-independent Kimi K3 DSpark scheduler groups.

    Target and causal draft attention layers require the same full-sequence
    block ownership. Putting them in one UniformType group lets them share one
    scheduler block table while preserving a separate physical page per layer.
    Recurrent layers are split into the fewest balanced groups whose size does
    not exceed the attention group. This minimizes scheduler groups while
    keeping the recurrent groups balanced.

    Block and page sizes are resolved by the runtime and intentionally not
    fixed here: TP8 and TP16 produce different sizes but the same ownership
    relation. An unrecognized or incompatible signature falls back to vLLM's
    generic hybrid grouping.
    """
    target_attention_specs = {
        name: spec
        for name, spec in kv_cache_spec.items()
        if name.startswith(_KIMI_K3_TARGET_LAYER_PREFIX) and isinstance(spec, FullAttentionSpec)
    }
    draft_attention_specs = {
        name: spec
        for name, spec in kv_cache_spec.items()
        if name.startswith(_KIMI_K3_DRAFT_LAYER_PREFIX) and isinstance(spec, FullAttentionSpec)
    }
    mamba_specs = {
        name: spec
        for name, spec in kv_cache_spec.items()
        if name.startswith(_KIMI_K3_TARGET_LAYER_PREFIX) and isinstance(spec, MambaSpec)
    }

    matched_layer_count = len(target_attention_specs) + len(draft_attention_specs) + len(mamba_specs)
    if (
        not target_attention_specs
        or not draft_attention_specs
        or not mamba_specs
        or matched_layer_count != len(kv_cache_spec)
    ):
        return None

    attention_specs = [*target_attention_specs.values(), *draft_attention_specs.values()]
    all_specs = [*attention_specs, *mamba_specs.values()]
    # Only attention layers share a scheduler block table. Mamba keeps its
    # own groups and may use max_model_len as block_size when cache_mode=none.
    if (
        len({spec.block_size for spec in attention_specs}) != 1
        or len({spec.page_size_bytes for spec in all_specs}) != 1
    ):
        return None

    first_mamba_spec = next(iter(mamba_specs.values()))
    if any(spec != first_mamba_spec for spec in mamba_specs.values()):
        return None

    # Insert target attention first. generate_scheduler_kv_cache_config unwraps a
    # UniformType group to its first spec, and this representative is registered
    # with the FullAttentionManager needed by both target and draft attention.
    mixed_attention_specs = {**target_attention_specs, **draft_attention_specs}
    mixed_attention_spec = UniformTypeKVCacheSpecs.from_specs(mixed_attention_specs)
    if mixed_attention_spec is None:
        return None

    groups = [
        KVCacheGroupSpec(
            layer_names=list(mixed_attention_specs),
            kv_cache_spec=mixed_attention_spec,
        )
    ]
    mamba_layer_names = list(mamba_specs)
    mamba_group_count = cdiv(len(mamba_layer_names), len(mixed_attention_specs))
    for group_idx in range(mamba_group_count):
        layer_names = mamba_layer_names[group_idx::mamba_group_count]
        group_specs = {name: mamba_specs[name] for name in layer_names}
        uniform_mamba_spec = UniformTypeKVCacheSpecs.from_specs(group_specs)
        assert uniform_mamba_spec is not None
        groups.append(
            KVCacheGroupSpec(
                layer_names=layer_names,
                kv_cache_spec=uniform_mamba_spec,
            )
        )

    logger.info(
        "Using Kimi K3 DSpark mixed KV grouping: %d target + %d draft attention layers, followed by Mamba groups %s",
        len(target_attention_specs),
        len(draft_attention_specs),
        [len(group.layer_names) for group in groups[1:]],
    )
    return groups


def _get_kv_cache_groups_uniform_page_size(
    kv_cache_spec: dict[str, KVCacheSpec],
) -> list[KVCacheGroupSpec]:
    kimi_k3_groups = _get_kimi_k3_dspark_mixed_kv_cache_groups(kv_cache_spec)
    if kimi_k3_groups is not None:
        return kimi_k3_groups
    return _orig_get_kv_cache_groups_uniform_page_size(kv_cache_spec)


def _kv_cache_config_has_mamba_layers(self: KVCacheConfig) -> bool:
    """Recognize Mamba layers nested in UniformType cache groups."""
    return any(get_kv_cache_spec_kind(group.kv_cache_spec) == KVCacheSpecKind.MAMBA for group in self.kv_cache_groups)


def group_and_unify_kv_cache_specs(
    kv_cache_spec: dict[str, KVCacheSpec],
) -> list[UniformTypeKVCacheSpecs] | None:
    """
    Group the KV cache specs and unify each group into one UniformTypeKVCacheSpecs.
    Currently, this is only used for DeepseekV4.
    """
    if not any(isinstance(spec, SlidingWindowMLASpec) for spec in kv_cache_spec.values()):
        return None

    logical_block_specs: dict[int, dict[str, KVCacheSpec]] = defaultdict(dict)
    grouped_swa_mla_specs: dict[int, dict[str, KVCacheSpec]] = defaultdict(dict)
    for name, spec in kv_cache_spec.items():
        if isinstance(spec, SlidingWindowMLASpec):
            grouped_swa_mla_specs[spec.block_size][name] = spec
        elif isinstance(spec, MLAAttentionSpec):
            logical_block_specs[spec.block_size][name] = spec

    mla_uniform_specs = []
    for block_size in sorted(logical_block_specs):
        spec_dict = logical_block_specs[block_size]
        assert len(spec_dict) > 0
        mla_uniform_specs.append(UniformTypeKVCacheSpecs.from_specs(spec_dict))
    assert mla_uniform_specs is not None

    swa_uniform_specs: list[UniformTypeKVCacheSpecs] = []
    for spec_dict in grouped_swa_mla_specs.values():
        uniform_spec = UniformTypeKVCacheSpecs.from_specs(spec_dict)
        assert uniform_spec is not None
        swa_uniform_specs.append(uniform_spec)

    return [*mla_uniform_specs, *swa_uniform_specs]


def _get_kv_cache_groups_uniform_groups(
    grouped_specs: list[UniformTypeKVCacheSpecs],
) -> list[KVCacheGroupSpec]:
    """
    Generate the KV cache groups from the grouped specs.
    """
    assert len(grouped_specs) > 0 and all(isinstance(spec, UniformTypeKVCacheSpecs) for spec in grouped_specs)
    if any(
        getattr(spec, "cache_dtype_str", None) == TURBOQUANT_CACHE_DTYPE
        for group in grouped_specs
        for spec in group.kv_cache_specs.values()
    ):
        return tq_cache.group_specs(grouped_specs)
    # For now, we restrict the first grouped_spec to be UniformTypeKVCacheSpecs
    # containing only MLAAttentionSpec.
    full_mla_spec = grouped_specs[0]
    full_mla_c128_spec = grouped_specs[1]

    assert all(isinstance(spec, MLAAttentionSpec) for spec in full_mla_spec.kv_cache_specs.values())
    full_mla_group = KVCacheGroupSpec(
        layer_names=list(full_mla_spec.kv_cache_specs.keys()),
        kv_cache_spec=full_mla_spec,
    )
    full_mla_c128_group = KVCacheGroupSpec(
        layer_names=list(full_mla_c128_spec.kv_cache_specs.keys()),
        kv_cache_spec=full_mla_c128_spec,
    )

    # We define a layer tuple as a group of layers with different page sizes, and
    # one UniformTypeKVCacheSpecs contains a list of layer tuples.
    # For example, if we have 11 C4 layers and 10 C128 layers, we can define a layer
    # tuple as [C4I, C4A, C128], and the full_mla_group will contain "11" layer tuples.
    # The other uniform KV cache specs will be similarly partitioned into layer tuples.
    # Say we have 21 SWA layers, all with the same page size, then we will have "21"
    # layer tuples.
    num_layer_tuples_per_group: list[int] = [_get_max_layers_per_page_size(g_spec) for g_spec in grouped_specs]
    # Choose `num_layer_tuples` to minimize total padding across groups.
    num_layer_tuples = _approximate_gcd(num_layer_tuples_per_group, lower_bound=num_layer_tuples_per_group[0])
    # Round up to the nearest multiple of `num_layer_tuples` (i.e., padding)
    num_layer_tuples_per_group = [round_up(x, num_layer_tuples) for x in num_layer_tuples_per_group]

    # TODO(cmq): this is not general enough
    swa_mla_specs = grouped_specs[2:]

    assert all(
        isinstance(spec, SlidingWindowMLASpec) for group in swa_mla_specs for spec in group.kv_cache_specs.values()
    )

    # Split each SWA UniformKV group into smaller groups to align their #(layer tuples)
    # Possibly padding layer tuples for this.
    # Additionally, we also pad KV blocks in each SWA layer, to align the page size
    # with the corresponding layer in the full-MLA group.
    all_page_sizes = _page_sizes(full_mla_spec)
    swa_mla_groups = []
    for sm_spec in swa_mla_specs:
        sm_page_sizes = _page_sizes(sm_spec)
        layers_per_size: dict[int, list[str]] = defaultdict(list)
        assert max(sm_page_sizes) <= max(all_page_sizes)

        # Unify page size by padding layers' page_size to the nearest larger page_size.
        # Compute candidate (nearest larger page_size) for each unique page size.
        size_to_candidate: dict[int, int] = {}
        for ps in sm_page_sizes:
            size_to_candidate[ps] = min(x for x in all_page_sizes if x >= ps)
        # Pad and collect layer names per page size.
        for layer_name, layer_spec in sm_spec.kv_cache_specs.items():
            current_size = layer_spec.page_size_bytes
            candidate = size_to_candidate[current_size]
            if current_size < candidate:
                object.__setattr__(layer_spec, "page_size_padded", candidate)
            layers_per_size[candidate].append(layer_name)
        # NOTE(yifan): for now, inside a UniformKV group, each page_size should
        # have the same number of layers. This also means we don't need to pad layers
        # inside a partial-full layer tuple.
        assert len(set(len(layers) for layers in layers_per_size.values())) == 1
        num_layers_per_size = len(next(iter(layers_per_size.values())))

        # Split layers inside each UniformKV group for aligned #(layers).
        # See `_get_kv_cache_groups_uniform_page_size` for more details.
        num_tuple_groups = cdiv(num_layers_per_size, num_layer_tuples)
        layer_tuples = list(zip(*layers_per_size.values()))
        for i in range(num_tuple_groups):
            group_layer_tuples = layer_tuples[i::num_tuple_groups]
            # Flatten tuples and build dict for from_specs
            group_layer_names = [name for layer_tuple in group_layer_tuples for name in layer_tuple]
            group_layer_specs = {name: sm_spec.kv_cache_specs[name] for name in group_layer_names}
            sub_sm_spec = UniformTypeKVCacheSpecs.from_specs(group_layer_specs)
            assert sub_sm_spec is not None
            swa_mla_groups.append(
                KVCacheGroupSpec(
                    layer_names=group_layer_names,
                    kv_cache_spec=sub_sm_spec,
                )
            )

    return [full_mla_group, full_mla_c128_group, *swa_mla_groups]


def _get_max_layers_per_page_size(spec: UniformTypeKVCacheSpecs) -> int:
    """Bridge the UniformTypeKVCacheSpecs helper renamed by vLLM #53896."""
    return spec.get_max_layers_per_page_size()


def _ascend_get_packed_kv_cache_groups(
    vllm_config: VllmConfig,
    kv_cache_spec: dict[str, KVCacheSpec],
) -> list[KVCacheGroupSpec] | None:
    """Preserve Ascend's DSV4 grouping on the live packed-group hook."""
    if is_deepseek_v41_cache(kv_cache_spec):
        grouped_specs = group_cache_specs(kv_cache_spec)
        groups = make_cache_groups(grouped_specs)
    else:
        grouped_specs = group_and_unify_kv_cache_specs(kv_cache_spec)
        if grouped_specs is None:
            assert _orig_get_packed_kv_cache_groups is not None
            return _orig_get_packed_kv_cache_groups(vllm_config, kv_cache_spec)
        groups = _get_kv_cache_groups_uniform_groups(grouped_specs)
    vllm.v1.core.kv_cache_utils._annotate_eagle_groups(
        vllm_config,
        kv_cache_spec,
        groups,
        use_deepseek_v4_fallback=True,
    )
    vllm.v1.core.kv_cache_utils._warn_if_unannotated_eagle_mamba(
        vllm_config,
        groups,
    )
    return groups


def _get_deepseek_v4_cache_layout(
    kv_cache_groups: list[KVCacheGroupSpec],
) -> tuple[list[int], list[dict[int, list[str]]], list[str], int, int]:
    """Return the geometry shared by DSV4 planning and rank normalization.

    Precondition: kv_cache_groups[0] is the full-MLA group; its page sizes
    define the canonical bucket set. Non-full-MLA groups must have been
    page_size-padded upstream (see _get_kv_cache_groups_uniform_groups) so
    every layer's page_size matches one of the full-MLA bucket sizes.

    For each group, bucket its layers by page_size_bytes and place each layer
    at tuple_idx = position-within-bucket.
    """
    full_mla_spec = kv_cache_groups[0].kv_cache_spec
    assert isinstance(full_mla_spec, UniformTypeKVCacheSpecs)
    page_sizes = sorted(_page_sizes(full_mla_spec))

    # Pre-bucket each group's layers by page_size (registration order within
    # bucket). bucketed[g_idx][page_size] = [layer_name, ...].
    mtp_layer_names = []
    mtp_page_size = 0
    bucketed: list[dict[int, list[str]]] = []
    for group in kv_cache_groups:
        assert isinstance(group.kv_cache_spec, UniformTypeKVCacheSpecs)
        specs = group.kv_cache_spec.kv_cache_specs
        b: dict[int, list[str]] = defaultdict(list)
        for name in group.layer_names:
            if "mtp" not in name:
                b[specs[name].page_size_bytes].append(name)
            else:
                mtp_layer_names.append(name)
                mtp_page_size = specs[name].page_size_bytes
        bucketed.append(b)

    # num_layer_tuples = longest bucket list across all groups. For the
    # full-MLA group this equals the count of layers in the largest
    # per-page-size bucket (= get_num_layer_tuples()); for SWA sub-groups
    # this equals the sub-group size (each has a single page_size).
    num_layer_tuples = max(len(layers) for b in bucketed for layers in b.values()) + len(mtp_layer_names)

    return page_sizes, bucketed, mtp_layer_names, mtp_page_size, num_layer_tuples


def _get_kv_cache_config_deepseek_v4_main(
    vllm_config: VllmConfig,
    kv_cache_groups: list[KVCacheGroupSpec],
    available_memory: int,
) -> tuple[int, list[KVCacheTensor]]:
    if tq_cache.uses_turboquant_groups(kv_cache_groups):
        return tq_cache.cache_config(vllm_config, kv_cache_groups, available_memory)
    (
        page_sizes,
        bucketed,
        mtp_layer_names,
        mtp_page_size,
        num_tuple_slots,
    ) = _get_deepseek_v4_cache_layout(kv_cache_groups)

    bytes_per_tuple = sum(page_sizes)
    num_blocks = available_memory // (bytes_per_tuple * num_tuple_slots)
    num_blocks = may_override_num_blocks(vllm_config, num_blocks)

    tuple_stride = bytes_per_tuple * num_blocks
    backing_size = tuple_stride * num_tuple_slots

    # Within every tuple slot, page-size buckets are placed consecutively.
    page_offsets: dict[int, int] = {}
    page_prefix = 0
    for page_size in page_sizes:
        page_offsets[page_size] = page_prefix * num_blocks
        page_prefix += page_size

    tensors: list[KVCacheTensor] = []

    # Keep each descriptor inside one cache group. Descriptors from different
    # groups alias corresponding tuple slots by using identical geometry.
    for group_buckets in bucketed:
        for page_size in page_sizes:
            layer_names = group_buckets.get(page_size)
            if not layer_names:
                continue

            tensors.append(
                KVCacheTensor(
                    size=backing_size,
                    layers=list(layer_names),
                    offset=page_offsets[page_size],
                    layer_stride=tuple_stride,
                    block_stride=page_size,
                )
            )

    # MTP layers receive trailing tuple slots. Unused page buckets remain
    # padding so num_blocks and memory accounting retain the existing contract.
    normal_tuple_slots = num_tuple_slots - len(mtp_layer_names)
    for index, layer_name in enumerate(mtp_layer_names):
        slot = normal_tuple_slots + index
        tensors.append(
            KVCacheTensor(
                size=backing_size,
                layers=[layer_name],
                offset=slot * tuple_stride + page_offsets[mtp_page_size],
                layer_stride=0,
                block_stride=mtp_page_size,
            )
        )

    return num_blocks, tensors


def _is_deepseek_v4_groups(kv_cache_groups: list[KVCacheGroupSpec]) -> bool:
    if (
        is_deepseek_v41_cache(kv_cache_groups)
        or not kv_cache_groups
        or not all(isinstance(group.kv_cache_spec, UniformTypeKVCacheSpecs) for group in kv_cache_groups)
    ):
        return False
    for group in kv_cache_groups:
        group_spec = group.kv_cache_spec
        assert isinstance(group_spec, UniformTypeKVCacheSpecs)
        specs = group_spec.kv_cache_specs.values()
        if any(getattr(spec, "model_version", None) == "deepseek_v4" for spec in specs):
            return True
    return False


def _ascend_pool_bytes_per_block(kv_cache_groups: list[KVCacheGroupSpec]) -> int:
    """Use the same DSV4 divisor as Ascend's shared-tuple planner.

    vLLM #51718 re-plans ranks with more KV memory using
    ``min_num_blocks * _pool_bytes_per_block(groups)``. Its standardized
    per-group layout has a different divisor from Ascend's DSV4 shared-tuple
    layout, so using the upstream value changes ``num_blocks`` during the
    re-plan and leaves ranks inconsistent.
    """
    if tq_cache.uses_turboquant_groups(kv_cache_groups):
        return tq_cache.pool_bytes_per_block(kv_cache_groups)
    if is_deepseek_v41_cache(kv_cache_groups):
        return get_deepseek_v41_pool_bytes_per_block(kv_cache_groups)
    if _get_glm5_next_cache_layout(kv_cache_groups) is not None:
        return get_glm5_next_pool_bytes_per_block(kv_cache_groups)
    if not _is_deepseek_v4_groups(kv_cache_groups):
        return _orig_pool_bytes_per_block(kv_cache_groups)

    page_sizes, _, _, _, num_layer_tuples = _get_deepseek_v4_cache_layout(kv_cache_groups)
    return sum(page_sizes) * num_layer_tuples


def _ascend_max_memory_usage_bytes_from_groups(
    vllm_config: VllmConfig,
    kv_cache_groups: list[KVCacheGroupSpec],
) -> int:
    """Keep the pre-#51718 DSV4 admission formula for its shared tuples."""
    if tq_cache.uses_turboquant_groups(kv_cache_groups):
        return tq_cache.max_memory_usage(vllm_config, kv_cache_groups)
    if _get_glm5_next_cache_layout(kv_cache_groups) is not None:
        return get_glm5_next_max_memory_usage(vllm_config, kv_cache_groups)
    if not _is_deepseek_v4_groups(kv_cache_groups):
        return _orig_max_memory_usage_bytes_from_groups(vllm_config, kv_cache_groups)

    assert all(isinstance(group.kv_cache_spec, UniformTypeKVCacheSpecs) for group in kv_cache_groups)
    full_mla_spec = kv_cache_groups[0].kv_cache_spec
    assert isinstance(full_mla_spec, UniformTypeKVCacheSpecs)
    layer_tuple_bytes = sum(_page_sizes(full_mla_spec))
    num_layer_tuples = max(
        _get_max_layers_per_page_size(group.kv_cache_spec)
        for group in kv_cache_groups
        if isinstance(group.kv_cache_spec, UniformTypeKVCacheSpecs)
    )
    return sum(
        num_layer_tuples * group.kv_cache_spec.max_memory_usage_pages(vllm_config) * layer_tuple_bytes
        for group in kv_cache_groups
        if isinstance(group.kv_cache_spec, UniformTypeKVCacheSpecs)
    )


def _c8_mixed_overlay_true_page_specs(
    vllm_config: VllmConfig,
    kv_cache_spec: dict[str, KVCacheSpec],
) -> bool:
    """Recognize the C8 mixed-page overlay spec set (true per-group pages).

    The platform skip-align hook kept ``skip_page_size_padded`` unset, so the
    sliding specs carry their true bf16 pages while the C8 specs carry the
    264 B/token quantized page. ``unify_kv_cache_spec_page_size`` would pad
    the smaller page back up to the max (recovering the waste this mode
    removes), so the caller must group these specs WITHOUT unification.
    """
    from vllm_ascend.utils import is_c8_mixed_kv_overlay_enabled

    if not kv_cache_spec or len({s.page_size_bytes for s in kv_cache_spec.values()}) <= 1:
        return False
    has_c8 = any(
        type(spec) is FullAttentionSpec and spec.dtype == torch.float8_e4m3fn
        for spec in kv_cache_spec.values()
    )
    if not has_c8:
        return False
    if not all(type(spec) in (FullAttentionSpec, SlidingWindowSpec) for spec in kv_cache_spec.values()):
        return False
    if any(getattr(spec, "page_size_padded", None) is not None for spec in kv_cache_spec.values()):
        # A padded spec means the uniform-page lift still ran; keep the
        # upstream path rather than mixing geometries.
        return False
    return is_c8_mixed_kv_overlay_enabled(vllm_config)


def _get_c8_mixed_overlay_groups(
    vllm_config: VllmConfig,
    kv_cache_spec: dict[str, KVCacheSpec],
) -> list[KVCacheGroupSpec]:
    """Bucket the overlay spec set into groups WITHOUT size normalization.

    ``_get_kv_cache_groups_uniform_page_size`` ends with an n:1-pattern
    heuristic (group size = smallest bucket, then split + pad every bucket to
    that size) that exists for the uniform shared-page pool. The block-outer
    overlay does not need equal layer counts across groups - each group
    aliases the same backing from byte 0 and a block ID belongs to one group
    at a time - and the heuristic degenerates to one group per layer when a
    singleton bucket exists (e.g. the MTP draft layer's own
    ``SlidingWindowSpec(512)``): 89 single-layer groups shatter the pool
    geometry.

    Two deviations from the upstream bucket+merge loop matter here:

    * Buckets form by EXACT spec equality, not ``merge()``. merge() only
      asserts page geometry, so all 63 same-geometry sliding layers collapse
      into one bucket - a 63-layer sliding group pins bytes_per_block at
      63 x page (16.5 MiB post-lift) and ~80% of every full-group slot
      becomes waste. The design's repeat-pattern split (one group per
      sliding position within the hybrid block) keeps the pin at
      21 x page (5,505,024 B, the anchor bpb).
    * A sliding bucket larger than one hybrid-block run splits by the
      layer's position within its sliding run (k % per_run). For IQuest-Q1
      the hybrid block is [full, sliding, sliding, sliding] x 21 repeats,
      so the 63 sliding layers split into three 21-layer groups - exactly
      the S1 production grouping [26 full, 21, 21, 21] with the draft now
      carried by its own 512-window spec instead of the full group.
    """
    same_type_layers: dict[KVCacheSpec, list[str]] = defaultdict(list)
    for layer_name, layer_spec in kv_cache_spec.items():
        same_type_layers[layer_spec].append(layer_name)

    per_run = 0
    hybrid_block = getattr(vllm_config.model_config.hf_config, "hybrid_layers_types_block", None)
    if hybrid_block:
        per_run = sum(1 for t in hybrid_block if t != "full_attention")

    layer_buckets: list[list[str]] = []
    for layer_spec, layer_names in same_type_layers.items():
        if (
            per_run > 1
            and isinstance(layer_spec, SlidingWindowSpec)
            and len(layer_names) > per_run
        ):
            ordered = sorted(layer_names, key=extract_layer_index)
            for pos in range(per_run):
                layer_buckets.append(ordered[pos::per_run])
        else:
            layer_buckets.append(list(layer_names))
    return vllm.v1.core.kv_cache_utils.create_kv_cache_group_specs(kv_cache_spec, layer_buckets)


def _c8_mixed_overlay_lift_block_sizes(
    kv_cache_spec: dict[str, KVCacheSpec],
) -> None:
    """Lift heterogeneous kernel block sizes to their LCM, keep true pages.

    The platform hook skips the upstream ``_align_heterogeneous_kv_block_size``
    wholesale, but that call does two unrelated things: lift kernel block
    sizes across spec kinds AND pad pages to a shared size. The overlay needs
    only the lift. Without it the sliding specs keep their native block (128
    on IQuest, regardless of ``--block-size``), so a 131,072-token request
    demands ``cdiv(N, 128) + 1 = 161`` sliding pages per group instead of 41,
    which measured 18.01x max concurrency on Ascend 950DT (2026-11-08, node6
    s2-c8 engine log) - the full-C8 saving entirely eaten by sliding block
    demand. With the lift: sliding pages 512 x 512 = 262,144, bpb
    max(26 x 135,168, 21 x 262,144) = 5,505,024, and the concurrency lands at
    the design doc's 1.238x over the lifted-bf16 block-512 baseline.

    Mutates ``kv_cache_spec`` in place so every downstream consumer (groups,
    block-size resolution, backend builders) sees one geometry.
    ``page_size_bytes`` is a derived property on the spec dataclasses, so
    ``replace(block_size=...)`` recomputes pages; ``page_size_padded`` stays
    unset, which keeps the no-padding contract of this mode. If the LCM ever
    exceeded a sliding window, the lift is skipped entirely (capacity gain
    lost, read path preserved) rather than mixing block sizes.
    """
    blocks = {spec.block_size for spec in kv_cache_spec.values()}
    if len(blocks) <= 1:
        return
    common = math.lcm(*blocks)
    windows = [
        spec.sliding_window
        for spec in kv_cache_spec.values()
        if type(spec) is SlidingWindowSpec
    ]
    if windows and common > min(windows):
        logger.warning(
            "C8 mixed-page KV overlay: skip block lift to %d, sliding window "
            "%d is smaller; sliding groups keep their native block and the "
            "capacity gain is reduced.",
            common,
            min(windows),
        )
        return
    for name, spec in kv_cache_spec.items():
        if spec.block_size != common:
            kv_cache_spec[name] = replace(spec, block_size=common)


def _get_glm5_next_kv_cache_groups(
    vllm_config: VllmConfig,
    kv_cache_spec: dict[str, KVCacheSpec],
) -> list[KVCacheGroupSpec]:
    if any(is_glm5_next_cache_spec(spec) for spec in kv_cache_spec.values()):
        return get_glm5_next_kv_cache_groups(vllm_config, kv_cache_spec)
    if _c8_mixed_overlay_true_page_specs(vllm_config, kv_cache_spec):
        # True per-group pages: lift kernel block sizes to their LCM (the
        # sliding specs arrive with their native 128-token block and would
        # otherwise inflate per-request sliding block demand 4x), then bucket
        # by exact spec value (the draft's own 512-window spec stays a
        # singleton group) and split the sliding bucket by within-run
        # position (63 sliding layers -> three 21-layer groups), skipping
        # ``unify_kv_cache_spec_page_size`` which pads the smaller page back
        # to the max. Buckets are NOT normalized to an equal layer count
        # (see _get_c8_mixed_overlay_groups).
        _c8_mixed_overlay_lift_block_sizes(kv_cache_spec)
        groups = _get_c8_mixed_overlay_groups(vllm_config, kv_cache_spec)
        vllm.v1.core.kv_cache_utils._annotate_eagle_groups(vllm_config, kv_cache_spec, groups)
        vllm.v1.core.kv_cache_utils._warn_if_unannotated_eagle_mamba(vllm_config, groups)
        group_pages = [
            sorted({spec.page_size_bytes for spec in group.kv_cache_spec.kv_cache_specs.values()})
            if isinstance(group.kv_cache_spec, UniformTypeKVCacheSpecs)
            else [group.kv_cache_spec.page_size_bytes]
            for group in groups
        ]
        logger.info(
            "C8 mixed-page KV overlay: grouped %d layers into %d groups with "
            "true per-group pages %s",
            len(kv_cache_spec),
            len(groups),
            group_pages,
        )
        return groups
    return _orig_get_kv_cache_groups(vllm_config, kv_cache_spec)


def _get_c8_mixed_overlay_kv_cache_config(
    vllm_config: VllmConfig,
    kv_cache_groups: list[KVCacheGroupSpec],
    available_memory: int,
) -> KVCacheConfig:
    """Build block-outer descriptors over one shared pool of true pages.

    Every group aliases the whole backing from byte 0: layer ``i`` of block
    ``b`` lives at ``i * page_g + b * bytes_per_block`` with
    ``bytes_per_block = max(group sum of true pages)``. A block ID belongs to
    one group at a time, so groups may reuse the same physical slots. The
    largest group pins ``bytes_per_block``; with the C8-mixed layer split
    that is the bf16 sliding group (21 x 262,144 = 5,505,024 per 512-token
    block), and ``num_blocks = available_memory // bytes_per_block`` is the
    token-normalized budget the design doc requires (no per-layer scaling).
    """
    if not vllm_config.use_v2_model_runner:
        raise ValueError(
            "The C8 mixed-page KV overlay requires the V2 model runner "
            "(set VLLM_USE_V2_MODEL_RUNNER=1)."
        )
    group_pages: list[int] = []
    for group in kv_cache_groups:
        specs = (
            group.kv_cache_spec.kv_cache_specs.values()
            if isinstance(group.kv_cache_spec, UniformTypeKVCacheSpecs)
            else (group.kv_cache_spec,)
        )
        pages = {spec.page_size_bytes for spec in specs}
        if len(pages) != 1:
            raise ValueError(
                "C8 mixed-page overlay groups must carry one page size per group, "
                f"got {sorted(pages)}."
            )
        group_pages.append(len(group.layer_names) * pages.pop())
    bytes_per_block = max(group_pages)
    num_blocks = may_override_num_blocks(vllm_config, available_memory // bytes_per_block)
    if num_blocks <= 0:
        raise ValueError(
            f"C8 mixed-page overlay needs a positive block count, got {num_blocks} "
            f"from {available_memory} bytes // {bytes_per_block}."
        )
    size = bytes_per_block * num_blocks
    kv_cache_tensors: list[KVCacheTensor] = []
    for group in kv_cache_groups:
        spec = group.kv_cache_spec
        page = (
            next(iter(spec.kv_cache_specs.values())).page_size_bytes
            if isinstance(spec, UniformTypeKVCacheSpecs)
            else spec.page_size_bytes
        )
        kv_cache_tensors.append(
            KVCacheTensor(
                size=size,
                layers=list(group.layer_names),
                offset=0,
                layer_stride=page,
                block_stride=bytes_per_block,
            )
        )
    logger.info(
        "C8 mixed-page KV overlay: bytes_per_block=%d over %d groups, "
        "num_blocks=%d, pool=%d MiB, %d B/token per block slot",
        bytes_per_block,
        len(kv_cache_groups),
        num_blocks,
        size >> 20,
        bytes_per_block
        // _get_per_layer_spec(kv_cache_groups[0], kv_cache_groups[0].layer_names[0]).block_size,
    )
    return KVCacheConfig(
        num_blocks=num_blocks,
        kv_cache_tensors=kv_cache_tensors,
        kv_cache_groups=kv_cache_groups,
        prefix_cache_retention_interval=vllm_config.cache_config.prefix_cache_retention_interval,
    )


def _ascend_get_kv_cache_config_from_groups(
    vllm_config: VllmConfig,
    kv_cache_groups: list[KVCacheGroupSpec],
    available_memory: int,
) -> KVCacheConfig:
    """Restore Ascend's DSV4 shared-tuple planner removed by vLLM #51718.

    Attach ``kv_transfer_config`` onto every built config: ``KVCacheConfig``
    has no native field, and the PD prefill-producer role read back by the
    kv-cache coordinator (see ``patch_kv_cache_coordinator.py``) must ride on
    the config object - the coordinator factory receives ``KVCacheConfig``
    but not ``VllmConfig``. The attribute survives the scheduler-side
    ``copy.deepcopy`` in ``generate_scheduler_kv_cache_config`` and is
    dropped by worker pickle IPC, which never reads it.
    """
    from vllm_ascend.utils import is_c8_mixed_kv_overlay_enabled

    if is_deepseek_v41_cache(kv_cache_groups):
        kv_cache_config = get_deepseek_v41_kv_cache_config(vllm_config, kv_cache_groups, available_memory)
    elif _get_glm5_next_cache_layout(kv_cache_groups) is not None:
        kv_cache_config = get_glm5_next_kv_cache_config(vllm_config, kv_cache_groups, available_memory)
    elif any(
        type(_get_per_layer_spec(group, group.layer_names[0])) is FullAttentionSpec
        and _get_per_layer_spec(group, group.layer_names[0]).dtype == torch.float8_e4m3fn
        for group in kv_cache_groups
        if group.layer_names
    ) and is_c8_mixed_kv_overlay_enabled(vllm_config):
        kv_cache_config = _get_c8_mixed_overlay_kv_cache_config(vllm_config, kv_cache_groups, available_memory)
    elif not _is_deepseek_v4_groups(kv_cache_groups):
        kv_cache_config = _orig_get_kv_cache_config_from_groups(vllm_config, kv_cache_groups, available_memory)
    else:
        num_blocks, kv_cache_tensors = _get_kv_cache_config_deepseek_v4_main(
            vllm_config,
            kv_cache_groups,
            available_memory,
        )
        kv_cache_config = KVCacheConfig(
            num_blocks=num_blocks,
            kv_cache_tensors=kv_cache_tensors,
            kv_cache_groups=kv_cache_groups,
            prefix_cache_retention_interval=vllm_config.cache_config.prefix_cache_retention_interval,
        )
    kv_cache_config.kv_transfer_config = getattr(vllm_config, "kv_transfer_config", None)
    speculative_config = getattr(vllm_config, "speculative_config", None)
    if speculative_config is not None and speculative_config.method == "dspark":
        from vllm_ascend.distributed.kv_transfer.kv_p2p.sfa_pd_rd2h.dspark_context import resident_mla_context_group_ids

        # Upstream scheduler conversion keeps only the first spec in a uniform
        # group, losing draft residency when target and draft share block IDs.
        # Preserve the worker-derived ownership through that deepcopy. Remove
        # this bridge once upstream KVCacheConfig carries per-layer ownership.
        kv_cache_config.dspark_context_group_ids = resident_mla_context_group_ids(kv_cache_config.kv_cache_groups)
        draft_only_group_ids = []
        for group_id, group in enumerate(kv_cache_config.kv_cache_groups):
            group_spec = group.kv_cache_spec
            specs = (
                tuple(group_spec.kv_cache_specs.values())
                if isinstance(group_spec, UniformTypeKVCacheSpecs)
                else (group_spec,)
            )
            # Keep mixed target/draft groups in the ordinary target lookup.
            # Empty PP-local layer names still carry the global group spec.
            if specs and all(
                isinstance(spec, MLAAttentionSpec) and spec.non_causal_multi_token_decode for spec in specs
            ):
                draft_only_group_ids.append(group_id)
        kv_cache_config.dspark_draft_only_group_ids = tuple(draft_only_group_ids)
    return kv_cache_config


vllm.v1.core.kv_cache_utils.resolve_kv_cache_block_sizes = _ascend_resolve_kv_cache_block_sizes
assert _orig_get_packed_kv_cache_groups is not None
vllm.v1.core.kv_cache_utils._get_packed_kv_cache_groups = _ascend_get_packed_kv_cache_groups
vllm.v1.core.kv_cache_utils._get_kv_cache_groups_uniform_page_size = _get_kv_cache_groups_uniform_page_size
vllm.v1.core.kv_cache_utils.get_kv_cache_groups = _get_glm5_next_kv_cache_groups
KVCacheConfig.has_mamba_layers = property(  # type: ignore[assignment]
    _kv_cache_config_has_mamba_layers
)
vllm.v1.core.kv_cache_utils.get_kv_cache_config_from_groups = _ascend_get_kv_cache_config_from_groups
vllm.v1.core.kv_cache_utils._max_memory_usage_bytes_from_groups = _ascend_max_memory_usage_bytes_from_groups
vllm.v1.core.kv_cache_utils._pool_bytes_per_block = _ascend_pool_bytes_per_block

# Also patch the reference used by engine/core.py which imports the function directly.
import vllm.v1.engine.core  # noqa: E402

vllm.v1.engine.core.resolve_kv_cache_block_sizes = _ascend_resolve_kv_cache_block_sizes
