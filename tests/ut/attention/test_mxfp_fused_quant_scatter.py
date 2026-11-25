# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project

from types import SimpleNamespace
from unittest.mock import patch

from tests.ut.base import TestBase
from vllm_ascend.attention.attention_c8_mxfp import (
    FUSED_QUANT_SCATTER_MAX_TOKENS,
    should_fuse_quant_scatter,
)

_CFG = "vllm_ascend.attention.attention_c8_mxfp.get_ascend_config"


def _with(enabled: int):
    return patch(_CFG, lambda: SimpleNamespace(enable_c8_fused_quant_scatter=enabled))


class TestShouldFuseQuantScatter(TestBase):
    def test_disabled_by_default(self):
        with _with(0):
            self.assertFalse(should_fuse_quant_scatter(1))
            self.assertFalse(should_fuse_quant_scatter(FUSED_QUANT_SCATTER_MAX_TOKENS))

    def test_enabled_decode_sized_batches(self):
        with _with(1):
            self.assertTrue(should_fuse_quant_scatter(1))
            self.assertTrue(should_fuse_quant_scatter(32))
            self.assertTrue(should_fuse_quant_scatter(FUSED_QUANT_SCATTER_MAX_TOKENS))

    def test_enabled_but_prefill_sized_batch_falls_back(self):
        with _with(1):
            self.assertFalse(should_fuse_quant_scatter(FUSED_QUANT_SCATTER_MAX_TOKENS + 1))
            self.assertFalse(should_fuse_quant_scatter(65536))

    def test_empty_batch_never_fuses(self):
        with _with(1):
            self.assertFalse(should_fuse_quant_scatter(0))
