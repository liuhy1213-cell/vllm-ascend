import unittest
import weakref
from unittest.mock import patch

import torch


class _KernelRecorder:
    """CPU stand-in for the Triton precompute kernel."""

    def __init__(self):
        self.calls = []

    def __getitem__(self, grid):
        return self._launch

    def _launch(self, positions, cos_sin_cache, out, *args, **kwargs):
        self.calls.append(positions.clone())
        out.fill_(float(len(self.calls)))


class _ForwardContext:
    pass


class TestRopeCosSinStepCache(unittest.TestCase):
    def test_default_off_recomputes_when_positions_change(self):
        from vllm_ascend.ops.triton.linearnorm import split_qkv_rmsnorm_rope_simt as rope

        # The original implementation defaulted to enabled and returned this
        # stale tensor without invoking the precompute kernel.
        self.assertFalse(rope._ENABLE_ROPE_COS_SIN_STEP_CACHE)
        ctx = _ForwardContext()
        stale = torch.full((2, 4), 99.0)
        original_cache = rope._ROPE_COS_SIN_CACHE.copy()
        rope._ROPE_COS_SIN_CACHE.update(ctx_ref=weakref.ref(ctx), tensor=stale)
        recorder = _KernelRecorder()
        try:
            with patch.object(rope, "precompute_rope_cos_sin_kernel", recorder):
                first = rope._rope_cos_sin_for_step(torch.tensor([10, 11]), None, 2, 4, 1, "cpu")
                second = rope._rope_cos_sin_for_step(torch.tensor([20, 21]), None, 2, 4, 1, "cpu")
        finally:
            rope._ROPE_COS_SIN_CACHE.clear()
            rope._ROPE_COS_SIN_CACHE.update(original_cache)

        self.assertEqual(len(recorder.calls), 2)
        self.assertEqual(recorder.calls[0].tolist(), [10, 11])
        self.assertEqual(recorder.calls[1].tolist(), [20, 21])
        self.assertTrue(torch.equal(first, torch.ones(2, 4)))
        self.assertTrue(torch.equal(second, torch.full((2, 4), 2.0)))
        self.assertNotEqual(first.data_ptr(), second.data_ptr())


if __name__ == "__main__":
    unittest.main()
