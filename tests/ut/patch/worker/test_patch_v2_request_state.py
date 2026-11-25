import importlib.util
import unittest
from pathlib import Path
import torch

spec = importlib.util.spec_from_file_location('draft_fix', Path(__file__).resolve().parents[4] / 'vllm_ascend/patch/worker/patch_v2/patch_request_state.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)

class TestInvalidDraftSlots(unittest.TestCase):
    def test_new_and_reused_slots(self):
        class State:
            def __init__(self):
                self.draft_tokens = torch.zeros((2, 5), dtype=torch.int64)
                self.req_id_to_index = {}
            def add_request(self, req_id, slot):
                self.req_id_to_index[req_id] = slot
                self.draft_tokens[slot].zero_()
                return 'preserved'
        module.patch_request_state_draft_tokens(State)
        wrapped = State.add_request
        module.patch_request_state_draft_tokens(State)
        self.assertIs(State.add_request, wrapped)
        state = State()
        self.assertEqual(state.add_request('first', 0), 'preserved')
        self.assertTrue(torch.all(state.draft_tokens[0] == -1))
        # A real proposal, including a genuine EOS, must stay untouched until reuse.
        state.draft_tokens[0] = torch.tensor([0, 42, 43, 44, 45])
        state.add_request('other', 1)
        self.assertEqual(state.draft_tokens[0, 0].item(), 0)
        state.add_request('reused', 0)
        self.assertTrue(torch.all(state.draft_tokens[0] == -1))

if __name__ == '__main__':
    unittest.main()
