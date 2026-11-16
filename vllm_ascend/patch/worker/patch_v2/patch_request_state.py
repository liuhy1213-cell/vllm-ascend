# SPDX-License-Identifier: Apache-2.0
"""Keep unproposed V2 draft slots invalid, including reused request slots."""
from functools import wraps


def patch_request_state_draft_tokens(request_state_cls=None):
    if request_state_cls is None:
        from vllm.v1.worker.gpu.states import RequestState
        request_state_cls = RequestState
    original = request_state_cls.add_request
    if getattr(original, "_ascend_invalid_draft_slots", False):
        return

    @wraps(original)
    def add_request(self, req_id, *args, **kwargs):
        result = original(self, req_id, *args, **kwargs)
        # PD/async scheduling can verify draft slots before the first proposal.
        # Zero is a valid token (and IQuest-Q1 EOS); -1 is the rejection
        # sampler's existing invalid-draft sentinel. Reset on slot reuse too.
        self.draft_tokens[self.req_id_to_index[req_id]].fill_(-1)
        return result

    add_request._ascend_invalid_draft_slots = True
    request_state_cls.add_request = add_request
