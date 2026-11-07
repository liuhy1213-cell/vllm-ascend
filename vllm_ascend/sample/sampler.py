import torch
import vllm.envs as envs
from vllm.logger import logger
from vllm.triton_utils import HAS_TRITON
from vllm.v1.sample.metadata import SamplingMetadata
from vllm.v1.sample.ops.topk_topp_sampler import TopKTopPSampler
from vllm.v1.sample.sampler import Sampler

from vllm_ascend.device.hardware_profile import HardwareCapability, get_current_hardware_profile
from vllm_ascend.sample.penalties import apply_all_penalties
from vllm_ascend.utils import global_stream, npu_stream_switch

DEFAULT_LOGPROBS_MODE = "raw_logprobs"

_SAMPLING_EPS = 1e-5


def random_sample(
    probs: torch.Tensor,
    generators: dict[int, torch.Generator],
) -> torch.Tensor:
    """Randomly sample from the probabilities.

    We use this function instead of torch.multinomial because torch.multinomial
    causes CPU-NPU synchronization.
    """
    # NOTE(woosuk): To batch-process the requests without their own seeds,
    # which is the common case, we first assume that every request does
    # not have its own seed. Then, we overwrite the values for the requests
    # that have their own seeds.
    with npu_stream_switch(global_stream()):
        q = torch.empty_like(probs)
        if len(generators) != probs.shape[0]:
            q.exponential_()
        if generators:
            # TODO(woosuk): This can be slow because we handle each request
            # one by one. Optimize this.
            for i, generator in generators.items():
                q[i].exponential_(generator=generator)
    torch.npu.current_stream().wait_stream(global_stream())
    q.record_stream(torch.npu.current_stream())
    return probs.div_(q).argmax(dim=-1).view(-1)


class AscendSampler(Sampler):
    @staticmethod
    def apply_penalties(
        logits: torch.Tensor,
        sampling_metadata: SamplingMetadata,
        output_token_ids: list[list[int]],
    ) -> torch.Tensor:
        """Use Triton-Ascend penalties on NPU when Triton is available; else vLLM default."""
        if not HAS_TRITON:
            logger.warning_once(
                "[sample/sampler] Triton not available, falling back to vLLM default "
                "penalty implementation. Penalty performance may be degraded on NPU. "
            )
            return Sampler.apply_penalties(logits, sampling_metadata, output_token_ids)

        if sampling_metadata.no_penalties:
            return logits
        assert sampling_metadata.prompt_token_ids is not None
        return apply_all_penalties(
            logits,
            sampling_metadata.prompt_token_ids,
            sampling_metadata.presence_penalties,
            sampling_metadata.frequency_penalties,
            sampling_metadata.repetition_penalties,
            output_token_ids,
        )

    def __init__(self, logprobs_mode=DEFAULT_LOGPROBS_MODE):
        # TODO: support logprobs_mode in vllm-ascend
        super().__init__(logprobs_mode=logprobs_mode)
        self.topk_topp_sampler = AscendTopKTopPSampler(logprobs_mode=logprobs_mode)
        logger.debug(
            "[sample/sampler] AscendSampler initialized. logprobs_mode=%s, triton_available=%s",
            logprobs_mode,
            HAS_TRITON,
        )

    @staticmethod
    def greedy_sample(logits: torch.Tensor) -> torch.Tensor:
        return logits.argmax(dim=-1).view(-1)


class AscendTopKTopPSampler(TopKTopPSampler):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.apply_top_k_top_p = apply_top_k_top_p

    def forward_native(self, logits, generators, k, p):
        """Override pytorch native implementation to torch_npu"""
        # when batch_invariant mode is enabled, we should use vllm's implementation.
        # or it will make batch_invariant mode not working.
        if envs.VLLM_BATCH_INVARIANT:
            logger.debug_once(
                "[sample/sampler] BATCH_INVARIANT mode enabled, "
                "falling back to vLLM native top-k/top-p implementation.",
            )
            return super().forward_native(logits, generators, k, p)

        logits = self.apply_top_k_top_p(logits, k, p)
        logits_to_return = None
        if self.logprobs_mode == "processed_logits":
            logits_to_return = logits
        elif self.logprobs_mode == "processed_logprobs":
            logits_to_return = logits.log_softmax(dim=-1, dtype=torch.float32)

        probs = logits.softmax(dim=-1, dtype=torch.float32)
        return random_sample(probs, generators), logits_to_return


# On wide-vocab rows the full [T, V] ascending sort + cumsum is pathological on
# NPU: aclnnCumsum scans each 160k-wide row serially (~31 GB/s on [120, 160000]
# fp32, ~5 ms per decode step). Only the first few sorted entries can ever
# matter: top-k keeps k <= cap tokens, and the top-p cutoff either falls inside
# the top-cap prefix (computable from it) or beyond it, in which case the
# combined (top-k, top-p) cutoff equals the top-k cutoff. See
# _apply_top_k_top_p_topk_mask for the equivalence argument.
_TOPK_TOPP_FAST_CAP = 128
_TOPK_TOPP_FAST_MIN_VOCAB = 8192


def _apply_top_k_top_p_topk_mask(
    logits: torch.Tensor,  # [B, V]
    probs: torch.Tensor,  # [B, V], softmax of logits
    k: torch.Tensor | None,  # [B] or None
    p: torch.Tensor | None,  # [B] or None
) -> torch.Tensor:
    """Sort-free top-k/top-p masking via a narrow topk prefix.

    Equivalent to the sort-based implementation below because
    ``masked_fill_(probs < c_k)`` then ``masked_fill_(probs < c_p)`` is the
    same as a single fill with ``max(c_k, c_p)``:

    - top-k rows (k <= cap): cutoff_k = vals[k - 1] is exact.
    - top-p: with vals the descending top-cap probs and c the descending
      cumsum, the top-p cutoff is vals[m - 1] where m is the first index with
      c[m - 1] > p. If no such index exists within the cap, the true top-p
      cutoff lies beyond position cap, hence below vals[cap - 1]; combined
      with any valid cutoff_k (>= vals[k - 1] >= vals[cap - 1]) the max is
      cutoff_k, so skipping top-p for that row is exact. Rows without a valid
      k degrade leniently (no top-k truncation), never over-truncate.
    """
    vocab = probs.shape[-1]
    cap = min(_TOPK_TOPP_FAST_CAP, vocab)
    vals, _ = torch.topk(probs, cap, dim=-1)  # [B, cap], descending

    neg_inf = float("-inf")
    cutoff = None
    if k is not None:
        valid_k = (k >= 1) & (k <= cap)
        gather_idx = k.clamp(1, cap).unsqueeze(-1)  # [B, 1], keep gather in range
        cutoff_k = vals.gather(-1, gather_idx - 1).squeeze(-1)  # [B]
        cutoff_k = torch.where(valid_k, cutoff_k, neg_inf)
        cutoff = cutoff_k
    if p is not None:
        hit = vals.cumsum(-1) > p.unsqueeze(-1)  # [B, cap]
        has_hit = hit.any(-1)
        first_hit = hit.to(torch.long).argmax(-1, keepdim=True)  # 0 if no hit
        cutoff_p = vals.gather(-1, first_hit).squeeze(-1)  # [B]
        cutoff_p = torch.where(has_hit, cutoff_p, neg_inf)
        cutoff = cutoff_p if cutoff is None else torch.maximum(cutoff, cutoff_p)

    assert cutoff is not None
    logits.masked_fill_(probs < cutoff.unsqueeze(-1), -float("inf"))
    return logits


def _apply_top_k_top_p_pytorch(
    logits: torch.Tensor,  # [B, V_local]
    k: torch.Tensor,  # [B] or None
    p: torch.Tensor,  # [B] or None
) -> torch.Tensor:
    if p is None and k is None:
        return logits

    probs = logits.softmax(dim=-1)

    # Wide-vocab rows: replace the [B, V] sort + cumsum (pathologically slow
    # aclnnCumsum on wide rows) with a narrow topk-based mask. Small vocabs
    # keep the original sort-based path, where sort/cumsum are cheap.
    if probs.shape[-1] >= _TOPK_TOPP_FAST_MIN_VOCAB:
        return _apply_top_k_top_p_topk_mask(logits, probs, k, p)

    probs_sort, _ = probs.sort(dim=-1, descending=False)

    if k is not None:
        top_k_count = probs_sort.size(1) - k.to(torch.long)  # shape: (batch, )
        top_k_count = top_k_count.unsqueeze(dim=1)
        top_k_cutoff = probs_sort.gather(-1, top_k_count)

        # Make sure the no top-k rows are no-op.
        no_top_k_mask = (k == logits.shape[1]).unsqueeze(dim=1)
        top_k_cutoff.masked_fill_(no_top_k_mask, -float("inf"))

        elements_to_discard = probs < top_k_cutoff
        logits.masked_fill_(elements_to_discard, -float("inf"))

    if p is not None:
        cumprob = torch.cumsum(probs_sort, dim=-1)
        top_p_mask = cumprob <= 1 - p.unsqueeze(dim=1)
        top_p_mask[:, -1] = False  # at least one

        top_p_count = top_p_mask.sum(dim=-1).unsqueeze(1)
        top_p_cutoff = probs_sort.gather(-1, top_p_count)
        elements_to_discard = probs < top_p_cutoff
        logits.masked_fill_(elements_to_discard, -float("inf"))

    return logits


def _apply_top_k_top_p_torch_npu(
    logits: torch.Tensor,
    k: torch.Tensor,
    p: torch.Tensor,
) -> torch.Tensor:
    # Use sort-based pytorch implementation.
    # npu_top_k_top_p degrades severely (5-28ms) when k is large or batch
    # contains mixed k values, while sort+mask is consistently ~1ms.
    if p is None and k is None:
        return logits
    return _apply_top_k_top_p_pytorch(logits, k, p)


apply_top_k_top_p = (
    _apply_top_k_top_p_torch_npu
    if get_current_hardware_profile().supports(HardwareCapability.NPU_TOP_K_TOP_P)
    else _apply_top_k_top_p_pytorch
)
