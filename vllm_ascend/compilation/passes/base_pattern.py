import weakref
from abc import ABC, abstractmethod
from collections.abc import Callable

import torch
import torch._inductor.pattern_matcher as pm
from torch._inductor.pattern_matcher import PatternMatcherPass
from vllm.config import VllmConfig

try:
    import npugraph_ex as nge
except ImportError:
    import torchair as nge

from vllm_ascend.compilation.passes.utils.npugraph_ex_utils_check import extra_stream_scope_check

# Patterns already handed to the process-global npugraph_ex backend registry.
_registered_patterns: set[str] = set()
# Pattern keys already registered into a given matcher. A module-level set cannot
# be used here because every compiled graph builds its own PatternMatcherPass; a
# weak-keyed map keeps the bookkeeping off the third-party matcher object and
# does not extend its lifetime.
_registered_by_matcher: "weakref.WeakKeyDictionary[PatternMatcherPass, set[str]]" = weakref.WeakKeyDictionary()


class BasePattern(ABC):
    def __init__(self, vllm_config: VllmConfig, eps: float = 1e-6):
        self.vllm_config = vllm_config
        self.dtype = vllm_config.model_config.dtype
        self.eps = eps

    @abstractmethod
    def get_inputs(self) -> list[torch.Tensor]:
        pass

    @abstractmethod
    def get_pattern(self) -> Callable:
        pass

    @abstractmethod
    def get_replacement(self) -> Callable:
        pass

    def get_extra_stream_scope_check(self):
        return extra_stream_scope_check

    def get_extra_check(self):
        """Optional shape/dtype guard evaluated on every real match.

        The inductor matcher only compares operator targets and argument
        structure, so patterns whose replacement has stricter input contracts
        than the matched ops (for example same-shape additions) can override
        this to reject incompatible matches.
        """
        return None

    def get_search_fn_pattern(self):
        """Optional pre-built pattern used instead of tracing ``get_pattern``.

        Dynamo keeps custom-op calls as ``torch.ops.<ns>.<op>`` packets while
        ``make_fx`` records the resolved ``<op>.default`` overload, so a
        normally traced pattern can only match one of the two graph forms.
        A pattern built from a packet target keeps the packet and all of its
        overloads in the target set and matches either form.
        """
        return None

    def pattern_key(self) -> str:
        return f"{self.__class__.__name__}_{self.eps}"

    def register(self, pm_pass: PatternMatcherPass) -> None:
        # NOTE: every GraphFusionPassManager instance owns its own PatternMatcherPass,
        # and a new instance is built for each compiled graph (prefill range, decode
        # range, speculator, ...). Registering into ``pm_pass`` must therefore happen
        # unconditionally: an earlier global de-duplication skipped it for every graph
        # after the first one, which left their matchers empty and silently disabled
        # every FX fusion pass. Only the npugraph_ex backend registry is global, so
        # only that call is de-duplicated.
        pattern_id = self.pattern_key()

        registered = _registered_by_matcher.get(pm_pass)
        if registered is None:
            registered = set()
            _registered_by_matcher[pm_pass] = registered

        pattern_fn = self.get_pattern()
        replacement_fn = self.get_replacement()
        example_inputs = self.get_inputs()

        # The two registries are independent, so they are guarded and recorded
        # separately: a pattern that fails to register must stay retryable, and a
        # failure in one registry must not suppress the other. Record the key only
        # after the registration actually succeeded.
        search_fn_pattern = self.get_search_fn_pattern()
        extra_check = self.get_extra_check()

        if pattern_id not in registered:
            register_kwargs = {}
            if extra_check is not None:
                register_kwargs["extra_check"] = extra_check
            if search_fn_pattern is not None:
                register_kwargs["search_fn_pattern"] = search_fn_pattern
            pm.register_replacement(
                pattern_fn, replacement_fn, example_inputs, pm.fwd_only, pm_pass, **register_kwargs
            )
            registered.add(pattern_id)

        if pattern_id not in _registered_patterns:
            stream_scope_check = self.get_extra_stream_scope_check()
            own_extra_check = self.get_extra_check()

            def nge_extra_check(match) -> bool:
                # The npugraph_ex registry applies its own stream/scope checks
                # around this callback; keep the legacy stream check for older
                # backends and add the pattern's own shape/dtype guard.
                if not stream_scope_check(match):
                    return False
                return True if own_extra_check is None else own_extra_check(match)

            nge_kwargs = {"extra_check": nge_extra_check}
            if search_fn_pattern is not None:
                nge_kwargs["search_fn_pattern"] = search_fn_pattern
            nge.register_replacement(
                search_fn=pattern_fn,
                replace_fn=replacement_fn,
                example_inputs=example_inputs,
                **nge_kwargs,
            )
            # Mark this pattern as registered in the global backend registry
            _registered_patterns.add(pattern_id)
