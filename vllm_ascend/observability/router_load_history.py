# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
"""Observe native admission statistics, never issue extra cache lookups.

State belongs to one scheduler/DP rank. One-second buckets bound memory to the
last five minutes. Samples are connector lookup hits at admission, not proof
that a subsequent external transfer completed successfully.
"""

import time
import uuid
from collections import deque
from functools import wraps

WINDOW_SECONDS = 300


class HitWindow:
    def __init__(self):
        self.buckets = deque()
        self.requests = self.queries = self.hits = 0
        self.invalid_batches = 0

    def prune(self, now):
        # Drop a complete bucket at its 300-second boundary. This may discard
        # less than one extra second, but never retains samples over 5 minutes.
        cutoff = now - WINDOW_SECONDS
        while self.buckets and self.buckets[0][0] <= cutoff:
            _, requests, queries, hits = self.buckets.popleft()
            self.requests -= requests
            self.queries -= queries
            self.hits -= hits

    def add(self, requests, queries, hits, now):
        self.prune(now)
        if requests == queries == hits == 0:
            return
        if requests <= 0 or queries <= 0 or not 0 <= hits <= queries:
            # Observability must not interrupt model execution on a malformed
            # statistics batch. Expose the loss rather than using a bogus rate.
            self.invalid_batches += 1
            return
        second = int(now)
        if self.buckets and self.buckets[-1][0] == second:
            bucket = self.buckets[-1]
            bucket[1] += requests
            bucket[2] += queries
            bucket[3] += hits
        else:
            self.buckets.append([second, requests, queries, hits])
        self.requests += requests
        self.queries += queries
        self.hits += hits

    def snapshot(self, now):
        self.prune(now)
        return {
            "requests": self.requests,
            "query_tokens": self.queries,
            "hit_tokens": self.hits,
            "hit_rate": self.hits / self.queries if self.queries else None,
        }


class LoadHistory:
    def __init__(self):
        self.instance_id = uuid.uuid4().hex
        self.reset_count = 0
        self.normal = HitWindow()
        self.preempted = HitWindow()
        self.skip_stats = None
        self.skip_counts = None

    @staticmethod
    def counts(stats):
        return tuple(
            int(getattr(stats, name, 0))
            for name in ("requests", "queries", "hits", "preempted_requests", "preempted_queries", "preempted_hits")
        )

    def clear(self, pending_stats=None):
        self.normal = HitWindow()
        self.preempted = HitWindow()
        self.reset_count += 1
        # A prefix flush need not drain the native statistics object. Do not
        # re-import samples from before that flush on the next make_stats().
        self.skip_stats = pending_stats
        self.skip_counts = self.counts(pending_stats) if pending_stats is not None else None

    def observe(self, stats, now):
        if stats is None:
            return
        counts = self.counts(stats)
        if stats is self.skip_stats:
            counts = tuple(max(0, a - b) for a, b in zip(counts, self.skip_counts))
        elif getattr(stats, "reset", False):
            self.clear()
        self.skip_stats = self.skip_counts = None
        self.normal.add(*counts[:3], now)
        self.preempted.add(*counts[3:], now)

    def snapshot(self, now):
        return {
            "window_seconds": WINDOW_SECONDS,
            "bucket_seconds": 1,
            "reset_count": self.reset_count,
            "source": "native_connector_admission_lookup",
            "invalid_batches": self.normal.invalid_batches + self.preempted.invalid_batches,
            "normal": self.normal.snapshot(now),
            "preempted": self.preempted.snapshot(now),
        }


def history_for(scheduler):
    history = getattr(scheduler, "_router_load_history", None)
    if history is None:
        history = LoadHistory()
        scheduler._router_load_history = history
    return history


def install_history_hooks():
    # Lazy: importing a scheduler during platform discovery can recurse.
    from vllm.v1.core.sched.scheduler import Scheduler

    if not getattr(Scheduler.make_stats, "_ascend_router_load_history", False):
        original_stats = Scheduler.make_stats

        @wraps(original_stats)
        def make_stats(self, *args, **kwargs):
            stats = original_stats(self, *args, **kwargs)
            if stats is not None:
                history_for(self).observe(stats.connector_prefix_cache_stats, time.monotonic())
            return stats

        make_stats._ascend_router_load_history = True
        Scheduler.make_stats = make_stats

    if not getattr(Scheduler.reset_prefix_cache, "_ascend_router_load_history", False):
        original_reset = Scheduler.reset_prefix_cache

        @wraps(original_reset)
        def reset_prefix_cache(self, *args, **kwargs):
            success = original_reset(self, *args, **kwargs)
            if success:
                history_for(self).clear(getattr(self, "connector_prefix_cache_stats", None))
            return success

        reset_prefix_cache._ascend_router_load_history = True
        Scheduler.reset_prefix_cache = reset_prefix_cache
