# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
"""Enable load reporting through Ascend platform registration, without flags."""

from functools import wraps


def install_default_load_hooks():
    # Keep platform discovery lightweight: EngineCore imports can resolve the
    # platform themselves. Defer them until the normal engine initialization.
    import vllm.plugins as plugins

    if not getattr(plugins.load_general_plugins, "_ascend_router_load", False):
        original_general = plugins.load_general_plugins

        @wraps(original_general)
        def load_general_plugins():
            original_general()
            from vllm_ascend.observability.router_load import install_engine_method

            install_engine_method()

        load_general_plugins._ascend_router_load = True
        plugins.load_general_plugins = load_general_plugins

    if not getattr(plugins.load_endpoint_plugins, "_ascend_router_load", False):
        original_endpoints = plugins.load_endpoint_plugins

        @wraps(original_endpoints)
        def load_endpoint_plugins(supported_tasks=None):
            endpoints = original_endpoints(supported_tasks)
            if supported_tasks is not None and "generate" in supported_tasks:
                from vllm_ascend.observability.router_load_api import RouterLoadEndpoint

                if not any(endpoint.name == RouterLoadEndpoint.name for endpoint in endpoints):
                    endpoints.append(RouterLoadEndpoint())
            return endpoints

        load_endpoint_plugins._ascend_router_load = True
        plugins.load_endpoint_plugins = load_endpoint_plugins
