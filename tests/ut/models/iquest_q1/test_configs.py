# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Config contract tests for IQuest-Q1's MoE expert width."""

from vllm_ascend.models.iquest_q1.configs import IQuestQ1Config, IQuestQ1MTPConfig


def test_moe_intermediate_size_defaults_to_intermediate_size():
    config = IQuestQ1Config()
    assert config.moe_intermediate_size == config.intermediate_size == 1536


def test_moe_intermediate_size_override_is_preserved():
    config = IQuestQ1Config(intermediate_size=1536, moe_intermediate_size=2048)
    assert config.intermediate_size == 1536
    assert config.moe_intermediate_size == 2048


def test_mtp_draft_inherits_moe_intermediate_size():
    draft = IQuestQ1MTPConfig(target_config=IQuestQ1Config().to_dict())
    assert draft.moe_intermediate_size == draft.intermediate_size == 1536
