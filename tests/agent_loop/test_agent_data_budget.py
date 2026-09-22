"""Prompt and response share one trainable context budget."""

from types import SimpleNamespace

import pytest
from omegaconf import OmegaConf
from psrl.workers.agent_loop.agent_data import ConversationAgentData
from psrl.workers.agent_loop.agent_data.base import SessionData, Trajectory
from psrl.workers.gen.utils import rollout_token_budget

pytestmark = pytest.mark.cpu_test

_PROMPT_LENGTH = 32768
_RESPONSE_LENGTH = 32768
_BUDGET = _PROMPT_LENGTH + _RESPONSE_LENGTH


def test_rollout_token_budget_reads_an_omegaconf_node() -> None:
    # The production config reaches the loops as an OmegaConf node, not as the
    # structured dataclass, so the helper must accept both.
    config = OmegaConf.create({"prompt_length": 2048, "response_length": 4096})

    assert rollout_token_budget(config) == 6144


def _agent_data(trajectories: list[Trajectory]) -> ConversationAgentData:
    data = ConversationAgentData.__new__(ConversationAgentData)
    data.session_data = SessionData(request_id=1, trajectories=trajectories)
    data.config = SimpleNamespace(
        gen_actor_rollout_ref=SimpleNamespace(
            rollout=SimpleNamespace(
                prompt_length=_PROMPT_LENGTH,
                response_length=_RESPONSE_LENGTH,
                agent=SimpleNamespace(traj_reward_mode="step"),
            )
        ),
        reward=SimpleNamespace(launch_reward_fn_async=False),
    )
    return data


def _trajectory(prompt_tokens: int, response_tokens: int) -> Trajectory:
    return Trajectory(
        prompt_ids=list(range(prompt_tokens)),
        response_ids=list(range(response_tokens)),
        response_mask=[1] * response_tokens,
        response_logprobs=[-0.1] * response_tokens,
    )


@pytest.mark.asyncio
async def test_finalize_output_keeps_a_response_longer_than_response_length() -> None:
    data = _agent_data([_trajectory(prompt_tokens=1000, response_tokens=40000)])

    output = await data.finalize_output()

    assert len(output.response_ids) == 40000
    assert len(output.response_mask) == 40000


@pytest.mark.asyncio
async def test_finalize_output_clamps_only_at_the_shared_budget() -> None:
    data = _agent_data([_trajectory(prompt_tokens=30000, response_tokens=40000)])

    output = await data.finalize_output()

    assert len(output.response_ids) == _BUDGET - 30000
    assert len(output.response_mask) == len(output.response_ids)
    assert len(output.response_log_probs) == len(output.response_ids)


@pytest.mark.asyncio
async def test_finalize_output_budgets_each_trajectory_against_its_own_prompt() -> None:
    data = _agent_data(
        [
            _trajectory(prompt_tokens=1000, response_tokens=40000),
            _trajectory(prompt_tokens=60000, response_tokens=40000),
        ]
    )

    outputs = await data.finalize_output()

    assert [len(output.response_ids) for output in outputs] == [40000, _BUDGET - 60000]
