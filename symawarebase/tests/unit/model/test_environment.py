# pylint: disable=missing-function-docstring, missing-class-docstring, redefined-outer-name, no-self-use, unused-argument, abstract-class-instantiated, invalid-name, attribute-defined-outside-init, protected-access, unused-variable, unused-import, redefined-outer-name, no-self-use, unused-argument, abstract-class-instantiated, invalid-name, attribute-defined-outside-init, protected-access, unused-variable, unused-import
import asyncio
from unittest.mock import Mock

import pytest

from symaware.base import Agent, Environment, TimeIntervalAsyncLoopLock


class TestEnvironment:

    def test_add_agents_single(self, environment: Environment, agent: Agent):
        assert len(environment.entities) == 0

        environment.add_agents(agent)

        assert len(environment.entities) == 1
        assert agent.entity in environment.entities

    def test_add_agents_multiple(self, environment: Environment, agent: Agent):
        assert len(environment.entities) == 0

        environment.add_agents((agent, agent))

        assert len(environment._agent_entities) == 1
        assert len(environment.entities) == 1
        assert agent.entity in environment.entities

    def test_get_agent_state(self, environment: Environment, agent: Agent):
        environment.add_agents(agent)

        state = environment.get_agent_state(agent)

        assert isinstance(state, Mock)

    def test_get_entity_state(self, environment: Environment, agent: Agent):
        environment.add_agents(agent)

        state = environment.get_entity_state(agent.entity)

        assert isinstance(state, Mock)

    def test_initialise(self, environment: Environment):
        environment.initialise()

    def test_step(self, environment: Environment):
        environment.step()

    def test_stop(self, environment: Environment):
        environment.stop()

    @pytest.mark.asyncio
    async def test_async_run_missing_lock(self, environment: Environment):
        with pytest.raises(AttributeError):
            await environment.async_run()

    @pytest.mark.asyncio
    async def test_async_stop_missing_lock(self, environment: Environment):
        with pytest.raises(AssertionError):
            await environment.async_stop()

    @pytest.mark.asyncio
    @pytest.mark.parametrize("environment", (TimeIntervalAsyncLoopLock(0.01),), indirect=True)
    async def test_async_run(self, environment: Environment):
        asyncio.get_running_loop().call_later(0.05, asyncio.create_task, environment.async_stop())
        await environment.async_run()
        assert environment.step.call_count == 0.05 / 0.01

    @pytest.mark.asyncio
    @pytest.mark.parametrize("environment", (TimeIntervalAsyncLoopLock(0.01),), indirect=True)
    async def test_async_stop(self, environment: Environment):
        await environment.async_stop()
        environment.step.assert_not_called()
