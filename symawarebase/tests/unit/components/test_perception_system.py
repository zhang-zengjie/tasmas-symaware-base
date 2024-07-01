# pylint: disable=missing-function-docstring, missing-class-docstring, redefined-outer-name, no-self-use, unused-argument, abstract-class-instantiated, invalid-name, pointless-statement, protected-access
from unittest.mock import Mock

import pytest

from symaware.base import (
    Agent,
    AsyncLoopLock,
    Environment,
    MultiAgentAwarenessVector,
    MultiAgentKnowledgeDatabase,
    PerceptionSystem,
)


class TestPerceptionSystem:
    def test_perception_system_init_sync(self, PatchedPerceptionSystem: type[PerceptionSystem]):
        ID = 1
        environment = Mock(spec=Environment)
        perception_system = PatchedPerceptionSystem(ID, environment)
        assert perception_system is not None
        assert perception_system.is_initialised is False
        assert perception_system.agent_id == ID
        assert perception_system.can_loop is False
        with pytest.raises(AttributeError):
            perception_system.async_loop_lock
        assert perception_system.environment == environment

    def test_perception_system_init_async(
        self, PatchedPerceptionSystem: type[PerceptionSystem], PatchedAsyncLoopLock: type[AsyncLoopLock]
    ):
        ID = 1
        environment = Mock(spec=Environment)
        perception_system = PatchedPerceptionSystem(ID, environment, async_loop_lock=PatchedAsyncLoopLock())
        assert perception_system is not None
        assert perception_system.is_initialised is False
        assert perception_system.agent_id == ID
        assert perception_system.can_loop is True
        assert perception_system.async_loop_lock is not None
        assert perception_system.environment == environment

    def test_perception_system_perceive_self_state_uninitialised(self, PatchedPerceptionSystem: type[PerceptionSystem]):
        perception_system = PatchedPerceptionSystem(1, Mock(spec=Environment))
        with pytest.raises(RuntimeError):
            perception_system.perceive_self_state()

    def test_perception_system_perceive_information_uninitialised(
        self, PatchedPerceptionSystem: type[PerceptionSystem]
    ):
        perception_system = PatchedPerceptionSystem(1, Mock(spec=Environment))
        with pytest.raises(RuntimeError):
            perception_system.compute()

    def test_perception_system_perceive_self_state_missing(
        self,
        PatchedPerceptionSystem: type[PerceptionSystem],
        agent: Agent,
        environment: Environment,
        awareness_vector: MultiAgentAwarenessVector,
        knowledge_database: MultiAgentKnowledgeDatabase,
    ):
        perception_system = PatchedPerceptionSystem(agent.id, environment)
        perception_system.initialise_component(agent, awareness_vector, knowledge_database)

        with pytest.raises(KeyError):
            perception_system.perceive_self_state()
        with pytest.raises(KeyError):
            environment.get_agent_state(agent.id)

    def test_perception_system_perceive_information(
        self,
        PatchedPerceptionSystem: type[PerceptionSystem],
        agent: Agent,
        environment: Environment,
        awareness_vector: MultiAgentAwarenessVector,
        knowledge_database: MultiAgentKnowledgeDatabase,
    ):
        perception_system = PatchedPerceptionSystem(agent.id, environment)

        callback1 = Mock()
        callback2 = Mock()
        perception_system.add_on_computing(callback1)
        perception_system.add_on_computed(callback2)

        perception_system.initialise_component(agent, awareness_vector, knowledge_database)
        result = perception_system.compute()

        assert result is not None
        callback1.assert_called_once_with(agent)
        callback2.assert_called_once_with(agent, result)
        # Add assertions for the expected behavior of compute()

    @pytest.mark.asyncio
    async def test_perception_system_async_perceive_information(
        self,
        PatchedPerceptionSystem: type[PerceptionSystem],
        agent: Agent,
        environment: Environment,
        awareness_vector: MultiAgentAwarenessVector,
        knowledge_database: MultiAgentKnowledgeDatabase,
    ):
        perception_system = PatchedPerceptionSystem(agent.id, environment)

        callback1 = Mock()
        callback2 = Mock()
        perception_system.add_on_computing(callback1)
        perception_system.add_on_computed(callback2)

        await perception_system.async_initialise_component(agent, awareness_vector, knowledge_database)
        result = await perception_system.async_compute()

        assert result is not None
        callback1.assert_called_once_with(agent)
        callback2.assert_called_once_with(agent, result)

    def test_perception_system_add_on_computing(self, PatchedPerceptionSystem: type[PerceptionSystem]):
        callback = Mock()
        perception_system = PatchedPerceptionSystem(1, Mock(spec=Environment))
        perception_system.add_on_computing(callback)
        assert perception_system._callbacks["computing"] == {callback}
        perception_system.remove_on_computing(callback)
        assert perception_system._callbacks["computing"] == set()

    def test_perception_system_add_on_computed(self, PatchedPerceptionSystem: type[PerceptionSystem]):
        callback = Mock()
        perception_system = PatchedPerceptionSystem(1, Mock(spec=Environment))
        perception_system.add_on_computed(callback)
        assert perception_system._callbacks["computed"] == {callback}
        perception_system.remove_on_computed(callback)
        assert perception_system._callbacks["computed"] == set()
