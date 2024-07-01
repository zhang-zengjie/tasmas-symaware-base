# pylint: disable=missing-function-docstring, missing-class-docstring, redefined-outer-name, no-self-use, unused-argument, abstract-class-instantiated, invalid-name, pointless-statement, protected-access
from unittest.mock import Mock

import pytest

from symaware.base import (
    Agent,
    AsyncLoopLock,
    Component,
    MultiAgentAwarenessVector,
    MultiAgentKnowledgeDatabase,
)


class TestComponent:
    def test_component_init_sync(self, PatchedComponent: type[Component]):
        ID = 1
        component = PatchedComponent(ID)
        assert component is not None
        assert component.is_initialised is False
        assert component.agent_id == ID
        assert component.can_loop is False
        with pytest.raises(AttributeError):
            component.async_loop_lock

    def test_component_init_async(self, PatchedComponent: type[Component], PatchedAsyncLoopLock: type[AsyncLoopLock]):
        ID = 1
        component = PatchedComponent(ID, async_loop_lock=PatchedAsyncLoopLock())
        assert component is not None
        assert component.is_initialised is False
        assert component.agent_id == ID
        assert component.can_loop is True
        assert component.async_loop_lock is not None

    def test_component_initialise(
        self,
        PatchedComponent: type[Component],
        agent: Agent,
        awareness_vector: MultiAgentAwarenessVector,
        knowledge_database: MultiAgentKnowledgeDatabase,
    ):
        component = PatchedComponent(agent.id)

        callback = Mock()
        component.add_on_initialised(callback)

        component.initialise_component(
            agent,
            awareness_vector,
            knowledge_database,
        )

        assert component.is_initialised is True
        assert component._agent is agent
        callback.assert_called_once_with(agent, awareness_vector, knowledge_database)

    @pytest.mark.asyncio
    async def test_component_initialise_async(
        self,
        PatchedComponent: type[Component],
        agent: Agent,
        awareness_vector: MultiAgentAwarenessVector,
        knowledge_database: MultiAgentKnowledgeDatabase,
    ):
        component = PatchedComponent(agent.id)

        callback = Mock()
        component.add_on_initialised(callback)

        await component.async_initialise_component(
            agent,
            awareness_vector,
            knowledge_database,
        )

        assert component.is_initialised is True
        assert component._agent is agent
        callback.assert_called_once_with(agent, awareness_vector, knowledge_database)

    def test_component_add_on_initialised(self, PatchedComponent: type[Component]):
        callback = Mock()
        component = PatchedComponent(1)
        component.add_on_initialised(callback)
        assert component._callbacks["initialised"] == {callback}
        component.remove_on_initialised(callback)
        assert component._callbacks["initialised"] == set()
