# pylint: disable=missing-function-docstring, missing-class-docstring, redefined-outer-name, no-self-use, unused-argument, abstract-class-instantiated, invalid-name, pointless-statement, protected-access
from unittest.mock import Mock

import numpy as np
import pytest

from symaware.base import (
    Agent,
    AsyncLoopLock,
    Controller,
    DefaultController,
    DynamicalModel,
    MultiAgentAwarenessVector,
    MultiAgentKnowledgeDatabase,
    TimeSeries,
)


class TestController:
    def test_controller_init_sync(self, PatchedController: type[Controller]):
        ID = 1
        controller = PatchedController(ID)
        assert controller is not None
        assert controller.is_initialised is False
        assert controller.agent_id == ID
        assert controller.can_loop is False
        with pytest.raises(AttributeError):
            controller.async_loop_lock

    def test_controller_init_async(
        self, PatchedController: type[Controller], PatchedAsyncLoopLock: type[AsyncLoopLock]
    ):
        ID = 1
        controller = PatchedController(ID, async_loop_lock=PatchedAsyncLoopLock())
        assert controller is not None
        assert controller.is_initialised is False
        assert controller.agent_id == ID
        assert controller.can_loop is True
        assert controller.async_loop_lock is not None

    def test_controller_initialise(
        self,
        PatchedController: type[Controller],
        agent: Agent,
        awareness_vector: MultiAgentAwarenessVector,
        knowledge_database: MultiAgentKnowledgeDatabase,
    ):
        callback = Mock()

        controller = PatchedController(agent.id)
        controller.add_on_initialised(callback)
        controller.initialise_component(
            agent,
            awareness_vector,
            knowledge_database,
        )

        assert controller.is_initialised is True
        assert controller._agent is agent
        assert controller.dynamical_model == agent.model
        callback.assert_called_once_with(agent, awareness_vector, knowledge_database)

    def test_controller_compute_control_input_uninitialised(self, PatchedController: type[Controller]):
        controller = PatchedController(1)
        with pytest.raises(RuntimeError):
            controller.compute()

    def test_controller_compute_control_input(
        self,
        PatchedController: type[Controller],
        agent: Agent,
        awareness_vector: MultiAgentAwarenessVector,
        knowledge_database: MultiAgentKnowledgeDatabase,
    ):
        controller = PatchedController(agent.id)

        callback1 = Mock()
        callback2 = Mock()
        controller.add_on_computing(callback1)
        controller.add_on_computed(callback2)

        controller.initialise_component(agent, awareness_vector, knowledge_database)
        control_input, time_series = controller.compute()

        assert control_input is not None
        assert time_series is not None
        callback1.assert_called_once_with(agent)
        callback2.assert_called_once_with(agent, (control_input, time_series))

    @pytest.mark.asyncio
    async def test_controller_async_compute_control_input(
        self,
        PatchedController: type[Controller],
        agent: Agent,
        awareness_vector: MultiAgentAwarenessVector,
        knowledge_database: MultiAgentKnowledgeDatabase,
    ):
        controller = PatchedController(agent.id)

        callback1 = Mock()
        callback2 = Mock()
        controller.add_on_computing(callback1)
        controller.add_on_computed(callback2)

        await controller.async_initialise_component(agent, awareness_vector, knowledge_database)
        control_input, time_series = await controller.async_compute()

        assert control_input is not None
        assert time_series is not None
        assert callback1.called
        callback1.assert_called_once_with(agent)
        callback2.assert_called_once_with(agent, (control_input, time_series))

    def test_controller_add_on_computing_control_input(self, PatchedController: type[Controller]):
        callback = Mock()
        controller = PatchedController(1)
        controller.add_on_computing(callback)
        assert controller._callbacks["computing"] == {callback}
        controller.remove_on_computing(callback)
        assert controller._callbacks["computing"] == set()

    def test_controller_add_on_computed_control_input(self, PatchedController: type[Controller]):
        callback = Mock()
        controller = PatchedController(1)
        controller.add_on_computed(callback)
        assert controller._callbacks["computed"] == {callback}
        controller.remove_on_computed(callback)
        assert controller._callbacks["computed"] == set()


class TestDefaultController:
    def test_default_controller_compute_control_input(
        self,
        agent: Agent,
        awareness_vector: MultiAgentAwarenessVector,
        knowledge_database: MultiAgentKnowledgeDatabase,
        PatchedDynamicalModel: type[DynamicalModel],
    ):
        controller = DefaultController(agent.id)
        control_input = np.zeros((1, 1))
        object.__setattr__(agent._entity, "model", PatchedDynamicalModel(agent.id, control_input))

        controller.initialise_component(agent, awareness_vector, knowledge_database)
        control_input, time_series = controller.compute()

        assert isinstance(control_input, np.ndarray)
        assert isinstance(time_series, TimeSeries)
        assert control_input is not None
        assert time_series is not None
        assert len(control_input) == agent.model.control_input_size
        assert control_input.shape == agent.model.control_input_shape
        assert np.all(control_input == 0)
        assert len(time_series) == 0
