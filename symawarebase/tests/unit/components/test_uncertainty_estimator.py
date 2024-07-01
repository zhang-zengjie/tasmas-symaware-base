# pylint: disable=missing-function-docstring, missing-class-docstring, redefined-outer-name, no-self-use, unused-argument, abstract-class-instantiated, invalid-name, pointless-statement, protected-access
from unittest.mock import Mock

import numpy as np
import pytest

from symaware.base import (
    Agent,
    AsyncLoopLock,
    DefaultUncertaintyEstimator,
    MultiAgentAwarenessVector,
    MultiAgentKnowledgeDatabase,
    TimeSeries,
    UncertaintyEstimator,
)


class TestUncertaintyEstimator:
    def test_uncertainty_estimator_init_sync(self, PatchedUncertaintyEstimator: type[UncertaintyEstimator]):
        ID = 1
        uncertainty_estimator = PatchedUncertaintyEstimator(ID)
        assert uncertainty_estimator is not None
        assert uncertainty_estimator.is_initialised is False
        assert uncertainty_estimator.agent_id == ID
        assert uncertainty_estimator.can_loop is False
        with pytest.raises(AttributeError):
            uncertainty_estimator.async_loop_lock

    def test_uncertainty_estimator_init_async(
        self, PatchedUncertaintyEstimator: type[UncertaintyEstimator], PatchedAsyncLoopLock: type[AsyncLoopLock]
    ):
        ID = 1
        uncertainty_estimator = PatchedUncertaintyEstimator(ID, async_loop_lock=PatchedAsyncLoopLock())
        assert uncertainty_estimator is not None
        assert uncertainty_estimator.is_initialised is False
        assert uncertainty_estimator.agent_id == ID
        assert uncertainty_estimator.can_loop is True
        assert uncertainty_estimator.async_loop_lock is not None

    def test_compute_uncertainty_uninitialised(self, PatchedUncertaintyEstimator: type[UncertaintyEstimator]):
        uncertainty_estimator = PatchedUncertaintyEstimator(1)

        with pytest.raises(RuntimeError):
            uncertainty_estimator.compute()

    def test_uncertainty_estimator_compute_uncertainty(
        self,
        PatchedUncertaintyEstimator: type[UncertaintyEstimator],
        agent: Agent,
        awareness_vector: MultiAgentAwarenessVector,
        knowledge_database: MultiAgentKnowledgeDatabase,
    ):
        uncertainty_estimator = PatchedUncertaintyEstimator(agent.id)

        callback1 = Mock()
        callback2 = Mock()
        uncertainty_estimator.add_on_computing(callback1)
        uncertainty_estimator.add_on_computed(callback2)
        uncertainty_estimator._compute_uncertainty = lambda awareness_database, knowledge_database: TimeSeries(
            {0: np.array([0.5])}
        )

        uncertainty_estimator.initialise_component(agent, awareness_vector, knowledge_database)
        uncertainty = uncertainty_estimator.compute()

        # Assertions
        assert uncertainty is not None
        callback1.assert_called_once_with(agent)
        callback2.assert_called_once_with(agent, uncertainty)

    @pytest.mark.asyncio
    async def test_uncertainty_estimator_async_compute_uncertainty(
        self,
        PatchedUncertaintyEstimator: type[UncertaintyEstimator],
        agent: Agent,
        awareness_vector: MultiAgentAwarenessVector,
        knowledge_database: MultiAgentKnowledgeDatabase,
    ):
        uncertainty_estimator = PatchedUncertaintyEstimator(agent.id)

        callback1 = Mock()
        callback2 = Mock()
        uncertainty_estimator.add_on_computing(callback1)
        uncertainty_estimator.add_on_computed(callback2)
        uncertainty_estimator._compute_uncertainty = lambda awareness_database, knowledge_database: TimeSeries(
            {0: np.array([0.5])}
        )

        await uncertainty_estimator.async_initialise_component(agent, awareness_vector, knowledge_database)
        uncertainty = await uncertainty_estimator.async_compute()

        assert uncertainty is not None
        callback1.assert_called_once_with(agent)
        callback2.assert_called_once_with(agent, uncertainty)

    def test_uncertainty_estimator_add_on_computing_uncertainty(
        self, PatchedUncertaintyEstimator: type[UncertaintyEstimator]
    ):
        callback = Mock()
        uncertainty_estimator = PatchedUncertaintyEstimator(1)
        uncertainty_estimator.add_on_computing(callback)
        assert uncertainty_estimator._callbacks["computing"] == {callback}
        uncertainty_estimator.remove_on_computing(callback)
        assert uncertainty_estimator._callbacks["computing"] == set()

    def test_uncertainty_estimator_add_on_computed_uncertainty(
        self, PatchedUncertaintyEstimator: type[UncertaintyEstimator]
    ):
        callback = Mock()
        uncertainty_estimator = PatchedUncertaintyEstimator(1)
        uncertainty_estimator.add_on_computed(callback)
        assert uncertainty_estimator._callbacks["computed"] == {callback}
        uncertainty_estimator.remove_on_computed(callback)
        assert uncertainty_estimator._callbacks["computed"] == set()


class TestDefaultUncertaintyEstimator:
    def test_default_uncertainty_estimator_compute_uncertainty(
        self,
        agent: Agent,
        awareness_vector: MultiAgentAwarenessVector,
        knowledge_database: MultiAgentKnowledgeDatabase,
    ):
        uncertainty_estimator = DefaultUncertaintyEstimator(agent.id)

        agent._awareness_database = awareness_vector
        uncertainty_estimator.initialise_component(agent, awareness_vector, knowledge_database)
        uncertainty = uncertainty_estimator.compute()

        assert uncertainty is not None
        assert len(uncertainty) == 0

    @pytest.mark.asyncio
    async def test_default_uncertainty_estimator_async_compute_uncertainty(
        self,
        agent: Agent,
        awareness_vector: MultiAgentAwarenessVector,
        knowledge_database: MultiAgentKnowledgeDatabase,
    ):
        uncertainty_estimator = DefaultUncertaintyEstimator(agent.id)

        agent._awareness_database = awareness_vector
        await uncertainty_estimator.async_initialise_component(agent, awareness_vector, knowledge_database)
        uncertainty = await uncertainty_estimator.async_compute()

        assert uncertainty is not None
        assert len(uncertainty) == 0

    def test_default_uncertainty_estimator_compute_uncertainty_from_old(
        self,
        agent: Agent,
        awareness_vector: MultiAgentAwarenessVector,
        knowledge_database: MultiAgentKnowledgeDatabase,
    ):
        uncertainty_estimator = DefaultUncertaintyEstimator(agent.id)

        awareness_vector[agent.id].uncertainty = TimeSeries({0: np.array([0.5])})

        agent._awareness_database = awareness_vector
        uncertainty_estimator.initialise_component(agent, awareness_vector, knowledge_database)
        uncertainty = uncertainty_estimator.compute()

        assert uncertainty is not None
        assert len(uncertainty) == 1
        assert uncertainty[0] == np.array([0.5])

    @pytest.mark.asyncio
    async def test_default_uncertainty_estimator_async_compute_uncertainty_from_old(
        self,
        agent: Agent,
        awareness_vector: MultiAgentAwarenessVector,
        knowledge_database: MultiAgentKnowledgeDatabase,
    ):
        uncertainty_estimator = DefaultUncertaintyEstimator(agent.id)

        awareness_vector[agent.id].uncertainty = TimeSeries({0: np.array([0.5])})

        agent._awareness_database = awareness_vector
        await uncertainty_estimator.async_initialise_component(agent, awareness_vector, knowledge_database)
        uncertainty = await uncertainty_estimator.async_compute()

        assert uncertainty is not None
        assert len(uncertainty) == 1
        assert uncertainty[0] == np.array([0.5])
