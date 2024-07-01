# pylint: disable=missing-function-docstring, missing-class-docstring, redefined-outer-name, no-self-use, unused-argument, abstract-class-instantiated, invalid-name, pointless-statement, protected-access
from unittest.mock import Mock

import numpy as np
import pytest

from symaware.base import (
    Agent,
    AsyncLoopLock,
    DefaultRiskEstimator,
    MultiAgentAwarenessVector,
    MultiAgentKnowledgeDatabase,
    RiskEstimator,
    TimeSeries,
)


class TestRiskEstimator:
    def test_risk_estimator_init_sync(self, PatchedRiskEstimator: type[RiskEstimator]):
        ID = 1
        risk_estimator = PatchedRiskEstimator(ID)
        assert risk_estimator is not None
        assert risk_estimator.is_initialised is False
        assert risk_estimator.agent_id == ID
        assert risk_estimator.can_loop is False
        with pytest.raises(AttributeError):
            risk_estimator.async_loop_lock

    def test_risk_estimator_init_async(
        self, PatchedRiskEstimator: type[RiskEstimator], PatchedAsyncLoopLock: type[AsyncLoopLock]
    ):
        ID = 1
        risk_estimator = PatchedRiskEstimator(ID, async_loop_lock=PatchedAsyncLoopLock())
        assert risk_estimator is not None
        assert risk_estimator.is_initialised is False
        assert risk_estimator.agent_id == ID
        assert risk_estimator.can_loop is True
        assert risk_estimator.async_loop_lock is not None

    def test_compute_risk_uninitialised(self, PatchedRiskEstimator: type[RiskEstimator]):
        risk_estimator = PatchedRiskEstimator(1)

        with pytest.raises(RuntimeError):
            risk_estimator.compute()

    def test_risk_estimator_compute_risk(
        self,
        PatchedRiskEstimator: type[RiskEstimator],
        agent: Agent,
        awareness_vector: MultiAgentAwarenessVector,
        knowledge_database: MultiAgentKnowledgeDatabase,
    ):
        risk_estimator = PatchedRiskEstimator(agent.id)

        callback1 = Mock()
        callback2 = Mock()
        risk_estimator.add_on_computing(callback1)
        risk_estimator.add_on_computed(callback2)
        risk_estimator._compute = lambda: TimeSeries({0: np.array([0.5])})

        risk_estimator.initialise_component(agent, awareness_vector, knowledge_database)
        risk = risk_estimator.compute()

        # Assertions
        assert risk is not None
        callback1.assert_called_once_with(agent)
        callback2.assert_called_once_with(agent, risk)

    @pytest.mark.asyncio
    async def test_risk_estimator_async_compute_risk(
        self,
        PatchedRiskEstimator: type[RiskEstimator],
        agent: Agent,
        awareness_vector: MultiAgentAwarenessVector,
        knowledge_database: MultiAgentKnowledgeDatabase,
    ):
        risk_estimator = PatchedRiskEstimator(agent.id)

        callback1 = Mock()
        callback2 = Mock()
        risk_estimator.add_on_computing(callback1)
        risk_estimator.add_on_computed(callback2)
        risk_estimator._compute = lambda: TimeSeries({0: np.array([0.5])})

        await risk_estimator.async_initialise_component(agent, awareness_vector, knowledge_database)
        risk = await risk_estimator.async_compute()

        assert risk is not None
        callback1.assert_called_once_with(agent)
        callback2.assert_called_once_with(agent, risk)

    def test_risk_estimator_add_on_computing_risk(self, PatchedRiskEstimator: type[RiskEstimator]):
        callback = Mock()
        risk_estimator = PatchedRiskEstimator(1)
        risk_estimator.add_on_computing(callback)
        assert risk_estimator._callbacks["computing"] == {callback}
        risk_estimator.remove_on_computing(callback)
        assert risk_estimator._callbacks["computing"] == set()

    def test_risk_estimator_add_on_computed_risk(self, PatchedRiskEstimator: type[RiskEstimator]):
        callback = Mock()
        risk_estimator = PatchedRiskEstimator(1)
        risk_estimator.add_on_computed(callback)
        assert risk_estimator._callbacks["computed"] == {callback}
        risk_estimator.remove_on_computed(callback)
        assert risk_estimator._callbacks["computed"] == set()


class TestDefaultRiskEstimator:
    def test_default_risk_estimator_compute_risk(
        self,
        agent: Agent,
        awareness_vector: MultiAgentAwarenessVector,
        knowledge_database: MultiAgentKnowledgeDatabase,
    ):
        risk_estimator = DefaultRiskEstimator(agent.id)

        agent._awareness_database = awareness_vector
        risk_estimator.initialise_component(agent, awareness_vector, knowledge_database)
        risk = risk_estimator.compute()

        assert risk is not None
        assert len(risk) == 0

    @pytest.mark.asyncio
    async def test_default_risk_estimator_async_compute_risk(
        self,
        agent: Agent,
        awareness_vector: MultiAgentAwarenessVector,
        knowledge_database: MultiAgentKnowledgeDatabase,
    ):
        risk_estimator = DefaultRiskEstimator(agent.id)

        agent._awareness_database = awareness_vector
        await risk_estimator.async_initialise_component(agent, awareness_vector, knowledge_database)
        risk = await risk_estimator.async_compute()

        assert risk is not None
        assert len(risk) == 0

    def test_default_risk_estimator_compute_risk_from_old(
        self,
        agent: Agent,
        awareness_vector: MultiAgentAwarenessVector,
        knowledge_database: MultiAgentKnowledgeDatabase,
    ):
        risk_estimator = DefaultRiskEstimator(agent.id)

        awareness_vector[agent.id].risk = TimeSeries({0: np.array([0.5])})

        agent._awareness_database = awareness_vector
        risk_estimator.initialise_component(agent, awareness_vector, knowledge_database)
        risk = risk_estimator.compute()

        assert risk is not None
        assert len(risk) == 1
        assert risk[0] == np.array([0.5])

    @pytest.mark.asyncio
    async def test_default_risk_estimator_async_compute_risk_from_old(
        self,
        agent: Agent,
        awareness_vector: MultiAgentAwarenessVector,
        knowledge_database: MultiAgentKnowledgeDatabase,
    ):
        risk_estimator = DefaultRiskEstimator(agent.id)

        awareness_vector[agent.id].risk = TimeSeries({0: np.array([0.5])})

        agent._awareness_database = awareness_vector
        await risk_estimator.async_initialise_component(agent, awareness_vector, knowledge_database)
        risk = await risk_estimator.async_compute()

        assert risk is not None
        assert len(risk) == 1
        assert risk[0] == np.array([0.5])
