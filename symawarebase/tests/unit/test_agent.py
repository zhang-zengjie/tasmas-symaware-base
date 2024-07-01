# pylint: disable=missing-function-docstring, missing-class-docstring, redefined-outer-name, no-self-use, unused-argument, abstract-class-instantiated, invalid-name
import pytest

from symaware.base import (
    Agent,
    DefaultCommunicationReceiver,
    DefaultCommunicationSender,
    DefaultController,
    DefaultPerceptionSystem,
    DefaultRiskEstimator,
    DefaultUncertaintyEstimator,
    Environment,
    MultiAgentAwarenessVector,
    MultiAgentKnowledgeDatabase,
    NullObject,
)


class TestAgent:
    def test_agent_init(self, agent: Agent):
        assert agent.entity is not None
        assert agent.entity.id == agent.id
        assert agent.entity.model is not None
        assert agent.is_initialised is False
        for component in agent.components:
            assert component is not None
            assert isinstance(component, NullObject)

    def test_agent_add_components(self, agent: Agent, environment: Environment):
        agent.add_components(
            DefaultPerceptionSystem(agent.id, environment),
            DefaultCommunicationReceiver(agent.id),
            DefaultRiskEstimator(agent.id),
            DefaultUncertaintyEstimator(agent.id),
            DefaultController(agent.id),
            DefaultCommunicationSender(agent.id),
        )
        for component in agent.components:
            assert component is not None
            assert not isinstance(component, NullObject)
        assert len(agent.components) == 6

    @pytest.mark.parametrize("agent", (1,), indirect=True)
    @pytest.mark.parametrize("awareness_vector", (2,), indirect=True)
    def test_agent_initialise_mismatch_id(
        self,
        agent: Agent,
        awareness_vector: MultiAgentAwarenessVector,
        knowledge_database: MultiAgentKnowledgeDatabase,
        environment: Environment,
    ):
        agent.add_components(
            DefaultPerceptionSystem(agent.id, environment),
            DefaultCommunicationReceiver(agent.id),
            DefaultRiskEstimator(agent.id),
            DefaultUncertaintyEstimator(agent.id),
            DefaultController(agent.id),
            DefaultCommunicationSender(agent.id),
        )
        with pytest.raises(ValueError):
            agent.initialise_agent(awareness_vector, knowledge_database)

    def test_agent_initialise(
        self,
        agent: Agent,
        awareness_vector: MultiAgentAwarenessVector,
        knowledge_database: MultiAgentKnowledgeDatabase,
        environment: Environment,
    ):
        agent.add_components(
            DefaultPerceptionSystem(agent.id, environment),
            DefaultCommunicationReceiver(agent.id),
            DefaultRiskEstimator(agent.id),
            DefaultUncertaintyEstimator(agent.id),
            DefaultController(agent.id),
            DefaultCommunicationSender(agent.id),
        )
        agent.initialise_agent(awareness_vector, knowledge_database)
        assert agent.is_initialised is True
        for component in agent.components:
            assert component.is_initialised is True
