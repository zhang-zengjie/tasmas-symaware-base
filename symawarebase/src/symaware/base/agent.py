import asyncio
from typing import TYPE_CHECKING, Generic, TypeVar

import numpy as np

from symaware.base.components import (
    CommunicationReceiver,
    CommunicationSender,
    Component,
    Controller,
    NullCommunicationReceiver,
    NullCommunicationSender,
    NullController,
    NullPerceptionSystem,
    NullRiskEstimator,
    NullUncertaintyEstimator,
    PerceptionSystem,
    RiskEstimator,
    UncertaintyEstimator,
)
from symaware.base.data import (
    AwarenessVector,
    Identifier,
    KnowledgeDatabase,
    MultiAgentAwarenessVector,
    MultiAgentKnowledgeDatabase,
)
from symaware.base.models import DynamicalModel, Entity
from symaware.base.utils import get_logger, log

if TYPE_CHECKING:
    from typing import Iterable

# Define the type of the agent based on the type of the awareness vector
T = TypeVar("T", bound=KnowledgeDatabase)


class Agent(Generic[T]):
    # pylint: disable=line-too-long
    """
    An agent represents an entity that can interact with the environment and other agents.
    It is made of several components.
    While the exact kind and number is up to the instance, usually these are the most common ones:

    - An :class:`symaware.base.Entity` representing the simulated physical entity of the agent, with a dynamical model.
    - A :class:`.Controller` that computes the control input of the agent based on the current information.
    - A :class:`.RiskEstimator` that computes the risk of the agent based on the current information.
    - An :class:`.UncertaintyEstimator` that computes the uncertainty of the agent based on the current information.
    - A :class:`.PerceptionSystem` that collects information from the environment.
    - A :class:`.CommunicationSystem` that collects information from other agents and sends information to other agents.

    Each component can easily be replaced by a custom one by subclassing the corresponding class.

    Agents support for synchronous and asynchronous execution.

    Note
    ----
    To simplify coordination between multiple agents, the use of a :class:`.AgentCoordinator` is recommended.

    Example
    -------
    Create an agent with a simple dynamical model and the default components using the pybullet simulator:

    >>> # doctest: +SKIP
    >>> import pybullet as p
    >>> import numpy as np
    >>> from symaware.simulators.pybullet import Environment, RacecarEntity, RacecarModel
    >>> from symaware.base import Agent, TimeIntervalAsyncLoopLock, KnowledgeDatabase, AgentCoordinator, AwarenessVector
    >>> from symaware.base import DefaultCommunicationSender, DefaultCommunicationReceiver
    >>> from symaware.base import DefaultInfoUpdater, DefaultController
    >>> from symaware.base import DefaultPerceptionSystem, DefaultRiskEstimator, DefaultUncertaintyEstimator
    >>> agent_id = 0
    >>> env = Environment(connection_method=p.DIRECT, async_loop_lock=TimeIntervalAsyncLoopLock(0.1))
    >>> entity = RacecarEntity(agent_id, model=RacecarModel(agent_id))
    >>> agent = Agent[KnowledgeDatabase](agent_id, entity)
    >>> env.add_agents(agent)
    >>> agent.add_components(
    ...     DefaultController(agent_id, TimeIntervalAsyncLoopLock(0.1)),
    ...     DefaultRiskEstimator(agent_id, TimeIntervalAsyncLoopLock(0.1)),
    ...     DefaultUncertaintyEstimator(agent_id, TimeIntervalAsyncLoopLock(0.1)),
    ...     DefaultPerceptionSystem(agent_id, env, TimeIntervalAsyncLoopLock(0.1)),
    ...     DefaultCommunicationSender(agent_id, TimeIntervalAsyncLoopLock(0.1)),
    ...     DefaultCommunicationReceiver(agent_id, TimeIntervalAsyncLoopLock(0.1)),
    ... )
    >>> agent.initialise_agent(
    ...     initial_awareness_database=AwarenessVector(agent_id, np.zeros(7)),
    ...     initial_knowledge_database={agent_id: KnowledgeDatabase(goal_reached=False, goal_pos=np.zeros(3))},
    ... )
    >>> AgentCoordinator(env, agent).run()

    Args
    ----
    ID:
        Identifier of the agent
    entity:
        Entity representing the simulated physical entity of the agent
    """

    _LOGGER = get_logger(__name__, "Agent")

    def __init__(self, ID: Identifier, entity: Entity):
        if not isinstance(ID, Identifier):
            raise TypeError("Agent ID must be an Identifier")
        if not isinstance(entity, Entity):
            raise TypeError("Agent entity must be an Entity")
        if not ID == entity.id:
            raise ValueError("Agent ID and entity ID must be the same, otherwise the controller will not work")

        self._ID = ID
        self._entity = entity

        # set your controller and respective perception and risk module
        self._components: set[Component] = set()

        # these are set up calling the function set_up
        # dictionary containing all the awareness vector from this agent and its neighbours
        self._awareness_database: MultiAgentAwarenessVector = {}

        # awareness database. Each entry is a dictionary associates with a unique key value.
        # The subdictionaries can contain any kind of free entry.
        # The keys represent the agents ID to which the knowledge refers to
        self._knowledge_database: MultiAgentKnowledgeDatabase[T] = {}
        self._is_initialised: bool = False  # flag to check if the agent has been initialised or not

        # Flag used in the async_run method to stop the agent
        self._running = False
        self._has_finished = asyncio.Event()

    # these properties must be set up by the user when defining the child concrete class
    @property
    def id(self) -> Identifier:
        """Agent ID"""
        return self._ID

    @property
    def entity(self) -> Entity:
        """Agent entity"""
        return self._entity

    @property
    def model(self) -> DynamicalModel:
        """Dynamical model for the agent"""
        return self._entity.model

    @property
    def current_state(self) -> np.ndarray:
        """Current state of the agent"""
        return self._awareness_database[self._ID].state

    @property
    def awareness_database(self) -> MultiAgentAwarenessVector:
        """Agent's awareness database"""
        return self._awareness_database

    @property
    def self_awareness(self) -> AwarenessVector:
        """Agent's awareness vector"""
        return self._awareness_database[self._ID]

    @property
    def control_input(self) -> np.ndarray:
        """Agent's control input"""
        return self._entity.model.control_input

    @property
    def self_state(self) -> np.ndarray:
        """Agent's state"""
        return self._awareness_database[self._ID].state

    @property
    def knowledge_database(self) -> MultiAgentKnowledgeDatabase[T]:
        """Agent's knowledge database"""
        return self._knowledge_database

    @property
    def is_initialised(self) -> bool:
        """Check if the :meth:`initialise_agent` method has been called before"""
        return self._is_initialised

    ##############
    # Components #
    ##############
    @property
    def risk_estimators(self) -> tuple[RiskEstimator, ...]:
        """Risk estimators for the agent"""
        return tuple(component for component in self._components if isinstance(component, RiskEstimator))

    @property
    def uncertainty_estimators(self) -> tuple[UncertaintyEstimator, ...]:
        """Uncertainty estimator for the agent"""
        return tuple(component for component in self._components if isinstance(component, UncertaintyEstimator))

    @property
    def perception_systems(self) -> tuple[PerceptionSystem, ...]:
        """Perception systems for the agent"""
        return tuple(component for component in self._components if isinstance(component, PerceptionSystem))

    @property
    def controllers(self) -> tuple[Controller, ...]:
        """Controllers for the agent"""
        return tuple(component for component in self._components if isinstance(component, Controller))

    @property
    def communication_senders(self) -> tuple[CommunicationSender, ...]:
        """Communication system for the agent"""
        return tuple(component for component in self._components if isinstance(component, CommunicationSender))

    @property
    def communication_receivers(self) -> tuple[CommunicationReceiver, ...]:
        """Communication receivers for the agent"""
        return tuple(component for component in self._components if isinstance(component, CommunicationReceiver))

    @property
    def components(self) -> tuple[Component, ...]:
        """Tuple containing all the components of the agent"""
        return tuple(self._components)

    @property
    def risk_estimator(self) -> RiskEstimator:
        """Risk estimator for the agent. Null component is none is present"""
        return next(iter(self.risk_estimators), NullRiskEstimator.instance())

    @property
    def uncertainty_estimator(self) -> UncertaintyEstimator:
        """Uncertainty estimator for the agent. Null component is none is present"""
        return next(iter(self.uncertainty_estimators), NullUncertaintyEstimator.instance())

    @property
    def perception_system(self) -> PerceptionSystem:
        """Perception system for the agent. Null component is none is present"""
        return next(iter(self.perception_systems), NullPerceptionSystem.instance())

    @property
    def controller(self) -> Controller:
        """Controller for the agent. Null component is none is present"""
        return next(iter(self.controllers), NullController.instance())

    @property
    def communication_sender(self) -> CommunicationSender:
        """Communication system for the agent. Null component is none is present"""
        return next(iter(self.communication_senders), NullCommunicationSender.instance())

    @property
    def communication_receiver(self) -> CommunicationReceiver:
        """Communication receiver for the agent. Null component is none is present"""
        return next(iter(self.communication_receivers), NullCommunicationReceiver.instance())

    def add_components(self, *components: "Component"):
        """
        Add a component or a list of components to the agent.

        Args
        ----
        components:
            single component or a list of components to add to the agent
        """
        if self._is_initialised:
            self._LOGGER.error("Adding components to an initialised agent. This may not work as expected")
        for component in components:
            if not isinstance(component, Component):
                raise TypeError("component must be a Component object")
            if not component.agent_id == self._ID:
                raise ValueError("Component ID and agent ID must be the same, otherwise the component may not work")
            self._components.add(component)

    def remove_components(self, components: "Iterable[Component] | Component"):
        """
        Remove a component or a list of components from the agent.

        Args
        ----
        components:
            single component or a list of components to remove from the agent

        Raises
        ------
        TypeError: The component is not a Component object
        KeyError: The component is not present in the agent
        """
        if isinstance(components, Component):
            components = (components,)
        for component in components:
            if not isinstance(component, Component):
                raise TypeError("component must be a Component object")
            self._components.remove(component)

    @log(_LOGGER)
    def initialise_agent(
        self,
        initial_awareness_database: "AwarenessVector | MultiAgentAwarenessVector",
        initial_knowledge_database: "MultiAgentKnowledgeDatabase[T]",
    ):
        """
        Initializes the agent with the given initial awareness database and initial knowledge database.

        Args
        ----
        initial_awareness_database:
            The initial awareness database containing the state of the agent.
            If it is a dictionary, it must at least contain the state of the agent itself.
            Otherwise, it is assumed to be representing the state of the agent.
        initial_knowledge_database:
            The initial knowledge database.
            If it is a dictionary, it must at least contain the state of the agent itself.
            Otherwise, it is assumed to be representing the state of the agent.

        Raises
        ------
        ValueError: The agent ID is not present in the initial awareness database
            or if the dimension of the awareness vector does not match the dimension of the dynamical model state input.
        """
        if isinstance(initial_awareness_database, AwarenessVector):
            initial_awareness_database = {self._ID: initial_awareness_database}

        if not self._ID in initial_awareness_database.keys():
            raise ValueError(
                "The agent ID must be present in the initial awareness database. "
                "This database also includes the state of the agent itself"
            )
        if self._ID not in initial_knowledge_database.keys():
            raise ValueError("The agent ID must be present in the initial knowledge database.")

        self._awareness_database = initial_awareness_database
        self._knowledge_database = initial_knowledge_database
        # Initialise all components
        for component in self.components:
            component.initialise_component(
                agent=self,
                initial_awareness_database=initial_awareness_database,
                initial_knowledge_database=initial_knowledge_database,
            )

        self._is_initialised = True

    @log(_LOGGER)
    def step(self):
        """
        Step function for the agent.
        Will run at each time step.
        It will, in order:

        - Collect information from the environment
        - Receive messages from other agents
        - Compute and update the risk
        - Compute and update the uncertainty
        - Compute the control input
        - Send the awareness and knowledge databases to other agents

        Note
        ----
        If multiple components of the same kind are present,
        the order in which they are added is the order in which they will be executed.

        Example
        -------
        Override the :meth:`step` method in a subclass to change the order of components' execution.

        >>> from symaware.base import Agent
        >>> class MyAgent(Agent):
        ...     def step(self):
        ...         # Your implementation here
        ...         # Example:
        ...         # Compute all the components in the order they were added
        ...         for component in self.components:
        ...             component.compute_and_update()

        Raises
        ------
        ValueError: The agent must be initialised before running the step function.
        """
        if not self.is_initialised:
            raise ValueError("The agent must be initialised before running the step function.")
        perception_systems = self.perception_systems
        communication_receivers = self.communication_receivers
        risk_estimators = self.risk_estimators
        uncertainty_estimators = self.uncertainty_estimators
        controllers = self.controllers
        communication_senders = self.communication_senders
        for perception_system in perception_systems:
            perception_system.compute_and_update()
        for communication_receiver in communication_receivers:
            communication_receiver.compute_and_update()
        for risk_estimator in risk_estimators:
            risk_estimator.compute_and_update()
        for uncertainty_estimator in uncertainty_estimators:
            uncertainty_estimator.compute_and_update()
        for controller in controllers:
            controller.compute_and_update()
        for communication_sender in communication_senders:
            communication_sender.compute_and_update()

    @log(_LOGGER)
    async def async_initialise_agent(
        self,
        initial_awareness_database: "AwarenessVector | MultiAgentAwarenessVector",
        initial_knowledge_database: "MultiAgentKnowledgeDatabase[T]",
    ):
        """
        Initializes the agent with the given initial awareness database and initial knowledge database.

        Args
        ----
        initial_awareness_database:
            The initial awareness database containing the state of the agent.
            If it is a dictionary, it must at least contain the state of the agent itself.
            Otherwise, it is assumed to be representing the state of the agent.
        initial_knowledge_database:
            The initial knowledge database.
            If it is a dictionary, it must at least contain the state of the agent itself.
            Otherwise, it is assumed to be representing the state of the agent.

        Raises
        ------
        ValueError: The agent ID is not present in the initial awareness database
            or if the dimension of the awareness vector does not match the dimension of the dynamical model state input.
        """
        if isinstance(initial_awareness_database, AwarenessVector):
            initial_awareness_database = {self._ID: initial_awareness_database}

        if not self._ID in initial_awareness_database.keys():
            raise ValueError(
                "The agent ID must be present in the initial awareness database. "
                "This database also includes the state of the agent itself"
            )
        if self._ID not in initial_knowledge_database.keys():
            raise ValueError("The agent ID must be present in the initial knowledge database.")

        self._awareness_database = initial_awareness_database
        self._knowledge_database = initial_knowledge_database

        # Initialise all components
        await asyncio.gather(
            *(
                component.async_initialise_component(
                    agent=self,
                    initial_awareness_database=initial_awareness_database,
                    initial_knowledge_database=initial_knowledge_database,
                )
                for component in self.components
            )
        )

        self._is_initialised = True

    async def async_run(self):
        """
        Run the agent asynchronously.
        It will run all the components asynchronously.
        """
        if not self.is_initialised:
            raise ValueError("The agent must be initialised before running the step function.")
        self._running = True
        await asyncio.gather(*(self._async_step(component) for component in self.components if component.can_loop))
        self._has_finished.set()

    async def async_stop(self):
        """
        Stop the agent asynchronously.
        Makes sure all the components have stopped running to stop the agent gracefully.
        """
        self._running = False
        await asyncio.gather(*(component.async_stop() for component in self.components if component.can_loop))
        await self._has_finished.wait()

    async def _async_step(self, component: Component):
        """
        Asynchronously run the step function for the component.
        Each component will compute its new value and update the agent accordingly.
        The frequency of the loop is determined by the component's :class:`.AsyncLoopLock`.

        Args
        ----
        component:
            Component to run asynchronously
        """
        assert self.is_initialised
        await component.next_loop()
        while self._running:
            await component.async_compute_and_update()
            await component.next_loop()
