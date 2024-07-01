from abc import ABC, abstractmethod
from collections.abc import Iterable
from typing import TYPE_CHECKING

import numpy as np

from symaware.base.data import Identifier
from symaware.base.utils import AsyncLoopLockable, get_logger

from .entity import Entity

if TYPE_CHECKING:
    # Forwards declaration
    # String type hinting to support python 3.9
    from symaware.base.utils import AsyncLoopLock

    from ..agent import Agent


class Environment(ABC, AsyncLoopLockable):
    """
    Just a support class to have multiple agents working in the same environment

    Args
    ----
    async_loop_lock:
        async loop lock to use with the environment
    """

    def __init__(self, async_loop_lock: "AsyncLoopLock | None" = None):
        super().__init__(async_loop_lock)
        self._entities: set[Entity] = set()
        # stores the states of the agent in the system
        self._agent_entities: dict[Identifier, Entity] = {}
        self._running = False

    @abstractmethod
    def _add_entity(self, entity: Entity):
        """
        Add an entity to the environment, initialising it.
        The actual implementation should be done in the derived class, based on the
        simulated environment API.
        The entity's :func:`initialise_entity` function should be called within this function
        with the appropriate arguments.

        Args
        ----
        entity:
            entity to initialise
        """
        pass

    @abstractmethod
    def get_entity_state(self, entity: Entity) -> np.ndarray:
        """
        Get the state of an entity in the environment.
        The actual implementation should be done in the derived class, based on the
        simulated environment API.

        Args
        ----
        entity:
            entity to get the state of

        Returns
        -------
            State of the entity within the environment
        """
        pass

    def get_agent_state(self, agent: "Identifier | Agent") -> np.ndarray:
        if isinstance(agent, Identifier):
            return self.get_entity_state(self._agent_entities[agent])
        return self.get_entity_state(self._agent_entities[agent.id])

    @property
    def entities(self) -> set[Entity]:
        """Set of entities in the environment"""
        return self._entities

    @property
    def agent_states(self) -> dict[Identifier, np.ndarray]:
        """Dictionary mapping agent identifiers to their states in the environment"""
        return {agent_id: self.get_agent_state(agent_id) for agent_id in self._agent_entities}

    def add_agents(self, agents: "Iterable[Agent] | Agent"):
        """
        Abstract high level interface this class exposes to add entities to the environment.
        The :func:`_add_single_obstacle` will add the entity to the underlying physics engine.

        Args
        ----
        agent:
            agent to add to the environment
        """
        if not isinstance(agents, Iterable):
            agents = (agents,)

        for agent in agents:
            self.add_entities(agent.entity)
            self._agent_entities[agent.id] = agent.entity

    def add_entities(self, entities: "Iterable[Entity] | Entity"):
        """
        Abstract high level interface this class exposes to add entities to the environment.
        The :func:`_add_single_obstacle` will add the entity to the underlying physics engine.
        Once the entity is added, it is also stored in the internal set of entities, to avoid
        adding and initialising it multiple times.

        Args
        ----
        entities:
            single instance of :class:`symaware.base.Entity` or an iterable of them
        """
        if isinstance(entities, Entity):
            entities = (entities,)
        if not isinstance(entities, Iterable):
            raise TypeError(f"Expected Iterable, got {type(entities)}")

        for entity in entities:
            if not isinstance(entity, Entity):
                raise TypeError(f"Expected entity, got {type(entity)}")
            if entity in self._entities:
                get_logger(__name__, "Environment").warning("Entity %s already present in the environment", entity)
                continue
            self._add_entity(entity)
            self._entities.add(entity)

    @abstractmethod
    def initialise(self):
        """
        Initialise the simulation, allocating the required resources.
        Should be called when the simulation has been set up and is ready to be run.
        Some environment implementations may call it automatically when the environment is created.
        It is their responsibility to ensure that the method is idempotent.
        """
        pass

    @abstractmethod
    def step(self):
        """
        It can be called repeatedly step the environment forward in time, updating the state of all the entities.
        """
        pass

    @abstractmethod
    def stop(self):
        """
        Terminate the simulation, releasing the resources.
        Should be called when the simulation has been running manually and needs to be stopped.
        Some environment implementations may call it automatically when the environment is destroyed.
        It is their responsibility to ensure that the method is idempotent.

        Warning
        -------
        Depending on the simulator implementation, calling this method may invalidate all the entities
        previously added to the environment.
        In that case, entities and the environment should be recreated from scratch.
        """
        pass

    async def async_run(self):
        """
        Start the environment loop asynchronously.
        It will run the environment until :meth:`async_stop` is called.
        The frequency of the loop is determined by the :class:`.AsyncLoopLock` used to initialise the environment.
        """
        self._running = True
        await self.next_loop()
        while self._running:
            self.step()
            await self.next_loop()

    async def async_stop(self):
        """
        Gracefully stop the environment loop asynchronously.
        Once the last cycle is completed, the control is returned to the caller.
        """
        self._running = False
        await AsyncLoopLockable.async_stop(self)
        self.stop()
