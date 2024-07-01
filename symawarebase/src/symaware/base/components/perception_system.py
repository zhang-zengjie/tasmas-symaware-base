from abc import abstractmethod
from typing import TYPE_CHECKING

import numpy as np

from symaware.base.data import AwarenessVector, Identifier, StateObservation
from symaware.base.models import Environment
from symaware.base.utils import NullObject, Tasynclooplock

from .component import Component

if TYPE_CHECKING:
    import sys
    from typing import Any, Callable

    from symaware.base.agent import Agent

    if sys.version_info >= (3, 10):
        from typing import TypeAlias
    else:
        from typing_extensions import TypeAlias
    PerceivingInformationCallback: TypeAlias = Callable[[Agent], Any]
    PerceivedInformationCallback: TypeAlias = Callable[[Agent, dict[Identifier, StateObservation]], Any]


class PerceptionSystem(Component[Tasynclooplock, "PerceivingInformationCallback", "PerceivedInformationCallback"]):
    """
    Generic perception system of an :class:`.symaware.base.Agent`.
    It is used to perceive the state of the :class:`.Agent` the :class:`symaware.base.Environment`.
    The implementation could extend this and include more complex interaction, like sensors or omniscient perception.
    The information collected is then used to update the awareness vector and knowledge database of the agent.

    Args
    ----
    agent_id:
        Identifier of the agent this component belongs to
    environment:
        :class:`symaware.base.Environment` the perception system will observe
    async_loop_lock:
        Async loop lock to use for the perception system
    """

    def __init__(self, agent_id: int, environment: Environment, async_loop_lock: "Tasynclooplock | None" = None):
        super().__init__(agent_id, async_loop_lock)
        self._env = environment

    @property
    def environment(self) -> Environment:
        """Environment"""
        return self._env

    def perceive_self_state(self) -> np.ndarray:
        """
        Perceive the state of the agent itself

        Returns
        -------
            State of the agent

        Raises
        ------
        RuntimeError: The perception system has not been initialised yet
        """
        if not self._initialised:
            raise RuntimeError("The perception system has not been initialised yet.")
        return self._env.get_agent_state(self._agent_id)

    @abstractmethod
    def _compute(self) -> dict[Identifier, StateObservation]:
        """
        Perceive the state of each :class:`.Agent` in the :class:`symaware.base.Environment`.
        It is used to update the knowledge database of the agent.

        Example
        -------
        Create a new perception system by subclassing the :class:`.PerceptionSystem` and implementing the
        :meth:`_compute` method.

        >>> from symaware.base import PerceptionSystem, StateObservation
        >>> class MyPerceptionSystem(PerceptionSystem):
        ...     def _compute(self):
        ...         # Your implementation here
        ...         # Example:
        ...         # Get only the agent's own state
        ...         if not self._agent_id in self._env.agent_states:
        ...             return {}
        ...         return {self._agent_id: StateObservation(self._agent_id, self._env.agent_states[self._agent_id])}

        Returns
        -------
            Information perceived by the agent. Each agent's state is identified by its id.
        """
        pass

    def _update(self, perceived_information: dict[Identifier, StateObservation]):
        """
        Update the agent's model with the new perceived information.

        Example
        -------
        A new perception system could decide to override the default :meth:`_update` method.

        >>> from symaware.base import PerceptionSystem, StateObservation, Identifier
        >>> class MyPerceptionSystem(PerceptionSystem):
        ...     def _update(self, perceived_information: dict[Identifier, StateObservation]):
        ...         # Your implementation here
        ...         # Example:
        ...         # Simply override the state of the agent
        ...         self._agent.awareness_database[self._agent_id].state = perceived_information[self._agent_id].state

        Args
        ----
        perceived_information:
            Information perceived by the agent. Each agent's state is identified by its id.
        """
        for agent_id, state_observation in perceived_information.items():
            awareness_vector = self._agent.awareness_database.setdefault(
                agent_id, AwarenessVector(agent_id, state_observation.state)
            )
            awareness_vector.state = state_observation.state


class NullPerceptionSystem(PerceptionSystem[Tasynclooplock], NullObject):
    """
    Default perception system used as a placeholder.
    It is used when no perception system is set for an agent.
    An exception is raised if this object is used in any way.
    """

    def __init__(self):
        super().__init__(-1, None)  # type: ignore

    def _compute(self):
        pass


class DefaultPerceptionSystem(PerceptionSystem[Tasynclooplock]):
    """
    Default implementation of perception system.
    It returns the values collected from the environment.

    Args
    ----
    agent_id:
        Identifier of the agent this component belongs to
    """

    def _compute(self) -> dict[Identifier, StateObservation]:
        """
        Compute the information perceived by the agent.
        It returns the values collected from the environment.

        Returns
        -------
            Information perceived by the agent. Each agent's state is identified by its id.
        """
        return {agent_id: StateObservation(agent_id, state) for agent_id, state in self._env.agent_states.items()}
