from abc import abstractmethod
from typing import TYPE_CHECKING

import numpy as np

from symaware.base.data import (
    MultiAgentAwarenessVector,
    MultiAgentKnowledgeDatabase,
    TimeSeries,
)
from symaware.base.models import DynamicalModel, NullDynamicalModel
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

    ComputingControlInputCallback: TypeAlias = Callable[[Agent], Any]
    ComputedControlInputCallback: TypeAlias = Callable[[Agent, tuple[np.ndarray, TimeSeries]], Any]


class Controller(Component[Tasynclooplock, "ComputingControlInputCallback", "ComputedControlInputCallback"]):
    """
    Abstract class for the controller.
    Compute the control input for the agent.
    Normally the agent would use its information available at a certain time to compute a control input.
    This could translate in a lot of information being passed to the :meth:`compute`.

    Args
    ----
    agent_id:
        Identifier of the agent this component belongs to
    async_loop_lock:
        Async loop lock to use for the controller
    """

    def __init__(self, agent_id, async_loop_lock: "Tasynclooplock | None" = None):
        super().__init__(agent_id, async_loop_lock)
        self._dynamical_model: DynamicalModel = NullDynamicalModel.instance()

    @property
    def dynamical_model(self) -> DynamicalModel:
        """
        Dynamical system model for the agent
        """
        return self._dynamical_model

    def initialise_component(
        self,
        agent: "Agent",
        initial_awareness_database: MultiAgentAwarenessVector,
        initial_knowledge_database: MultiAgentKnowledgeDatabase,
    ):
        self._dynamical_model = agent.model
        super().initialise_component(agent, initial_awareness_database, initial_knowledge_database)

    @abstractmethod
    def _compute(self) -> tuple[np.ndarray, TimeSeries]:
        """
        Compute the control input for the agent.
        Normally the agent would use its information available at a certain time to compute a control input.
        This could translate in a lot of information being passed to this method.

        This method must be implemented in any custom controller.

        Example
        -------
        Create a new controller by subclassing the :class:`.Controller` and implementing the
        :meth:`_compute` method.

        >>> from symaware.base import Controller, MultiAgentAwarenessVector, MultiAgentKnowledgeDatabase, TimeSeries
        >>> import numpy as np
        >>> class MyController(Controller):
        ...     def _compute(self) -> tuple[np.ndarray, TimeSeries]:
        ...         # Your implementation here
        ...         # Example:
        ...         # Get the state of the agent
        ...         state = self._agent.awareness_database[self.agent_id].state
        ...         # Get the goal position from the knowledge database
        ...         goal_pos = self._agent.knowledge_database[self.agent_id]["goal_pos"]
        ...         # Compute the control input as the difference between the goal position and the current state
        ...         control_input = goal_pos - state
        ...         # Return the control input and an empty TimeSeries
        ...         return control_input, TimeSeries()


        Returns
        -------
            - New state of the agent the controller wants to reach,
            - Time series of intents of the controller, Can be empty
        """
        pass

    def _update(self, control_input_and_intent: tuple[np.ndarray, TimeSeries]):
        """
        Update the agent's model with the computed control input
        and store the intent in the agent's awareness vector.

        Example
        -------
        A new controller could decide to override the default :meth:`_update` method.

        >>> from symaware.base import Controller, TimeSeries
        >>> class MyController(Controller):
        ...     def _update(self, control_input: np.ndarray, intent: TimeSeries):
        ...         # Your implementation here
        ...         # Example:
        ...         # Simply override the control input and intent of the agent
        ...         self._agent.model.control_input = control_input
        ...         self._agent.self_awareness.intent = intent

        Args
        ----
        control_input:
            New control input to apply to the agent's model
        intent:
            New intent to store in the agent's awareness vector
        """
        control_input, intent = control_input_and_intent
        self._agent.model.control_input = control_input
        self._agent.self_awareness.intent = intent


class NullController(Controller, NullObject):
    """
    Default controller used as a placeholder.
    It is used when no controller is set for an agent.
    An exception is raised if this object is used in any way.
    """

    def __init__(self):
        super().__init__(-1)

    def _compute(self, awareness_database, knowledge_database):
        pass


class DefaultController(Controller):
    """
    Default implementation of the controller.
    It returns a zero vector of the right size as control input.

    Args
    ----
    agent_id:
        Identifier of the agent this component belongs to
    async_loop_lock:
        Async loop lock to use for the controller
    """

    def __init__(self, agent_id, async_loop_lock: "Tasynclooplock | None" = None):
        super().__init__(agent_id, async_loop_lock)
        self._control_input = np.zeros(0)

    def initialise_component(self, agent, initial_awareness_database, initial_knowledge_database):
        super().initialise_component(agent, initial_awareness_database, initial_knowledge_database)
        self._control_input = np.zeros(agent.model.control_input_shape)

    def _compute(self):
        return self._control_input, TimeSeries()
