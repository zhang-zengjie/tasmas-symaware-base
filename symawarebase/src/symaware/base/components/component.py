from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Callable, Generic, TypeVar

from symaware.base.data import (
    Identifier,
    MultiAgentAwarenessVector,
    MultiAgentKnowledgeDatabase,
)
from symaware.base.utils import AsyncLoopLockable, Publisher, Tasynclooplock

if TYPE_CHECKING:
    import sys

    from symaware.base.agent import Agent

    if sys.version_info >= (3, 10):
        from typing import TypeAlias
    else:
        from typing_extensions import TypeAlias

    InitialisedCallback: TypeAlias = Callable[[Agent, MultiAgentAwarenessVector, MultiAgentKnowledgeDatabase], Any]
    # ComputingCallback: TypeAlias = Callable[..., Any]
    # ComputedCallback: TypeAlias = Callable[..., Any]

ComputingCallback = TypeVar("ComputingCallback", bound=Callable[..., Any])
ComputedCallback = TypeVar("ComputedCallback", bound=Callable[..., Any])


class Component(
    Publisher, AsyncLoopLockable[Tasynclooplock], ABC, Generic[Tasynclooplock, ComputingCallback, ComputedCallback]
):
    """
    Generic component of an :class:`.Agent`.
    All specialized components should inherit from this class.

    Args
    ----
    agent_id:
        Identifier of the agent this component belongs to
    async_loop_lock:
        Async loop lock to use to wait for each loop if the component is used asynchronously
    """

    def __init__(self, agent_id: "Identifier", async_loop_lock: "Tasynclooplock | None" = None):
        Publisher.__init__(self)
        AsyncLoopLockable.__init__(self, async_loop_lock)
        self._agent: "Agent"
        self._agent_id = agent_id
        self._initialised = False

    @property
    def agent_id(self) -> Identifier:
        """Agent identifier"""
        return self._agent_id

    @property
    def is_initialised(self) -> bool:
        """Flag indicating whether the component has been initialised"""
        return self._initialised

    def initialise_component(
        self,
        agent: "Agent",
        initial_awareness_database: MultiAgentAwarenessVector,
        initial_knowledge_database: MultiAgentKnowledgeDatabase,
    ):
        """
        Initialize the component with some custom logic.
        For example you can instantiate new attributes and properties of the component.
        This function is called upon initialization by the :class:`.Agent`.
        To get the agent's initial awareness and knowledge database you can use the arguments of this function.
        Make sure to call the super method at the end of your implementation to notify the subscribers of the event.

        Note
        ----
        Invoking this method will notify the subscribers
        of the event ``initialised`` added with :meth:`add_on_initialised`.

        Warning
        -------
        The implementation of the method in :class:`.Component` notifies the subscribers of the event ``initialised``,
        sets the attribute ``_agent`` to the agent passed as argument and the flag ``_initialised`` to ``True``.
        If you override this method, make sure to call the super method at the end of your implementation.

        Example
        -------
        The :meth:`initialise_component` method can be overwritten to provide some custom initialisation logic.

        >>> import time
        >>> import numpy as np
        >>> from symaware.base import Component
        >>> class MyComponent(Component):
        ...     def initialise_component(
        ...            self, agent, initial_awareness_database, initial_knowledge_database
        ...         ):
        ...         self._my_model = agent.entity.model
        ...         self._my_state = initial_awareness_database[self.agent_id].state
        ...         self._my_time = time.time()
        ...         self._my_list = []
        ...         super().initialise_component(entity, initial_awareness_database, initial_knowledge_database)

        Args
        ----
        agent:
            Agent this component has been attached to
        initial_awareness_database:
            Awareness database of the agent
        initial_knowledge_database:
            Knowledge database of the agent
        """
        assert agent.id == self._agent_id
        self._agent = agent
        self._initialised = True
        self._notify("initialised", agent, initial_awareness_database, initial_knowledge_database)

    async def async_initialise_component(
        self,
        agent: "Agent",
        initial_awareness_database: MultiAgentAwarenessVector,
        initial_knowledge_database: MultiAgentKnowledgeDatabase,
    ):
        """
        Initialize the component with some custom logic asynchronously.

        Note
        ----
        Check :meth:`initialise_component` for more information about the method
        and :class:`.AsyncLoopLockable` for more information about the async loop.

        Args
        ----
        agent:
            Agent this component has been attached to
        initial_awareness_database:
            Awareness database of the agent
        initial_knowledge_database:
            Knowledge database of the agent
        """
        return self.initialise_component(agent, initial_awareness_database, initial_knowledge_database)

    @abstractmethod
    def _compute(self, *args, **kwargs) -> Any:
        """
        This very generic method is to be implemented by the specialised components.
        Given the state of the agent, compute a value or a set of values.
        Those values can be used to update the agent's state or to compute the control input of the agent.

        This method must be implemented in any custom component.

        Example
        -------
        The :meth:`compute` method can be implemented to perform some custom computation.

        >>> import numpy as np
        >>> from symaware.base import Component
        >>> class MyComponent(Component):
        ...     def _compute(self):
        ...         state = self._agent.self_state
        ...         risk = self._agent.self_awareness.risk.get(0, np.zeros(1))
        ...         return state * np.maximum(0, risk)

        Args
        ----
        args:
            Additional arguments
        kwargs:
            Additional keyword arguments

        Returns
        -------
            Computed value or set of values
        """
        pass

    async def _async_compute(self, *args, **kwargs) -> Any:
        """
        Given the state of the agent, compute a value or a set of values asynchronously.

        Note
        ----
        Check :meth:`_compute` for more information about the method
        and :class:`.AsyncLoopLockable` for more information about the async loop.

        Args
        ----
        args:
            Additional arguments
        kwargs:
            Additional keyword arguments

        Returns
        -------
            Computed value or set of values
        """
        return self._compute(*args, **kwargs)

    @abstractmethod
    def _update(self, *args, **kwargs):
        """
        This very generic method is to be implemented by the specialised components.
        Update the agent with some new values, usually computed by the :meth:`compute` method.

        This method must be implemented in any custom component.

        Example
        -------
        The :meth:`update` method can be implemented to perform some custom updates.

        >>> import numpy as np
        >>> from symaware.base import Component, TimeSeries
        >>> class MyComponent(Component):
        ...     def _update(self, new_risk: TimeSeries, new_state: np.ndarray):
        ...         self._agent.self_awareness.risk = new_risk
        ...         self._agent.self_state = new_state

        Args
        ----
        args:
            Additional arguments
        kwargs:
            Additional keyword arguments
        """
        pass

    async def _async_update(self, *args, **kwargs):
        """
        Update the agent with some new values, usually computed by the :meth:`compute` method, asynchronously.

        Note
        ----
        Check :meth:`_update` for more information about the method
        and :class:`.AsyncLoopLockable` for more information about the async loop.

        Args
        ----
        args:
            Additional arguments
        kwargs:
            Additional keyword arguments
        """
        return self._update(*args, **kwargs)

    def compute(self, *args, **kwargs) -> Any:
        """
        Given the state of the agent, compute a value or a set of values.
        Those values can be used to update the agent's state or to compute the control input of the agent.
        Internally, this method calls the :meth:`_compute` method.
        Although not necessary, it is possible to override it with the correct signature.
        Just make sure to call the super method.

        Note
        ----
        Invoking this method will notify the subscribers
        of the events `computing` and ``computed`` added with
        :meth:`add_on_computing` and :meth:``add_on_computed`` respectively.

        Args
        ----
        args:
            Additional arguments
        kwargs:
            Additional keyword arguments

        Returns
        -------
            Computed value or set of values

        Raises
        ------
        RuntimeError: If the controller has not been initialised yet
        """
        if not self.is_initialised:
            raise RuntimeError("Component has not been initialised")
        self._notify("computing", self._agent, *args, **kwargs)
        res = self._compute(*args, **kwargs)
        self._notify("computed", self._agent, res)
        return res

    async def async_compute(self, *args, **kwargs) -> Any:
        """
        Given the state of the agent, compute a value or a set of values.

        Note
        ----
        Check :meth:`compute` for more information about the method
        and :class:`.AsyncLoopLockable` for more information about the async loop.

        Args
        ----
        args:
            Additional arguments
        kwargs:
            Additional keyword arguments

        Returns
        -------
            Computed value or set of values

        Raises
        ------
        RuntimeError: If the controller has not been initialised yet
        """
        if not self.is_initialised:
            raise RuntimeError("Component has not been initialised")
        self._notify("computing", self._agent, *args, **kwargs)
        res = await self._async_compute(*args, **kwargs)
        self._notify("computed", self._agent, res)
        return res

    def update(self, *args, **kwargs):
        """
        Given some new values, update the state of the agent.
        Internally, this method calls the :meth:`_update` method.
        Although not necessary, it is possible to override it with the correct signature.
        Just make sure to call the super method.

        Note
        ----
        Invoking this method will notify the subscribers
        of the events `updating` and ``updated`` added with
        :meth:`add_on_updating` and :meth:``add_on_updated`` respectively.

        Args
        ----
        args:
            Additional arguments
        kwargs:
            Additional keyword arguments

        Raises
        ------
        RuntimeError: If the controller has not been initialised yet
        """
        if not self.is_initialised:
            raise RuntimeError("Component has not been initialised")
        self._notify("updating", self._agent, *args, **kwargs)
        self._update(*args, **kwargs)
        self._notify("updated", self._agent)

    async def async_update(self, *args, **kwargs):
        """
        Given some new values, update the state of the agent asynchronously.

        Note
        ----
        Check :meth:`update` for more information about the method
        and :class:`.AsyncLoopLockable` for more information about the async loop.

        Args
        ----
        args:
            Additional arguments
        kwargs:
            Additional keyword arguments

        Raises
        ------
        RuntimeError: If the controller has not been initialised yet
        """
        if not self.is_initialised:
            raise RuntimeError("Component has not been initialised")
        self._notify("updating", self._agent, *args, **kwargs)
        await self._async_update(*args, **kwargs)
        self._notify("updated", self._agent)

    def compute_and_update(self, *args, **kwargs) -> Any:
        """
        Compute and update the component in one go.
        See :meth:`compute` and :meth:`update` for more information.

        Returns:
            Computed value or set of values
        """
        res = self.compute(*args, **kwargs)
        self.update(res)
        return res

    async def async_compute_and_update(self, *args, **kwargs) -> Any:
        """
        Compute and update the component in one go asynchronously.
        See :meth:`compute` and :meth:`update` for more information.

        Note
        ----
        Check :meth:`compute_and_update` for more information about the method
        and :class:`.AsyncLoopLockable` for more information about the async loop.

        Returns:
            Computed value or set of values
        """
        res = await self.async_compute(*args, **kwargs)
        await self.async_update(res)
        return res

    def add_on_initialised(self, callback: "InitialisedCallback"):
        """
        Add a callback to the event ``initialised``

        Args
        ----
        callback:
            Callback to add
        """
        self._add("initialised", callback)

    def remove_on_initialised(self, callback: "InitialisedCallback"):
        """
        Remove a callback from the event ``initialised``

        Args
        ----
        callback:
            Callback to remove
        """
        self._remove("initialised", callback)

    def add_on_computing(self, callback: "ComputingCallback"):
        """
        Add a callback to the event ``computing``

        Args
        ----
        callback:
            Callback to add
        """
        self._add("computing", callback)

    def remove_on_computing(self, callback: "ComputingCallback"):
        """
        Remove a callback from the event ``computing``

        Args
        ----
        callback:
            Callback to remove
        """
        self._remove("computing", callback)

    def add_on_computed(self, callback: "ComputedCallback"):
        """
        Add a callback to the event ``computed``

        Args
        ----
        callback:
            Callback to add
        """
        self._add("computed", callback)

    def remove_on_computed(self, callback: "ComputedCallback"):
        """
        Remove a callback from the event ``computed``

        Args
        ----
        callback:
            Callback to remove
        """
        self._remove("computed", callback)

    def add_on_updating(self, callback: "ComputingCallback"):
        """
        Add a callback to the event ``updating``

        Args
        ----
        callback:
            Callback to add
        """
        self._add("updating", callback)
