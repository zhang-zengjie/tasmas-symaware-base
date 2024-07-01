import asyncio
import signal
import time
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Coroutine,
    Generic,
    Iterable,
    NoReturn,
    TypeVar,
)

from symaware.base.data import (
    Identifier,
    KnowledgeDatabase,
    MultiAgentAwarenessVector,
    MultiAgentKnowledgeDatabase,
)
from symaware.base.models import Environment
from symaware.base.utils import get_logger

from .agent import Agent

if TYPE_CHECKING:
    # Forward declarations and 3.9 type hinting
    import sys

    if sys.version_info >= (3, 10):
        from typing import TypeAlias
    else:
        from typing_extensions import TypeAlias

T = TypeVar("T", bound=KnowledgeDatabase)
CoroutineCallbacks: "TypeAlias" = Callable[["AgentCoordinator"], Coroutine[Any, Any, None]]
InitialisationInfo: "TypeAlias" = dict[Identifier, tuple[MultiAgentAwarenessVector, MultiAgentKnowledgeDatabase[T]]]


class AgentCoordinator(Generic[T]):  # pylint: disable=function-redefined
    """
    The agent coordinator is responsible for running all agents and the environment in an asynchronous manner.
    It is advised to use this class instead of running the agents and environment manually.

    After loading the agents and the environment,
    the coordinator can be started by calling the :meth:`run` or :meth:`async_run` methods.
    It is possible to add a callback that will be executed after the initialisation
    and one after the termination of the agents and the environment.

    Args
    ----
    env:
        Environment the agents will interact with and the coordinator will run
    agents:
        Collection of agents that will be run by the coordinator
    post_init:
        Callback that will be executed after the initialisation of the agents and the environment
    post_stop:
        Callback that will be executed after the termination of the agents and the environment
    async_post_init:
        Asynchronous callback that will be executed after the initialisation of the agents and the environment
    async_post_stop:
        Asynchronous callback that will be executed after the termination of the agents and the environment
    """

    __LOGGER = get_logger(__name__, "AgentCoordinator")

    def __init__(
        self,
        env: Environment,
        agents: "Iterable[Agent] | Agent | None" = None,
        post_init: "Callable[[AgentCoordinator], None] | None" = None,
        post_stop: "Callable[[AgentCoordinator], None] | None" = None,
        async_post_init: "CoroutineCallbacks | None" = None,
        async_post_stop: "CoroutineCallbacks | None" = None,
    ):
        self._agents: list[Agent] = []
        self._env = env
        self._running = False
        self._loop: "asyncio.AbstractEventLoop | None" = None
        self._post_init = post_init
        self._post_stop = post_stop
        self._async_post_init = async_post_init
        self._async_post_stop = async_post_stop
        self._run_task: "asyncio.Task | None" = None

        self.add_agents(agents or [])

    def add_agents(self, agents: "Iterable[Agent] | Agent"):
        """
        Add agents to the coordinator.

        Args
        ----
        agents:
            Collection of agents that will be run by the coordinator

        Raises
        ------
        ValueError: Agents must be an Agent or an Iterable of Agent
        """
        if isinstance(agents, Agent):
            self._agents.append(agents)
        elif isinstance(agents, Iterable):
            self._agents.extend(agents)
        else:
            raise ValueError("agents must be an Agent or an Iterable of Agent")

    @property
    def agents(self) -> list[Agent]:
        """Collection of agents that will be run by the coordinator"""
        return self._agents

    @property
    def env(self) -> Environment:
        """Environment the agents will interact with and the coordinator will run"""
        return self._env

    @property
    def running(self) -> bool:
        """Whether the coordinator is running"""
        return self._running

    @property
    def is_initialised(self) -> bool:
        """Whether all agents are initialised"""
        return all(agent.is_initialised for agent in self._agents)

    async def _initialise(self, initialise_info: "InitialisationInfo"):
        """
        Initialise all agents asynchronously.
        Only the agents that have not been initialised before
        and are present in the `initialise_info` will be initialised.

        Args
        ----
        initialise_info:
            Dictionary containing the initialisation information for each agent.
            The agent id is the key and the value is a tuple
            containing the awareness vector and the knowledge database
        """
        self._env.initialise()
        for agent in self._agents:
            if agent.is_initialised or agent.id not in initialise_info:
                continue
            await agent.async_initialise_agent(*initialise_info[agent.id])
            self.__LOGGER.debug("Agent %s has been initialised: %s", agent.id, initialise_info)
        self.__LOGGER.info("Initialised")

    async def _run(self):
        """
        Run all agents and the environment asynchronously.
        The method will return when all agents and the environment have been stopped.
        """
        self._running = True
        self.__LOGGER.info("Running all agents and environment")
        agent_runs = (agent.async_run() for agent in self._agents)
        await asyncio.gather(*agent_runs, self._env.async_run())

    async def _stop(self):
        """
        Stop all agents and the environment gracefully.
        The method will return when all agents and the environment have been stopped.
        """
        assert self._run_task is not None
        self._running = False
        agent_stop = (agent.async_stop() for agent in self._agents)
        await asyncio.gather(*agent_stop, self._env.async_stop())
        await self._run_task  # Wait for the run task to finish gracefully
        self.__LOGGER.info("All agents and the environment have been stopped gracefully")

    def _set_loop_stop_signals(self, stop_signals: tuple[signal.Signals, ...]):
        """
        Set the stop signals for the event loop.
        By default, the signals SIGINT, SIGTERM and SIGABRT are used.

        Args
        ----
        stop_signals:
            Signals that will cause the event loop to stop gracefully
        """
        assert self._loop is not None

        if len(stop_signals) == 0:
            stop_signals = (signal.SIGINT, signal.SIGTERM, signal.SIGABRT)

        try:
            for sig in stop_signals:
                self._loop.add_signal_handler(sig, self._raise_system_exit)
        except NotImplementedError as exc:
            self.__LOGGER.warning(
                "Could not add signal handlers for the stop signals %s due to "
                "exception `%s`. If your event loop does not implement `add_signal_handler`,"
                " please pass `stop_signals=None`.",
                stop_signals,
                exc,
            )

    def run(self, time_step: float, timeout: float = -1, initialise_info: "InitialisationInfo | None" = None):
        """
        Start running all the agents and the environment.
        All agents will begin running their components synchronously.
        The frequency at which each :class:`.Component` is executed is determined by the `time_step`.
        All agents that have not yet been initialised will be initialised before running, as long as the
        `initialise_info` contains the initialisation information for the agent,
        namely a tuple with the awareness vector and the knowledge database.

        Example
        -------
        The following example shows how to create an environment, agents and a coordinator.

        >>> from symaware.base import (
        ...     AgentCoordinator,
        ...     Agent,
        ...     Environment,
        ...     TimeIntervalAsyncLoopLock,
        ...     Entity,
        ...     DynamicalModel,
        ... )
        >>>
        >>> class MyEnvironment(Environment):
        ...     ...
        >>> class MyEntity(Entity):
        ...     ...
        >>> class MyModel(DynamicalModel):
        ...     ...
        >>>
        >>> # Create the environment
        >>> env = MyEnvironment(async_loop_lock=TimeIntervalAsyncLoopLock(0.1)) # doctest: +SKIP
        >>>
        >>> # Create the agents
        >>> agents = [Agent(i, MyEntity(i, MyModel(i))) for i in range(1, 5)] # doctest: +SKIP
        >>>
        >>> # Set the components for each agent
        >>> for agent in agents: # doctest: +SKIP
        ...     agent.set_components(...) # doctest: +SKIP
        >>>
        >>> # Create the coordinator
        >>> coordinator = AgentCoordinator(env, agents) # doctest: +SKIP
        >>>
        >>> # Run the coordinator
        >>> coordinator.run() # doctest: +SKIP

        Args
        ----
        time_step:
            Time step representing the interval at which the agents and the environment will be updated
        timeout:
            Time in seconds after which the event loop will stop.
            If the timeout is negative, the event loop will run indefinitely
        initialise_info:
            Dictionary containing the initialisation information for each agent.
            The agent id is the key and the value is a tuple
            containing the awareness vector, the knowledge database and the time

        Raises
        ------
        Exception: An unforeseen exception was raised during execution
        """
        for agent in self._agents:
            self._env.initialise()
            if agent.is_initialised or initialise_info is None or agent.id not in initialise_info:
                continue
            agent.initialise_agent(*initialise_info[agent.id])
            self.__LOGGER.debug("Agent %s has been initialised: %s", agent.id, initialise_info)
        self.__LOGGER.info("Initialised")
        if self._post_init is not None:
            self._post_init(self)
        try:
            starting_time = time.time()
            end_time = time.time()
            while timeout < 0 or end_time - starting_time < timeout:
                self._env.step()
                for agent in self._agents:
                    agent.step()
                time.sleep(time_step)
                end_time = time.time()
        except (KeyboardInterrupt, SystemExit):
            self.__LOGGER.info("Received stop signal. Stopping agents.")
        except Exception as exc:
            self.__LOGGER.error("An error occurred: %s", exc)
            raise exc
        finally:
            self._env.stop()
            if self._post_stop is not None:
                self._post_stop(self)

    def async_run(
        self,
        initialise_info: "InitialisationInfo | None" = None,
        timeout: float = -1,
        loop: "asyncio.AbstractEventLoop | None" = None,
        stop_signals: "tuple[signal.Signals, ...] | None" = None,
        close_loop: bool = True,
    ):
        """
        Start running the event loop.
        All agents will begin running their components asynchronously.
        The frequency at which each :class:`.Component` is executed is determined by their :class:`.AsyncLoopLock`.
        All agents that have not yet been initialised will be initialised before running, as long as the
        `initialise_info` contains the initialisation information for the agent,
        namely a tuple with the awareness vector and the knowledge database.

        Example
        -------
        The following example shows how to create an environment, agents and a coordinator.

        >>> from symaware.base import (
        ...     AgentCoordinator,
        ...     Agent,
        ...     Environment,
        ...     TimeIntervalAsyncLoopLock,
        ...     Entity,
        ...     DynamicalModel,
        ... )
        >>>
        >>> class MyEnvironment(Environment):
        ...     ...
        >>> class MyEntity(Entity):
        ...     ...
        >>> class MyModel(DynamicalModel):
        ...     ...
        >>>
        >>> # Create the environment
        >>> env = MyEnvironment(async_loop_lock=TimeIntervalAsyncLoopLock(0.1)) # doctest: +SKIP
        >>>
        >>> # Create the agents
        >>> agents = [Agent(i, MyEntity(i, MyModel(i))) for i in range(1, 5)] # doctest: +SKIP
        >>>
        >>> # Set the components for each agent
        >>> for agent in agents: # doctest: +SKIP
        ...     agent.set_components(...) # doctest: +SKIP
        >>>
        >>> # Create the coordinator
        >>> coordinator = AgentCoordinator(env, agents) # doctest: +SKIP
        >>>
        >>> # Run the coordinator
        >>> coordinator.async_run() # doctest: +SKIP

        Args
        ----
        initialise_info:
            Dictionary containing the initialisation information for each agent.
            The agent id is the key and the value is a tuple
            containing the awareness vector, the knowledge database and the time
        timeout:
            Time in seconds after which the event loop will stop.
            If the timeout is negative, the event loop will run indefinitely
        loop:
            event loop. If none is provided, the default one will be used
        stop_signals:
            Signals that will cause the event loop to stop gracefully
        close_loop:
            Whether to close the event loop upon termination

        Raises
        ------
        Exception: An unforeseen exception was raised during execution
        """
        self._loop = loop or asyncio.get_event_loop()
        self._set_loop_stop_signals(stop_signals or tuple())

        try:
            self._loop.run_until_complete(self._initialise(initialise_info or {}))
            if self._async_post_init is not None:
                self._loop.run_until_complete(self._async_post_init(self))
            self._run_task = self._loop.create_task(self._run())
            if timeout > 0:
                self._loop.call_later(timeout, self._raise_sigint)
            self._loop.run_forever()
        except (KeyboardInterrupt, SystemExit):
            self.__LOGGER.info("Received stop signal. Stopping agents.")
        except Exception as exc:
            self.__LOGGER.error("An error occurred: %s", exc)
            raise exc
        finally:
            try:
                if self.running:
                    self._loop.run_until_complete(self._stop())
                if self._async_post_stop is not None:
                    self._loop.run_until_complete(self._async_post_stop(self))
            finally:
                if close_loop:
                    self._loop.close()

    @staticmethod
    def _raise_sigint():
        """
        Raise a SIGINT signal to stop the event loop gracefully.
        """
        signal.raise_signal(signal.SIGINT)

    @staticmethod
    def _raise_system_exit() -> NoReturn:
        """
        Raise a SystemExit exception.
        Used to stop the event loop gracefully.

        Raises
        ------
        SystemExit: A signal was received to stop the event loop
        """
        raise SystemExit
