import asyncio
from abc import ABC, abstractmethod
from typing import TypeVar


class AsyncLoopLock(ABC):
    """
    Generic interface for a lock used to synchronize an async loop.
    Used by :class:`.AsyncLoopLockable` objects, in order to synchronize themselves in an async loop.
    The next iteration of the loop will run only after the object has completed its task and the lock has been released.
    In other words, the time between each iteration of the loop it determined by which of the two takes longer.

    Example
    -------
    A lock that synchronizes an async loop at a fixed time interval of 1.
    Until the task takes less than 1 second, the next loop will wait for the lock before starting,
    meaning that the task will be executed at most once per second.
    If the task takes more than 1 second, the next loop will start immediately after the task is completed.

    >>> import asyncio
    >>> from symaware.base.utils import TimeIntervalAsyncLoopLock
    >>> # Create the lock
    >>> lock = TimeIntervalAsyncLoopLock(1)
    >>>
    >>> async def task(delay: float):
    ...     asyncio.sleep(delay)
    ...
    >>> # This could be an object that implements the AsyncLoopLockable interface
    >>> async def run():
    ...     task_time = 0
    ...     while True:
    ...         await task(task_time) # Do something that takes "task_time" seconds
    ...         await lock.next_loop() # Wait for the next loop to start
    ...         task_time += 0.1
    ...

    Args
    ----
    loop:
        event loop. If none is provided, the default one will be used
    """

    def __init__(self, loop: "asyncio.AbstractEventLoop | None" = None):
        self._loop: "asyncio.AbstractEventLoop | None" = loop or asyncio.get_event_loop()
        self._released = False
        self._lock_task: "asyncio.Task[None] | None" = None

    @property
    def released(self) -> bool:
        """Whether the lock has been released. It usually means that the simulation is stopping."""
        return self._released

    async def release_loop(self):
        """
        Release the lock, causing the object to complete all future loops immediately.
        """
        self._released = True
        if self._lock_task is not None:
            self._lock_task.cancel()

    @abstractmethod
    async def next_loop(self):
        """
        Wait for the next loop to start.

        Example
        -------
        >>> async def run(self):
        ...     while True:
        ...         # Do something
        ...         await self.next_loop() # Wait for the next loop to start
        """
        pass


class TimeIntervalAsyncLoopLock(AsyncLoopLock):
    """
    Lock used to synchronize an async loop at a fixed time interval.

    Args
    ----
    time_interval:
        time interval in seconds
    loop:
        event loop. If none is provided, the default one will be used
    """

    def __init__(self, time_interval: float, loop: "asyncio.AbstractEventLoop | None" = None):
        super().__init__(loop)
        self._time_interval = time_interval
        self._last_timestamp = 0.0

    @property
    def time_interval(self) -> float:
        return self._time_interval

    @property
    def last_timestamp(self) -> float:
        return self._last_timestamp

    async def next_loop(self):
        if self._released:
            return
        assert self._loop is not None
        time_passed = self._loop.time() - self._last_timestamp
        self._lock_task = asyncio.create_task(asyncio.sleep(self.time_interval - time_passed))
        try:
            await self._lock_task
        except asyncio.CancelledError:
            pass
        self._last_timestamp = self._loop.time()


class EventAsyncLoopLock(AsyncLoopLock):
    """
    Lock used to synchronize an async loop using an event.

    Args
    ----
    loop:
        event loop. If none is provided, the default one will be used
    """

    def __init__(self, loop: "asyncio.AbstractEventLoop | None" = None):
        super().__init__(loop)
        self._event = asyncio.Event()

    @property
    def event(self) -> asyncio.Event:
        return self._event

    async def release_loop(self):
        self._released = True
        self._event.set()

    def trigger(self):
        """
        Trigger the event, causing the next loop to start.
        """
        self._event.set()

    def clear(self):
        """
        Clear the event, resetting it to the non-set state.
        The next loop will not start until the event is triggered again.
        """
        self._event.clear()

    async def next_loop(self):
        if self._released:
            return
        await self._event.wait()
        self._event.clear()


class ConditionAsyncLoopLock(AsyncLoopLock):
    """
    Lock used to synchronize an async loop using a condition.

    Args
    ----
    loop:
        event loop. If none is provided, the default one will be used
    """

    def __init__(self, loop: "asyncio.AbstractEventLoop | None" = None):
        super().__init__(loop)
        self._condition = asyncio.Condition()

    @property
    def condition(self) -> asyncio.Condition:
        return self._condition

    async def release_loop(self):
        self._released = True
        async with self._condition:
            self._condition.notify_all()

    async def trigger(self):
        """
        Trigger the event, causing the next loop to start.
        """
        async with self.condition:
            self.condition.notify_all()

    async def next_loop(self):
        if self._released:
            return
        async with self.condition:
            await self.condition.wait()


class DefaultAsyncLoopLock(AsyncLoopLock):
    """
    Default lock used to synchronize an async loop.
    It does not wait for the next loop to start.

    Args
    ----
    loop:
        event loop. If none is provided, the default one will be used
    """

    async def next_loop(self):
        await asyncio.sleep(0)


Tasynclooplock = TypeVar("Tasynclooplock", bound=AsyncLoopLock)
