from typing import Generic

from .async_loop_lock import Tasynclooplock


class AsyncLoopLockable(Generic[Tasynclooplock]):
    """
    Generic interface for objects that need to be synchronized in an async loop.

    Args
    ----
    async_loop_lock:
        Lock used to synchronize the component in an async loop
    """

    def __init__(self, async_loop_lock: "Tasynclooplock | None" = None):
        self._async_loop_lock = async_loop_lock

    @property
    def can_loop(self) -> bool:
        return self._async_loop_lock is not None

    async def next_loop(self):
        """
        Wait for the next loop to start

        Example
        -------
        This function is used by the :class:`.Agent` to synchronize the :class:`.Component` in an async loop.
        """
        if self._async_loop_lock is None:
            raise AttributeError(
                "Async loop lock is not set. Please set it before using the component in an async loop."
            )
        await self._async_loop_lock.next_loop()

    async def async_stop(self):
        """
        Release the async loop lock.
        It means that the object will complete the current loop immediately and then stop.

        Example
        -------
        This function is used by the :class:`.Agent` when it stops the async loop.
        """
        assert self._async_loop_lock is not None
        await self._async_loop_lock.release_loop()

    @property
    def async_loop_lock(self) -> "Tasynclooplock":
        """Async loop lock"""
        if self._async_loop_lock is None:
            raise AttributeError(
                "Async loop lock is not set. Please set it before using the component in an async loop."
            )
        return self._async_loop_lock

    @async_loop_lock.setter
    def async_loop_lock(self, value: "Tasynclooplock"):
        self._async_loop_lock = value

    @async_loop_lock.deleter
    def async_loop_lock(self):
        self._async_loop_lock = None
