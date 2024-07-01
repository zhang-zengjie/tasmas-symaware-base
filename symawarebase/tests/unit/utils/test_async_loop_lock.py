# pylint: disable=missing-function-docstring, missing-class-docstring, redefined-outer-name, no-self-use, unused-argument, abstract-class-instantiated, invalid-name, attribute-defined-outside-init, protected-access, unused-variable, unused-import, redefined-outer-name, no-self-use, unused-argument, abstract-class-instantiated, invalid-name, attribute-defined-outside-init, protected-access, unused-variable, unused-import
import asyncio

import pytest

from symaware.base.utils import (
    ConditionAsyncLoopLock,
    EventAsyncLoopLock,
    TimeIntervalAsyncLoopLock,
)


@pytest.mark.asyncio
class TestEventAsyncLoopLock:
    async def test_event_async_loop_lock_init(self):
        lock = EventAsyncLoopLock()
        assert not lock.released
        assert not lock.event.is_set()
        assert lock._loop is not None
        with pytest.raises(asyncio.TimeoutError):
            try:
                await asyncio.wait_for(lock.next_loop(), timeout=0.1)
            except asyncio.CancelledError:
                pass

    async def test_event_async_loop_lock_trigger(self):
        lock = EventAsyncLoopLock()
        lock.trigger()
        assert not lock.released
        assert lock.event.is_set()
        await asyncio.wait_for(lock.next_loop(), timeout=0.1)

    async def test_event_async_loop_lock_acquire(self):
        lock = EventAsyncLoopLock()
        lock.trigger()
        lock.clear()
        assert not lock.released
        assert not lock.event.is_set()
        with pytest.raises(asyncio.TimeoutError):
            try:
                await asyncio.wait_for(lock.next_loop(), timeout=0.1)
            except asyncio.CancelledError:
                pass

    async def test_event_async_loop_lock_release_loop(self):
        lock = EventAsyncLoopLock()
        await lock.release_loop()
        assert lock.released
        assert lock.event.is_set()
        await lock.next_loop()


@pytest.mark.asyncio
class TestTimeIntervalAsyncLoopLock:
    async def test_time_interval_async_loop_lock_init(self):
        lock = TimeIntervalAsyncLoopLock(10)
        assert not lock.released
        assert lock._loop is not None
        assert lock.time_interval == 10
        assert lock.last_timestamp == 0.0
        await asyncio.wait_for(lock.next_loop(), timeout=0.1)
        last_timestamp = lock.last_timestamp
        await asyncio.wait_for(lock.next_loop(), timeout=0.1)
        assert lock.last_timestamp != 0.0
        assert lock.last_timestamp - last_timestamp < 10

    async def test_time_interval_async_loop_lock_next_loop_in_time(self):
        time_interval = 0.05
        lock = TimeIntervalAsyncLoopLock(time_interval)
        assert lock.last_timestamp < lock._loop.time()
        last_timestamp = lock.last_timestamp
        await asyncio.wait_for(lock.next_loop(), timeout=0.1)
        assert lock.last_timestamp != 0.0
        assert lock.last_timestamp - last_timestamp >= time_interval
        last_timestamp = lock.last_timestamp
        await asyncio.wait_for(lock.next_loop(), timeout=0.1)
        assert lock.last_timestamp != 0.0
        assert lock.last_timestamp - last_timestamp >= time_interval

    async def test_time_interval_async_loop_lock_next_loop_slow(self):
        time_interval = 0.2
        lock = TimeIntervalAsyncLoopLock(time_interval)
        assert lock.last_timestamp < lock._loop.time()
        last_timestamp = lock.last_timestamp
        await asyncio.wait_for(lock.next_loop(), timeout=0.1)
        assert lock.last_timestamp != 0.0
        assert lock.last_timestamp - last_timestamp >= time_interval
        await asyncio.sleep(1.5)
        await asyncio.wait_for(lock.next_loop(), timeout=0.1)
        assert lock.last_timestamp != 0.0
        assert lock.last_timestamp - last_timestamp >= time_interval

    async def test_time_interval_async_loop_lock_release_loop(self):
        lock = TimeIntervalAsyncLoopLock(10)
        await asyncio.wait_for(lock.next_loop(), timeout=0.1)
        await lock.release_loop()
        assert lock.released
        await lock.next_loop()


@pytest.mark.asyncio
class TestConditionAsyncLoopLock:
    async def test_condition_async_loop_lock_init(self):
        lock = ConditionAsyncLoopLock()
        assert not lock.released
        assert lock._loop is not None
        assert lock._condition is not None
        assert not lock._condition.locked()
        assert not lock._condition._waiters

    async def test_condition_async_loop_lock_next_loop(self):
        lock = ConditionAsyncLoopLock()
        with pytest.raises(asyncio.TimeoutError):
            try:
                await asyncio.wait_for(lock.next_loop(), timeout=0.1)
            except asyncio.CancelledError:
                pass
        assert not lock.released
        assert not lock._condition.locked()

    async def test_condition_async_loop_lock_trigger(self):
        lock = ConditionAsyncLoopLock()
        lock._loop.call_later(0.05, asyncio.create_task, lock.trigger())
        await asyncio.wait_for(lock.next_loop(), timeout=0.5)
        assert not lock.released
        assert not lock._condition.locked()

    async def test_condition_async_loop_lock_release_loop(self):
        lock = ConditionAsyncLoopLock()
        await lock.release_loop()
        await lock.next_loop()
        assert lock.released
