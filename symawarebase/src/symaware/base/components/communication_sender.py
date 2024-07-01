from abc import abstractmethod
from typing import TYPE_CHECKING, Any, Iterable

from symaware.base.data import (
    Identifier,
    Message,
    MultiAgentAwarenessVector,
    MultiAgentKnowledgeDatabase,
)
from symaware.base.utils import NullObject, Tasynclooplock

from .component import Component

if TYPE_CHECKING:
    import sys
    from typing import Callable

    from symaware.base.agent import Agent

    if sys.version_info >= (3, 10):
        from typing import TypeAlias
    else:
        from typing_extensions import TypeAlias

    SendingCommunicationCallback: TypeAlias = Callable[
        [
            Agent,
            MultiAgentAwarenessVector,
            MultiAgentKnowledgeDatabase,
            "Identifier | Iterable[Identifier] | None",
        ],
        Any,
    ]
    SentCommunicationCallback: TypeAlias = Callable[[Agent, Iterable[Message]], Any]


class CommunicationSender(Component[Tasynclooplock, "SendingCommunicationCallback", "SentCommunicationCallback"]):
    """
    Generic communication system of an :class:`.symaware.base.Agent`.
    It is used to communicate with other agents.
    The information collected is then used to update the knowledge database of the agent.

    Before sending a message, it will be enqueued in the message queue.
    On the next iteration, the messages will be sent to the receiver agent(s) through the communication channel.

    Args
    ----
    agent_id:
        Identifier of the agent this component belongs to
    async_loop_lock:
        Async loop lock to use for the communication sender
    send_to_self:
        If True, the agent is able to send messages to itself
    """

    def __init__(
        self,
        agent_id: Identifier,
        async_loop_lock: "Tasynclooplock | None" = None,
        send_to_self: bool = False,
    ):
        super().__init__(agent_id, async_loop_lock)
        self._send_to_self = send_to_self
        self._message_queue: list[Message] = []

    def enqueue_messages(self, *messages: Message):
        """
        Enqueue a message to be sent to another agent.

        Args
        ----
        message:
            Message to send to the receiver agent
        """
        self._message_queue.extend(messages)

    def dequeue_message(self, message: Message):
        """
        Dequeue a message from the message queue.

        Args
        ----
        message:
            Message to remove from the message queue
        """
        self._message_queue.remove(message)

    @abstractmethod
    def _send_communication_through_channel(self, message: Message):
        """
        The information the agents wants to share has been encoded into a message and it sent to another agent
        using a communication channel.

        Example
        -------
        Create a new communication system by subclassing the :class:`.CommunicationSender` and implementing the
        :meth:`_send_communication_through_channel` method.

        >>> from symaware.base import CommunicationSender, Message
        >>> class MyCommunicationSender(CommunicationSender):
        ...     def _send_communication_through_channel(self, message: Message):
        ...         # Your implementation here
        ...         # Example:
        ...         # Print the message to the console
        ...         print(f"Sending message to {message.receiver_id}: {message}")

        Args
        ----
        message:
            Message to send to the receiver agent through the communication channel
        """
        pass

    async def _async_send_communication_through_channel(self, message: Message):
        """
        The information the agents wants to share has been encoded into a message and it sent to another agent
        using a communication channel asynchronously.

        Note
        ----
        Check :meth:`_send_communication_through_channel` for more information about the method
        and :class:`.AsyncLoopLockable` for more information about the async loop.

        Args
        ----
        message:
            Message to send to the receiver agent through the communication channel
        """
        return self._send_communication_through_channel(message)

    def _compute(self) -> Iterable[Message]:
        """
        All the messages in the message queue are sent to the receiver agent(s) through the communication channel.
        Some implementations may put a limit on the number of messages sent per iteration,
        to avoid flooding the channel.

        Returns
        -------
            Message(s) sent to the receiver agent(s)
        """
        message_sent: list[Message] = []
        while len(self._message_queue) > 0:
            message = self._message_queue.pop(0)
            if message.receiver_id == self._agent_id and not self._send_to_self:
                continue
            self._send_communication_through_channel(message)
            message_sent.append(message)
        return message_sent

    async def _async_compute(self) -> Iterable[Message]:
        """
        All the messages in the message queue are sent to the receiver agent(s)
        through the communication channel asynchronously.

        Note
        ----
        Check :meth:`_compute` for more information about the method
        and :class:`.AsyncLoopLockable` for more information about the async loop.

        Returns:
            Message(s) sent to the receiver agent(s)
        """
        message_sent: list[Message] = []
        while len(self._message_queue) > 0:
            message = self._message_queue.pop(0)
            if message.receiver_id == self._agent_id and not self._send_to_self:
                continue
            await self._async_send_communication_through_channel(message)
            message_sent.append(message)
        return message_sent

    def _update(self, messages_sent: Iterable[Message]):
        pass


class NullCommunicationSender(CommunicationSender, NullObject):
    """
    Default communication system used as a placeholder.
    It is used when no communication system is set for an agent.
    An exception is raised if this object is used in any way.
    """

    def __init__(self):
        super().__init__(-1)

    def _send_communication_through_channel(self, message: Message):
        pass


class DefaultCommunicationSender(CommunicationSender[Tasynclooplock]):
    """
    Default implementation of the info updater.
    It does not send or receive any message.

    Args
    ----
    agent_id:
        Identifier of the agent this component belongs to
    """

    def _send_communication_through_channel(self, message: Message):
        pass
