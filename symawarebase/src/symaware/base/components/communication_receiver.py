from abc import abstractmethod
from typing import TYPE_CHECKING, Any, Iterable

from symaware.base.data import (
    Identifier,
    InfoMessage,
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

    ReceivingCommunicationCallback: TypeAlias = Callable[[Agent], Any]
    ReceivedCommunicationCallback: TypeAlias = Callable[
        [Agent, dict[Identifier, tuple[MultiAgentAwarenessVector, MultiAgentKnowledgeDatabase]]], Any
    ]


class CommunicationReceiver(
    Component[Tasynclooplock, "ReceivingCommunicationCallback", "ReceivedCommunicationCallback"]
):
    """
    Generic communication system of an :class:`.symaware.base.Agent`.
    It is used to communicate with other agents.
    The information collected is then used to update the knowledge database of the agent.

    The internal communication channel is read and any number of messages collected are returned.
    The messages are then translated into a dictionary of :class:`.AwarenessVector` and :class:`.KnowledgeDatabase`
    ready to be used to update the agent's state.

    To implement a new communication system, you need to indicate how the new messages update the agent by implementing
    the :meth:`_update` method.

    Example
    -------
    Create a new communication system by subclassing the :class:`.CommunicationReceiver` and implementing the
    :meth:`_update` method.

    >>> from symaware.base import CommunicationReceiver, Message
    >>> class MyCommunicationReceiver(CommunicationReceiver):
    ...     def _update(self, *args, **kwargs):
    ...         # Your implementation here
    ...         # Example:
    ...         # Update the awareness database with the messages received
    ...         for sender_id, (awareness_vector, knowledge_database) in kwargs.items():
    ...             self._agent.awareness_database[sender_id] = awareness_vector
    ...             self._agent.knowledge_database[sender_id] = knowledge_database

    Args
    ----
    agent_id:
        Identifier of the agent this component belongs to
    async_send_loop_lock:
        Async loop lock to use for the communication system
    """

    @abstractmethod
    def _decode_message(
        self, messages: Iterable[Message]
    ) -> dict[Identifier, tuple[MultiAgentAwarenessVector, MultiAgentKnowledgeDatabase]]:
        """
        The messages received from the communication channel are decoded to produce the output
        of the :meth:`_compute` method.
        The default implementation assumes that the messages are of type :class:`.InfoMessage`,
        and produces a dictionary of Awareness vector and Knowledge database.

        Although the method must be implemented, you may choose to sure the super implementation if the messages
        are of type :class:`.InfoMessage`.

        Example
        -------
        To support a new message type, you need to implement the :meth:`_decode_message` method.

        >>> from typing import Iterable
        >>> from symaware.base import CommunicationSender, Message, InfoMessage, Identifier
        >>> class MyCommunicationReceiver(CommunicationReceiver):
        ...     def _decode_message(self, messages: Iterable[Message]) -> dict[Identifier, str]:
        ...         # Your implementation here
        ...         # Example:
        ...         # Decode a message of type InfoMessage
        ...         res = {}
        ...         for message in messages:
        ...             if isinstance(message, InfoMessage):
        ...                 res[message.sender_id] = (message.awareness_database, message.knowledge_database)
        ...             else:
        ...                 raise TypeError(f"Unknown message type {type(message)}")
        ...         return res

        Args
        ----
        messages:
            Iterable of messages to parse

        Returns
        -------
            Dictionary of Awareness vector and Knowledge database.
            The key is the sender id and the value is a tuple of Awareness vector and Knowledge database
        """
        res = {}
        for message in messages:
            if isinstance(message, InfoMessage):
                res[message.sender_id] = (message.awareness_database, message.knowledge_database)
            else:
                raise TypeError(f"Unknown message type {type(message)}")
        return res

    @abstractmethod
    def _receive_communication_from_channel(self) -> Iterable[Message]:
        """
        The internal communication channel is read and any number of messages collected are returned.

        Example
        -------
        Create a new communication system by subclassing the :class:`.CommunicationReceiver` and implementing the
        :meth:`_receive_communication_from_channel` method.

        >>> from typing import Iterable
        >>> from symaware.base import CommunicationReceiver, Message
        >>> class MyCommunicationReceiver(CommunicationReceiver):
        ...     def _receive_communication_from_channel(self) -> Iterable[Message]:
        ...         # Your implementation here
        ...         # Example:
        ...         # Return an empty list. No messages are ever received
        ...         return []

        Returns
        -------
            Messages received by another agent
        """
        pass

    async def _async_receive_communication_from_channel(self) -> Iterable[Message]:
        """
        The internal communication channel is read and any number of messages collected are returned asynchronously.

        Note
        ----
        Check :meth:`_receive_communication_from_channel` for more information about the method
        and :class:`.AsyncLoopLockable` for more information about the async loop.

        Returns:
            Iterable[Message]: _description_
        """
        return self._receive_communication_from_channel()

    def _compute(self) -> "dict[Identifier, tuple[MultiAgentAwarenessVector, MultiAgentKnowledgeDatabase]]":
        """
        Upon receiving a message, it is decoded and translated into useful information.
        The simplest one would be returning directly what you receive.


        Returns
        -------
            Messages received by other agents as a dictionary of Awareness vector and Knowledge database
        """
        messages = self._receive_communication_from_channel()
        return self._decode_message(messages)

    async def _async_compute(self) -> "dict[Identifier, tuple[MultiAgentAwarenessVector, MultiAgentKnowledgeDatabase]]":
        """
        Upon receiving a message, it is decoded and translated into useful information asynchronously.

        Note
        ----
        Check :meth:`_compute` for more information about the method
        and :class:`.AsyncLoopLockable` for more information about the async loop.

        Returns:
            Messages received by other agents as a dictionary of Awareness vector and Knowledge database
        """
        messages = await self._async_receive_communication_from_channel()
        return self._decode_message(messages)


class NullCommunicationReceiver(CommunicationReceiver[Tasynclooplock], NullObject):
    """
    Default communication system used as a placeholder.
    It is used when no communication system is set for an agent.
    An exception is raised if this object is used in any way.
    """

    def __init__(self):
        super().__init__(-1)

    def _decode_message(self, messages: Iterable[Message]):
        pass

    def _receive_communication_from_channel(self):
        pass

    def _update(self):
        pass


class DefaultCommunicationReceiver(CommunicationReceiver[Tasynclooplock]):
    """
    Default implementation of the info updater.
    It does not send or receive any message.

    Args
    ----
    agent_id:
        Identifier of the agent this component belongs to
    """

    def _decode_message(self, messages: Iterable[Message]) -> dict:
        return {}

    def _receive_communication_from_channel(self) -> Iterable[Message]:
        return []

    def _update(self):
        pass
