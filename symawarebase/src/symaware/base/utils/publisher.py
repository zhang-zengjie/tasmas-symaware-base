from abc import ABC
from typing import Any, Callable


class Publisher(ABC):
    """
    The Publisher interface declares a set of methods for managing subscribers (observers) and notifying them of
    state changes, implementing the Publisher-Subscriber pattern.
    The publisher will invoke the callback provided by the subscriber when a determinate event occurs.

    Other Packages can use this class to get notified of events, exchanging information and triggering actions.

    Attributes
    ----------
    _callbacks:
        A dictionary of set of callbacks added to this publisher.
        The key for each set is the identifier of the event the callback is interested in.
        Note that each set does not allow duplicates.
    """

    def __init__(self):
        self._callbacks: dict[str, set[Callable]] = {}

    def add(self, event: str, callback: Callable):
        """
        Add a callback to the publisher.
        From now on, every time the publisher's state changes, it will invoke the callback.
        Keep in mind that duplicate callbacks are not allowed, so calling this method multiple times with the same
        one will not add it multiple times.
        This said, it is easy to create multiple semantically equals callbacks, for example by using lambda functions.

        Warning
        -------
        Using the :meth:`add` method directly is discouraged, for no static type checking is performed.
        It is recommended to use the specific method provided by the concrete subclass of Publisher.

        Example
        -------
        Adding callbacks to the publisher and notifying them of an event.
        Only the callbacks added to that specific event will be invoked.

        >>> from symaware.base.utils import Publisher
        >>> publisher = Publisher() # concrete subclass of Publisher
        >>> publisher.add("event", lambda: print("Hello World!"))
        >>> publisher.add("another_event", lambda: print("OK"))
        >>> publisher._notify("event")
        Hello World!

        Args
        ----
        event:
            Identifier of the event the callback will be invoked for.
        callback:
            The callback to attach. It is a callable object that will be invoked when the publisher's state changes.
        """
        self._add(event, callback)

    def remove(self, event: str, callback: Callable):
        """
        Remove a callback from the publisher.
        It won't be invoked anymore when a notification is sent.

        Warning
        -------
        Using the :meth:`remove` method directly is discouraged, for no static type checking is performed.
        It is recommended to use the specific method provided by the concrete subclass of Publisher.

        Example
        -------
        Removing a callbacks from the publisher and notifying the remaining ones of an event.

        >>> from symaware.base.utils import Publisher
        >>> publisher = Publisher() # concrete subclass of Publisher
        >>> fun = lambda: print("Hello World!")
        >>> publisher.add("event", fun)
        >>> publisher.add("event", lambda: print("Hello"))
        >>> publisher.remove("event", fun)
        >>> publisher._notify("event")
        Hello

        Args
        ----
        event:
            Identifier of the event the callback was attached to.
        callback:
            The callback to remove.
        """
        self._remove(event, callback)

    def _add(self, event: str, callback: Callable):
        """
        Add a callback to the internal dictionary of callbacks.

        Args
        ----
        event:
            Identifier of the event the callback will be invoked for
        callback:
            The callback to attach. It is a callable object that will be invoked when the publisher's state changes
        """
        self._callbacks.setdefault(event, set())
        self._callbacks[event].add(callback)

    def _remove(self, event: str, callback: Callable):
        """
        Remove a callback from the internal dictionary of callbacks.

        Args
        ----
        event:
            Identifier of the event the callback was attached to
        callback:
            The callback to remove
        """
        if event in self._callbacks and callback in self._callbacks[event]:
            self._callbacks[event].remove(callback)

    def _notify(self, event: str, *data: Any):
        """Notify the subscribers of the event.

        Example
        -------
        Notifying the subscribers of an event will invoke the callbacks attached to that event.

        >>> from symaware.base.utils import Publisher
        >>> phrases = []
        >>> publisher = Publisher() # concrete subclass of Publisher
        >>> publisher.add("event", lambda: phrases.append("Hello World!"))
        >>> publisher.add("event", lambda: phrases.append("Hello"))
        >>> publisher.add("another_event", lambda: phrases.append("OK"))
        >>> publisher._notify("event")
        >>> print(" - ".join(sorted(phrases)))
        Hello - Hello World!

        Args
        ----
        event:
            event identifier
        data:
            data to invoke the callbacks with
        """
        if event in self._callbacks:
            for callback in self._callbacks[event]:
                callback(*data)
