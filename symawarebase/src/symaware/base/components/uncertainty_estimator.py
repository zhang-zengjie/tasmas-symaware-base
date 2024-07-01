from abc import abstractmethod
from typing import TYPE_CHECKING

from symaware.base.data import TimeSeries
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
    ComputingUncertaintyCallback: TypeAlias = Callable[[Agent], Any]
    ComputedUncertaintyCallback: TypeAlias = Callable[[Agent, TimeSeries], Any]


class UncertaintyEstimator(Component[Tasynclooplock, "ComputingUncertaintyCallback", "ComputedUncertaintyCallback"]):
    """
    Generic uncertainty computer of an :class:`.symaware.base.Agent`.
    It is used to compute the uncertainty of an agent.
    The result of the computation is stored in the :class:`.AwarenessVector` of the agent.

    Args
    ----
    agent_id:
        Identifier of the agent this component belongs to
    async_loop_lock:
        Async loop lock to use for the uncertainty estimator
    """

    @abstractmethod
    def _compute(self) -> TimeSeries:
        """
        Compute the uncertainty for the agent

        Example
        -------
        Create a new uncertainty estimator by subclassing the :class:`.UncertaintyEstimator` and implementing the
        :meth:`_compute` method.

        >>> from symaware.base import UncertaintyEstimator, TimeSeries
        >>> class MyUncertaintyEstimator(UncertaintyEstimator):
        ...     def _compute(self):
        ...         # Your implementation here
        ...         # Example:
        ...         # Get the last value of the uncertainty stored in the awareness database
        ...         awareness_database = self._agent.awareness_database
        ...         if len(awareness_database[self._agent_id].uncertainty) == 0:
        ...             return TimeSeries({0: np.array([0])})
        ...         last_value_idx = sorted(awareness_database[self._agent_id].uncertainty)
        ...         last_value = awareness_database[self._agent_id].uncertainty[last_value_idx[-1]]
        ...         # Inver the last value and return it as a TimeSeries
        ...         return TimeSeries({0: np.array([1 - last_value])})

        Returns
        -------
            Uncertainty of the agent
        """
        pass

    def _update(self, uncertainty: TimeSeries):
        """
        Update the uncertainty of the agent in the awareness database.

        Example
        -------

        >>> from symaware.base import UncertaintyEstimator, TimeSeries
        >>> class MyUncertaintyEstimator(UncertaintyEstimator):
        ...     def _update(self, uncertainty: TimeSeries):
        ...         # Your implementation here
        ...         # Example:
        ...         # Simply override the uncertainty of the agent
        ...         self._agent.self_awareness.uncertainty = uncertainty


        Args
        ----
        uncertainty:
            Uncertainty to update in the awareness database
        """
        self._agent.self_awareness.uncertainty = uncertainty


class NullUncertaintyEstimator(UncertaintyEstimator[Tasynclooplock], NullObject):
    """
    Default uncertainty estimator used as a placeholder.
    It is used when no uncertainty estimator is set for an agent.
    An exception is raised if this object is used in any way.
    """

    def __init__(self):
        super().__init__(-1)

    def _compute(self):
        pass


class DefaultUncertaintyEstimator(UncertaintyEstimator[Tasynclooplock]):
    """
    Default implementation of uncertainty estimator.
    It returns the uncertainty stored in the awareness database of the agent.

    Args
    ----
    agent_id:
        Identifier of the agent this component belongs to
    """

    def _compute(self):
        """
        Compute the uncertainty for the agent
        by returning the uncertainty stored in the awareness database of the agent.

        Returns
        -------
            Uncertainty of the agent
        """
        return self._agent.self_awareness.uncertainty
