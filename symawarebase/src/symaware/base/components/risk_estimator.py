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
    ComputingRiskCallback: TypeAlias = Callable[[Agent], Any]
    ComputedRiskCallback: TypeAlias = Callable[[Agent, TimeSeries], Any]


class RiskEstimator(Component[Tasynclooplock, "ComputingRiskCallback", "ComputedRiskCallback"]):
    """
    Generic risk estimator of an :class:`.symaware.base.Agent`.
    It is used to compute the risk of an agent.
    The result of the computation is stored in the :class:`.AwarenessVector` of the agent.

    Args
    ----
    agent_id:
        Identifier of the agent this component belongs to
    async_loop_lock:
        Async loop lock to use for the risk estimator
    """

    @abstractmethod
    def _compute(self) -> TimeSeries:
        """
        Compute the risk for the agent.

        This method must be implemented in any custom risk estimator.

        Example
        -------
        Create a new risk estimator by subclassing the :class:`.RiskEstimator` and implementing the
        :meth:`_compute` method.

        >>> from symaware.base import RiskEstimator, TimeSeries
        >>> class MyRiskEstimator(RiskEstimator):
        ...     def _compute(self):
        ...         # Your implementation here
        ...         # Example:
        ...         # Get the last value of the risk stored in the awareness database
        ...         awareness_database = self._agent.awareness_database
        ...         if len(awareness_database[self._agent_id].risk) == 0:
        ...             return TimeSeries({0: np.array([0])})
        ...         last_value_idx = next(iter(awareness_database[self._agent_id].risk))
        ...         last_value = awareness_database[self._agent_id].risk[last_value_idx]
        ...         # Invert the last value and return it as a TimeSeries
        ...         return TimeSeries({0: np.array([1 - last_value])})
        """
        pass

    def _update(self, risk: TimeSeries):
        """
        Update the agent's model with the new risk time series.

        Example
        -------
        A new risk estimator could decide to override the default :meth:`_update` method.

        >>> from symaware.base import RiskEstimator, TimeSeries
        >>> class MyRiskEstimator(RiskEstimator):
        ...     def _update(self, risk: TimeSeries):
        ...         # Your implementation here
        ...         # Example:
        ...         # Simply override the risk of the agent
        ...         self._agent.self_awareness.risk = risk

        Args
        ----
        risk:
            New risk estimation to apply to store in the agent's awareness vector
        """
        self._agent.self_awareness.risk = risk


class NullRiskEstimator(RiskEstimator[Tasynclooplock], NullObject):
    """
    Default risk estimator used as a placeholder.
    It is used when no risk estimator is set for an agent.
    An exception is raised if this object is used in any way.
    """

    def __init__(self):
        super().__init__(-1)

    def _compute(self):
        pass


class DefaultRiskEstimator(RiskEstimator[Tasynclooplock]):
    """
    Default implementation of risk estimator.
    It returns the risk stored in the awareness database of the agent.

    Args
    ----
    agent_id:
        Identifier of the agent this component belongs to
    """

    def _compute(self) -> TimeSeries:
        return self._agent.self_awareness.risk
