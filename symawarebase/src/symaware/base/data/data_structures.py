import sys
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, TypedDict, TypeVar

import numpy as np

from .types import Identifier

if TYPE_CHECKING:
    # String type hinting to support python 3.9

    if sys.version_info >= (3, 10):
        from typing import TypeAlias
    else:
        from typing_extensions import TypeAlias


@dataclass(frozen=True)
class Observation:
    """help class for an observation

    Args
    ----
    observed_object_id:
        id of the agent that is the subject of the observation
    """

    observed_object_id: Identifier


@dataclass(frozen=True)
class StateObservation(Observation):
    """help class for an observation

    Args
    ----
    observed_object_id:
        id of the agent that is the subject of the observation
    state:
        state of the observed agent
    """

    state: np.ndarray


@dataclass(frozen=True)
class Message:
    """
    Simple class to store messages between agents.

    Args
    ----
    sender_id:
        id of the sender agent
    receiver_id:
        id of the receiver agent
    """

    sender_id: Identifier
    receiver_id: Identifier


@dataclass(frozen=True)
class InfoMessage(Message):
    """
    Simple class to store messages between agents that share their awareness vector.

    Args
    ----
    sender_id:
        id of the sender agent
    receiver_id:
        id of the receiver agent
    awareness_vector:
        awareness_vector communicated by the sender to the receiver
    """

    awareness_database: "MultiAgentAwarenessVector"
    knowledge_database: "MultiAgentKnowledgeDatabase"


class TimeSeries(dict):
    """
    Simple call to store timed numpy array.
    It is a dictionary that you can look up for the current time and get the corresponding numpy array.

    Note
    ----
    Consistency of the array is not check for now

    Args
    ----
    iterable:
        dictionary used to initialize the instance
    """

    def __init__(self, iterable: "dict | None" = None):
        if iterable is None:
            iterable = {}
        for key, item in iterable.items():
            if not isinstance(item, np.ndarray):
                raise TypeError(f"Expected numpy array, got {type(item)}")

            if not isinstance(key, (float, int)):
                raise TypeError(f"Expected float/int as a key, got {type(key)}")

        super().__init__(iterable)

    def __setitem__(self, key: "float | int", item: np.ndarray):
        if not isinstance(item, np.ndarray):
            raise TypeError(f"Expected numpy array, got {type(item)}")
        if not isinstance(key, (float, int)):
            raise TypeError(f"Expected float/int as a key, got {type(key)}")

        super().__setitem__(key, item)

    def __getitem__(self, key: "float | int") -> np.ndarray:
        try:
            return super().__getitem__(key)
        except Exception as e:
            raise KeyError(
                f"The requested time {key} is not available in the time series. "
                f"Available name are are {list(self.__dict__.keys())}"
            ) from e

    def __eq__(self, value: Any) -> bool:
        if not isinstance(value, TimeSeries):
            raise TypeError(f"Expected {self.__class__.__name__}, received {type(value)}")
        if len(self) != len(value):
            return False
        for (k, v), (k2, v2) in zip(self.items(), value.items()):
            if k != k2:
                return False
            if not np.array_equal(v, v2):
                return False
        return True

    def __ne__(self, value: object) -> bool:
        return not self.__eq__(value)

    def __or__(self, value: Any) -> "TimeSeries":
        if not isinstance(value, TimeSeries):
            raise TypeError(f"Expected TimeSeries, got {type(value)}")
        out = TimeSeries(self)
        for key, item in value.items():
            out[key] = item
        return out


class KnowledgeDatabase(TypedDict):
    """
    Simple dictionary to store knowledge about the system.
    It is highly recommended to use TypedDict to enforce the structure of the knowledge database.
    """


class AwarenessVector:
    """
    Class used to store state,id,intent,discrete modes of an agent and its neighbours.

    Note
    ----
    An agent has basically an opinion on its awareness vector and its neighbours state

    Args
    ----
    ID:
        id of the agent
    current_state:
        current continuous state of the agent expressed as a numpy array
    intent:
        intent vector. Should be updated by the controller
    risk:
        risk evaluation vector. Should be updated by the risk estimator
    uncertainty:
        uncertainty evaluation vector. Should be updated by the uncertainty estimator
    """

    def __init__(
        self,
        ID: Identifier,
        current_state: np.ndarray,
        intent: TimeSeries = TimeSeries(),
        risk: TimeSeries = TimeSeries(),
        uncertainty: TimeSeries = TimeSeries(),
    ):
        if not isinstance(ID, Identifier):
            raise TypeError(f"Expected integer ID, got {type(ID)}")
        if not isinstance(current_state, np.ndarray):
            raise TypeError(f"Expected numpy array, got {type(current_state)}")
        if not isinstance(intent, TimeSeries):
            raise TypeError(f"Expected TimeSeries, got {type(intent)}")
        if not isinstance(risk, TimeSeries):
            raise TypeError(f"Expected TimeSeries, got {type(risk)}")
        if not isinstance(uncertainty, TimeSeries):
            raise TypeError(f"Expected TimeSeries, got {type(uncertainty)}")

        self._state = current_state  # intended as current continuous state
        self._ID = ID
        self._intent = intent
        self._risk = risk
        self._uncertainty = uncertainty
        self._state_dim = len(current_state)

    @property
    def state(self) -> np.ndarray:
        return self._state

    @state.setter
    def state(self, value: np.ndarray):
        if not isinstance(value, np.ndarray):
            raise TypeError(f"Expected numpy array, got {type(value)}")
        if not len(value) == self._state_dim:
            raise ValueError(f"Expected numpy array of length {self._state_dim}, got {len(value)}")
        self._state = value

    @property
    def state_dim(self) -> int:
        return self._state_dim

    @property
    def ID(self) -> int:
        return self._ID

    @property
    def intent(self) -> TimeSeries:
        return self._intent

    @intent.setter
    def intent(self, value: TimeSeries):
        if not isinstance(value, TimeSeries):
            raise TypeError(f"Expected TimeSeries, got {type(value)}")
        self._intent = value

    @property
    def risk(self) -> TimeSeries:
        return self._risk

    @risk.setter
    def risk(self, value: TimeSeries):
        if not isinstance(value, TimeSeries):
            raise TypeError(f"Expected TimeSeries, got {type(value)}")
        self._risk = value

    @property
    def uncertainty(self) -> TimeSeries:
        return self._uncertainty

    @uncertainty.setter
    def uncertainty(self, value: TimeSeries):
        if not isinstance(value, TimeSeries):
            raise TypeError(f"Expected TimeSeries, got {type(value)}")
        self._uncertainty = value

    def copy(self) -> "AwarenessVector":
        return AwarenessVector(
            ID=self.ID,
            current_state=self.state,
            intent=self.intent,
            risk=self.risk,
            uncertainty=self.uncertainty,
        )

    def __or__(self, value: Any) -> "AwarenessVector":
        if not isinstance(value, AwarenessVector):
            raise TypeError(f"Expected AwarenessVector, got {type(value)}")
        if not self._ID == value.ID:
            raise ValueError(f"Expected same ID, got {self._ID} and {value.ID}")
        return AwarenessVector(
            ID=self.ID,
            current_state=value.state,
            intent=self.intent | value.intent,
            risk=self.risk | value.risk,
            uncertainty=self.uncertainty | value.uncertainty,
        )

    def __eq__(self, value: Any) -> bool:
        if not isinstance(value, AwarenessVector):
            raise TypeError(f"Expected {self.__class__.__name__}, received {type(value)}")
        return (
            self.ID == value.ID
            and np.array_equal(self.state, value.state)
            and self.intent == value.intent
            and self.risk == value.risk
            and self.uncertainty == value.uncertainty
        )

    def __str__(self) -> str:
        return (
            f"Agent ID: {self.ID}\n"
            f"Current State: {self.state}\n"
            f"Intent: {self.intent}\n"
            f"Risk: {self.risk}\n"
            f"Uncertainty: {self.uncertainty}\n"
        )


T = TypeVar("T", bound=KnowledgeDatabase)
MultiAgentAwarenessVector: "TypeAlias" = dict[Identifier, AwarenessVector]
MultiAgentKnowledgeDatabase: "TypeAlias" = dict[Identifier, T]
