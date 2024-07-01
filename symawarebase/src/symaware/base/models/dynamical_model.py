from abc import ABC, abstractmethod
from typing import TypedDict, cast

import numpy as np

from symaware.base.data import Identifier
from symaware.base.utils import NullObject


class DynamicalModelSubinputs(TypedDict):
    pass


class DynamicalModel(ABC):
    """
    Abstract class for dynamical models that will simulate the behaviour of an entity.
    At each step of the simulation, the model is fed the control input from the :class:`.Controller`.

    Args
    ----
    ID:
        Identifier of the agent this model belongs to.
        A value of -1 indicates that no agent is associated with this model.
    control_input:
        Initial control input of the model. Is is also used to check the shape of future control inputs
    """

    def __init__(self, ID: Identifier, control_input: np.ndarray):
        self._ID = ID
        self._control_input = control_input

    @property
    def id(self) -> Identifier:
        """
        Identifier of the agent this model belongs to.
        A value of -1 indicates that no agent is associated with this model.
        """
        return self._ID

    @property
    def control_input(self) -> np.ndarray:
        """Last control input received by the model."""
        return self._control_input

    @control_input.setter
    def control_input(self, value: np.ndarray):
        if not isinstance(value, np.ndarray):
            raise TypeError(f"Expected numpy array, got {type(value)}")
        if not value.shape == self.control_input_shape:
            raise ValueError(f"Expected numpy array of shape {self.control_input_shape}, got {value.shape}")
        self._control_input = value

    @abstractmethod
    def initialise(self, *args, **kwargs):
        """
        Initialise the dynamical model.
        Called by the associated entity when it is added to the environment.

        Args
        ----
        args:
            Additional arguments
        kwargs:
            Additional keyword arguments
        """
        pass

    @abstractmethod
    def step(self, *args, **kwargs) -> np.ndarray:
        """
        Run a simulation step of the dynamical model.
        Called by the associated entity at each simulation step.

        Args
        ----
        args:
            Additional arguments
        kwargs:
            Additional keyword arguments
        """
        pass

    @property
    def control_input_size(self) -> int:
        """Size of the control input vector."""
        return self.control_input.size

    @property
    def control_input_shape(self) -> tuple[int, ...]:
        """Shape of the control input vector."""
        return self.control_input.shape

    @property
    @abstractmethod
    def subinputs_dict(self) -> DynamicalModelSubinputs:
        """
        The input of a system is the composition of subinputs.

        Example
        -------
        A car input [vx, vy, vz, s] is composed by the velocity and the steering angle.

        >>> from symaware.base import DynamicalModel
        >>> class CarModel(DynamicalModel):
        ...     @property
        ...     def subinputs_dict(self):
        ...         return {
        ...             "velocity": self._control_input[:3],
        ...             "steering_angle": self._control_input[3]
        ...         }
        ...

        Important
        ---------
        The order of the subinputs in the list must be the same as the order of the subinputs in the input vector
        """
        pass

    def __str__(self) -> str:
        sub_inputs = "\n".join(
            f"key  : {key}\nsize : {cast(np.ndarray, subinput).size}" for key, subinput in self.subinputs_dict.items()
        )
        return (
            f"{self.__class__.__name__} ({self._ID})(\n"
            f"control input size: {self.control_input_size}\n"
            "List of subinputs (subinputs are stacked in the given order to give the input vector):\n"
            f"{sub_inputs}\n)"
        )


class NullDynamicalModel(DynamicalModel, NullObject):
    """
    Default dynamical model used as a placeholder.
    It is used when no dynamical model has been defined for an entity.
    The entity will just ignore it and won't have any step function associated with it.
    An exception is raised if this object is used in any way.
    """

    def __init__(self):
        super().__init__(-1, np.zeros(0))

    def initialise(self):
        pass

    def step(self):
        pass

    @property
    def subinputs_dict(self):
        pass
