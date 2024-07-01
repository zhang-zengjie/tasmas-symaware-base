from abc import ABC, abstractmethod
from dataclasses import dataclass, field

from symaware.base.data import Identifier

from .dynamical_model import DynamicalModel, NullDynamicalModel


@dataclass(frozen=True)
class Entity(ABC):
    """
    Generic class that represents an entity in the simulation environment.
    It can be linked to an :class:`Agent`.
    Additionally, can support a dynamical model that determines its behaviour during the simulation,
    based on the control inputs it receives.

    Note
    ----
    The initialise and step method will be called by the simulation environment, and should be implemented
    by the subclasses to define the characteristics of the entity according to the chosen simulator.

    Args
    ----
    id:
        id of the agent this entity is linked to
    model:
        dynamical model that determines the behaviour of this entity, if any
    """

    id: Identifier = field(default=-1)
    model: DynamicalModel = field(default_factory=NullDynamicalModel)

    @abstractmethod
    def initialise(self, *args, **kwargs):
        pass

    def step(self):
        if isinstance(self.model, NullDynamicalModel):
            return
        self.model.step()
