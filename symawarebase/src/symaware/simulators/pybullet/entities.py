from abc import abstractmethod
from dataclasses import dataclass, field

import numpy as np
import pybullet as p
from symaware.base import Entity as BaseEntity
from symaware.base import NullDynamicalModel

from .urdf import URDF
from .dynamical_model import DynamicalModel


@dataclass(frozen=True)
class Entity(BaseEntity):
    """
    Abstract class for the entities using the PyBullet physics engine.
    All the internal identifiers are set to -1 by default, and will be set
    to the correct values during the initialisation.

    Args
    ----
    id:
        Identifier of the entity, if linked to an :class:`Agent`
    mass:
        Mass of the entity in the physics simulation
    position:
        (3)-shaped initial position of the entity. If the size is less than 3, the missing values are set to 0
    velocity:
        (3)-shaped initial velocity of the entity. If the size is less than 3, the missing values are set to 0
    orientation:
        (3,4)-shaped initial orientation of the entity.
        A 3-sized array is interpreted as Euler angles and converted to a 4-sized quaternion array
    model:
        Dynamical model associated with the entity. Must be a subclass of :class:`.PybulletDynamicalModel`
    """

    mass: float = field(default=1)
    position: np.ndarray = field(default_factory=lambda: np.zeros(3))
    velocity: np.ndarray = field(default_factory=lambda: np.zeros(3))
    orientation: np.ndarray = field(default_factory=lambda: np.array([0, 0, 0, 1]))
    model: "DynamicalModel" = field(default_factory=NullDynamicalModel)

    _entity_id: int = field(default=-1)
    _col_id: int = field(default=-1)
    _vis_id: int = field(default=-1)
    _bod_id: int = field(default=-1)

    @property
    def entity_id(self):
        return self._entity_id

    def __post_init__(self):
        object.__setattr__(
            self, "position", np.pad(self.position[:3], (0, max(0, 3 - len(self.position))), mode="constant")
        )
        object.__setattr__(
            self, "velocity", np.pad(self.velocity[:3], (0, max(0, 3 - len(self.velocity))), mode="constant")
        )
        if len(self.orientation) == 3:
            object.__setattr__(self, "orientation", np.array(p.getQuaternionFromEuler(self.orientation)))
        elif len(self.orientation) == 4:
            object.__setattr__(self, "orientation", np.array(self.orientation))
        else:
            raise ValueError(f"Invalid orientation {self.orientation}")

    @abstractmethod
    def initialise(self):
        object.__setattr__(
            self, "_bod_id", p.createMultiBody(self.mass, self._col_id, self._vis_id, self.position, self.orientation)
        )

    def __hash__(self) -> int:
        return hash((super().__hash__(), self._entity_id, self._vis_id, self._bod_id, self._col_id))


@dataclass(frozen=True)
class URDFEntity(Entity):
    """
    Generic entity loaded from a URDF file

    Args
    ----
    id:
        Identifier of the entity, if linked to an :class:`Agent`
    mass:
        Mass of the entity in the physics simulation
    position:
        (3)-shaped initial position of the entity. If the size is less than 3, the missing values are set to 0
    velocity:
        (3)-shaped initial velocity of the entity. If the size is less than 3, the missing values are set to 0
    orientation:
        (3,4)-shaped initial orientation of the entity.
        A 3-sized array is interpreted as Euler angles and converted to a 4-sized quaternion array
    model:
        Dynamical model associated with the entity. Must be a subclass of :class:`.PybulletDynamicalModel`
    urdf_path:
        Path to the URDF file containing the definition of the entity
    """

    urdf_path: str = field(default="")
    init_flags: int = field(default=0)

    def initialise(self):
        if self.urdf_path == "":
            raise ValueError("URDF path not set")
        object.__setattr__(
            self, "_entity_id", p.loadURDF(self.urdf_path, self.position, self.orientation, flags=self.init_flags)
        )
        if self._entity_id < 0:
            raise RuntimeError(f"Failed to initialise {self.__class__.__name__}: negative id = {self._entity_id}")
        if not isinstance(self.model, NullDynamicalModel):
            self.model.initialise(self._entity_id)

    def __hash__(self) -> int:  # pylint: disable=useless-parent-delegation
        return super().__hash__()


@dataclass(frozen=True)
class RacecarEntity(URDFEntity):
    """
    Racecar entity

    Args
    ----
    id:
        Identifier of the entity, if linked to an :class:`Agent`
    mass:
        Mass of the entity in the physics simulation
    position:
        (3)-shaped initial position of the entity. If the size is less than 3, the missing values are set to 0
    velocity:
        (3)-shaped initial velocity of the entity. If the size is less than 3, the missing values are set to 0
    orientation:
        (3,4)-shaped initial orientation of the entity.
        A 3-sized array is interpreted as Euler angles and converted to a 4-sized quaternion array
    model:
        Dynamical model associated with the entity. Must be a subclass of :class:`.PybulletDynamicalModel`
    """

    urdf_path: str = field(default=URDF.RACECAR.urdf)

    def __hash__(self) -> int:  # pylint: disable=useless-parent-delegation
        return super().__hash__()


@dataclass(frozen=True)
class DroneCf2pEntity(URDFEntity):
    """
    Drone entity

    Args
    ----
    id:
        Identifier of the entity, if linked to an :class:`Agent`
    mass:
        Mass of the entity in the physics simulation
    position:
        (3)-shaped initial position of the entity. If the size is less than 3, the missing values are set to 0
    velocity:
        (3)-shaped initial velocity of the entity. If the size is less than 3, the missing values are set to 0
    orientation:
        (3,4)-shaped initial orientation of the entity.
        A 3-sized array is interpreted as Euler angles and converted to a 4-sized quaternion array
    model:
        Dynamical model associated with the entity. Must be a subclass of :class:`.PybulletDynamicalModel`
    """

    urdf_path: str = field(default=URDF.DRONE_CF2P.urdf)

    def __hash__(self) -> int:  # pylint: disable=useless-parent-delegation
        return super().__hash__()


@dataclass(frozen=True)
class DroneCf2xEntity(URDFEntity):
    """
    Drone entity

    Args
    ----
    id:
        Identifier of the entity, if linked to an :class:`Agent`
    mass:
        Mass of the entity in the physics simulation
    position:
        (3)-shaped initial position of the entity. If the size is less than 3, the missing values are set to 0
    velocity:
        (3)-shaped initial velocity of the entity. If the size is less than 3, the missing values are set to 0
    orientation:
        (3,4)-shaped initial orientation of the entity.
        A 3-sized array is interpreted as Euler angles and converted to a 4-sized quaternion array
    model:
        Dynamical model associated with the entity. Must be a subclass of :class:`.PybulletDynamicalModel`
    """

    urdf_path: str = field(default=URDF.DRONE_CF2X.urdf)

    def __hash__(self) -> int:  # pylint: disable=useless-parent-delegation
        return super().__hash__()


@dataclass(frozen=True)
class DroneRacerEntity(URDFEntity):
    """
    Drone entity

    Args
    ----
    id:
        Identifier of the entity, if linked to an :class:`Agent`
    mass:
        Mass of the entity in the physics simulation
    position:
        (3)-shaped initial position of the entity. If the size is less than 3, the missing values are set to 0
    velocity:
        (3)-shaped initial velocity of the entity. If the size is less than 3, the missing values are set to 0
    orientation:
        (3,4)-shaped initial orientation of the entity.
        A 3-sized array is interpreted as Euler angles and converted to a 4-sized quaternion array
    model:
        Dynamical model associated with the entity. Must be a subclass of :class:`.PybulletDynamicalModel`
    """

    urdf_path: str = field(default=URDF.DRONE_RACER.urdf)

    def __hash__(self) -> int:  # pylint: disable=useless-parent-delegation
        return super().__hash__()


@dataclass(frozen=True)
class SphereEntity(Entity):
    """
    Sphere entity

    Args
    ----
    id:
        Identifier of the entity, if linked to an :class:`Agent`
    mass:
        Mass of the entity in the physics simulation
    position:
        (3)-shaped initial position of the entity. If the size is less than 3, the missing values are set to 0
    velocity:
        (3)-shaped initial velocity of the entity. If the size is less than 3, the missing values are set to 0
    orientation:
        (3,4)-shaped initial orientation of the entity.
        A 3-sized array is interpreted as Euler angles and converted to a 4-sized quaternion array
    model:
        Dynamical model associated with the entity. Must be a subclass of :class:`.PybulletDynamicalModel`
    radius:
        Radius of the sphere
    """

    radius: float = field(default=1.0)

    def initialise(self):
        object.__setattr__(self, "_col_id", p.createCollisionShape(p.GEOM_SPHERE, radius=self.radius))
        object.__setattr__(self, "_vis_id", p.createVisualShape(p.GEOM_SPHERE, radius=self.radius))
        super().initialise()

    def __hash__(self) -> int:  # pylint: disable=useless-parent-delegation
        return super().__hash__()


@dataclass(frozen=True)
class BoxEntity(Entity):
    """
    Box entity

    Args
    ----
    id:
        Identifier of the entity, if linked to an :class:`Agent`
    mass:
        Mass of the entity in the physics simulation
    position:
        (3)-shaped initial position of the entity. If the size is less than 3, the missing values are set to 0
    velocity:
        (3)-shaped initial velocity of the entity. If the size is less than 3, the missing values are set to 0
    orientation:
        (3,4)-shaped initial orientation of the entity.
        A 3-sized array is interpreted as Euler angles and converted to a 4-sized quaternion array
    model:
        Dynamical model associated with the entity. Must be a subclass of :class:`.PybulletDynamicalModel`
    sizes:
        Size of the box
    """

    sizes: np.ndarray = field(default_factory=lambda: np.ones(3))

    def __post_init__(self):
        super().__post_init__()
        object.__setattr__(self, "sizes", np.pad(self.sizes[:3], (0, max(0, 3 - len(self.sizes))), mode="constant"))

    def initialise(self):
        object.__setattr__(self, "_col_id", p.createCollisionShape(p.GEOM_BOX, halfExtents=self.sizes))
        object.__setattr__(self, "_vis_id", p.createVisualShape(p.GEOM_BOX, halfExtents=self.sizes))
        super().initialise()

    def __hash__(self) -> int:  # pylint: disable=useless-parent-delegation
        return super().__hash__()
