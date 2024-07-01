import os
from typing import TYPE_CHECKING

import numpy as np
import pybullet as p
import pybullet_data
from symaware.base.models import Environment as BaseEnvironment
from symaware.base.utils import get_logger, log

from .entities import Entity

if TYPE_CHECKING:
    # String type hinting to support python 3.9
    from symaware.base.utils import AsyncLoopLock


class Environment(BaseEnvironment):
    """
    Environment based on the PyBullet physics engine.

    Args
    ----
    real_time_interval:
        If set to a strictly positive value, pybullet will run the simulation in real time.
        Otherwise, the simulation will run when :func:`step` is called.
    connection_method:
        Method used to connect to the pybullet server. See the pybullet documentation for more information.
    async_loop_lock:
        Async loop lock to use for the environment
    """

    __LOGGER = get_logger(__name__, "Pybullet.Environment")

    def __init__(
        self,
        real_time_interval: float = 0,
        connection_method: int = p.GUI,
        async_loop_lock: "AsyncLoopLock | None" = None,
    ):
        super().__init__(async_loop_lock)
        self._is_pybullet_initialized = False
        self._real_time_interval = real_time_interval
        self._connection_method = connection_method

    @property
    def use_real_time(self) -> bool:
        return self._real_time_interval > 0

    @log(__LOGGER)
    def get_entity_state(self, entity: Entity) -> np.ndarray:
        if not isinstance(entity, Entity):
            raise TypeError(f"Expected SpatialEntity, got {type(entity)}")
        position, orientation = p.getBasePositionAndOrientation(entity.entity_id)
        return np.array(position + orientation)

    @log(__LOGGER)
    def _add_entity(self, entity: Entity):
        if not isinstance(entity, Entity):
            raise TypeError(f"Expected SpatialEntity, got {type(entity)}")
        if not self._is_pybullet_initialized:
            self.initialise()
        entity.initialise()

    def initialise(self):
        if self._is_pybullet_initialized:
            return
        self._is_pybullet_initialized = True
        p.connect(self._connection_method)
        p.resetSimulation()
        p.setRealTimeSimulation(self.use_real_time)
        p.setGravity(0, 0, -10)
        p.loadURDF(os.path.join(pybullet_data.getDataPath(), "plane.urdf"))

    def step(self):
        for entity in self._agent_entities.values():
            entity.step()
        if not self.use_real_time:
            p.stepSimulation()

    def stop(self):
        self._is_pybullet_initialized = False
        p.disconnect()
