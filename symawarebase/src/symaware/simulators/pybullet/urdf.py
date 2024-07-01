import os
from importlib import resources
from enum import Enum
import pybullet_data


class URDF(Enum):
    RACECAR = "racecar"
    DRONE_RACER = "drone_racer"
    DRONE_CF2P = "drone_cf2p"
    DRONE_CF2X = "drone_cf2x"

    @property
    def urdf(self) -> str:
        if self == URDF.RACECAR:
            return os.path.join(pybullet_data.getDataPath(), "racecar", "racecar_differential.urdf")
        elif self == URDF.DRONE_RACER:
            return os.path.join(resources.files("symaware.simulators.pybullet"), "assets", "racer.urdf")
        elif self == URDF.DRONE_CF2P:
            return os.path.join(resources.files("symaware.simulators.pybullet"), "assets", "cf2p.urdf")
        elif self == URDF.DRONE_CF2X:
            return os.path.join(resources.files("symaware.simulators.pybullet"), "assets", "cf2x.urdf")
        else:
            raise ValueError(f"Unknown urdp {self}")
