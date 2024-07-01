from abc import abstractmethod
from typing import TypedDict
import time

import numpy as np
import pybullet as p
from symaware.base import DynamicalModel as BaseDynamicalModel
from symaware.base import Identifier, get_logger
import xml.etree.ElementTree as etxml
from .urdf import URDF


class RacecarModelSubinputs(TypedDict):
    target_velocity: np.ndarray
    steering_angle: np.ndarray


class DroneModelSubinputs(TypedDict):
    rpm1: float
    rpm2: float
    rpm3: float
    rpm4: float


class DynamicalModel(BaseDynamicalModel):
    """
    Abstract class for the dynamical models using the PyBullet physics engine.

    Args
    ----
    ID:
        Identifier of the agent this model belongs to
    control_input:
        Initial control input of the agent. It also used to validate the size of future control inputs
    """

    def __init__(self, ID: Identifier, control_input: np.ndarray):
        super().__init__(ID, control_input=control_input)
        self._entity_id = -1

    def initialise(self, entity_id: int):
        self._entity_id = entity_id
        if self._entity_id < 0:
            raise RuntimeError(f"Failed to initialise {self.__class__.__name__}: negative id = {self._ID}")

    @abstractmethod
    def step(self):
        pass


class RacecarModel(DynamicalModel):
    """
    PyBullet dynamical model for the racecar.

    Args
    ----
    ID:
        Identifier of the agent this model belongs to
    control_input:
        Initial control input of the agent. It also used to validate the size of future control inputs
    max_force:
        Maximum force that can be applied to the wheels
    steering_links:
        Tuple of the two links that are used to steer the car
    motorized_wheels:
        Tuple of the two links that are used to drive the car
    """

    __LOGGER = get_logger(__name__, "RacecarModel")

    def __init__(
        self,
        ID: Identifier,
        max_force: float = 20.0,
        steering_links: tuple[int, int] = (0, 2),
        motorized_wheels: tuple[int, int] = (8, 15),
    ):
        super().__init__(ID, control_input=np.zeros(2))
        self._max_force = max_force
        self._steering_links = steering_links
        self._motorized_wheels = motorized_wheels

    @property
    def subinputs_dict(self) -> RacecarModelSubinputs:
        return {"target_velocity": self.control_input[0], "steering_angle": self.control_input[1]}

    def initialise(self, entity_id: int):
        super().initialise(entity_id)

        for wheel in range(p.getNumJoints(self._entity_id)):
            p.setJointMotorControl2(self._entity_id, wheel, p.VELOCITY_CONTROL, targetVelocity=0, force=0)

        c = p.createConstraint(
            self._entity_id,
            9,
            self._entity_id,
            11,
            jointType=p.JOINT_GEAR,
            jointAxis=[0, 1, 0],
            parentFramePosition=[0, 0, 0],
            childFramePosition=[0, 0, 0],
        )
        p.changeConstraint(c, gearRatio=1, maxForce=10000)

        c = p.createConstraint(
            self._entity_id,
            10,
            self._entity_id,
            13,
            jointType=p.JOINT_GEAR,
            jointAxis=[0, 1, 0],
            parentFramePosition=[0, 0, 0],
            childFramePosition=[0, 0, 0],
        )
        p.changeConstraint(c, gearRatio=-1, maxForce=10000)

        c = p.createConstraint(
            self._entity_id,
            9,
            self._entity_id,
            13,
            jointType=p.JOINT_GEAR,
            jointAxis=[0, 1, 0],
            parentFramePosition=[0, 0, 0],
            childFramePosition=[0, 0, 0],
        )
        p.changeConstraint(c, gearRatio=-1, maxForce=10000)

        c = p.createConstraint(
            self._entity_id,
            16,
            self._entity_id,
            18,
            jointType=p.JOINT_GEAR,
            jointAxis=[0, 1, 0],
            parentFramePosition=[0, 0, 0],
            childFramePosition=[0, 0, 0],
        )
        p.changeConstraint(c, gearRatio=1, maxForce=10000)

        c = p.createConstraint(
            self._entity_id,
            16,
            self._entity_id,
            19,
            jointType=p.JOINT_GEAR,
            jointAxis=[0, 1, 0],
            parentFramePosition=[0, 0, 0],
            childFramePosition=[0, 0, 0],
        )
        p.changeConstraint(c, gearRatio=-1, maxForce=10000)

        c = p.createConstraint(
            self._entity_id,
            17,
            self._entity_id,
            19,
            jointType=p.JOINT_GEAR,
            jointAxis=[0, 1, 0],
            parentFramePosition=[0, 0, 0],
            childFramePosition=[0, 0, 0],
        )
        p.changeConstraint(c, gearRatio=-1, maxForce=10000)

        c = p.createConstraint(
            self._entity_id,
            1,
            self._entity_id,
            18,
            jointType=p.JOINT_GEAR,
            jointAxis=[0, 1, 0],
            parentFramePosition=[0, 0, 0],
            childFramePosition=[0, 0, 0],
        )
        p.changeConstraint(c, gearRatio=-1, gearAuxLink=15, maxForce=10000)
        c = p.createConstraint(
            self._entity_id,
            3,
            self._entity_id,
            19,
            jointType=p.JOINT_GEAR,
            jointAxis=[0, 1, 0],
            parentFramePosition=[0, 0, 0],
            childFramePosition=[0, 0, 0],
        )
        p.changeConstraint(c, gearRatio=-1, gearAuxLink=15, maxForce=10000)

    def step(self):
        target_velocity, steering_angle = self._control_input

        for motor in self._motorized_wheels:
            p.setJointMotorControl2(
                self._entity_id,
                motor,
                p.VELOCITY_CONTROL,
                targetVelocity=target_velocity,
                force=self._max_force,
            )
        for steer in self._steering_links:
            p.setJointMotorControl2(self._entity_id, steer, p.POSITION_CONTROL, targetPosition=steering_angle)


class DroneModel(DynamicalModel):
    __LOGGER = get_logger(__name__, "DroneModel")

    def __init__(self, ID: Identifier, urp_path: str, debug: bool = False):
        """Abstract class for the dynamical models using the PyBullet physics engine.

        Args
        ----
        ID:
            Identifier of the agent this model belongs to
        urp_path:
            Path to the URDF file containing the drone's parameters
        control_input:
            Initial control input of the agent. It also used to validate the size of future control inputs
        """
        super().__init__(ID, control_input=np.ndarray(4))
        self._urp_path = urp_path
        self._load_urdf_args()
        #### Compute constants #####################################
        self.debug = debug
        self.gravity = 9.8 * self.m
        self.hover_rpm = np.sqrt(self.gravity / (4 * self.kf))
        self.max_rpm = np.sqrt((self.thrust_to_weight_ratio * self.gravity) / (4 * self.kf))
        self.max_thrust = 4 * self.kf * self.max_rpm**2
        self.max_z_torque = 2 * self.km * self.max_rpm**2
        self.gnd_eff_h_clip = (
            0.25 * self.prop_radius * np.sqrt((15 * self.max_rpm**2 * self.kf * self.gnd_eff_coeff) / self.max_thrust)
        )

        self.pos: tuple[float, float, float]
        self.quat: tuple[float, float, float, float]
        self.vel: tuple[float, float, float]
        self.ang_v: tuple[float, float, float]
        self.rpy_rates: np.ndarray
        self._last_timestamp = time.time()

    @property
    def normalized_control_input(self):
        return np.where(
            self.control_input <= self.hover_rpm,
            -1 + self.control_input / self.hover_rpm,
            1 + (self.control_input - self.hover_rpm) / (self.max_rpm - self.hover_rpm),
        )

    @normalized_control_input.setter
    def normalized_control_input(self, action: np.ndarray):
        self.control_input = self.normalized_control_input_to_rpm(action)

    @property
    def subinputs_dict(self) -> DroneModelSubinputs:
        return {
            "rpm1": self.control_input[0],
            "rpm2": self.control_input[1],
            "rpm3": self.control_input[2],
            "rpm4": self.control_input[3],
        }

    def _load_urdf_args(self):
        """Loads parameters from an URDF file.

        This method is nothing more than a custom XML parser for the .urdf
        files in folder `assets/`.

        """
        URDF_TREE = etxml.parse(self._urp_path).getroot()
        self.m = float(URDF_TREE[1][0][1].attrib["value"])
        self.l = float(URDF_TREE[0].attrib["arm"])
        self.thrust_to_weight_ratio = float(URDF_TREE[0].attrib["thrust2weight"])
        self.ixx = float(URDF_TREE[1][0][2].attrib["ixx"])
        self.iyy = float(URDF_TREE[1][0][2].attrib["iyy"])
        self.izz = float(URDF_TREE[1][0][2].attrib["izz"])
        self.j = np.diag([self.ixx, self.iyy, self.izz])
        self.j_inv = np.linalg.inv(self.j)
        self.kf = float(URDF_TREE[0].attrib["kf"])
        self.km = float(URDF_TREE[0].attrib["km"])
        self.collision_h = float(URDF_TREE[1][2][1][0].attrib["length"])
        self.collision_r = float(URDF_TREE[1][2][1][0].attrib["radius"])
        self.collision_shape_offsets = [float(s) for s in URDF_TREE[1][2][0].attrib["xyz"].split(" ")]
        self.collision_z_offset = self.collision_shape_offsets[2]
        self.max_speed_kmh = float(URDF_TREE[0].attrib["max_speed_kmh"])
        self.gnd_eff_coeff = float(URDF_TREE[0].attrib["gnd_eff_coeff"])
        self.prop_radius = float(URDF_TREE[0].attrib["prop_radius"])
        self.drag_coeff_xy = float(URDF_TREE[0].attrib["drag_coeff_xy"])
        self.drag_coeff_z = float(URDF_TREE[0].attrib["drag_coeff_z"])
        self.drag_coeff = np.array([self.drag_coeff_xy, self.drag_coeff_xy, self.drag_coeff_z])
        self.dw_coeff_1 = float(URDF_TREE[0].attrib["dw_coeff_1"])
        self.dw_coeff_2 = float(URDF_TREE[0].attrib["dw_coeff_2"])
        self.dw_coeff_3 = float(URDF_TREE[0].attrib["dw_coeff_3"])
        self.__LOGGER.info("loaded parameters from the drone's .urdf")
        self.__LOGGER.info("m %f, L %f,", self.m, self.l)
        self.__LOGGER.info("ixx %f, iyy %f, izz %f,", self.j[0, 0], self.j[1, 1], self.j[2, 2])
        self.__LOGGER.info("kf %f, km %f,", self.kf, self.km)
        self.__LOGGER.info("t2w %f, max_speed_kmh %f,", self.thrust_to_weight_ratio, self.max_speed_kmh)
        self.__LOGGER.info("gnd_eff_coeff %f, prop_radius %f,", self.gnd_eff_coeff, self.prop_radius)
        self.__LOGGER.info("drag_xy_coeff %f, drag_z_coeff %f,", self.drag_coeff[0], self.drag_coeff[2])
        self.__LOGGER.info(
            "dw_coeff_1 %f, dw_coeff_2 %f, dw_coeff_3 %f", self.dw_coeff_1, self.dw_coeff_2, self.dw_coeff_3
        )

    def _show_drone_local_axes(self):
        """Draws the local frame of the n-th drone in PyBullet's GUI."""
        if self.debug:
            AXIS_LENGTH = 2 * self.l
            self.X_AX = p.addUserDebugLine(
                lineFromXYZ=[0, 0, 0],
                lineToXYZ=[AXIS_LENGTH, 0, 0],
                lineColorRGB=[1, 0, 0],
                parentObjectUniqueId=self._entity_id,
                parentLinkIndex=-1,
                replaceItemUniqueId=int(self.X_AX),
            )
            self.Y_AX = p.addUserDebugLine(
                lineFromXYZ=[0, 0, 0],
                lineToXYZ=[0, AXIS_LENGTH, 0],
                lineColorRGB=[0, 1, 0],
                parentObjectUniqueId=self._entity_id,
                parentLinkIndex=-1,
                replaceItemUniqueId=int(self.Y_AX),
            )
            self.Z_AX = p.addUserDebugLine(
                lineFromXYZ=[0, 0, 0],
                lineToXYZ=[0, 0, AXIS_LENGTH],
                lineColorRGB=[0, 0, 1],
                parentObjectUniqueId=self._entity_id,
                parentLinkIndex=-1,
                replaceItemUniqueId=int(self.Z_AX),
            )

    def normalized_control_input_to_rpm(self, normalized_control_input: np.ndarray):
        """De-normalizes the [-1, 1] range to the [0, max_rpm] range.

        Args
        ----
        normalized_control_input:
            (4)-shaped array of ints containing an input in the [-1, 1] range.

        Returns
        -------
            (4)-shaped array of ints containing RPMs for the 4 motors in the [0, max_rpm] range.
        """
        if np.any(np.abs(normalized_control_input) > 1):
            self.__LOGGER.error("_normalized_action_to_rpm(), out-of-bound action: %s", normalized_control_input)
        return np.where(
            normalized_control_input <= 0,
            (normalized_control_input + 1) * self.hover_rpm,
            self.hover_rpm + (self.max_rpm - self.hover_rpm) * normalized_control_input,
        )

    def _update_kinematic_info(self):
        """Updates and stores the drones kinematic information.

        This method is meant to limit the number of calls to PyBullet in each step
        and improve performance (at the expense of memory).
        """
        self.pos, self.quat = p.getBasePositionAndOrientation(self._entity_id)
        self.rpy = p.getEulerFromQuaternion(self.quat)
        self.vel, self.ang_v = p.getBaseVelocity(self._entity_id)

    def _integrate_q(self, quat: np.ndarray, omega: np.ndarray, dt: float) -> np.ndarray:
        omega_norm = np.linalg.norm(omega)
        b, q, r = omega
        if np.isclose(omega_norm, 0):
            return quat
        lambda_ = np.array([[0, r, -q, b], [-r, 0, b, q], [q, -b, 0, r], [-b, -q, -r, 0]]) * 0.5
        theta = omega_norm * dt / 2
        quat = np.dot(np.eye(4) * np.cos(theta) + 2 / omega_norm * lambda_ * np.sin(theta), quat)
        return quat

    def _dynamics(self, rpm: np.ndarray):
        """Explicit dynamics implementation.

        Based on code written at the Dynamic Systems Lab by James Xu.

        Args
        ----
        rpm : ndarray
            (4)-shaped array of ints containing the RPMs values of the 4 motors.
        nth_drone : int
            The ordinal number/position of the desired drone in list self.DRONE_IDS.

        """
        #### Current state #########################################
        time_step = time.time() - self._last_timestamp
        pos = self.pos
        quat = self.quat
        vel = self.vel
        rpy_rates = self.rpy_rates
        rotation = np.array(p.getMatrixFromQuaternion(quat)).reshape(3, 3)
        #### Compute forces and torques ############################
        forces = np.array(rpm**2) * self.kf
        thrust = np.array([0, 0, np.sum(forces)])
        thrust_world_frame = np.dot(rotation, thrust)
        force_world_frame = thrust_world_frame - np.array([0, 0, self.gravity])
        x_torque, y_torque, z_torque = self._get_torque(rpm, forces)
        torques = np.array([x_torque, y_torque, z_torque])
        torques = torques - np.cross(rpy_rates, np.dot(self.j, rpy_rates))
        rpy_rates_deriv = np.dot(self.j_inv, torques)
        no_pybullet_dyn_accs = force_world_frame / self.m
        #### Update state ##########################################
        vel = vel + time_step * no_pybullet_dyn_accs
        rpy_rates = rpy_rates + time_step * rpy_rates_deriv
        pos = pos + time_step * vel
        quat = self._integrate_q(quat, rpy_rates, time_step)
        #### Set PyBullet's state ##################################
        p.resetBasePositionAndOrientation(self._entity_id, pos, quat)
        #### Note: the base's velocity only stored and not used ####
        p.resetBaseVelocity(self._entity_id, vel, np.dot(rotation, rpy_rates))
        #### Store the roll, pitch, yaw rates for the next step ####
        self.rpy_rates = rpy_rates
        self._last_timestamp = time.time()

    def _downwash(self):
        """PyBullet implementation of a ground effect model.

        Based on experiments conducted at the Dynamic Systems Lab by SiQi Zhou.
        """
        delta_z = self.pos[2] - self.pos[2]
        delta_xy = np.linalg.norm(np.array(self.pos[0:2]) - np.array(self.pos[0:2]))
        if delta_z > 0 and delta_xy < 10:  # Ignore drones more than 10 meters away
            alpha = self.dw_coeff_1 * (self.prop_radius / (4 * delta_z)) ** 2
            beta = self.dw_coeff_2 * delta_z + self.dw_coeff_3
            downwash = [0, 0, -alpha * np.exp(-0.5 * (delta_xy / beta) ** 2)]
            p.applyExternalForce(
                self._entity_id,
                4,
                forceObj=downwash,
                posObj=[0, 0, 0],
                flags=p.LINK_FRAME,
            )

    def _drag(self, rpm: np.ndarray):
        """PyBullet implementation of a drag model.

        Based on the the system identification in (Forster, 2015).

        Args
        ----
        rpm : ndarray
            (4)-shaped array of ints containing the RPMs values of the 4 motors.
        """
        #### Rotation matrix of the base ###########################
        base_rot = np.array(p.getMatrixFromQuaternion(self.quat)).reshape(3, 3)
        #### Simple draft model applied to the base/center of mass #
        drag_factors = -1 * self.drag_coeff * np.sum(np.array(2 * np.pi * rpm / 60))
        drag = np.dot(base_rot.T, drag_factors * np.array(self.vel))
        p.applyExternalForce(
            self._entity_id,
            4,
            forceObj=drag,
            posObj=[0, 0, 0],
            flags=p.LINK_FRAME,
        )

    def _ground_effect(self, rpm: np.ndarray):
        """PyBullet implementation of a ground effect model.

        Inspired by the analytical model used for comparison in (Shi et al., 2019).

        Args
        ----
        rpm : ndarray
            (4)-shaped array of ints containing the RPMs values of the 4 motors.
        """
        #### Kin. info of all links (propellers and center of mass)
        link_states = p.getLinkStates(
            self._entity_id,
            linkIndices=[0, 1, 2, 3, 4],
            computeLinkVelocity=1,
            computeForwardKinematics=1,
        )
        #### Simple, per-propeller ground effects ##################
        prop_heights = np.array(
            [link_states[0][0][2], link_states[1][0][2], link_states[2][0][2], link_states[3][0][2]]
        )
        prop_heights = np.clip(prop_heights, self.gnd_eff_h_clip, np.inf)
        gnd_effects = np.array(rpm**2) * self.kf * self.gnd_eff_coeff * (self.prop_radius / (4 * prop_heights)) ** 2
        if np.abs(self.rpy[0]) < np.pi / 2 and np.abs(self.rpy[1]) < np.pi / 2:
            for i in range(4):
                p.applyExternalForce(
                    self._entity_id,
                    i,
                    forceObj=[0, 0, gnd_effects[i]],
                    posObj=[0, 0, 0],
                    flags=p.LINK_FRAME,
                )

    def step(self):
        """Advances the environment by one simulation step."""
        forces = np.array(self._control_input**2) * self.kf
        _, _, z_torque = self._get_torque(self._control_input, forces)
        for i in range(4):
            p.applyExternalForce(
                self._entity_id,
                i,
                forceObj=[0, 0, forces[i]],
                posObj=[0, 0, 0],
                flags=p.LINK_FRAME,
            )
        p.applyExternalTorque(self._entity_id, 4, torqueObj=[0, 0, z_torque], flags=p.LINK_FRAME)

        self._update_kinematic_info()

    @abstractmethod
    def _get_torque(self, rpm: float, forces: np.ndarray) -> tuple[float, float, float]:
        pass

    def _get_state(self):
        """Returns the state vector of the n-th drone.

        Returns
        -------
            (16)-shaped array of floats containing the state vector of the n-th drone.
            Check the only line in this method and `_updateAndStoreKinematicInformation()`
            to understand its format.

        """
        state = np.hstack([self.pos, self.quat, self.rpy, self.vel, self.ang_v])
        return state.reshape(16)


class DroneRacerModel(DroneModel):

    def __init__(self, ID: Identifier, debug: bool = False):
        """Abstract class for the dynamical models using the PyBullet physics engine.

        Args
        ----
        ID:
            Identifier of the agent this model belongs to
        """
        super().__init__(ID, urp_path=URDF.DRONE_RACER.urdf, debug=debug)
        self.max_xy_torque = (2 * self.l * self.kf * self.max_rpm**2) / np.sqrt(2)

    def _get_torque(self, rpm: float, forces: np.ndarray) -> tuple[float, float, float]:
        z_torques = -(np.array(rpm**2) * self.km)
        z_torque = -z_torques[0] + z_torques[1] - z_torques[2] + z_torques[3]
        x_torque = (forces[0] + forces[1] - forces[2] - forces[3]) * (self.l / np.sqrt(2))
        y_torque = (-forces[0] + forces[1] + forces[2] - forces[3]) * (self.l / np.sqrt(2))
        return x_torque, y_torque, z_torque


class DroneCf2xModel(DroneModel):

    def __init__(self, ID: Identifier, debug: bool = False):
        """Abstract class for the dynamical models using the PyBullet physics engine.

        Args
        ----
        ID:
            Identifier of the agent this model belongs to
        """
        super().__init__(ID, urp_path=URDF.DRONE_CF2X.urdf, debug=debug)
        self.max_xy_torque = (2 * self.l * self.kf * self.max_rpm**2) / np.sqrt(2)

    def _get_torque(self, rpm: float, forces: np.ndarray) -> tuple[float, float, float]:
        z_torques = np.array(rpm**2) * self.km
        z_torque = -z_torques[0] + z_torques[1] - z_torques[2] + z_torques[3]
        x_torque = (forces[0] + forces[1] - forces[2] - forces[3]) * (self.l / np.sqrt(2))
        y_torque = (-forces[0] + forces[1] + forces[2] - forces[3]) * (self.l / np.sqrt(2))
        return x_torque, y_torque, z_torque


class DroneCf2pModel(DroneModel):

    def __init__(self, ID: Identifier, debug: bool = False):
        """Abstract class for the dynamical models using the PyBullet physics engine.

        Args
        ----
        ID:
            Identifier of the agent this model belongs to
        """
        super().__init__(ID, urp_path=URDF.DRONE_CF2P.urdf, debug=debug)
        self.max_xy_torque = self.l * self.kf * self.max_rpm**2

    def _get_torque(self, rpm: float, forces: np.ndarray) -> tuple[float, float, float]:
        z_torques = np.array(rpm**2) * self.km
        z_torque = -z_torques[0] + z_torques[1] - z_torques[2] + z_torques[3]
        x_torque = (forces[1] - forces[3]) * self.l
        y_torque = (-forces[0] + forces[2]) * self.l
        return x_torque, y_torque, z_torque
