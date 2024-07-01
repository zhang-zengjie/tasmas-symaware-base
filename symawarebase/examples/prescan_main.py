import os

import numpy as np

from symaware.base import (
    Agent,
    AgentCoordinator,
    AwarenessVector,
    Controller,
    DefaultPerceptionSystem,
    KnowledgeDatabase,
    MultiAgentAwarenessVector,
    MultiAgentKnowledgeDatabase,
    TimeIntervalAsyncLoopLock,
    TimeSeries,
    get_logger,
    initialize_logger,
)

PRESCAN_DIR = "C:/Program Files/Simcenter Prescan/Prescan_2403"  # Adjust if needed
os.add_dll_directory(f"{PRESCAN_DIR}/bin")
os.environ["PATH"] = f"{PRESCAN_DIR}/bin;{os.environ['PATH']}"

try:
    from symaware.simulators.prescan import (
        AmesimDynamicalModel,
        Environment,
        Gear,
        TrackModel,
        ExistingEntity,
        Entity,
        DeerEntity,
        AirSensor,
        LmsSensor,
    )
except ImportError as e:
    raise ImportError(
        "symaware-pybullet non found. "
        "Try running `pip install symaware-pybullet` or `pip install symaware[simulators]`"
    ) from e


class MyKnowledgeDatabase(KnowledgeDatabase):
    pass


class MyController(Controller):
    """
    Simple controller that tries to reach a goal position, adjusting the speed and the steering angle,
    """

    __LOGGER = get_logger(__name__, "MyController")

    def __init__(self, agent_id: int, async_loop_lock=None):
        super().__init__(agent_id, async_loop_lock)
        self.__air: AirSensor
        self.__lms: LmsSensor

    def initialise_component(
        self,
        agent: Agent,
        initial_awareness_database: MultiAgentAwarenessVector,
        initial_knowledge_database: MultiAgentKnowledgeDatabase[MyKnowledgeDatabase],
    ):
        super().initialise_component(agent, initial_awareness_database, initial_knowledge_database)
        assert isinstance(self._agent.entity, Entity)
        self.__air: AirSensor = self._agent.entity.get_sensor_of_type(AirSensor)
        self.__lms: LmsSensor = self._agent.entity.get_sensor_of_type(LmsSensor)

    def _compute(self, compute_control_input: bool = False) -> tuple[np.ndarray, TimeSeries]:
        """
        Given the awareness and knowledge databases, the controller computes the control input
        and the goal position for the agent.
        In this example, the vehicle will accelerate when far from the target and start to brake
        when getting closer, while also following the street's lines.

        Args
        ----
        awareness_database:
            Awareness database of the agent
        knowledge_database:
            Knowledge database of the agent

        Returns
        -------
            - Control input to be applied to the agent
            - TimeSeries representing the intent of the agent
        """
        if not compute_control_input:
            return np.array([np.nan, np.nan, np.nan, np.nan]), TimeSeries()
        awareness_database = self._agent.awareness_database
        assert isinstance(self._agent._entity, Entity)
        x, y, z, roll, pitch, yaw, v, yaw_rate = awareness_database[self._agent_id].state
        if v == 0:
            v += 0.01
        lines = self.__lms.lms_lines

        time_gap = 2.0
        distance_to_target = self.__air.range if len(self.__air.data) > 0 else 9999
        time_to_collision = distance_to_target / v
        throttle = (time_to_collision - time_gap) / 3.0
        brake = self.__clamp((time_to_collision - time_gap) / -5.0, -10, 0)

        aebs_brake = 1.6
        aebs_min = 1.0
        aebs_stop_distance = 0.5

        if time_to_collision <= aebs_brake:
            brake += 0.4
        if time_to_collision <= aebs_min or distance_to_target <= aebs_stop_distance:
            brake = 1
            throttle = 0

        steering_wheel_gain = 150 * 3.14 / 180
        if len(lines) > 1 and len(lines[0]) > 0 and len(lines[1]) > 0:
            steering = (lines[0][0].y + lines[1][0].y) * steering_wheel_gain
            print(f"lines[0][0]: {lines[0][0].y}\tlines[0][1]: {lines[0][1].y}\tSteering angle: {steering}")
        else:
            throttle = 0
            brake = 1
            steering = 0

        return np.array([self.__clamp(throttle, 0, 1), self.__clamp(brake, 0, 1), steering, Gear.Forward]), TimeSeries()

    def on_pre_step(self):
        self._agent.entity.model.control_input, _ = self._compute(True)
        self._update((self._agent.entity.model.control_input, _))

    def _update(self, control_input_and_intent: tuple[np.ndarray, TimeSeries]):
        if np.isnan(control_input_and_intent[0][0]):
            return
        super()._update(control_input_and_intent)

    def __clamp(self, value: float, min_value: float, max_value: float) -> float:
        return max(min(value, max_value), min_value)


def main():

    ###########################################################
    # 0. Parameters                                           #
    ###########################################################
    TIME_INTERVAL = 0.05
    LOG_LEVEL = "INFO"

    initialize_logger(LOG_LEVEL)

    ###########################################################
    # 1. Load the environment from the file and add entities  #
    ###########################################################
    env = Environment(
        filename="Demo_AmesimPreconfiguredDynamics.pb", async_loop_lock=TimeIntervalAsyncLoopLock(TIME_INTERVAL)
    )
    env.add_entities(DeerEntity(position=np.array([0, 4, 0]), orientation=np.array((0, 0, -90))))

    agent_coordinator = AgentCoordinator[MyKnowledgeDatabase](env)

    target_entity = ExistingEntity(id=-1, object_name="BMW_X5_SUV_1", model=TrackModel(-1, existing=True))
    env.add_entities(target_entity)

    ###########################################################
    # 2. Add the agent in the environment                     #
    ###########################################################

    agent_entity = ExistingEntity(
        id=1,
        object_name="Audi_A8_Sedan_1",
        sensors=(AirSensor(existing=True), LmsSensor(existing=True)),
        model=AmesimDynamicalModel(1, is_flat_ground=False, existing=True),
    )
    agent = Agent[MyKnowledgeDatabase](1, agent_entity)
    env.add_agents(agent)

    ###########################################################
    # 3. Create and set the component of the agent            #
    ###########################################################
    controller = MyController(agent.id, TimeIntervalAsyncLoopLock(TIME_INTERVAL))
    agent.add_components(controller, DefaultPerceptionSystem(agent.id, env, TimeIntervalAsyncLoopLock(TIME_INTERVAL)))
    env.set_on_pre_step(controller.on_pre_step)

    ###########################################################
    # 4. Initialise the agent with some starting information  #
    ###########################################################
    awareness_vector = AwarenessVector(agent.id, np.zeros(8))
    knowledge_database = MyKnowledgeDatabase(goal_reached=False)
    agent.initialise_agent(awareness_vector, {agent.id: knowledge_database})

    ###########################################################
    # 5. Run the simulation                                   #
    ###########################################################
    agent_coordinator.add_agents(agent)

    agent_coordinator.async_run(timeout=40)


if __name__ == "__main__":
    main()
