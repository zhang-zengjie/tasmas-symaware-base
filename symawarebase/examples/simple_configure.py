import math

import numpy as np

from symaware.base import (
    Agent,
    AwarenessVector,
    Controller,
    DefaultPerceptionSystem,
    KnowledgeDatabase,
    MultiAgentAwarenessVector,
    MultiAgentKnowledgeDatabase,
    SymawareConfig,
    TimeIntervalAsyncLoopLock,
    TimeSeries,
    get_logger,
    log,
)

try:
    from symaware.simulators.pybullet import (
        BoxEntity,
        Environment,
        RacecarEntity,
        RacecarModel,
        SphereEntity,
    )
    import pybullet as p
except ImportError as e:
    raise ImportError(
        "symaware-pybullet non found. "
        "Try running `pip install symaware-pybullet` or `pip install symaware[simulators]`"
    ) from e


class MyKnowledgeDatabase(KnowledgeDatabase):
    goal_reached: bool
    goal_pos: np.ndarray
    tolerance: float
    speed: float
    steering_tolerance: float


class MyController(Controller):
    """
    Simple controller that tries to reach a goal position, adjusting the speed and the steering angle,
    """

    __LOGGER = get_logger(__name__, "SimpleController")

    def __init__(self, agent_id: int, async_loop_lock=None):
        super().__init__(agent_id, async_loop_lock)
        self._tolerance = 0.0
        self._goal_pos = np.zeros(3)
        self._steering_tolerance = 0.0

    @log(__LOGGER)
    def initialise_component(
        self,
        agent: Agent,
        initial_awareness_database: MultiAgentAwarenessVector,
        initial_knowledge_database: MultiAgentKnowledgeDatabase[MyKnowledgeDatabase],
    ):
        """
        Initialise the controller with the initial awareness and knowledge databases.
        It also becomes aware of the dynamical model of the agent, stored in the entity.
        Furthermore, the controller uses the knowledge database to set some parameters

        Args
        ----
        entity:
            The entity that the controller will control. It contains the dynamical model of the agent.
        initial_awareness_database:
            Initial awareness database of the agent
        initial_knowledge_database:
            Initial knowledge database of the agent
        """
        self._goal_pos = initial_knowledge_database[self.agent_id]["goal_pos"]
        self._tolerance = initial_knowledge_database[self.agent_id]["tolerance"]
        self._steering_tolerance = initial_knowledge_database[self.agent_id]["steering_tolerance"]
        super().initialise_component(agent, initial_awareness_database, initial_knowledge_database)

    @log(__LOGGER)
    def _compute(self) -> tuple[np.ndarray, TimeSeries]:
        """
        Given the awareness and knowledge databases, the controller computes the control input
        and the goal position for the agent.
        In this example, tries to reach the goal position, adjusting the speed and the steering angle,
        stopping when the goal is within a certain tolerance range.

        Returns
        -------
            - Control input to be applied to the agent
            - TimeSeries representing the intent of the agent
        """
        # Check remaining distance
        awareness_database = self._agent.awareness_database
        knowledge_database = self._agent.knowledge_database
        pos = awareness_database[self._agent_id].state[:2]
        goal_direction = self._goal_pos - pos
        goal_direction_norm = np.linalg.norm(goal_direction)
        if goal_direction_norm < self._tolerance:
            return np.zeros(2), TimeSeries({0: self._goal_pos})

        ## Steering angle
        goal_direction_normalized = goal_direction / goal_direction_norm
        current_rz = math.atan2(goal_direction_normalized[1], goal_direction_normalized[0])
        _, _, rz = p.getEulerFromQuaternion(awareness_database[self._agent_id].state[3:])
        difference_rz = current_rz - rz
        if difference_rz > self._steering_tolerance:
            steering_angle = 1
        elif difference_rz < -self._steering_tolerance:
            steering_angle = -1
        else:
            steering_angle = 0

        # Speed
        speed = knowledge_database[self._agent_id]["speed"]
        risk_dampener: float = 1 - awareness_database[self._agent_id].risk.get(0, 0)
        return np.array([speed * risk_dampener, steering_angle]), TimeSeries({0: self._goal_pos})

def configure_symaware(num_agents: int, time_step: float, disable_async: bool) -> SymawareConfig:
    """
    This function will be called by the main method of the symaware.base package.
    It returns the configuration for the symaware simulation.

    Example
    -------
    Run the simulation with 5 agents and a time step of 0.05:

    ```bash
    python3 -m symaware.base --num-agents 5 --time-step 0.05
    ```

    Args
    ----
    num_agents:
        Number of agents in the simulation
    time_step:
        Time step of the simulation
    disable_async:
        If True, the agents run synchronously

    Returns:
        Symaware configuration
    """

    def time_loop_lock() -> "TimeIntervalAsyncLoopLock | None":
        return TimeIntervalAsyncLoopLock(time_step) if not disable_async else None

    #############################################################
    # 1. Create the environment and add the obstacles           #
    #############################################################
    env = Environment(real_time_interval=time_step if disable_async else 0, async_loop_lock=time_loop_lock())
    env.add_entities((BoxEntity(position=np.array([2, 2, 2])), SphereEntity(position=np.array([-4, -4, 2]))))

    #############################################################
    # 2. Create the agents and add them to the environment      #
    #############################################################
    entities = [RacecarEntity(i, model=RacecarModel(i), position=np.array([i, i, 0.1])) for i in range(num_agents)]
    agents = [Agent[MyKnowledgeDatabase](i, entities[i]) for i in range(num_agents)]
    env.add_agents(agents)

    #############################################################
    # 3. Create the knowledge database and the awareness vector #
    #############################################################
    knowledge_databases = [
        {
            i: MyKnowledgeDatabase(
                goal_reached=False,
                goal_pos=np.array([2, 2]),
                tolerance=0.1,
                speed=20 + 10 * i,
                steering_tolerance=0.01,
            )
        }
        for i in range(num_agents)
    ]
    awareness_vectors = [AwarenessVector(i, np.zeros(7)) for i in range(num_agents)]

    #############################################################
    # 4. Return the configuration                               #
    #############################################################
    return SymawareConfig(
        agent=agents,
        controller=[MyController(i, time_loop_lock()) for i in range(num_agents)],
        knowledge_database=knowledge_databases,
        awareness_vector=awareness_vectors,
        risk_estimator=[None for i in range(num_agents)],
        uncertainty_estimator=[None for i in range(num_agents)],
        perception_system=[DefaultPerceptionSystem(i, env, time_loop_lock()) for i in range(num_agents)],
        communication_receiver=[None for i in range(num_agents)],
        communication_sender=[None for i in range(num_agents)],
        environment=env,
    )
