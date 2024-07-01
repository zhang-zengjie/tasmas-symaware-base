import math, os, sys
import numpy as np

root_path = os.getcwd()
sys.path.append(os.path.join(root_path, 'src'))

from symaware.base import (
    Agent,
    DefaultPerceptionSystem,
    AgentCoordinator,
    AwarenessVector,
    Controller,
    KnowledgeDatabase,
    MultiAgentAwarenessVector,
    MultiAgentKnowledgeDatabase,
    TimeIntervalAsyncLoopLock,
    TimeSeries,
    get_logger,
    initialize_logger,
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


def main():
    ###########################################################
    # 0. Parameters                                           #
    ###########################################################
    TIME_INTERVAL = 0.01
    NUM_AGENTS = 3
    LOG_LEVEL = "INFO"

    initialize_logger(LOG_LEVEL)

    ###########################################################
    # 1. Create the environment and add the obstacles         #
    ###########################################################
    
    env = Environment(async_loop_lock=TimeIntervalAsyncLoopLock(TIME_INTERVAL))
    env.add_entities((BoxEntity(position=np.array([2, 2, 2])), SphereEntity(position=np.array([-4, -4, 2]))))
    
    ###########################################################
    # For each agent in the simulation...                     #
    ###########################################################
    agent_coordinator = AgentCoordinator[MyKnowledgeDatabase](env)
    for i in range(NUM_AGENTS):
        ###########################################################
        # 2. Create the entity and model the agent will use       #
        ###########################################################
        agent_entity = RacecarEntity(i, model=RacecarModel(i), position=np.array([0, i, 0.1]))

        ###########################################################
        # 3. Create the agent and assign it an entity             #
        ###########################################################
        agent = Agent[MyKnowledgeDatabase](i, agent_entity)

        ###########################################################
        # 4. Add the agent to the environment                     #
        ###########################################################
        env.add_agents(agent)

        ###########################################################
        # 5. Create and set the component of the agent            #
        ###########################################################
        # In this example, wee need a controller and a perception system
        # They both run at the same frequency
        agent.add_components(
            MyController(agent.id, TimeIntervalAsyncLoopLock(TIME_INTERVAL)),
            DefaultPerceptionSystem(agent.id, env, TimeIntervalAsyncLoopLock(TIME_INTERVAL)),
        )

        ###########################################################
        # 6. Initialise the agent with some starting information  #
        ###########################################################
        awareness_vector = AwarenessVector(agent.id, np.zeros(7))
        knowledge_database = MyKnowledgeDatabase(
            goal_reached=False, goal_pos=np.array([2, 2]), tolerance=0.1, speed=20 + 10 * i, steering_tolerance=0.01
        )
        agent.initialise_agent(awareness_vector, {agent.id: knowledge_database})

        ###########################################################
        # 7. Add the agent to the coordinator                     #
        ###########################################################
        agent_coordinator.add_agents(agent)

    ###########################################################
    # 8. Run the simulation                                   #
    ###########################################################
    agent_coordinator.run(TIME_INTERVAL)


if __name__ == "__main__":
    main()
