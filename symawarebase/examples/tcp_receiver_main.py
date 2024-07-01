import asyncio
import math
import pickle
from typing import Any, Callable, Iterable
from dataclasses import dataclass
import numpy as np

from symaware.base import (
    Agent,
    AgentCoordinator,
    AwarenessVector,
    Controller,
    DefaultAsyncLoopLock,
    Identifier,
    KnowledgeDatabase,
    Message,
    MultiAgentAwarenessVector,
    MultiAgentKnowledgeDatabase,
    PerceptionSystem,
    StateObservation,
    Tasynclooplock,
    TimeIntervalAsyncLoopLock,
    TimeSeries,
    get_logger,
    CommunicationReceiver,
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
    goal_pos: np.ndarray
    tolerance: float
    next_agent_id: int
    idle: bool


@dataclass(frozen=True)
class MyMessage(Message):
    idle: bool


class MyPerceptionSystem(PerceptionSystem):
    """
    Unfortunately, this perception system is very limited, and can only perceive the state of the agent itself.
    """

    def _compute(self) -> dict[Identifier, StateObservation]:
        """
        Discard the information about any other agent.
        Only return the information related to the agent itself.
        """
        return {self._agent_id: StateObservation(self._agent_id, self._env.get_agent_state(self.agent_id))}


class MyCommunicationReceiver(CommunicationReceiver[DefaultAsyncLoopLock]):
    """
    Create a simple communication system that can only be used asynchronously.
    It sends and receives messages using tcp sockets.
    It means that the two agents could run on completely different machines
    and still communicate with each other via internet.
    """

    __LOGGER = get_logger(__name__, "MyCommunicationSystem")

    def __init__(
        self,
        agent_id: int,
        self_ip: str = "0.0.0.0",
        self_port: int = 9999,
    ):
        super().__init__(agent_id, DefaultAsyncLoopLock())
        self._get_message_task: asyncio.Task
        self._server: asyncio.Server
        self._self_ip = self_ip
        self._self_port = self_port
        self._message_queue: asyncio.Queue[Message] = asyncio.Queue()

    @log(__LOGGER)
    async def async_initialise_component(
        self,
        agent: Agent,
        initial_awareness_database: MultiAgentAwarenessVector,
        initial_knowledge_database: MultiAgentKnowledgeDatabase[MyKnowledgeDatabase],
    ):
        await super().async_initialise_component(agent, initial_awareness_database, initial_knowledge_database)
        self._server = await asyncio.start_server(self._put_message_in_queue, self._self_ip, self._self_port)

    @log(__LOGGER)
    async def _put_message_in_queue(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter):
        data = await reader.read()
        message = pickle.loads(data)
        self._message_queue.put_nowait(message)
        writer.close()
        await writer.wait_closed()

    def _receive_communication_from_channel(self) -> Iterable[Message]:
        raise NotImplementedError("This communication system can only be used asynchronously")

    def _compute(self):
        raise NotImplementedError("This communication system can only be used asynchronously")

    def _decode_message(self, messages: tuple[MyMessage]) -> bool:
        if len(messages) == 0 or messages[0].receiver_id != self.agent_id:
            return None
        return messages[0].idle

    @log(__LOGGER)
    async def _async_receive_communication_from_channel(self) -> "Iterable[Message]":
        self.__LOGGER.info("Agent %d: Waiting for a message", self.agent_id)
        try:
            self._get_message_task = asyncio.create_task(self._message_queue.get())
            message = await self._get_message_task
            self.__LOGGER.info(
                "Agent %d: Received the message %s from agent %d", self.agent_id, message, message.sender_id
            )
            return (message,)
        except asyncio.CancelledError:
            self.__LOGGER.info("Agent %d: Stopping waiting for new messages", self.agent_id)
            return tuple()

    async def async_stop(self):
        if self._server.is_serving():
            self._server.close()
        self._get_message_task.cancel()
        return await super().async_stop()

    def _update(self, idle: "None | bool"):
        if idle is not None:
            self._agent.knowledge_database[self._agent_id]["idle"] = idle


class MyController(Controller[Tasynclooplock]):
    """
    Simple controller that tries to reach a goal position, adjusting the speed and the steering angle,
    """

    __LOGGER = get_logger(__name__, "SimpleController")

    def __init__(self, agent_id: int, async_loop_lock=None):
        super().__init__(agent_id, async_loop_lock)
        self._tolerance = 0.0
        self._goal_pos = np.zeros(3)
        self._steering_tolerance = 0.0
        self._message_sent = False

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
        knowledge_database = self._agent.knowledge_database
        awareness_database = self._agent.awareness_database
        # If the agent is idle, stand still
        if knowledge_database[self._agent_id]["idle"]:
            return np.zeros(2), TimeSeries({0: self._goal_pos})

        # Check remaining distance
        pos = awareness_database[self._agent_id].state[:2]
        goal_direction = self._goal_pos - pos
        goal_direction_norm = np.linalg.norm(goal_direction)
        if goal_direction_norm < self._tolerance:
            self.__send_message(knowledge_database)
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

    def __send_message(self, knowledge_database: MultiAgentKnowledgeDatabase[MyKnowledgeDatabase]):
        if not self._message_sent:
            # When the goal is reached, send a message to the next agent once
            self.__LOGGER.info(
                "Agent %d: Goal reached. Sending a message to the next agent %d",
                self.agent_id,
                knowledge_database[self._agent_id]["next_agent_id"],
            )
            self._notify("goal_reached", self._agent)
            self._message_sent = True

    def add_on_goal_reached(self, callback: Callable[[Agent], Any]):
        self._add("goal_reached", callback)

    def remove_on_goal_reached(self, callback: Callable[[Agent], Any]):
        self._remove("goal_reached", callback)


def main():
    ###########################################################
    # 0. Parameters                                           #
    ###########################################################
    TIME_INTERVAL = 0.01
    LOG_LEVEL = "INFO"
    SELF_AGENT = 9999
    OTHER_AGENT = 7777
    SELF_IDLE = True

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
    initialise_info = {}

    ###########################################################
    # 2. Create the entity and model the agent will use       #
    ###########################################################

    agent_entity = RacecarEntity(SELF_AGENT, model=RacecarModel(SELF_AGENT), position=np.array([0, 0, 0.1]))

    ###########################################################
    # 3. Create the agent and assign it an entity             #
    ###########################################################
    agent = Agent[MyKnowledgeDatabase](SELF_AGENT, agent_entity)

    ###########################################################
    # 4. Add the agent to the environment                     #
    ###########################################################
    env.add_agents(agent)

    ###########################################################
    # 5. Create and set the component of the agent            #
    ###########################################################
    # In this example, all components run at the same frequency
    controller = MyController(agent.id, TimeIntervalAsyncLoopLock(TIME_INTERVAL))
    # Send a message to the next agent when the goal is reached

    agent.add_components(
        controller,
        MyPerceptionSystem(agent.id, env, TimeIntervalAsyncLoopLock(TIME_INTERVAL)),
        MyCommunicationReceiver(agent.id, self_ip="0.0.0.0", self_port=9999),
    )

    ###########################################################
    # 6. Initialise the agent with some starting information  #
    ###########################################################
    awareness_vector = AwarenessVector(agent.id, np.zeros(7))
    knowledge_database = MyKnowledgeDatabase(
        goal_pos=np.array([2, 2]),
        tolerance=0.1,
        next_agent_id=OTHER_AGENT,
        idle=SELF_IDLE,
        speed=20,
        steering_tolerance=0.01,
    )
    initialise_info[agent.id] = (awareness_vector, {agent.id: knowledge_database})

    ###########################################################
    # 7. Add the agent to the coordinator                     #
    ###########################################################
    agent_coordinator.add_agents(agent)

    ###########################################################
    # 8. Run the simulation asynchronously                    #
    ###########################################################
    agent_coordinator.async_run(initialise_info)


if __name__ == "__main__":
    main()
