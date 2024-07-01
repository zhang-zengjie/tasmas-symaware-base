import asyncio
import math
import pickle
from typing import Any, Callable
from dataclasses import dataclass

import numpy as np

from symaware.base import (
    Agent,
    AgentCoordinator,
    AwarenessVector,
    CommunicationSender,
    Controller,
    DefaultRiskEstimator,
    DefaultUncertaintyEstimator,
    EventAsyncLoopLock,
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
    initialize_logger,
    log,
)


try:
    from symaware.simulators.pymunk import (
        BoxEntity,
        Environment,
        VelocityModel,
        SphereEntity,
    )
except ImportError as e:
    raise ImportError(
        "symaware-pymunk non found. Try running `pip install symaware-pymunk` or `pip install symaware[simulators]`"
    ) from e


class MyKnowledgeDatabase(KnowledgeDatabase):
    goal_pos: np.ndarray
    tolerance: float
    next_agent_id: int
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


@dataclass(frozen=True)
class MyMessage(Message):
    idle: bool


class MyCommunicationSender(CommunicationSender[EventAsyncLoopLock]):
    """
    Create a simple communication system that can only be used asynchronously.
    It sends and receives messages using tcp sockets.
    It means that the two agents could run on completely different machines
    and still communicate with each other via internet.
    """

    __LOGGER = get_logger(__name__, "MyCommunicationSystem")

    def __init__(self, agent_id: int, dest_ip: str = "127.0.0.1", dest_port: int = 9999):
        super().__init__(agent_id, EventAsyncLoopLock())
        self._dest_ip = dest_ip
        self._dest_port = dest_port

    def _send_communication_through_channel(self, message: Message):
        raise NotImplementedError("This communication system can only be used asynchronously")

    @log(__LOGGER)
    async def _async_send_communication_through_channel(self, message: Message):
        try:
            _, writer = await asyncio.open_connection(self._dest_ip, self._dest_port)
        except ConnectionRefusedError:
            self.__LOGGER.warning("Agent %d: Connection to agent %d refused", self.agent_id, message.receiver_id)
            return
        self.__LOGGER.info("Sending: %r", message)
        data = pickle.dumps(message)
        writer.write(data)
        await writer.drain()
        writer.close()
        await writer.wait_closed()


class MyController(Controller[Tasynclooplock]):
    """

    Simple controller that tries to reach a goal position, adjusting the velocity.

    """

    __LOGGER = get_logger(__name__, "MyController")

    def __init__(self, agent_id: int, async_loop_lock: "Tasynclooplock | None" = None):
        super().__init__(agent_id, async_loop_lock)
        self._tolerance = 0.0
        self._goal_pos = np.zeros(3)
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
        agent:
            The entity that the controller will control. It contains the dynamical model of the agent.
        initial_awareness_database:
            Initial awareness database of the agent
        initial_knowledge_database:
            Initial knowledge database of the agent
        """
        super().initialise_component(agent, initial_awareness_database, initial_knowledge_database)
        self._goal_pos = initial_knowledge_database[self.agent_id]["goal_pos"]
        self._tolerance = initial_knowledge_database[self.agent_id]["tolerance"]

    @log(__LOGGER)
    def _compute(self) -> tuple[np.ndarray, TimeSeries]:
        """
        Given the awareness and knowledge databases, the controller computes the control input
        and the goal position for the agent.
        In this example, tries to reach the goal position, adjusting the speed of the object.
        It only moves if the agent is not idle.

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
        # Rotate the goal direction to the agent frame
        goal_direction_norm = np.linalg.norm(goal_direction)

        if goal_direction_norm < self._tolerance:
            self.__send_message(knowledge_database)
            return np.zeros(2), TimeSeries({0: self._goal_pos})

        goal_direction_normalized = goal_direction / goal_direction_norm
        return goal_direction_normalized * 50, TimeSeries({0: self._goal_pos})

    def add_on_goal_reached(self, callback: Callable[[Agent], Any]):
        self._add("goal_reached", callback)

    def remove_on_goal_reached(self, callback: Callable[[Agent], Any]):
        self._remove("goal_reached", callback)

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


def main():
    ###########################################################
    # 0. Parameters                                           #
    ###########################################################
    TIME_INTERVAL = 0.1
    LOG_LEVEL = "INFO"
    SELF_AGENT = 7777
    OTHER_AGENT = 9999
    SELF_IDLE = False

    initialize_logger(LOG_LEVEL)

    ###########################################################
    # 1. Create the environment and add the obstacles         #
    ###########################################################
    env = Environment(async_loop_lock=TimeIntervalAsyncLoopLock(TIME_INTERVAL))
    box = BoxEntity(position=np.array([300, 300]), sizes=np.array([30, 10]))
    env.add_entities(box)

    def send_message_callback(agent: Agent[MyKnowledgeDatabase]):
        receiver_id = agent.knowledge_database[agent.id]["next_agent_id"]
        communication_sender = agent.communication_sender
        assert isinstance(communication_sender, MyCommunicationSender)
        communication_sender.enqueue_messages(MyMessage(agent.id, receiver_id, False))
        # Without this line, the message would just be enqueued
        communication_sender.async_loop_lock.trigger()  # Send the enqueued messages in the next iteration.

    agent_coordinator = AgentCoordinator[MyKnowledgeDatabase](env)
    initialise_info = {}

    ###########################################################
    # 2. Create the entity and model the agent will use       #
    ###########################################################

    agent_entity = SphereEntity(
        SELF_AGENT,
        model=VelocityModel(SELF_AGENT),
        position=np.array([10 + 580, 10 + 580 * ((0) // 2)]),
        radius=10,
        angle=math.pi,
    )

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
    controller.add_on_goal_reached(send_message_callback)
    agent.add_components(
        controller,
        MyPerceptionSystem(agent.id, env, TimeIntervalAsyncLoopLock(TIME_INTERVAL)),
        MyCommunicationSender(agent.id, dest_ip="127.0.0.1", dest_port=9999),
    )

    ###########################################################
    # 6. Initialise the agent with some starting information  #
    ###########################################################
    awareness_vector = AwarenessVector(agent.id, np.zeros(8))
    knowledge_database = MyKnowledgeDatabase(
        goal_pos=np.array([300, 300]),
        tolerance=15,
        next_agent_id=OTHER_AGENT,
        idle=SELF_IDLE,
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
