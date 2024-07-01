import numpy as np
import asyncio
from typing import Iterable
from dataclasses import dataclass

from symaware.base import (
    Agent,
    AgentCoordinator,
    AwarenessVector,
    Controller,
    KnowledgeDatabase,
    TimeIntervalAsyncLoopLock,
    TimeSeries,
    get_logger,
    initialize_logger,
    log,
    Identifier,
    PerceptionSystem,
    StateObservation,
    EventAsyncLoopLock,
    DefaultAsyncLoopLock,
    InfoMessage,
    CommunicationSender,
    Message,
    CommunicationReceiver,
)

try:
    from symaware.simulators.pybullet import (
        Environment,
        DroneRacerEntity,
        DroneCf2pEntity,
        DroneCf2xEntity,
        DroneRacerModel,
        DroneCf2xModel,
        DroneModel,
    )
except ImportError as e:
    raise ImportError(
        "symaware-pybullet non found. "
        "Try running `pip install symaware-pybullet` or `pip install symaware[simulators]`"
    ) from e


class MyKnowledgeDatabase(KnowledgeDatabase):
    pass


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
    position: np.ndarray


class MyCommunicationSender(CommunicationSender[EventAsyncLoopLock]):
    __LOGGER = get_logger(__name__, "MyCommunicationSender")

    def __init__(self, agent_id: Identifier):
        super().__init__(agent_id, EventAsyncLoopLock())

    @log(__LOGGER)
    def _send_communication_through_channel(self, message: Message):
        self.__LOGGER.info(
            "Agent %d: Sending the message %s to the agent %d", self.agent_id, message, message.receiver_id
        )
        if message.receiver_id in MyCommunicationReceiver.message_queue:
            MyCommunicationReceiver.message_queue[message.receiver_id].put_nowait(message)
        else:
            self.__LOGGER.warning("Agent %d: Message %s could not be sent", self.agent_id, message)


class MyCommunicationReceiver(CommunicationReceiver[DefaultAsyncLoopLock]):
    __LOGGER = get_logger(__name__, "MyCommunicationReceiver")
    message_queue: dict[Identifier, asyncio.Queue[InfoMessage]] = {}

    def __init__(self, agent_id: int, min_distance_to_explode: float):
        super().__init__(agent_id, DefaultAsyncLoopLock())
        self.message_queue.setdefault(agent_id, asyncio.Queue())
        self._get_message_task: asyncio.Task
        self._min_distance_to_explode = min_distance_to_explode

    @log(__LOGGER)
    def _receive_communication_from_channel(self) -> Iterable[Message]:
        raise NotImplementedError("This communication system can only be used asynchronously")

    @log(__LOGGER)
    def _decode_message(self, messages: tuple[MyMessage]) -> "np.ndarray | None":
        if len(messages) == 0 or messages[0].receiver_id != self.agent_id:
            return None
        return messages[0].position

    @log(__LOGGER)
    async def _async_receive_communication_from_channel(self) -> Iterable[Message]:
        self.__LOGGER.info("Agent %d: Waiting for a message", self.agent_id)
        try:
            self._get_message_task = asyncio.create_task(self.message_queue[self.agent_id].get())
            message: Message = await self._get_message_task
            self.__LOGGER.info(
                "Agent %d: Received the message %s from agent %d", self.agent_id, message, message.sender_id
            )
            return (message,)
        except asyncio.CancelledError:
            self.__LOGGER.info("Agent %d: Stopping waiting for new messages", self.agent_id)
            return tuple()

    def _update(self, pos: "np.ndarray | None"):
        if pos is None or len(self._agent.self_state) < 3:
            return
        self_pos = self._agent.self_state[:3]
        # calculate the distance between the agent and the sender
        distance = np.linalg.norm(self_pos - pos)
        self.__LOGGER.info("Agent %d: Distance to the sender: %f", self.agent_id, distance)
        controller = self._agent.controller
        assert isinstance(controller, MyController)
        controller.set_offset((1, 1, 0, 0) if distance < self._min_distance_to_explode else (0, 0, 0, 0))

    async def async_stop(self):
        self._get_message_task.cancel()
        return await super().async_stop()


class MyController(Controller):
    """
    Simple controller that keeps the drone at a fixed altitude.
    """

    __LOGGER = get_logger(__name__, "SimpleController")

    def __init__(
        self,
        agent_id,
        offset: tuple[float, float, float, float] = (0, 0, 0, 0),
        async_loop_lock: TimeIntervalAsyncLoopLock | None = None,
    ):
        super().__init__(agent_id, async_loop_lock)
        self._offset = offset

    def set_offset(self, offset: tuple[float, float, float, float]):
        self._offset = offset

    @log(__LOGGER)
    def _compute(self) -> tuple[np.ndarray, TimeSeries]:
        """
        Compute the control input to be applied to the agent.
        The control input represents the rpm of each of the four rotors of the drone.

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
        assert isinstance(self._agent.model, DroneModel)
        # Normalized input in the range [-1, 1] where -1 means no thrust, 1 means full thrust
        normalized_control_input = np.array([0, 0, 0, 0]) + np.array(self._offset)
        control_input = self._agent.model.normalized_control_input_to_rpm(normalized_control_input)
        self.__LOGGER.debug("Control input: %s", control_input)
        return control_input, TimeSeries()


def send_message_callback(agent: Agent, new_state: dict[Identifier, StateObservation]):
    agent.communication_sender.enqueue_messages(MyMessage(agent.id, (1 - agent.id), new_state[agent.id].state[:3]))
    agent.communication_sender.async_loop_lock.trigger()


def main():
    ###########################################################
    # 0. Parameters                                           #
    ###########################################################
    TIME_INTERVAL = 0.01
    LOG_LEVEL = "INFO"
    MIN_DISTANCE_TO_EXPLODE = 1

    initialize_logger(LOG_LEVEL)

    entities: tuple[DroneCf2xEntity, DroneCf2pEntity, DroneRacerEntity] = (
        DroneCf2xEntity(0, model=DroneCf2xModel(0), position=np.array([1, 1, 2])),
        DroneRacerEntity(1, model=DroneRacerModel(1), position=np.array([0, 0, 2])),
    )

    ###########################################################
    # 1. Create the environment and add the obstacles         #
    ###########################################################
    env = Environment(async_loop_lock=TimeIntervalAsyncLoopLock(TIME_INTERVAL))
    # env.add_entities((DroneCf2pEntity(position=np.array([1, 1, 2])),))
    # env.add_entities((DroneCf2xEntity(position=np.array([2, 2, 2])),))
    # env.add_entities((DroneRacerEntity(position=np.array([0, 0, 2])),))

    ###########################################################
    # For each agent in the simulation...                     #
    ###########################################################
    agent_coordinator = AgentCoordinator[MyKnowledgeDatabase](env)
    for i, entity in enumerate(entities):
        ###########################################################
        # 2. Create the agent and assign it an entity             #
        ###########################################################
        agent = Agent[MyKnowledgeDatabase](i, entity)

        ###########################################################
        # 3. Add the agent to the environment                     #
        ###########################################################
        env.add_agents(agent)

        ###########################################################
        # 4. Create and set the component of the agent            #
        ###########################################################
        # In this example, all components run at the same frequency
        perception = MyPerceptionSystem(agent.id, env, TimeIntervalAsyncLoopLock(TIME_INTERVAL))
        perception.add_on_computed(send_message_callback)
        agent.add_components(
            perception,
            MyController(agent.id, async_loop_lock=TimeIntervalAsyncLoopLock(TIME_INTERVAL)),
            MyCommunicationSender(agent.id),
            MyCommunicationReceiver(agent.id, MIN_DISTANCE_TO_EXPLODE),
        )

        ###########################################################
        # 5. Initialise the agent with some starting information  #
        ###########################################################
        agent.initialise_agent(AwarenessVector(agent.id, np.zeros(7)), {agent.id: MyKnowledgeDatabase()})

        ###########################################################
        # 6. Add the agent to the coordinator                     #
        ###########################################################
        agent_coordinator.add_agents(agent)

    ###########################################################
    # 7. Run the simulation                                   #
    ###########################################################
    agent_coordinator.async_run()


if __name__ == "__main__":
    main()
