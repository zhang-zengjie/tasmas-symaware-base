import argparse
import importlib
import logging
import os
from typing import TYPE_CHECKING

import numpy as np

from symaware.base.components import (
    DefaultCommunicationReceiver,
    DefaultCommunicationSender,
    DefaultController,
    DefaultPerceptionSystem,
    DefaultRiskEstimator,
    DefaultUncertaintyEstimator,
)
from symaware.base.data import AwarenessVector, KnowledgeDatabase, SymawareConfig
from symaware.base.utils import TimeIntervalAsyncLoopLock, initialize_logger

from .agent import Agent
from .agent_coordinator import AgentCoordinator

try:
    from symaware.base._version import __version__
except ImportError:
    __version__ = "0.0.0"

if TYPE_CHECKING:
    from typing import Sequence

    class CLIArgs(argparse.Namespace):
        """
        Type hinting for the command line arguments
        """

        version: str
        config_file: str
        num_agents: int
        agent_entity_idx: int
        verbose: int
        time_step: float
        disable_async: bool


VERBOSITY_TO_LOG_LEVEL = {0: logging.CRITICAL, 1: logging.ERROR, 2: logging.WARNING, 3: logging.INFO, 4: logging.DEBUG}
DEFAULT_INTERVAL = 0.5


def parse_args(args: "Sequence[str] | None" = None) -> "CLIArgs":
    """
    Parse the command line arguments

    Args
    ----
    args:
        A sequence of strings representing the command line arguments.
        If None, the arguments are taken from sys.argv

    Returns
    -------
        data structure containing the command line arguments
    """

    def file_exists(path: str) -> str:
        if path != "" and not os.path.exists(path):
            raise argparse.ArgumentTypeError(f"File '{path}' does not exist")
        return path

    parser = argparse.ArgumentParser(description="SymAware Base")
    parser.add_argument("--version", action="version", version=__version__)
    parser.add_argument("-c", "--config_file", help="Path to the configuration file", default="", type=file_exists)
    # Override the default values of the configuration
    parser.add_argument("-n", "--num-agents", help="Number of agents in the simulation", type=int, default=1)
    parser.add_argument(
        "-v",
        "--verbose",
        help="Increase the verbosity of the logs with the number of times the flag appears. "
        "E.g. -v = ERROR | -vvvv = DEBUG",
        action="count",
        default=0,
    )
    parser.add_argument(
        "--disable-async",
        help="Agents run asynchronously by default. This flag disables it and runs the agents synchronously.",
        default=False,
        action="store_true",
    )
    parser.add_argument("-t", "--time-step", help="Time step of the simulation", type=float, default=0.05)
    return parser.parse_args(args)  # type: ignore


def configure_symaware(num_agents: int, time_step: float, disable_async: bool) -> SymawareConfig:
    """
    Default configuration for the symaware simulation

    Args
    ----
    num_agents:
        Number of agents in the simulation
    time_step:
        Time step of the simulation
    disable_async:
        If True, the agents run synchronously

    Returns:
        Configuration for the symaware simulation
    """
    # pylint: disable=import-outside-toplevel
    try:
        from symaware.base.simulators.pybullet import (
            PyBulletEnvironment,
            PyBulletRacecar,
            PybulletRacecarModel,
        )
    except ImportError as e:
        raise ImportError(
            "symaware-pybullet non found. "
            "Try running `pip install symaware-pybullet` or `pip install symaware[simulators]`"
        ) from e

    def time_loop_lock() -> "TimeIntervalAsyncLoopLock | None":
        return TimeIntervalAsyncLoopLock(time_step) if not disable_async else None

    # Environment
    env = PyBulletEnvironment(time_step if disable_async else 0, async_loop_lock=time_loop_lock())
    # Agents
    entities = [PyBulletRacecar(i, model=PybulletRacecarModel(i)) for i in range(num_agents)]
    for i, entity in enumerate(entities):
        entity.position = np.array([i, i, 0.1])
    agents = [Agent[KnowledgeDatabase](i, entities[i]) for i in range(num_agents)]
    env.add_agents(agents)

    return SymawareConfig(
        agent=agents,
        knowledge_database=[{i: KnowledgeDatabase()} for i in range(num_agents)],
        awareness_vector=[{i: AwarenessVector(i, np.zeros(7))} for i in range(num_agents)],
        environment=env,
        controller=[DefaultController(i, time_loop_lock()) for i in range(num_agents)],
        risk_estimator=[DefaultRiskEstimator(i, time_loop_lock()) for i in range(num_agents)],
        uncertainty_estimator=[DefaultUncertaintyEstimator(i, time_loop_lock()) for i in range(num_agents)],
        communication_receiver=[DefaultCommunicationReceiver(i, time_loop_lock()) for i in range(num_agents)],
        communication_sender=[DefaultCommunicationSender(i, time_loop_lock()) for i in range(num_agents)],
        perception_system=[DefaultPerceptionSystem(i, env, time_loop_lock()) for i in range(num_agents)],
    )


def main():
    ###########################################################
    # 0. Parse the command line and setup logging             #
    ###########################################################
    args = parse_args()
    if args.verbose in VERBOSITY_TO_LOG_LEVEL:
        initialize_logger(VERBOSITY_TO_LOG_LEVEL[args.verbose])

    ###########################################################
    # 1. Load the configuration                               #
    ###########################################################
    if args.config_file != "":
        res = importlib.import_module(args.config_file.replace(".py", "").replace("/", "."))
        if not hasattr(res, "configure_symaware"):
            raise ValueError("The configuration file must contain a function called 'configure_symaware'")
        config: SymawareConfig = res.configure_symaware(args.num_agents, args.time_step, args.disable_async)
    else:
        config = configure_symaware(args.num_agents, args.time_step, args.disable_async)
    assert (
        len(config["agent"])
        == len(config["controller"])
        == len(config["knowledge_database"])
        == len(config["awareness_vector"])
        == len(config["risk_estimator"])
        == len(config["uncertainty_estimator"])
        == len(config["perception_system"])
        == len(config["communication_receiver"])
        == len(config["communication_sender"])
    ), "The number of agents, components and info must be the same"

    ###########################################################
    # For each agent in the simulation...                     #
    ###########################################################
    initialise_info = {}
    agents: list[Agent] = []
    for i, agent in enumerate(config["agent"]):

        ###########################################################
        # 2. Add the components in the agent                      #
        ###########################################################
        for component in (
            config["controller"][i],
            config["risk_estimator"][i],
            config["uncertainty_estimator"][i],
            config["perception_system"][i],
            config["communication_receiver"][i],
            config["communication_sender"][i],
        ):
            if component is not None:
                agent.add_components(component)  # type: ignore
        initialise_info[agent.id] = (config["awareness_vector"][i], config["knowledge_database"][i])
        agents.append(agent)

    ###########################################################
    # 3. Initialise the agent and run the simulation          #
    ###########################################################
    env = config["environment"]
    agent_coordinator = AgentCoordinator[KnowledgeDatabase](env, agents)
    if args.disable_async:
        agent_coordinator.run(time_step=args.time_step, initialise_info=initialise_info)
    else:
        agent_coordinator.async_run(initialise_info=initialise_info)


if __name__ == "__main__":
    main()
