# 🤖 SymAware: Agent Simulation Base Model 🚀

[![MIT License](https://img.shields.io/badge/license-BSD3-green)](https://gitlab.mpi-sws.org/sadegh/eicsymaware/-/blob/base/LICENSE)
[![Python Version](https://img.shields.io/badge/python-3.9+-blue)](https://www.python.org/downloads/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![CI/CD](https://gitlab.mpi-sws.org/sadegh/eicsymaware/badges/base/pipeline.svg)](https://gitlab.mpi-sws.org/sadegh/eicsymaware/-/commits/base)
[![Test coverage](https://gitlab.mpi-sws.org/sadegh/eicsymaware/badges/base/coverage.svg)](https://gitlab.mpi-sws.org/sadegh/eicsymaware/-/commits/base)
[![symaware.base](https://gitlab.mpi-sws.org/sadegh/eicsymaware/-/badges/release.svg)](https://gitlab.mpi-sws.org/sadegh/eicsymaware/-/packages)
[![Documentation](https://img.shields.io/badge/Documentation-sphinx-purple)](https://sadegh.pages.mpi-sws.org/eicsymaware/)

## Introduction

Welcome to SymAware, the Symbolic Logic Framework for Situational Awareness in Mixed Autonomy, a cutting-edge project funded by the European Union.
Imagine an environment where autonomous agents - be it robots, drones, or cars - collaborate seamlessly with humans to complete complex, dynamically evolving tasks.
SymAware is designed to equip these agents with the crucial capability of obtaining situational awareness and enhancing their risk perception abilities.
This base model is pivotal for creating sustainable, real-world autonomy in mixed human-agent environments.

## Overview

This repository contains code of the `symaware.base` package, the foundation used to develop and run simulations for multi-agent systems with different dynamical models using the framework developed for the SymAware project.
This repository is a work in progress.

### Structure

The repository is structured as follows:

```bash
symaware-base/
├── .gitlab-ci.yml      # CI/CD configuration
├── Dockerfile          # Docker configuration
├── pyproject.toml      # Python package configuration
├── requirements.txt    # Python package requirements
├── tox.ini             # Dev environment configuration
├── README.md           # Project documentation
├── docs/               # Project documentation
├── examples/           # Example scripts
├── src/
│   └── symaware/       # Namespace package (symaware)
│       └── base/       # Base package (symaware.base)
└── tests/              # Test suite
```

## Installation

The package can either be installed from source or using the python package manager [pip](https://pypi.org/project/pip/).

### From Source

clone the repository and run the following command in the root directory:

```bash
# Clone the repository
git clone --branch base git-rts@gitlab.mpi-sws.org:sadegh/eicsymaware.git
# Change directory
cd eicsymaware
# Install the package (optionally use a virtual environment)
pip install . .[simulators]
```

### With pip

```bash
pip install symaware-base --index-url https://gitlab.mpi-sws.org/api/v4/projects/2668/packages/pypi/simple
```

## Additional Requirements

Based on what the user wants to do, additional requirements may require installation alongside the package.

### From Source

```bash
# Run tests
pip install .[test]
# Lint the code
pip install .[lint]
# Run the examples with the provided simulators
pip install .[simulators]
```

### With pip

```bash
# Run the examples with the provided simulators
pip install symaware-base[simulators] --index-url https://gitlab.mpi-sws.org/api/v4/projects/2668/packages/pypi/simple
```

## Examples

The `examples` directory contains a bunch of scripts that demonstrate some common use-cases.

There are two ways of using this package:

- Including it in your own project as a library and importing the classes and functions you need (main)
- Running the main script of the package providing your own configuration script (configure)

In both cases, the `symaware.base` package needs to be [installed](#installation) along with the simulators.

```bash
# Run the examples (main)
python3 examples/simple_main.py
# Run the examples (configure)
python3 -m symaware.base examples/simple_configure.py
```

### Main Example

The [`simple_main.py`](./examples/simple_main.py) script demonstrates how to use the package as a library.

```python
# All the external import you need

from symaware.base import (
    # All the classes and functions you need fom the package
)
from symaware.base.simulators.pybullet import (
    # You can either import one of the already implemented simulators
    # or implement/import your own
)

class MyKnowledgeDatabase(KnowledgeDatabase):
    # Define your own knowledge database


class MyController(Controller):
    # Extend all the components you need in your simulation.
    # All of them have a reasonable default implementation

    # Optionally use of the provided logger functionality
    __LOGGER = get_logger(__name__, "SimpleController")

    @log(__LOGGER)
    def compute_control_input(
        self,
        awareness_database: MultiAgentAwarenessVector,
        knowledge_database: MultiAgentKnowledgeDatabase[MyKnowledgeDatabase],
    ) -> tuple[np.ndarray, TimeSeries]:
        # Check remaining distance
        ...
        # Compute the steering angle
        steering_angle = ...
        # Compute the speed
        speed = ...
        return ... # Tuple containing the ( Control input, Intent )

    @log(__LOGGER)
    def __get_euler_from_quaternion(self, Q: np.ndarray | tuple[float, float, float, float]) -> np.ndarray:
        # Covert a quaternion into a full three-dimensional rotation matrix.


def main():
    ###########################################################
    # 0. Parameters                                           #
    ###########################################################
    TIME_INTERVAL = 0.1
    NUM_AGENTS = 3
    ...

    ###########################################################
    # 1. Create the environment and add the obstacles         #
    ###########################################################
    env = PyBulletEnvironment(async_loop_lock=TimeIntervalAsyncLoopLock(TIME_INTERVAL))
    env.add_entities((PybulletBox(position=np.array([2, 2, 2])), PybulletSphere(position=np.array([-4, -4, 2]))))

    ###########################################################
    # For each agent in the simulation...                     #
    ###########################################################
    agent_coordinator = AgentCoordinator[MyKnowledgeDatabase](env)
    for i in range(NUM_AGENTS):
        ###########################################################
        # 2. Create the entity and model the agent will use       #
        ###########################################################
        agent_entity = PyBulletRacecar(i, model=PybulletRacecarModel(i), position=np.array([0, i, 0.1]))

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
        # In this example, all components run at the same frequency
        controller = MyController(agent.id, TimeIntervalAsyncLoopLock(TIME_INTERVAL))
        risk = DefaultRiskEstimator(agent.id, TimeIntervalAsyncLoopLock(TIME_INTERVAL))
        uncertainty = DefaultUncertaintyEstimator(agent.id, TimeIntervalAsyncLoopLock(TIME_INTERVAL))
        perception = DefaultPerceptionSystem(agent.id, env, TimeIntervalAsyncLoopLock(TIME_INTERVAL))
        communication = DefaultCommunicationSystem(
            agent.id, TimeIntervalAsyncLoopLock(TIME_INTERVAL), TimeIntervalAsyncLoopLock(TIME_INTERVAL)
        )
        info_updater = DefaultInfoUpdater(agent.id)
        agent.set_components(controller, risk, uncertainty, perception, communication, info_updater)

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
    agent_coordinator.run()


if __name__ == "__main__":
    main()
```

### Configure Example

The [`simple_configure.py`](./examples/simple_configure.py) script demonstrates how to leverage the package's main method by providing a custom configuration script.

```python
# All the external import you need

from symaware.base import (
    # All the classes and functions you need fom the package
)
from symaware.base.simulators.pybullet import (
    # You can either import one of the already implemented simulators
    # or implement/import your own
)

class MyKnowledgeDatabase(KnowledgeDatabase):
    # Define your own knowledge database


class MyController(Controller):
    # Extend all the components you need in your simulation.
    # All of them have a reasonable default implementation

    # Optionally use of the provided logger functionality
    __LOGGER = get_logger(__name__, "SimpleController")

    @log(__LOGGER)
    def compute_control_input(
        self,
        awareness_database: MultiAgentAwarenessVector,
        knowledge_database: MultiAgentKnowledgeDatabase[MyKnowledgeDatabase],
    ) -> tuple[np.ndarray, TimeSeries]:
        # Check remaining distance
        ...
        # Compute the steering angle
        steering_angle = ...
        # Compute the speed
        speed = ...
        return ... # Tuple containing the ( Control input, Intent )

    @log(__LOGGER)
    def __get_euler_from_quaternion(self, Q: np.ndarray | tuple[float, float, float, float]) -> np.ndarray:
        # Covert a quaternion into a full three-dimensional rotation matrix.


def configure_symaware(num_agents: int, time_step: float, disable_async: bool) -> SymawareConfig:
    # Configuration function with a specific signature that will be called from the packages.
    # It has to return a dictionary containing the configuration.

    def time_loop_lock() -> "TimeIntervalAsyncLoopLock | None":
        return TimeIntervalAsyncLoopLock(time_step) if not disable_async else None

    #############################################################
    # 1. Create the environment and add the obstacles           #
    #############################################################
    env = PyBulletEnvironment(real_time_interval=time_step if disable_async else 0, async_loop_lock=time_loop_lock())
    env.add_entities((PybulletBox(position=np.array([2, 2, 2])), PybulletSphere(position=np.array([-4, -4, 2]))))

    #############################################################
    # 2. Create the agents and add them to the environment      #
    #############################################################
    entities = [PyBulletRacecar(i, model=PybulletRacecarModel(i)) for i in range(num_agents)]
    for i, entity in enumerate(entities):
        entity.position = np.array([i, i, 0.1])
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
        risk_estimator=[DefaultRiskEstimator(i, time_loop_lock()) for i in range(num_agents)],
        uncertainty_estimator=[DefaultUncertaintyEstimator(i, time_loop_lock()) for i in range(num_agents)],
        communication_system=[
            DefaultCommunicationSystem(i, time_loop_lock(), time_loop_lock()) for i in range(num_agents)
        ],
        info_updater=[DefaultInfoUpdater(i, time_loop_lock()) for i in range(num_agents)],
        perception_system=[DefaultPerceptionSystem(i, env, time_loop_lock()) for i in range(num_agents)],
        environment=env,
    )
```

## Documentation

The documentation for this package is available [here](https://sadegh.pages.mpi-sws.org/eicsymaware/).

## Contacts

- [Gregorio Marchesini](mailto:gremar@kth.se)
- [Arabinda Ghosh](mailto:arabinda@mpi-sws.org)
- [Zengjie Zhang](mailto:z.zhang3@tue.nl)
- [Ernesto Casablanca](mailto:casablancaernesto@gmail.com)
