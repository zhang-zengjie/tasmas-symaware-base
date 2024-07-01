"""
Base package for SymAware.
It contains the basic components and classes to create an agent and its environment.
Most of the classes are abstract and need to be implemented by the user, 
although some default implementations are provided.

To run a simulation using the package's own main function,
the user needs to create a configuration python file 
that contains a function called :func:`configure_symaware`.
The function must return a :class:`.SymawareConfig` object.

Example
-------
Given a configuration file called `my_config.py` which contains a function like the following:

>>> # Import all the classes that will be used in the configuration
>>> from symaware.base import SymawareConfig  # ... other imports
>>> def configure_symaware(num_agents: int, time_step: float, disable_async: bool) -> SymawareConfig:
...
...    def time_loop_lock() -> "TimeIntervalAsyncLoopLock | None":
...        return TimeIntervalAsyncLoopLock(time_step) if not disable_async else None
...
...    #############################################################
...    # 1. Create the environment and add any number of obstacles #
...    #############################################################
...    env = PyBulletEnvironment(real_time_interval=time_step if disable_async else 0, async_loop_lock=time_loop_lock())
...    env.add_entities((PybulletBox(position=np.array([2, 2, 2])), PybulletSphere(position=np.array([-4, -4, 2]))))
...
...    #############################################################
...    # 2. Create the agents and add them to the environment      #
...    #############################################################
...    entities = [PyBulletRacecar(i, model=PybulletRacecarModel(i)) for i in range(num_agents)]
...    for i, entity in enumerate(entities):
...        entity.position = np.array([i, i, 0.1])
...    agents = [Agent[MyKnowledgeDatabase](i, entities[i]) for i in range(num_agents)]
...    env.add_agents(agents)
...
...    #############################################################
...    # 3. Create the knowledge database and the awareness vector #
...    #############################################################
...    knowledge_databases = [
...        {
...            i: MyKnowledgeDatabase(
...                goal_reached=False,
...                goal_pos=np.array([2, 2]),
...                tolerance=0.1,
...                speed=20 + 10 * i,
...                steering_tolerance=0.01,
...            )
...        }
...        for i in range(num_agents)
...    ]
...    awareness_vectors = [AwarenessVector(i, np.zeros(7)) for i in range(num_agents)]
...
...    #############################################################
...    # 4. Return the configuration                               #
...    #############################################################
...    return SymawareConfig(
...        agent=agents,
...        controller=[MyController(i, time_loop_lock()) for i in range(num_agents)],
...        knowledge_database=knowledge_databases,
...        awareness_vector=awareness_vectors,
...        risk_estimator=[DefaultRiskEstimator(i, time_loop_lock()) for i in range(num_agents)],
...        uncertainty_estimator=[DefaultUncertaintyEstimator(i, time_loop_lock()) for i in range(num_agents)],
...        communication_system=[
...            DefaultCommunicationSystem(i, time_loop_lock(), time_loop_lock()) for i in range(num_agents)
...        ],
...        info_updater=[DefaultInfoUpdater(i, time_loop_lock()) for i in range(num_agents)],
...        perception_system=[DefaultPerceptionSystem(i, env, time_loop_lock()) for i in range(num_agents)],
...        environment=env,
...    )
"""

from .agent import *
from .agent import Agent
from .agent_coordinator import AgentCoordinator
from .components import *
from .data import *
from .models import *
from .utils import *
