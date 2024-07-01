# Software design

## High level goal

The goal is to create a collection python packages an user can easily install on their machine.

The software must allow for different component implementations to be swapped easily.

<!-- .element: class="fragment" -->

<!-- New section -->

## Project structure

```mermaid
%%{ init : { "flowchart" : { "curve" : "basis" }}}%%
flowchart LR
    env_input{{External inputs from environment}}
    agent_input{{External inputs from another agent}}

    subgraph agent_j[Agent j]
        direction TB
        transmitted_output([Received Communication])
    end

    subgraph human[Human agent]
        direction LR
        goal([Goal])
        preferences([User preferences])
    end

    subgraph agent_i[Agent i]
        direction LR
        perceptual_information[Perceptual Information]
        knowledge[Knowledge]
        received_communication[Received Communication]
        chosen_action[Chosen Action]
        physical_state[Physical State of the system]
        communication_human[Communication Interface with Human Agent]
        communication_agent[Communication Interface with Agent j]

        subgraph situational_awareness[Situational Awareness]
            direction LR
            state([State])
            intent([Intent])
            uncertainty([Uncertainty])
            risk([Risk])
        end
    end

    env_input --> perceptual_information
    agent_input --> received_communication

    received_communication --> knowledge
    received_communication --> situational_awareness

    perceptual_information --> knowledge
    perceptual_information --> communication_agent
    perceptual_information --> situational_awareness

    knowledge --> situational_awareness
    knowledge --> communication_agent

    situational_awareness --> communication_agent
    situational_awareness --> chosen_action
    situational_awareness --> communication_human

    chosen_action --> communication_agent
    chosen_action --> communication_human
    chosen_action --> physical_state

    physical_state --> perceptual_information

    communication_human --> human
    communication_agent --> agent_j

classDef yellow stroke:#50623A,stroke-width:1px
classDef red stroke:red,stroke-width:1px
classDef green stroke:green,stroke-width:1px
classDef blue stroke:blue,stroke-width:1px
classDef background fill:#00000015

class state,intent,uncertainty,risk,situational_awareness,knowledge red;
class human,communication_human,goal,preferences green;
class agent_j,communication_agent,transmitted_output,agent_input,received_communication blue;
class env_input,perceptual_information,physical_state yellow;
style chosen_action stroke:magenta,stroke-width:1px;

class agent_i,situational_awareness,human,agent_j background;
```

<!-- New section -->

## Package architecture

All packages will be namespaced under the `symaware` namespace.
From the `symaware.base` package, each team will then develop their own implementation of one or more elements of the system.

```mermaid
---
title: Package architecture from the prospective of the user
---
flowchart RL
    subgraph symawarep["symaware (namepsace)"]
        symaware["symaware.symaware"]
    end

classDef background fill:#00000015
class symawarep background;
```

<!-- .element: class="fragment fade-in-then-out m-unset" -->

```mermaid
---
title: The base package provides the abstract interface of the system
---
flowchart RL
    subgraph symawarep["symaware (namepsace)"]
        base[symaware.base]
        symaware[symaware.symaware]
    end

    symaware --> base

classDef background fill:#00000015
class symawarep background;
```

<!-- .element: class="fragment fade-in-then-out m-unset" -->

```mermaid
---
title: |
    Each team will build their own implementation of some elements of the system
    for them to be combined together in the public package
---
flowchart RL
    subgraph symawarep["symaware (namepsace)"]
        base[symaware.]
        mpi[symaware.mpi]
        kth[symaware.kth]
        tue[symaware.tue]
        uu[symaware.uu]
        nlr[symaware.nlr]
        sisw[symaware.sisw]
        base[symaware.base]
        symaware[symaware.symaware]
    end

    mpi --> base
    kth --> base
    tue --> base
    uu --> base
    nlr --> base
    sisw --> base
    symaware --> mpi
    symaware --> kth
    symaware --> tue
    symaware --> uu
    symaware --> nlr
    symaware --> sisw

classDef background fill:#00000015
class symawarep background;
```

<!-- .element: class="fragment fade-in-then-out m-unset" -->

<!-- New section -->

### Software design of `symaware.base`

The main elements of the software have been divided in subpackages to enforce a coarse but clear separation of concerns.

```mermaid
---
title: Explicit dependencies
---
flowchart TB
    user{{User}}
    subgraph base["symaware.base"]
        direction TB
        agent([base.Agent])
        simulators[base.simulators]
        components[base.components]
        models[base.models]
        utils[base.utils]
        data[base.data]
    end

user --> agent
user --> simulators
agent --> models
agent --> components
agent --> data
simulators --> components
simulators --> models
simulators --> data
components --> utils
components --> models
components --> data
models --> utils
models --> data

classDef background fill:#00000015
classDef yellow stroke:#50623A,stroke-width:1px
classDef red stroke:red,stroke-width:1px
classDef green stroke:green,stroke-width:1px
classDef blue stroke:blue,stroke-width:1px
classDef orange stroke:orange,stroke-width:1px
classDef magenta stroke:magenta,stroke-width:1px

class simulators red;
class agent orange;
class components green;
class models blue;
class utils yellow;
class data magenta;
class base background;
```

<!-- .element: class="fragment fade-in-then-out m-unset" -->

```mermaid
---
title: Assuming transitive dependencies
---
flowchart TB
    user{{User}}
    subgraph base["symaware.base"]
        direction TB
        agent([base.Agent])
        simulators[base.simulators]
        components[base.components]
        models[base.models]
        data[base.data]
        utils[base.utils]
    end

user --> agent
user --> simulators
agent --> components
simulators --> components
components --> models
models --> data
models --> utils

classDef background fill:#00000015
classDef yellow stroke:#50623A,stroke-width:1px
classDef red stroke:red,stroke-width:1px
classDef green stroke:green,stroke-width:1px
classDef blue stroke:blue,stroke-width:1px
classDef orange stroke:orange,stroke-width:1px
classDef magenta stroke:magenta,stroke-width:1px

class simulators red;
class agent orange;
class components green;
class models blue;
class utils yellow;
class data magenta;
class base background;
```

<!-- .element: class="fragment fade-in-then-out m-unset" -->

```mermaid
---
title: Explicit dependencies
---
flowchart TB
    user{{User}}
    subgraph base["symaware.base"]
        direction TB
        agent([base.Agent])
        subgraph simulators[base.simulators]
            direction TB
            simulator_pybullet([simulators.pybullet])
            simulator_pymunk([simulators.pymunk])
        end
        subgraph components[base.components]
            direction TB
            controller([components.Controller])
            perception_system([components.PerceptionSystem])
            communication_system([components.CommunicationSystem])
            risk_evaluator([components.RiskEvaluator])
            uncertainty_evaluator([components.UncertaintyEvaluator])
        end
        subgraph models[base.models]
        direction TB
            dynamical_model([models.DynamicModel])
            environment([models.Environment])
            entity([models.Entity])
        end
        subgraph utils[base.utils]
            direction TB
            logger[utils.log]
        end
        subgraph data[base.data]
            direction TB
            knowledge([data.Knowledge])
            awareness_vector([data.AwarenessVector])
        end
    end

user --> agent
user --> simulators
agent --> components
simulators --> components
components --> models
models --> data
models --> utils


classDef background fill:#00000015
classDef yellow stroke:#50623A,stroke-width:1px
classDef red stroke:red,stroke-width:1px
classDef green stroke:green,stroke-width:1px
classDef blue stroke:blue,stroke-width:1px
classDef orange stroke:orange,stroke-width:1px
classDef magenta stroke:magenta,stroke-width:1px

class simulators red;
class agent orange;
class components green;
class models blue;
class utils yellow;
class data magenta;
class base,simulators,components,models,utils,data background;
```

<!-- .element: class="fragment fade-in-then-out m-unset" -->

<!-- New subsection -->

### Sequence diagram

The following sequence diagram shows the interaction between the different components of the system.

```mermaid
sequenceDiagram
    participant p as Perception System
    participant cs as Communication System
    participant a as Agent
    participant ru as Risk Evaluator<br>Uncertainty Evaluator
    participant c as Controller

p ->> a: Perceptual Information
cs ->> a: Received Communication
note over a: InfoUpdater<br>State = Awareness + Knowledge
a ->> ru: Current state
ru ->> a: Risk/Uncertainty
a ->> c: Updated state
c ->> a: Chosen action
a ->> cs: Updated state
```

<!-- New section -->

## Asynchronous model

Instead of relying in a strict sequence of events, the system is designed to be asynchronous ([asyncio](https://docs.python.org/3/library/asyncio.html) is used for this purpose).

Each component is independent and can run concurrently with the others, with its own fire frequency or event trigger.

<!-- .element: class="fragment" -->

Most of the added complexity is hidden in the `symaware.base`.
Components can be developed only using standard, synchronous code.

<!-- .element: class="fragment" -->

<!-- New subsection -->

### AsyncLoopLock

The `AsyncLoopLock` class determines how often the component will run.

- `TimeIntervalAsyncLoopLock` runs the component at a fixed interval
- `EventAsyncLoopLock` runs the component when a specific event is triggered
- `DefaultAsyncLoopLock` the component will ruu continuously. Needs to be used in combination with a custom lock mechanism

<!-- .element: class="fragment" -->

See the [AsyncLoopLock documentation](https://sadegh.pages.mpi-sws.org/eicsymaware/api/symaware.base.utils.html#symaware.base.utils.async_loop_lock.AsyncLoopLock) or the [example](https://gitlab.mpi-sws.org/sadegh/eicsymaware/-/blob/base/examples/messages_lib.py) for more information.

<!-- .element: class="fragment" -->

<!-- New section -->

## Extending the system

The system is designed to be easily extensible.

There are two core aspects to this:

<!-- .element: class="fragment" data-fragment-index="1" -->

- **Adding new components**: components determine the behavior of the agent
- **Adding new models**: models simulate the environment and the physical state of the system

<!-- .element: class="fragment" data-fragment-index="1" -->

<!-- New subsection -->

### Adding new components

To add a new component, you must define a new class that inherits from the specific component they want to extend, which in turns inherits from `symaware.base.components.Component`.

The new component must implement its specific behavior in the abstract method the superclass provides.

<!-- .element: class="fragment" -->

For more information and examples, see the [Component documentation](https://sadegh.pages.mpi-sws.org/eicsymaware/api/symaware.base.components.html) or the [components subpackage](https://gitlab.mpi-sws.org/sadegh/eicsymaware/-/tree/base/src/symaware/base/components).

<!-- .element: class="fragment" -->

<!-- New subsection -->

#### Example: adding a new controller

```python
from symaware.base import Controller
class MyController(Controller):
    def __init__(self, agent_id, async_loop_lock = None):
        super().__init__(agent_id, async_loop_lock)
        self._control_input = np.zeros(0)

    def initialise_component(self, agent, initial_awareness_database, initial_knowledge_database):
        # Custom initialisation
        super().initialise_component(agent, initial_awareness_database, initial_knowledge_database)
        self._control_input = np.zeros(agent.model.control_input_shape)

    def _compute_control_input(self, awareness_database, knowledge_database):
        # Controller specific implementation
        return self._control_input, TimeSeries()
```

<!-- New subsection -->

#### Example: minimal controller

```python
from symaware.base import Controller

class MyController(Controller):

    def _compute_control_input(self, awareness_database, knowledge_database):
        # Controller specific implementation
        return np.zeros(self._agent.model.control_input_shape), TimeSeries()
```

<!-- New subsection -->

### Adding new models

Adding a new model is slightly more involved

It requires at least tree classes to be defined:

<!-- .element: class="fragment" data-fragment-index="1" -->

- `Environment`: the environment in which the agent operates
- `Entity`: the entities that populate the environment
- `DynamicModel`: the dynamic model of the entity

<!-- .element: class="fragment" data-fragment-index="1" -->

For more information and examples, see the [Model documentation](https://sadegh.pages.mpi-sws.org/eicsymaware/api/symaware.base.models.html) or the [simulators subpackage](https://gitlab.mpi-sws.org/sadegh/eicsymaware/-/blob/base/src/symaware/base/simulators).

<!-- .element: class="fragment" data-fragment-index="2" -->

<!-- New subsection -->

#### Example: adding a new environment

```python
from symaware.base import Environment
class PyBulletEnvironment(Environment):
    def __init__(self, async_loop_lock = None):
        super().__init__(async_loop_lock)
        self._initialise_pybullet()

    def _initialise_pybullet(self):
        p.connect(p.GUI)
        # ... more initialisation code

    def get_entity_state(self, entity: Entity) -> np.ndarray:
        return np.array(p.getBasePositionAndOrientation(entity.entity_id))

    def _add_entity(self, entity: Entity):
        entity.initialise()

    def step(self):
        for entity in self._agent_entities.values():
            entity.step()
        p.stepSimulation()
```

<!-- New subsection -->

#### Example: adding a new entity

```python
from symaware.base import Entity
@dataclass
class PybulletSphere(Entity):
    model: PybulletDynamicalModel = field(default_factory=NullDynamicalModel)
    pos: np.ndarray
    angle: np.array
    radius: float

    def initialise(self):
        col_id = p.createCollisionShape(p.GEOM_SPHERE, radius=self.radius)
        vis_id = p.createVisualShape(p.GEOM_SPHERE, radius=self.radius)
        entity_id = p.createMultiBody(1, col_id, vis_id, self.pos, self.angle)
        if not isinstance(self.model, NullDynamicalModel):
            self.model.initialise(entity_id)
```

<!-- New subsection -->

#### Example: adding a new dynamic model

```python
from symaware.base import DynamicModel
class PybulletRacecarModel(DynamicModel):
    def __init__(self, ID, max_force):
        super().__init__(ID, control_input=np.zeros(2), state=np.zeros(7))
    @property
    def subinputs_dict(self) -> PybulletRacecarModelSubinputs:
        return {"velocity": self.control_input[0], "angle": self.control_input[1]}

    def initialise(self, entity_id: int):
        self._entity_id = entity_id

    def step(self):
        target_velocity, steering_angle = self._control_input
        # Just steer the front wheels
        for steer in (0, 2):
            p.setJointMotorControl2(self._entity_id, steer, p.POSITION_CONTROL, targetPosition=steering_angle)
```

<!-- New section -->
