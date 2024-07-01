import numpy as np
import os, sys
from itertools import product
import gurobipy as gp
from gurobipy import GRB
import logging
from typing import TypeVar
import time
T = TypeVar('T')

root_path = os.getcwd()
sys.path.append(os.path.join(root_path, 'eicsymaware', 'src'))

from tasmas.utils.functions import PRT, calculate_probabilities, calculate_risks, checkout_largest_in_dict
from tasmas.probstlpy.systems.linear import LinearSystem
from tasmas.probstlpy.solvers.gurobi.gurobi_micp import GurobiMICPSolver as MICPSolver
from tasmas.config import agent_model, agent_specs

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


class TasMasKnowledge(KnowledgeDatabase):
    simulation_horizon: float
    tlg: list
    tll: list
    slg: list
    sll: list


class TasMasCoordinator(AgentCoordinator[T]):

    __LOGGER = get_logger(__name__, "TasMasController")

    def __init__(self, env, agents):
        super().__init__(env, agents)

        self.t = 0
        self.specs = None
        self.m = None
        self.m_range = None

        self.bidding_list = None
        self.price_list = None
        self.assignment_list = None

        self.model = gp.Model("AUCTION_IP")

    @log(__LOGGER)
    def initialise_specs(self, knowledge):

        self.horizon = knowledge['simulation_horizon']
        self.tlg = knowledge['tlg']
        self.tll = knowledge['tll']
        self.slg = knowledge['slg']                       # LG: the list of global specifications (slg) and its time list (tlg) 
        self.sll = knowledge['sll']                     # LG: the list of local specifications (sll) and its time list (tlg)


    @log(__LOGGER)
    def initialize_agents(self):

        self.n = len(self._agents)
        for agent in self._agents:
            assert(len(agent.controllers) == 1)
            agent.controllers[0].x = self.model.addMVar((self.n, ), vtype=GRB.BINARY, name="assignment")

    @log(__LOGGER)
    def update_specs(self, specs):
        self.specs = specs
        self.m = len(self.specs)
        self.m_range = [i for i in range(self.m)]
        self.clean_bidding_list()
        self.clean_price_list()
        self.clean_assignment_list()

    @log(__LOGGER)
    def clean_bidding_list(self):
        self.bidding_list = {agent._ID: [False for i in self.m_range] for agent in self._agents}

    @log(__LOGGER)
    def clean_price_list(self):
        self.price_list = {agent._ID: [1 for i in self.m_range] for agent in self._agents}

    @log(__LOGGER)
    def clean_assignment_list(self):
        self.assignment_list = {agent._ID: [False for i in self.m_range] for agent in self._agents}

    @log(__LOGGER)
    def select_agents(self):
        
        for spec_index in self.m_range:
            rho = {agent._ID: self.specs[spec_index].robustness(agent.controllers[0].xx, 0)[0] for agent in self._agents}
            selected_agents = checkout_largest_in_dict(rho, self.m)
            for id in selected_agents:
                self.bidding_list[id][spec_index] = True
        
    @log(__LOGGER)
    def auction(self):
        
        self.model.remove(self.model.getConstrs())
        p = 0
        sum = [0 for i in self.m_range]
        for agent in self._agents:
            for j in self.m_range:
                if self.bidding_list[agent._ID][j] & (self.price_list[agent._ID][j] < 1):
                    p += agent.controllers[0].x[j] * self.price_list[agent._ID][j]
                else:
                    self.model.addConstr(agent.controllers[0].x[j] == 0)
                sum[j] += agent.controllers[0].x[j]
            self.model.addConstr(agent.controllers[0].x.sum() <= 1)

        if self.n < self.m:
            for agent, j in product(self._agents, range(self.n, self.m)):
                self.model.addConstr(agent.controllers[0].x[j] == 0)

        for j in self.m_range:
            self.model.addConstr(sum[j] == 1)
        self.model.setObjective(p, GRB.MINIMIZE)
        self.model.optimize()

        if self.model.status == GRB.OPTIMAL:

            for agent in self._agents:
                for j in self.m_range:
                    self.assignment_list[agent._ID][j] = agent.controllers[0].x.X[j]

        else:
            logging.info("Assignment not feasible. Rejecting the new tasks!")

    @log(__LOGGER)
    def bidding(self, t):
        for agent in self._agents:
            for j in self.m_range:
                self.price_list[agent._ID][j] = agent.controllers[0].bid(t, self.specs[j]) if self.bidding_list[agent._ID][j] else 1
                logging.info("Agent " + str(agent._ID) + " has probed task " + self.specs[j].name + "!")

    @log(__LOGGER)
    def assign(self, t):
        for id, item in self.assignment_list.items():
            if 1 in item:
                self._agents[id].controllers[0].accept_task(t, self.specs[item.index(1)])

    @log(__LOGGER)
    def update_control(self, t):
        for agent in self._agents:
            agent.controllers[0].apply_control(t, agent.probe_task(t))

    @log(__LOGGER)
    def assign_global(self, t, specs_list):
        self.update_specs(specs_list)
        self.select_agents()
        self.bidding(t)
        self.auction()
        self.assign(t)

    @log(__LOGGER)
    def assign_local(self, t, specs_list):

        for id, spec in specs_list.items():
            sln = self._agents[id].controllers[0].probe_task(t, spec)
            if sln[-1] != GRB.OPTIMAL:
                self._agents[id].controllers[0].reject_task(t, spec)
            else:
                self._agents[id].controllers[0].accept_task(t, spec)

    @log(__LOGGER)
    def run(self, time_step: float, timeout: float = -1, initialise_info: "InitialisationInfo | None" = None):
        """
        Start running all the agents and the environment.
        All agents will begin running their components synchronously.
        The frequency at which each :class:`.Component` is executed is determined by the `time_step`.
        All agents that have not yet been initialised will be initialised before running, as long as the
        `initialise_info` contains the initialisation information for the agent,
        namely a tuple with the awareness vector and the knowledge database.

        Example
        -------
        The following example shows how to create an environment, agents and a coordinator.

        Args
        ----
        time_step:
            Time step representing the interval at which the agents and the environment will be updated
        timeout:
            Time in seconds after which the event loop will stop.
            If the timeout is negative, the event loop will run indefinitely
        initialise_info:
            Dictionary containing the initialisation information for each agent.
            The agent id is the key and the value is a tuple
            containing the awareness vector, the knowledge database and the time

        Raises
        ------
        Exception: An unforeseen exception was raised during execution
        """

        """""""""""""""""
        Initialize
        """""""""""""""""

        self.initialize_agents()
        self.t = 0
        
        for agent in self._agents:
            self._env.initialise()
            if agent.is_initialised or initialise_info is None or agent.id not in initialise_info:
                continue
            agent.initialise_agent(*initialise_info[agent.id])
            self.__LOGGER.debug("Agent %s has been initialised: %s", agent.id, initialise_info)
        self.__LOGGER.info("Initialised")
        if self._post_init is not None:
            self._post_init(self)
        try:
            starting_time = time.time()
            end_time = time.time()

            while timeout < 0 or end_time - starting_time < timeout:

                """""""""""""""
                Iteration
                """""""""""""""
                if self.t < self.horizon:

                    if self.t in self.tlg:
                        self.assign_global(self.t, self.slg[self.tlg.index(self.t)])
                    if self.t in self.tll:
                        self.assign_local(self.t, self.sll[self.tll.index(self.t)])

                    self._env.step()
                    for agent in self._agents:
                        agent.step(self.t)

                    self.t += 1

                else:
                    print('Maximal horizon reached ... standing by ...')

                time.sleep(time_step)
                end_time = time.time()
        except (KeyboardInterrupt, SystemExit):
            self.__LOGGER.info("Received stop signal. Stopping agents.")
        except Exception as exc:
            self.__LOGGER.error("An error occurred: %s", exc)
            raise exc
        finally:
            self._env.stop()
            if self._post_stop is not None:
                self._post_stop(self)


class TasMasAgentModel(KnowledgeDatabase):
    A: np.ndarray
    B: np.ndarray
    C: np.ndarray
    D: np.ndarray
    mu: np.ndarray
    Sigma: np.ndarray


class TasMasAgent(Agent[T]):

    __LOGGER = get_logger(__name__, "TasMasController")

    @log(__LOGGER)
    def initialise_agent(
        self,
        initial_awareness_database,
        initial_knowledge_database
    ):
        if isinstance(initial_awareness_database, AwarenessVector):
            initial_awareness_database = {self._ID: initial_awareness_database}

        if not self._ID in initial_awareness_database.keys():
            raise ValueError(
                "The agent ID must be present in the initial awareness database. "
                "This database also includes the state of the agent itself"
            )
        if self._ID not in initial_knowledge_database.keys():
            raise ValueError("The agent ID must be present in the initial knowledge database.")

        self._awareness_database = initial_awareness_database
        self._knowledge_database = initial_knowledge_database
        # Initialise all components
        for component in self.components:
            component.initialise_component(
                agent=self,
                initial_awareness_database=initial_awareness_database,
                initial_knowledge_database=initial_knowledge_database,
            )

        self._is_initialised = True

    @log(__LOGGER)
    def step(self, t):
        # Your implementation here
        # Example:
        # Compute all the components in the order they were added

        controller = self.controllers[0]
        controller._compute(t, controller.probe_task(t))


class TasMasController(Controller):
    """
    The tasmas controller by TU/e
    """

    __LOGGER = get_logger(__name__, "TasMasController")

    def __init__(self, agent_id, sys_model, spec, N, x0, Q, R, ub, async_loop_lock=None):
        super().__init__(agent_id, async_loop_lock)

        self.N = N              # Time-horizon
        self.psi = spec
        self.x0 = x0
        
        self.sys = None
        self.w = None  # uncomment to check with no disturbance

        self.Q = Q
        self.R = R

        # Compute 1) relation between tube and probability & 2) Stabilizing feedback gain
        self.construct_model(sys_model)
        self.diag_sigma_inf, self.K = PRT(self.sys, self.Q, self.R)

        # Initialize Memory
        self.xx = np.zeros([2, self.N + 1])
        self.zz = np.zeros([2, self.N + 1])

        self.xx[:, 0] = self.x0
        self.zz[:, 0] = self.x0

        self.z = np.zeros([2, self.N + 1])
        self.zh = []
        
        self.err = np.zeros([2, self.N + 1])
        self.vv = np.zeros([2, self.N])
        self.v = np.zeros([2, self.N])
        
        self.u_limits = np.array([[-ub, ub], [-ub, ub]])

        self.accept_meas = []

        # The lists of accepted specifications
        self.accept_phi = [spec]                                # Accepted specifications
        self.accept_prob = [np.ones((self.N, ))]                # Acceptance probabilities
        self.accept_time = [0]                                  # The time instants of acceptation

        self.risk = None
        self.x = None

        self.solver = MICPSolver(self.psi, self.sys, self.zz, self.N, self.diag_sigma_inf, uu=self.vv, riskH=self.risk, mark=0, verbose=False)
        self.solver.AddSTLConstraints(self.psi)
        self.solver.AddControlBounds(self.u_limits[:, 0], self.u_limits[:, 1])
        

    @log(__LOGGER)
    def construct_model(self, sys_model):
        A = sys_model[self.agent_id]['A']
        B = sys_model[self.agent_id]['B']
        C = sys_model[self.agent_id]['C']
        D = sys_model[self.agent_id]['D']
        mu = sys_model[self.agent_id]['mu']
        Sigma = sys_model[self.agent_id]['Sigma']

        self.sys = LinearSystem(A, B, C, D, mu, Sigma)
        self.w = np.random.multivariate_normal(mu, Sigma, self.N).T

    @log(__LOGGER)
    def update_measurement(self, t):
        self.zz[:, t] = self.xx[:, t]
        return None

    @log(__LOGGER)
    def update_probabilities(self, t):

        spec_prob = calculate_probabilities(self.accept_phi, self.risk)

        for i in range(len(self.accept_phi)):
            self.accept_prob[i][t] = spec_prob[i]
        return None
    
    @log(__LOGGER)
    def update_memory(self, t):

        # Update Memory
        self.zh += [self.z]
        if t == 0:
            self.err[:, 1] = self.w[:, 0]
        else:
            self.err[:, t + 1] = (self.sys.A - self.sys.B @ self.K) @ self.err[:, t] + self.w[:, t]
        self.xx[:, t + 1] = self.z[:, t + 1] + self.err[:, t + 1]
        self.vv[:, t] = self.v[:, t]

        return None

    @log(__LOGGER)
    def probe_task(self, t, spec=None):

        self.solver.mark = t
        self.solver.riskH = self.risk

        if spec is not None:
            STLConstrs = self.solver.AddSTLConstraints(spec)
            STLProbConstrs = self.solver.AddNewSTLProbConstraint(spec)
            
        DynConstrs = self.solver.AddDynamicsConstraints()
        RiskConstrs = self.solver.AddRiskConstraints()
        
        self.solver.AddRiskCost()
        self.solver.AddQuadraticInputCost(self.R)
        z, v, risk, flag, _ = self.solver.Solve()

        self.solver.model.remove(DynConstrs)
        self.solver.model.remove(RiskConstrs)

        if spec is not None:
            self.solver.model.remove(STLConstrs)
            self.solver.model.remove(STLProbConstrs)

        self.solver.model.update()

        return (z, v, risk, flag)           

    @log(__LOGGER)
    def accept_task(self, t, spec):

        self.psi = self.psi & spec
        self.accept_phi += [spec]
        self.accept_prob += [np.ones((self.N, ))]
        self.accept_time += [t]
        self.solver.AddSTLConstraints(spec)
        self.solver.AddNewSTLProbConstraint(spec)

        logging.info("Agent " + str(self.agent_id) + " has accepted the new task " + spec.name + " at step " + str(t) + "!")

    @log(__LOGGER)
    def reject_task(self, t, spec):
        
        logging.info("Agent " + str(self.agent_id) + " has rejected the new task " + spec.name + " at step " + str(t) + "!")

    @log(__LOGGER)
    def _compute(self, t, solution):
    # def apply_control(self, t, solution):
        
        #solver.AddControlBounds(self.u_limits[:, 0], self.u_limits[:, 1])
        #solver.AddQuadraticInputCost(self.R)
        zNew, vNew, riskNew, flag = solution

        # Check whether a solution has been found!
        if flag != GRB.OPTIMAL:

            self.zz[:, t] = self.z[:, t]
            logging.info("Agent " + str(self.agent_id) + " has rejected the new measurement at step " + str(t) + "!")

        else:
            # Update control strategy
            self.z = zNew
            self.v = vNew
            self.risk = riskNew
            self.err[:, t] = 0
            self.accept_meas += [t]
            
            logging.info("Agent " + str(self.agent_id) + " has accepted the new measurement at step " + str(t) + "!")
        
        self.update_memory(t)
        self.update_probabilities(t)
        self.update_measurement(t+1)

    @log(__LOGGER)
    def bid(self, t, spec):
        _, _, risk, flag = self.probe_task(t, spec)
        if flag != GRB.OPTIMAL:
            price = 1
        else:
            price = calculate_risks([spec], risk)[0]
        return price


def main():
    ###########################################################
    # 0. Parameters                                           #
    ###########################################################
    TIME_INTERVAL = 0.01
    NUM_AGENTS = 4
    LOG_LEVEL = "INFO"
    CONTROL_HORIZON = 25
    agents = []

    initial_states = [[5, 5], [15, 5], [25, 5], [35, 5]]
    control_bounds = [4, 5, 6, 7]

    initialize_logger(LOG_LEVEL)

    model = agent_model(2)
    specs = agent_specs(2, CONTROL_HORIZON)

    ###########################################################
    # 1. Create the environment and add the obstacles         #
    ###########################################################
    
    env = Environment(async_loop_lock=TimeIntervalAsyncLoopLock(TIME_INTERVAL))
    # env.add_entities((BoxEntity(position=np.array([2, 2, 2])), SphereEntity(position=np.array([-4, -4, 2]))))
    
    ###########################################################
    # For each agent in the simulation...                     #
    ###########################################################
    agent_coordinator = TasMasCoordinator[TasMasKnowledge](env, agents)
    specs_knowledge = TasMasKnowledge(
        simulation_horizon = CONTROL_HORIZON,
        tlg = specs.tlg,
        tll = specs.tll,
        slg = specs.slg,
        sll = specs.sll,
    )

    agent_coordinator.initialise_specs(specs_knowledge)
    for i in range(NUM_AGENTS):
        ###########################################################
        # 2. Create the entity and model the agent will use       #
        ###########################################################
        agent_entity = RacecarEntity(i, model=RacecarModel(i), position=np.array([0, i, 0.1]))

        ###########################################################
        # 3. Create the agent and assign it an entity             #
        ###########################################################
        agent = TasMasAgent[TasMasAgentModel](i, agent_entity)

        ###########################################################
        # 6. Initialise the agent with some starting information  #
        ###########################################################
        awareness_vector = AwarenessVector(agent.id, np.zeros(7))
        agent_knowledge = TasMasAgentModel(
            A = model.A,
            B = model.B,
            C = model.C,
            D = model.D,

            # Disturbance variables
            mu = model.mu,
            Sigma = model.Sigma
        )
        agent.initialise_agent({agent.id: awareness_vector}, {agent.id: agent_knowledge})

        ###########################################################
        # 4. Add the agent to the environment                     #
        ###########################################################
        env.add_agents(agent)

        ###########################################################
        # 5. Create and set the component of the agent            #
        ###########################################################
        # In this example, wee need a controller and a perception system
        # They both run at the same frequency

        agent_controller = TasMasController(
                i,
                sys_model = agent._knowledge_database,
                spec = specs.safety, 
                N = CONTROL_HORIZON, 
                x0 = np.array(np.array(initial_states[i])), 
                Q = model.Q,
                R = model.R,
                ub = control_bounds[i], 
                async_loop_lock = TimeIntervalAsyncLoopLock(TIME_INTERVAL)
            )

        agent.add_components(
            agent_controller,
            DefaultPerceptionSystem(i, env, TimeIntervalAsyncLoopLock(TIME_INTERVAL))
        )

        ###########################################################
        # 7. Add the agent to the coordinator                     #
        ###########################################################
        agent_coordinator.add_agents(agent)

    ###########################################################
    # 8. Run the simulation                                   #
    ###########################################################
    agent_coordinator.run(TIME_INTERVAL)


if __name__ == "__main__":

    np.random.seed(3)
    main()
