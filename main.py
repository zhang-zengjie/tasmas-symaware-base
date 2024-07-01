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


class TasMasAgentModel(KnowledgeDatabase):
    A: np.ndarray
    B: np.ndarray
    C: np.ndarray
    D: np.ndarray
    mu: np.ndarray
    Sigma: np.ndarray


class TasMasCoordinator(AgentCoordinator[T]):

    __LOGGER = get_logger(__name__, "TasMasController")

    def __init__(self, env):
        super().__init__(env)

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
    def bidding(self):
        for agent in self._agents:
            for j in self.m_range:
                self.price_list[agent._ID][j] = agent.controllers[0].bid(self.specs[j]) if self.bidding_list[agent._ID][j] else 1
                logging.info("Agent " + str(agent._ID) + " has probed task " + self.specs[j].name + "!")

    @log(__LOGGER)
    def assign(self):
        for id, item in self.assignment_list.items():
            if 1 in item:
                self._agents[id].controllers[0].accept_task(self.specs[item.index(1)])

    @log(__LOGGER)
    def assign_global(self, specs_list):
        self.update_specs(specs_list)
        self.select_agents()
        self.bidding()
        self.auction()
        self.assign()

    @log(__LOGGER)
    def assign_local(self, specs_list):

        for id, spec in specs_list.items():
            sln = self._agents[id].controllers[0].probe_task(spec)
            if sln[-1] != GRB.OPTIMAL:
                self._agents[id].controllers[0].reject_task(spec)
            else:
                self._agents[id].controllers[0].accept_task(spec)

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
        sim_step = 0
        
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

                self._env.step()
                if sim_step < self.horizon:

                    if sim_step in self.tlg:
                        self.assign_global(self.slg[self.tlg.index(sim_step)])
                    if sim_step in self.tll:
                        self.assign_local(self.sll[self.tll.index(sim_step)])

                    
                    for agent in self._agents:
                        for controller in agent.controllers:
                            controller.set_time(sim_step)
                        agent.step()

                    sim_step += 1

                elif sim_step == self.horizon:
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


class TasMasController(Controller):
    """
    The tasmas controller by TU/e
    """

    __LOGGER = get_logger(__name__, "TasMasController")

    def __init__(self, agent_id, spec, N, x0, Q, R, ub, async_loop_lock=None):
        super().__init__(agent_id, async_loop_lock)

        self.time = 0
        self.N = N              # Time-horizon
        self.psi = spec
        self.x0 = x0
        
        self.sys = None
        self.w = None  # uncomment to check with no disturbance

        self.Q = Q
        self.R = R

        # Compute 1) relation between tube and probability & 2) Stabilizing feedback gain
        
        self.diag_sigma_inf, self.K = None, None

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
        
    @log(__LOGGER)
    def set_time(self, time):
        self.time = time

    @log(__LOGGER)
    def construct_solver(self):
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
        self.diag_sigma_inf, self.K = PRT(self.sys, self.Q, self.R)

    @log(__LOGGER)
    def update_measurement(self):
        self.zz[:, self.time + 1] = self.xx[:, self.time + 1]
        return None

    @log(__LOGGER)
    def update_probabilities(self):

        spec_prob = calculate_probabilities(self.accept_phi, self.risk)

        for i in range(len(self.accept_phi)):
            self.accept_prob[i][self.time] = spec_prob[i]
        return None
    
    @log(__LOGGER)
    def update_memory(self):

        # Update Memory
        self.zh += [self.z]
        if self.time == 0:
            self.err[:, 1] = self.w[:, 0]
        else:
            self.err[:, self.time + 1] = (self.sys.A - self.sys.B @ self.K) @ self.err[:, self.time] + self.w[:, self.time]
        self.xx[:, self.time + 1] = self.z[:, self.time + 1] + self.err[:, self.time + 1]
        self.vv[:, self.time] = self.v[:, self.time]

        return None

    @log(__LOGGER)
    def probe_task(self, spec=None):

        self.solver.mark = self.time
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
    def accept_task(self, spec):

        self.psi = self.psi & spec
        self.accept_phi += [spec]
        self.accept_prob += [np.ones((self.N, ))]
        self.accept_time += [self.time]
        self.solver.AddSTLConstraints(spec)
        self.solver.AddNewSTLProbConstraint(spec)

        logging.info("Agent " + str(self.agent_id) + " has accepted the new task " + spec.name + " at step " + str(self.time) + "!")

    @log(__LOGGER)
    def reject_task(self, spec):
        
        logging.info("Agent " + str(self.agent_id) + " has rejected the new task " + spec.name + " at step " + str(self.time) + "!")

    @log(__LOGGER)
    def _compute(self, solution):

        #solver.AddControlBounds(self.u_limits[:, 0], self.u_limits[:, 1])
        #solver.AddQuadraticInputCost(self.R)
        zNew, vNew, riskNew, flag = solution

        # Check whether a solution has been found!
        if flag != GRB.OPTIMAL:

            self.zz[:, self.time] = self.z[:, self.time]
            logging.info("Agent " + str(self.agent_id) + " has rejected the new measurement at step " + str(self.time) + "!")

        else:
            # Update control strategy
            self.z = zNew
            self.v = vNew
            self.risk = riskNew
            self.err[:, self.time] = 0
            self.accept_meas += [self.time]
            
            logging.info("Agent " + str(self.agent_id) + " has accepted the new measurement at step " + str(self.time) + "!")
        
        self.update_memory()
        self.update_probabilities()
        self.update_measurement()

    @log(__LOGGER)
    def compute_and_update(self):
        solution = self.probe_task()
        self._compute(solution)


    @log(__LOGGER)
    def bid(self, spec):
        _, _, risk, flag = self.probe_task(spec)
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

    initial_states = [[5, 5, 0], [15, 5, 0], [25, 5, 0], [35, 5, 0]]
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
    agent_coordinator = TasMasCoordinator[TasMasKnowledge](env)
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
        
        agent_entity = RacecarEntity(i, model=RacecarModel(i), 
                                     position=0.1*np.array(initial_states[i]),
                                     orientation=p.getQuaternionFromEuler([0, 0, 3.14159 / 2])
                                     )

        ###########################################################
        # 3. Create the agent and assign it an entity             #
        ###########################################################
        agent = Agent[TasMasAgentModel](i, agent_entity)

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
            TasMasController(
                i,
                spec = specs.safety, 
                N = CONTROL_HORIZON, 
                x0 = np.array(np.array(initial_states[i][:2])), 
                Q = model.Q,
                R = model.R,
                ub = control_bounds[i], 
                async_loop_lock = TimeIntervalAsyncLoopLock(TIME_INTERVAL)
            ),
            DefaultPerceptionSystem(i, env, TimeIntervalAsyncLoopLock(TIME_INTERVAL))
        )

        ###########################################################
        # 6. Initialise the agent with some starting information  #
        ###########################################################

        awareness_vector = AwarenessVector(agent.id, np.zeros(7))
        agent_knowledge = TasMasAgentModel(
            A = model.A, B = model.B, C = model.C, D = model.D, mu = model.mu, Sigma = model.Sigma
        )

        agent.initialise_agent({agent.id: awareness_vector}, {agent.id: agent_knowledge})

        for controller in agent.controllers:
            controller.construct_model(agent._knowledge_database)
            controller.construct_solver()

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
