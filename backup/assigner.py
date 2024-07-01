from tasmasbase.commons.functions import checkout_largest_in_dict
import gurobipy as gp
from gurobipy import GRB
from itertools import product
import logging


class Assigner:

    def __init__(self, agents):

        self.agents = agents
        self.n = len(self.agents)

        self.specs = None
        self.m = None
        self.m_range = None

        self.bidding_list = None
        self.price_list = None
        self.assignment_list = None

        self.model = gp.Model("AUCTION_IP")

        for agent in agents.values():
            agent.x = self.model.addMVar((self.n, ), vtype=GRB.BINARY, name="assignment")

    def update_specs(self, specs):
        self.specs = specs
        self.m = len(self.specs)
        self.m_range = [i for i in range(self.m)]
        self.clean_bidding_list()
        self.clean_price_list()
        self.clean_assignment_list()

    def clean_bidding_list(self):
        self.bidding_list = {name: [False for i in self.m_range] for name in self.agents.keys()}

    def clean_price_list(self):
        self.price_list = {name: [1 for i in self.m_range] for name in self.agents.keys()}

    def clean_assignment_list(self):
        self.assignment_list = {name: [False for i in self.m_range] for name in self.agents.keys()}

    def select_agents(self):
        
        for spec_index in self.m_range:
            rho = {name: self.specs[spec_index].robustness(agent.xx, 0)[0] for name, agent in self.agents.items()}
            selected_agents = checkout_largest_in_dict(rho, self.m)
            for name in selected_agents:
                self.bidding_list[name][spec_index] = True
        
    def auction(self):
        
        self.model.remove(self.model.getConstrs())
        p = 0
        sum = [0 for i in self.m_range]
        for name, agent in self.agents.items():
            for j in self.m_range:
                if self.bidding_list[name][j] & (self.price_list[name][j] < 1):
                    p += agent.x[j] * self.price_list[agent.name][j]
                else:
                    self.model.addConstr(agent.x[j] == 0)
                sum[j] += agent.x[j]
            self.model.addConstr(agent.x.sum() <= 1)

        if self.n < self.m:
            for agent, j in product(self.agents.values(), range(self.n, self.m)):
                self.model.addConstr(agent.x[j] == 0)

        for j in self.m_range:
            self.model.addConstr(sum[j] == 1)
        self.model.setObjective(p, GRB.MINIMIZE)
        self.model.optimize()

        if self.model.status == GRB.OPTIMAL:

            for name, agent in self.agents.items():
                for j in self.m_range:
                    self.assignment_list[name][j] = agent.x.X[j]

        else:
            logging.info("Assignment not feasible. Rejecting the new tasks!")


    def bidding(self, t):
        for name, agent in self.agents.items():
            for j in self.m_range:
                self.price_list[name][j] = agent.bid(t, self.specs[j]) if self.bidding_list[name][j] else 1
                logging.info(name + " has probed task " + self.specs[j].name + "!")


    def assign(self, t):
        for name, item in self.assignment_list.items():
            if 1 in item:
                self.agents[name].accept_task(t, self.specs[item.index(1)])


    def update_control(self, t):
        for agent in self.agents.values():
            agent.apply_control(t, agent.probe_task(t))

    def assign_global(self, t, specs_list):
        self.update_specs(specs_list)
        self.select_agents()
        self.bidding(t)
        self.auction()
        self.assign(t)

    def assign_local(self, t, specs_list):

        for name, spec in specs_list.items():
            sln = self.agents[name].probe_task(t, spec)
            if sln[-1] != GRB.OPTIMAL:
                self.agents[name].reject_task(t, spec)
            else:
                self.agents[name].accept_task(t, spec)
