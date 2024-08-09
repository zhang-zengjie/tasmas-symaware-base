from probstlpy.benchmarks.common import inside_rectangle_formula, outside_rectangle_formula
import numpy as np
from probstlpy.systems.linear import LinearSystem


# Areas of interest

SAFETY = (0, 40, 0, 40)

PICKUP = (30, 40, 0, 10)

HOME_1 = (0, 5, 5, 15)
HOME_2 = (0, 5, 20, 30)

CASE_1 = (5, 10, 35, 40)
CASE_2 = (15, 20, 35, 40)
CASE_3 = (25, 30, 35, 40)

OBSTACLE_1 = (17, 22, 0, 10)
OBSTACLE_2 = (23, 40, 17, 22)

HOMES = {'H-1': HOME_1, 'H-2': HOME_2}
CASES = {'CP-1': CASE_1, 'CP-2': CASE_2, 'CP-3': CASE_3}
OBSTACLES = {'W-1': OBSTACLE_1, 'W-2': OBSTACLE_2}


def safe_spec(n, N):

    mu_safety = inside_rectangle_formula(SAFETY, 0, 1, n)
    for obstacle in OBSTACLES.values():
        mu_safety &= outside_rectangle_formula(obstacle, 0, 1, n)
    phi_safety = mu_safety.always(0, N)
    phi_safety.name = "SAFETY"

    return phi_safety


def fetch_spec(t, T, n, via, goal, name):

    # t: the starting time of the task
    # T: the time interval needed for a task unit

    mu_pickup = inside_rectangle_formula(via, 0, 1, n)
    not_mu_pickup = outside_rectangle_formula(via, 0, 1, n)
    mu_case = inside_rectangle_formula(goal, 0, 1, n)

    from_pickup_to_case = not_mu_pickup | mu_case.eventually(0, T)
    fetch = mu_pickup.eventually(t, t + T) & from_pickup_to_case.always(t, t + T)
    fetch.name = name

    return fetch


def go_home(t, T, n, home, name):

    mu_home = inside_rectangle_formula(home, 0, 1, n)
    go_home = mu_home.always(0, 2).eventually(t, t+T-2)
    go_home.name = name
    return go_home


class agent_specs:

    def __init__(self, n, N):

        # The safety specification
        self.safety = safe_spec(n, N)

        # The time list of global specifications 
        self.tlg = [1, 15]
        # The list of global specifications
        self.slg = [[fetch_spec(self.tlg[0], 10, 2, PICKUP, CASES['GP-I'], 'FETCH GP-I'),
                     fetch_spec(self.tlg[0]+1, 7, 2, PICKUP, CASES['GP-II'], 'FETCH GP-II'),
                     fetch_spec(self.tlg[0]+2, 7, 2, PICKUP, CASES['GP-III'], 'FETCH GP-III')],
                    [fetch_spec(self.tlg[1], 10, 2, PICKUP, CASES['GP-I'], 'FETCH GP-I'),
                     fetch_spec(self.tlg[1], 10, 2, PICKUP, CASES['GP-II'], 'FETCH GP-II')]]

        # The time list of local specifications 
        self.tll = [30]
        # The list of local specifications
        self.sll = [{0: go_home(self.tll[0], 10, 2, HOME_1, 'BACK TO H-1'),
                1: go_home(self.tll[0], 10, 2, HOME_1, 'BACK TO H-1'),
                2: go_home(self.tll[0], 10, 2, HOME_2, 'BACK TO H-2'),
                3: go_home(self.tll[0], 10, 2, HOME_2, 'BACK TO H-2')}
            ]

        # LG: the list of global specifications (slg) and its time list (tlg) 
        # LG: the list of local specifications (sll) and its time list (tlg)

class agent_model:

    def __init__(self, n):

        self.A = np.eye(n)
        self.B = np.eye(n)
        self.C = np.eye(n)
        self.D = np.zeros([n, n])

        # Disturbance variables
        self.mu = np.zeros(n)
        self.Sigma = 0.01 * np.eye(n)

        # Initialize System
        self.sys = LinearSystem(self.A, self.B, self.C, self.D, self.mu, self.Sigma)

        # Quadratic Cost function (nonzero & SPD)
        self.Q = np.eye(self.sys.n) * 0.001
        self.R = np.eye(self.sys.m) * 0.001
