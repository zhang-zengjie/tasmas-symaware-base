from probstlpy.benchmarks.common import inside_rectangle_formula, outside_rectangle_formula
import numpy as np
from probstlpy.systems.linear import LinearSystem


# Areas of interest

SAFETY = (0, 40, 0, 40)

HOME_A = (0, 10, 0, 5)
HOME_B = (10, 20, 0, 5)
HOME_C = (20, 30, 0, 5)
HOME_D = (30, 40, 0, 5)

CASE_1 = (0, 10, 35, 40)
CASE_2 = (10, 20, 35, 40)
CASE_3 = (20, 30, 35, 40)
CASE_4 = (30, 40, 35, 40)

OBSTACLE_1 = (10, 18, 16, 24)
OBSTACLE_2 = (22, 30, 16, 24)

HOMES = {'T-0': HOME_A, 'T-1': HOME_B, 'T-2': HOME_C, 'T-3': HOME_D}
CASES = {'GP-I': CASE_1, 'GP-II': CASE_2, 'GP-III': CASE_3, 'GP-IV': CASE_4}
OBSTACLES = {'B-1': OBSTACLE_1, 'B-2': OBSTACLE_2}

LOAD = (34, 40, 17, 23)


def safe_spec(n, N, safety, obstacles):

    mu_safety = inside_rectangle_formula(safety, 0, 1, n)
    for obstacle in obstacles.values():
        mu_safety &= outside_rectangle_formula(obstacle, 0, 1, n)
    phi_safety = mu_safety.always(0, N)
    phi_safety.name = "SAFETY"

    return phi_safety


def fetch_spec(t, T, n, via, goal, name):

    # t: the starting time of the task
    # T: the time interval needed for a task unit

    mu_case = inside_rectangle_formula(via, 0, 1, n)
    not_mu_case = outside_rectangle_formula(via, 0, 1, n)
    mu_load = inside_rectangle_formula(goal, 0, 1, n)

    from_case_to_load = not_mu_case | mu_load.eventually(0, T)
    fetch = mu_case.eventually(t, t + T) & from_case_to_load.always(t, t + T)
    fetch.name = name

    return fetch


def go_home(t, T, n, home, name):

    mu_home = inside_rectangle_formula(home, 0, 1, n)
    go_home = mu_home.always(0, 2).eventually(t, t+T-2)
    go_home.name = name
    return go_home


def get_assigner(N, n):

    A = np.eye(n)
    B = np.eye(n)
    C = np.eye(n)
    D = np.zeros([n, n])

    # Disturbance variables
    mu = np.zeros(n)
    Sigma = 0.01 * np.eye(n)

    # Initialize System
    sys = LinearSystem(A, B, C, D, mu, Sigma)

    # Quadratic Cost function (nonzero & SPD)
    Q = np.eye(sys.n) * 0.001
    R = np.eye(sys.m) * 0.001

    # Specifications

    phi_0 = safe_spec(n, N, SAFETY, OBSTACLES)

    # The time list of global specifications 
    tlg = [2, 8]
    # The list of global specifications
    slg = [[fetch_spec(tlg[0], 7, 2, CASES['GP-I'], LOAD, 'FETCH GP-I'), fetch_spec(tlg[0]+1, 7, 2, CASES['GP-II'], LOAD, 'FETCH GP-II'), fetch_spec(tlg[0]+2, 7, 2, CASES['GP-IV'], LOAD, 'FETCH GP-IV')],
          [fetch_spec(tlg[1], 7, 2, CASES['GP-III'], LOAD, 'FETCH GP-III'), fetch_spec(tlg[1]+1, 7, 2, CASES['GP-II'], LOAD, 'FETCH GP-II')]]

    # The time list of local specifications 
    tll = [15]
    # The list of local specifications
    sll = [{0: go_home(tll[0], 10, 2, HOME_A, 'BACK TO T-0'),
            1: go_home(tll[0], 10, 2, HOME_B, 'BACK TO T-1'),
            2: go_home(tll[0], 10, 2, HOME_C, 'BACK TO T-2'),
            3: go_home(tll[0], 10, 2, HOME_D, 'BACK TO T-3')}
          ]
    '''
    A = Agent(sys, phi_0, N, x0=np.array([5, 5]), Q=Q, R=R, ub=4, name=0)
    B = Agent(sys, phi_0, N, x0=np.array([15, 5]), Q=Q, R=R, ub=5, name=1)
    C = Agent(sys, phi_0, N, x0=np.array([25, 5]), Q=Q, R=R, ub=6, name=2)
    D = Agent(sys, phi_0, N, x0=np.array([35, 5]), Q=Q, R=R, ub=7, name=3)

    agents = {0: A, 1: B, 2: C, 3: D}
    TA = Assigner(agents)
    '''

    return (tlg, slg), (tll, sll)


def get_phi_0(n, N):
    phi_0 = safe_spec(n, N, SAFETY, OBSTACLES)

    return phi_0