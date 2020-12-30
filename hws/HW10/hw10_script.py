from functools import partial
import numpy as np
import pandas as pd
import logging
import cvxpy as cvx
import scipy as sp
from scipy import sparse as spr
from scipy.special import erf
logging.basicConfig(level=logging.INFO)

#region Q2
def synt_team_data():
    n = 10
    m = 45
    m_test = 45
    sigma= 0.250
    train=np.array([[1, 2, 1],
    [1, 3, 1],
    [1, 4, 1],
    [1, 5, 1],
    [1, 6, 1],
    [1, 7, 1],
    [1, 8, 1],
    [1, 9, 1],
    [1, 10, 1],
    [2, 3 ,-1],
    [2, 4 ,-1],
    [2, 5 ,-1],
    [2, 6 ,-1],
    [2, 7 ,-1],
    [2, 8 ,-1],
    [2, 9 ,-1],
    [2, 10 ,-1],
    [3, 4, 1],
    [3, 5 ,-1],
    [3, 6 ,-1],
    [3, 7, 1],
    [3, 8, 1],
    [3, 9, 1],
    [3, 10, 1],
    [4, 5 ,-1],
    [4, 6 ,-1],
    [4, 7, 1],
    [4, 8, 1],
    [4, 9 ,-1],
    [4, 10 ,-1],
    [5, 6, 1],
    [5, 7, 1],
    [5, 8, 1],
    [5, 9 ,-1],
    [5, 10, 1],
    [6, 7, 1],
    [6, 8, 1],
    [6, 9 ,-1],
    [6, 10 ,-1],
    [7, 8, 1],
    [7, 9, 1],
    [7, 10 ,-1],
    [8, 9 ,-1],
    [8, 10 ,-1],
    [9, 10, 1]])

    test=np.array([[1, 2, 1],
    [1, 3, 1],
    [1, 4, 1],
    [1, 5, 1],
    [1, 6, 1],
    [1, 7, 1],
    [1, 8, 1],
    [1, 9, 1],
    [1, 10, 1],
    [2, 3 ,-1],
    [2, 4, 1],
    [2, 5 ,-1],
    [2, 6 ,-1],
    [2, 7 ,-1],
    [2, 8, 1],
    [2, 9 ,-1],
    [2, 10 ,-1],
    [3, 4, 1],
    [3, 5 ,-1],
    [3, 6, 1],
    [3, 7, 1],
    [3, 8, 1],
    [3, 9 ,-1],
    [3, 10, 1],
    [4, 5 ,-1],
    [4, 6 ,-1],
    [4, 7 ,-1],
    [4, 8, 1],
    [4, 9 ,-1],
    [4, 10 ,-1],
    [5, 6 ,-1],
    [5, 7, 1],
    [5, 8, 1],
    [5, 9, 1],
    [5, 10, 1],
    [6, 7, 1],
    [6, 8, 1],
    [6, 9, 1],
    [6, 10, 1],
    [7, 8, 1],
    [7, 9 ,-1],
    [7, 10, 1],
    [8, 9 ,-1],
    [8, 10 ,-1],
    [9, 10, 1]
    ])

    A_train = build_game_incidence_matrix(train, n)
    A_test = build_game_incidence_matrix(test, n)
    return n, m, sigma, m_test, train, test, A_train, A_test

def build_game_incidence_matrix(data, num_teams):
    num_games = data.shape[0]
    GIM = np.zeros((num_games, num_teams))  # Game Incidence Matrix
    for p, r in enumerate(data):
        j, k, y = r
        GIM[p, j - 1] = y   # -1 because of matlab indexing
        GIM[p, k - 1] = -y  # -1 because of matlab indexing
    GIM_sp = spr.csc_matrix(GIM)
    return GIM_sp

def cvx_log_noramcdf(x: cvx.Variable, mu, sigma, dt:float=10**-5):
    x_ = (x-mu)/(sigma*np.sqrt(2))
    cdf = 0.5 * (1 + erf(x_))
    return cdf

# def erf(x: cvx.Variable, dt:float):
#     t = np.linspace(0 ,x, int(1./dt))


# endregion

#region Q3

def synt_interdict_alloc_data():
    np.random.seed(0)
    (n, m) = (10, 20)
    edges = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (1, 5), \
             (2, 4), (2, 5), (3, 5), (3, 6), (4, 6), (4, 7), \
             (5, 6), (5, 7), (6, 7), (6, 8), (6, 9), (7, 8), \
             (7, 9), (8, 9)]
    a = 2*np.random.rand(m, 1)
    x_max = 1+np.random.rand(m, 1)
    B = m/2.0
    A = np.zeros((n,m))
    for k, e in enumerate(edges):
        i,j = e
        A[i,k] = -1
        A[j, k] = 1
    return n, m, a, x_max, B, edges, A

#endregion
if "__main__" == __name__:
    pass