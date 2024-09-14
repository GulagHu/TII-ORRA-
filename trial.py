import numpy as np
from Myalgos import *
# from gurobipy import *
# import time


if __name__ == '__main__':
    T = 1000
    I = 10
    K = 5  # 20%, 50%, 80%
    a_bar = np.random.randint(4, 9, (I,), dtype=int)
    w_bar = np.random.randint(2, 21, (I,), dtype=int)
    d_bar = 20

    # J_bar = a_bar * w_bar * d_bar
    # V_bar = a_bar * d_bar
    [C, A, D, W, J_bar, V_bar] = CADW_generation(I, T, K, a_bar, d_bar, w_bar)
    delta = 0.02
    xi = max(a_bar / C)
    CR = min(C)*(1-2*np.sqrt(xi))/(2*max(V_bar))
    print(xi)
    print(CR)

    reward_Offline = Offline(W, A, D, C, d_bar)
    print(reward_Offline)
    # reward_Greedy = Greedy(W, A, D, C, d_bar)
    # print(reward_Greedy)
    reward_NMWU_S = NMWU_S(delta, a_bar, d_bar, J_bar, V_bar, W, A, D, C)
    print(reward_NMWU_S)
    reward_NMWU = NMWU(delta, a_bar, d_bar, J_bar, V_bar, W, A, D, C)
    print(reward_NMWU)
