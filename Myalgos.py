import numpy as np
import gurobipy as gp
from gurobipy import *

'''--------------------Run Offline Benchmarks------------------------'''


def Offline(W, A, D, C, d_bar):  # Give one sample path, output one reward
    reward = 0
    T = A.shape[1]
    pi = BP_C(W, A, D, C, d_bar)
    for t in range(T):
        k = pi[t]
        reward = reward + np.sum(W[:, t, k] * A[:, t, k] * D[:, t])
    reward = reward / T

    return reward


# def Offline1(W, A, D, C, d_bar):  # Give one sample path, output one reward
#     reward = 0
#     T = A.shape[1]
#     pi = LP_S(W, A, D, C, d_bar)
#     for t in range(T):
#         k = pi[t]
#         reward = reward + np.sum(W[:, t, k] * A[:, t, k] * D[:, t])
#     reward = reward / T
#
#     return reward


'''-------------------------Run Greedy--------------------------------'''


def Greedy(W, A, D, C, d_bar):
    reward = 0
    I = A.shape[0]
    T = A.shape[1]
    K = A.shape[2]

    Service = range(I)
    pi = np.zeros((T,), dtype=int)
    C_t = C + np.zeros((I, ))

    for t in range(T):
        # Return Resource
        t_start = max(t - d_bar+1, 0)
        for tau in range(t_start, t):  # Replenish services
            k = pi[tau]
            for i in Service:
                if k > 0 and t == tau + D[i, tau]:  # Accepted and ought to be returned now
                    C_t[i] = C_t[i] + A[i, tau, k]

        # Greedily allocate resource
        reward_t = 0
        for k in range(K):
            if (C_t >= A[:, t, k]).all() and np.sum(W[:, t, k] * A[:, t, k] * D[:, t]) > reward_t:  # Enough resources?
                reward_t = np.sum(W[:, t, k] * A[:, t, k] * D[:, t])
                pi[t] = k
        C_t = C_t - A[:, t, pi[t]]
        reward = reward + reward_t

    reward = reward / T

    # return reward, pi
    return reward


'''------------------------NMWU algorithm-----------------------------'''


def NMWU(delta, a_bar, d_bar, J_bar, V_bar, W, A, D, C):  # one sample each time
    I = A.shape[0]
    T = A.shape[1]
    K = A.shape[2]
    pi = np.zeros((T,), dtype=int)
    Service = range(I)
    reward = 0
    xi = max(a_bar / C)
    beta = np.sqrt(xi) * np.log(I / np.sqrt(xi))     # beta < 1

    C_t = C + np.zeros((I, ))
    span = d_bar
    halfspan = 0
    phi = 0
    psi = 0

    q = 0  # Phase 0, warm up, greedy policy
    for t in range(span):  # 0 to span-1
        # Returning resources
        for t_ in range(t):
            k = pi[t_]
            for i in Service:
                if pi[t_] and t == t_ + D[i, t_]:  # accepted and ought to be returned now
                    C_t[i] = C_t[i] + A[i, t_, k]

        reward_t = 0
        for k in range(K):
            if (C_t >= A[:, t, k]).all() and np.sum(W[:, t, k] * A[:, t, k] * D[:, t]) >= reward_t:
                reward_t = np.sum(W[:, t, k] * A[:, t, k] * D[:, t])
                pi[t] = k
        C_t = C_t - A[:, t, pi[t]]
        reward = reward + reward_t

    for t in range(span, T):  # Phase 1,2,...
        if (t+1) % span == 1:
            halfspan = span
            span = 2 * span  # \tau^{(q)}
            D_1 = D[:, :halfspan]
            W_1 = W[:, :halfspan, :]
            A_1 = A[:, :halfspan, :]

            q = q + 1
            eta = np.sqrt(np.log(2*I)/halfspan)
            [phi, psi] = MWU(W_1, A_1, D_1, eta, J_bar, V_bar)

        for t_ in range(t - d_bar, t):  # replenish services
            for i in Service:
                k = pi[t_]
                if k > 0 and t == t_ + D[i, t_]:  # accepted and ought to be returned now
                    C_t[i] = C_t[i] + A[i, t_, k]

        epsilon = 1 + np.sqrt(np.log(1 / delta) / (2 * halfspan)) - 2 * np.sqrt(np.log(2 * I) / halfspan) + np.sqrt(np.log(I / delta) / (2
                                                                                                                   * halfspan))

        idx_w = np.random.randint(halfspan)
        phi_s = phi[:, idx_w]
        psi_s = psi[:, idx_w]
        [idk, _, _] = kappa(phi_s, psi_s, W[:, t, :], A[:, t, :], D[:, t], J_bar, V_bar)
        B_tau = min(C) / (max(V_bar)*(epsilon + beta))
        if idk > 0 and np.random.rand() <= B_tau and (C_t >= A[:, t, idk]).all():
            pi[t] = idk
            C_t = C_t - A[:, t, idk]
            reward = reward + np.sum(W[:, t, idk] * A[:, t, idk] * D[:, t])
        else:  # Lack resources?
            pi[t] = 0
    reward = reward / T

    return reward


'''------------------------NMWU-S algorithm-----------------------------'''


def NMWU_S(delta, a_bar, d_bar, J_bar, V_bar, W, A, D, C):  # one sample each time
    I = A.shape[0]
    T = A.shape[1]
    K = A.shape[2]
    pi = np.zeros((T,), dtype=int)
    Service = range(I)
    reward = 0
    xi = max(a_bar / C)
    beta = np.sqrt(xi) * np.log(I / np.sqrt(xi))     # beta < 1

    C_t = C + np.zeros((I, ))
    span = d_bar
    halfspan = 0
    phi = 0
    psi = 0

    q = 0  # Phase 0, warm up, greedy policy
    for t in range(span):  # 0 to span-1
        # Returning resources
        for t_ in range(t):
            k = pi[t_]
            for i in Service:
                if pi[t_] and t == t_ + D[i, t_]:  # accepted and ought to be returned now
                    C_t[i] = C_t[i] + A[i, t_, k]

        reward_t = 0
        for k in range(K):
            if (C_t >= A[:, t, k]).all() and np.sum(W[:, t, k] * A[:, t, k] * D[:, t]) >= reward_t:
                reward_t = np.sum(W[:, t, k] * A[:, t, k] * D[:, t])
                pi[t] = k
        C_t = C_t - A[:, t, pi[t]]
        reward = reward + reward_t

    for t in range(span, T):  # Phase 1,2,...
        if (t+1) % span == 1:
            halfspan = span
            span = 2 * span  # \tau^{(q)}
            D_1 = D[:, :halfspan]
            W_1 = W[:, :halfspan, :]
            A_1 = A[:, :halfspan, :]

            q = q + 1
            eta = np.sqrt(np.log(2*I)/halfspan)
            [phi, psi] = MWU_S(W_1, A_1, D_1, eta, J_bar, V_bar)

        for t_ in range(t - d_bar, t):  # replenish services
            for i in Service:
                k = pi[t_]
                if k > 0 and t == t_ + D[i, t_]:  # accepted and ought to be returned now
                    C_t[i] = C_t[i] + A[i, t_, k]

        epsilon = 1 + np.sqrt(np.log(1 / delta) / (2 * halfspan)) - 2 * np.sqrt(np.log(2 * I) / halfspan) + np.sqrt(np.log(I / delta) / (2
                                                                                                                   * halfspan))
        phi_s = phi
        psi_s = psi
        [idk, _, _] = kappa(phi_s, psi_s, W[:, t, :], A[:, t, :], D[:, t], J_bar, V_bar)
        B_tau = min(C) / (max(V_bar)*(epsilon + beta))
        if idk > 0 and np.random.rand() <= B_tau and (C_t >= A[:, t, idk]).all():
            pi[t] = idk
            C_t = C_t - A[:, t, idk]
            reward = reward + np.sum(W[:, t, idk] * A[:, t, idk] * D[:, t])
        else:  # Lack resources?
            pi[t] = 0
    reward = reward / T

    return reward


'''-------------------------------Auxiliary Algorithms--------------------------------'''


def BP_C(W, A, D, C, d_bar):
    model = Model("BP")
    I = A.shape[0]
    T = A.shape[1]
    K = A.shape[2]
    Service = range(I)
    Horizon = range(T)
    Action = range(K)
    X = model.addVars(T, K, vtype=GRB.BINARY)
    model.update()

    model.setObjective(quicksum(W[i, s, k] * A[i, s, k] * D[i, s] * X[s, k] for s in Horizon for k in Action for i in Service) / T,
                       GRB.MAXIMIZE)

    for t in Horizon:
        model.addConstr(quicksum(X[t, k] for k in Action) <= 1)
        for i in Service:
            t_start = max(t - d_bar + 1, 0)
            model.addConstr(quicksum(bool(tau+D[i, tau] > t)*A[i, tau, k]*X[tau, k] for k in Action for tau in range(t_start, t+1)) <= C[i])

    model.update()
    model.setParam('OutputFlag', 0)
    model.optimize()

    pi = np.zeros((T,), dtype=int)
    for t in Horizon:
        for k in Action:
            if X[t, k].X == 1:
                pi[t] = k

    return pi


def kappa(phi_s, psi_s, W_s, A_s, D_s, J_bar, V_bar):
    I = W_s.shape[0]
    K = W_s.shape[1]
    M = np.zeros((I, K))
    N = np.zeros((I, K))

    for k in range(K):
        M[:, k] = W_s[:, k] * A_s[:, k] * D_s/J_bar
        for i in range(I):
            if A_s[i, k] * D_s[i] != 0:
                N[i, k] = 1 - A_s[i, k] * D_s[i] / V_bar[i]

    value = 0
    index = 0
    for k in range(K):
        temp = np.sum(phi_s * M[:, k]) + np.sum(psi_s * N[:, k])
        if temp > value:
            value = temp
            index = k

    return index, M[:, index], N[:, index]


def MWU(W, A, D, eta, J_bar, V_bar):  # W, A, D are the first half in \tau^{(q-1)}
    I = A.shape[0]
    T = A.shape[1]
    phi = np.zeros((I, T))  # weights for J/M
    psi = np.zeros((I, T))  # weights for V/N
    phi[:, 0] = np.ones((I,)) / (2 * I)
    psi[:, 0] = np.ones((I,)) / (2 * I)

    Phi = np.zeros((I, ))
    Psi = np.zeros((I, ))

    for s in range(T):
        if s + 1 < T:
            [_, m, n] = kappa(phi[:, s], psi[:, s], W[:, s, :], A[:, s, :], D[:, s], J_bar, V_bar)
            denom = np.sum(np.exp(eta * Phi)) + np.sum(np.exp(eta * Psi))
            phi[:, s + 1] = np.exp(eta * Phi) / denom
            psi[:, s + 1] = np.exp(eta * Psi) / denom
            Phi = Phi + m
            Psi = Psi + n

    return phi, psi


def MWU_S(W, A, D, eta, J_bar, V_bar):  # W, A, D are the first half in \tau^{(q-1)}
    I = A.shape[0]
    T = A.shape[1]
    phi = np.ones((I,)) / (2 * I)
    psi = np.ones((I,)) / (2 * I)

    Phi = np.zeros((I, ))
    Psi = np.zeros((I, ))

    for s in range(T):
        if s + 1 < T:
            [_, m, n] = kappa(phi, psi, W[:, s, :], A[:, s, :], D[:, s], J_bar, V_bar)
            denom = np.sum(np.exp(eta * Phi)) + np.sum(np.exp(eta * Psi))
            phi = np.exp(eta * Phi) / denom
            psi = np.exp(eta * Psi) / denom
            Phi = Phi + m
            Psi = Psi + n

    return phi, psi


def CADW_generation(I, T, K, a_bar, d_bar, w_bar):  # Fulfillment
    C = np.random.randint(100, 201, (I, ), dtype=int)  # C is an I-by-1 vector

    Service = range(I)  # 20%, 50%, 80%
    Horizon = range(T)
    Action = range(K)

    A = np.zeros((I, T, K))
    W = np.zeros((I, T, K))
    A_r = [0, 0.2, 0.5, 0.8, 1]
    W_r = [1, 1, 1, 0.95, 0.9]
    D = np.random.randint(0, d_bar, (I, T), dtype=int)
    for i in Service:
        a = np.random.randint(0, a_bar[i], (T,), dtype=int)
        w = np.random.randint(1, w_bar[i], (T,), dtype=int)
        for k in Action:
            A[i, :, k] = np.floor(A_r[k] * a)
            W[i, :, k] = np.ceil(W_r[k] * w)

    V_bar = np.zeros((I, ))
    J_bar = np.zeros((I, ))
    for t in Horizon:
        for i in Service:
            if A[i, t, K-1] == 0:
                D[i, t] = 0
            if A[i, t, K-1] * D[i, t] > V_bar[i]:
                V_bar[i] = A[i, t, K-1] * D[i, t]
            if W[i, t, K-1] * A[i, t, K-1] * D[i, t] > J_bar[i]:
                J_bar[i] = W[i, t, K-1] * A[i, t, K-1] * D[i, t]

    return C, A, D, W, J_bar, V_bar
