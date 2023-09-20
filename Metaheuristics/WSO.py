import numpy as np
import random
from Problem.Benchmark.Problem import fitness as f

def iterarWSO(max_iter, iter, dim, whiteSharks, WSO_Positions, lb, ub, v, gbest, wbest ):
    
    fmax = 0.75
    fmin = 0.07
    tau = 4.125
    mu = 2 / abs(2 - tau - np.sqrt(tau ** 2 - 4 * tau))
    pmin = 0.5
    pmax = 1.5
    a0 = 6.250
    a1 = 100
    a2 = 0.0005
    mv = 1 / (a0 + np.exp((max_iter / 2.0 - iter) / a1))
    s_s = np.abs(1 - np.exp(-a2 * iter / max_iter))
    p1 = pmax + (pmax - pmin) * np.exp(-(4 * iter / max_iter) ** 2)
    p2 = pmin + (pmax - pmin) * np.exp(-(4 * iter / max_iter) ** 2)
    if lb is None and ub is None:
        lb = np.zeros(dim)
        ub = np.ones(dim)

    nu = np.floor(whiteSharks * np.random.rand(whiteSharks)).astype(int)
    for i in range(whiteSharks):
        v[i, :] = mu * (v[i, :] + p1 * (gbest - WSO_Positions[i, :]) * random.random() + p2 * (wbest[nu[i], :] - WSO_Positions[i, :]) * random.random())

    for i in range(whiteSharks):
        f = fmin + (fmax - fmin) / (fmax + fmin)
        a = np.sign(WSO_Positions[i, :] - ub) > 0
        b = np.sign(WSO_Positions[i, :] - lb) < 0
        wo = np.logical_xor(a, b)
        if random.random() < mv:
            WSO_Positions[i, :] = WSO_Positions[i, :] * (~wo) + (ub * a + lb * b)
        else:
            WSO_Positions[i, :] = WSO_Positions[i, :] + v[i, :] / f

    for i in range(whiteSharks):
        for j in range(dim):
            if random.random() < s_s:
                dist = np.abs(random.random() * (gbest[j] - 1 * WSO_Positions[i, j]))
                if i == 0:
                    WSO_Positions[i, j] = gbest[j] + random.random() * dist * np.sign(random.random() - 0.5)
                else:
                    WSO_Pos = gbest[j] + random.random() * dist * np.sign(random.random() - 0.5)
                    WSO_Positions[i, j] = (WSO_Pos + WSO_Positions[i - 1, j]) / 2 * random.random()

    return WSO_Positions, v

