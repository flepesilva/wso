import numpy as np
import random

def WSO(max_iter, whiteSharks, dim, fobj):
    ccurve = np.zeros(max_iter)
    gbest = None
    wbest = None

    # Inicializar las posiciones de los tiburones blancos como soluciones binarias
    WSO_Positions = initialization(whiteSharks, dim)

    v = np.zeros_like(WSO_Positions)
    fit = np.zeros(whiteSharks)

    for i in range(whiteSharks):
        fit[i] = fobj(WSO_Positions[i, :])

    fitness = fit
    fmin0 = np.min(fit)
    index = np.argmin(fit)
    gbest = WSO_Positions[index, :]
    wbest = np.copy(WSO_Positions)

    fmax = 0.75
    fmin = 0.07
    tau = 4.125
    mu = 2 / abs(2 - tau - np.sqrt(tau ** 2 - 4 * tau))
    pmin = 0.5
    pmax = 1.5
    a0 = 6.250
    a1 = 100
    a2 = 0.0005

    for ite in range(max_iter):
        mv = 1 / (a0 + np.exp((max_iter / 2.0 - ite) / a1))
        s_s = np.abs(1 - np.exp(-a2 * ite / max_iter))
        p1 = pmax + (pmax - pmin) * np.exp(-(4 * ite / max_iter) ** 2)
        p2 = pmin + (pmax - pmin) * np.exp(-(4 * ite / max_iter) ** 2)

        nu = np.floor(whiteSharks * np.random.rand(whiteSharks)).astype(int)
        for i in range(whiteSharks):
            v[i, :] = mu * (v[i, :] + p1 * (gbest - WSO_Positions[i, :]) * random.random() + p2 * (wbest[nu[i], :] - WSO_Positions[i, :]) * random.random())

        for i in range(whiteSharks):
            f = fmin + (fmax - fmin) / (fmax + fmin)
            a = np.sign(WSO_Positions[i, :] - 1) > 0  # Limites superior
            b = np.sign(WSO_Positions[i, :]) < 0  # Limites inferior
            wo = np.logical_xor(a, b)
            if random.random() < mv:
                WSO_Positions[i, :] = WSO_Positions[i, :] * (~wo) + (1 * a + 0 * b)  # Valores binarios
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

        for i in range(whiteSharks):
            if fit[i] < fitness[i]:
                wbest[i, :] = WSO_Positions[i, :]
                fitness[i] = fit[i]
            if fitness[i] < fmin0:
                fmin0 = fitness[i]
                gbest = wbest[index, :]

        outmsg = f'Iteration# {ite}  Fitness= {fmin0}'
        print(outmsg)
        ccurve[ite] = fmin0

    return fmin0, gbest, ccurve

def initialization(whiteSharks, dim):
    pos = np.random.randint(2, size=(whiteSharks, dim))  # Genera soluciones binarias aleatorias
    return pos

# Función de ejemplo de la función objetivo para valores binarios
def example_function(x):
    return np.sum(x)

max_iter = 1000
whiteSharks = 30
dim = 10  # Por ejemplo, 10 variables binarias
ub = 1  # Límite superior para variables binarias
lb = 0  # Límite inferior para variables binarias

fmin0, gbest, ccurve = WSO(max_iter, whiteSharks, dim, example_function)