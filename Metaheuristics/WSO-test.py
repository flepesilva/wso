import numpy as np
import random

def WSO(max_iter, whiteSharks, lb, ub, dim, fobj):
    ccurve = np.zeros(max_iter) #guardamos los curva de convergencia 
    gbest = None
    wbest = None

    # Inicializar las posiciones de los tiburones blancos
    # whitSharks = Numero de tiburones blancos (poblacion)
    # dim = Dimensión del problema
    # ub = Limite superior
    # lb = Limite inferior

    # WSO_Positions seria algo como el dimension del problema, en este caso con 30 tiburones
    # en una dimension de 10, daria 300 posiciones, 10 por cada tiburon
    WSO_Positions = initialization(whiteSharks, dim, ub, lb)#

    # La "v" representa la velocidad de los tiburones, se inicializa en cero para ir actualizando
    v = np.zeros_like(WSO_Positions)
    # Se inicializa el fitness de cada posicion y se calcula con la funcion objetivo
    fit = np.zeros(whiteSharks)
    # Se calcula el fitness de cada tiburon de la matriz inicial en funcion de las posiciones iniciales
    for i in range(whiteSharks):
        fit[i] = fobj(WSO_Positions[i, :])

    # fitness de los 30 tiburones
    fitness = fit

    #El fitnes minimo
    fmin0 = np.min(fit)

    # Devuelve el indice del tiburon con el fitness mas bajo
    index = np.argmin(fit)

    # Asignamos al peor valor como el mejor para ir actualizando en cada iteracion
    gbest = WSO_Positions[index, :]

    # A lo largo del algoritmo WSO, se actualiza la matriz wbest
    # con las mejores posiciones locales encontradas por cada tiburón blanco en la población.
    wbest = np.copy(WSO_Positions)

    #Parametros fijos
    # frecuencia maxima de movimiento ondulante
    fmax = 0.75

    # frecuencia minima de movimiento ondulante
    fmin = 0.07

    # Coeficiente de aceleracion
    tau = 4.125

    # factor de restricción sugerido en WSO para analizar
    # el comportamiento de convergencia de los tiburones blancos
    mu = 2 / abs(2 - tau - np.sqrt(tau ** 2 - 4 * tau))

    # velocidades inciales y subornidad para un buen movimiento
    pmin = 0.5
    pmax = 1.5

    # constantes positivas utilizadas para gestionar los comportamientos de exploración y explotación.
    a0 = 6.250
    a1 = 100

    # una constante positiva utilizada para controlar los comportamientos de exploración y explotación.
    a2 = 0.0005

    # 3.1 Comienzo del proceso de iteracion del WSO
    for ite in range(max_iter):
        # función del tiempo para asegurar la convergencia al acelerar la velocidad de búsqueda
        # y fortalecer las características de exploración y explotación del algoritmo propuesto
        mv = 1 / (a0 + np.exp((max_iter / 2.0 - ite) / a1))

        # S_s es un parámetro sugerido para expresar la fuerza de los sentidos del olfato y la vista
        # de los tiburones blancos cuando siguen a otros tiburones blancos que están cerca de la presa óptima
        s_s = np.abs(1 - np.exp(-a2 * ite / max_iter))

        # p1 y p2 p1 y p2 representan las fuerzas de los tiburones blancos que controlan
        # el efecto de w_(〖gbest〗_k ) y w_best^(〖v^i〗_k ) en w_k^i, respectivamente
        p1 = pmax + (pmax - pmin) * np.exp(-(4 * ite / max_iter) ** 2)
        p2 = pmin + (pmax - pmin) * np.exp(-(4 * ite / max_iter) ** 2)

        # 3.2 Actualizar la velocidad de los tiburones blancos en el agua
        #--------------------------------------------------------------------------------------
        nu = np.floor(whiteSharks * np.random.rand(whiteSharks)).astype(int)
        for i in range(whiteSharks):
            v[i, :] = mu * (v[i, :] + p1 * (gbest - WSO_Positions[i, :]) * random.random() + p2 * (wbest[nu[i], :] - WSO_Positions[i, :]) * random.random())
        #---------------------------------------------------------------------------------------

        # 3.3 Actualizar la posición del tiburón blanco
        #--------------------------------------------------------------------------------------
        for i in range(whiteSharks):
            f = fmin + (fmax - fmin) / (fmax + fmin)
            a = np.sign(WSO_Positions[i, :] - ub) > 0
            b = np.sign(WSO_Positions[i, :] - lb) < 0
            wo = np.logical_xor(a, b)
            if random.random() < mv:
                WSO_Positions[i, :] = WSO_Positions[i, :] * (~wo) + (ub * a + lb * b)
            else:
                WSO_Positions[i, :] = WSO_Positions[i, :] + v[i, :] / f
        #--------------------------------------------------------------------------------------
        # Actualizar las posiciones de los tiburones blancos teniendo en cuenta la escuela de pesca
        for i in range(whiteSharks):
            for j in range(dim):
                if random.random() < s_s:
                    dist = np.abs(random.random() * (gbest[j] - 1 * WSO_Positions[i, j]))
                    if i == 0:
                        WSO_Positions[i, j] = gbest[j] + random.random() * dist * np.sign(random.random() - 0.5)
                    else:
                        WSO_Pos = gbest[j] + random.random() * dist * np.sign(random.random() - 0.5)
                        WSO_Positions[i, j] = (WSO_Pos + WSO_Positions[i - 1, j]) / 2 * random.random()

        #--------------------------------------------------------------------------------------
        # Actualizar las posiciones global, best and new positions
        fit = np.zeros(whiteSharks)
        for i in range(whiteSharks):
            #Manejo de violaciones de límites.
            if np.all(lb <= WSO_Positions[i, :]) and np.all(WSO_Positions[i, :] <= ub):
                fit[i] = fobj(WSO_Positions[i, :])
                #Evaluar el fitness
                if fit[i] < fitness[i]:
                    wbest[i, :] = WSO_Positions[i, :] #actualizar la mejor posicion
                    fitness[i] = fit[i] #Actualizar el fitness
                # Encontrar las mejores posiciones.
                if fitness[i] < fmin0:
                    fmin0 = fitness[i]
                    gbest = wbest[index, :] #Actualizar las mejores posiciones globales.

        #resultados
        outmsg = f'Iteration# {ite}  Fitness= {fmin0}'
        print(outmsg)

        # Mejor valor encontrado hasta la iteración "ite".
        ccurve[ite] = fmin0
        if ite > 2:
            pass  # Plotting the convergence curve

    return fmin0, gbest, ccurve

# Función de inicialización de la población
def initialization(whiteSharks, dim, ub, lb):
    
    pos = np.random.rand(whiteSharks, dim) * (ub - lb) + lb
    return pos

# Función de ejemplo de la función objetivo
def example_function(x):
    return np.sum(np.abs(x)) + np.prod(np.abs(x))

# Parámetros del algoritmo
max_iter = 1000  # Número máximo de iteraciones
whiteSharks = 30  # Número de tiburones blancos (población)
dim = 2  # Dimensión del problema
ub = 50 * np.ones(2)
lb = -50 * np.ones(2)

# Ejecutar el algoritmo WSO
fmin0, gbest, ccurve = WSO(max_iter, whiteSharks, lb, ub, dim, example_function)