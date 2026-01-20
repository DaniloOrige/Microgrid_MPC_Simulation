import numpy as np
import matplotlib.pyplot as plt
import cvxpy as cp

def G_matrix(Nu, Np, Nsim, K1):

    # Step response of battery SoC to a step change in charging power

    step_test = 1 # [W]

    SoC_coef = np.zeros(Nsim)
    SoC_coef[0] = 0  # Initial SoC for step response simulation

    for k in range(1, Nsim):
            SoC = SoC_coef[k-1] + K1 * step_test
            SoC = np.clip(SoC, 0, 100)
            SoC_coef[k] = SoC

    g = SoC_coef[:Np]  # Coefficients for the first Np steps

    G = []
    for j in range(Nu):
        coluna_j = np.hstack([np.zeros(j), g[:Np-j]])
        G.append(coluna_j)

    G = np.vstack(G).T
    return G

