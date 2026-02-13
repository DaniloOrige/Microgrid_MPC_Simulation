import numpy as np
import matplotlib.pyplot as plt
import cvxpy as cp
import pandas as pd

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

def load_microgrid_data():      
    'Load and process microgrid data from CSV file and calculate mean hourly for each variable'

    data = pd.read_csv('microgrid_data/inverter2.csv', comment='#')
    data = data[['_time', '_field', '_value']].copy()

    # Changing data orientation to use time as an index and power generated/consumed as columns (long to wide)
    data = data.pivot(index='_time', columns='_field', values='_value').reset_index()

    # Converting time column to datetime format
    data['_time'] = pd.to_datetime(data['_time'],format ='ISO8601' ,utc=True)  
    data['_hour'] = data['_time'].dt.hour

    mean_hourly = data.groupby('_hour').mean()
    hours     = mean_hourly.index.to_numpy()
    pv_mean   = mean_hourly["Sun_PV_Power"].to_numpy()
    grid_mean = mean_hourly["Potencia_total_rede"].to_numpy()
    load_mean = mean_hourly["Total_load_power"].to_numpy()

    # Linear interpolation to match sampling time (15 minutes)
    pv_mean   = np.interp(np.arange(0, 24, 0.25), hours, pv_mean)
    grid_mean = np.interp(np.arange(0, 24, 0.25), hours, grid_mean)
    load_mean = np.interp(np.arange(0, 24, 0.25), hours, load_mean)

    return {
        "pv_power": pv_mean,
        "load_power": load_mean,
        "grid_power": grid_mean,
        "hours": hours,
    }




def CARIMA(A, B, N, Nu):

    # Initial definitions
    
    delta = np.array([1.0, -1.0])
    A_delta = np.convolve(A, delta)  # Integrated A polynomial

    # Initialization of lists to store the polynomials at each step
    E_list = []
    F_list = []

    # The first remainder is the unity polynomial (numerator of the transfer function)
    for j in range(1, N + 1):
        # Initialize E and F for step j
        e = np.zeros(j)
        
        # Simplified polynomial division to compute E_j and F_j
        # The current numerator starts as [1, 0, 0, ...]
        numerator = np.zeros(j + len(A_delta) - 1)
        numerator[0] = 1.0
        
        # Perform long division
        for i in range(j):
            e[i] = numerator[i] / A_delta[0]
            # Update the numerator by subtracting e[i] * shifted A_delta
            numerator[i:i+len(A_delta)] -= e[i] * A_delta
        
        # The remaining part of the numerator after the divisions is the F polynomial (shifted)
        f = numerator[j:]
        
        E_list.append(e)
        F_list.append(f)


    F = np.vstack(F_list)
    # Reshape E to be a column vector
    E = np.array(E_list[N-1])
    E = E.reshape(len(E), 1)

    Ej = np.empty((N, Nu))

    for j in range(Nu):
        aux = np.zeros((j, 1))
        temp = np.vstack([aux, E[:N-j]])
        Ej[:, j] = temp[:, 0]

    G = Ej * B

    return {
         "G": G, 
         "F": F
    }

    print("G Matrix:", G)
    print("F Matrix:", F)





