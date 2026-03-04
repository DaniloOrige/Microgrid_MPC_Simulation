import numpy as np
import matplotlib.pyplot as plt
import cvxpy as cp
import pandas as pd

class Battery:
    def __init__(self, capacity, ch_efficiency, dis_efficiency, IC):
        self.capacity = capacity  # Battery capacity in Wh
        self.ch_efficiency = ch_efficiency  # Charging efficiency (0-1)
        self.dis_efficiency = dis_efficiency  # Discharging efficiency (0-1)
        self.SoC = IC  # Initial State of Charge in percentage (0-100)


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


def step_microgrid_open_loop(SoC, P_bat, P_load, P_pv, tariff, K_ch, K_dis, dt, eta, ch_bat, allow_export = False):
    
    SoC_avail = (SoC/100)*ch_bat  # Available energy in the battery [Wh]
    SoC_room = ((100 - SoC)/100)*ch_bat # Available room for charging in the battery [Wh]
    P_ch_max = SoC_room/ (eta*dt) # Maximum charging power based on available room and charging efficiency [W]
    P_dis_max = (SoC_avail*eta)/(dt) # Maximum discharging power based on available energy and charging efficiency [W]

    if P_bat >= 0: # Discharging
        P_bat = min(P_bat, P_dis_max)
        dSoC = -K_dis*P_bat
    else:          # Charging
        P_bat = -min(-P_bat, P_ch_max)
        dSoC = -K_ch*P_bat

    P_grid_raw = P_load - P_pv - P_bat # [W]

    if allow_export:
        E_grid = P_grid_raw*dt
        E_curt = 0.0 # Curtailment power is zero when excess power can be exported
    else:
        E_grid = max(P_grid_raw, 0.0)*dt # No export allowed, grid power cannot be negative
        E_curt = max(-P_grid_raw, 0.0)*dt # Curtailment power is the excess power that cannot be exported

    cost = E_grid * tariff  # Cost for the current time step [R$]


    SoC = np.clip(SoC + dSoC, 0.0, 100.0) # Update SoC and ensure it stays within bounds

    


    return SoC, E_grid, E_curt, cost, P_bat




