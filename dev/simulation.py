
# Importing libraries
import numpy as np
import matplotlib.pyplot as plt
import cvxpy as cp
import control as ct


##------------------------------MPC MICROGRID SIMULATION------------------------------##

########## Parameters

# Battery parameters
C_bat = [2000]  # Battery capacity [Wh]
ch_bat = [0.9] # Charge efficiency
SoC_CI = [50]  # Initial state of charge [%]

# Sampling parameters
ts = 15 * 60 # Sampling time [s]
samples_hour = 3600/ts # Hour in samples 

# Controller parameters
Nu = int(4 * samples_hour)  # Control horizon
Np = int(24 * samples_hour)  # Prediction horizon

# Simulation parameters
t_sim = 48  # Simulation time [h]
Nsim = int(t_sim * samples_hour)  # Number of simulation steps


# Tariff Rules 
# White tariff rules for residential consumer supplied voltages lower than 2.3kV in Florianopolis, SC, Brazil in 2025
# Green flag: Off peak hours
# Yellow flag: Mid-peak hours
# Red flag: Peak hours
g_t =  0.59715 # Green flag tariff [R$/kWh]
y_t =  0.79279 # Yellow flag tariff [R$/kWh]
r_t =  1.17082 # Red flag tariff [R$/kWh]

# Tariff in a full day [R$/kWh]
tariff_daily = np.hstack((
                g_t*np.ones(int(17.5 * samples_hour)),
                y_t*np.ones(int(1 * samples_hour)),
                r_t*np.ones(int(3 * samples_hour)),
                y_t*np.ones(int(1 * samples_hour)),
                g_t*np.ones(int(1.5 * samples_hour))
))

tariff_sim = np.hstack((tariff_daily, tariff_daily))  # Tariff for the whole simulation (2 days) [R$/kWh]

# Tariff plot
tx_plot = np.arange(0, 24, ts/3600)
plt.figure()
plt.plot(tx_plot, tariff_daily, color = 'red')
plt.xticks(np.arange(0, 26, 2))
plt.yticks(np.arange(0, 1.5, 0.1))
plt.title('Electricity Tariff - Florianopolis, SC, Brazil (2025)')
plt.xlabel('Time [hours]')
plt.ylabel('Price [R$/kWh]')
plt.grid()
plt.show()

# Load and PV generation profiles
# Load profile [W]  


########## Models

# Baterry Model
#SoC_a = previous SoC [Wh]
K = 100*ch_bat[0]/C_bat[0]   # Charging constant 
#SoC = SoC_a + K * Pentregue(kWh)



"""def battery_model(C_bat, ch_bat, CI_SOC, ts):
   
    Create a discrete-time state-space model of the battery.

    Parameters:
    C_bat (float): Battery capacity [Wh]
    ch_bat (float): Charge efficiency
    CI_SOC (float): Initial state of charge [%]
    ts (float): Sampling time [s]

    Returns:
    A_bat (ndarray): State matrix
    B_bat (ndarray): Input matrix
    C_bat (ndarray): Output matrix
    D_bat (ndarray): Feedthrough matrix
    x0_bat (ndarray): Initial state
    
    # Continuous-time state-space matrices
    A_c = np.array([[0]])
    B_c = np.array([[ch_bat / C_bat]])
    C_c = np.array([[1]])
    D_c = np.array([[0]])

    # Discretization using zero-order hold
    sys_c = ct.ss(A_c, B_c, C_c, D_c)
    sys_d = ct.sample_system(sys_c, ts)

    A_bat = np.array(sys_d.A)
    B_bat = np.array(sys_d.B)
    C_bat = np.array(sys_d.C)
    D_bat = np.array(sys_d.D)

    # Initial state
    x0_bat = np.array([[CI_SOC / 100 * C_bat]])

    return A_bat, B_bat, C_bat, D_bat, x0_bat  """          



