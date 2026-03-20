import numpy as np
import matplotlib.pyplot as plt
import cvxpy as cp
import pandas as pd
import control as ct

class Batterytest:
    def __init__(self, capacity, ch_efficiency, dis_efficiency, IC, K_ch, K_dis, alpha):
        self.capacity = capacity 
        self.ch_efficiency = ch_efficiency 
        self.dis_efficiency = dis_efficiency 
        self.K_ch = K_ch 
        self.K_dis = K_dis 
        self.alpha = alpha 
        self.SoC = IC
       

    def step_open_loop(self, SoC, P_bat, P_load, P_pv, tariff, dt, allow_export = False):

        # SoC_avail = (SoC/100)*self.capacity  # Battery available energy [Wh]
        # SoC_room = ((100 - SoC)/100)*self.capacity  # Available room for charging in the battery [Wh]
        # P_ch_max = SoC_room/ (self.ch_efficiency*dt) # Maximum charging power based on available room and charging efficiency [W]
        # P_dis_max = (SoC_avail*self.dis_efficiency)/(dt) # Maximum discharging power based on available energy and discharging efficiency [W]

        Pch = max(P_bat, 0)
        Pdis = max(-P_bat, 0)
      
        
        
        P_grid_raw = P_load - P_pv - Pdis + Pch # [W]
        


        if allow_export:
            E_grid = P_grid_raw*dt
            E_curt = 0.0 # Curtailment power is zero when excess power can be exported
        else:
            E_grid = max(P_grid_raw, 0.0)*dt # No export allowed, grid power cannot be negative
            E_curt = max(-P_grid_raw, 0.0)*dt # Curtailment power is the excess power that cannot be exported

        cost = E_grid * tariff  # Cost for the current time step [R$]

        dSoC = self.K_ch*Pch - self.K_dis*Pdis
       

        SoC = np.clip(0.995*SoC + dSoC, 0.0, 100.0)
     

        return SoC, E_grid, E_curt, cost, P_bat



class Battery:
    def __init__(self, capacity, ch_efficiency, dis_efficiency, IC, K_ch, K_dis, alpha):
        self.capacity = capacity if isinstance(capacity, list) else [capacity]
        self.ch_efficiency = ch_efficiency if isinstance(ch_efficiency, list) else [ch_efficiency]
        self.dis_efficiency = dis_efficiency if isinstance(dis_efficiency, list) else [dis_efficiency]
        self.SoC = IC if isinstance(IC, list) else [IC]
        self.K_ch = K_ch if isinstance(K_ch, list) else [K_ch]
        self.K_dis = K_dis if isinstance(K_dis, list) else [K_dis]
        self.alpha = alpha if isinstance(alpha, list) else [alpha]
        self.nx = len(self.capacity)


        self.dis_fact = []
        self.ch_fact = []
        for i in range(self.nx):
            self.dis_fact.append(1 / (dis_efficiency[i] * capacity[0]))
            self.ch_fact.append(ch_efficiency[i]/ capacity[i])
        
        # Number of batteries
        



    def state_space_oldversion(self, nx, nu, ts):
        A = np.array([[self.alpha]])
        B = np.array([[self.K_ch, self.K_dis]])
        # Since the state and the output are the same, we can use C as an identity matrix
        C = np.eye(nx)
        D = np.zeros((nx, 2*nx))


        systemss = ct.ss(A, B, C, D, dt = ts)
        self.ss = systemss
        return systemss
    
    def state_space(self, ts):
        # Create diagonal matrices for multiple batteries
        A = np.eye(self.nx)  # nx × nx diagonal matrix
        A = A * self.alpha
        
        # Stack charging and discharging coefficient matrices
        B_ch = np.diag(self.K_ch)  # nx × nx for charging
        B_dis = np.diag(self.K_dis)  # nx × nx for discharging
        B = np.hstack((B_ch, B_dis))  # nx × (2*nx)
        
        
        C = np.eye(self.nx)  # nx × nx identity
        D = np.zeros((self.nx, 2 * self.nx))  # nx × (2*nx)
        
        # Use control.ss for continuous-time, then sample
        sysd = ct.ss(A, B, C, D, dt = ts)
        
        
        return sysd
    
    def simulate(self, x0, P_pos, P_neg, ts):
        
        sysd = self.state_space(ts)
      

        B_dis = sysd.B[:, : self.nx]
        B_ch = sysd.B[:, self.nx :]

        xf = sysd.A @ x0 + B_dis @ P_pos + B_ch @ P_neg

        return xf
        
    def simulate_oldversion(self, x_IC, Pbat_pos, Pbat_neg, nx, nu, ts):

        systemss = self.state_space(nx, nu, ts)  # Ensure the state-space model is defined  
   
        
        Bch = systemss.B[:, 0]  # Charging input matrix
        Bdis = systemss.B[:, 1]  # Discharging input matrix

        dx = systemss.A @ x_IC + Bch * Pbat_pos + Bdis * Pbat_neg
        

        return dx
        

    

    
    def step_open_loop(self,SoC, P_bat, P_load, P_pv, tariff, dt, allow_export = False):

        # SoC_avail = (SoC/100)*self.capacity  # Battery available energy [Wh]
        # SoC_room = ((100 - SoC)/100)*self.capacity  # Available room for charging in the battery [Wh]
        # P_ch_max = SoC_room/ (self.ch_efficiency*dt) # Maximum charging power based on available room and charging efficiency [W]
        # P_dis_max = (SoC_avail*self.dis_efficiency)/(dt) # Maximum discharging power based on available energy and discharging efficiency [W]

        Pch = max(P_bat, 0)
        Pdis = max(-P_bat, 0)
      
        
        
        P_grid_raw = P_load - P_pv - Pdis + Pch # [W]
        


        if allow_export:
            E_grid = P_grid_raw*dt
            E_curt = 0.0 # Curtailment power is zero when excess power can be exported
        else:
            E_grid = max(P_grid_raw, 0.0)*dt # No export allowed, grid power cannot be negative
            E_curt = max(-P_grid_raw, 0.0)*dt # Curtailment power is the excess power that cannot be exported

        cost = E_grid * tariff  # Cost for the current time step [R$]

        dSoC = self.K_ch*Pch - self.K_dis*Pdis
       

        SoC = np.clip(0.995*SoC + dSoC, 0.0, 100.0)
     

        return SoC, E_grid, E_curt, cost, P_bat
        



class Controller:

    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)
    
        self.opt_problem()  # Set up the optimization problem when the controller is initialized

    def obj_SoC(self, x_SoC, x_SoC_wide, x_SoC_narr):
        objective = 0

        #objective += cp.quad_form(x_SoC - SoC_ref, cp.diag(self.Q_SoC))  # Penalize deviation from SoC reference
        objective += cp.quad_form(x_SoC_wide, cp.diag(self.Q_SoC_wide))  # Penalize soft constraint violations (20 - 80)
        objective += cp.quad_form(x_SoC_narr, cp.diag(self.Q_SoC_narr))  # Penalize soft constraint violations (40 - 60) 

        return objective
    
    def var_bounds(self, var_lb, var_ub, var): 
        '''
            Creates bounds constraints for a given variable 
        '''
        cons = []
        
        cons.append(var >= var_lb)  # Lower bound constraint
        cons.append(var <= var_ub)  # Upper bound constraint
        return cons 
    
    def bound_SoC(self, x_SoC, x_SoC_wide, x_SoC_narr):
        cons = []

        # Hard bounds 
        cons.extend(self.var_bounds(self.y_lb, self.y_ub, x_SoC))  # SoC bounds

        # Soft bounds (20 - 80)
        cons.append(self.y_lb_wide - x_SoC_wide <= x_SoC)
        cons.append(x_SoC <= self.y_ub_wide + x_SoC_wide)

        # Soft bounds (40 - 60)
        cons.append(self.y_lb_narr - x_SoC_narr <= x_SoC)
        cons.append(x_SoC <= self.y_ub_narr + x_SoC_narr)


        return cons
    
    def opt_problem(self):
        # Horizons and sampling time
        Np, Nu = self.Np, self.Nu
        ts = self.ts
        # Problem dimensions
        nu, nx = self.nu, self.nx
        # Weights and system model 
        Q_bat, Q_dbat = self.Q_bat, self.Q_dbat
        Q_grid = self.Q_grid
        battery = self.battery
        # Bounds 
        u_lb, u_ub = self.u_lb, self.u_ub
        du_lb, du_up = self.du_lb, self.du_ub


        # Decision variables
        Pgrid = cp.Variable(Np, name = "Pgrid")


        Pbat_pos = cp.Variable((nu, Nu), name = "Pbat_pos", nonneg = True)
        Pbat_neg = cp.Variable((nu, Nu), name = "Pbat_neg", nonpos = True)
        Pbat = Pbat_pos + Pbat_neg  
        setattr(self, "Pbat", Pbat)  # Store Pbat as an attribute 

        x_SoC = cp.Variable((nx, Np + 1), name = "x_SoC", nonneg = True)
        x_SoC_wide = cp.Variable((nx, Np + 1), name = "x_SoC_wide", nonneg = True)
        x_SoC_narr = cp.Variable((nx, Np + 1), name = "x_SoC_narr", nonneg = True)


        # Parameters 
        Ppv = cp.Parameter(Np, name = "Ppv", nonneg = True)
        Pload = cp.Parameter(Np, name = "Pload", nonneg = True)
        tariff = cp.Parameter(Np, name = "tariff", nonneg = True)

        SoC_IC = cp.Parameter(nx, name = "SoC_IC", nonneg = True)
        #SoC_ref = cp.Parameter(nx, name = "SoC_ref", nonneg = True)

        u_past = cp.Parameter(nu, name = "u_past")

        aux = cp.reshape(u_past, (nu, 1), order = "C")  # Reshape u_past to be a column vector
        du = Pbat[:, :] - cp.hstack([aux, Pbat[:, :-1]])  # Change in battery power from the last control input
        setattr(self, "du", du)  # Store du as an attribute

        objective = 0
        cons = []

        cons.append(x_SoC[:, 0] == SoC_IC)  # Initial SoC constraint

        for k in range(Np):
            cons.extend(self.bound_SoC(x_SoC[:, k], x_SoC_wide[:, k], x_SoC_narr[:, k]))  # SoC bounds constraints
            objective += self.obj_SoC(x_SoC[:, k], x_SoC_wide[:, k], x_SoC_narr[:, k])  # SoC objective

         
            #cons.append(Pgrid[k] >= 0)
            Pbought = cp.maximum(Pgrid[k], 0)
            # Minimizing Pbought
            objective += Q_grid*(tariff[k] * Pgrid[k] * ts/3600)**2

            if k < Nu:
                Pbat_k = Pbat[:, k]

                #objective += cp.quad_form(du[:, k], cp.diag(Q_dbat))
                #cons.extend(self.var_bounds(du_lb, du_up, du[:, k]))

                Pbat_pos_k = Pbat_pos[:, k]
                Pbat_neg_k = Pbat_neg[:, k]

                objective += cp.quad_form(Pbat_pos_k, cp.diag(Q_bat))
                objective += cp.quad_form(Pbat_neg_k, cp.diag(Q_bat))

                cons.append(Pbat_pos_k <= u_ub)
                cons.append(Pbat_neg_k >= u_lb)

            else:
                # Use last Pbat input 
                Pbat_k = Pbat[:, -1]
                Pbat_pos_k = Pbat_pos[:, -1]
                Pbat_neg_k = Pbat_neg[:, -1]


                #cons.append(Pbat_k == 0)
            
            dx = battery.simulate(x_SoC[:, k], Pbat_pos_k, Pbat_neg_k, ts)
            cons.append(x_SoC[:, k + 1] == dx)
            

            cons.append(Pload[k] == Pgrid[k] + Ppv[k] - cp.sum(Pbat_k))  # Power balance constraint

        
        self.prob = cp.Problem(cp.Minimize(objective), cons)








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




def load_generator(days, steps, load_dict):
    load = np.zeros(steps)
    for key, value in load_dict.items():
        power = value['power']
        
        if value['type'] == 'continuous':
            for start, end in value['hours']:
                start_i = int(start * 4)
                end_i = int(end * 4)

                if start_i > end_i: # handle loads starting night time for example
                    load[start_i:] += power    # CORRIGIDO AQUI
                    load[:end_i] += power      # CORRIGIDO AQUI
                else:
                    load[start_i:end_i] += power # CORRIGIDO AQUI
                    
        if value['type'] == 'cyclic':
            # converting min to number of time blocks
            t_on = max(1, value['t_on'] // 15)
            t_off = max(1, value['t_off'] // 15)        

            for i in range(0, steps, t_on + t_off):
                end_cycle = min(i + t_on, steps)
                load[i:end_cycle] += power       # CORRIGIDO AQUI
                
    load = np.tile(load, days)
    return load

    





