import numpy as np
import matplotlib.pyplot as plt
import pandas as pd







def solar_base_curve(hours, sunrise=6.0, sunset=19.0, alpha=3):
    """
    Normalized base curve (0, 1)
    """
    base = np.zeros_like(hours, dtype=float)
    mask = (hours >= sunrise) & (hours <= sunset)

    base[mask] = np.sin(np.pi * ((hours[mask] - sunrise) / (sunset - sunrise))) ** alpha

    return base

def add_clouds(factor, hours, rng, events, amp_min, amp_max, min_width, max_width):
    """
    Adds drops in PV due to clouds. 
    """
    out = factor.copy()

    for _ in range(events):
        center = rng.uniform(8.0, 16.5)
        amplitude = rng.uniform(amp_min, amp_max)
        width = rng.uniform(min_width, max_width)

        dip = 1.0 - amplitude * np.exp(-0.5 * ((hours - center) / width) ** 2)
        out *= dip

    return out

def pv_generator(day, nominal_Ppv, ts_min=15, seed=42):
    """
    Returns a DF with Ppv in 24h
    """
    rng = np.random.default_rng(seed)

    n = int(24 * 60 / ts_min)
    hours = np.arange(n) * ts_min / 60.0

    base = solar_base_curve(hours)

    if day == "sunny":
        climate = 0.97 + 0.01 * rng.normal(size=n)
        climate = np.clip(climate, 0.93, 1.00)
        climate = add_clouds(climate, hours, rng, events=1,
                                 amp_min=0.03, amp_max=0.08,
                                 min_width=0.15, max_width=0.35)

    elif day == "average":
        climate = 0.62 + 0.05 * rng.normal(size=n)
        climate = np.clip(climate, 0.45, 0.75)
        climate = add_clouds(climate, hours, rng, events=3,
                                 amp_min=0.15, amp_max=0.35,
                                 min_width=0.20, max_width=0.60)

    elif day == "cloudy":
        climate = 0.8 + 0.05 * rng.normal(size=n)
        climate = np.clip(climate, 0.02, 0.20)

        # small openings in the sky
        for _ in range(5):
            center = rng.uniform(10.0, 15.0)
            width = rng.uniform(0.20, 0.50)
            opening = rng.uniform(0.03, 0.08) * np.exp(-0.5 * ((hours - center) / width) ** 2)
            climate += opening

        climate = np.clip(climate, 0.02, 0.25)

    else:
        raise ValueError("day must be: 'sunny', 'average' ou 'cloudy'")

    Ppv = nominal_Ppv * base * climate
    Ppv[base == 0] = 0.0  # garante zero à noite

    df = pd.DataFrame({
        "hour": hours,
        "Ppv": Ppv,
        "solar_base": base,
        "climate_factor": climate
    })

    return df

# # Generating 3 different days
# df_sunny = pv_generator('sunny', seed=1)
# df_average = pv_generator('average', seed=2)
# df_cloudy = pv_generator('cloudy', seed=3)

# # Plot
# plt.figure(figsize=(10,5))
# plt.plot(df_sunny["hour"], df_sunny["Ppv"], label="Ensolarado")
# plt.plot(df_average["hour"],      df_average["Ppv"],      label="Médio")
# plt.plot(df_cloudy["hour"],    df_cloudy["Ppv"],    label="Nublado/chuva")
# plt.xlabel("Hora do dia")
# plt.ylabel("Potência FV [kW]")
# plt.title("Perfis de geração fotovoltaica")
# plt.grid(True)
# plt.legend()
# plt.show()




def load_generator(days, steps, load_dict):
    load = np.zeros(steps)
    for key, value in load_dict.items():
        power = value['power']
        
        if value['type'] == 'continuous':
            for start, end in value['hours']:
                start_i = int(start * 4)
                end_i = int(end * 4)

                if start_i > end_i: # handle loads starting night time for example
                    load[start_i:] += power    
                    load[:end_i] += power      
                else:
                    load[start_i:end_i] += power 
                    
        if value['type'] == 'cyclic':
            # converting min to number of time blocks
            t_on = max(1, value['t_on'] // 15)
            t_off = max(1, value['t_off'] // 15)        

            for i in range(0, steps, t_on + t_off):
                end_cycle = min(i + t_on, steps)
                load[i:end_cycle] += power       
                
    load = np.tile(load, days)
    return load

    


































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