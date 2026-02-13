import pandas as pd
import numpy as np

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



