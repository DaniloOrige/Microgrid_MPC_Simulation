import pandas as pd
import numpy as np
from pathlib import Path


def load_microgrid_data():

    '''
    Creates arrays with inverter data from CSV file and calculates mean values by hour for each variable.
    '''

    base_dir = Path(__file__).parent

    csv_path = base_dir / "inverter2.csv"

    df_raw = pd.read_csv(
        csv_path,
        comment="#",   # ignora as linhas #group, #datatype, #default
    )

    df = df_raw[["_time", "_value", "_field"]].copy()
    df["_time"] = pd.to_datetime(df["_time"],format ="ISO8601" ,utc=True)  # Convert to datetime

    df_wide = df.pivot(index="_time", columns="_field", values="_value").reset_index()   # Pivot to wide format

    pv_power = df_wide["Sun_PV_Power"].to_numpy()
    load_power = df_wide["Total_load_power"].to_numpy()
    grid_power = df_wide["Potencia_total_rede"].to_numpy()
    df_wide = df_wide.reset_index()
    time = df_wide["_time"].to_numpy()

    df_wide["hour"] = df_wide["_time"].dt.hour # day hours column

    vars_of_interest = [
        "Sun_PV_Power",
        "Potencia_total_rede",
        "Total_load_power",
    ]

    mean_by_hour = (
        df_wide
        .groupby("hour")[vars_of_interest]
        .mean()             
        .reset_index()
    )


    hours          = mean_by_hour["hour"].to_numpy()
    pv_hour_mean   = mean_by_hour["Sun_PV_Power"].to_numpy()
    grid_hour_mean = mean_by_hour["Potencia_total_rede"].to_numpy()
    load_hour_mean = mean_by_hour["Total_load_power"].to_numpy()

    #print("Hours:", hours)
    #print("Mean PV Power by Hour:", pv_hour_mean)
    #print("Mean Grid Power by Hour:", grid_hour_mean)
    #print("Mean Load Power by Hour:", load_hour_mean)

    # Linear interpolation to match sampling time (15 minutes)
    interp_PV = np.interp(np.arange(0, 24, 0.25), hours, pv_hour_mean)
    interp_load = np.interp(np.arange(0, 24, 0.25), hours, load_hour_mean)
    interp_grid = np.interp(np.arange(0, 24, 0.25), hours, grid_hour_mean)

    return {
        "pv_power": interp_PV,
        "load_power": interp_load,
        "grid_power": interp_grid,
        "hours": hours,
    }












