import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib
matplotlib.use('TkAgg')

sns.set(style="whitegrid")

def preprocess_data(data_path):
    df = pd.read_csv(data_path, parse_dates=["timestamp"], index_col="timestamp")
    df.fillna(method='ffill', inplace=True)
    return df

def smooth_data(df_resampled, window_size=7):
    return df_resampled['value'].rolling(window=window_size, center=True).mean()

def save_plot(filename):
    plt.tight_layout()
    plt.savefig(filename, format='png')
    plt.close()

def line_plot_visualise(data_path, val):
    df_resampled = preprocess_data(data_path)
    smoothed_values = smooth_data(df_resampled)
    plt.figure(figsize=(14, 8))
    sns.lineplot(data=smoothed_values, color='blue', linewidth=2)
    plt.title(f"Smoothed Daily Average of {val} Over Time", fontsize=16)
    plt.xlabel("Timestamp", fontsize=14)
    plt.ylabel(f"{val}", fontsize=14)
    plt.xticks(rotation=45, ha='right', fontsize=12)
    plt.yticks(fontsize=12)
    save_plot(f"{val}_line_plot.png")

def hist_visualise(data_path, val):
    df_resampled = preprocess_data(data_path)
    plt.figure(figsize=(14, 8))
    sns.histplot(df_resampled['value'], bins=50, color='blue', kde=True)
    plt.title(f"Histogram of Daily Average of {val}", fontsize=16)
    plt.xlabel(f"{val}", fontsize=14)
    plt.ylabel("Frequency", fontsize=14)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    save_plot(f"{val}_histogram.png")

data_parameters = ["BatteryTemperature", "BusVoltage", "TotalBusCurrent", "WheelRPM", "WheelTemperature"]
for data_parameter in data_parameters:
    data_path = f'/home/vishwajitsarnobat/Workspace/SPIT/PDS/Satellite-Anomaly-Detection/data/{data_parameter}.csv'
    line_plot_visualise(data_path, data_parameter)
    hist_visualise(data_path, data_parameter)
