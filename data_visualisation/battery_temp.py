import pandas as pd
import matplotlib.pyplot as plt
import matplotlib

matplotlib.use('TkAgg')

data_path = '/home/vishwajitsarnobat/Workspace/SPIT/PDS/Satellite-Anomaly-Detection/data/BatteryTemperature.csv'

df = pd.read_csv(data_path)
df['timestamp'] = pd.to_datetime(df['timestamp'])

df_resampled = df.resample('H', on='timestamp').mean()

print(df_resampled.head())

plt.figure(figsize=(10, 6))
plt.plot(df_resampled.index, df_resampled['battery_temperature'], marker='o', linestyle='-', color='b')
plt.title('Battery Temperature Over Time (Hourly Resampling)')
plt.xlabel('Timestamp')
plt.ylabel('Battery Temperature (°C)')
plt.grid(True)
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

