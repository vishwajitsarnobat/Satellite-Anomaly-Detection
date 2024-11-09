import pandas as pd
import os

def get_unique_dates_in_csv(data_path):
    df = pd.read_csv(data_path, parse_dates=["timestamp"], index_col="timestamp")
    df['date'] = df.index.date
    return set(df['date'].unique())

def find_common_dates(directory_path):
    csv_files = [f for f in os.listdir(directory_path) if f.endswith('.csv')]
    common_dates = None

    for csv_file in csv_files:
        data_path = os.path.join(directory_path, csv_file)
        unique_dates = get_unique_dates_in_csv(data_path)
        
        if common_dates is None:
            common_dates = unique_dates
        else:
            common_dates &= unique_dates

    return common_dates

directory_path = '/home/vishwajitsarnobat/Workspace/SPIT/PDS/data_satellite'
common_dates = find_common_dates(directory_path)

print(len(common_dates))
