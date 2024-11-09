import os
import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.covariance import EllipticEnvelope
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow.keras import layers, Model
import matplotlib.pyplot as plt
from scipy import stats

def preprocess_data(data, value_col='value'):
    data[value_col] = data[value_col].fillna(method='ffill').fillna(method='bfill')
    scaler = StandardScaler()
    data[value_col] = scaler.fit_transform(data[value_col].values.reshape(-1, 1))
    return data, scaler

def zscore_detection(data, value_col='value', threshold=3):
    print("Running Z-Score Detection...")
    z_scores = np.abs(stats.zscore(data[value_col]))
    anomalies = z_scores > threshold
    data['zscore_anomaly'] = anomalies
    data['zscore_score'] = z_scores
    print(f"Z-Score anomalies detected: {anomalies.sum()} out of {len(data)} points")
    return data

def isolation_forest_detection(data, value_col='value', contamination=0.1):
    print("Running Isolation Forest Detection...")
    iso_forest = IsolationForest(contamination=contamination, random_state=42)
    values = data[value_col].values.reshape(-1, 1)
    anomalies = iso_forest.fit_predict(values)
    scores = iso_forest.score_samples(values)
    data['iforest_anomaly'] = anomalies == -1
    data['iforest_score'] = scores
    print(f"Isolation Forest detected {anomalies.sum()} anomalies")
    return data

def robust_covariance_detection(data, value_col='value', contamination=0.1):
    print("Running Robust Covariance Detection...")
    robust_cov = EllipticEnvelope(contamination=contamination, random_state=42)
    values = data[value_col].values.reshape(-1, 1)
    anomalies = robust_cov.fit_predict(values)
    scores = robust_cov.score_samples(values)
    data['robust_cov_anomaly'] = anomalies == -1
    data['robust_cov_score'] = scores
    print(f"Robust Covariance detected {anomalies.sum()} anomalies")
    return data

def rolling_statistics_detection(data, value_col='value', window=24, threshold=3):
    print("Running Rolling Statistics Detection...")
    rolling_mean = data[value_col].rolling(window=window).mean()
    rolling_std = data[value_col].rolling(window=window).std()
    z_scores = np.abs((data[value_col] - rolling_mean) / rolling_std)
    anomalies = z_scores > threshold
    data['rolling_anomaly'] = anomalies
    data['rolling_score'] = z_scores
    print(f"Rolling statistics detected {anomalies.sum()} anomalies")
    return data

def build_autoencoder(data, value_col='value', sequence_length=24):
    print("Running Autoencoder Anomaly Detection...")
    sequences = []
    for i in range(len(data) - sequence_length + 1):
        sequences.append(data[value_col].values[i:(i + sequence_length)])
    sequences = np.array(sequences).reshape(-1, sequence_length, 1)
    
    # Define the autoencoder model
    input_layer = layers.Input(shape=(sequence_length, 1))
    encoded = layers.LSTM(16, activation='relu')(input_layer)
    encoded = layers.Dense(8, activation='relu')(encoded)
    decoded = layers.Dense(16, activation='relu')(encoded)
    decoded = layers.RepeatVector(sequence_length)(decoded)
    decoded = layers.LSTM(16, activation='relu', return_sequences=True)(decoded)
    decoded = layers.TimeDistributed(layers.Dense(1))(decoded)
    
    autoencoder = Model(input_layer, decoded)
    autoencoder.compile(optimizer='adam', loss='mse')
    autoencoder.fit(sequences, sequences, epochs=50, batch_size=32, verbose=0)
    
    reconstructed_sequences = autoencoder.predict(sequences)
    mse = np.mean(np.power(sequences - reconstructed_sequences, 2), axis=(1,2))
    threshold = np.percentile(mse, 95)
    
    data['autoencoder_score'] = np.nan
    data['autoencoder_anomaly'] = False
    data.iloc[sequence_length-1:, -2] = mse
    data.iloc[sequence_length-1:, -1] = mse > threshold
    print(f"Autoencoder detected {np.sum(mse > threshold)} anomalies")
    return data

def ensemble_detection(data):
    print("Running Ensemble Detection...")
    data = zscore_detection(data)
    data = isolation_forest_detection(data)
    data = robust_covariance_detection(data)
    data = rolling_statistics_detection(data)
    data = build_autoencoder(data)
    
    anomaly_columns = [col for col in data.columns if col.endswith('_anomaly')]
    data['ensemble_anomaly'] = data[anomaly_columns].sum(axis=1) >= 2
    print(f"Ensemble detection identified {data['ensemble_anomaly'].sum()} anomalies")
    return data

def plot_results(data, timestamp_col='timestamp', value_col='value', param_name="parameter"):
    print("Plotting and saving results...")
    plt.figure(figsize=(15, 10))
    plt.subplot(2, 1, 1)
    plt.plot(data[timestamp_col], data[value_col], 'b-', label='Original Data')
    
    colors = ['red', 'green', 'purple', 'orange', 'brown']
    methods = ['zscore', 'iforest', 'robust_cov', 'rolling', 'autoencoder']
    for method, color in zip(methods, colors):
        anomalies = data[data[f'{method}_anomaly']]
        plt.scatter(anomalies[timestamp_col], anomalies[value_col], 
                    c=color, label=f'{method.capitalize()} Anomalies', alpha=0.5)
    
    plt.title(f'Anomaly Detection Results for {param_name}')
    plt.legend()

    plt.subplot(2, 1, 2)
    plt.plot(data[timestamp_col], data[value_col], 'b-', label='Original Data')
    ensemble_anomalies = data[data['ensemble_anomaly']]
    plt.scatter(ensemble_anomalies[timestamp_col], ensemble_anomalies[value_col], 
                c='red', label='Ensemble Anomalies', alpha=0.5)
    
    plt.title(f'Ensemble Anomaly Detection Results for {param_name}')
    plt.legend()
    plt.tight_layout()
    
    output_path = f"{param_name}_anomaly_detection.png"
    plt.savefig(output_path)
    print(f"Plot saved as {output_path}")
    plt.show()

def process_telemetry_file(file_path, param_name="parameter"):
    print(f"Processing file: {file_path}...")
    data = pd.read_csv(file_path)
    data['timestamp'] = pd.to_datetime(data['timestamp'])
    data = data.sort_values('timestamp')
    
    data, scaler = preprocess_data(data)
    
    results = ensemble_detection(data)
    plot_results(results, param_name=param_name)
    return results

def analyze_all_parameters(directory_path):
    print(f"Analyzing all parameters in directory: {directory_path}...")
    file_paths = {file.split('.')[0]: os.path.join(directory_path, file) for file in os.listdir(directory_path) if file.endswith('.csv')}
    results = {}
    for param, file_path in file_paths.items():
        print(f"Processing {param}...")
        results[param] = process_telemetry_file(file_path, param_name=param)
    print("All files processed.")
    return results

directory_path = '/home/vishwajitsarnobat/Workspace/SPIT/PDS/data_satellite'

results = analyze_all_parameters(directory_path)