import os
import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.covariance import EllipticEnvelope
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from silence_tensorflow import silence_tensorflow
silence_tensorflow()
import tensorflow as tf
from tensorflow.keras import layers, Model
import matplotlib.pyplot as plt
from scipy import stats


def preprocess_data(data, value_col='value'):
    print(f"{'='*50}\nPreprocessing Data for Column: {value_col}\n{'='*50}")
    data[value_col] = data[value_col].fillna(method='ffill').fillna(method='bfill')
    scaler = StandardScaler()
    data[value_col] = scaler.fit_transform(data[value_col].values.reshape(-1, 1))
    print(f"Data standardized using StandardScaler.")
    return data, scaler

def zscore_detection(data, value_col='value', threshold=3):
    print(f"{'='*50}\nZ-Score Detection\n{'='*50}")
    z_scores = np.abs(stats.zscore(data[value_col]))
    anomalies = z_scores > threshold
    data['zscore_anomaly'] = anomalies
    data['zscore_score'] = z_scores
    print(f"Detected {anomalies.sum()} anomalies using Z-Score (Threshold={threshold}).")
    return data

def isolation_forest_detection(data, value_col='value', contamination=0.1):
    print(f"{'='*50}\nIsolation Forest Detection\n{'='*50}")
    iso_forest = IsolationForest(contamination=contamination, random_state=42)
    values = data[value_col].values.reshape(-1, 1)
    anomalies = iso_forest.fit_predict(values)
    scores = iso_forest.score_samples(values)
    data['iforest_anomaly'] = anomalies == -1
    data['iforest_score'] = scores
    print(f"Detected {anomalies.sum()} anomalies using Isolation Forest (Contamination={contamination}).")
    return data

def robust_covariance_detection(data, value_col='value', contamination=0.1):
    print(f"{'='*50}\nRobust Covariance Detection\n{'='*50}")
    robust_cov = EllipticEnvelope(contamination=contamination, random_state=42)
    values = data[value_col].values.reshape(-1, 1)
    anomalies = robust_cov.fit_predict(values)
    scores = robust_cov.score_samples(values)
    data['robust_cov_anomaly'] = anomalies == -1
    data['robust_cov_score'] = scores
    print(f"Detected {anomalies.sum()} anomalies using Robust Covariance (Contamination={contamination}).")
    return data

def rolling_statistics_detection(data, value_col='value', window=24, threshold=3):
    print(f"{'='*50}\nRolling Statistics Detection\n{'='*50}")
    rolling_mean = data[value_col].rolling(window=window).mean()
    rolling_std = data[value_col].rolling(window=window).std()
    z_scores = np.abs((data[value_col] - rolling_mean) / rolling_std)
    anomalies = z_scores > threshold
    data['rolling_anomaly'] = anomalies
    data['rolling_score'] = z_scores
    print(f"Detected {anomalies.sum()} anomalies using Rolling Statistics (Window={window}, Threshold={threshold}).")
    return data

def build_autoencoder(data, value_col='value', sequence_length=24):
    print(f"{'='*50}\nAutoencoder Detection\n{'='*50}")
    sequences = []
    for i in range(len(data) - sequence_length + 1):
        sequences.append(data[value_col].values[i:(i + sequence_length)])
    sequences = np.array(sequences).reshape(-1, sequence_length, 1)
    
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
    mse = np.mean(np.power(sequences - reconstructed_sequences, 2), axis=(1, 2))
    threshold = np.percentile(mse, 95)
    
    data['autoencoder_score'] = np.nan
    data['autoencoder_anomaly'] = False
    data.iloc[sequence_length-1:, -2] = mse
    data.iloc[sequence_length-1:, -1] = mse > threshold
    print(f"Detected {np.sum(mse > threshold)} anomalies using Autoencoder.")
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
    
    # Evaluate individual model performance using ensemble anomalies as ground truth
    true_anomalies = data['ensemble_anomaly']
    
    z_score_acc = accuracy_score(true_anomalies, data['zscore_anomaly'])
    z_score_precision = precision_score(true_anomalies, data['zscore_anomaly'])
    z_score_recall = recall_score(true_anomalies, data['zscore_anomaly'])
    z_score_f1 = f1_score(true_anomalies, data['zscore_anomaly'])
    print(f"Z-Score Performance: Accuracy={z_score_acc:.2f}, Precision={z_score_precision:.2f}, Recall={z_score_recall:.2f}, F1-Score={z_score_f1:.2f}")
    
    iforest_acc = accuracy_score(true_anomalies, data['iforest_anomaly'])
    iforest_precision = precision_score(true_anomalies, data['iforest_anomaly'])
    iforest_recall = recall_score(true_anomalies, data['iforest_anomaly'])
    iforest_f1 = f1_score(true_anomalies, data['iforest_anomaly'])
    print(f"Isolation Forest Performance: Accuracy={iforest_acc:.2f}, Precision={iforest_precision:.2f}, Recall={iforest_recall:.2f}, F1-Score={iforest_f1:.2f}")
    
    robust_cov_acc = accuracy_score(true_anomalies, data['robust_cov_anomaly'])
    robust_cov_precision = precision_score(true_anomalies, data['robust_cov_anomaly'])
    robust_cov_recall = recall_score(true_anomalies, data['robust_cov_anomaly'])
    robust_cov_f1 = f1_score(true_anomalies, data['robust_cov_anomaly'])
    print(f"Robust Covariance Performance: Accuracy={robust_cov_acc:.2f}, Precision={robust_cov_precision:.2f}, Recall={robust_cov_recall:.2f}, F1-Score={robust_cov_f1:.2f}")
    
    rolling_acc = accuracy_score(true_anomalies, data['rolling_anomaly'])
    rolling_precision = precision_score(true_anomalies, data['rolling_anomaly'])
    rolling_recall = recall_score(true_anomalies, data['rolling_anomaly'])
    rolling_f1 = f1_score(true_anomalies, data['rolling_anomaly'])
    print(f"Rolling Statistics Performance: Accuracy={rolling_acc:.2f}, Precision={rolling_precision:.2f}, Recall={rolling_recall:.2f}, F1-Score={rolling_f1:.2f}")
    
    autoencoder_acc = accuracy_score(true_anomalies, data['autoencoder_anomaly'])
    autoencoder_precision = precision_score(true_anomalies, data['autoencoder_anomaly'])
    autoencoder_recall = recall_score(true_anomalies, data['autoencoder_anomaly'])
    autoencoder_f1 = f1_score(true_anomalies, data['autoencoder_anomaly'])
    print(f"Autoencoder Performance: Accuracy={autoencoder_acc:.2f}, Precision={autoencoder_precision:.2f}, Recall={autoencoder_recall:.2f}, F1-Score={autoencoder_f1:.2f}")
    
    ensemble_acc = accuracy_score(true_anomalies, data['ensemble_anomaly'])
    ensemble_precision = precision_score(true_anomalies, data['ensemble_anomaly'])
    ensemble_recall = recall_score(true_anomalies, data['ensemble_anomaly'])
    ensemble_f1 = f1_score(true_anomalies, data['ensemble_anomaly'])
    print(f"Ensemble Performance: Accuracy={ensemble_acc:.2f}, Precision={ensemble_precision:.2f}, Recall={ensemble_recall:.2f}, F1-Score={ensemble_f1:.2f}")
    
    return data

def data_explore(parameter, path):
    print(f"{'='*50}\nExploring Data: {parameter}\n{'='*50}")
    df = pd.read_csv(path)
    data = df['value']
    print(f"Total Observations       : {len(data)}")
    print(f"Min Value               : {min(data):.2f}")
    print(f"Max Value               : {max(data):.2f}")
    print(f"Mean Value              : {data.mean():.2f}")
    print(f"Standard Deviation      : {data.std():.2f}")
    print(f"Median                  : {data.median():.2f}\n")

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
    file_paths = {file.split('.')[0]: os.path.join(directory_path, file) for file in os.listdir(directory_path) if file.endswith('.csv') and not file.startswith('Wheel')}
    print(file_paths)
    for param, file_path in file_paths.items():
        data_explore(param, file_path)

    results = {}
    for param, file_path in file_paths.items():
        print(f"Processing {param}...")
        results[param] = process_telemetry_file(file_path, param_name=param)
    print("All files processed.")
    return results

directory_path = '/home/vishwajitsarnobat/Workspace/SPIT/PDS/data_satellite'
results = analyze_all_parameters(directory_path)