# Multi-Layered Model for Satellite Telemetry Anomaly Detection  

## By  
- Vishwajit Sarnobat (2023300195)  
- Arsalan Sayad (2023300197)  
- Harshav Shah (2023300207)  
- Harshal Shah (2023300206)  

### Guided by  
**Abhijeet Salunke Sir (Professor - SPIT, Andheri)**  

---

## Abstract  
Satellite telemetry anomaly detection is crucial for ensuring mission safety and spacecraft reliability. Traditional detection methods are often insufficient to address the complexity and continuous nature of telemetry data. This project presents an AI-driven, real-time anomaly detection system that integrates statistical, machine learning, and deep learning techniques in a multi-layered framework.  

### Key Techniques:  
- **Z-Score Analysis**: Detects deviations from the mean.  
- **Rolling Statistics**: Captures temporal changes over sliding windows.  
- **Robust Covariance**: Identifies multivariate outliers.  
- **Isolation Forest**: Detects patterns significantly different from normal data.  
- **Autoencoders**: Utilizes reconstruction errors for anomaly detection.  

Anomalies are flagged if detected by at least two methods, ensuring robust monitoring of satellite telemetry.  

---

## Problem Statement  
Satellite telemetry data is continuous, complex, and prone to subtle variations that may indicate potential failures. Traditional methods like threshold monitoring and visual inspection fail to detect these irregularities effectively. This project addresses the challenge by building a real-time, autonomous, AI-driven anomaly detection system.  

---

## Objective  
To develop a multi-layered anomaly detection framework combining:  
1. **Statistical Techniques**: Baseline analysis using Z-Score and Rolling Statistics.  
2. **Machine Learning**: Isolation Forest for identifying intricate patterns.  
3. **Deep Learning**: Autoencoders for detecting anomalies through reconstruction errors.  

This integrated system enhances detection accuracy and reliability.  

---

## Motivation  
The growing complexity of satellite missions necessitates advanced monitoring solutions. AI-driven anomaly detection can significantly improve monitoring precision, reduce risks, and increase operational efficiency for organizations like ISRO and NASA.  

---

## Dataset Description  
The dataset comprises five time-series CSV files containing telemetry data:  
- **Battery Temperature**  
- **Bus Voltage**  
- **Total Bus Current**  
- **Wheel RPM**  
- **Wheel Temperature**  

## Methodology  
### Techniques Used:  
- **Z-Score Analysis**: Flags deviations from the mean.  
- **Rolling Statistics**: Captures gradual shifts over time.  
- **Robust Covariance**: Detects multivariate outliers.  
- **Isolation Forest**: Isolates anomalies in high-dimensional data.  
- **Autoencoders**: Flags anomalies using reconstruction errors.  

Anomalies were flagged if detected by at least two techniques, leveraging the strengths of individual methods.  

---

## Results  
- **Isolation Forest** and **Autoencoders** were highly sensitive to both sudden and gradual changes.  
- **Rolling Statistics** captured gradual deviations effectively, while **Z-Score** identified abrupt changes.  
- The hybrid approach delivered robust anomaly detection.  

### Visualizations  
Scatter plots and line graphs highlight detected anomalies over telemetry time series for comparison.  

---

## Tools and Technologies  
- **Programming**: Python  
- **Libraries**: Pandas, NumPy, Scikit-learn, TensorFlow, Matplotlib  
- **Environment**: Google Colab, Fedora OS  
- **Hardware**: Intel i5 (13th Gen), NVIDIA RTX 2050 (4GB VRAM), 16GB RAM  

---

## Limitations  
1. Limited dataset size may affect generalization.  
2. Ensemble methods could further improve performance.  

---

## Conclusion  
This project successfully demonstrates a multi-layered AI-driven approach to real-time satellite telemetry anomaly detection. Future enhancements could include ensemble techniques and larger datasets for improved accuracy.  

---

## Folder Structure  

1. **`data.zip`**: Contains the CSV files for telemetry data.  
2. **`metadata/`**: Includes metadata about the dataset (recording intervals, time range, etc.).  
3. **`codes/`**: Contains the following scripts:  
   - **`count_days.py`**: Calculates the total number of common days across all files.  
   - **`visualisation.py`**: Generates histograms and line plots for data visualization.  
   - **`model.py`**: Contains the anomaly detection code.  
4. **`visualisation_results/`**: Stores visualization outputs.  
5. **`data_features/`**: Extracted features from the dataset for processing.  
6. **`performance_metrices/`**: Performance metrics of the anomaly detection models.  
7. **`final_results/`**: Results of the trained models in line plot format.  
8. **`full_training_logs/`**: Logs from training the models.  

---

## Usage  
1. Extract the `data.zip` file to access the telemetry data in CSV format.  
2. Refer to the `metadata` folder for dataset information.  
3. Run the scripts in the `codes` folder for visualization and model training.  
4. Use outputs from `visualisation_results` and `final_results` for analysis.  

