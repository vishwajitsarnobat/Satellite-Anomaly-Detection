# Multi-Layered Model for Satellite Telemetry Anomaly Detection  

## By  
- Vishwajit Sarnobat (2023300195)  
- Arsalan Sayad (2023300197)  
- Harshav Shah (2023300207)  
- Harshal Shah (2023300206)  

### Guided by  
**Abhijeet Salunke**  

---

## Abstract  
Satellite telemetry anomaly detection is essential for ensuring mission safety and spacecraft integrity. Telemetry data often presents subtle variations that may indicate critical issues if left undetected. Traditional detection methods are not well-suited for the continuous and complex nature of telemetry data. This project develops an AI-driven, real-time anomaly detection system that integrates statistical, machine learning, and deep learning techniques for a multi-layered detection framework.  

Key methods used:  
- **Z-Score Analysis**: Identifies deviations from the mean.  
- **Rolling Statistics**: Captures temporal shifts over sliding windows.  
- **Robust Covariance**: Isolates outliers in multivariate data.  
- **Isolation Forest**: Flags data points significantly different from normal behavior.  
- **Autoencoders**: Uses reconstruction errors to detect anomalies in telemetry signals.  

Anomalies are flagged if detected by at least two methods. This system demonstrates the potential of AI to improve anomaly detection in high-stakes environments, ensuring reliable, real-time monitoring of satellite telemetry.  

---

## Problem Statement  
Satellite telemetry data is continuous, complex, and often exhibits subtle variations that may point to critical issues. Traditional anomaly detection techniques, like visual inspection and threshold monitoring, fail to capture nuanced patterns or sudden changes in telemetry. This project aims to build a robust, AI-driven system for autonomous and real-time anomaly detection, enabling early detection of potential failures.  

---

## Objective  
To develop a multi-layered anomaly detection system by integrating:  
1. **Statistical Techniques**: Baseline behavior analysis using Z-Score and Rolling Statistics.  
2. **Machine Learning**: Isolation Forest for intricate patterns and outliers.  
3. **Deep Learning**: Autoencoders to identify deviations through reconstruction errors.  

This combination ensures enhanced reliability and actionable insights for telemetry monitoring.  

---

## Motivation  
The increasing complexity of space missions demands advanced anomaly detection systems. Current systems rely heavily on rule-based approaches, which lack the sophistication to detect subtle irregularities. By leveraging AI-driven systems, agencies like ISRO and NASA can enhance monitoring precision, reduce mission risks, and improve cost-efficiency.  

---

## Dataset Description  
The dataset, sourced from an open GitHub repository, includes five CSV files with time-series telemetry data:  
- **Battery Temperature**  
- **Bus Voltage**  
- **Total Bus Current**  
- **Wheel RPM**  
- **Wheel Temperature**  

### Data Characteristics:  
- **Duration**: Common subset from 2001–2018.  
- **Size**: Approximately 500,000–750,000 records per file.  
- **Features**: Time-stamped telemetry readings essential for anomaly detection.  

---

## Preprocessing  
1. **Handling Missing Values**: Addressed using backfilling and front filling to maintain temporal consistency.  
2. **Outlier Management**: Outliers retained as indicators of potential anomalies.  
3. **Data Normalization**: Applied Standard Scaling to standardize features.  

---

## Methodology  

### Techniques Used  
1. **Z-Score**: Flags anomalies based on deviations from the mean.  
2. **Rolling Statistics**: Captures gradual shifts in telemetry behavior.  
3. **Robust Covariance**: Identifies multivariate outliers.  
4. **Isolation Forest**: Isolates anomalies in high-dimensional data.  
5. **Autoencoder**: Flags anomalies via reconstruction errors.  

### Implementation  
Each method independently analyzed telemetry parameters. An anomaly was flagged if detected by at least two methods, ensuring accuracy and reliability.  

---

## Results  
### Key Findings  
- **Isolation Forest** and **Autoencoder** showed the highest sensitivity, detecting both sudden spikes and gradual shifts.  
- **Rolling Statistics** effectively identified gradual deviations, while **Z-Score** was optimal for abrupt changes.  
- The hybrid approach leveraged the strengths of all methods, providing robust anomaly detection.  

### Visualizations  
Scatter plots and line graphs were used to overlay detected anomalies on the telemetry time series for comparative analysis.  

---

## Tools and Technologies  
- **Programming**: Python  
- **Libraries**: Pandas, NumPy, Scikit-learn, TensorFlow, Matplotlib  
- **Environment**: Google Colab, Fedora OS  
- **Hardware**: Intel i5 (13th Gen), NVIDIA RTX 2050 (4GB VRAM), 16GB RAM  

---

## Limitations  
1. Limited data size affects the ability to generalize patterns.  
2. Independent model assessments could benefit from ensemble techniques.  

---

## Conclusion  
This project successfully demonstrates a multi-layered approach to satellite telemetry anomaly detection. By combining statistical, machine learning, and deep learning techniques, the system identifies subtle anomalies in real-time. Future improvements could involve ensemble methods and larger datasets for enhanced accuracy.  

---  

### References  
- GitHub Repository: [Link to dataset]  
- WebTCAD and LASP Systems: [Link to more information]  

