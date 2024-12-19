# Enhancement of Military Industrial Control Systems Using Deep Learning Techniques

## Project Overview
This project aims to enhance the cybersecurity of military Supervisory Control and Data Acquisition (SCADA) systems through the application of deep learning (DL) techniques. SCADA systems play a pivotal role in managing and monitoring essential operations within military infrastructures, making them prime targets for cyber threats. To address these vulnerabilities, this project proposes a dual-model framework focused on attack prediction and anomaly detection.

## Project Goals and Objectives
The main goal of this project is to bolster the cybersecurity of military SCADA systems. The following specific objectives have been set to achieve this goal:
- Identify and characterize the cybersecurity vulnerabilities specific to military SCADA systems.
- Review and critique existing cybersecurity measures for SCADA systems.
- Develop a deep learning-based cybersecurity framework for military SCADA systems.
- Propose and validate two deep learning models: one for attack prediction and one for anomaly detection.

## Problem Specification and Motivation
### Challenges Faced by Military SCADA Systems
1. **Cybersecurity Threats**: Military SCADA systems face threats from state-sponsored cyber espionage, terrorist attacks, and ransomware threats.
2. **Inherent Vulnerabilities**: Legacy components, lack of encryption, and integration with IT networks increase exposure to attacks.
3. **Operational Importance**: Military SCADA systems support critical national defense infrastructure, which requires continuous operation and high reliability.

### Motivation
Addressing these challenges necessitates the development of proactive and intelligent cybersecurity measures. Deep learning provides a means to identify, predict, and respond to cyber threats with improved accuracy and speed. This project aims to shift military SCADA systems from a reactive to a proactive defense posture.

## Proposed Models
### 1. **Attack Prediction Model**
- **Objective**: Predict future cyber-attacks in military SCADA systems.
- **Dataset Used**: BATADAL (Battle of the Attack Detection Algorithms) dataset.
- **Approach**: A deep neural network (DNN) is trained using labeled operational and attack scenarios.
- **Implementation**: The model predicts attack types and triggers early warnings, enabling operators to act before an attack unfolds.

### 2. **Anomaly Detection Model**
- **Objective**: Detect anomalies in real-time operations that could indicate cyber threats.
- **Dataset Used**: Secure Water Treatment (SWaT) dataset.
- **Approach**: The model integrates Long Short-Term Memory (LSTM) networks to analyze and detect deviations from normal operating conditions.
- **Implementation**: Anomalies are flagged in real-time, enabling immediate action to contain cyber threats.

## Methodology
1. **Data Collection**: Use the BATADAL and SWaT datasets for training and testing the models.
2. **Model Development**:
   - Design and implement DNN for attack prediction.
   - Develop an LSTM-based model for anomaly detection.
3. **Training and Testing**: Train models on datasets and validate their performance using test data.
4. **Evaluation**: Evaluate models using accuracy, precision, recall, and F1 score metrics.
5. **Deployment**: Develop a framework to integrate these models into military SCADA systems for real-time cybersecurity monitoring.

## Key Features of the Project
- **Dual-Model Approach**: Separate models for attack prediction and anomaly detection.
- **Real-Time Detection**: Anomalies and attacks are detected in real-time.
- **Machine Learning Pipelines**: End-to-end ML pipelines are established for data preprocessing, model training, and model deployment.
- **Explainability**: Model decisions are explainable, ensuring transparency in attack prediction and anomaly detection.

## Datasets Used
1. **BATADAL Dataset**
   - Description: Data from water distribution system simulations including normal operations and cyber-attack scenarios.
   - Features: Flow rates, pressure sensors, valve positions, pump control, and attack flags.
   - Purpose: Used to train the attack prediction model.

2. **SWaT Dataset**
   - Description: Secure Water Treatment testbed dataset with normal operational data and attack scenarios.
   - Features: Sensor and actuator signals from a water treatment plant.
   - Purpose: Used to train the anomaly detection model.

## Results and Performance
- **Attack Prediction**: The DNN model achieved high accuracy in predicting attack types, demonstrating its capability for early warning.
- **Anomaly Detection**: The LSTM model effectively identified anomalies, showcasing its ability to detect deviations from normal behavior.

## Tools and Technologies
- **Programming Languages**: Python (TensorFlow, Keras, NumPy, Pandas, Scikit-learn).
- **Machine Learning**: Neural Networks, LSTM.
- **Datasets**: BATADAL and SWaT datasets.
- **Evaluation Metrics**: Accuracy, precision, recall, F1 score.

## Challenges and Limitations
- **Data Quality**: Ensuring the datasets are free of noise and errors is critical for model training.
- **Adversarial Attacks**: DL models are susceptible to adversarial inputs, requiring robust defenses.
- **Scalability**: Ensuring the models scale effectively to large SCADA environments is a key focus.

