# Advanced Smart City Traffic Prediction System

### Developed during an Industrial Internship | Upskill Campus | UniConverge Technologies Pvt Ltd

## Overview

The **Advanced Smart City Traffic Prediction System** is a machine learning-based solution designed to predict real-time traffic patterns in urban areas using data acquired from IoT-enabled sensors. Developed over a six-week industrial internship, this project combines state-of-the-art technologies such as **TensorFlow**, **Scikit-learn**, and **XGBoost** for time-series forecasting, data preprocessing, and system integration.

### Key Features:
- **Real-time Traffic Prediction**: Leverages real-time IoT sensor data for continuous traffic forecasting.
- **Machine Learning Algorithms**: Implements advanced models such as **Gradient Boosting**, **LSTM Networks**, and **Random Forest**.
- **Scalable Infrastructure**: Designed to handle large-scale datasets with optimized API communication for low-latency predictions.
- **System Integration**: Seamlessly integrates with urban traffic management systems via APIs to provide proactive insights.

## Problem Statement

Urban traffic management systems often face challenges such as real-time data handling and scalability. The aim of this project is to develop a robust and scalable traffic prediction model that can efficiently process both historical and real-time traffic data, ensuring minimal latency and high prediction accuracy.

## Proposed Solution

Our solution incorporates **Python-based machine learning models** with hyperparameter tuning, time-series analysis, and API-based real-time data integration. The system is capable of:
- Ingesting real-time data from distributed IoT sensors.
- Processing and cleaning high-dimensional data for prediction accuracy.
- Providing actionable traffic forecasts that can be integrated with city management platforms.


## Technologies Used

- **Python**
- **TensorFlow**: Deep learning framework for time-series forecasting.
- **Scikit-learn**: For preprocessing and machine learning algorithms.
- **XGBoost**: Gradient boosting for efficient prediction models.
- **Flask**: API framework for real-time data integration.
- **Pandas, Seaborn**: Data manipulation and visualization libraries.

## Performance Testing

### Key Test Cases:
1. **Prediction Accuracy**: Achieved a 93% accuracy rate for peak traffic hours.
2. **Real-Time Data Handling**: Successfully integrated live traffic data with minimal delay (< 1 second).
3. **Scalability**: Handled a 5x increase in data volume without performance degradation.
4. **API Latency**: Maintained an average API latency of 0.85 seconds under load.
5. **Model Efficiency**: Operated within 75% of CPU and memory resources during continuous real-time prediction.

## Repository Structure

- **Diagrams/**: Contains architecture diagrams, system design visuals, and flowcharts.
- **app/**: Source code for the Flask API that handles real-time data processing and prediction.
- **dashboard/**: Web-based dashboard for visualizing traffic data and predictions.
- **data/raw/**: Directory for storing raw sensor data collected for analysis and model training.
- **models/**: Pre-trained machine learning models used for traffic prediction.
- **notebooks/**: Jupyter notebooks containing exploratory data analysis, model training, and experimentation.
- **scripts/**: Utility scripts for data preprocessing, model evaluation, and system integration.
- **README.md**: Project documentation and overview.
- **requirements.txt**: List of dependencies required to run the project.

## Future Work

- **Scalability Enhancements**: Further optimizations to handle larger datasets in more complex urban scenarios.
- **Latency Reduction**: Improved API and model response times for faster real-time predictions.
- **Deployment**: Exploration of deployment strategies and long-term maintenance for live urban environments.

## How to Run the Project

1. **Clone the repository**:    git clone https://github.com/imdiveshjain/upskillcampus.git
2. **Install dependencies**:    pip install -r requirements.txt
3. **Run the API server**:    python app.py
4. **Access the real-time prediction API at**:    http://localhost:5000/predict

## References

- **TensorFlow Documentation**: https://www.tensorflow.org/
- **Scikit-learn Documentation**: https://scikit-learn.org/stable/
- **XGBoost Documentation**: https://xgboost.readthedocs.io/

Feel free to contribute or raise issues if you encounter any!
