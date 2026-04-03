# Grade-Prediction-Using-Machine-Learning-Algorithm

# Pass/Fail Prediction ML Application

## Overview
A web-based machine learning application for training classification models and predicting pass/fail outcomes. Built with Streamlit and scikit-learn.

## Project Structure
- `app.py` - Main Streamlit application with ML functionality
- `.streamlit/config.toml` - Streamlit server configuration
- `pyproject.toml` - Python dependencies

## Features
- **Data Loading**: Upload CSV files or use built-in sample dataset
- **Model Training**: Support for Logistic Regression and Random Forest
- **Performance Metrics**: Accuracy, Precision, Recall, F1 Score, Confusion Matrix
- **Interactive Predictions**: Enter values to get real-time predictions
- **Visualizations**: Feature importance charts and prediction probabilities

## Running the Application
```bash
streamlit run app.py --server.port 5000
```

## Dependencies
- streamlit
- scikit-learn
- pandas
- numpy
- matplotlib
- seaborn

## Recent Changes
- December 08, 2025: Initial creation of ML pass/fail prediction app
