# Fraud Detection System

## Project Overview

The Fraud Detection System is a Machine Learning project designed to identify fraudulent financial transactions. The model analyzes transaction details and predicts whether a transaction is legitimate or fraudulent.

This project demonstrates the complete machine learning workflow, including data preprocessing, exploratory data analysis, handling class imbalance, model training, evaluation, and deployment using Streamlit.


## Problem Statement

Financial institutions process millions of transactions every day. Detecting fraudulent transactions quickly is essential to reduce financial losses and improve customer security.

The objective of this project is to build a machine learning model that accurately classifies transactions as fraudulent or legitimate.


## Dataset

The dataset contains transaction information with the following features:

- transaction_id
- amount
- transaction_hour
- merchant_category
- foreign_transaction
- location_mismatch
- device_trust_score
- velocity_last_24h
- cardholder_age
- is_fraud (Target Variable)

Target Variable:

- 0 = Legitimate Transaction
- 1 = Fraudulent Transaction


## Technologies Used

- Python
- Jupyter Notebook
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- Imbalanced-learn (SMOTE)
- Joblib
- Streamlit


## Machine Learning Workflow

1. Data Loading
2. Data Cleaning
3. Exploratory Data Analysis
4. Feature Encoding
5. Train-Test Split
6. Handle Class Imbalance using SMOTE
7. Feature Scaling
8. Random Forest Model Training
9. Isolation Forest Model
10. Model Evaluation
11. Save Trained Model
12. Streamlit Deployment


## Project Structure

```
Fraud_Detection_System/

│
├── credit_card_fraud.csv
├── Fraud_Detection.ipynb
├── app.py
├── fraud_detection_model.pkl
├── scaler.pkl
├── README.md
├── requirements.txt
└── report.pdf
```


## Performance Metrics

The model is evaluated using:

- Accuracy
- Precision
- Recall
- F1 Score
- ROC-AUC Score
- Confusion Matrix


## Model Used

- Random Forest Classifier
- Isolation Forest (Anomaly Detection)


## Installation

Clone the repository

```bash
git clone https://github.com/yourusername/Fraud_Detection_System.git
```

Move into the project directory

```bash
cd Fraud_Detection_System
```

Install the required libraries

```bash
pip install -r requirements.txt
```


## Run the Application

Execute the following command:

```bash
streamlit run app.py
```

The application will open in your default web browser.


## Features

- Detect fraudulent transactions
- User-friendly Streamlit interface
- Handles imbalanced data using SMOTE
- Machine Learning based prediction
- Displays fraud probability
- Model saved using Joblib


## Future Improvements

- Deploy the application on Streamlit Cloud
- Add XGBoost model
- Real-time transaction monitoring
- Database integration
- API deployment using Flask or FastAPI


## Author

Shivam Kumar

B.Tech Artificial Intelligence and Data Science


## License

This project is developed for educational and learning purposes.
