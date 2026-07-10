import streamlit as st
import pandas as pd
import joblib

# Load the trained model and scaler
model = joblib.load("fraud_detection_model.pkl")
scaler = joblib.load("scaler.pkl")

# Page configuration
st.set_page_config(page_title="Fraud Detection System")

st.title("Fraud Detection System")
st.write("Enter the transaction details below to predict whether the transaction is Fraudulent or Legitimate.")

# User Inputs
amount = st.number_input("Transaction Amount", min_value=0.0, value=100.0)

transaction_hour = st.slider("Transaction Hour", 0, 23, 12)

merchant_category = st.selectbox(
    "Merchant Category",
    [
        "Electronics",
        "Entertainment",
        "Grocery",
        "Healthcare",
        "Restaurant",
        "Shopping",
        "Travel"
    ]
)

foreign_transaction = st.selectbox(
    "Foreign Transaction",
    ["No", "Yes"]
)

location_mismatch = st.selectbox(
    "Location Mismatch",
    ["No", "Yes"]
)

device_trust_score = st.slider(
    "Device Trust Score",
    min_value=0.0,
    max_value=1.0,
    value=0.80
)

velocity_last_24h = st.number_input(
    "Transactions in Last 24 Hours",
    min_value=0,
    value=5
)

cardholder_age = st.number_input(
    "Cardholder Age",
    min_value=18,
    max_value=100,
    value=30
)

# Encoding dictionaries
merchant_map = {
    "Electronics": 0,
    "Entertainment": 1,
    "Grocery": 2,
    "Healthcare": 3,
    "Restaurant": 4,
    "Shopping": 5,
    "Travel": 6
}

foreign_map = {
    "No": 0,
    "Yes": 1
}

location_map = {
    "No": 0,
    "Yes": 1
}

# Prediction
if st.button("Predict"):

    input_data = pd.DataFrame({
        "amount": [amount],
        "transaction_hour": [transaction_hour],
        "merchant_category": [merchant_map[merchant_category]],
        "foreign_transaction": [foreign_map[foreign_transaction]],
        "location_mismatch": [location_map[location_mismatch]],
        "device_trust_score": [device_trust_score],
        "velocity_last_24h": [velocity_last_24h],
        "cardholder_age": [cardholder_age]
    })

    # Scale the input
    input_scaled = scaler.transform(input_data)

    # Predict
    prediction = model.predict(input_scaled)[0]
    probability = model.predict_proba(input_scaled)[0][1]

    st.subheader("Prediction Result")

    if prediction == 1:
        st.error("Fraudulent Transaction")
    else:
        st.success("Legitimate Transaction")

    st.write("Fraud Probability: {:.2f}%".format(probability * 100))