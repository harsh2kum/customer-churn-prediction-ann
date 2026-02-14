import streamlit as st
import pandas as pd
import numpy as np
import pickle
import tensorflow as tf

# Load model and preprocessors
model = tf.keras.models.load_model("model.h5")

with open("onehot_encoder_geo.pkl", "rb") as f:
    onehot_encoder_geo = pickle.load(f)

with open("label_encoder_gender.pkl", "rb") as f:
    label_encoder_gender = pickle.load(f)

with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

st.title("Customer Churn Prediction")

# User inputs
credit_score = st.number_input("Credit Score", value=600)
geography = st.selectbox("Geography", ["France", "Germany", "Spain"])
gender = st.selectbox("Gender", ["Male", "Female"])
age = st.number_input("Age", value=40)
tenure = st.number_input("Tenure", value=3)
balance = st.number_input("Balance", value=60000.0)
num_products = st.number_input("Number of Products", value=2)
has_cr_card = st.selectbox("Has Credit Card", [0, 1])
is_active = st.selectbox("Is Active Member", [0, 1])
salary = st.number_input("Estimated Salary", value=50000.0)

# Encode Geography
geo_encoded = onehot_encoder_geo.transform([[geography]]).toarray()
geo_columns = onehot_encoder_geo.get_feature_names_out(["Geography"])
geo_df = pd.DataFrame(geo_encoded, columns=geo_columns)

# Encode Gender
gender_encoded = label_encoder_gender.transform([gender])[0]

# Create input dataframe
input_data = pd.DataFrame({
    "CreditScore": [credit_score],
    "Gender": [gender_encoded],
    "Age": [age],
    "Tenure": [tenure],
    "Balance": [balance],
    "NumOfProducts": [num_products],
    "HasCrCard": [has_cr_card],
    "IsActiveMember": [is_active],
    "EstimatedSalary": [salary]
})

# Merge encoded geography
input_data = pd.concat([input_data.reset_index(drop=True), geo_df], axis=1)

# Add missing columns if any
for col in scaler.feature_names_in_:
    if col not in input_data.columns:
        input_data[col] = 0

# Ensure correct column order
input_data = input_data[scaler.feature_names_in_]

# Scale
input_scaled = scaler.transform(input_data)

# Prediction
prediction = model.predict(input_scaled)
prob = prediction[0][0]

st.subheader("Prediction Result")

if prob > 0.5:
    st.error(f"Customer likely to churn (Probability: {prob:.2f})")
else:
    st.success(f"Customer likely to stay (Probability: {prob:.2f})")
