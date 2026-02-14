import streamlit as st
import numpy as np
import pandas as pd
import tensorflow as tf
import pickle

# ---------------------------
# Load model and preprocessors
# ---------------------------
model = tf.keras.models.load_model("regression_model.h5")

with open("label_encoder_gender.pkl", "rb") as file:
    label_encoder_gender = pickle.load(file)

with open("onehot_encoder_geo.pkl", "rb") as file:
    onehot_encoder_geo = pickle.load(file)

with open("scaler.pkl", "rb") as file:
    scaler = pickle.load(file)

# ---------------------------
# Streamlit UI
# ---------------------------
st.set_page_config(page_title="Salary Prediction", page_icon="💰")

st.title("💰 Estimated Salary Prediction (ANN)")
st.write("Enter customer details to predict estimated salary.")

# Inputs
geography = st.selectbox("Geography", onehot_encoder_geo.categories_[0])
gender = st.selectbox("Gender", label_encoder_gender.classes_)
age = st.slider("Age", 18, 92, 30)
balance = st.number_input("Balance", min_value=0.0)
credit_score = st.number_input("Credit Score", min_value=300, max_value=900, value=600)
tenure = st.slider("Tenure", 0, 10, 3)
num_of_products = st.slider("Number of Products", 1, 4, 1)
has_cr_card = st.selectbox("Has Credit Card", [0, 1])
is_active_member = st.selectbox("Is Active Member", [0, 1])
exited = st.selectbox("Exited", [0, 1])

# ---------------------------
# Prediction Button
# ---------------------------
if st.button("Predict Salary"):

    # Encode gender
    gender_encoded = label_encoder_gender.transform([gender])[0]

    # Create dataframe
    input_data = pd.DataFrame({
        "CreditScore": [credit_score],
        "Gender": [gender_encoded],
        "Age": [age],
        "Tenure": [tenure],
        "Balance": [balance],
        "NumOfProducts": [num_of_products],
        "HasCrCard": [has_cr_card],
        "IsActiveMember": [is_active_member],
        "Exited": [exited]
    })

    # One-hot encode geography
    geo_encoded = onehot_encoder_geo.transform([[geography]])
    geo_encoded_df = pd.DataFrame(
        geo_encoded,
        columns=onehot_encoder_geo.get_feature_names_out(["Geography"])
    )

    # Combine features
    input_data = pd.concat([input_data, geo_encoded_df], axis=1)

    # Scale
    input_data_scaled = scaler.transform(input_data)

    # Predict
    prediction = model.predict(input_data_scaled)
    predicted_salary = prediction[0][0]

    st.success(f"💰 Predicted Estimated Salary: ₹ {predicted_salary:,.2f}")
