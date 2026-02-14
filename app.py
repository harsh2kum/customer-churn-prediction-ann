import streamlit as st
import pandas as pd
import numpy as np
import pickle
import tensorflow as tf
import plotly.graph_objects as go

# ---------- Load CSS ----------
def load_css():
    try:
        with open("assets/style.css") as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
    except:
        pass

load_css()

# ---------- Load Model & Artifacts (Cached) ----------
@st.cache_resource
def load_artifacts():
    model = tf.keras.models.load_model("model.h5")

    with open("onehot_encoder_geo.pkl", "rb") as f:
        onehot_encoder_geo = pickle.load(f)

    with open("label_encoder_gender.pkl", "rb") as f:
        label_encoder_gender = pickle.load(f)

    with open("scaler.pkl", "rb") as f:
        scaler = pickle.load(f)

    return model, onehot_encoder_geo, label_encoder_gender, scaler


model, onehot_encoder_geo, label_encoder_gender, scaler = load_artifacts()

# ---------- App Title ----------
st.title("💳 Customer Churn Prediction")

st.markdown("Predict whether a customer is likely to churn using a trained ANN model.")

# ---------- Layout ----------
col1, col2 = st.columns(2)

with col1:
    st.subheader("📋 Customer Details")

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

# ---------- Preprocessing ----------
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

# Ensure all columns match training data
for col in scaler.feature_names_in_:
    if col not in input_data.columns:
        input_data[col] = 0

input_data = input_data[scaler.feature_names_in_]

# Scale input
input_scaled = scaler.transform(input_data)

# ---------- Prediction ----------
if st.button("🔮 Predict Churn"):

    prediction = model.predict(input_scaled, verbose=0)
    prob = prediction[0][0]

    with col2:
        st.subheader("📊 Prediction Result")

        if prob > 0.5:
            st.error(f"⚠️ Customer likely to churn (Probability: {prob:.2f})")
        else:
            st.success(f"✅ Customer likely to stay (Probability: {prob:.2f})")

        # ---------- Gauge Chart ----------
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=prob * 100,
            title={'text': "Churn Probability"},
            gauge={
                'axis': {'range': [0, 100]},
                'bar': {'thickness': 0.3},
                'steps': [
                    {'range': [0, 50], 'color': "green"},
                    {'range': [50, 75], 'color': "orange"},
                    {'range': [75, 100], 'color': "red"}],
            }))
        st.plotly_chart(fig, use_container_width=True)

        # ---------- Feature Influence ----------
        importance = np.abs(input_data.iloc[0])
        importance_df = pd.DataFrame({
            "Feature": input_data.columns,
            "Impact": importance
        }).sort_values(by="Impact", ascending=False)

        st.subheader("📈 Feature Influence (Approx)")
        st.bar_chart(importance_df.set_index("Feature"))
