import tensorflow as tf
import pickle
import numpy as np

# Load model
model = tf.keras.models.load_model("model.h5")

with open("label_encoder_gender.pkl", "rb") as f:
    label_encoder_gender = pickle.load(f)

with open("onehot_encoder_geo.pkl", "rb") as f:
    onehot_encoder_geo = pickle.load(f)

with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

def preprocess(data):
    gender = label_encoder_gender.transform([data.Gender])[0]
    geo = onehot_encoder_geo.transform([[data.Geography]])

    input_data = np.array([[
        data.CreditScore,
        gender,
        data.Age,
        data.Tenure,
        data.Balance,
        data.NumOfProducts,
        data.HasCrCard,
        data.IsActiveMember,
        data.EstimatedSalary
    ]])

    input_data = np.concatenate((input_data, geo), axis=1)
    input_data = scaler.transform(input_data)

    return input_data
