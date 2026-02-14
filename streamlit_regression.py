import streamlit as st 
import numpy as np 
import pandas as pd 
import tensorflow as tf 
from sklearn.preprocessing import StandardScaler, LabelEncoder,OneHotEncoder
import pickle

# Load the trained model 
model = tf.keras.models.load_model('regression_model.h5')

# Load the encoders and scaler 

with open('label_encoder_gender.pkl','rb') as file:
    label_encoder_gender = pickle.load(file)
    
