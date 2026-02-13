# 🚀 End-to-End Customer Churn Prediction using ANN

## 📌 Project Overview

This project builds an **Artificial Neural Network (ANN)** model to predict customer churn using the Churn Modelling dataset.  

The objective is to identify customers who are likely to leave the bank, enabling businesses to take proactive retention actions.

This project covers the complete machine learning lifecycle:

- Data preprocessing
- Feature encoding & scaling
- Model building with TensorFlow/Keras
- Hyperparameter tuning
- Model evaluation
- Deployment using Streamlit

---

## 📊 Business Problem

Customer churn is one of the biggest challenges for subscription-based businesses and banks.

By predicting churn in advance, companies can:

- Improve customer retention
- Reduce revenue loss
- Target high-risk customers with personalized strategies

---

## 🧠 Tech Stack

- Python 3.11
- TensorFlow / Keras
- NumPy
- Pandas
- Scikit-learn
- Streamlit
- Pickle
- Conda (Environment Management)

---

## 📂 Project Structure

```
.
├── Churn_Modelling.csv
├── experiments.ipynb
├── hyperparametertuningann.ipynb
├── prediction.ipynb
├── salaryregression.ipynb
│
├── model.h5
├── scaler.pkl
├── label_encoder_gender.pkl
├── onehot_encoder_geo.pkl
│
├── app.py
├── requirements.txt
└── README.md
```

---

## 🔎 Data Preprocessing

- Removed irrelevant columns
- Label Encoding (Gender)
- One-Hot Encoding (Geography)
- Feature Scaling using StandardScaler
- Train-Test Split

---

## 🧠 Model Architecture

- Input Layer
- Hidden Dense Layers with ReLU activation
- Output Layer with Sigmoid activation

Loss Function:
- Binary Crossentropy

Optimizer:
- Adam

Evaluation Metric:
- Accuracy

---

## 📈 Model Performance

- Binary Classification (Churn / No Churn)
- Outputs churn probability
- Evaluated on test dataset

(Future improvement: Add confusion matrix & ROC-AUC score)

---

## ⚙️ How to Run the Project

### 1️⃣ Clone the Repository

```bash
git clone https://github.com/your-username/end-to-end-customer-churn-ann.git
cd end-to-end-customer-churn-ann
```

---

### 2️⃣ Create Conda Environment

```bash
conda create -n churn_env python=3.11 -y
conda activate churn_env
```

---

### 3️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

---

### 4️⃣ Run Streamlit Application

```bash
streamlit run app.py
```

The app will open in your browser where you can input customer details and get churn prediction in real time.

---

## 🖥️ Deployment

This project includes a Streamlit app for interactive prediction.

Future deployment options:
- Render
- Hugging Face Spaces
- AWS EC2
- Docker containerization

---

## 🔥 Key Highlights

- End-to-End Deep Learning Pipeline
- ANN-based Classification
- Feature Engineering & Encoding
- Model Persistence using Pickle
- Real-Time Prediction via Streamlit
- Hyperparameter Tuning Notebook Included

---

## 🚀 Future Enhancements

- Add ROC Curve & Confusion Matrix
- Perform Cross-Validation
- Convert to FastAPI backend
- Add Docker support
- Implement MLOps pipeline

---

## 👨‍💻 Author

**Harsh Kumar**  
B.Tech – Computer Science & Data Science  
Aspiring Machine Learning Engineer  

---
