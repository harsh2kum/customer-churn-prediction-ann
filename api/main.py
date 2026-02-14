from fastapi import FastAPI, HTTPException
from api.schema import CustomerData
from api.model_loader import model, preprocess

app = FastAPI(title="Customer Churn Prediction API")

@app.post("/predict")
def predict(data: CustomerData):
    try:
        processed = preprocess(data)
        prediction = model.predict(processed)[0][0]

        return {
            "churn_probability": float(prediction),
            "prediction": "Churn" if prediction > 0.5 else "No Churn"
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))



# http://127.0.0.1:8000/docs#/default/predict_predict_post