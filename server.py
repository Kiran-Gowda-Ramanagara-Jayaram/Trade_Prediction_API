from fastapi import FastAPI
from pydantic import BaseModel
import pickle
import numpy as np

# Load the trained model
with open("tradeprediction.pkl", "rb") as f:
    model = pickle.load(f)

# Initialize FastAPI app
app = FastAPI()

# Define input data structure for prediction requests
class PredictionRequest(BaseModel):
    date_block_num: int
    shop_id: int
    item_id: int
    month: int
    year: int
    rolling_mean_3: float
    rolling_std_3: float
    rolling_mean_6: float
    rolling_std_6: float
    item_cnt_month_lag_1: float

# Define prediction endpoint
import pandas as pd  # Import pandas

@app.post("/predict")
async def predict(data: PredictionRequest):
    # Convert input data to a DataFrame
    input_data = pd.DataFrame([[
        data.date_block_num, data.shop_id, data.item_id, data.month, data.year, 
        data.rolling_mean_3, data.rolling_std_3, data.rolling_mean_6, data.rolling_std_6, 
        data.item_cnt_month_lag_1
    ]], columns=[
        'date_block_num', 'shop_id', 'item_id', 'month', 'year', 
        'rolling_mean_3', 'rolling_std_3', 'rolling_mean_6', 'rolling_std_6', 
        'item_cnt_month_lag_1'
    ])  # Ensure these column names match those used in your model training

    # Make prediction
    prediction = model.predict(input_data)
    return {"item_cnt_month": prediction[0]}
