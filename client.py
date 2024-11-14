import requests

# Define the FastAPI server URL
url = "http://127.0.0.1:8000/predict"

# Sample input data based on the model's expected features
data = {
    "date_block_num": 34,
    "shop_id": 5,
    "item_id": 5037,
    "month": 6,
    "year": 2015,
    "rolling_mean_3": 2.5,
    "rolling_std_3": 1.3,
    "rolling_mean_6": 2.8,
    "rolling_std_6": 1.5,
    "item_cnt_month_lag_1": 1.2
}

# Send a POST request to the FastAPI server
response = requests.post(url, json=data)

# Check if the request was successful and print the result
if response.status_code == 200:
    print("Prediction:", response.json())
else:
    print("Failed to get prediction:", response.status_code, response.text)
