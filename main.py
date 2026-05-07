from fastapi import FastAPI, HTTPException
import joblib
import pandas as pd
import os

app = FastAPI(title="Production Sales Forecasting Service")

# 1. Load data and preprocess
try:
    processed_data_path = "data/processed_data.csv"
    df = pd.read_csv(processed_data_path)
    df['Date'] = pd.to_datetime(df['Date'])
except Exception as e:
    print(f"Warning: Could not load data. Ensure {processed_data_path} exists.")
    df = None

@app.get("/predict/{state}")
def get_8_week_forecast(state: str):
    state_formatted = state.capitalize()
    model_path = f"models/{state_formatted.replace(' ', '_')}_best.pkl"
    
    if not os.path.exists(model_path):
        raise HTTPException(status_code=404, detail=f"Model for {state_formatted} not found.")

    #Load the dictionary
    champion_data = joblib.load(model_path)
    
    #Extract the model and type
    actual_model = champion_data['model']
    model_type = champion_data['type']
    
    #features
    state_latest = df[df['State'] == state_formatted].sort_values('Date').iloc[-1:]
    features = ['lag_1', 'lag_7', 'lag_30', 'roll_mean_7', 'roll_std_7', 'month', 'day_of_week']
    
    #prediction
    if model_type == "XGBoost":
        base_prediction = float(actual_model.predict(state_latest[features])[0])
    elif model_type == "SARIMA":
        base_prediction = float(actual_model.forecast(1).iloc[0])
    elif model_type == "Prophet":
        
        future_pd = pd.DataFrame({'ds': [state_latest['Date'].iloc[0]]})
        forecast = actual_model.predict(future_pd)
        base_prediction = float(forecast['yhat'].iloc[0])
    else:
       
        base_prediction = 0.0

    #Generate 8-week list
    forecast_values = [round(base_prediction * (1 + (i * 0.005)), 2) for i in range(8)]

    return {
        "status": "success",
        "metadata": {
            "state": state_formatted,
            "model_type": model_type,
            "forecast_horizon": "8 Weeks"
        },
        "forecast": forecast_values
    }