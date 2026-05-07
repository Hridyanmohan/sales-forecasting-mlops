import pandas as pd
import numpy as np
import joblib
import os
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from statsmodels.tsa.statespace.sarimax import SARIMAX
from xgboost import XGBRegressor
from prophet import Prophet
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

#Create models directory
os.makedirs('models', exist_ok=True)

def train_tournament():
    df = pd.read_csv("data/processed_data.csv")
    features = ['lag_1', 'lag_7', 'lag_30', 'roll_mean_7', 'roll_std_7', 'month', 'day_of_week']
    states = df['State'].unique()

    for state in states:
        state_df = df[df['State'] == state].sort_values('Date')
        train = state_df.iloc[:-8] # Use all data except last 8 weeks
        val = state_df.iloc[-8:]    # Last 8 weeks for validation
        
        results = {}

        #SARIMA
        try:
            # Handles Seasonality and Trend
            model_sarima = SARIMAX(train['Total'], order=(1,1,1), seasonal_order=(1,1,1,12)).fit(disp=False)
            preds = model_sarima.forecast(8)
            results['SARIMA'] = (np.sqrt(mean_squared_error(val['Total'], preds)), model_sarima)
        except: pass

        #XGBoost
        try:
            model_xgb = XGBRegressor(n_estimators=100)
            model_xgb.fit(train[features], train['Total'])
            preds = model_xgb.predict(val[features])
            results['XGBoost'] = (np.sqrt(mean_squared_error(val['Total'], preds)), model_xgb)
        except: pass

        #Prophet
        try:
            p_df = train[['Date', 'Total']].rename(columns={'Date': 'ds', 'Total': 'y'})
            m = Prophet(weekly_seasonality=True, yearly_seasonality=True).fit(p_df)
            future = m.make_future_dataframe(periods=8)
            forecast = m.predict(future)
            preds = forecast['yhat'].iloc[-8:]
            results['Prophet'] = (np.sqrt(mean_squared_error(val['Total'], preds)), m)
        except: pass

        #LSTM
        try:
            
            scaler = MinMaxScaler()
            train_scaled = scaler.fit_transform(train[['Total']])
            
            
            X_lstm, y_lstm = [], []
            for i in range(4, len(train_scaled)):
                X_lstm.append(train_scaled[i-4:i, 0])
                y_lstm.append(train_scaled[i, 0])
            X_lstm, y_lstm = np.array(X_lstm), np.array(y_lstm)
            X_lstm = np.reshape(X_lstm, (X_lstm.shape[0], X_lstm.shape[1], 1))

            model_lstm = Sequential([
                LSTM(50, activation='relu', input_shape=(4, 1)),
                Dense(1)
            ])
            model_lstm.compile(optimizer='adam', loss='mse')
            model_lstm.fit(X_lstm, y_lstm, epochs=10, verbose=0)
            
            
            results['LSTM'] = (float('inf'), model_lstm) # Placeholder for RMSE
        except: pass

        #selection of model with lowest RMSE
        if results:
            
            leaderboard = sorted(results.items(), key=lambda x: x[1][0])
            
            print(f"\nTournament Results for {state}:")
            for m_name, (m_rmse, _) in leaderboard:
                print(f"  - {m_name}: RMSE = {m_rmse:.2f}")

            best_type = leaderboard[0][0] # The winner
            best_score, best_model = leaderboard[0][1]
            
            save_path = f"models/{state.replace(' ', '_')}_best.pkl"
            joblib.dump({"model": best_model, "type": best_type, "rmse": best_score}, save_path)

if __name__ == "__main__":
    train_tournament()