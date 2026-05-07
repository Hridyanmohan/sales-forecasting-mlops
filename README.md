Data Science (End-to-End Time Series Forecasting System with API) 
Objective:
Build a production-ready forecasting system that: 
1. Trains multiple forecasting algorithms 
2. Compares and selects the best model 
3. Exposes predictions via a REST API 
4. Should be designed like real backend service 

Problem Statement:
Forecast next 8weeks of sales for each state using historical data. Your solution must: 
• Handle missing dates / missing values (if any) 
• Handle seasonality & trend 
• Automatically select the best performing model 
• Serve predictions via API

Mandatory Models to Implement 
Train and compare at least: 
1. ARIMA / SARIMA 
2. Facebook Prophet 
3. XGBoost (with lag features) 
4. LSTM (deep learning)
   
Feature Engineering
You must create: 
• Lag features (t-1, t-7, t-30) 
• Rolling mean / std 
• Day of week, month, holiday flag 
• Train / validation split using time series logic (no leakage) 

Retail Sales Forecasting System (MLOps Pipeline)


Objective: To forecast 8-week sales for 50 states using an automated "Champion-Challenger" framework.

Key Features:Feature Engineering: Implemented $t-1, t-7, t-30$ lags and rolling statistics to capture seasonality.Model Tournament: Automated comparison of SARIMA, Prophet, XGBoost, and LSTM.
Selection Metric: Models are selected based on the lowest RMSE via a time-series validation split.
Production API: Built with FastAPI for high-performance inference.
Interactive UI: Built with Streamlit for business stakeholder visualization.


# 📈 End-to-End Sales Forecasting MLOps System

A production-grade forecasting solution designed to predict 8-week sales across 43 unique states. This project implements a **Champion-Challenger** architecture, transitioning from raw data to a live, decoupled web service.

## System Architecture
The system follows a service-oriented design to ensure modularity and scalability:
* **Data Engineering:** Automated pipeline handling lag features ($t-1, t-7, t-30$) and rolling statistics.
* **Model Tournament:** Competitive evaluation of **SARIMA, Facebook Prophet, XGBoost, and LSTM**.
* **Serving Layer:** **FastAPI** backend for high-performance RESTful inference.
* **Presentation Layer:** **Streamlit** dashboard for interactive business visualization.

## Tech Stack
* **Core:** Python 3.10+
* **ML:** Scikit-learn, XGBoost, Prophet, Statsmodels, TensorFlow
* **Web:** FastAPI, Uvicorn, Streamlit
* **Documentation:** LaTeX

##  Execution Order

### 1. Environment Setup
```bash
python -m venv venv
.\venv\Scripts\activate  # Windows
pip install -r requirements.txt

# Clean data and engineer features
python preprocessing.py

# Run tournament and save champion models
python trainer.py

# Terminal 1: Start Backend API
uvicorn main:app --reload

# Terminal 2: Start UI Dashboard
streamlit run app.py


## Documentation
You can find the full technical report here: https://github.com/Hridyanmohan/sales-forecasting-mlops/blob/main/Time_Series_Forecasting_System_with_API_Project_Report.pdf
