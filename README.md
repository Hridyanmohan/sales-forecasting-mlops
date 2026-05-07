Retail Sales Forecasting System (MLOps Pipeline)


Objective: To forecast 8-week sales for 50 states using an automated "Champion-Challenger" framework.

Key Features:Feature Engineering: Implemented $t-1, t-7, t-30$ lags and rolling statistics to capture seasonality.Model Tournament: Automated comparison of SARIMA, Prophet, XGBoost, and LSTM.
Selection Metric: Models are selected based on the lowest RMSE via a time-series validation split.
Production API: Built with FastAPI for high-performance inference.
Interactive UI: Built with Streamlit for business stakeholder visualization.
## Documentation
You can find the full technical report here:[https://github.com/Hridyanmohan/sales-forecasting-mlops/blob/main/Time_Series_Forecasting_System_with_API_Project_Report.pdf)]
