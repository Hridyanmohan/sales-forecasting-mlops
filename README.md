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
```bash
Key Features & Logic
Automated Champion Selection: The system doesn't rely on a "one-size-fits-all" model. It automatically promotes the best-performing architecture per state based on the lowest RMSE.

Time-Series Integrity: Strict chronological splitting ensures zero data leakage during the validation phase.

System Resilience: Built-in logic identifies and bypasses numerical instabilities (such as Exploding Gradients in Deep Learning models) to maintain production uptime.

Decoupled Workflow: The Frontend and Backend communicate via a standardized API, allowing for independent scaling of either layer.

📄 Documentation
The full technical report, including mathematical challenges, feature engineering deep-dives, and system integration workflows, is available here:


### Download Technical Report (PDF)
You can find the full technical report here: https://github.com/Hridyanmohan/sales-forecasting-mlops/blob/main/Time_Series_Forecasting_System_with_API_Project_Report.pdf



Developed as part of a Data Science & MLOps Case Study.
