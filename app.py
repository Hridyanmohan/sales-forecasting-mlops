import streamlit as st
import requests
import pandas as pd
import plotly.express as px

# Page Configuration
st.set_page_config(page_title="Sales Forecasting Dashboard", layout="wide")

# Custom CSS 
st.markdown("""
    <style>
    .main { background-color: #f5f7f9; }
    .stMetric { background-color: #ffffff; padding: 15px; border-radius: 10px; box-shadow: 0 2px 4px rgba(0,0,0,0.05); }
    </style>
    """, unsafe_allow_html=True)

st.title("📈 State-wise Sales Forecasting Service")
st.markdown("---")

# Sidebar 
st.sidebar.header("User Input")
state_list = ["Alabama", "Arizona", "Arkansas", "California", "Colorado", "Connecticut", 
              "Florida", "Georgia", "Illinois", "Indiana", "Iowa", "Kansas", "Kentucky", 
              "Louisiana", "Maine", "Maryland", "Massachusetts", "Michigan", "Minnesota", 
              "Mississippi", "Missouri", "Nebraska", "Nevada", "New Hampshire", "New Jersey", 
              "New Mexico", "New York", "North Carolina", "Ohio", "Oklahoma", "Oregon", 
              "Pennsylvania", "Rhode Island", "South Carolina", "South Dakota", "Tennessee", 
              "Texas", "Utah", "Vermont", "Virginia", "Washington", "West Virginia", 
              "Wisconsin", "Wyoming"]

selected_state = st.sidebar.selectbox("Select State Target", state_list)

# API
if st.sidebar.button("Generate Forecast", type="primary"):
    try:
        # Call the FastAPI
        with st.spinner(f"Requesting Champion Model for {selected_state}..."):
            response = requests.get(f"http://127.0.0.1:8000/predict/{selected_state}")
        
        if response.status_code == 200:
            data = response.json()
            forecast_values = data['forecast']
            model_type = data['metadata']['model_type']
            
            
            
            st.info(f"🛡️ {selected_state} is currently using the **{model_type}** model.")

            
            col_m1, col_m2, col_m3 = st.columns(3)
            
            
            col_m1.metric(
                label="Starting Forecast (W1)", 
                value=f"₹{forecast_values[0]:,.0f}"
            )
            
            
            col_m2.metric(
                label="Ending Forecast (W8)", 
                value=f"₹{forecast_values[-1]:,.0f}",
                delta=f"{((forecast_values[-1]/forecast_values[0])-1)*100:.2f}% (Trend)"
            )
            
            
            col_m3.metric(label="Model Status", value="Active", delta="Optimal RMSE")

            st.markdown("---")

            # Data Visualization
            chart_data = pd.DataFrame({
                "Timeline": [f"Week {i+1}" for i in range(8)],
                "Predicted Sales": forecast_values
            })

            fig_col, table_col = st.columns([2, 1])

            with fig_col:
                fig = px.line(chart_data, x="Timeline", y="Predicted Sales", markers=True,
                              title=f"Visual Trend",
                              template="plotly_white")
                fig.update_traces(line_color='#1f77b4', line_width=3)
                st.plotly_chart(fig, use_container_width=True)

            with table_col:
                st.subheader(f"8-Week Sales Forecast for {selected_state}")
                st.dataframe(chart_data, hide_index=True, use_container_width=True)
                
        else:
            st.error(f"Backend Error: {response.status_code}. Model might not be trained for this state.")
            
    except Exception as e:
        st.error("Connection Failed: Ensure your FastAPI backend (Uvicorn) is running on port 8000.")
else:
    st.write("👈 Select a state and click 'Generate Forecast' to begin.")