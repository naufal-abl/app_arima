import streamlit as st
import pandas as pd
import numpy as np
import pickle
from statsmodels.tsa.arima.model import ARIMA

# Custom CSS for styling
st.markdown("""
    <style>
    /* Background */
    .stApp {
        background: linear-gradient(to right, #2c3e50, #4ca1af);
        color: white;
    }

    /* Judul */
    .stTitle {
        font-size: 36px;
        font-weight: bold;
        text-align: center;
        margin-bottom: 20px;
        padding: 10px;
        border-radius: 10px;
        background: rgba(255, 255, 255, 0.1);
        box-shadow: 2px 2px 10px rgba(0, 0, 0, 0.2);
    }

    /* Number Input */
    .stNumberInput label {
        font-size: 18px;
        font-weight: bold;
        color: white;
    }

    /* Tombol */
    .stButton button {
        background: linear-gradient(135deg, #1e88e5, #1565c0) !important;
        color: white !important;
        font-size: 18px !important;
        font-weight: bold !important;
        padding: 12px 20px !important;
        border-radius: 12px !important;
        border: none !important;
        box-shadow: 2px 2px 10px rgba(0, 0, 0, 0.3);
        transition: 0.3s;
    }
    
    .stButton button:hover {
        background: linear-gradient(135deg, #1565c0, #0d47a1) !important;
        transform: scale(1.05);
        box-shadow: 2px 2px 15px rgba(255, 255, 255, 0.5);
    }

    /* Card Style */
    .stCard {
        background: rgba(255, 255, 255, 0.2);
        border-radius: 15px;
        padding: 20px;
        box-shadow: 3px 3px 10px rgba(0, 0, 0, 0.3);
        margin-bottom: 20px;
    }

    /* Tabel */
    .stDataFrame {
        border-radius: 10px;
        overflow: hidden;
        background-color: rgba(255, 255, 255, 0.9);
        color: black;
        padding: 10px;
    }

    .stDataFrame tbody tr:hover {
        background-color: rgba(0, 0, 0, 0.1);
    }

    </style>
    """, unsafe_allow_html=True)

# Load the model and data
try:
    with open('arima_model.pkl', 'rb') as file:
        model = pickle.load(file)
except FileNotFoundError:
    st.error("Error: arima_model.pkl not found. Please upload the model file.")
    st.stop()

try:
    df = pd.read_csv('F.csv', index_col='Date', parse_dates=True)
    df.index = pd.to_datetime(df.index)  # Ensure index is datetime
except FileNotFoundError:
    st.error("Error: F.csv not found. Please upload the data file.")
    st.stop()


# Function to make predictions
def predict_price(days):
    try:
        last_date = df.index[-1]
        future_dates = pd.date_range(start=last_date + pd.DateOffset(1), periods=days)
        predictions = []

        # Gunakan nilai terakhir sebagai awal prediksi
        history = df["Close"].tolist()

        for _ in range(days):
            model_fit = ARIMA(history, order=model.order).fit()
            pred = model_fit.forecast(steps=1)[0]
            predictions.append(pred)
            history.append(pred)  # Update history untuk prediksi selanjutnya

        future_predictions = pd.DataFrame(predictions, index=future_dates, columns=["Close"])
        return future_predictions
    except Exception as e:
        return f"Prediction error: {e}"


# Streamlit app
st.markdown('<h1 class="stTitle">📈 Ford Motor Company Stock Price Prediction</h1>', unsafe_allow_html=True)

# Input number of days for prediction
st.markdown('<div class="stCard">', unsafe_allow_html=True)
days = st.number_input('🔮 Enter the number of days to predict:', min_value=1, max_value=30, value=7)
st.markdown('</div>', unsafe_allow_html=True)

if st.button('🚀 Predict'):
    future_predictions = predict_price(days)

    if isinstance(future_predictions, str):  # Handle prediction errors
        st.error(future_predictions)
    else:
        st.subheader('📊 Predicted Prices')
        st.markdown('<div class="stCard">', unsafe_allow_html=True)
        st.write(future_predictions.style.set_table_styles(
            [{'selector': 'thead th', 'props': [('background-color', '#1e88e5'), ('color', 'white')]}]
        ))
        st.markdown('</div>', unsafe_allow_html=True)

        # Tampilkan grafik
        st.markdown('<div class="stCard">', unsafe_allow_html=True)
        st.line_chart(future_predictions)
        st.markdown('</div>', unsafe_allow_html=True)
