import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
import joblib

# Load model and scaler
model = load_model("model.h5")
scaler = joblib.load("scaler.pkl")

st.set_page_config(page_title="Next Day Closing Price", page_icon="📈")
st.title("📊 Predict Next Day's Closing Price (LSTM Model)")
st.markdown("Enter the last 10 closing prices:")

# Input: 10 closing prices
close_inputs = []
for i in range(10):
    val = st.number_input(f"Day {-10 + i + 1} Closing Price", value=100.0)
    close_inputs.append(val)

if st.button("📍 Predict Next Closing Price"):
    input_array = np.array(close_inputs).reshape(-1, 1)
    input_scaled = scaler.transform(input_array)
    input_scaled = input_scaled.reshape((1, 10, 1))

    prediction_scaled = model.predict(input_scaled)[0][0]
    prediction = scaler.inverse_transform([[prediction_scaled]])[0][0]

    st.success(f"✅ Predicted Next Closing Price: ₹{prediction:.2f}")
