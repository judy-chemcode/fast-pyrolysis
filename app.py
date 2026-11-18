#!/usr/bin/env python
# coding: utf-8

# In[1]:


app_code = """
import streamlit as st
import numpy as np
import pandas as pd
import joblib

st.set_page_config(page_title="Fast Pyrolysis Dashboard & LCA", layout="wide")

st.title("🔥 Fast Pyrolysis AI Dashboard & Thermodynamic LCA Simulator")
st.markdown("Predict Biochar & Bio-oil yields and calculate net carbon footprints interactively.")

st.sidebar.header("Feed Stock & Operating Parameters")
carbon = st.sidebar.slider("Carbon (wt. %)", 20.0, 70.0, 48.5)
hydrogen = st.sidebar.slider("Hydrogen (wt. %)", 2.0, 15.0, 6.5)
nitrogen = st.sidebar.slider("Nitrogen (wt. %)", 0.0, 10.0, 2.6)
oxygen = st.sidebar.slider("Oxygen (wt. %)", 20.0, 60.0, 42.0)
temperature = st.sidebar.slider("Pyrolysis Temperature (°C)", 300.0, 900.0, 520.0)
heating_rate = st.sidebar.slider("Heating Rate (°C/min)", 1.0, 100.0, 10.0)
moisture = st.sidebar.slider("Moisture (%)", 0.0, 40.0, 6.7)
vm = st.sidebar.slider("Volatile Matter (VM, %)", 10.0, 95.0, 72.7)
ash = st.sidebar.slider("Ash (%)", 0.1, 30.0, 6.7)
fc = st.sidebar.slider("Fixed Carbon (FC, %)", 0.1, 35.0, 14.0)
particle_size = st.sidebar.slider("Particle Size (mm)", 0.05, 10.0, 1.0)

# Feature Engineering
h_c_ratio = hydrogen / (carbon + 1e-5)
o_c_ratio = oxygen / (carbon + 1e-5)
fc_ratio = fc / (vm + 1e-5)

input_data = np.array([[
    carbon, hydrogen, nitrogen, oxygen, temperature, heating_rate, 
    moisture, vm, ash, fc, particle_size, h_c_ratio, o_c_ratio, fc_ratio
]])

@st.cache_resource
def load_model():
    return joblib.load("pyrolysis_ml_model.pkl")

try:
    model = load_model()
except:
    st.error("Model file 'pyrolysis_ml_model.pkl' not found! Please run the training cells first.")
    st.stop()

if st.button("Run Prediction & LCA Analysis"):
    preds = model.predict(input_data)[0]
    biochar = max(0.0, min(100.0, preds[0]))
    bio_oil = max(0.0, min(100.0, preds[1]))
    gas_loss = max(0.0, 100.0 - (biochar + bio_oil))

    col1, col2, col3 = st.columns(3)
    col1.metric("Solid Biochar Yield", f"{biochar:.2f} wt.%")
    col2.metric("Liquid Bio-oil Yield", f"{bio_oil:.2f} wt.%")
    col3.metric("Gas + Losses", f"{gas_loss:.2f} wt.%")

    # LCA Calculations
    cp, delta_h = 1.6, 2260.0
    q_thermal = 1.0 * cp * (temperature - 25.0) + 1.0 * (moisture / 100.0) * delta_h
    e_process = (q_thermal / 0.80 / 3600.0) * 0.45
    c_seq = 1.0 * (biochar / 100.0) * (12.0 / 44.0) * 0.75 * 0.80
    net_co2 = e_process - c_seq

    st.subheader("🌱 Environmental & Life Cycle Assessment (LCA)")
    st.metric(
        "Net Carbon Footprint", 
        f"{net_co2:.3f} kg CO₂-eq/kg feed", 
        delta="Net Carbon Sink" if net_co2 < 0 else "Net Emitter",
        delta_color="inverse"
    )
"""

with open('app.py', 'w', encoding='utf-8') as f:
  f.write(app_code)

print(
    'File app.py created successfully! Ready for local execution or cloud'
    ' deployment.'
)


# In[ ]:




