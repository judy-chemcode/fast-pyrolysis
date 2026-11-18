
import joblib
import numpy as np
import pandas as pd
import re
from sklearn.ensemble import ExtraTreesRegressor
import streamlit as st

st.set_page_config(
    page_title='Fast Pyrolysis Dashboard & LCA', layout='wide'
)

st.title('🔥 Fast Pyrolysis AI Dashboard & Thermodynamic LCA Simulator')
st.markdown(
    'Predict Biochar & Bio-oil yields and calculate net carbon footprints'
    ' interactively.'
)


# دالة تدريب وتجهيز النموذج أوتوماتيكياً من ملف الإكسيل مباشرة
@st.cache_resource
def get_trained_model():
  # قراءة وتنظيف بيانات Book1.xlsx أوتوماتيك
  df = pd.read_excel('Book1.xlsx')

  def clean_numeric(val):
    if pd.isna(val):
      return np.nan
    if isinstance(val, (int, float)):
      return float(val)
    nums = re.findall(r'[-+]?\d*\.\d+|\d+', str(val))
    return float(nums[0]) if nums else np.nan

  for col in [
      'Oxygen',
      'Heating Rate (C/min)',
      'Biochar Yield (%)',
      'Bio Oil Yield (%)',
  ]:
    if col in df.columns:
      df[col] = df[col].apply(clean_numeric)
  df = df.dropna()

  # هندسة الخصائص (Feature Engineering)
  df['H_C_Ratio'] = df['Hydrogen'] / (df['Carbon'] + 1e-5)
  df['O_C_Ratio'] = df['Oxygen'] / (df['Carbon'] + 1e-5)
  df['Fixed_Carbon_Ratio'] = df['FC'] / (df['VM'] + 1e-5)

  X_cols = [
      'Carbon',
      'Hydrogen',
      'Nitrogen',
      'Oxygen',
      'Temperature (C )',
      'Heating Rate (C/min)',
      'Moisture (%)',
      'VM',
      'Ash',
      'FC',
      'Particle Size (mm)',
      'H_C_Ratio',
      'O_C_Ratio',
      'Fixed_Carbon_Ratio',
  ]
  X = df[X_cols]
  y = df[['Biochar Yield (%)', 'Bio Oil Yield (%)']]

  # تدريب النموذج بـ 50 شجرة عشان يكون سريع جداً وخفيف على السيرفر
  model = ExtraTreesRegressor(n_estimators=50, random_state=42, n_jobs=-1)
  model.fit(X, y)
  return model


# تحميل النموذج المبرمج ذاتياً
try:
  model = get_trained_model()
except Exception as e:
  st.error(
      f"Error loading data or training model: {e}. Please ensure 'Book1.xlsx'"
      ' is in the repository.'
  )
  st.stop()

# Sidebar inputs للتحكم في المعاملات
st.sidebar.header('Feed Stock & Operating Parameters')
carbon = st.sidebar.slider('Carbon (wt. %)', 20.0, 70.0, 48.5)
hydrogen = st.sidebar.slider('Hydrogen (wt. %)', 2.0, 15.0, 6.5)
nitrogen = st.sidebar.slider('Nitrogen (wt. %)', 0.0, 10.0, 2.6)
oxygen = st.sidebar.slider('Oxygen (wt. %)', 20.0, 60.0, 42.0)
temperature = st.sidebar.slider('Pyrolysis Temperature (°C)', 300.0, 900.0, 520.0)
heating_rate = st.sidebar.slider('Heating Rate (°C/min)', 1.0, 100.0, 10.0)
moisture = st.sidebar.slider('Moisture (%)', 0.0, 40.0, 6.7)
vm = st.sidebar.slider('Volatile Matter (VM, %)', 10.0, 95.0, 72.7)
ash = st.sidebar.slider('Ash (%)', 0.1, 30.0, 6.7)
fc = st.sidebar.slider('Fixed Carbon (FC, %)', 0.1, 35.0, 14.0)
particle_size = st.sidebar.slider('Particle Size (mm)', 0.05, 10.0, 1.0)

# حساب الـ Ratios المستخدمة في التدريب
h_c_ratio = hydrogen / (carbon + 1e-5)
o_c_ratio = oxygen / (carbon + 1e-5)
fc_ratio = fc / (vm + 1e-5)

input_data = np.array([[
    carbon,
    hydrogen,
    nitrogen,
    oxygen,
    temperature,
    heating_rate,
    moisture,
    vm,
    ash,
    fc,
    particle_size,
    h_c_ratio,
    o_c_ratio,
    fc_ratio,
]])

if st.button('Run Prediction & LCA Analysis'):
  preds = model.predict(input_data)[0]
  biochar = max(0.0, min(100.0, preds[0]))
  bio_oil = max(0.0, min(100.0, preds[1]))
  gas_loss = max(0.0, 100.0 - (biochar + bio_oil))

  col1, col2, col3 = st.columns(3)
  col1.metric('Solid Biochar Yield', f'{biochar:.2f} wt.%')
  col2.metric('Liquid Bio-oil Yield', f'{bio_oil:.2f} wt.%')
  col3.metric('Gas + Losses', f'{gas_loss:.2f} wt.%')

  # Thermodynamic LCA Calculations
  cp, delta_h = 1.6, 2260.0
  q_thermal = (
      1.0 * cp * (temperature - 25.0) + 1.0 * (moisture / 100.0) * delta_h
  )
  e_process = (q_thermal / 0.80 / 3600.0) * 0.45
  c_seq = 1.0 * (biochar / 100.0) * (12.0 / 44.0) * 0.75 * 0.80
  net_co2 = e_process - c_seq

  st.subheader('🌱 Environmental & Life Cycle Assessment (LCA)')
  st.metric(
      'Net Carbon Footprint',
      f'{net_co2:.3f} kg CO₂-eq/kg feed',
      delta='Net Carbon Sink' if net_co2 < 0 else 'Net Emitter',
      delta_color='inverse',
  )
