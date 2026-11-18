#!/usr/bin/env python
# coding: utf-8

# In[47]:


import re
import numpy as np
import pandas as pd

# 1. Load dataset from Excel file
df = pd.read_excel('Book1.xlsx')
print(f'Original dataset shape: {df.shape}')


# 2. Function to clean mixed/string numerical entries
def clean_numeric(val):
  if pd.isna(val):
    return np.nan
  if isinstance(val, (int, float)):
    return float(val)
  nums = re.findall(r'[-+]?\d*\.\d+|\d+', str(val))
  return float(nums[0]) if nums else np.nan


# Clean columns that might contain text or formatting noise
cleaning_cols = [
    'Oxygen',
    'Heating Rate (C/min)',
    'Biochar Yield (%)',
    'Bio Oil Yield (%)',
]
for col in cleaning_cols:
  if col in df.columns:
    df[col] = df[col].apply(clean_numeric)

# Drop missing values to ensure high data integrity
df = df.dropna()

# Enforce mass balance: Derived Gas + Losses = 100 - (Biochar + Bio-oil)
df['Gas_Losses (%)'] = 100.0 - (
    df['Biochar Yield (%)'] + df['Bio Oil Yield (%)']
)

print(f'Cleaned dataset records remaining: {len(df)} rows.')
display(df
)


# In[48]:


# 1. Feature Engineering: Atomic & Proximate Ratios
df['H_C_Ratio'] = df['Hydrogen'] / (df['Carbon'] + 1e-5)
df['O_C_Ratio'] = df['Oxygen'] / (df['Carbon'] + 1e-5)
df['Fixed_Carbon_Ratio'] = df['FC'] / (df['VM'] + 1e-5)

# 2. Statistical Summary of Main Outputs
yield_stats = df[['Biochar Yield (%)', 'Bio Oil Yield (%)']].agg(
    ['mean', 'median', 'std', 'min', 'max']
)
print('--- Statistical Summary of Pyrolysis Yields ---')
display(yield_stats)


# In[49]:


from sklearn.ensemble import ExtraTreesRegressor
from sklearn.metrics import mean_absolute_percentage_error, r2_score
from sklearn.model_selection import KFold

# Define Features and Targets
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

# K-Fold Cross Validation setup
kf = KFold(n_splits=5, shuffle=True, random_state=42)
model = ExtraTreesRegressor(n_estimators=300, random_state=42, n_jobs=-1)

r2_char, mape_char = [], []
r2_oil, mape_oil = [], []

for train_idx, val_idx in kf.split(X):
  X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
  y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

  model.fit(X_train, y_train)
  preds = model.predict(X_val)

  r2_char.append(r2_score(y_val['Biochar Yield (%)'], preds[:, 0]))
  mape_char.append(
      mean_absolute_percentage_error(y_val['Biochar Yield (%)'], preds[:, 0])
  )

  r2_oil.append(r2_score(y_val['Bio Oil Yield (%)'], preds[:, 1]))
  mape_oil.append(
      mean_absolute_percentage_error(y_val['Bio Oil Yield (%)'], preds[:, 1])
  )

print('--- Model Cross-Validation Performance ---')
print(
    f'Biochar (Solid) R2 Score: {np.mean(r2_char):.4f} | MAPE Error:'
    f' {np.mean(mape_char)*100:.2f}%'
)
print(
    f'Bio-oil (Liquid) R2 Score: {np.mean(r2_oil):.4f} | MAPE Error:'
    f' {np.mean(mape_oil)*100:.2f}%'
)


# In[50]:


def calculate_carbon_footprint(
    y_solid,
    ft,
    moisture,
    eta_thermal=0.80,
    ef_grid=0.45,
    f_c=0.75,
    r_100=0.80,
    m_feed=1.0,
):
  """Calculates thermodynamic heat duties and net CO2 emissions (kg CO2-eq/kg feed)."""
  cp = 1.6  # kJ/kg K (Specific heat of biomass)
  delta_h_vap = 2260.0  # kJ/kg (Latent heat of water vaporization)

  # 1. Thermal energy requirement (sensible heat + moisture vaporization)
  q_thermal = m_feed * cp * (ft - 25.0) + m_feed * (moisture / 100.0) * delta_h_vap
  q_electric = q_thermal / eta_thermal

  # 2. Process emissions from electricity/heat
  e_process = (q_electric / 3600.0) * ef_grid

  # 3. Carbon sequestration credit from recalcitrant biochar
  c_sequestered = m_feed * (y_solid / 100.0) * (12.0 / 44.0) * f_c * r_100

  # 4. Net Carbon Footprint
  net_co2 = e_process - c_sequestered
  return {
      'E_process': e_process,
      'C_sequestered': c_sequestered,
      'Net_CO2': net_co2,
  }


# Test LCA calculation with sample values
sample_lca = calculate_carbon_footprint(y_solid=38.0, ft=500.0, moisture=9.37)
print('--- Thermodynamic LCA Sample Results ---')
for k, v in sample_lca.items():
  print(f'{k}: {v:.4f}')


# In[51]:


import joblib

# Fit final model on 100% of the cleaned data
model.fit(X, y)

# Save model artifact to disk
joblib.dump(model, 'pyrolysis_ml_model.pkl')
print('Model successfully trained and saved as pyrolysis_ml_model.pkl !')


# In[53]:


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





# In[ ]:





# In[ ]:





# In[ ]:




