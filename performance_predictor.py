import streamlit as st
import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, time

# ----------------------------- Page Config -----------------------------
st.set_page_config(
    page_title="Solar Thermal Collector Performance Predictor",
    page_icon="☀️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ----------------------------- Load Model & Artifacts -----------------------------
@st.cache_resource
def load_artifacts():
    model = joblib.load("best_model_lightgbm_noleak.joblib")
    scaler = joblib.load("scaler_noleak.joblib")
    feature_config = joblib.load("feature_config_noleak.joblib")  # list of feature names
    return model, scaler, feature_config

model, scaler, feature_names = load_artifacts()

# Collector area (gross) - from the Graz dataset
COLLECTOR_AREA = 516  # m²

# ----------------------------- Realistic Limits (from data stats) -----------------------------
limits = {
    "mass_flow": (0.10, 2.60),       # kg/s (slightly buffered)
    "T_inlet": (273, 371),           # K
    "G_poa": (55, 1252),             # W/m²
    "G_poa_beam": (0, 967),          # W/m²
    "G_poa_diffuse": (0, 670),       # W/m²
    "T_ambient": (265, 309),         # K
    "wind_speed": (0.00, 3.50),      # m/s
    "solar_elevation": (11, 67),     # degrees
    "shadowed": (0, 1),              # fraction
}

# ----------------------------- Helper Functions -----------------------------
def celsius_to_kelvin(c):
    return c + 273.15

def kelvin_to_celsius(k):
    return k - 273.15

def compute_cyclic(time_obj, date_obj):
    hour = time_obj.hour + time_obj.minute / 60.0
    day_of_year = date_obj.timetuple().tm_yday
    hour_sin = np.sin(2 * np.pi * hour / 24)
    hour_cos = np.cos(2 * np.pi * hour / 24)
    day_sin = np.sin(2 * np.pi * day_of_year / 365.25)
    day_cos = np.cos(2 * np.pi * day_of_year / 365.25)
    return hour_sin, hour_cos, day_sin, day_cos

def is_daytime(elevation, gpoa):
    return (elevation > 10) and (gpoa > 50)

def predict_specific_power(input_df):
    scaled = scaler.transform(input_df[feature_names])
    pred = model.predict(scaled)
    return np.clip(pred, -60, 850)  # physical clipping (small negative allowed for losses)

# ----------------------------- Sidebar - User Inputs -----------------------------
st.sidebar.header("☀️ Input Parameters")

col1, col2 = st.sidebar.columns(2)

with col1:
    st.subheader("Date & Time")
    input_date = st.date_input("Date", value=datetime(2017, 7, 1))
    input_time = st.time_input("Time", value=time(12, 0))

with col2:
    st.subheader("Location-Derived")
    solar_elevation = st.slider("Solar Elevation (°)", 
                                min_value=11, max_value=67, value=40, step=1,
                                help="Angle of sun above horizon")
    shadowed = st.slider("Shadowed Fraction", 
                         min_value=0.0, max_value=1.0, value=0.0, step=0.05,
                         help="0 = no shadow, 1 = fully shadowed")

# Automatic cyclic features
hour_sin, hour_cos, day_sin, day_cos = compute_cyclic(input_time, input_date)

st.sidebar.subheader("Operating & Weather Conditions")

mass_flow = st.sidebar.slider("Mass Flow Rate (kg/s)", 
                              min_value=0.10, max_value=2.60, value=1.76, step=0.01)

T_inlet_c = st.sidebar.slider("Inlet Temperature (°C)", 
                              min_value=0, max_value=97, value=63, step=1)
T_inlet = celsius_to_kelvin(T_inlet_c)

G_poa = st.sidebar.slider("Global POA Irradiance (W/m²)", 
                          min_value=55, max_value=1252, value=613, step=10)

# Beam / Diffuse split (user can adjust ratio)
beam_ratio = st.sidebar.slider("Beam Fraction of G_poa", 
                               min_value=0.0, max_value=1.0, value=0.66, step=0.01,
                               help="0 = fully diffuse, 1 = fully direct")
G_poa_beam = G_poa * beam_ratio
G_poa_diffuse = G_poa * (1 - beam_ratio)

T_ambient_c = st.sidebar.slider("Ambient Temperature (°C)", 
                                min_value=-8, max_value=35, value=20, step=1)
T_ambient = celsius_to_kelvin(T_ambient_c)

wind_speed = st.sidebar.slider("Wind Speed (m/s)", 
                               min_value=0.0, max_value=3.5, value=0.77, step=0.05)

# ----------------------------- Main Dashboard -----------------------------
st.title("☀️ Solar Thermal Collector Performance Predictor")
st.markdown("""
**Large-Scale Flat-Plate Collector Array** (516 m² gross area, Graz 2017 dataset)  
Physics-based ML model (LightGBM) — predicts performance **without** using outlet temperature.
""")

# Prepare input dataframe
input_data = {
    "mass_flow": mass_flow,
    "T_inlet": T_inlet,
    "G_poa": G_poa,
    "G_poa_beam": G_poa_beam,
    "G_poa_diffuse": G_poa_diffuse,
    "T_ambient": T_ambient,
    "wind_speed": wind_speed,
    "solar_elevation": solar_elevation,
    "shadowed": shadowed,
    "hour_sin": hour_sin,
    "hour_cos": hour_cos,
    "day_sin": day_sin,
    "day_cos": day_cos,
    "is_day": 1.0
}

input_df = pd.DataFrame([input_data])

# Prediction
specific_power = predict_specific_power(input_df)[0]
total_power = specific_power * COLLECTOR_AREA / 1000  # kW
efficiency = specific_power / G_poa if G_poa > 50 else 0.0

# Day/night flag
day_active = is_daytime(solar_elevation, G_poa)

# ----------------------------- Metrics Display -----------------------------
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("Specific Thermal Power", f"{specific_power:.1f} W/m²",
              delta=None if day_active else "Low irradiance/flow")
with col2:
    st.metric("Total Thermal Power", f"{total_power:.1f} kW")
with col3:
    st.metric("Instantaneous Efficiency", f"{efficiency*100:.1f}%")
with col4:
    status = "Active" if day_active else "Inactive (night/low sun)"
    st.metric("System Status", status)

if not day_active:
    st.warning("⚠️ Low solar elevation or irradiance — collector likely inactive. "
               "Power output will be near zero or negative (losses).")

# ----------------------------- Interactive Plots -----------------------------
st.subheader("Predicted Performance Trends")

# 1. Diurnal trend (vary hour, fixed date)
hours = np.arange(0, 24, 0.5)
diurnal_power = []
for h in hours:
    h_sin = np.sin(2 * np.pi * h / 24)
    h_cos = np.cos(2 * np.pi * h / 24)
    temp_df = input_df.copy()
    temp_df["hour_sin"] = h_sin
    temp_df["hour_cos"] = h_cos
    # Approximate elevation change (simple sinusoidal model for demo)
    max_elev = 66  # summer max
    elev = max(0, max_elev * np.sin(np.pi * (h - 6) / 12)) if 6 <= h <= 18 else 0
    temp_df["solar_elevation"] = elev
    temp_df["is_day"] = 1 if elev > 10 else 0
    p = predict_specific_power(temp_df)[0]
    diurnal_power.append(p)

fig1 = go.Figure()
fig1.add_trace(go.Scatter(x=hours, y=diurnal_power, mode='lines', name='Specific Power',
                          line=dict(width=3)))
fig1.update_layout(title="Diurnal Power Profile (Fixed Date & Weather)",
                   xaxis_title="Hour of Day", yaxis_title="Specific Thermal Power (W/m²)",
                   template="plotly_white")
st.plotly_chart(fig1, use_container_width=True)

# 2. Efficiency vs Irradiance (vary G_poa)
gpoa_range = np.linspace(55, 1250, 100)
eff_vs_g = []
for g in gpoa_range:
    temp_df = input_df.copy()
    temp_df["G_poa"] = g
    temp_df["G_poa_beam"] = g * beam_ratio
    temp_df["G_poa_diffuse"] = g * (1 - beam_ratio)
    p = predict_specific_power(temp_df)[0]
    eff = p / g if g > 50 else 0
    eff_vs_g.append(eff)

fig2 = go.Figure()
fig2.add_trace(go.Scatter(x=gpoa_range, y=eff_vs_g, mode='lines', name='Efficiency',
                          line=dict(color='orange', width=3)))
fig2.update_layout(title="Collector Efficiency vs Irradiance",
                   xaxis_title="G_poa (W/m²)", yaxis_title="Efficiency η",
                   yaxis=dict(tickformat=".1%"), template="plotly_white")
st.plotly_chart(fig2, use_container_width=True)

# ----------------------------- Footer -----------------------------
st.markdown("---")
st.markdown("""
**Model Details**  
• Trained on 2017 high-precision data from a 516 m² flat-plate collector array in Graz, Austria  
• LightGBM regressor (no data leakage — outlet temperature excluded)  
• Test R²: 0.851 | MAE: 34.3 W/m²  
• All predictions are physics-constrained and validated against real operational limits.
""")
