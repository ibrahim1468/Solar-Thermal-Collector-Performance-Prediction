import streamlit as st
import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime, time

# ----------------------------- Page Config -----------------------------
st.set_page_config(
    page_title="Solar Thermal Collector Performance Predictor",
    page_icon="☀️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ----------------------------- Load Model & Artifacts (FIXED) -----------------------------
@st.cache_resource
def load_artifacts():
    try:
        model = joblib.load("best_model_lightgbm_noleak.joblib")
        scaler = joblib.load("scaler_noleak.joblib")
        config = joblib.load("feature_config_noleak.joblib")
        
        # --- CRITICAL FIX: Robustly extract feature names as list ---
        if isinstance(config, list):
            feature_names = config
        elif isinstance(config, dict):
            # Try common keys
            if 'feature_names' in config:
                feature_names = config['feature_names']
            elif 'features' in config:
                feature_names = config['features']
            else:
                feature_names = list(config.keys())
        elif hasattr(config, '__dict__'):
            feature_names = list(config.__dict__.keys())
        else:
            # Fallback: assume it's model-related and extract from model if possible
            if hasattr(model, 'feature_name_'):
                feature_names = model.feature_name_()
            elif hasattr(model, 'feature_names'):
                feature_names = model.feature_names
            else:
                raise ValueError("Could not extract feature names from config or model.")
        
        # Final cleanup: ensure list of strings
        feature_names = [str(name).strip() for name in feature_names]
        
        st.success(f"Model and artifacts loaded successfully! Using {len(feature_names)} features.")
        return model, scaler, feature_names
    
    except Exception as e:
        st.error(f"Failed to load model artifacts: {str(e)}")
        st.stop()

model, scaler, feature_names = load_artifacts()

# Hardcoded fallback (safety net) - replace with your actual 14 features if needed
EXPECTED_FEATURES = [
    "mass_flow", "T_inlet", "G_poa", "G_poa_beam", "G_poa_diffuse",
    "T_ambient", "wind_speed", "solar_elevation", "shadowed",
    "hour_sin", "hour_cos", "day_sin", "day_cos", "is_day"
]

if len(feature_names) != len(EXPECTED_FEATURES):
    st.warning(f"Feature count mismatch. Expected {len(EXPECTED_FEATURES)}, got {len(feature_names)}. Using loaded ones.")

# ----------------------------- Collector Area -----------------------------
COLLECTOR_AREA = 516  # m² (gross)

# ----------------------------- Helper Functions -----------------------------
def celsius_to_kelvin(c):
    return c + 273.15

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
    try:
        # Ensure exact column order and selection
        input_ordered = input_df[feature_names]
        scaled = scaler.transform(input_ordered)
        pred = model.predict(scaled)
        return np.clip(pred, -60, 850)
    except Exception as e:
        st.error(f"Prediction failed: {str(e)}")
        return np.array([0.0])

# ----------------------------- Sidebar Inputs -----------------------------
st.sidebar.header("☀️ Input Parameters")

col1, col2 = st.sidebar.columns(2)

with col1:
    st.subheader("Date & Time")
    input_date = st.date_input("Date", value=datetime(2017, 7, 1))
    input_time = st.time_input("Time", value=time(12, 0))

with col2:
    st.subheader("Solar Position")
    solar_elevation = st.slider("Solar Elevation (°)", 11, 67, 40, step=1)
    shadowed = st.slider("Shadowed Fraction", 0.0, 1.0, 0.0, step=0.05)

hour_sin, hour_cos, day_sin, day_cos = compute_cyclic(input_time, input_date)

st.sidebar.subheader("Operating & Weather Conditions")

mass_flow = st.sidebar.slider("Mass Flow Rate (kg/s)", 0.10, 2.60, 1.76, step=0.01)
T_inlet_c = st.sidebar.slider("Inlet Temperature (°C)", 0, 97, 63, step=1)
T_inlet = celsius_to_kelvin(T_inlet_c)

G_poa = st.sidebar.slider("Global POA Irradiance (W/m²)", 55, 1252, 613, step=10)
beam_ratio = st.sidebar.slider("Beam Fraction", 0.0, 1.0, 0.66, step=0.01)
G_poa_beam = G_poa * beam_ratio
G_poa_diffuse = G_poa * (1 - beam_ratio)

T_ambient_c = st.sidebar.slider("Ambient Temperature (°C)", -8, 35, 20, step=1)
T_ambient = celsius_to_kelvin(T_ambient_c)

wind_speed = st.sidebar.slider("Wind Speed (m/s)", 0.0, 3.5, 0.77, step=0.05)

# ----------------------------- Build Input DataFrame -----------------------------
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

# ----------------------------- Prediction -----------------------------
specific_power = predict_specific_power(input_df)[0]
total_power_kw = specific_power * COLLECTOR_AREA / 1000
efficiency = specific_power / G_poa if G_poa > 50 else 0.0
day_active = is_daytime(solar_elevation, G_poa)

# ----------------------------- Dashboard Display -----------------------------
st.title("☀️ Solar Thermal Collector Performance Predictor")
st.markdown("**Large-Scale Flat-Plate Collector Array** — 516 m² (Graz, Austria 2017) | Physics-based ML Prediction")

col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Specific Thermal Power", f"{specific_power:.1f} W/m²")
with col2:
    st.metric("Total Power Output", f"{total_power_kw:.1f} kW")
with col3:
    st.metric("Efficiency η", f"{efficiency:.1%}")
with col4:
    st.metric("System Status", "Active" if day_active else "Inactive")

if not day_active:
    st.info("🌙 Low sun or irradiance — minimal/no useful output expected.")

# ----------------------------- Plots -----------------------------
st.subheader("Performance Trends")

# Diurnal Profile
hours = np.arange(0, 24, 0.5)
diurnal_vals = []
for h in hours:
    h_sin = np.sin(2 * np.pi * h / 24)
    h_cos = np.cos(2 * np.pi * h / 24)
    elev = max(0, 66 * np.sin(np.pi * (h - 6)/12)) if 6 <= h <= 18 else 0
    df_temp = input_df.copy()
    df_temp["hour_sin"] = h_sin
    df_temp["hour_cos"] = h_cos
    df_temp["solar_elevation"] = elev
    df_temp["is_day"] = 1 if elev > 10 else 0
    p = predict_specific_power(df_temp)[0]
    diurnal_vals.append(p)

fig1 = go.Figure()
fig1.add_trace(go.Scatter(x=hours, y=diurnal_vals, mode='lines', name='Power', line=dict(width=3, color='orange')))
fig1.update_layout(title="Diurnal Power Profile", xaxis_title="Hour", yaxis_title="Specific Power (W/m²)", template="plotly_white")
st.plotly_chart(fig1, use_container_width=True)

# Efficiency Curve
gpoa_range = np.linspace(55, 1250, 80)
effs = []
for g in gpoa_range:
    df_temp = input_df.copy()
    df_temp["G_poa"] = g
    df_temp["G_poa_beam"] = g * beam_ratio
    df_temp["G_poa_diffuse"] = g * (1 - beam_ratio)
    p = predict_specific_power(df_temp)[0]
    effs.append(p / g if g > 50 else 0)

fig2 = go.Figure()
fig2.add_trace(go.Scatter(x=gpoa_range, y=effs, mode='lines', name='Efficiency', line=dict(width=3, color='green')))
fig2.update_layout(title="Efficiency vs Irradiance", xaxis_title="G_poa (W/m²)", yaxis_title="Efficiency", yaxis_tickformat=".1%", template="plotly_white")
st.plotly_chart(fig2, use_container_width=True)

# ----------------------------- Footer -----------------------------
st.markdown("---")
st.caption("""
**Model**: LightGBM (no data leakage) | Trained on high-precision 2017 operational data  
**Test Performance**: R² = 0.851 | MAE = 34.3 W/m²  
**Note**: Predictions are constrained to physically realistic ranges based on real system behavior.
""")
