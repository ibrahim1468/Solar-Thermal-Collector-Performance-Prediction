import streamlit as st
import joblib
import json
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
import io

# ============================================================================
# PAGE CONFIGURATION
# ============================================================================

st.set_page_config(
    page_title="Solar Thermal Collector Performance",
    page_icon="☀️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================================
# LOAD MODEL AND METADATA
# ============================================================================

@st.cache_resource
def load_model_artifacts():
    """Load trained model, scaler, and metadata"""
    try:
        model = joblib.load('solar_thermal_deltaT_model_RandomForest_20260104_145610.joblib')
        scaler = joblib.load('solar_thermal_deltaT_scaler_20260104_145610.joblib')
        with open('solar_thermal_deltaT_metadata_20260104_145610.json', 'r') as f:
            metadata = json.load(f)
        return model, scaler, metadata
    except FileNotFoundError as e:
        st.error(f"Model files not found: {e}")
        st.stop()

model, scaler, metadata = load_model_artifacts()

# ============================================================================
# CONSTANTS
# ============================================================================

RADIATION_THRESHOLD = 50  # W/m² - minimum for active operation
RMSE = metadata['rmse']
MAE = metadata['mae']

# Approximate training ranges (adjust if exact ranges are available in metadata)
TRAINING_RANGES = {
    'Radiation': (0, 1100),
    'Ta': (5, 40),
    'Tin': (15, 65),
    'Tank_Mean': (15, 75),
    'cos_incidence': (0.0, 1.0),
    'Stratification': (0, 35),
    'Tank_loss_driver': (0, 55)
}

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def predict_delta_t(inputs):
    features = metadata['features']
    input_array = [inputs.get(feat, 0.0) for feat in features]
    input_df = pd.DataFrame([input_array], columns=features)
    input_scaled = scaler.transform(input_df)
    return model.predict(input_scaled)[0]

def get_operating_mode(radiation):
    return "active" if radiation >= RADIATION_THRESHOLD else "inactive"

def interpret_delta_t(delta_t):
    if delta_t > 2:
        return "🔥 Collector is effectively gaining heat from solar radiation"
    elif delta_t > 0:
        return "☀️ Collector is gaining heat (moderate efficiency)"
    elif delta_t > -1:
        return "⚖️ Thermal equilibrium (minimal net heat transfer)"
    else:
        return "❄️ Heat losses dominate over solar gain"

def calculate_cyclical_features(hour, day_of_year):
    hour_sin = np.sin(2 * np.pi * hour / 24)
    hour_cos = np.cos(2 * np.pi * hour / 24)
    doy_sin = np.sin(2 * np.pi * day_of_year / 365)
    doy_cos = np.cos(2 * np.pi * day_of_year / 365)
    return hour_sin, hour_cos, doy_sin, doy_cos

def check_out_of_range(inputs):
    warnings = []
    for key, (min_v, max_v) in TRAINING_RANGES.items():
        value = inputs.get(key, None)
        if value is not None:
            if value < min_v or value > max_v:
                warnings.append(key)
    return warnings

def generate_physics_story(radiation, tin, ta, delta_t):
    if radiation < RADIATION_THRESHOLD:
        return "Solar radiation is below the active threshold. The collector is inactive, with no net heat gain expected."
    
    irr_level = "high" if radiation > 700 else "moderate" if radiation > 300 else "low"
    temp_level = "high" if tin > 45 else "moderate" if tin > 25 else "low"
    
    story = f"At {radiation:.0f} W/m² solar irradiance and an inlet temperature of {tin:.1f}°C, "
    story += f"the collector is expected to raise the fluid temperature by approximately {delta_t:.1f}°C. "
    story += f"This performance is consistent with typical flat-plate collector behavior under {irr_level} irradiance "
    story += f"and {temp_level} operating temperatures. "
    if delta_t > 15:
        story += "Strong solar input overcomes thermal losses effectively."
    elif delta_t > 5:
        story += "Moderate net gain is achieved despite some heat losses."
    else:
        story += "Heat losses partially offset solar gains."
    return story

# ============================================================================
# PRESET SCENARIOS
# ============================================================================

def set_preset_morning():
    return {
        "radiation": 300.0, "ta": 18.0, "tin": 25.0, "tank_mean": 30.0,
        "hour": 9, "day_of_year": 172, "cos_incidence": 0.85,
        "stratification": 12.0, "tank_loss_driver": 18.0
    }

def set_preset_noon():
    return {
        "radiation": 850.0, "ta": 30.0, "tin": 40.0, "tank_mean": 45.0,
        "hour": 12, "day_of_year": 172, "cos_incidence": 1.0,
        "stratification": 8.0, "tank_loss_driver": 12.0
    }

def set_preset_afternoon():
    return {
        "radiation": 500.0, "ta": 28.0, "tin": 45.0, "tank_mean": 48.0,
        "hour": 16, "day_of_year": 172, "cos_incidence": 0.75,
        "stratification": 15.0, "tank_loss_driver": 20.0
    }

# ============================================================================
# MAIN APP
# ============================================================================

def main():
    # Title and header
    st.title("☀️ Solar Thermal Collector Performance Predictor")
    st.markdown("### Real-Time Temperature Rise Estimation")
    st.markdown("---")
    
    # Sidebar - About Section (UPDATED TEXT)
    with st.sidebar:
        st.header("ℹ️ About This Collector")
        st.markdown("""
Flat‑plate solar thermal collectors are engineered to absorb solar radiation efficiently, transfer that heat to a working fluid (often water), and minimize thermal losses. The absorber plate is the most crucial component so it must have high thermal conductivity and good corrosion resistance. The solar collector material was copper due to its superior heat transfer properties and durability. The absorber was coated in a spectrally‑selective surface to maximize solar absorptance and reduce radiative loss. Around the absorber, mineral wool was used to reduce heat loss from the back and sides, while plastic glazing covered the top to allow sunlight in and reduce convective heat loss. Collector frames were made of aluminum for structural support.
        """)
        
        st.markdown("---")
        st.markdown("**Model Details**")
        st.info(f"""
- **Type:** Random Forest Regressor  
- **Target:** ΔT = Tₒᵤₜ – Tᵢₙ (°C)  
- **Expected Accuracy:** ±{RMSE:.2f}°C (RMSE)  
- **Valid for:** Radiation ≥ 50 W/m² (active mode)
        """)
        
        st.markdown("---")
        st.markdown("**Preset Scenarios**")
        col_preset1, col_preset2 = st.columns(2)
        with col_preset1:
            if st.button("🌅 Morning"):
                for k, v in set_preset_morning().items():
                    st.session_state[k] = v
        with col_preset2:
            if st.button("☀️ Noon Peak"):
                for k, v in set_preset_noon().items():
                    st.session_state[k] = v
        if st.button("🌇 Late Afternoon"):
            for k, v in set_preset_afternoon().items():
                st.session_state[k] = v
        
        if st.button("🔄 Reset to Defaults"):
            default_vals = {
                "radiation": 500.0, "ta": 25.0, "tin": 30.0, "tank_mean": 35.0,
                "hour": 12, "day_of_year": 200, "cos_incidence": 0.9,
                "stratification": 10.0, "tank_loss_driver": 15.0
            }
            for k, v in default_vals.items():
                st.session_state[k] = v

    # Initialize session state for inputs
    if "radiation" not in st.session_state:
        st.session_state.update({
            "radiation": 500.0, "ta": 25.0, "tin": 30.0, "tank_mean": 35.0,
            "hour": 12, "day_of_year": 200, "cos_incidence": 0.9,
            "stratification": 10.0, "tank_loss_driver": 15.0
        })

    # Main content in two columns
    col1, col2 = st.columns([1, 1])
    
    # ========================================================================
    # LEFT COLUMN - INPUT PARAMETERS
    # ========================================================================
    
    with col1:
        st.header("🔧 System Parameters")
        
        st.subheader("☀️ Environmental Conditions")
        
        radiation = st.slider(
            "Solar Radiation (W/m²)",
            min_value=0.0, max_value=1200.0, step=10.0,
            value=st.session_state.radiation,
            help="Global horizontal irradiance measured on the collector plane. Typical peak values ~800–1000 W/m²."
        )
        st.session_state.radiation = radiation
        
        ta = st.slider(
            "Ambient Temperature (°C)",
            min_value=0.0, max_value=50.0, step=0.5,
            value=st.session_state.ta,
            help="Outdoor air temperature surrounding the collector."
        )
        st.session_state.ta = ta
        
        st.subheader("🌡️ System State")
        
        tin = st.slider(
            "Inlet Temperature (°C)",
            min_value=10.0, max_value=80.0, step=0.5,
            value=st.session_state.tin,
            help="Temperature of fluid entering the collector from storage."
        )
        st.session_state.tin = tin
        
        tank_mean = st.slider(
            "Tank Mean Temperature (°C)",
            min_value=10.0, max_value=80.0, step=0.5,
            value=st.session_state.tank_mean,
            help="Average temperature in the thermal storage tank."
        )
        st.session_state.tank_mean = tank_mean
        
        st.subheader("⏰ Time Parameters")
        
        col_time1, col_time2 = st.columns(2)
        with col_time1:
            hour = st.slider(
                "Hour of Day",
                min_value=0, max_value=23, value=st.session_state.hour,
                help="Local solar time (hour). Influences incidence angle and ambient conditions."
            )
            st.session_state.hour = hour
        with col_time2:
            day_of_year = st.slider(
                "Day of Year",
                min_value=1, max_value=365, value=st.session_state.day_of_year,
                help="Influences seasonal solar elevation and ambient temperature patterns."
            )
            st.session_state.day_of_year = day_of_year
        
        st.subheader("🔬 Advanced Parameters")
        
        cos_incidence = st.slider(
            "Cosine of Incidence Angle",
            min_value=0.0, max_value=1.0, step=0.05,
            value=st.session_state.cos_incidence,
            help="Cos(θ) where θ is angle between sun rays and collector normal. Reduces effective irradiance."
        )
        st.session_state.cos_incidence = cos_incidence
        
        stratification = st.slider(
            "Tank Stratification Index (°C)",
            min_value=0.0, max_value=40.0, step=1.0,
            value=st.session_state.stratification,
            help="Temperature difference between top and bottom of storage tank."
        )
        st.session_state.stratification = stratification
        
        tank_loss_driver = st.slider(
            "Tank Loss Driver (°C)",
            min_value=0.0, max_value=60.0, step=1.0,
            value=st.session_state.tank_loss_driver,
            help="Effective temperature difference driving tank heat losses."
        )
        st.session_state.tank_loss_driver = tank_loss_driver

    # ========================================================================
    # RIGHT COLUMN - RESULTS AND VISUALIZATION
    # ========================================================================
    
    with col2:
        st.header("📊 Prediction Results")
        
        operating_mode = get_operating_mode(radiation)
        
        # Prepare inputs for prediction
        hour_sin, hour_cos, doy_sin, doy_cos = calculate_cyclical_features(hour, day_of_year)
        
        inputs = {
            'Ta': ta,
            'Radiation': radiation,
            'cos_incidence': cos_incidence,
            'Tank_Mean': tank_mean,
            'Stratification': stratification,
            'Tank_loss_driver': tank_loss_driver,
            'Tin_lag1': tin,
            'Radiation_lag1': radiation,
            'hour_sin': hour_sin,
            'hour_cos': hour_cos,
            'doy_sin': doy_sin,
            'doy_cos': doy_cos,
        }
        
        # Out-of-range warning
        out_of_range = check_out_of_range(inputs)
        if out_of_range:
            st.warning(f"⚠️ Input outside experimental range ({', '.join(out_of_range)}) — predictions may be less reliable.")
        
        if operating_mode == "inactive":
            st.error("🌙 **SOLAR-INACTIVE MODE**")
            st.markdown("Solar radiation insufficient for active operation. Collector inactive. ΔT set to 0°C by physical rule.")
            delta_t_pred = 0.0
            tout_pred = tin
        else:
            st.success("☀️ **SOLAR-ACTIVE MODE**")
            st.markdown("Collector operational. Using ML model for prediction.")
            delta_t_pred = predict_delta_t(inputs)
            tout_pred = tin + delta_t_pred
        
        st.markdown("---")
        
        col_res1, col_res2 = st.columns(2)
        with col_res1:
            st.metric(label="**Predicted ΔT (°C)**", value=f"{delta_t_pred:.2f}")
        with col_res2:
            st.metric(label="**Outlet Temperature (°C)**", value=f"{tout_pred:.2f}", delta=f"{delta_t_pred:.2f}°C")
        
        st.markdown("---")
        st.subheader("🔍 Physical Interpretation")
        st.info(interpret_delta_t(delta_t_pred))
        
        st.markdown("---")
        st.subheader("📝 Physics-Based Assessment")
        story = generate_physics_story(radiation, tin, ta, delta_t_pred)
        st.info(story)
        
        st.markdown("---")
        st.subheader("📈 Model Confidence")
        confidence_pct = max(0, min(100, 100 - (RMSE / 10 * 100)))
        st.progress(confidence_pct / 100)
        st.caption(f"Expected accuracy: ±{RMSE:.2f}°C (RMSE) | ±{MAE:.2f}°C (MAE)")

    # ========================================================================
    # EXPORT RESULTS
    # ========================================================================
    
    st.markdown("---")
    st.subheader("💾 Export Results")
    
    results_df = pd.DataFrame([{
        "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "Solar Radiation (W/m²)": radiation,
        "Ambient Temp (°C)": ta,
        "Inlet Temp (°C)": tin,
        "Tank Mean (°C)": tank_mean,
        "Hour": hour,
        "Day of Year": day_of_year,
        "Cos Incidence": cos_incidence,
        "Stratification (°C)": stratification,
        "Tank Loss Driver (°C)": tank_loss_driver,
        "Predicted ΔT (°C)": round(delta_t_pred, 3),
        "Outlet Temp (°C)": round(tout_pred, 3),
        "Operating Mode": "Active" if operating_mode == "active" else "Inactive"
    }])
    
    csv = results_df.to_csv(index=False).encode()
    st.download_button(
        label="📄 Download Results as CSV",
        data=csv,
        file_name=f"solar_collector_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
        mime="text/csv"
    )

    # ========================================================================
    # SENSITIVITY ANALYSIS
    # ========================================================================
    
    st.markdown("---")
    st.header("🔬 Sensitivity Explorer")
    
    param_options = {
        "Solar Radiation (W/m²)": ("radiation", 0, 1200, radiation),
        "Inlet Temperature (°C)": ("tin", 10, 80, tin),
        "Ambient Temperature (°C)": ("ta", 0, 50, ta),
        "Tank Mean Temperature (°C)": ("tank_mean", 10, 80, tank_mean)
    }
    
    selected_param = st.selectbox("Select parameter to analyze:", options=list(param_options.keys()))
    param_name, min_val, max_val, current_val = param_options[selected_param]
    
    param_range = np.linspace(min_val, max_val, 60)
    delta_t_values = []
    
    for val in param_range:
        temp_radiation = val if param_name == "radiation" else radiation
        temp_tin = val if param_name == "tin" else tin
        temp_ta = val if param_name == "ta" else ta
        temp_tank_mean = val if param_name == "tank_mean" else tank_mean
        
        if temp_radiation < RADIATION_THRESHOLD:
            delta_t_values.append(0.0)
        else:
            h_sin, h_cos, d_sin, d_cos = calculate_cyclical_features(hour, day_of_year)
            temp_inputs = {
                'Ta': temp_ta, 'Radiation': temp_radiation, 'cos_incidence': cos_incidence,
                'Tank_Mean': temp_tank_mean, 'Stratification': stratification,
                'Tank_loss_driver': tank_loss_driver, 'Tin_lag1': temp_tin,
                'Radiation_lag1': temp_radiation, 'hour_sin': h_sin, 'hour_cos': h_cos,
                'doy_sin': d_sin, 'doy_cos': d_cos
            }
            delta_t_values.append(predict_delta_t(temp_inputs))
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=param_range, y=delta_t_values,
        mode='lines', line=dict(color='#FF6B35', width=4), name='ΔT (°C)'
    ))
    fig.add_trace(go.Scatter(
        x=[current_val], y=[delta_t_pred],
        mode='markers', marker=dict(size=14, color='#D41111', symbol='diamond'),
        name='Current Operating Point'
    ))
    
    if param_name == "radiation":
        fig.add_vline(x=RADIATION_THRESHOLD, line_dash="dash", line_color="orange",
                      annotation_text="Active threshold")
    
    fig.update_layout(
        title=f"Temperature Rise vs {selected_param}",
        xaxis_title=selected_param,
        yaxis_title="ΔT (°C)",
        template="plotly_white",
        height=450,
        hovermode='x unified'
    )
    st.plotly_chart(fig, use_container_width=True)

    # ========================================================================
    # NOMINAL OPERATING CURVES
    # ========================================================================
    
    st.markdown("---")
    st.header("📈 Nominal Performance Curves")
    st.markdown("Typical collector performance under standard reference conditions.")
    
    col_curve1, col_curve2 = st.columns(2)
    
    with col_curve1:
        st.subheader("ΔT vs Solar Radiation")
        rad_range = np.linspace(50, 1100, 50)
        curves = []
        tin_levels = [20, 35, 50]
        for tin_val in tin_levels:
            dt_vals = []
            for rad in rad_range:
                h_sin, h_cos, d_sin, d_cos = calculate_cyclical_features(12, 172)
                inp = {
                    'Ta': 25.0, 'Radiation': rad, 'cos_incidence': 1.0,
                    'Tank_Mean': tin_val + 10, 'Stratification': 10.0,
                    'Tank_loss_driver': 15.0, 'Tin_lag1': tin_val,
                    'Radiation_lag1': rad, 'hour_sin': h_sin, 'hour_cos': h_cos,
                    'doy_sin': d_sin, 'doy_cos': d_cos
                }
                dt_vals.append(predict_delta_t(inp))
            curves.append(go.Scatter(x=rad_range, y=dt_vals, name=f"Tᵢₙ = {tin_val}°C"))
        
        fig1 = go.Figure(curves)
        fig1.update_layout(title="ΔT vs Irradiance (Tₐ = 25°C, noon)",
                           xaxis_title="Solar Radiation (W/m²)", yaxis_title="ΔT (°C)",
                           template="plotly_white", height=400)
        st.plotly_chart(fig1, use_container_width=True)
    
    with col_curve2:
        st.subheader("ΔT vs Inlet Temperature")
        tin_range = np.linspace(15, 70, 50)
        curves2 = []
        rad_levels = [400, 700, 1000]
        for rad_val in rad_levels:
            dt_vals = []
            for tin_v in tin_range:
                h_sin, h_cos, d_sin, d_cos = calculate_cyclical_features(12, 172)
                inp = {
                    'Ta': 25.0, 'Radiation': rad_val, 'cos_incidence': 1.0,
                    'Tank_Mean': tin_v + 10, 'Stratification': 10.0,
                    'Tank_loss_driver': 15.0, 'Tin_lag1': tin_v,
                    'Radiation_lag1': rad_val, 'hour_sin': h_sin, 'hour_cos': h_cos,
                    'doy_sin': d_sin, 'doy_cos': d_cos
                }
                dt_vals.append(predict_delta_t(inp) if rad_val >= RADIATION_THRESHOLD else 0)
            curves2.append(go.Scatter(x=tin_range, y=dt_vals, name=f"G = {rad_val} W/m²"))
        
        fig2 = go.Figure(curves2)
        fig2.update_layout(title="ΔT vs Inlet Temp (Tₐ = 25°C, noon)",
                           xaxis_title="Inlet Temperature (°C)", yaxis_title="ΔT (°C)",
                           template="plotly_white", height=400)
        st.plotly_chart(fig2, use_container_width=True)

    # ========================================================================
    # FOOTER
    # ========================================================================
    
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: gray;'>
    <small>
    Solar Thermal Collector Performance Predictor | Physics-Informed ML Model | 
    Valid for Radiation ≥ 50 W/m² | Developed January 2026
    </small>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
