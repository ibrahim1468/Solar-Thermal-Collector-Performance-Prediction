import streamlit as st
import joblib
import json
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime

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

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def predict_delta_t(inputs):
    """
    Predict temperature rise using loaded model
    
    Parameters:
    -----------
    inputs : dict
        Dictionary containing all required features
        
    Returns:
    --------
    float : Predicted ΔT (°C)
    """
    # Create feature array in correct order
    features = metadata['features']
    
    # Build input array
    input_array = []
    for feat in features:
        if feat in inputs:
            input_array.append(inputs[feat])
        else:
            st.error(f"Missing feature: {feat}")
            return 0.0
    
    # Reshape and scale
    input_df = pd.DataFrame([input_array], columns=features)
    input_scaled = scaler.transform(input_df)
    
    # Predict
    delta_t = model.predict(input_scaled)[0]
    
    return delta_t

def get_operating_mode(radiation):
    """Determine if collector is in active or inactive mode"""
    return "active" if radiation >= RADIATION_THRESHOLD else "inactive"

def interpret_delta_t(delta_t):
    """Provide physical interpretation of ΔT"""
    if delta_t > 2:
        return "🔥 Collector is effectively gaining heat from solar radiation"
    elif delta_t > 0:
        return "☀️ Collector is gaining heat (moderate efficiency)"
    elif delta_t > -1:
        return "⚖️ Thermal equilibrium (minimal net heat transfer)"
    else:
        return "❄️ Heat losses dominate over solar gain"

def calculate_cyclical_features(hour, day_of_year):
    """Convert hour and day to sine/cosine features"""
    hour_sin = np.sin(2 * np.pi * hour / 24)
    hour_cos = np.cos(2 * np.pi * hour / 24)
    doy_sin = np.sin(2 * np.pi * day_of_year / 365)
    doy_cos = np.cos(2 * np.pi * day_of_year / 365)
    return hour_sin, hour_cos, doy_sin, doy_cos

# ============================================================================
# MAIN APP
# ============================================================================

def main():
    # Title and header
    st.title("☀️ Solar Thermal Collector Performance Predictor")
    st.markdown("### Real-Time Temperature Rise Estimation")
    st.markdown("---")
    
    # Sidebar - About Section
    with st.sidebar:
        st.header("ℹ️ About This Tool")
        st.markdown("""
        **Purpose:**  
        Predict the net temperature rise (ΔT) across a solar thermal collector based on operating conditions.
        
        **How It Works:**
        - Data collected from experimental solar thermal system
        - Solar-active and inactive regimes separated
        - Machine learning predicts ΔT only during active operation  
        - Night behavior handled deterministically
        
        **Model Details:**
        - Type: Random Forest Regressor
        - Target: ΔT = Tout - Tin
        - Expected Accuracy: ±{:.2f}°C
        """.format(RMSE))
        
        st.markdown("---")
        st.markdown("**⚠️ Physical Constraints**")
        st.info("""
        Model valid only for:
        - Solar radiation ≥ 50 W/m²
        - Experimental operating range
        - Daytime active operation
        """)
    
    # Main content in two columns
    col1, col2 = st.columns([1, 1])
    
    # ========================================================================
    # LEFT COLUMN - INPUT PARAMETERS
    # ========================================================================
    
    with col1:
        st.header("🔧 System Parameters")
        
        # Environmental Conditions
        st.subheader("☀️ Environmental Conditions")
        
        radiation = st.slider(
            "Solar Radiation (W/m²)",
            min_value=0.0,
            max_value=1000.0,
            value=500.0,
            step=10.0,
            help="Global horizontal solar irradiance"
        )
        
        ta = st.slider(
            "Ambient Temperature (°C)",
            min_value=0.0,
            max_value=45.0,
            value=25.0,
            step=0.5,
            help="Outside air temperature"
        )
        
        # System State
        st.subheader("🌡️ System State")
        
        tin = st.slider(
            "Inlet Temperature (°C)",
            min_value=10.0,
            max_value=60.0,
            value=30.0,
            step=0.5,
            help="Fluid temperature entering the collector"
        )
        
        tank_mean = st.slider(
            "Tank Mean Temperature (°C)",
            min_value=10.0,
            max_value=70.0,
            value=35.0,
            step=0.5,
            help="Average storage tank temperature"
        )
        
        # Temporal Parameters
        st.subheader("⏰ Time Parameters")
        
        col_time1, col_time2 = st.columns(2)
        
        with col_time1:
            hour = st.slider(
                "Hour of Day",
                min_value=0,
                max_value=23,
                value=12,
                help="Hour of day (0-23)"
            )
        
        with col_time2:
            day_of_year = st.slider(
                "Day of Year",
                min_value=1,
                max_value=365,
                value=200,
                help="Day of year (1-365)"
            )
        
        # Additional parameters (set to reasonable defaults)
        st.subheader("🔬 Advanced Parameters")
        
        cos_incidence = st.slider(
            "Cosine of Incidence Angle",
            min_value=0.0,
            max_value=1.0,
            value=0.9,
            step=0.05,
            help="Accounts for angle-dependent absorption"
        )
        
        stratification = st.slider(
            "Tank Stratification Index",
            min_value=0.0,
            max_value=30.0,
            value=10.0,
            step=1.0,
            help="Temperature difference between tank layers"
        )
        
        tank_loss_driver = st.slider(
            "Tank Loss Driver",
            min_value=0.0,
            max_value=50.0,
            value=15.0,
            step=1.0,
            help="Thermal loss coefficient"
        )
    
    # ========================================================================
    # RIGHT COLUMN - RESULTS AND VISUALIZATION
    # ========================================================================
    
    with col2:
        st.header("📊 Prediction Results")
        
        # Operating mode detection
        operating_mode = get_operating_mode(radiation)
        
        if operating_mode == "inactive":
            # INACTIVE MODE - DETERMINISTIC BEHAVIOR
            st.error("🌙 **SOLAR-INACTIVE MODE**")
            st.markdown("""
            **Solar radiation insufficient for active operation.**  
            Collector inactive. ΔT set to 0°C by physical rule.
            
            ⚠️ Machine learning model is **bypassed** in this regime.
            """)
            
            delta_t_pred = 0.0
            tout_pred = tin
            
        else:
            # ACTIVE MODE - ML PREDICTION
            st.success("☀️ **SOLAR-ACTIVE MODE**")
            st.markdown("Collector operational. Using ML model for prediction.")
            
            # Calculate cyclical features
            hour_sin, hour_cos, doy_sin, doy_cos = calculate_cyclical_features(hour, day_of_year)
            
            # Prepare inputs (with lag features set to current values as approximation)
            inputs = {
                'Ta': ta,
                'Radiation': radiation,
                'cos_incidence': cos_incidence,
                'Tank_Mean': tank_mean,
                'Stratification': stratification,
                'Tank_loss_driver': tank_loss_driver,
                'Tin_lag1': tin,  # Approximate with current value
                'Radiation_lag1': radiation,  # Approximate with current value
                'hour_sin': hour_sin,
                'hour_cos': hour_cos,
                'doy_sin': doy_sin,
                'doy_cos': doy_cos,
            }
            
            # Predict
            delta_t_pred = predict_delta_t(inputs)
            tout_pred = tin + delta_t_pred
        
        # Display results
        st.markdown("---")
        
        # Main prediction
        col_res1, col_res2 = st.columns(2)
        
        with col_res1:
            st.metric(
                label="**Predicted ΔT**",
                value=f"{delta_t_pred:.2f}°C",
                help="Net temperature rise across collector"
            )
            st.caption("Net thermal gain across collector")
        
        with col_res2:
            st.metric(
                label="**Outlet Temperature**",
                value=f"{tout_pred:.2f}°C",
                delta=f"{delta_t_pred:.2f}°C from inlet",
                help="Tout = Tin + ΔT"
            )
            st.caption("Reconstructed from Tin + ΔT")
        
        # Physical interpretation
        st.markdown("---")
        st.subheader("🔍 Physical Interpretation")
        interpretation = interpret_delta_t(delta_t_pred)
        st.info(interpretation)
        
        # Confidence meter
        st.markdown("---")
        st.subheader("📈 Model Confidence")
        
        # Progress bar showing uncertainty
        confidence_pct = max(0, min(100, 100 - (RMSE / 10 * 100)))
        st.progress(confidence_pct / 100)
        st.caption(f"Expected accuracy: ±{RMSE:.2f}°C (RMSE) | ±{MAE:.2f}°C (MAE)")
    
    # ========================================================================
    # SENSITIVITY ANALYSIS SECTION
    # ========================================================================
    
    st.markdown("---")
    st.header("🔬 Sensitivity Explorer")
    st.markdown("Explore how ΔT varies with individual parameters (others held constant)")
    
    # Select parameter to vary
    param_options = {
        "Solar Radiation": ("radiation", 0, 1000, radiation, "W/m²"),
        "Inlet Temperature": ("tin", 10, 60, tin, "°C"),
        "Ambient Temperature": ("ta", 0, 45, ta, "°C"),
        "Tank Mean Temperature": ("tank_mean", 10, 70, tank_mean, "°C")
    }
    
    selected_param = st.selectbox(
        "Select parameter to analyze:",
        options=list(param_options.keys())
    )
    
    param_name, min_val, max_val, current_val, unit = param_options[selected_param]
    
    # Generate sensitivity data
    param_range = np.linspace(min_val, max_val, 50)
    delta_t_values = []
    
    for val in param_range:
        # Update the selected parameter
        temp_radiation = val if param_name == "radiation" else radiation
        temp_tin = val if param_name == "tin" else tin
        temp_ta = val if param_name == "ta" else ta
        temp_tank_mean = val if param_name == "tank_mean" else tank_mean
        
        # Check operating mode
        if temp_radiation < RADIATION_THRESHOLD:
            delta_t_values.append(0.0)
        else:
            hour_sin, hour_cos, doy_sin, doy_cos = calculate_cyclical_features(hour, day_of_year)
            
            inputs = {
                'Ta': temp_ta,
                'Radiation': temp_radiation,
                'cos_incidence': cos_incidence,
                'Tank_Mean': temp_tank_mean,
                'Stratification': stratification,
                'Tank_loss_driver': tank_loss_driver,
                'Tin_lag1': temp_tin,
                'Radiation_lag1': temp_radiation,
                'hour_sin': hour_sin,
                'hour_cos': hour_cos,
                'doy_sin': doy_sin,
                'doy_cos': doy_cos,
            }
            
            delta_t_values.append(predict_delta_t(inputs))
    
    # Plot
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=param_range,
        y=delta_t_values,
        mode='lines',
        name='ΔT',
        line=dict(color='#FF6B35', width=3)
    ))
    
    # Add current value marker
    current_delta_t = delta_t_pred
    fig.add_trace(go.Scatter(
        x=[current_val],
        y=[current_delta_t],
        mode='markers',
        name='Current Value',
        marker=dict(size=15, color='red', symbol='diamond')
    ))
    
    # Add threshold line for radiation
    if param_name == "radiation":
        fig.add_hline(
            y=0,
            line_dash="dash",
            line_color="gray",
            annotation_text="Inactive threshold",
            annotation_position="right"
        )
        fig.add_vline(
            x=RADIATION_THRESHOLD,
            line_dash="dash",
            line_color="orange",
            annotation_text=f"{RADIATION_THRESHOLD} W/m²",
            annotation_position="top"
        )
    
    fig.update_layout(
        title=f"ΔT vs {selected_param}",
        xaxis_title=f"{selected_param} ({unit})",
        yaxis_title="Temperature Rise ΔT (°C)",
        hovermode='x unified',
        height=400,
        template="plotly_white"
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Physical insights
    st.markdown("**📌 Physical Insights:**")
    if param_name == "radiation":
        st.markdown("""
        - ΔT increases with solar radiation (more energy input)
        - Below 50 W/m², collector is inactive (ΔT ≈ 0)
        - Relationship is typically non-linear due to thermal losses
        """)
    elif param_name == "tin":
        st.markdown("""
        - Higher inlet temperature typically reduces ΔT
        - Thermal losses increase with fluid temperature
        - Collector efficiency decreases at higher operating temperatures
        """)
    elif param_name == "ta":
        st.markdown("""
        - Higher ambient temperature reduces heat losses
        - ΔT typically increases with Ta (less convective loss)
        - Effect is moderate compared to radiation
        """)
    elif param_name == "tank_mean":
        st.markdown("""
        - Tank temperature affects system thermal state
        - Higher tank temperature may indicate recent solar gain
        - Influences thermal stratification and circulation
        """)
    
    # ========================================================================
    # FOOTER
    # ========================================================================
    
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: gray;'>
    <small>
    Solar Thermal Collector Performance Predictor | 
    Physics-Based ML Model | 
    Valid for Radiation ≥ 50 W/m²
    </small>
    </div>
    """, unsafe_allow_html=True)

# ============================================================================
# RUN APP
# ============================================================================

if __name__ == "__main__":
    main()