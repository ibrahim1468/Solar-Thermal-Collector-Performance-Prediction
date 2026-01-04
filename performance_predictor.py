import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import joblib
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# PAGE CONFIGURATION
# ============================================================================
st.set_page_config(
    page_title="Solar Thermal Collector Performance Dashboard",
    page_icon="☀️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for professional styling
st.markdown("""
    <style>
    .main {
        padding: 0rem 1rem;
    }
    .stAlert {
        padding: 0.5rem;
        margin: 0.5rem 0;
    }
    div[data-testid="metric-container"] {
        background-color: #f0f2f6;
        border: 1px solid #d0d0d0;
        padding: 10px;
        border-radius: 5px;
    }
    .css-1d391kg {
        padding-top: 2rem;
    }
    h1 {
        color: #1f77b4;
        padding-bottom: 1rem;
    }
    h2 {
        color: #2ca02c;
        padding-top: 1rem;
    }
    h3 {
        color: #ff7f0e;
    }
    </style>
    """, unsafe_allow_html=True)

# ============================================================================
# LOAD MODELS AND CONFIGURATIONS
# ============================================================================
@st.cache_resource
def load_models():
    """Load the trained model, scaler, and feature configuration"""
    try:
        model = joblib.load('best_model_lightgbm_noleak.joblib')
        scaler = joblib.load('scaler_noleak.joblib')
        feature_config = joblib.load('feature_config_noleak.joblib')
        return model, scaler, feature_config
    except Exception as e:
        st.error(f"Error loading models: {e}")
        st.stop()

model, scaler, feature_config = load_models()

# ============================================================================
# PHYSICAL CONSTRAINTS (from data statistics)
# ============================================================================
CONSTRAINTS = {
    'mass_flow': {'min': 0.108727, 'max': 2.553023, 'default': 1.713578, 'unit': 'kg/s'},
    'T_inlet': {'min': 273.575395, 'max': 370.207014, 'default': 338.320834, 'unit': 'K'},
    'G_poa': {'min': 55.275, 'max': 1251.375, 'default': 617.466667, 'unit': 'W/m²'},
    'G_poa_beam': {'min': 0.0, 'max': 966.322036, 'default': 430.652458, 'unit': 'W/m²'},
    'G_poa_diffuse': {'min': 0.0, 'max': 669.648715, 'default': 186.087444, 'unit': 'W/m²'},
    'T_ambient': {'min': 265.381667, 'max': 308.51875, 'default': 294.386375, 'unit': 'K'},
    'wind_speed': {'min': 0.018333, 'max': 3.49, 'default': 0.719167, 'unit': 'm/s'},
    'solar_elevation': {'min': 10.956602, 'max': 66.37248, 'default': 39.049221, 'unit': '°'},
    'shadowed': {'min': 0.0, 'max': 1.0, 'default': 0.0, 'unit': 'binary'},
}

# Physical constants
C_P_WATER = 4186  # J/(kg·K) - specific heat capacity of water

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================
def kelvin_to_celsius(k):
    """Convert Kelvin to Celsius"""
    return k - 273.15

def celsius_to_kelvin(c):
    """Convert Celsius to Kelvin"""
    return c + 273.15

def validate_inputs(inputs):
    """Validate physics-based constraints"""
    warnings = []
    
    # Check if beam + diffuse <= total irradiance
    if inputs['G_poa_beam'] + inputs['G_poa_diffuse'] > inputs['G_poa'] * 1.05:
        warnings.append("⚠️ Beam + Diffuse irradiance exceeds total irradiance")
    
    # Check if inlet temperature is reasonable vs ambient
    if inputs['T_inlet'] < inputs['T_ambient'] - 5:
        warnings.append("⚠️ Inlet temperature is lower than ambient (unusual)")
    
    # Low irradiance check
    if inputs['G_poa'] < 100:
        warnings.append("ℹ️ Low irradiance - system may not be producing significant power")
    
    # Night time check
    if inputs['solar_elevation'] < 15:
        warnings.append("🌙 Low solar elevation - likely dawn/dusk or nighttime")
    
    # Low flow check
    if inputs['mass_flow'] < 0.5:
        warnings.append("⚠️ Very low mass flow rate - may indicate startup or shutdown")
    
    return warnings

def calculate_derived_metrics(inputs, prediction):
    """Calculate physics-based derived metrics"""
    metrics = {}
    
    # Thermal efficiency (η = Power / (Area × G_poa))
    # Assuming 1 m² collector for specific power
    if inputs['G_poa'] > 10:
        metrics['efficiency'] = max(0, min(100, (prediction / inputs['G_poa']) * 100))
    else:
        metrics['efficiency'] = 0
    
    # Predicted temperature rise (ΔT = Power / (mass_flow × c_p))
    if inputs['mass_flow'] > 0.01:
        metrics['delta_T'] = prediction / (inputs['mass_flow'] * C_P_WATER)
        metrics['T_outlet_pred'] = inputs['T_inlet'] + metrics['delta_T']
    else:
        metrics['delta_T'] = 0
        metrics['T_outlet_pred'] = inputs['T_inlet']
    
    # Theoretical maximum power (if 100% efficient)
    metrics['max_theoretical_power'] = inputs['G_poa']
    
    # Performance ratio
    if metrics['max_theoretical_power'] > 0:
        metrics['performance_ratio'] = (prediction / metrics['max_theoretical_power']) * 100
    else:
        metrics['performance_ratio'] = 0
    
    # Heat loss estimation (simplified)
    delta_T_amb = inputs['T_inlet'] - inputs['T_ambient']
    metrics['estimated_heat_loss'] = delta_T_amb * inputs['wind_speed'] * 5  # Simplified
    
    # Day/Night classification
    metrics['is_operational'] = inputs['G_poa'] > 50 and inputs['mass_flow'] > 0.2
    
    return metrics

def create_hourly_profile(base_inputs, hour_range=24, day_of_year=None):
    """Generate predictions for a full day profile"""
    hours = np.arange(0, hour_range)
    predictions = []
    efficiencies = []
    irradiances = []
    solar_elevations = []
    
    # Use provided day_of_year or extract from base_inputs
    if day_of_year is None:
        # Back-calculate day from sin/cos if needed
        day_of_year = 180  # Default to summer solstice
    
    # Calculate day angle features (constant for the whole day)
    day_angle = 2 * np.pi * day_of_year / 365
    day_sin = np.sin(day_angle)
    day_cos = np.cos(day_angle)
    
    for hour in hours:
        # Calculate hour angle features from actual hour
        hour_angle = 2 * np.pi * hour / 24
        hour_sin = np.sin(hour_angle)
        hour_cos = np.cos(hour_angle)
        
        inputs_copy = base_inputs.copy()
        
        # Set temporal features
        inputs_copy['hour_sin'] = hour_sin
        inputs_copy['hour_cos'] = hour_cos
        inputs_copy['day_sin'] = day_sin
        inputs_copy['day_cos'] = day_cos
        
        # Adjust solar elevation (simplified solar position model)
        # Peak at solar noon (hour 12), accounting for day of year
        declination = 23.45 * np.sin(2 * np.pi * (284 + day_of_year) / 365)
        hour_angle_deg = 15 * (hour - 12)  # 15° per hour from solar noon
        
        # Simplified solar elevation calculation
        latitude = 35  # Assumed latitude (adjust as needed)
        elevation = np.arcsin(
            np.sin(np.radians(latitude)) * np.sin(np.radians(declination)) +
            np.cos(np.radians(latitude)) * np.cos(np.radians(declination)) * 
            np.cos(np.radians(hour_angle_deg))
        )
        elevation_deg = max(0, np.degrees(elevation))
        
        inputs_copy['solar_elevation'] = elevation_deg
        
        # Adjust irradiance based on solar elevation
        if elevation_deg > 5:  # Sun is above horizon
            # Air mass calculation
            air_mass = 1 / (np.sin(np.radians(elevation_deg)) + 0.50572 * (elevation_deg + 6.07995)**(-1.6364))
            air_mass = min(air_mass, 10)  # Cap air mass
            
            # Atmospheric attenuation
            irr_factor = 0.7**(air_mass**0.678)
            
            # Scale base irradiance
            inputs_copy['G_poa'] = base_inputs['G_poa'] * irr_factor
            inputs_copy['G_poa_beam'] = base_inputs['G_poa_beam'] * irr_factor
            inputs_copy['G_poa_diffuse'] = base_inputs['G_poa_diffuse'] * irr_factor * 0.8
            inputs_copy['is_day'] = 1.0
        else:
            # Nighttime
            inputs_copy['G_poa'] = 0
            inputs_copy['G_poa_beam'] = 0
            inputs_copy['G_poa_diffuse'] = 0
            inputs_copy['is_day'] = 0.0
        
        # Make prediction
        pred = make_prediction(inputs_copy)
        predictions.append(max(0, pred))
        irradiances.append(inputs_copy['G_poa'])
        solar_elevations.append(elevation_deg)
        
        # Calculate efficiency
        if inputs_copy['G_poa'] > 10:
            eff = (pred / inputs_copy['G_poa']) * 100
            efficiencies.append(max(0, min(100, eff)))
        else:
            efficiencies.append(0)
    
    return hours, predictions, efficiencies, irradiances, solar_elevations

def make_prediction(inputs):
    """Make prediction using the loaded model"""
    try:
        # Prepare feature vector in correct order
        features = feature_config['features']
        X = np.array([[inputs[f] for f in features]])
        
        # Scale and predict
        X_scaled = scaler.transform(X)
        prediction = model.predict(X_scaled)[0]
        
        # Physical constraints on prediction
        prediction = max(-50, min(850, prediction))  # Based on data constraints
        
        return prediction
    except Exception as e:
        st.error(f"Prediction error: {e}")
        return 0

# ============================================================================
# MAIN APPLICATION
# ============================================================================
def main():
    # Header
    st.title("☀️ Solar Thermal Collector Performance Dashboard")
    st.markdown("### Physics-Based Prediction System | No Data Leakage")
    
    # Sidebar for inputs
    st.sidebar.header("⚙️ System Parameters")
    
    # Mode selection
    mode = st.sidebar.radio(
        "Operation Mode",
        ["Single Point Prediction", "Daily Profile Analysis", "Sensitivity Analysis"],
        help="Choose the analysis mode"
    )
    
    st.sidebar.markdown("---")
    
    # ========================================================================
    # INPUT SECTION
    # ========================================================================
    st.sidebar.subheader("📊 Operating Conditions")
    
    # Temperature inputs (show in Celsius, convert to Kelvin)
    col1, col2 = st.sidebar.columns(2)
    with col1:
        T_inlet_C = st.number_input(
            "Inlet Temp (°C)",
            min_value=kelvin_to_celsius(CONSTRAINTS['T_inlet']['min']),
            max_value=kelvin_to_celsius(CONSTRAINTS['T_inlet']['max']),
            value=kelvin_to_celsius(CONSTRAINTS['T_inlet']['default']),
            step=1.0,
            help="Fluid temperature entering the collector"
        )
    
    with col2:
        T_ambient_C = st.number_input(
            "Ambient Temp (°C)",
            min_value=kelvin_to_celsius(CONSTRAINTS['T_ambient']['min']),
            max_value=kelvin_to_celsius(CONSTRAINTS['T_ambient']['max']),
            value=kelvin_to_celsius(CONSTRAINTS['T_ambient']['default']),
            step=1.0,
            help="Outside air temperature"
        )
    
    mass_flow = st.sidebar.slider(
        "Mass Flow Rate (kg/s)",
        min_value=float(CONSTRAINTS['mass_flow']['min']),
        max_value=float(CONSTRAINTS['mass_flow']['max']),
        value=float(CONSTRAINTS['mass_flow']['default']),
        step=0.1,
        help="Flow rate of heat transfer fluid"
    )
    
    st.sidebar.markdown("---")
    st.sidebar.subheader("☀️ Solar Conditions")
    
    G_poa = st.sidebar.slider(
        "Total Irradiance (W/m²)",
        min_value=float(CONSTRAINTS['G_poa']['min']),
        max_value=float(CONSTRAINTS['G_poa']['max']),
        value=float(CONSTRAINTS['G_poa']['default']),
        step=10.0,
        help="Total solar irradiance on collector plane"
    )
    
    col1, col2 = st.sidebar.columns(2)
    with col1:
        G_poa_beam = st.number_input(
            "Beam (W/m²)",
            min_value=float(CONSTRAINTS['G_poa_beam']['min']),
            max_value=min(float(CONSTRAINTS['G_poa_beam']['max']), G_poa),
            value=min(float(CONSTRAINTS['G_poa_beam']['default']), G_poa * 0.7),
            step=10.0,
            help="Direct beam irradiance"
        )
    
    with col2:
        G_poa_diffuse = st.number_input(
            "Diffuse (W/m²)",
            min_value=float(CONSTRAINTS['G_poa_diffuse']['min']),
            max_value=min(float(CONSTRAINTS['G_poa_diffuse']['max']), G_poa),
            value=min(float(CONSTRAINTS['G_poa_diffuse']['default']), G_poa * 0.3),
            step=10.0,
            help="Diffuse sky irradiance"
        )
    
    solar_elevation = st.sidebar.slider(
        "Solar Elevation (°)",
        min_value=float(CONSTRAINTS['solar_elevation']['min']),
        max_value=float(CONSTRAINTS['solar_elevation']['max']),
        value=float(CONSTRAINTS['solar_elevation']['default']),
        step=1.0,
        help="Sun angle above horizon"
    )
    
    st.sidebar.markdown("---")
    st.sidebar.subheader("🌬️ Environmental Conditions")
    
    wind_speed = st.sidebar.slider(
        "Wind Speed (m/s)",
        min_value=float(CONSTRAINTS['wind_speed']['min']),
        max_value=float(CONSTRAINTS['wind_speed']['max']),
        value=float(CONSTRAINTS['wind_speed']['default']),
        step=0.1,
        help="Wind speed affects convective heat loss"
    )
    
    shadowed = st.sidebar.checkbox(
        "Collector Shadowed",
        value=False,
        help="Check if collector is partially or fully shadowed"
    )
    
    # Advanced options
    with st.sidebar.expander("🔧 Advanced Options"):
        hour_of_day = st.slider("Hour of Day", 0, 23, 12, help="For temporal features")
        day_of_year = st.slider("Day of Year", 1, 365, 180, help="For seasonal features")
    
    # Calculate temporal features
    hour_angle = 2 * np.pi * hour_of_day / 24
    day_angle = 2 * np.pi * day_of_year / 365
    
    # Prepare inputs dictionary
    inputs = {
        'mass_flow': mass_flow,
        'T_inlet': celsius_to_kelvin(T_inlet_C),
        'G_poa': G_poa,
        'G_poa_beam': G_poa_beam,
        'G_poa_diffuse': G_poa_diffuse,
        'T_ambient': celsius_to_kelvin(T_ambient_C),
        'wind_speed': wind_speed,
        'solar_elevation': solar_elevation,
        'shadowed': 1.0 if shadowed else 0.0,
        'hour_sin': np.sin(hour_angle),
        'hour_cos': np.cos(hour_angle),
        'day_sin': np.sin(day_angle),
        'day_cos': np.cos(day_angle),
        'is_day': 1.0 if solar_elevation > 15 else 0.0
    }
    
    # ========================================================================
    # VALIDATION
    # ========================================================================
    validation_warnings = validate_inputs(inputs)
    if validation_warnings:
        with st.expander("⚠️ Input Validation Warnings", expanded=True):
            for warning in validation_warnings:
                st.warning(warning)
    
    # ========================================================================
    # PREDICTION
    # ========================================================================
    if mode == "Single Point Prediction":
        st.header("📈 Single Point Prediction")
        
        # Make prediction
        prediction = make_prediction(inputs)
        derived_metrics = calculate_derived_metrics(inputs, prediction)
        
        # Display main results
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                "Predicted Power",
                f"{prediction:.2f} W/m²",
                help="Specific thermal power output"
            )
        
        with col2:
            st.metric(
                "Thermal Efficiency",
                f"{derived_metrics['efficiency']:.1f}%",
                help="η = Power / Irradiance"
            )
        
        with col3:
            st.metric(
                "Temperature Rise",
                f"{derived_metrics['delta_T']:.2f} K",
                help="ΔT = Power / (ṁ × cp)"
            )
        
        with col4:
            status = "🟢 Operational" if derived_metrics['is_operational'] else "🔴 Inactive"
            st.metric(
                "System Status",
                status,
                help="Based on irradiance and flow rate"
            )
        
        # Additional metrics
        st.markdown("---")
        st.subheader("🔬 Detailed Physics Analysis")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Predicted Outlet Temp", 
                     f"{kelvin_to_celsius(derived_metrics['T_outlet_pred']):.1f} °C")
            st.metric("Inlet Temperature", 
                     f"{T_inlet_C:.1f} °C")
            st.metric("Ambient Temperature", 
                     f"{T_ambient_C:.1f} °C")
        
        with col2:
            st.metric("Total Irradiance", f"{G_poa:.1f} W/m²")
            st.metric("Beam Component", f"{G_poa_beam:.1f} W/m²")
            st.metric("Diffuse Component", f"{G_poa_diffuse:.1f} W/m²")
        
        with col3:
            st.metric("Performance Ratio", 
                     f"{derived_metrics['performance_ratio']:.1f}%")
            st.metric("Max Theoretical Power", 
                     f"{derived_metrics['max_theoretical_power']:.1f} W/m²")
            st.metric("Est. Heat Loss", 
                     f"{derived_metrics['estimated_heat_loss']:.1f} W/m²")
        
        # Visualization
        st.markdown("---")
        st.subheader("📊 Energy Balance Breakdown")
        
        # Create energy balance chart
        fig = go.Figure()
        
        categories = ['Solar Input', 'Useful Heat', 'Losses', 'Efficiency']
        values = [
            G_poa,
            prediction,
            G_poa - prediction if G_poa > prediction else 0,
            derived_metrics['efficiency']
        ]
        colors = ['#FDB462', '#80B1D3', '#FB8072', '#B3DE69']
        
        fig.add_trace(go.Bar(
            x=categories[:3],
            y=values[:3],
            marker_color=colors[:3],
            text=[f"{v:.1f}" for v in values[:3]],
            textposition='auto',
        ))
        
        fig.update_layout(
            title="Energy Flow Analysis",
            yaxis_title="Power (W/m²)",
            showlegend=False,
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Feature importance visualization
        col1, col2 = st.columns(2)
        
        with col1:
            # Irradiance components pie chart
            fig_pie = go.Figure(data=[go.Pie(
                labels=['Beam', 'Diffuse'],
                values=[G_poa_beam, G_poa_diffuse],
                hole=0.3,
                marker_colors=['#FFD700', '#87CEEB']
            )])
            fig_pie.update_layout(title="Irradiance Composition", height=300)
            st.plotly_chart(fig_pie, use_container_width=True)
        
        with col2:
            # Temperature profile
            temps = {
                'Ambient': T_ambient_C,
                'Inlet': T_inlet_C,
                'Predicted Outlet': kelvin_to_celsius(derived_metrics['T_outlet_pred'])
            }
            fig_temp = go.Figure(data=[go.Bar(
                x=list(temps.keys()),
                y=list(temps.values()),
                marker_color=['#1f77b4', '#ff7f0e', '#2ca02c']
            )])
            fig_temp.update_layout(
                title="Temperature Profile",
                yaxis_title="Temperature (°C)",
                height=300
            )
            st.plotly_chart(fig_temp, use_container_width=True)
    
    # ========================================================================
    # DAILY PROFILE MODE
    # ========================================================================
    elif mode == "Daily Profile Analysis":
        st.header("🌅 Daily Performance Profile")
        
        st.info("Simulating collector performance over 24 hours with current conditions")
        
        # Generate daily profile
        hours, predictions, efficiencies = create_hourly_profile(inputs)
        
        # Create subplots
        fig = make_subplots(
            rows=2, cols=1,
            subplot_titles=('Thermal Power Output', 'Thermal Efficiency'),
            vertical_spacing=0.15
        )
        
        # Power plot
        fig.add_trace(
            go.Scatter(x=hours, y=predictions, mode='lines+markers',
                      name='Power', line=dict(color='#1f77b4', width=3),
                      fill='tozeroy'),
            row=1, col=1
        )
        
        # Efficiency plot
        fig.add_trace(
            go.Scatter(x=hours, y=efficiencies, mode='lines+markers',
                      name='Efficiency', line=dict(color='#2ca02c', width=3),
                      fill='tozeroy'),
            row=2, col=1
        )
        
        fig.update_xaxes(title_text="Hour of Day", row=2, col=1)
        fig.update_yaxes(title_text="Power (W/m²)", row=1, col=1)
        fig.update_yaxes(title_text="Efficiency (%)", row=2, col=1)
        
        fig.update_layout(height=700, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
        
        # Daily statistics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Peak Power", f"{max(predictions):.1f} W/m²")
        with col2:
            st.metric("Average Power", f"{np.mean(predictions):.1f} W/m²")
        with col3:
            st.metric("Daily Energy", f"{np.trapz(predictions) / 1000:.2f} kWh/m²")
        with col4:
            st.metric("Avg Efficiency", f"{np.mean(efficiencies):.1f}%")
        
        # Download option
        df_profile = pd.DataFrame({
            'Hour': hours,
            'Power (W/m²)': predictions,
            'Efficiency (%)': efficiencies
        })
        
        csv = df_profile.to_csv(index=False)
        st.download_button(
            label="📥 Download Daily Profile",
            data=csv,
            file_name="daily_profile.csv",
            mime="text/csv"
        )
    
    # ========================================================================
    # SENSITIVITY ANALYSIS MODE
    # ========================================================================
    elif mode == "Sensitivity Analysis":
        st.header("🔍 Sensitivity Analysis")
        
        param_choice = st.selectbox(
            "Select Parameter to Vary",
            ["G_poa", "mass_flow", "T_inlet", "wind_speed", "solar_elevation"]
        )
        
        # Get parameter range
        param_min = CONSTRAINTS[param_choice]['min']
        param_max = CONSTRAINTS[param_choice]['max']
        param_values = np.linspace(param_min, param_max, 50)
        
        predictions = []
        efficiencies = []
        
        # Calculate predictions for each value
        progress_bar = st.progress(0)
        for i, val in enumerate(param_values):
            inputs_copy = inputs.copy()
            inputs_copy[param_choice] = val
            
            pred = make_prediction(inputs_copy)
            predictions.append(max(0, pred))
            
            if inputs_copy['G_poa'] > 10:
                eff = (pred / inputs_copy['G_poa']) * 100
                efficiencies.append(max(0, min(100, eff)))
            else:
                efficiencies.append(0)
            
            progress_bar.progress((i + 1) / len(param_values))
        
        progress_bar.empty()
        
        # Convert temperature to Celsius for display if needed
        if param_choice in ['T_inlet', 'T_ambient']:
            param_values_display = [kelvin_to_celsius(v) for v in param_values]
            x_label = f"{param_choice} (°C)"
        else:
            param_values_display = param_values
            x_label = f"{param_choice} ({CONSTRAINTS[param_choice]['unit']})"
        
        # Create plots
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=('Power vs Parameter', 'Efficiency vs Parameter')
        )
        
        fig.add_trace(
            go.Scatter(x=param_values_display, y=predictions, mode='lines',
                      line=dict(color='#1f77b4', width=3)),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Scatter(x=param_values_display, y=efficiencies, mode='lines',
                      line=dict(color='#2ca02c', width=3)),
            row=1, col=2
        )
        
        fig.update_xaxes(title_text=x_label, row=1, col=1)
        fig.update_xaxes(title_text=x_label, row=1, col=2)
        fig.update_yaxes(title_text="Power (W/m²)", row=1, col=1)
        fig.update_yaxes(title_text="Efficiency (%)", row=1, col=2)
        
        fig.update_layout(height=500, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
        
        # Statistics
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Max Power", f"{max(predictions):.1f} W/m²")
        with col2:
            st.metric("Min Power", f"{min(predictions):.1f} W/m²")
        with col3:
            sensitivity = (max(predictions) - min(predictions)) / (param_max - param_min)
            st.metric("Sensitivity", f"{sensitivity:.2f} W/m² per unit")
    
    # ========================================================================
    # FOOTER
    # ========================================================================
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666;'>
        <p>🔬 Physics-Based Solar Thermal Collector Model | No Data Leakage</p>
        <p>Model: LightGBM | Features: 14 | Target: Specific Thermal Power (W/m²)</p>
    </div>
    """, unsafe_allow_html=True)

# ============================================================================
# RUN APPLICATION
# ============================================================================
if __name__ == "__main__":
    main()
