"""
Streamlit Web Application for Temperature Forecasting
Demonstration interface for 24-hour temperature prediction
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import os
import sys
import pickle

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

import config
from src.data_loader import WeatherDataLoader, DataPreprocessor


# Page configuration
st.set_page_config(
    page_title="Temperature Forecasting - Varna",
    page_icon="🌡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .stAlert {
        margin-top: 1rem;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_models():
    """Load all trained models"""
    from tensorflow import keras
    
    models = {}
    model_paths = {
        'LSTM': os.path.join(config.MODELS_DIR, 'lstm_best_model.keras'),
        'GRU': os.path.join(config.MODELS_DIR, 'gru_best_model.keras'),
        'Temporal CNN': os.path.join(config.MODELS_DIR, 'tcnn_best_model.keras')
    }
    
    for name, path in model_paths.items():
        if os.path.exists(path):
            models[name] = keras.models.load_model(path, compile=False)
            st.sidebar.success(f"✓ {name} loaded")
        else:
            st.sidebar.warning(f"✗ {name} not found")
    
    # Load Prophet separately
    prophet_path = os.path.join(config.MODELS_DIR, 'prophet_model.pkl')
    if os.path.exists(prophet_path):
        with open(prophet_path, 'rb') as f:
            models['Prophet'] = pickle.load(f)
        st.sidebar.success(f"✓ Prophet loaded")
    
    return models


@st.cache_resource
def load_preprocessor():
    """Load data preprocessor"""
    preprocessor_path = os.path.join(config.MODELS_DIR, 'preprocessor.pkl')
    if os.path.exists(preprocessor_path):
        with open(preprocessor_path, 'rb') as f:
            preprocessor = pickle.load(f)
        return preprocessor
    return None


@st.cache_data
def load_recent_data():
    """Load recent weather data"""
    loader = WeatherDataLoader()
    
    # Load last 30 days for context
    end_date = datetime.now().strftime('%Y-%m-%d')
    start_date = (datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d')
    
    df = loader.fetch_historical_data(start_date, end_date)
    return df


def prepare_input_sequence(df, preprocessor):
    """Prepare input sequence for prediction"""
    # Normalize
    df_normalized = preprocessor.normalize_data(df, fit=False)
    
    # Get last sequence_length points
    feature_cols = [col for col in df_normalized.columns if col != 'datetime']
    last_sequence = df_normalized[feature_cols].tail(config.SEQUENCE_LENGTH).values
    
    # Reshape for model input
    X = last_sequence.reshape(1, config.SEQUENCE_LENGTH, len(feature_cols))
    
    return X


def make_predictions(models, X, preprocessor):
    """Make predictions with all models"""
    predictions = {}
    
    for model_name, model in models.items():
        if model_name == 'Prophet':
            # Prophet handles differently
            continue
        
        # Predict
        y_pred = model.predict(X, verbose=0)
        
        # Inverse transform
        y_pred_actual = preprocessor.inverse_transform_temperature(y_pred.flatten())
        
        predictions[model_name] = y_pred_actual
    
    return predictions


def create_forecast_plot(predictions, current_temp, start_time):
    """Create interactive forecast plot"""
    fig = go.Figure()
    
    # Time steps (30-min intervals for 24 hours)
    time_steps = [start_time + timedelta(minutes=30*i) for i in range(config.FORECAST_HORIZON)]
    
    # Add traces for each model
    colors = {
        'LSTM': '#FF6B6B',
        'GRU': '#4ECDC4',
        'Temporal CNN': '#45B7D1',
        'Prophet': '#FFA07A'
    }
    
    for model_name, temps in predictions.items():
        fig.add_trace(go.Scatter(
            x=time_steps,
            y=temps,
            mode='lines+markers',
            name=model_name,
            line=dict(color=colors.get(model_name, '#333'), width=2),
            marker=dict(size=4)
        ))
    
    # Add current temperature point
    fig.add_trace(go.Scatter(
        x=[start_time],
        y=[current_temp],
        mode='markers',
        name='Current Temperature',
        marker=dict(color='red', size=15, symbol='star')
    ))
    
    fig.update_layout(
        title='24-Hour Temperature Forecast (30-minute intervals)',
        xaxis_title='Time',
        yaxis_title='Temperature (°C)',
        hovermode='x unified',
        height=500,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    
    return fig


def create_statistics_table(predictions):
    """Create statistics table for predictions"""
    stats = []
    
    for model_name, temps in predictions.items():
        stats.append({
            'Model': model_name,
            'Min (°C)': f"{np.min(temps):.2f}",
            'Max (°C)': f"{np.max(temps):.2f}",
            'Mean (°C)': f"{np.mean(temps):.2f}",
            'Std Dev (°C)': f"{np.std(temps):.2f}"
        })
    
    return pd.DataFrame(stats)


def create_hourly_table(predictions, start_time):
    """Create hourly forecast table"""
    # Get hourly predictions (every 2nd point since we have 30-min intervals)
    hours = []
    
    for i in range(0, 48, 2):  # 0, 2, 4, ... 46 (24 hours)
        hour_time = start_time + timedelta(minutes=30*i)
        row = {'Time': hour_time.strftime('%H:%M')}
        
        for model_name, temps in predictions.items():
            row[model_name] = f"{temps[i]:.1f}°C"
        
        hours.append(row)
    
    return pd.DataFrame(hours)


def main():
    # Header
    st.markdown('<div class="main-header">🌡️ Temperature Forecasting for Varna, Bulgaria</div>', 
                unsafe_allow_html=True)
    st.markdown("### 24-Hour Temperature Prediction with Machine Learning")
    st.markdown("---")
    
    # Sidebar
    st.sidebar.title("⚙️ Settings")
    st.sidebar.markdown("### Model Information")
    
    # Load models
    with st.spinner("Loading models..."):
        models = load_models()
        preprocessor = load_preprocessor()
    
    if not models or preprocessor is None:
        st.error("❌ Models not found! Please train the models first by running `python src/train.py`")
        st.stop()
    
    st.sidebar.markdown(f"**Loaded Models:** {len(models)}")
    st.sidebar.markdown(f"**Forecast Horizon:** 24 hours")
    st.sidebar.markdown(f"**Time Step:** 30 minutes")
    st.sidebar.markdown(f"**Total Predictions:** 48 steps")
    
    # Main content
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("📊 Current Weather Data")
        
    # Load recent data
    with st.spinner("Fetching current weather data..."):
        try:
            df_recent = load_recent_data()
            
            # Get current conditions
            latest = df_recent.iloc[-1]
            current_temp = latest['temperature_2m']
            current_humidity = latest['relative_humidity_2m']
            current_pressure = latest['surface_pressure']
            current_time = latest['datetime']
            
            # Display current conditions
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("🌡️ Temperature", f"{current_temp:.1f}°C")
            with col2:
                st.metric("💧 Humidity", f"{current_humidity:.1f}%")
            with col3:
                st.metric("📊 Pressure", f"{current_pressure:.1f} hPa")
            with col4:
                st.metric("🕐 Updated", current_time.strftime('%H:%M'))
            
            # Show info about data timestamp
            st.info(f"📅 Latest available data from API: {current_time.strftime('%Y-%m-%d %H:%M')} | Current time: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
            
            st.success("✅ Weather data loaded successfully!")
            
        except Exception as e:
            st.error(f"❌ Error loading data: {e}")
            st.stop()
    
    # Prediction button
    st.markdown("---")
    
    if st.button("🔮 Generate 24-Hour Forecast", type="primary", use_container_width=True):
        with st.spinner("Generating predictions..."):
            # Prepare input
            X = prepare_input_sequence(df_recent, preprocessor)
            
            # Make predictions
            predictions = make_predictions(models, X, preprocessor)
            
            # Display results
            st.markdown("## 📈 Forecast Results")
            
            # Plot
            fig = create_forecast_plot(predictions, current_temp, current_time)
            st.plotly_chart(fig, use_container_width=True)
            
            # Statistics
            st.markdown("### 📊 Forecast Statistics")
            stats_df = create_statistics_table(predictions)
            st.dataframe(stats_df, use_container_width=True, hide_index=True)
            
            # Hourly table
            st.markdown("### 🕐 Hourly Forecast Table")
            hourly_df = create_hourly_table(predictions, current_time)
            st.dataframe(hourly_df, use_container_width=True, hide_index=True)
            
            # Download button
            csv = hourly_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 Download Forecast (CSV)",
                data=csv,
                file_name=f"temperature_forecast_{current_time.strftime('%Y%m%d_%H%M')}.csv",
                mime="text/csv",
            )
            
            st.success("✅ Forecast generated successfully!")
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666;">
        <p><b>Temperature Forecasting System</b> | Varna, Bulgaria</p>
        <p>Models: LSTM, GRU, Temporal CNN, Prophet | Data: Open-Meteo API</p>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
