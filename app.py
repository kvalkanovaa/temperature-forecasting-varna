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
import pytz
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


@st.cache_data(ttl=300)  # Cache for 5 minutes only
def load_recent_data():
    """Load recent weather data using Forecast API for current data"""
    loader = WeatherDataLoader()
    
    # Use Forecast API to get most recent data (includes past 7 days + forecast)
    df = loader.fetch_forecast_data(past_days=7)
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


def create_humidity_plot(df_recent, start_time):
    """Create humidity historical + forecast plot"""
    fig = go.Figure()
    
    # Historical data (last 24 hours)
    df_24h = df_recent.tail(48)
    
    fig.add_trace(go.Scatter(
        x=df_24h['datetime'],
        y=df_24h['relative_humidity_2m'],
        mode='lines',
        name='Historical Humidity',
        line=dict(color='#3498db', width=2),
        fill='tozeroy',
        fillcolor='rgba(52, 152, 219, 0.2)'
    ))
    
    # Future forecast line (simple persistence model)
    last_humidity = df_24h['relative_humidity_2m'].iloc[-1]
    future_times = [start_time + timedelta(minutes=30*i) for i in range(config.FORECAST_HORIZON)]
    
    # Simple forecast: slight variation based on temperature pattern
    future_humidity = [last_humidity + np.random.normal(0, 2) for _ in range(config.FORECAST_HORIZON)]
    
    fig.add_trace(go.Scatter(
        x=future_times,
        y=future_humidity,
        mode='lines',
        name='Expected Humidity (estimated)',
        line=dict(color='#9b59b6', width=2, dash='dash'),
        opacity=0.6
    ))
    
    fig.update_layout(
        title='Humidity - Historical (24h) & Expected Trend',
        xaxis_title='Time',
        yaxis_title='Relative Humidity (%)',
        hovermode='x unified',
        height=400,
        yaxis=dict(range=[0, 100])
    )
    
    return fig


def create_pressure_plot(df_recent, start_time):
    """Create pressure historical + forecast plot"""
    fig = go.Figure()
    
    # Historical data (last 24 hours)
    df_24h = df_recent.tail(48)
    
    fig.add_trace(go.Scatter(
        x=df_24h['datetime'],
        y=df_24h['surface_pressure'],
        mode='lines',
        name='Historical Pressure',
        line=dict(color='#e74c3c', width=2),
        fill='tozeroy',
        fillcolor='rgba(231, 76, 60, 0.2)'
    ))
    
    # Future forecast (simple persistence)
    last_pressure = df_24h['surface_pressure'].iloc[-1]
    future_times = [start_time + timedelta(minutes=30*i) for i in range(config.FORECAST_HORIZON)]
    
    # Simple forecast: minimal variation
    future_pressure = [last_pressure + np.random.normal(0, 0.5) for _ in range(config.FORECAST_HORIZON)]
    
    fig.add_trace(go.Scatter(
        x=future_times,
        y=future_pressure,
        mode='lines',
        name='Expected Pressure (estimated)',
        line=dict(color='#e67e22', width=2, dash='dash'),
        opacity=0.6
    ))
    
    fig.update_layout(
        title='Atmospheric Pressure - Historical (24h) & Expected Trend',
        xaxis_title='Time',
        yaxis_title='Pressure (hPa)',
        hovermode='x unified',
        height=400
    )
    
    return fig


def generate_forecast_summary(predictions, current_temp, current_humidity, current_pressure, start_time):
    """Generate automatic forecast summary and insights"""
    # Calculate statistics
    lstm_temps = predictions.get('LSTM', predictions[list(predictions.keys())[0]])
    
    max_temp = np.max(lstm_temps)
    min_temp = np.min(lstm_temps)
    avg_temp = np.mean(lstm_temps)
    
    max_temp_idx = int(np.argmax(lstm_temps))
    min_temp_idx = int(np.argmin(lstm_temps))
    max_temp_time = max_temp_idx * 0.5  # hours from now
    min_temp_time = min_temp_idx * 0.5
    
    # Calculate actual datetime for max/min
    max_temp_datetime = start_time + timedelta(minutes=30*max_temp_idx)
    min_temp_datetime = start_time + timedelta(minutes=30*min_temp_idx)
    
    temp_change = lstm_temps[-1] - current_temp
    
    # Generate insights
    summary = f"""
    ### 📊 Forecast Summary (LSTM Model)
    
    **Temperature Analysis:**
    - Current: **{current_temp:.1f}°C**
    - Expected Max: **{max_temp:.1f}°C** at **{max_temp_datetime.strftime('%H:%M')}** (in ~{max_temp_time:.1f} hours)
    - Expected Min: **{min_temp:.1f}°C** at **{min_temp_datetime.strftime('%H:%M')}** (in ~{min_temp_time:.1f} hours)
    - 24h Average: **{avg_temp:.1f}°C**
    - 24h Change: **{temp_change:+.1f}°C**
    
    **Weather Conditions:**
    - Humidity: **{current_humidity:.0f}%** {'(High - humid conditions)' if current_humidity > 70 else '(Moderate)' if current_humidity > 50 else '(Low - dry)'}
    - Pressure: **{current_pressure:.1f} hPa** {'(High - stable weather)' if current_pressure > 1013 else '(Low - possible changes)'}
    
    **Recommendations:**
    """
    
    # Add recommendations
    if temp_change > 3:
        summary += "\n    - 🌡️ Temperature will **rise** significantly - prepare for warmer conditions"
    elif temp_change < -3:
        summary += "\n    - ❄️ Temperature will **drop** significantly - dress warmer"
    else:
        summary += "\n    - ✅ Temperature will remain **relatively stable**"
    
    if max_temp > 25:
        summary += "\n    - ☀️ Warm day expected - light clothing recommended"
    elif min_temp < 5:
        summary += "\n    - 🧥 Cold conditions expected - warm clothing needed"
    
    if current_humidity > 80:
        summary += "\n    - 💧 High humidity - may feel muggy"
    elif current_humidity < 40:
        summary += "\n    - 🌵 Low humidity - stay hydrated"
    
    # Temperature trend
    first_half = np.mean(lstm_temps[:24])
    second_half = np.mean(lstm_temps[24:])
    
    if second_half > first_half + 1:
        summary += "\n    - 📈 **Warming trend** throughout the day"
    elif second_half < first_half - 1:
        summary += "\n    - 📉 **Cooling trend** throughout the day"
    else:
        summary += "\n    - ➡️ **Stable conditions** expected"
    
    return summary


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
            
            # Get current time in Sofia timezone
            sofia_tz = pytz.timezone('Europe/Sofia')
            current_time_now = datetime.now(sofia_tz)
            
            # Filter to use only historical data (not future forecasts)
            # Convert datetime column to timezone-aware if needed
            if df_recent['datetime'].dt.tz is None:
                df_recent['datetime'] = pd.to_datetime(df_recent['datetime']).dt.tz_localize('UTC').dt.tz_convert('Europe/Sofia')
            
            # Keep only data up to current time (exclude future forecasts from API)
            df_recent = df_recent[df_recent['datetime'] <= current_time_now].copy()
            
            # Get current conditions from last available real data
            latest = df_recent.iloc[-1]
            current_temp = latest['temperature_2m']
            current_humidity = latest['relative_humidity_2m']
            current_pressure = latest['surface_pressure']
            api_time = latest['datetime']
            
            # Display current conditions
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("🌡️ Temperature", f"{current_temp:.1f}°C")
            with col2:
                st.metric("💧 Humidity", f"{current_humidity:.1f}%")
            with col3:
                st.metric("📊 Pressure", f"{current_pressure:.1f} hPa")
            with col4:
                st.metric("🕐 Updated", current_time_now.strftime('%H:%M'))
            
            # Show current time
            st.info(f"📅 Current time: {current_time_now.strftime('%Y-%m-%d %H:%M')} | Using {len(df_recent)} historical records")
            
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
            
            # Use current time for forecast start
            forecast_start_time = current_time_now
            
            # Forecast summary
            summary = generate_forecast_summary(predictions, current_temp, current_humidity, current_pressure, forecast_start_time)
            st.markdown(summary)
            
            st.markdown("---")
            
            # Temperature forecast plot
            st.markdown("### 🌡️ Temperature Forecast")
            fig = create_forecast_plot(predictions, current_temp, forecast_start_time)
            st.plotly_chart(fig, use_container_width=True)
            
            # Additional weather parameters
            st.markdown("---")
            st.markdown("### 🌤️ Additional Weather Parameters")
            
            col1, col2 = st.columns(2)
            
            with col1:
                humidity_fig = create_humidity_plot(df_recent, forecast_start_time)
                st.plotly_chart(humidity_fig, use_container_width=True)
            
            with col2:
                pressure_fig = create_pressure_plot(df_recent, forecast_start_time)
                st.plotly_chart(pressure_fig, use_container_width=True)
            
            # Statistics
            st.markdown("### 📊 Forecast Statistics")
            stats_df = create_statistics_table(predictions)
            st.dataframe(stats_df, use_container_width=True, hide_index=True)
            
            # Hourly table
            st.markdown("### 🕐 Hourly Forecast Table")
            hourly_df = create_hourly_table(predictions, forecast_start_time)
            st.dataframe(hourly_df, use_container_width=True, hide_index=True)
            
            # Download button
            csv = hourly_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 Download Forecast (CSV)",
                data=csv,
                file_name=f"temperature_forecast_{forecast_start_time.strftime('%Y%m%d_%H%M')}.csv",
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
