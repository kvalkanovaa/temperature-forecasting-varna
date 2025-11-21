"""
Prophet model for temperature forecasting
"""
import numpy as np
import pandas as pd
from prophet import Prophet
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
import config
import pickle


class ProphetForecaster:
    """Prophet model for temperature forecasting"""
    
    def __init__(self, seasonality_mode=config.PROPHET_SEASONALITY_MODE,
                 yearly_seasonality=config.PROPHET_YEARLY_SEASONALITY,
                 weekly_seasonality=config.PROPHET_WEEKLY_SEASONALITY,
                 daily_seasonality=config.PROPHET_DAILY_SEASONALITY):
        """
        Initialize Prophet model
        
        Args:
            seasonality_mode: 'additive' or 'multiplicative'
            yearly_seasonality: Enable yearly seasonality
            weekly_seasonality: Enable weekly seasonality
            daily_seasonality: Enable daily seasonality
        """
        self.model = Prophet(
            seasonality_mode=seasonality_mode,
            yearly_seasonality=yearly_seasonality,
            weekly_seasonality=weekly_seasonality,
            daily_seasonality=daily_seasonality,
            changepoint_prior_scale=0.05,
            seasonality_prior_scale=10.0
        )
        self.is_fitted = False
        
    def prepare_data(self, df, target_col='temperature_2m'):
        """
        Prepare data for Prophet (requires 'ds' and 'y' columns)
        
        Args:
            df: DataFrame with 'datetime' and target column
            target_col: Name of the target column
            
        Returns:
            DataFrame in Prophet format
        """
        prophet_df = pd.DataFrame({
            'ds': df['datetime'],
            'y': df[target_col]
        })
        return prophet_df
    
    def train(self, df, target_col='temperature_2m'):
        """
        Train the Prophet model
        
        Args:
            df: DataFrame with weather data
            target_col: Column to predict
            
        Returns:
            Fitted model
        """
        print("\n" + "="*60)
        print("PROPHET MODEL TRAINING")
        print("="*60)
        print(f"Training data: {len(df)} records")
        print(f"Date range: {df['datetime'].min()} to {df['datetime'].max()}")
        print("="*60 + "\n")
        
        # Prepare data
        prophet_df = self.prepare_data(df, target_col)
        
        # Add additional regressors
        if 'relative_humidity_2m' in df.columns:
            self.model.add_regressor('humidity')
            prophet_df['humidity'] = df['relative_humidity_2m'].values
        
        if 'surface_pressure' in df.columns:
            self.model.add_regressor('pressure')
            prophet_df['pressure'] = df['surface_pressure'].values
        
        # Fit the model
        self.model.fit(prophet_df)
        self.is_fitted = True
        
        print(f"\n✓ Prophet training complete!")
        
        return self.model
    
    def predict(self, df=None, periods=config.FORECAST_HORIZON, freq='30min'):
        """
        Make predictions
        
        Args:
            df: DataFrame with future data (if None, creates future dataframe)
            periods: Number of periods to forecast
            freq: Frequency of predictions
            
        Returns:
            DataFrame with predictions
        """
        if not self.is_fitted:
            raise ValueError("Model must be trained before making predictions")
        
        if df is None:
            # Create future dataframe
            future = self.model.make_future_dataframe(periods=periods, freq=freq)
        else:
            future = self.prepare_data(df)
        
        # Make predictions
        forecast = self.model.predict(future)
        
        return forecast
    
    def predict_next_24h(self, current_data=None):
        """
        Predict next 24 hours (48 steps with 30min intervals)
        
        Args:
            current_data: Current weather data (optional)
            
        Returns:
            Array of 48 temperature predictions
        """
        forecast = self.predict(periods=config.FORECAST_HORIZON, freq='30min')
        
        # Get last 48 predictions
        predictions = forecast['yhat'].tail(config.FORECAST_HORIZON).values
        
        return predictions
    
    def save_model(self, filepath):
        """Save the model"""
        with open(filepath, 'wb') as f:
            pickle.dump(self.model, f)
        print(f"✓ Prophet model saved to {filepath}")
    
    def load_model(self, filepath):
        """Load a saved model"""
        with open(filepath, 'rb') as f:
            self.model = pickle.load(f)
        self.is_fitted = True
        print(f"✓ Prophet model loaded from {filepath}")
    
    def plot_forecast(self, forecast):
        """Plot the forecast"""
        fig = self.model.plot(forecast)
        return fig
    
    def plot_components(self, forecast):
        """Plot forecast components"""
        fig = self.model.plot_components(forecast)
        return fig


if __name__ == "__main__":
    print("✓ Prophet forecaster module loaded successfully")
    print(f"  Forecast horizon: {config.FORECAST_HORIZON} steps (24 hours)")
    print(f"  Seasonality mode: {config.PROPHET_SEASONALITY_MODE}")
