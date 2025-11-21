"""
Data loading and preprocessing module for weather data from Open-Meteo API
"""
import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
import json
from tqdm import tqdm
import config
import urllib3

# Disable SSL warnings
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

class WeatherDataLoader:
    """Class for loading historical weather data from Open-Meteo API"""
    
    def __init__(self, latitude=config.LATITUDE, longitude=config.LONGITUDE):
        self.latitude = latitude
        self.longitude = longitude
        self.base_url = "https://archive-api.open-meteo.com/v1/archive"
        
    def fetch_historical_data(self, start_date, end_date, save_path=None):
        """
        Fetch historical weather data from Open-Meteo API
        
        Args:
            start_date: Start date in format 'YYYY-MM-DD'
            end_date: End date in format 'YYYY-MM-DD'
            save_path: Path to save the data (optional)
            
        Returns:
            DataFrame with weather data
        """
        print(f"Fetching weather data for {config.CITY_NAME}...")
        print(f"Period: {start_date} to {end_date}")
        
        params = {
            "latitude": self.latitude,
            "longitude": self.longitude,
            "start_date": start_date,
            "end_date": end_date,
            "hourly": "temperature_2m,relative_humidity_2m,surface_pressure",
            "timezone": "Europe/Sofia"
        }
        
        try:
            response = requests.get(self.base_url, params=params, verify=False)
            response.raise_for_status()
            data = response.json()
            
            # Parse the data
            df = pd.DataFrame({
                'datetime': pd.to_datetime(data['hourly']['time']),
                'temperature_2m': data['hourly']['temperature_2m'],
                'relative_humidity_2m': data['hourly']['relative_humidity_2m'],
                'surface_pressure': data['hourly']['surface_pressure']
            })
            
            # Resample to 30-minute intervals using interpolation
            df.set_index('datetime', inplace=True)
            df = df.resample('30min').interpolate(method='linear')
            df.reset_index(inplace=True)
            
            print(f"✓ Successfully fetched {len(df)} records")
            print(f"  Date range: {df['datetime'].min()} to {df['datetime'].max()}")
            print(f"  Features: {list(df.columns)}")
            
            # Save if path provided
            if save_path:
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                df.to_csv(save_path, index=False)
                print(f"✓ Data saved to {save_path}")
            
            return df
            
        except Exception as e:
            print(f"✗ Error fetching data: {e}")
            raise
    
    def load_or_fetch_data(self, start_date=config.START_DATE, 
                          end_date=config.END_DATE,
                          force_download=False):
        """
        Load data from file or fetch if not exists
        
        Args:
            start_date: Start date
            end_date: End date
            force_download: Force re-download even if file exists
            
        Returns:
            DataFrame with weather data
        """
        filename = f"weather_data_{config.CITY_NAME.lower()}_{start_date}_{end_date}.csv"
        filepath = os.path.join(config.DATA_DIR, filename)
        
        if os.path.exists(filepath) and not force_download:
            print(f"Loading existing data from {filepath}")
            df = pd.read_csv(filepath)
            df['datetime'] = pd.to_datetime(df['datetime'])
            print(f"✓ Loaded {len(df)} records")
            return df
        else:
            return self.fetch_historical_data(start_date, end_date, filepath)
    
    def get_data_statistics(self, df):
        """Print statistics about the dataset"""
        print("\n" + "="*60)
        print("DATASET STATISTICS")
        print("="*60)
        print(f"Total records: {len(df)}")
        print(f"Date range: {df['datetime'].min()} to {df['datetime'].max()}")
        print(f"Duration: {(df['datetime'].max() - df['datetime'].min()).days} days")
        print(f"\nMissing values:")
        print(df.isnull().sum())
        print(f"\nBasic statistics:")
        print(df.describe())
        print("="*60 + "\n")


class DataPreprocessor:
    """Class for preprocessing weather data for ML models"""
    
    def __init__(self, sequence_length=config.SEQUENCE_LENGTH,
                 forecast_horizon=config.FORECAST_HORIZON):
        self.sequence_length = sequence_length
        self.forecast_horizon = forecast_horizon
        self.scalers = {}
        
    def normalize_data(self, df, fit=True):
        """
        Normalize the data using MinMaxScaler
        
        Args:
            df: DataFrame with features
            fit: Whether to fit the scaler (True for training data)
            
        Returns:
            Normalized DataFrame
        """
        from sklearn.preprocessing import MinMaxScaler
        
        df_normalized = df.copy()
        feature_columns = [col for col in df.columns if col != 'datetime']
        
        for col in feature_columns:
            if fit:
                scaler = MinMaxScaler()
                df_normalized[col] = scaler.fit_transform(df[[col]])
                self.scalers[col] = scaler
            else:
                if col in self.scalers:
                    df_normalized[col] = self.scalers[col].transform(df[[col]])
                else:
                    raise ValueError(f"No scaler found for column {col}")
        
        return df_normalized
    
    def create_sequences(self, data, target_col='temperature_2m'):
        """
        Create sequences for time series prediction
        
        Args:
            data: Normalized data array
            target_col: Column name to predict
            
        Returns:
            X, y arrays for training
        """
        X, y = [], []
        
        # Get feature columns (exclude datetime)
        feature_cols = [col for col in data.columns if col != 'datetime']
        data_values = data[feature_cols].values.astype(np.float32)
        
        # Get target column index
        target_idx = feature_cols.index(target_col)
        
        for i in range(len(data) - self.sequence_length - self.forecast_horizon + 1):
            # Input sequence
            X.append(data_values[i:i + self.sequence_length])
            # Target: next 48 temperature values (24 hours)
            y.append(data_values[i + self.sequence_length:
                              i + self.sequence_length + self.forecast_horizon, 
                              target_idx])
        
        return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)
    
    def split_data(self, X, y, val_split=config.VALIDATION_SPLIT, 
                   test_split=config.TEST_SPLIT):
        """
        Split data into train, validation, and test sets
        
        Args:
            X: Input sequences
            y: Target sequences
            val_split: Validation split ratio
            test_split: Test split ratio
            
        Returns:
            X_train, X_val, X_test, y_train, y_val, y_test
        """
        n = len(X)
        n_test = int(n * test_split)
        n_val = int(n * val_split)
        n_train = n - n_test - n_val
        
        X_train, y_train = X[:n_train], y[:n_train]
        X_val, y_val = X[n_train:n_train+n_val], y[n_train:n_train+n_val]
        X_test, y_test = X[n_train+n_val:], y[n_train+n_val:]
        
        print(f"\nData split:")
        print(f"  Training: {len(X_train)} samples")
        print(f"  Validation: {len(X_val)} samples")
        print(f"  Test: {len(X_test)} samples")
        
        return X_train, X_val, X_test, y_train, y_val, y_test
    
    def inverse_transform_temperature(self, values):
        """Inverse transform temperature predictions"""
        if 'temperature_2m' in self.scalers:
            return self.scalers['temperature_2m'].inverse_transform(
                values.reshape(-1, 1)
            ).flatten()
        return values


if __name__ == "__main__":
    # Test the data loader
    loader = WeatherDataLoader()
    df = loader.load_or_fetch_data()
    loader.get_data_statistics(df)
    
    # Test preprocessing
    preprocessor = DataPreprocessor()
    df_normalized = preprocessor.normalize_data(df)
    print(f"\n✓ Data normalized successfully")
    print(f"  Shape: {df_normalized.shape}")
