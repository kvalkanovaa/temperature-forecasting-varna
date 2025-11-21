"""
Configuration file for temperature forecasting project
"""
import os

# Project paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, 'data')
MODELS_DIR = os.path.join(BASE_DIR, 'saved_models')

# Varna coordinates
LATITUDE = 43.2141
LONGITUDE = 27.9147
CITY_NAME = "Varna"

# Data parameters
START_DATE = "2020-01-01"
END_DATE = "2025-11-01"
FEATURES = ['temperature_2m', 'relative_humidity_2m', 'surface_pressure']

# Model parameters
SEQUENCE_LENGTH = 48  # 24 часа * 2 (30-мин интервали) = 48 стъпки назад
FORECAST_HORIZON = 48  # 24 часа напред със стъпка 30 мин
BATCH_SIZE = 128  # Увеличен за по-бързо обучение с 6+ години данни
EPOCHS = 15  # Намален заради по-голям dataset
LEARNING_RATE = 0.001
VALIDATION_SPLIT = 0.2
TEST_SPLIT = 0.1

# LSTM/GRU parameters
LSTM_UNITS = [128, 64, 32]
DROPOUT_RATE = 0.2

# Temporal CNN parameters
CNN_FILTERS = [64, 128, 64]
KERNEL_SIZE = 3

# Prophet parameters
PROPHET_SEASONALITY_MODE = 'multiplicative'
PROPHET_YEARLY_SEASONALITY = True
PROPHET_WEEKLY_SEASONALITY = True
PROPHET_DAILY_SEASONALITY = True

# Training parameters
EARLY_STOPPING_PATIENCE = 15
REDUCE_LR_PATIENCE = 5
RANDOM_SEED = 42
