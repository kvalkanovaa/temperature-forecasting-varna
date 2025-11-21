"""
Training script for all temperature forecasting models
"""
import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import tensorflow as tf

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

sys.path.append(os.path.dirname(__file__))
import config
from data_loader import WeatherDataLoader, DataPreprocessor
from models.lstm_model import LSTMForecaster
from models.gru_model import GRUForecaster
from models.tcnn_model import TemporalCNNForecaster
from models.prophet_model import ProphetForecaster


def plot_training_history(history, model_name, save_path):
    """Plot and save training history"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle(f'{model_name} Training History', fontsize=16)
    
    # Loss
    axes[0, 0].plot(history.history['loss'], label='Train Loss')
    axes[0, 0].plot(history.history['val_loss'], label='Val Loss')
    axes[0, 0].set_title('Model Loss (MSE)')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True)
    
    # MAE
    axes[0, 1].plot(history.history['mae'], label='Train MAE')
    axes[0, 1].plot(history.history['val_mae'], label='Val MAE')
    axes[0, 1].set_title('Mean Absolute Error')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('MAE')
    axes[0, 1].legend()
    axes[0, 1].grid(True)
    
    # MAPE
    axes[1, 0].plot(history.history['mape'], label='Train MAPE')
    axes[1, 0].plot(history.history['val_mape'], label='Val MAPE')
    axes[1, 0].set_title('Mean Absolute Percentage Error')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('MAPE (%)')
    axes[1, 0].legend()
    axes[1, 0].grid(True)
    
    # Learning rate (if available)
    if 'lr' in history.history:
        axes[1, 1].plot(history.history['lr'])
        axes[1, 1].set_title('Learning Rate')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('LR')
        axes[1, 1].set_yscale('log')
        axes[1, 1].grid(True)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Training history plot saved to {save_path}")


def train_deep_learning_models():
    """Train LSTM, GRU, and Temporal CNN models"""
    print("\n" + "="*70)
    print("TRAINING DEEP LEARNING MODELS")
    print("="*70)
    
    # Load and preprocess data
    print("\n1. Loading data...")
    loader = WeatherDataLoader()
    df = loader.load_or_fetch_data()
    loader.get_data_statistics(df)
    
    # Preprocess
    print("\n2. Preprocessing data...")
    preprocessor = DataPreprocessor()
    df_normalized = preprocessor.normalize_data(df, fit=True)
    
    # Create sequences
    print("\n3. Creating sequences...")
    feature_cols = [col for col in df_normalized.columns if col != 'datetime']
    data = df_normalized[['datetime'] + feature_cols].copy()
    
    X, y = preprocessor.create_sequences(data, target_col='temperature_2m')
    print(f"  X shape: {X.shape}")
    print(f"  y shape: {y.shape}")
    
    # Split data
    print("\n4. Splitting data...")
    X_train, X_val, X_test, y_train, y_val, y_test = preprocessor.split_data(X, y)
    
    # Input shape for models
    input_shape = (X_train.shape[1], X_train.shape[2])
    
    # Dictionary to store results
    results = {}
    
    # Train LSTM
    print("\n" + "="*70)
    print("TRAINING LSTM MODEL")
    print("="*70)
    lstm = LSTMForecaster(input_shape)
    lstm.build_model()
    lstm.compile_model()
    lstm.summary()
    
    lstm_path = os.path.join(config.MODELS_DIR, 'lstm_best_model.keras')
    lstm_history = lstm.train(
        X_train, y_train, X_val, y_val,
        model_path=lstm_path
    )
    
    plot_path = os.path.join(config.MODELS_DIR, 'lstm_training_history.png')
    plot_training_history(lstm_history, 'LSTM', plot_path)
    
    results['LSTM'] = {
        'model': lstm,
        'history': lstm_history,
        'best_val_loss': min(lstm_history.history['val_loss'])
    }
    
    # Train GRU
    print("\n" + "="*70)
    print("TRAINING GRU MODEL")
    print("="*70)
    gru = GRUForecaster(input_shape)
    gru.build_model()
    gru.compile_model()
    gru.summary()
    
    gru_path = os.path.join(config.MODELS_DIR, 'gru_best_model.keras')
    gru_history = gru.train(
        X_train, y_train, X_val, y_val,
        model_path=gru_path
    )
    
    plot_path = os.path.join(config.MODELS_DIR, 'gru_training_history.png')
    plot_training_history(gru_history, 'GRU', plot_path)
    
    results['GRU'] = {
        'model': gru,
        'history': gru_history,
        'best_val_loss': min(gru_history.history['val_loss'])
    }
    
    # Train Temporal CNN
    print("\n" + "="*70)
    print("TRAINING TEMPORAL CNN MODEL")
    print("="*70)
    tcnn = TemporalCNNForecaster(input_shape)
    tcnn.build_model()
    tcnn.compile_model()
    tcnn.summary()
    
    tcnn_path = os.path.join(config.MODELS_DIR, 'tcnn_best_model.keras')
    tcnn_history = tcnn.train(
        X_train, y_train, X_val, y_val,
        model_path=tcnn_path
    )
    
    plot_path = os.path.join(config.MODELS_DIR, 'tcnn_training_history.png')
    plot_training_history(tcnn_history, 'Temporal CNN', plot_path)
    
    results['Temporal_CNN'] = {
        'model': tcnn,
        'history': tcnn_history,
        'best_val_loss': min(tcnn_history.history['val_loss'])
    }
    
    # Save preprocessor
    import pickle
    preprocessor_path = os.path.join(config.MODELS_DIR, 'preprocessor.pkl')
    with open(preprocessor_path, 'wb') as f:
        pickle.dump(preprocessor, f)
    print(f"\n✓ Preprocessor saved to {preprocessor_path}")
    
    return results, preprocessor, (X_test, y_test)


def train_prophet_model():
    """Train Prophet model"""
    print("\n" + "="*70)
    print("TRAINING PROPHET MODEL")
    print("="*70)
    
    # Load data
    loader = WeatherDataLoader()
    df = loader.load_or_fetch_data()
    
    # Train Prophet
    prophet = ProphetForecaster()
    prophet.train(df, target_col='temperature_2m')
    
    # Save model
    prophet_path = os.path.join(config.MODELS_DIR, 'prophet_model.pkl')
    prophet.save_model(prophet_path)
    
    return prophet


def main():
    """Main training function"""
    start_time = datetime.now()
    
    # Create models directory if it doesn't exist
    os.makedirs(config.MODELS_DIR, exist_ok=True)
    
    print("\n" + "="*70)
    print("TEMPERATURE FORECASTING - MODEL TRAINING")
    print(f"City: {config.CITY_NAME}")
    print(f"Forecast horizon: {config.FORECAST_HORIZON} steps (24 hours)")
    print(f"Start time: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*70)
    
    # Train deep learning models
    dl_results, preprocessor, test_data = train_deep_learning_models()
    
    # Train Prophet
    prophet = train_prophet_model()
    
    # Print summary
    print("\n" + "="*70)
    print("TRAINING SUMMARY")
    print("="*70)
    for model_name, result in dl_results.items():
        print(f"{model_name}:")
        print(f"  Best validation loss: {result['best_val_loss']:.6f}")
    
    end_time = datetime.now()
    duration = end_time - start_time
    print(f"\n✓ All models trained successfully!")
    print(f"  Total training time: {duration}")
    print(f"  Models saved in: {config.MODELS_DIR}")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()
