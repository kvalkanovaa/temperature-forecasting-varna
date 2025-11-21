"""
LSTM model for temperature forecasting
"""
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
import config


class LSTMForecaster:
    """LSTM model for multi-step temperature forecasting"""
    
    def __init__(self, input_shape, output_steps=config.FORECAST_HORIZON,
                 units=config.LSTM_UNITS, dropout=config.DROPOUT_RATE):
        """
        Initialize LSTM model
        
        Args:
            input_shape: (sequence_length, n_features)
            output_steps: Number of steps to forecast (48 for 24h with 30min intervals)
            units: List of LSTM units for each layer
            dropout: Dropout rate
        """
        self.input_shape = input_shape
        self.output_steps = output_steps
        self.units = units
        self.dropout = dropout
        self.model = None
        self.history = None
        
    def build_model(self):
        """Build the LSTM model architecture"""
        model = keras.Sequential(name="LSTM_Forecaster")
        
        # First LSTM layer (return sequences for stacking)
        model.add(layers.LSTM(
            units=self.units[0],
            return_sequences=True,
            input_shape=self.input_shape,
            name="LSTM_Layer_1"
        ))
        model.add(layers.Dropout(self.dropout, name="Dropout_1"))
        
        # Second LSTM layer
        if len(self.units) > 1:
            model.add(layers.LSTM(
                units=self.units[1],
                return_sequences=True,
                name="LSTM_Layer_2"
            ))
            model.add(layers.Dropout(self.dropout, name="Dropout_2"))
        
        # Third LSTM layer (no return sequences)
        if len(self.units) > 2:
            model.add(layers.LSTM(
                units=self.units[2],
                return_sequences=False,
                name="LSTM_Layer_3"
            ))
            model.add(layers.Dropout(self.dropout, name="Dropout_3"))
        
        # Dense layers for output
        model.add(layers.Dense(128, activation='relu', name="Dense_1"))
        model.add(layers.Dropout(self.dropout, name="Dropout_Dense"))
        model.add(layers.Dense(64, activation='relu', name="Dense_2"))
        
        # Output layer - predicting next 48 temperature values
        model.add(layers.Dense(self.output_steps, name="Output"))
        
        self.model = model
        return model
    
    def compile_model(self, learning_rate=config.LEARNING_RATE):
        """Compile the model"""
        optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
        self.model.compile(
            optimizer=optimizer,
            loss='mse',
            metrics=['mae', 'mape']
        )
        
    def get_callbacks(self, model_path):
        """Get training callbacks"""
        callbacks = [
            keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=config.EARLY_STOPPING_PATIENCE,
                restore_best_weights=True,
                verbose=1
            ),
            keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=config.REDUCE_LR_PATIENCE,
                min_lr=1e-7,
                verbose=1
            ),
            keras.callbacks.ModelCheckpoint(
                filepath=model_path,
                monitor='val_loss',
                save_best_only=True,
                verbose=1
            )
        ]
        return callbacks
    
    def train(self, X_train, y_train, X_val, y_val, 
              epochs=config.EPOCHS, batch_size=config.BATCH_SIZE,
              model_path=None):
        """
        Train the LSTM model
        
        Args:
            X_train: Training input sequences
            y_train: Training targets
            X_val: Validation input sequences
            y_val: Validation targets
            epochs: Number of training epochs
            batch_size: Batch size
            model_path: Path to save the best model
            
        Returns:
            Training history
        """
        if self.model is None:
            self.build_model()
            self.compile_model()
        
        print("\n" + "="*60)
        print("LSTM MODEL TRAINING")
        print("="*60)
        print(f"Model: {self.model.name}")
        print(f"Input shape: {X_train.shape}")
        print(f"Output shape: {y_train.shape}")
        print(f"Total parameters: {self.model.count_params():,}")
        print("="*60 + "\n")
        
        if model_path is None:
            model_path = os.path.join(config.MODELS_DIR, 'lstm_best_model.keras')
        
        callbacks = self.get_callbacks(model_path)
        
        self.history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=1
        )
        
        print(f"\n✓ Training complete!")
        print(f"  Best val_loss: {min(self.history.history['val_loss']):.4f}")
        print(f"  Model saved to: {model_path}")
        
        return self.history
    
    def predict(self, X):
        """Make predictions"""
        return self.model.predict(X)
    
    def save_model(self, filepath):
        """Save the model"""
        self.model.save(filepath)
        print(f"✓ Model saved to {filepath}")
    
    def load_model(self, filepath):
        """Load a saved model"""
        self.model = keras.models.load_model(filepath)
        print(f"✓ Model loaded from {filepath}")
    
    def summary(self):
        """Print model summary"""
        if self.model:
            self.model.summary()
        else:
            print("Model not built yet. Call build_model() first.")


if __name__ == "__main__":
    # Test model creation
    input_shape = (config.SEQUENCE_LENGTH, len(config.FEATURES))
    
    lstm = LSTMForecaster(input_shape)
    lstm.build_model()
    lstm.compile_model()
    lstm.summary()
    
    print(f"\n✓ LSTM model created successfully")
    print(f"  Input: {input_shape}")
    print(f"  Output: {config.FORECAST_HORIZON} steps")
