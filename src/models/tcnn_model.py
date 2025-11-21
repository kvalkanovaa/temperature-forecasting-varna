"""
Temporal Convolutional Network (TCN) for temperature forecasting
"""
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
import config


class TemporalCNNForecaster:
    """Temporal CNN model for multi-step temperature forecasting"""
    
    def __init__(self, input_shape, output_steps=config.FORECAST_HORIZON,
                 filters=config.CNN_FILTERS, kernel_size=config.KERNEL_SIZE,
                 dropout=config.DROPOUT_RATE):
        """
        Initialize Temporal CNN model
        
        Args:
            input_shape: (sequence_length, n_features)
            output_steps: Number of steps to forecast
            filters: List of filter sizes for each Conv1D layer
            kernel_size: Kernel size for convolutions
            dropout: Dropout rate
        """
        self.input_shape = input_shape
        self.output_steps = output_steps
        self.filters = filters
        self.kernel_size = kernel_size
        self.dropout = dropout
        self.model = None
        self.history = None
        
    def build_model(self):
        """Build the Temporal CNN model architecture"""
        model = keras.Sequential(name="Temporal_CNN_Forecaster")
        
        # First Conv1D layer
        model.add(layers.Conv1D(
            filters=self.filters[0],
            kernel_size=self.kernel_size,
            padding='causal',
            activation='relu',
            input_shape=self.input_shape,
            name="Conv1D_1"
        ))
        model.add(layers.Dropout(self.dropout, name="Dropout_1"))
        model.add(layers.MaxPooling1D(pool_size=2, name="MaxPool_1"))
        
        # Second Conv1D layer
        if len(self.filters) > 1:
            model.add(layers.Conv1D(
                filters=self.filters[1],
                kernel_size=self.kernel_size,
                padding='causal',
                activation='relu',
                name="Conv1D_2"
            ))
            model.add(layers.Dropout(self.dropout, name="Dropout_2"))
            model.add(layers.MaxPooling1D(pool_size=2, name="MaxPool_2"))
        
        # Third Conv1D layer
        if len(self.filters) > 2:
            model.add(layers.Conv1D(
                filters=self.filters[2],
                kernel_size=self.kernel_size,
                padding='causal',
                activation='relu',
                name="Conv1D_3"
            ))
            model.add(layers.Dropout(self.dropout, name="Dropout_3"))
        
        # Global pooling to reduce dimensionality
        model.add(layers.GlobalAveragePooling1D(name="GlobalAvgPool"))
        
        # Dense layers
        model.add(layers.Dense(256, activation='relu', name="Dense_1"))
        model.add(layers.Dropout(self.dropout, name="Dropout_Dense_1"))
        model.add(layers.Dense(128, activation='relu', name="Dense_2"))
        model.add(layers.Dropout(self.dropout, name="Dropout_Dense_2"))
        model.add(layers.Dense(64, activation='relu', name="Dense_3"))
        
        # Output layer
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
        """Train the Temporal CNN model"""
        if self.model is None:
            self.build_model()
            self.compile_model()
        
        print("\n" + "="*60)
        print("TEMPORAL CNN MODEL TRAINING")
        print("="*60)
        print(f"Model: {self.model.name}")
        print(f"Input shape: {X_train.shape}")
        print(f"Output shape: {y_train.shape}")
        print(f"Total parameters: {self.model.count_params():,}")
        print("="*60 + "\n")
        
        if model_path is None:
            model_path = os.path.join(config.MODELS_DIR, 'tcnn_best_model.keras')
        
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
    
    tcnn = TemporalCNNForecaster(input_shape)
    tcnn.build_model()
    tcnn.compile_model()
    tcnn.summary()
    
    print(f"\n✓ Temporal CNN model created successfully")
    print(f"  Input: {input_shape}")
    print(f"  Output: {config.FORECAST_HORIZON} steps")
