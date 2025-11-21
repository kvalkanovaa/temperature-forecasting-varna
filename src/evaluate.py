"""
Evaluation script for temperature forecasting models
Calculate RMSE, MAE, MAPE metrics
"""
import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error, mean_absolute_error
import pickle

sys.path.append(os.path.dirname(__file__))
import config


def calculate_metrics(y_true, y_pred):
    """
    Calculate RMSE, MAE, and MAPE
    
    Args:
        y_true: True values
        y_pred: Predicted values
        
    Returns:
        Dictionary with metrics
    """
    # RMSE
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    
    # MAE
    mae = mean_absolute_error(y_true, y_pred)
    
    # MAPE
    mape = np.mean(np.abs((y_true - y_pred) / (y_true + 1e-8))) * 100
    
    return {
        'RMSE': rmse,
        'MAE': mae,
        'MAPE': mape
    }


def evaluate_model(model, X_test, y_test, model_name, preprocessor):
    """
    Evaluate a single model
    
    Args:
        model: Trained model
        X_test: Test input data
        y_test: Test target data
        model_name: Name of the model
        preprocessor: Data preprocessor for inverse transform
        
    Returns:
        Dictionary with metrics and predictions
    """
    print(f"\n{'='*60}")
    print(f"EVALUATING {model_name}")
    print(f"{'='*60}")
    
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Inverse transform to get actual temperatures
    y_true_actual = preprocessor.inverse_transform_temperature(y_test.flatten())
    y_pred_actual = preprocessor.inverse_transform_temperature(y_pred.flatten())
    
    # Reshape back to original shape
    y_true_actual = y_true_actual.reshape(y_test.shape)
    y_pred_actual = y_pred_actual.reshape(y_pred.shape)
    
    # Calculate metrics
    metrics = calculate_metrics(y_true_actual, y_pred_actual)
    
    print(f"Results:")
    print(f"  RMSE: {metrics['RMSE']:.4f} °C")
    print(f"  MAE: {metrics['MAE']:.4f} °C")
    print(f"  MAPE: {metrics['MAPE']:.4f} %")
    
    return {
        'metrics': metrics,
        'y_true': y_true_actual,
        'y_pred': y_pred_actual
    }


def plot_predictions(results_dict, save_path, num_samples=5):
    """
    Plot sample predictions vs actual values for all models
    
    Args:
        results_dict: Dictionary with results for each model
        save_path: Path to save the plot
        num_samples: Number of sample sequences to plot
    """
    n_models = len(results_dict)
    fig, axes = plt.subplots(num_samples, n_models, figsize=(5*n_models, 3*num_samples))
    
    if num_samples == 1:
        axes = axes.reshape(1, -1)
    
    model_names = list(results_dict.keys())
    
    for sample_idx in range(num_samples):
        for model_idx, model_name in enumerate(model_names):
            ax = axes[sample_idx, model_idx]
            
            y_true = results_dict[model_name]['y_true'][sample_idx]
            y_pred = results_dict[model_name]['y_pred'][sample_idx]
            
            time_steps = np.arange(len(y_true)) * 30  # 30-minute intervals
            
            ax.plot(time_steps, y_true, label='Actual', linewidth=2, alpha=0.7)
            ax.plot(time_steps, y_pred, label='Predicted', linewidth=2, alpha=0.7)
            ax.set_xlabel('Time (minutes)')
            ax.set_ylabel('Temperature (°C)')
            ax.set_title(f'{model_name} - Sample {sample_idx+1}')
            ax.legend()
            ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Predictions plot saved to {save_path}")


def plot_metrics_comparison(metrics_df, save_path):
    """
    Plot comparison of metrics across models
    
    Args:
        metrics_df: DataFrame with metrics for all models
        save_path: Path to save the plot
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    metrics = ['RMSE', 'MAE', 'MAPE']
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']
    
    for idx, (metric, color) in enumerate(zip(metrics, colors)):
        ax = axes[idx]
        values = metrics_df[metric].values
        models = metrics_df.index.values
        
        bars = ax.bar(models, values, color=color, alpha=0.7, edgecolor='black')
        ax.set_ylabel(metric)
        ax.set_title(f'{metric} Comparison')
        ax.grid(True, alpha=0.3, axis='y')
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.3f}',
                   ha='center', va='bottom', fontweight='bold')
        
        # Rotate x labels if needed
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Metrics comparison plot saved to {save_path}")


def plot_error_distribution(results_dict, save_path):
    """
    Plot error distribution for all models
    
    Args:
        results_dict: Dictionary with results for each model
        save_path: Path to save the plot
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    axes = axes.flatten()
    
    for idx, (model_name, results) in enumerate(results_dict.items()):
        if idx >= 4:
            break
            
        ax = axes[idx]
        
        errors = results['y_true'].flatten() - results['y_pred'].flatten()
        
        ax.hist(errors, bins=50, color='skyblue', edgecolor='black', alpha=0.7)
        ax.axvline(0, color='red', linestyle='--', linewidth=2, label='Zero Error')
        ax.set_xlabel('Prediction Error (°C)')
        ax.set_ylabel('Frequency')
        ax.set_title(f'{model_name} - Error Distribution')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Add statistics
        mean_error = np.mean(errors)
        std_error = np.std(errors)
        ax.text(0.02, 0.98, f'Mean: {mean_error:.3f}°C\nStd: {std_error:.3f}°C',
               transform=ax.transAxes, verticalalignment='top',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Error distribution plot saved to {save_path}")


def main():
    """Main evaluation function"""
    print("\n" + "="*70)
    print("MODEL EVALUATION")
    print("="*70)
    
    # Load preprocessor
    preprocessor_path = os.path.join(config.MODELS_DIR, 'preprocessor.pkl')
    with open(preprocessor_path, 'rb') as f:
        preprocessor = pickle.load(f)
    print(f"✓ Loaded preprocessor from {preprocessor_path}")
    
    # Load test data
    from data_loader import WeatherDataLoader, DataPreprocessor
    
    loader = WeatherDataLoader()
    df = loader.load_or_fetch_data()
    
    df_normalized = preprocessor.normalize_data(df, fit=False)
    feature_cols = [col for col in df_normalized.columns if col != 'datetime']
    data = df_normalized[['datetime'] + feature_cols].copy()
    
    X, y = preprocessor.create_sequences(data, target_col='temperature_2m')
    
    # Split to get test data
    n = len(X)
    n_test = int(n * config.TEST_SPLIT)
    X_test, y_test = X[-n_test:], y[-n_test:]
    
    print(f"Test data shape: X={X_test.shape}, y={y_test.shape}")
    
    # Load and evaluate models
    from tensorflow import keras
    
    models_to_evaluate = {
        'LSTM': os.path.join(config.MODELS_DIR, 'lstm_best_model.h5'),
        'GRU': os.path.join(config.MODELS_DIR, 'gru_best_model.h5'),
        'Temporal_CNN': os.path.join(config.MODELS_DIR, 'tcnn_best_model.h5')
    }
    
    results_dict = {}
    metrics_list = []
    
    for model_name, model_path in models_to_evaluate.items():
        if os.path.exists(model_path):
            print(f"\nLoading {model_name} from {model_path}")
            model = keras.models.load_model(model_path)
            
            results = evaluate_model(model, X_test, y_test, model_name, preprocessor)
            results_dict[model_name] = results
            
            metrics_list.append({
                'Model': model_name,
                **results['metrics']
            })
        else:
            print(f"✗ Model file not found: {model_path}")
    
    # Create metrics DataFrame
    metrics_df = pd.DataFrame(metrics_list)
    metrics_df.set_index('Model', inplace=True)
    
    print("\n" + "="*70)
    print("METRICS SUMMARY")
    print("="*70)
    print(metrics_df.to_string())
    print("="*70)
    
    # Save metrics
    metrics_path = os.path.join(config.MODELS_DIR, 'evaluation_metrics.csv')
    metrics_df.to_csv(metrics_path)
    print(f"\n✓ Metrics saved to {metrics_path}")
    
    # Generate plots
    print("\nGenerating visualization plots...")
    
    plot_predictions(
        results_dict,
        os.path.join(config.MODELS_DIR, 'predictions_comparison.png'),
        num_samples=5
    )
    
    plot_metrics_comparison(
        metrics_df,
        os.path.join(config.MODELS_DIR, 'metrics_comparison.png')
    )
    
    plot_error_distribution(
        results_dict,
        os.path.join(config.MODELS_DIR, 'error_distribution.png')
    )
    
    print("\n✓ Evaluation complete!")


if __name__ == "__main__":
    main()
