"""
Evaluation metrics for stock prediction models
Consolidates all metric calculations (RMSE, MAPE, Directional Accuracy)
"""
import numpy as np
import pandas as pd


class MetricsCalculator:
    """Unified metrics calculator for prediction evaluation"""
    
    @staticmethod
    def calculate_rmse(actual, predicted):
        """Calculate Root Mean Square Error"""
        try:
            actual_np = np.array(actual)
            predicted_np = np.array(predicted)
            
            # Filter out NaN values
            mask = ~(np.isnan(actual_np) | np.isnan(predicted_np))
            actual_clean = actual_np[mask]
            predicted_clean = predicted_np[mask]
            
            if len(actual_clean) == 0:
                return np.nan
            
            return np.sqrt(np.mean((actual_clean - predicted_clean) ** 2))
        except Exception as e:
            print(f"Error in calculate_rmse: {e}")
            return np.nan
    
    @staticmethod
    def calculate_mape(actual, predicted):
        """Calculate Mean Absolute Percentage Error"""
        try:
            actual_np = np.array(actual)
            predicted_np = np.array(predicted)
            
            # Filter out NaN and zero values
            mask = ~(np.isnan(actual_np) | np.isnan(predicted_np) | (actual_np == 0))
            actual_clean = actual_np[mask]
            predicted_clean = predicted_np[mask]
            
            if len(actual_clean) == 0:
                return np.nan
            
            return np.mean(np.abs((actual_clean - predicted_clean) / actual_clean)) * 100
        except Exception as e:
            print(f"Error in calculate_mape: {e}")
            return np.nan
    
    @staticmethod
    def calculate_directional_accuracy(actual, predicted, threshold_percent=1.0):
        """Calculate directional accuracy with threshold"""
        try:
            actual_np = np.array(actual)
            predicted_np = np.array(predicted)
            
            # Filter out NaN values
            mask = ~(np.isnan(actual_np) | np.isnan(predicted_np))
            actual_clean = actual_np[mask]
            predicted_clean = predicted_np[mask]
            
            if len(actual_clean) < 2:
                return np.nan
            
            # Calculate percentage changes
            actual_changes = np.diff(actual_clean) / actual_clean[:-1] * 100
            predicted_changes = np.diff(predicted_clean) / predicted_clean[:-1] * 100
            
            # Categorize directions with threshold
            def categorize_direction(changes, threshold=threshold_percent):
                directions = np.zeros(len(changes), dtype=int)
                directions[changes > threshold] = 1   # Up
                directions[changes < -threshold] = -1  # Down
                return directions
            
            actual_direction = categorize_direction(actual_changes)
            predicted_direction = categorize_direction(predicted_changes)
            
            # Calculate accuracy
            correct_predictions = np.sum(actual_direction == predicted_direction)
            total_predictions = len(actual_direction)
            
            return (correct_predictions / total_predictions) * 100
        except Exception as e:
            print(f"Error in calculate_directional_accuracy: {e}")
            return np.nan
    
    @staticmethod
    def calculate_time_based_metrics(actual, predicted, dates, periods=[3, 7, 14, 30]):
        """Calculate metrics for different time periods"""
        metrics = {}
        
        # Filter out NaN values first
        mask = ~(np.isnan(actual) | np.isnan(predicted))
        actual_clean = actual[mask]
        predicted_clean = predicted[mask]
        dates_clean = dates[mask]
        
        if len(actual_clean) == 0:
            return metrics
        
        for period in periods:
            if len(actual_clean) >= period:
                # Take first N days
                actual_period = actual_clean[:period]
                predicted_period = predicted_clean[:period]
                
                # Calculate metrics
                rmse = MetricsCalculator.calculate_rmse(actual_period, predicted_period)
                mape = MetricsCalculator.calculate_mape(actual_period, predicted_period)
                da = MetricsCalculator.calculate_directional_accuracy(actual_period, predicted_period)
                
                metrics[f"{period}d"] = {
                    'rmse': rmse,
                    'mape': mape,
                    'directional_accuracy': da,
                    'count': period
                }
        
        return metrics
    
    @staticmethod
    def evaluate_model_performance(predictions_df, model_name="Model"):
        """Comprehensive model evaluation"""
        # Ensure we have required columns
        required_cols = ['Actual_Price', 'Final_Prediction']
        if not all(col in predictions_df.columns for col in required_cols):
            raise ValueError(f"DataFrame must contain columns: {required_cols}")
        
        # Filter valid data
        valid_data = predictions_df.dropna(subset=required_cols)
        
        if len(valid_data) == 0:
            print(f"No valid data for {model_name} evaluation")
            return {}
        
        actual = valid_data['Actual_Price'].values
        predicted = valid_data['Final_Prediction'].values
        
        # Calculate metrics
        rmse = MetricsCalculator.calculate_rmse(actual, predicted)
        mape = MetricsCalculator.calculate_mape(actual, predicted)
        da = MetricsCalculator.calculate_directional_accuracy(actual, predicted)
        
        # Time-based metrics if dates available
        time_metrics = {}
        if 'Date' in valid_data.columns:
            dates = pd.to_datetime(valid_data['Date']).values
            time_metrics = MetricsCalculator.calculate_time_based_metrics(
                actual, predicted, dates
            )
        
        results = {
            'model_name': model_name,
            'total_predictions': len(valid_data),
            'rmse': rmse,
            'mape': mape,
            'directional_accuracy': da,
            'time_based_metrics': time_metrics
        }
        
        return results
    
    @staticmethod
    def print_evaluation_results(results):
        """Print formatted evaluation results"""
        print("="*60)
        print(f"MODEL EVALUATION: {results['model_name']}")
        print("="*60)
        print(f"Total Predictions: {results['total_predictions']}")
        print(f"RMSE: ${results['rmse']:.2f}")
        print(f"MAPE: {results['mape']:.2f}%")
        print(f"Directional Accuracy: {results['directional_accuracy']:.1f}%")
        
        if results['time_based_metrics']:
            print("\nTime-based Performance:")
            for period, metrics in results['time_based_metrics'].items():
                print(f"  {period}: RMSE=${metrics['rmse']:.2f}, "
                      f"DA={metrics['directional_accuracy']:.1f}%")
        
        print("="*60)