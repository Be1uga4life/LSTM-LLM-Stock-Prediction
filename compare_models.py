"""
Clean comparison script that replaces stock_comparison.py
Provides easy model comparison with better organization
"""
import argparse
import pandas as pd
import sys
import os

# Add src to path
sys.path.append('src')

from src.metrics import MetricsCalculator
from src.visualizer import PredictionVisualizer


def load_and_validate_data(enhanced_file, lstm_file, historical_file):
    """Load and validate all required data files"""
    try:
        # Load prediction files
        df_enhanced = pd.read_csv(enhanced_file, parse_dates=['Date'])
        df_lstm = pd.read_csv(lstm_file, parse_dates=['Date'])
        
        # Load historical data
        df_historical = pd.read_csv(historical_file, parse_dates=['Date'])
        
        # Validate columns
        required_pred_cols = ['Date', 'Final_Prediction']
        if not all(col in df_enhanced.columns for col in required_pred_cols):
            raise ValueError(f"Enhanced file missing columns: {required_pred_cols}")
        
        if not all(col in df_lstm.columns for col in required_pred_cols):
            raise ValueError(f"LSTM file missing columns: {required_pred_cols}")
        
        # Check for Close or Actual_Price in historical data
        if 'Close' in df_historical.columns:
            df_historical['Actual_Price'] = df_historical['Close']
        elif 'Actual_Price' not in df_historical.columns:
            raise ValueError("Historical data must contain 'Close' or 'Actual_Price' column")
        
        return df_enhanced, df_lstm, df_historical
        
    except Exception as e:
        print(f"Error loading data: {e}")
        raise


def compare_models(enhanced_file, lstm_file, historical_file, symbol=None, save_plots=True):
    """Compare enhanced vs LSTM-only models"""
    
    # Load data
    df_enhanced, df_lstm, df_historical = load_and_validate_data(
        enhanced_file, lstm_file, historical_file
    )
    
    # Extract symbol from filename if not provided
    if symbol is None:
        symbol = os.path.basename(enhanced_file).split('_')[0]
    
    print(f"Comparing models for {symbol}")
    print(f"Enhanced predictions: {len(df_enhanced)}")
    print(f"LSTM predictions: {len(df_lstm)}")
    print(f"Historical data points: {len(df_historical)}")
    print()
    
    # Calculate metrics for both models
    metrics_calc = MetricsCalculator()
    
    # Enhanced model metrics
    enhanced_results = metrics_calc.evaluate_model_performance(
        df_enhanced, "Enhanced (LSTM + LLM)"
    )
    
    # LSTM model metrics (need to align with enhanced data for fair comparison)
    merged_df = pd.merge(
        df_enhanced[['Date', 'Actual_Price']], 
        df_lstm[['Date', 'Final_Prediction']], 
        on='Date', how='inner'
    )
    
    if len(merged_df) > 0:
        lstm_results = {
            'model_name': 'LSTM Only',
            'total_predictions': len(merged_df),
            'rmse': metrics_calc.calculate_rmse(
                merged_df['Actual_Price'], merged_df['Final_Prediction']
            ),
            'mape': metrics_calc.calculate_mape(
                merged_df['Actual_Price'], merged_df['Final_Prediction']
            ),
            'directional_accuracy': metrics_calc.calculate_directional_accuracy(
                merged_df['Actual_Price'], merged_df['Final_Prediction']
            ),
            'time_based_metrics': {}
        }
    else:
        print("Warning: No overlapping dates between enhanced and LSTM predictions")
        lstm_results = enhanced_results.copy()
        lstm_results['model_name'] = 'LSTM Only'
    
    # Print comparison results
    print_comparison_results(enhanced_results, lstm_results)
    
    # Create visualizations
    if save_plots:
        visualizer = PredictionVisualizer()
        
        # Main comparison plot
        comparison_path = f"{symbol}_model_comparison.png"
        visualizer.create_comparison_plot(
            df_enhanced, df_historical, symbol, comparison_path
        )
        
        # Metrics comparison plot
        metrics_path = f"{symbol}_metrics_comparison.png"
        visualizer.create_metrics_comparison_plot(
            lstm_results, enhanced_results, metrics_path
        )
        
        print(f"Plots saved:")
        print(f"  - {comparison_path}")
        print(f"  - {metrics_path}")
    
    return enhanced_results, lstm_results


def print_comparison_results(enhanced_results, lstm_results):
    """Print formatted comparison results"""
    print("="*80)
    print("MODEL PERFORMANCE COMPARISON")
    print("="*80)
    
    # Print individual results
    MetricsCalculator.print_evaluation_results(enhanced_results)
    MetricsCalculator.print_evaluation_results(lstm_results)
    
    # Print winner analysis
    print("COMPARISON SUMMARY")
    print("="*80)
    
    enhanced_rmse = enhanced_results.get('rmse', float('inf'))
    lstm_rmse = lstm_results.get('rmse', float('inf'))
    enhanced_da = enhanced_results.get('directional_accuracy', 0)
    lstm_da = lstm_results.get('directional_accuracy', 0)
    
    if enhanced_rmse < lstm_rmse:
        rmse_winner = "Enhanced Model"
        rmse_improvement = ((lstm_rmse - enhanced_rmse) / lstm_rmse) * 100
        print(f"🏆 RMSE Winner: {rmse_winner} (${enhanced_rmse:.2f} vs ${lstm_rmse:.2f})")
        print(f"   Improvement: {rmse_improvement:.1f}% better")
    else:
        rmse_winner = "LSTM Only"
        rmse_improvement = ((enhanced_rmse - lstm_rmse) / enhanced_rmse) * 100
        print(f"🏆 RMSE Winner: {rmse_winner} (${lstm_rmse:.2f} vs ${enhanced_rmse:.2f})")
        print(f"   Improvement: {rmse_improvement:.1f}% better")
    
    if enhanced_da > lstm_da:
        da_winner = "Enhanced Model"
        da_improvement = enhanced_da - lstm_da
        print(f"📈 Directional Accuracy Winner: {da_winner} ({enhanced_da:.1f}% vs {lstm_da:.1f}%)")
        print(f"   Improvement: +{da_improvement:.1f} percentage points")
    else:
        da_winner = "LSTM Only"
        da_improvement = lstm_da - enhanced_da
        print(f"📈 Directional Accuracy Winner: {da_winner} ({lstm_da:.1f}% vs {enhanced_da:.1f}%)")
        print(f"   Improvement: +{da_improvement:.1f} percentage points")
    
    print("="*80)


def main():
    parser = argparse.ArgumentParser(description='Compare Enhanced vs LSTM-only predictions')
    
    parser.add_argument('enhanced_file', help='Enhanced predictions CSV file')
    parser.add_argument('lstm_file', help='LSTM-only predictions CSV file')
    parser.add_argument('historical_file', help='Historical stock data CSV file')
    parser.add_argument('--symbol', help='Stock symbol (auto-detected if not provided)')
    parser.add_argument('--no-plots', action='store_true', help='Skip plot generation')
    
    args = parser.parse_args()
    
    # Validate files exist
    for file_path in [args.enhanced_file, args.lstm_file, args.historical_file]:
        if not os.path.exists(file_path):
            print(f"Error: File not found: {file_path}")
            return 1
    
    try:
        enhanced_results, lstm_results = compare_models(
            args.enhanced_file,
            args.lstm_file, 
            args.historical_file,
            args.symbol,
            save_plots=not args.no_plots
        )
        return 0
        
    except Exception as e:
        print(f"Error during comparison: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())