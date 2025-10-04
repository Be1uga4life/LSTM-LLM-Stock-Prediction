"""
Complete LSTM + LLM prediction with accuracy testing
Uses CPU, applies LLM enhancements, then tests against actual data
"""
import os
import sys
import warnings
warnings.filterwarnings('ignore')

# Force CPU usage to avoid GPU issues
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# Add src to path
sys.path.append('src')

import numpy as np
import pandas as pd
import json
from datetime import datetime, timedelta
import matplotlib.pyplot as plt

# Import our modules
from src.lstm_model import LSTMPredictor
from src.llm_enhancer import LLMEnhancer
from src.metrics import MetricsCalculator
from src.data_handler import DataHandler

def load_actual_data_for_testing(actual_data_file, prediction_dates):
    """Load actual prices for the prediction period to test accuracy"""
    try:
        actual_data = pd.read_csv(actual_data_file, parse_dates=['Date'])
        actual_data.set_index('Date', inplace=True)
        
        actual_prices = []
        for pred_date in prediction_dates:
            if pred_date in actual_data.index:
                actual_prices.append(actual_data.loc[pred_date, 'Close'])
            else:
                actual_prices.append(None)
        
        print(f"✅ Loaded actual prices for {sum(1 for p in actual_prices if p is not None)} prediction days")
        return actual_prices
    except Exception as e:
        print(f"❌ Error loading actual data: {e}")
        return [None] * len(prediction_dates)

def create_comparison_plot(historical_data, prediction_dates, lstm_predictions, 
                          enhanced_predictions, actual_prices, symbol):
    """Create comprehensive comparison plot"""
    plt.figure(figsize=(15, 10))
    
    # Plot last 60 days of historical data
    hist_data = historical_data.tail(60)
    plt.plot(hist_data.index, hist_data['Close'], 
             label='Historical Price', color='black', linewidth=2)
    
    # Plot predictions
    plt.plot(prediction_dates, lstm_predictions, 
             label='LSTM Predictions', color='red', linewidth=2, linestyle='--', alpha=0.7)
    
    plt.plot(prediction_dates, enhanced_predictions, 
             label='Enhanced (LSTM + LLM)', color='blue', linewidth=3)
    
    # Plot actual prices if available
    actual_dates = []
    actual_values = []
    for i, (date, price) in enumerate(zip(prediction_dates, actual_prices)):
        if price is not None:
            actual_dates.append(date)
            actual_values.append(price)
    
    if actual_values:
        plt.plot(actual_dates, actual_values, 
                 label='Actual Prices', color='green', linewidth=2, marker='o', markersize=4)
    
    # Add vertical line at prediction start
    plt.axvline(x=prediction_dates[0], color='red', linestyle=':', alpha=0.5, 
                label='Prediction Start')
    
    plt.title(f'{symbol} Stock Price: LSTM + LLM Enhanced Predictions vs Actual')
    plt.xlabel('Date')
    plt.ylabel('Price ($)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    # Save plot
    os.makedirs('output', exist_ok=True)
    plot_file = f'output/{symbol}_lstm_llm_comparison.png'
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    return plot_file

def main():
    """Main prediction pipeline with LLM enhancement and accuracy testing"""
    print("🚀 LSTM + LLM Enhanced Stock Prediction with Accuracy Testing")
    print("=" * 70)
    
    # Configuration
    symbol = "TSLA"
    stock_file = "TSLA_historical.csv"
    news_file = f"news/{symbol}_news_latest.json"
    actual_data_file = "TSLA_historical.csv"  # Same file but we'll use future dates
    days_to_predict = 30
    
    # Check files exist
    if not os.path.exists(stock_file):
        print(f"❌ Stock data file not found: {stock_file}")
        return 1
    
    # Download news if not available
    if not os.path.exists(news_file):
        print(f"📰 News file not found, downloading news for {symbol}...")
        try:
            from download_news import NewsDownloader
            downloader = NewsDownloader()
            news_data_raw = downloader.download_news(symbol, days_back=90)
            if news_data_raw:
                news_file = downloader.save_news(news_data_raw, symbol)
                print(f"✅ News downloaded and saved to: {news_file}")
            else:
                print("❌ Failed to download news data")
                return 1
        except Exception as e:
            print(f"❌ Error downloading news: {e}")
            return 1
    
    # Load data
    print("📊 Loading data...")
    data_handler = DataHandler()
    
    # Load stock data
    stock_data = pd.read_csv(stock_file, parse_dates=['Date'])
    stock_data.set_index('Date', inplace=True)
    print(f"✅ Loaded {len(stock_data)} days of stock data")
    
    # Load news data
    news_data = data_handler.load_news_data(news_file, symbol)
    
    # Split data for training (use data up to a certain point for realistic testing)
    # Use data up to 30 days before the end for training, test on last 30 days
    split_date = stock_data.index[-days_to_predict]
    training_data = stock_data[stock_data.index < split_date]
    
    print(f"📚 Training data: {len(training_data)} days (up to {split_date.strftime('%Y-%m-%d')})")
    print(f"🎯 Testing period: {days_to_predict} days (from {split_date.strftime('%Y-%m-%d')})")
    
    # Step 1: Train LSTM Model
    print("\n🧠 Step 1: Training LSTM Model (CPU-only)...")
    lstm_predictor = LSTMPredictor(time_steps=60)
    
    try:
        lstm_predictor.train(training_data, epochs=30, batch_size=32)
        print("✅ LSTM training completed")
    except Exception as e:
        print(f"❌ LSTM training failed: {e}")
        return 1
    
    # Step 2: Generate LSTM Predictions
    print("\n📈 Step 2: Generating LSTM predictions...")
    try:
        lstm_predictions = lstm_predictor.generate_predictions(training_data, days_to_predict)
        print(f"✅ Generated {len(lstm_predictions)} LSTM predictions")
    except Exception as e:
        print(f"❌ LSTM prediction failed: {e}")
        return 1
    
    # Step 3: Apply LLM Enhancements
    print("\n🤖 Step 3: Applying LLM enhancements...")
    llm_enhancer = LLMEnhancer()
    
    # Generate prediction dates
    last_training_date = training_data.index[-1]
    prediction_dates = [last_training_date + timedelta(days=i+1) for i in range(days_to_predict)]
    
    enhanced_predictions = []
    adjustments_log = []
    
    for i, (lstm_pred, pred_date) in enumerate(zip(lstm_predictions, prediction_dates)):
        try:
            enhanced_pred, reasoning = llm_enhancer.get_adjustment(
                symbol, lstm_pred, news_data, pred_date
            )
            enhanced_predictions.append(enhanced_pred)
            
            adjustment_factor = (enhanced_pred - lstm_pred) / lstm_pred * 100
            adjustments_log.append({
                'date': pred_date,
                'lstm_prediction': lstm_pred,
                'enhanced_prediction': enhanced_pred,
                'adjustment_percent': adjustment_factor,
                'reasoning': reasoning[:100] + "..." if len(reasoning) > 100 else reasoning
            })
            
            print(f"  Day {i+1}: ${lstm_pred:.2f} → ${enhanced_pred:.2f} ({adjustment_factor:+.1f}%)")
            
        except Exception as e:
            print(f"  Day {i+1}: LLM failed, using LSTM prediction: {e}")
            enhanced_predictions.append(lstm_pred)
            adjustments_log.append({
                'date': pred_date,
                'lstm_prediction': lstm_pred,
                'enhanced_prediction': lstm_pred,
                'adjustment_percent': 0.0,
                'reasoning': f"LLM failed: {e}"
            })
    
    print(f"✅ Applied LLM enhancements to {len(enhanced_predictions)} predictions")
    
    # Step 4: Load Actual Data for Accuracy Testing
    print("\n🎯 Step 4: Loading actual data for accuracy testing...")
    actual_prices = load_actual_data_for_testing(actual_data_file, prediction_dates)
    
    # Step 5: Calculate Accuracy Metrics
    print("\n📊 Step 5: Calculating accuracy metrics...")
    metrics_calc = MetricsCalculator()
    
    # Filter out None values for metrics calculation
    valid_indices = [i for i, price in enumerate(actual_prices) if price is not None]
    
    if valid_indices:
        valid_actual = [actual_prices[i] for i in valid_indices]
        valid_lstm = [lstm_predictions[i] for i in valid_indices]
        valid_enhanced = [enhanced_predictions[i] for i in valid_indices]
        
        # LSTM-only metrics
        lstm_rmse = metrics_calc.calculate_rmse(valid_actual, valid_lstm)
        lstm_mape = metrics_calc.calculate_mape(valid_actual, valid_lstm)
        lstm_da = metrics_calc.calculate_directional_accuracy(valid_actual, valid_lstm)
        
        # Enhanced model metrics
        enhanced_rmse = metrics_calc.calculate_rmse(valid_actual, valid_enhanced)
        enhanced_mape = metrics_calc.calculate_mape(valid_actual, valid_enhanced)
        enhanced_da = metrics_calc.calculate_directional_accuracy(valid_actual, valid_enhanced)
        
        print(f"📈 LSTM-Only Performance:")
        print(f"   RMSE: ${lstm_rmse:.2f}")
        print(f"   MAPE: {lstm_mape:.2f}%")
        print(f"   Directional Accuracy: {lstm_da:.1f}%")
        
        print(f"🚀 Enhanced (LSTM + LLM) Performance:")
        print(f"   RMSE: ${enhanced_rmse:.2f}")
        print(f"   MAPE: {enhanced_mape:.2f}%")
        print(f"   Directional Accuracy: {enhanced_da:.1f}%")
        
        # Improvement analysis
        rmse_improvement = ((lstm_rmse - enhanced_rmse) / lstm_rmse) * 100
        da_improvement = enhanced_da - lstm_da
        
        print(f"\n🏆 Improvement Analysis:")
        print(f"   RMSE: {rmse_improvement:+.1f}% {'(Better)' if rmse_improvement > 0 else '(Worse)'}")
        print(f"   Directional Accuracy: {da_improvement:+.1f} percentage points")
        
    else:
        print("⚠️ No actual data available for accuracy testing")
    
    # Step 6: Save Results and Create Visualizations
    print("\n💾 Step 6: Saving results...")
    
    # Save predictions
    results_df = pd.DataFrame({
        'Date': prediction_dates,
        'LSTM_Prediction': lstm_predictions,
        'Enhanced_Prediction': enhanced_predictions,
        'Actual_Price': actual_prices,
        'LLM_Adjustment_Percent': [log['adjustment_percent'] for log in adjustments_log]
    })
    
    results_file = f'output/{symbol}_lstm_llm_predictions.csv'
    results_df.to_csv(results_file, index=False)
    print(f"📄 Predictions saved to: {results_file}")
    
    # Create comparison plot
    plot_file = create_comparison_plot(
        training_data, prediction_dates, lstm_predictions, 
        enhanced_predictions, actual_prices, symbol
    )
    print(f"📊 Comparison plot saved to: {plot_file}")
    
    # Save detailed log
    log_df = pd.DataFrame(adjustments_log)
    log_file = f'output/{symbol}_llm_adjustments_log.csv'
    log_df.to_csv(log_file, index=False)
    print(f"📝 LLM adjustments log saved to: {log_file}")
    
    print("\n" + "=" * 70)
    print("🎉 PREDICTION PIPELINE COMPLETED SUCCESSFULLY!")
    print("=" * 70)
    print(f"Symbol: {symbol}")
    print(f"Method: LSTM + LLM Enhancement")
    print(f"Predictions: {days_to_predict} days")
    print(f"Results: {results_file}")
    print(f"Visualization: {plot_file}")
    print(f"Adjustments Log: {log_file}")
    
    return 0

if __name__ == "__main__":
    exit(main())