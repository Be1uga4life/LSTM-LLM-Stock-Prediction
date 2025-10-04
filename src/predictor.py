"""
Main prediction orchestrator that combines LSTM and LLM components
This is the main interface for running predictions
"""
import pandas as pd
from datetime import datetime, timedelta
import os

from .data_handler import DataHandler
from .lstm_model import LSTMPredictor
from .llm_enhancer import LLMEnhancer
from .metrics import MetricsCalculator
from .visualizer import PredictionVisualizer


class StockPredictor:
    """Main predictor class that orchestrates LSTM and LLM components"""
    
    def __init__(self, symbol, model_dir='models', output_dir='output'):
        self.symbol = symbol
        self.model_dir = model_dir
        self.output_dir = output_dir
        
        # Initialize components
        self.data_handler = DataHandler()
        self.lstm_predictor = LSTMPredictor()
        self.llm_enhancer = LLMEnhancer()
        self.metrics_calculator = MetricsCalculator()
        self.visualizer = PredictionVisualizer()
        
        # Create output directory
        self.symbol_output_dir = os.path.join(output_dir, symbol)
        os.makedirs(self.symbol_output_dir, exist_ok=True)
    
    def train_model(self, start_date, end_date, save_model=True):
        """Train LSTM model on historical data"""
        print(f"Training model for {self.symbol}...")
        
        # Fetch training data
        stock_data = self.data_handler.fetch_stock_data(
            self.symbol, start_date, end_date
        )
        
        # Train LSTM model
        self.lstm_predictor.train(stock_data)
        
        # Save model if requested
        if save_model:
            self.lstm_predictor.save_model(self.model_dir, self.symbol)
        
        print(f"Model training completed for {self.symbol}")
        return stock_data
    
    def load_existing_model(self):
        """Load pre-trained model"""
        try:
            self.lstm_predictor.load_model(self.model_dir, self.symbol)
            print(f"Loaded existing model for {self.symbol}")
            return True
        except FileNotFoundError:
            print(f"No existing model found for {self.symbol}")
            return False
    
    def generate_predictions(self, training_data, news_file_path, days_to_predict=30, 
                           use_llm=True, actual_data_file=None):
        """Generate predictions using LSTM and optionally enhance with LLM"""
        print(f"Generating predictions for {self.symbol}...")
        
        # Generate LSTM predictions
        lstm_predictions = self.lstm_predictor.generate_predictions(
            training_data, days_to_predict
        )
        
        # Load news data
        news_data = self.data_handler.load_news_data(news_file_path, self.symbol)
        
        # Generate prediction dates
        last_date = training_data.index[-1]
        prediction_dates = [
            last_date + timedelta(days=i+1) for i in range(days_to_predict)
        ]
        
        # Enhance with LLM if requested
        enhanced_predictions = lstm_predictions.copy()
        adjustments_log = []
        
        if use_llm:
            print("Enhancing predictions with LLM...")
            enhanced_predictions, adjustments_log = self.llm_enhancer.enhance_predictions(
                self.symbol, lstm_predictions, news_data, last_date + timedelta(days=1)
            )
        
        # Load actual prices if available
        actual_prices = [None] * days_to_predict
        if actual_data_file and os.path.exists(actual_data_file):
            try:
                actual_data = pd.read_csv(actual_data_file, parse_dates=['Date'])
                actual_data.set_index('Date', inplace=True)
                
                for i, pred_date in enumerate(prediction_dates):
                    if pred_date in actual_data.index:
                        actual_prices[i] = actual_data.loc[pred_date, 'Close']
            except Exception as e:
                print(f"Error loading actual prices: {e}")
        
        # Prepare results
        results = {
            'dates': prediction_dates,
            'lstm_predictions': lstm_predictions,
            'enhanced_predictions': enhanced_predictions,
            'actual_prices': actual_prices,
            'adjustments_log': adjustments_log,
            'news_data': news_data
        }
        
        return results
    
    def save_results(self, results):
        """Save prediction results to CSV"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Prepare DataFrame
        df_results = pd.DataFrame({
            'Date': results['dates'],
            'LSTM_Prediction': results['lstm_predictions'],
            'Final_Prediction': results['enhanced_predictions'],
            'Actual_Price': results['actual_prices']
        })
        
        # Save predictions
        predictions_file = os.path.join(
            self.symbol_output_dir, f"{self.symbol}_predictions_{timestamp}.csv"
        )
        self.data_handler.save_predictions(df_results.to_dict('records'), predictions_file)
        
        # Save LSTM-only predictions for comparison
        df_lstm = df_results[['Date', 'LSTM_Prediction', 'Actual_Price']].copy()
        df_lstm.rename(columns={'LSTM_Prediction': 'Final_Prediction'}, inplace=True)
        
        lstm_file = os.path.join(
            self.symbol_output_dir, f"{self.symbol}_lstm_only_{timestamp}.csv"
        )
        self.data_handler.save_predictions(df_lstm.to_dict('records'), lstm_file)
        
        return predictions_file, lstm_file
    
    def evaluate_performance(self, predictions_file, lstm_file):
        """Evaluate model performance"""
        # Load prediction files
        df_enhanced = pd.read_csv(predictions_file, parse_dates=['Date'])
        df_lstm = pd.read_csv(lstm_file, parse_dates=['Date'])
        
        # Calculate metrics
        enhanced_results = self.metrics_calculator.evaluate_model_performance(
            df_enhanced, "Enhanced (LSTM + LLM)"
        )
        lstm_results = self.metrics_calculator.evaluate_model_performance(
            df_lstm, "LSTM Only"
        )
        
        # Print results
        self.metrics_calculator.print_evaluation_results(enhanced_results)
        self.metrics_calculator.print_evaluation_results(lstm_results)
        
        return enhanced_results, lstm_results
    
    def create_visualizations(self, predictions_file, historical_data):
        """Create visualization plots"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Load predictions
        df_predictions = pd.read_csv(predictions_file, parse_dates=['Date'])
        
        # Create comparison plot
        comparison_plot_path = os.path.join(
            self.symbol_output_dir, f"{self.symbol}_comparison_{timestamp}.png"
        )
        
        self.visualizer.create_comparison_plot(
            df_predictions, historical_data, self.symbol, comparison_plot_path
        )
        
        # Create accuracy plot if actual prices available
        if df_predictions['Actual_Price'].notna().any():
            accuracy_plot_path = os.path.join(
                self.symbol_output_dir, f"{self.symbol}_accuracy_{timestamp}.png"
            )
            self.visualizer.create_prediction_accuracy_plot(
                df_predictions, accuracy_plot_path
            )
        
        return comparison_plot_path
    
    def run_full_pipeline(self, train_start, train_end, news_file_path, 
                         days_to_predict=30, use_llm=True, actual_data_file=None):
        """Run the complete prediction pipeline"""
        print(f"Starting full prediction pipeline for {self.symbol}")
        print("="*60)
        
        # Step 1: Train or load model
        if not self.load_existing_model():
            training_data = self.train_model(train_start, train_end)
        else:
            # Still need training data for predictions
            training_data = self.data_handler.fetch_stock_data(
                self.symbol, train_start, train_end
            )
        
        # Step 2: Generate predictions
        results = self.generate_predictions(
            training_data, news_file_path, days_to_predict, use_llm, actual_data_file
        )
        
        # Step 3: Save results
        predictions_file, lstm_file = self.save_results(results)
        
        # Step 4: Evaluate performance
        enhanced_results, lstm_results = self.evaluate_performance(
            predictions_file, lstm_file
        )
        
        # Step 5: Create visualizations
        plot_path = self.create_visualizations(predictions_file, training_data)
        
        print("="*60)
        print("Pipeline completed successfully!")
        print(f"Predictions saved to: {predictions_file}")
        print(f"Visualization saved to: {plot_path}")
        
        return {
            'predictions_file': predictions_file,
            'lstm_file': lstm_file,
            'enhanced_results': enhanced_results,
            'lstm_results': lstm_results,
            'plot_path': plot_path
        }