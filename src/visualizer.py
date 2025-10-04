"""
Visualization utilities for stock prediction results
Handles all plotting and chart generation
"""
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd
import numpy as np
from datetime import timedelta
import os


class PredictionVisualizer:
    """Visualization tools for prediction results"""
    
    @staticmethod
    def create_comparison_plot(df_predictions, df_historical, symbol, save_path=None):
        """Create comprehensive comparison plot"""
        fig, ax = plt.subplots(figsize=(15, 10))
        
        # Find prediction start date
        prediction_start_date = df_predictions['Date'].min()
        
        # Limit historical data to 30 days before predictions
        historical_cutoff_date = prediction_start_date - timedelta(days=30)
        df_historical_limited = df_historical[
            (df_historical['Date'] >= historical_cutoff_date) & 
            (df_historical['Date'] <= prediction_start_date)
        ]
        
        # Plot historical data
        if not df_historical_limited.empty:
            ax.plot(df_historical_limited['Date'], df_historical_limited['Close'], 
                   color='black', linewidth=3, label='Historical Price', linestyle='-')
        
        # Plot actual prices during prediction period
        actual_mask = df_predictions['Actual_Price'].notna()
        if actual_mask.any():
            actual_data = df_predictions[actual_mask]
            ax.plot(actual_data['Date'], actual_data['Actual_Price'],
                   color='green', linewidth=2, label='Actual Price', 
                   linestyle='-', marker='o', markersize=4)
        
        # Plot LSTM predictions
        if 'LSTM_Prediction' in df_predictions.columns:
            ax.plot(df_predictions['Date'], df_predictions['LSTM_Prediction'],
                   color='red', linewidth=2, label='LSTM Prediction', 
                   linestyle='--', alpha=0.7)
        
        # Plot enhanced predictions
        if 'Final_Prediction' in df_predictions.columns:
            ax.plot(df_predictions['Date'], df_predictions['Final_Prediction'],
                   color='blue', linewidth=3, label='Enhanced Prediction', 
                   linestyle='-')
        
        # Add vertical line at prediction start
        ax.axvline(x=prediction_start_date, color='red', linestyle=':', 
                  alpha=0.5, label='Prediction Start')
        
        # Customize plot
        ax.set_title(f'{symbol} Stock Price: Prediction Performance Comparison', 
                    fontsize=16, fontweight='bold')
        ax.set_xlabel('Date', fontsize=12)
        ax.set_ylabel('Price ($)', fontsize=12)
        ax.legend(loc='upper left')
        ax.grid(True, alpha=0.3)
        
        # Format dates
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        ax.xaxis.set_major_locator(mdates.WeekdayLocator(interval=2))
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
        
        fig.tight_layout()
        
        # Save if path provided
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Plot saved to {save_path}")
        
        return fig
    
    @staticmethod
    def create_metrics_comparison_plot(lstm_results, enhanced_results, save_path=None):
        """Create bar chart comparing model metrics"""
        metrics = ['rmse', 'mape', 'directional_accuracy']
        lstm_values = [lstm_results.get(m, 0) for m in metrics]
        enhanced_values = [enhanced_results.get(m, 0) for m in metrics]
        
        x = np.arange(len(metrics))
        width = 0.35
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        bars1 = ax.bar(x - width/2, lstm_values, width, label='LSTM Only', alpha=0.8)
        bars2 = ax.bar(x + width/2, enhanced_values, width, label='LSTM + LLM', alpha=0.8)
        
        ax.set_xlabel('Metrics')
        ax.set_ylabel('Values')
        ax.set_title('Model Performance Comparison')
        ax.set_xticks(x)
        ax.set_xticklabels(['RMSE ($)', 'MAPE (%)', 'Directional Accuracy (%)'])
        ax.legend()
        
        # Add value labels on bars
        def add_value_labels(bars):
            for bar in bars:
                height = bar.get_height()
                ax.annotate(f'{height:.2f}',
                           xy=(bar.get_x() + bar.get_width() / 2, height),
                           xytext=(0, 3),
                           textcoords="offset points",
                           ha='center', va='bottom')
        
        add_value_labels(bars1)
        add_value_labels(bars2)
        
        fig.tight_layout()
        
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Metrics comparison saved to {save_path}")
        
        return fig
    
    @staticmethod
    def create_prediction_accuracy_plot(df_predictions, save_path=None):
        """Create plot showing prediction accuracy over time"""
        # Calculate daily accuracy
        df_predictions = df_predictions.copy()
        df_predictions['Prediction_Error'] = abs(
            df_predictions['Final_Prediction'] - df_predictions['Actual_Price']
        )
        df_predictions['Accuracy_Percent'] = (
            1 - df_predictions['Prediction_Error'] / df_predictions['Actual_Price']
        ) * 100
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
        
        # Plot 1: Prediction vs Actual
        valid_data = df_predictions.dropna(subset=['Actual_Price'])
        ax1.plot(valid_data['Date'], valid_data['Actual_Price'], 
                'o-', label='Actual Price', color='green')
        ax1.plot(valid_data['Date'], valid_data['Final_Prediction'], 
                's-', label='Predicted Price', color='blue', alpha=0.7)
        ax1.set_ylabel('Price ($)')
        ax1.set_title('Prediction vs Actual Prices')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Accuracy over time
        ax2.plot(valid_data['Date'], valid_data['Accuracy_Percent'], 
                'o-', color='red', alpha=0.7)
        ax2.axhline(y=95, color='green', linestyle='--', alpha=0.5, label='95% Accuracy')
        ax2.axhline(y=90, color='orange', linestyle='--', alpha=0.5, label='90% Accuracy')
        ax2.set_ylabel('Accuracy (%)')
        ax2.set_xlabel('Date')
        ax2.set_title('Prediction Accuracy Over Time')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Format dates
        for ax in [ax1, ax2]:
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
        
        fig.tight_layout()
        
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Accuracy plot saved to {save_path}")
        
        return fig