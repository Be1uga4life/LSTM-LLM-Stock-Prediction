"""
LSTM model implementation for stock price prediction
Handles model creation, training, and prediction generation
"""
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import pickle
import os
import shutil

# Configure TensorFlow to handle GPU issues gracefully
import tensorflow as tf
try:
    # Disable GPU if there are CUDA issues
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            # Enable memory growth to avoid allocation issues
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as e:
            print(f"GPU configuration warning: {e}")
            # Force CPU usage if GPU setup fails
            tf.config.set_visible_devices([], 'GPU')
            print("Falling back to CPU due to GPU issues")
except Exception as e:
    print(f"TensorFlow GPU setup warning: {e}")
    print("Using CPU for training")

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense


class LSTMPredictor:
    """LSTM model for stock price prediction"""
    
    def __init__(self, time_steps=60):
        self.time_steps = time_steps
        self.model = None
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.is_trained = False
    
    def prepare_data(self, stock_data):
        """Prepare stock data for LSTM training"""
        close_prices = np.array(stock_data['Close'].values).reshape(-1, 1)
        scaled_data = self.scaler.fit_transform(close_prices)
        return close_prices, scaled_data
    
    def create_sequences(self, scaled_data):
        """Create sequences for LSTM training"""
        x, y = [], []
        for i in range(self.time_steps, len(scaled_data)):
            x.append(scaled_data[i-self.time_steps:i, 0])
            y.append(scaled_data[i, 0])
        return np.array(x), np.array(y)
    
    def build_model(self, input_shape):
        """Build LSTM model architecture"""
        self.model = Sequential([
            LSTM(units=50, return_sequences=True, input_shape=input_shape),
            LSTM(units=50),
            Dense(units=1)
        ])
        self.model.compile(optimizer='adam', loss='mean_squared_error')
        return self.model
    
    def train(self, stock_data, epochs=50, batch_size=32):
        """Train the LSTM model"""
        print("Training LSTM model...")
        
        # Prepare data
        close_prices, scaled_data = self.prepare_data(stock_data)
        x_train, y_train = self.create_sequences(scaled_data)
        
        # Reshape for LSTM
        x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
        
        # Build and train model
        self.build_model((x_train.shape[1], 1))
        self.model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size, verbose=1)
        
        self.is_trained = True
        print("LSTM model training completed")
        return self.model
    
    def predict_next(self, last_sequence):
        """Predict next price given a sequence"""
        if not self.is_trained:
            raise ValueError("Model must be trained first")
        
        # Ensure proper shape
        if len(last_sequence.shape) == 1:
            last_sequence = last_sequence.reshape(1, -1, 1)
        elif len(last_sequence.shape) == 2:
            last_sequence = last_sequence.reshape(1, last_sequence.shape[0], 1)
        
        # Make prediction
        scaled_prediction = self.model.predict(last_sequence, verbose=0)
        prediction = self.scaler.inverse_transform(scaled_prediction)
        return prediction[0][0]
    
    def generate_predictions(self, stock_data, days_to_predict=30):
        """Generate multiple day predictions"""
        if not self.is_trained:
            raise ValueError("Model must be trained first")
        
        # Get last sequence from training data
        close_prices, scaled_data = self.prepare_data(stock_data)
        current_sequence = scaled_data[-self.time_steps:].flatten()
        
        predictions = []
        
        for day in range(days_to_predict):
            # Predict next price
            next_prediction_scaled = self.model.predict(
                current_sequence.reshape(1, self.time_steps, 1), verbose=0
            )
            
            # Convert back to actual price
            next_prediction = self.scaler.inverse_transform(next_prediction_scaled)[0][0]
            predictions.append(next_prediction)
            
            # Update sequence for next prediction
            current_sequence = np.append(current_sequence[1:], next_prediction_scaled[0][0])
        
        return predictions
    
    def save_model(self, model_dir, symbol):
        """Save trained model and scaler"""
        if not self.is_trained:
            raise ValueError("No trained model to save")
        
        os.makedirs(model_dir, exist_ok=True)
        
        # Save model with .keras extension for Keras 3 compatibility
        model_path = os.path.join(model_dir, f"{symbol}_lstm_model.keras")
        self.model.save(model_path)
        
        # Save scaler
        scaler_path = os.path.join(model_dir, f"{symbol}_scaler.pkl")
        with open(scaler_path, 'wb') as f:
            pickle.dump(self.scaler, f)
        
        print(f"Model saved to {model_path}")
        print(f"Scaler saved to {scaler_path}")
    
    def load_model(self, model_dir, symbol):
        """Load trained model and scaler"""
        from tensorflow.keras.models import load_model
        
        model_path = os.path.join(model_dir, f"{symbol}_lstm_model.keras")
        scaler_path = os.path.join(model_dir, f"{symbol}_scaler.pkl")
        
        # Check for both .keras and legacy formats
        legacy_model_path = os.path.join(model_dir, f"{symbol}_lstm_model")
        
        if os.path.exists(model_path):
            # Load .keras format
            self.model = load_model(model_path)
            print(f"Model loaded from {model_path}")
        elif os.path.exists(legacy_model_path):
            # Try to load legacy format and convert
            try:
                self.model = load_model(legacy_model_path)
                print(f"Legacy model loaded from {legacy_model_path}")
                # Re-save in new format
                self.model.save(model_path)
                print(f"Model converted and saved to {model_path}")
            except Exception as e:
                print(f"Error loading legacy model: {e}")
                raise FileNotFoundError("Cannot load model in any supported format")
        else:
            raise FileNotFoundError("Model file not found")
        
        # Load scaler
        if not os.path.exists(scaler_path):
            raise FileNotFoundError("Scaler file not found")
        
        with open(scaler_path, 'rb') as f:
            self.scaler = pickle.load(f)
        
        self.is_trained = True
        print(f"Scaler loaded from {scaler_path}")
    
    def cleanup_legacy_models(self, model_dir, symbol):
        """Clean up old model formats after successful conversion"""
        legacy_model_path = os.path.join(model_dir, f"{symbol}_lstm_model")
        if os.path.exists(legacy_model_path):
            try:
                # Remove the old SavedModel directory
                if os.path.isdir(legacy_model_path):
                    shutil.rmtree(legacy_model_path)
                    print(f"Cleaned up legacy model directory: {legacy_model_path}")
                else:
                    os.remove(legacy_model_path)
                    print(f"Cleaned up legacy model file: {legacy_model_path}")
            except Exception as e:
                print(f"Warning: Could not clean up legacy model: {e}")