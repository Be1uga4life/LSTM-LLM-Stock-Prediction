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
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau


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
        """Build LSTM model architecture with regularization as per paper methodology"""
        from tensorflow.keras.layers import Dropout
        
        # Two-layer LSTM with 50 units each (as specified in paper)
        # Dropout added for overfitting prevention (paper methodology section)
        self.model = Sequential([
            LSTM(units=50, return_sequences=True, input_shape=input_shape),
            Dropout(0.2),  # Dropout for regularization (paper methodology)
            LSTM(units=50),
            Dropout(0.2),  # Dropout for regularization (paper methodology)
            Dense(units=1)
        ])
        self.model.compile(optimizer='adam', loss='mean_squared_error')
        return self.model
    
    def train(self, stock_data, epochs=50, batch_size=32):
        """Train the LSTM model with overfitting prevention"""
        print("Training LSTM model with regularization...")
        
        # Prepare data
        close_prices, scaled_data = self.prepare_data(stock_data)
        x_train, y_train = self.create_sequences(scaled_data)
        
        # Temporal split for train/validation (80/20)
        split_idx = int(0.8 * len(x_train))
        x_train_split, x_val = x_train[:split_idx], x_train[split_idx:]
        y_train_split, y_val = y_train[:split_idx], y_train[split_idx:]
        
        print(f"Training samples: {len(x_train_split)}, Validation samples: {len(x_val)}")
        
        # Reshape for LSTM
        x_train_split = np.reshape(x_train_split, (x_train_split.shape[0], x_train_split.shape[1], 1))
        x_val = np.reshape(x_val, (x_val.shape[0], x_val.shape[1], 1))
        
        # Build model
        self.build_model((x_train_split.shape[1], 1))
        
        # Define callbacks for overfitting prevention
        from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
        
        callbacks = [
            EarlyStopping(
                monitor='val_loss',
                patience=10,
                restore_best_weights=True,
                verbose=1,
                mode='min'
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5,
                min_lr=0.0001,
                verbose=1,
                mode='min'
            )
        ]
        
        # Train with validation and callbacks
        print("Training with early stopping and learning rate scheduling...")
        history = self.model.fit(
            x_train_split, y_train_split,
            validation_data=(x_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=1
        )
        
        self.is_trained = True
        
        # Print training summary
        final_epoch = len(history.history['loss'])
        final_train_loss = history.history['loss'][-1]
        final_val_loss = history.history['val_loss'][-1]
        
        print(f"Training completed after {final_epoch} epochs")
        print(f"Final training loss: {final_train_loss:.6f}")
        print(f"Final validation loss: {final_val_loss:.6f}")
        print(f"Overfitting gap: {((final_val_loss - final_train_loss) / final_train_loss * 100):.1f}%")
        
        return history
    
    def plot_training_history(self, history, save_path=None):
        """Plot training history to visualize overfitting prevention"""
        import matplotlib.pyplot as plt
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        
        # Plot training & validation loss
        ax1.plot(history.history['loss'], label='Training Loss', color='blue')
        ax1.plot(history.history['val_loss'], label='Validation Loss', color='red')
        ax1.set_title('Model Loss During Training')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot learning rate if available
        if 'lr' in history.history:
            ax2.plot(history.history['lr'], label='Learning Rate', color='green')
            ax2.set_title('Learning Rate Schedule')
            ax2.set_xlabel('Epoch')
            ax2.set_ylabel('Learning Rate')
            ax2.set_yscale('log')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
        else:
            ax2.text(0.5, 0.5, 'Learning Rate\nHistory Not Available', 
                    ha='center', va='center', transform=ax2.transAxes)
            ax2.set_title('Learning Rate Schedule')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Training history plot saved to: {save_path}")
        
        return fig
    
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