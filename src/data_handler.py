"""
Data handling utilities for stock data and news data
Consolidates all data loading, caching, and preprocessing functions
"""
import os
import json
import time
import hashlib
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
from pathlib import Path


class DataHandler:
    """Unified data handler for stock and news data with caching"""
    
    def __init__(self, cache_dir='cache'):
        self.cache_config = {
            'base_dir': cache_dir,
            'stock_cache_dir': f'{cache_dir}/stock_data',
            'news_cache_dir': f'{cache_dir}/news_data',
            'stock_cache_expiry_hours': float('inf'),
            'news_cache_expiry_hours': float('inf'),
            'max_cache_size_mb': 500,
        }
        self._setup_cache_directories()
    
    def _setup_cache_directories(self):
        """Setup all cache directories"""
        for cache_dir in [self.cache_config['base_dir'], 
                         self.cache_config['stock_cache_dir'],
                         self.cache_config['news_cache_dir']]:
            os.makedirs(cache_dir, exist_ok=True)
    
    def _get_cache_key(self, *args):
        """Generate cache key from arguments"""
        key_string = '_'.join(str(arg) for arg in args)
        return hashlib.md5(key_string.encode()).hexdigest()
    
    def _is_cache_valid(self, cache_file_path, expiry_hours):
        """Check if cache file is still valid"""
        if not os.path.exists(cache_file_path):
            return False
        file_age = time.time() - os.path.getmtime(cache_file_path)
        return file_age < (expiry_hours * 3600)
    
    def fetch_stock_data(self, symbol, start_date, end_date):
        """Fetch stock data with caching and fallback options"""
        print(f"Fetching data for {symbol} from {start_date} to {end_date}...")
        
        cache_key = self._get_cache_key(symbol, start_date, end_date)
        cache_file = os.path.join(self.cache_config['stock_cache_dir'], f"{cache_key}.csv")
        
        # Check cache first
        if self._is_cache_valid(cache_file, self.cache_config['stock_cache_expiry_hours']):
            print(f"Loading cached stock data...")
            try:
                data = pd.read_csv(cache_file, index_col=0, parse_dates=True)
                return self._validate_stock_data(data)
            except Exception as e:
                print(f"Error loading cached data: {e}")
                if os.path.exists(cache_file):
                    os.remove(cache_file)
        
        # Try to find existing CSV files in current directory
        potential_files = [
            f"{symbol}_stock_data.csv",
            f"{symbol}_historical.csv",
            f"{symbol.lower()}_stock_data.csv",
            f"{symbol.lower()}_historical.csv"
        ]
        
        for filename in potential_files:
            if os.path.exists(filename):
                print(f"Found existing data file: {filename}")
                try:
                    data = pd.read_csv(filename, parse_dates=['Date'])
                    data.set_index('Date', inplace=True)
                    
                    # Filter by date range if possible
                    start_dt = pd.to_datetime(start_date)
                    end_dt = pd.to_datetime(end_date)
                    
                    if 'Date' in data.columns:
                        data['Date'] = pd.to_datetime(data['Date'])
                        data = data[(data['Date'] >= start_dt) & (data['Date'] <= end_dt)]
                    else:
                        # Data is already indexed by date
                        data = data[(data.index >= start_dt) & (data.index <= end_dt)]
                    
                    validated_data = self._validate_stock_data(data)
                    # Cache the data for future use
                    validated_data.to_csv(cache_file)
                    print(f"Loaded and cached data from {filename}")
                    return validated_data
                except Exception as e:
                    print(f"Error loading {filename}: {e}")
                    continue
        
        # Fetch fresh data as last resort
        print("Fetching fresh stock data from Yahoo Finance...")
        try:
            data = yf.download(symbol, start=start_date, end=end_date, progress=False, auto_adjust=False)
            if data.empty:
                raise ValueError("No data returned from Yahoo Finance")
            
            validated_data = self._validate_stock_data(data)
            validated_data.to_csv(cache_file)
            print(f"Downloaded and cached stock data")
            return validated_data
            
        except Exception as e:
            print(f"Error fetching fresh data: {e}")
            
            # Final fallback: look for any cached data (even expired)
            print("Looking for any cached data as fallback...")
            for file in os.listdir(self.cache_config['stock_cache_dir']):
                if file.endswith('.csv'):
                    try:
                        cache_path = os.path.join(self.cache_config['stock_cache_dir'], file)
                        data = pd.read_csv(cache_path, index_col=0, parse_dates=True)
                        validated_data = self._validate_stock_data(data)
                        print(f"Using expired cache data from {file}")
                        return validated_data
                    except:
                        continue
            
            # If all else fails, create minimal sample data for testing
            print("⚠️ Creating minimal sample data for testing purposes")
            return self._create_sample_data(symbol, start_date, end_date)
    
    def _validate_stock_data(self, data):
        """Validate and clean stock data"""
        if data.empty:
            raise ValueError("No data returned")
        if 'Close' not in data.columns:
            raise ValueError("No 'Close' column in data")
        
        data['Close'] = pd.to_numeric(data['Close'], errors='coerce')
        data = data.dropna(subset=['Close'])
        
        if data.empty:
            raise ValueError("No valid data after cleaning")
        
        return data
    
    def load_news_data(self, news_file_path, symbol=None):
        """Load news data from JSON file"""
        if not os.path.exists(news_file_path):
            print(f"News file not found: {news_file_path}")
            return self._get_fallback_news(symbol)
        
        try:
            with open(news_file_path, 'r', encoding='utf-8') as f:
                news_data = json.load(f)
            print(f"Loaded {len(news_data)} news items from {news_file_path}")
            return news_data
        except Exception as e:
            print(f"Error loading news data: {e}")
            return self._get_fallback_news(symbol)
    
    def _get_fallback_news(self, symbol):
        """Generate fallback news data"""
        return [{
            'title': f'Sample news about {symbol}',
            'summary': 'Sample news summary for testing',
            'published': (datetime.now() - timedelta(days=1)).strftime('%Y-%m-%d'),
            'publisher': 'Test Publisher',
            'link': 'https://example.com/news',
            'type': 'Fallback'
        }]
    
    def _create_sample_data(self, symbol, start_date, end_date):
        """Create sample stock data for testing when no real data is available"""
        import numpy as np
        
        print(f"Creating sample data for {symbol} (for testing only)")
        
        # Create date range
        dates = pd.date_range(start=start_date, end=end_date, freq='D')
        
        # Generate realistic-looking stock prices
        np.random.seed(42)  # For reproducibility
        base_price = 150.0  # Starting price
        returns = np.random.normal(0.001, 0.02, len(dates))  # Daily returns
        prices = [base_price]
        
        for ret in returns[1:]:
            prices.append(prices[-1] * (1 + ret))
        
        # Create DataFrame
        data = pd.DataFrame({
            'Open': [p * 0.99 for p in prices],
            'High': [p * 1.02 for p in prices],
            'Low': [p * 0.98 for p in prices],
            'Close': prices,
            'Volume': np.random.randint(1000000, 5000000, len(dates))
        }, index=dates)
        
        print(f"⚠️ WARNING: Using sample data! Results are for testing only.")
        return data
    
    def save_predictions(self, predictions_data, output_file):
        """Save predictions to CSV file"""
        df = pd.DataFrame(predictions_data)
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        df.to_csv(output_file, index=False)
        print(f"Predictions saved to {output_file}")