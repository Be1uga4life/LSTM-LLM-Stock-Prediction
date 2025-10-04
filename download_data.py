"""
Clean data download utility
Replaces download.py with better organization and error handling
"""
import argparse
import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
import os


def download_stock_data(symbol, start_date=None, end_date=None, filename=None):
    """
    Download stock data from yfinance
    
    Parameters:
    symbol (str): Stock symbol (e.g., 'TSLA', 'MSFT')
    start_date (str): Start date in 'YYYY-MM-DD' format
    end_date (str): End date in 'YYYY-MM-DD' format  
    filename (str): Output CSV filename
    """
    
    # Set default dates if not provided
    if end_date is None:
        end_date = datetime.now().strftime('%Y-%m-%d')
    
    if start_date is None:
        # Default to 1 year ago
        start_date = (datetime.now() - timedelta(days=365)).strftime('%Y-%m-%d')
    
    if filename is None:
        filename = f"{symbol}_stock_data.csv"
    
    print(f"Downloading {symbol} stock data...")
    print(f"Date range: {start_date} to {end_date}")
    
    try:
        # Download stock data
        ticker = yf.Ticker(symbol)
        hist = ticker.history(start=start_date, end=end_date)
        
        if hist.empty:
            raise ValueError(f"No data found for symbol {symbol}")
        
        # Reset index to make Date a column
        hist = hist.reset_index()
        
        # Format the Date column
        hist['Date'] = hist['Date'].dt.strftime('%Y-%m-%d')
        
        # Select and reorder columns
        formatted_data = hist[['Date', 'Open', 'High', 'Low', 'Close', 'Volume']].copy()
        
        # Create output directory if needed
        os.makedirs(os.path.dirname(filename) if os.path.dirname(filename) else '.', exist_ok=True)
        
        # Save to CSV
        formatted_data.to_csv(filename, index=False)
        
        print(f"✅ Successfully downloaded {symbol} stock data!")
        print(f"📊 Records: {len(formatted_data)}")
        print(f"💾 Saved to: {filename}")
        
        # Display sample data
        print(f"\nSample data (first 3 rows):")
        print(formatted_data.head(3).to_string(index=False))
        
        return formatted_data
        
    except Exception as e:
        print(f"❌ Error downloading data for {symbol}: {e}")
        raise


def main():
    parser = argparse.ArgumentParser(description='Download stock data from Yahoo Finance')
    
    parser.add_argument('symbol', help='Stock symbol (e.g., TSLA, MSFT, GOOGL)')
    parser.add_argument('--start', help='Start date (YYYY-MM-DD)')
    parser.add_argument('--end', help='End date (YYYY-MM-DD)')
    parser.add_argument('--output', help='Output filename (default: {SYMBOL}_stock_data.csv)')
    parser.add_argument('--days', type=int, help='Number of days back from today (alternative to --start)')
    
    args = parser.parse_args()
    
    # Handle days parameter
    start_date = args.start
    if args.days and not start_date:
        start_date = (datetime.now() - timedelta(days=args.days)).strftime('%Y-%m-%d')
    
    # Set output filename
    output_file = args.output or f"{args.symbol}_stock_data.csv"
    
    try:
        download_stock_data(
            symbol=args.symbol.upper(),
            start_date=start_date,
            end_date=args.end,
            filename=output_file
        )
        return 0
        
    except Exception as e:
        print(f"Download failed: {e}")
        return 1


if __name__ == "__main__":
    exit(main())