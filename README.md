# A Novel LLM-LSTM Integration Framework for Stock Price Prediction

**Research Implementation** - This repository contains the complete implementation for the research paper "A Novel LLM-LSTM Integration Framework for Stock Price Prediction" by Eric Yu and Tony Du.

## 📄 Paper Abstract

This research introduces an integration of Long Short-Term Memory (LSTM) and Large Language Model (LLM) that leverages quantitative price patterns and qualitative market sentiment for enhanced stock prediction. The framework generates base forecasts through LSTM analysis of historical price sequences, followed by LLM-based sentiment adjustments derived from financial news.

## 🎯 Research Methodology

The implementation follows the exact methodology described in the paper:
- **Two-layer LSTM** with 50 units per layer
- **60-day time sequences** for temporal modeling  
- **LLaMA 3 integration** via Ollama for sentiment analysis
- **±5% adjustment constraints** to prevent sentiment dominance
- **21-day evaluation period** (weekdays only)
- **13-week news collection** prior to prediction start

## 🏗️ Project Structure

```
├── src/                          # Core source code modules
│   ├── data_handler.py          # Data loading, caching, and preprocessing
│   ├── lstm_model.py            # LSTM model implementation
│   ├── llm_enhancer.py          # LLM-based prediction enhancement
│   ├── metrics.py               # Evaluation metrics (RMSE, MAPE, Directional Accuracy)
│   ├── visualizer.py            # Plotting and visualization utilities
│   └── predictor.py             # Main orchestrator class
├── main_clean.py                # Clean main entry point
├── compare_models.py            # Model comparison utility
├── download_data.py             # Stock data download utility
├── cache/                       # Cached data directory
├── models/                      # Trained model storage
├── output/                      # Prediction results and plots
├── news/                        # News data files
└── requirements.txt             # Python dependencies
```

## 🚀 Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Download Stock Data
```bash
python download_data.py TSLA --days 365 --output TSLA_historical.csv
```

### 3. Run Predictions
```bash
python main_clean.py TSLA news/TSLA_news_20250216_to_20250417.json \
    --train-start 2023-01-01 \
    --train-end 2024-12-31 \
    --days 30 \
    --actual-data TSLA_historical.csv
```

### 4. Compare Models
```bash
python compare_models.py \
    output/TSLA/TSLA_predictions_20250104_120000.csv \
    output/TSLA/TSLA_lstm_only_20250104_120000.csv \
    TSLA_historical.csv
```

## 📊 How It Works

### 1. LSTM Prediction
- Trains on historical stock price data
- Uses 60-day sequences to predict next day prices
- Generates baseline predictions for specified period

### 2. LLM Enhancement
- Analyzes news sentiment and market context
- Adjusts LSTM predictions based on news analysis
- Applies bounded adjustments (±5%) to prevent extreme modifications

### 3. Evaluation
- Calculates RMSE (Root Mean Square Error)
- Measures MAPE (Mean Absolute Percentage Error)
- Computes Directional Accuracy with 1% threshold
- Provides time-based performance metrics

## 🔧 Command Line Interface

### Main Prediction Script
```bash
python main_clean.py <symbol> <news_file> [options]

Required:
  symbol              Stock symbol (e.g., TSLA, MSFT, GOOGL)
  news_file           Path to news JSON file

Options:
  --train-start       Training start date (default: 2023-01-01)
  --train-end         Training end date (default: 2024-12-31)
  --days              Days to predict (default: 30)
  --no-llm            Skip LLM enhancement
  --actual-data       CSV file with actual prices for evaluation
  --model-dir         Model storage directory (default: models)
  --output-dir        Results output directory (default: output)
```

### Data Download
```bash
python download_data.py <symbol> [options]

Options:
  --start             Start date (YYYY-MM-DD)
  --end               End date (YYYY-MM-DD)
  --days              Days back from today
  --output            Output filename
```

### Model Comparison
```bash
python compare_models.py <enhanced_file> <lstm_file> <historical_file> [options]

Options:
  --symbol            Stock symbol
  --no-plots          Skip plot generation
```

## 📈 Output Files

### Prediction Results
- `{SYMBOL}_predictions_{timestamp}.csv` - Enhanced predictions with LLM adjustments
- `{SYMBOL}_lstm_only_{timestamp}.csv` - Baseline LSTM predictions
- `{SYMBOL}_comparison_{timestamp}.png` - Visual comparison plot
- `{SYMBOL}_accuracy_{timestamp}.png` - Prediction accuracy over time

### CSV Format
```csv
Date,LSTM_Prediction,Final_Prediction,Actual_Price
2025-01-01,150.25,152.10,151.80
2025-01-02,151.30,150.95,150.45
...
```

## 🎯 Key Features

### Modular Architecture
- **Separation of Concerns**: Each component has a single responsibility
- **Reusable Components**: Easy to swap LSTM models or LLM providers
- **Clean Interfaces**: Well-defined APIs between modules

### Robust Data Handling
- **Intelligent Caching**: Reduces API calls and improves performance
- **Error Recovery**: Graceful handling of missing data or API failures
- **Data Validation**: Ensures data quality throughout the pipeline


## 📋 Requirements

- Python 3.8+
- TensorFlow/Keras for LSTM models
- Local LLM server (Ollama) for enhancement
- News data in JSON format
- Historical stock data access

## 🤝 Contributing

This is a research project. For academic collaboration or questions about the methodology, please refer to the associated research paper.

## 📄 License

Academic use only. Please cite the associated research paper when using this code.
