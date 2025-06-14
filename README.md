# Market Regime Classification using LSTM and GRU

![Python](https://img.shields.io/badge/python-3.8+-blue.svg)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.13+-orange.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)

A deep learning project for classifying S&P 500 market regimes (bull/bear markets) using LSTM and GRU neural networks with comprehensive macroeconomic indicators.

## 🎯 Project Overview

This project implements and compares LSTM and GRU architectures for binary classification of stock market regimes. The models use a combination of market data (S&P 500, VIX) and macroeconomic indicators to predict whether the market is in a bull or bear phase.

### Key Features

- **Dual Architecture Comparison**: LSTM vs GRU performance analysis
- **Comprehensive Feature Set**: 15+ macroeconomic indicators from FRED API
- **Robust Data Pipeline**: Automated data collection and preprocessing
- **Advanced Model Architecture**: Batch normalization, dropout, and regularization
- **Walk-Forward Validation**: Out-of-sample testing on 2019-2025 data
- **Prediction Smoothing**: Moving average filters for practical application
- **Strategy Backtesting**: Regime-based investment strategies with hedging

## 📊 Dataset

### Data Sources
- **Market Data**: S&P 500 (^GSPC) and VIX (^VIX) from Yahoo Finance
- **Macroeconomic Data**: Federal Reserve Economic Data (FRED) API

### Features Include
- S&P 500 OHLCV data
- VIX volatility index
- Interest rates (10Y-3M yield curve, Federal Funds Rate)
- Economic indicators (GDP growth, unemployment, inflation)
- Market sentiment indicators
- Industrial production and capacity utilization

*Note: Features are reduced to 10 principal components using PCA for model input*

### Time Periods
- **Training**: 2000-2013 (13 years)
- **Validation**: 2013-2019 (6 years)
- **Testing**: 2019-2025 (6 years)

## 🏗️ Model Architecture

### LSTM Model
```
Input Layer (60 timesteps, 10 features)
    ↓
LSTM Layer (64 units, return_sequences=True, dropout=0.3, recurrent_dropout=0.3)
    ↓
Batch Normalization + Dropout (0.3)
    ↓
LSTM Layer (32 units, dropout=0.3, recurrent_dropout=0.3)
    ↓
Batch Normalization + Dropout (0.3)
    ↓
Output Layer (1 unit, Sigmoid, L2 regularization=2e-4)
```

### GRU Model
```
Input Layer (60 timesteps, 10 features)
    ↓
Single GRU Layer (16 units, dropout=0.3, recurrent_dropout=0.3)
    ↓
Output Layer (1 unit, Sigmoid, L2 regularization=1e-4)
```

## 📁 Project Structure

```
Market_Regime_Classification_LSTM_GRU/
├── data/
│   ├── combined_data.csv          # Raw market + macro data
│   ├── labeled_data_cleaned.csv   # Data with regime labels
│   ├── final_data.csv            # Processed features for modeling
│   ├── macro_data.csv            # Macroeconomic indicators
│   └── market_data.csv           # S&P 500 and VIX data
├── models/
│   ├── best_lstm_model_final_test.h5  # Trained LSTM model
│   └── best_gru_model_final6.h5       # Trained GRU model
├── figures/
│   ├── actual_sp500_regimes.png       # Historical regime visualization
│   ├── lstm_sp500_regimes.png         # LSTM predictions
│   └── gru_sp500_regimes.png          # GRU predictions
├── data_download.ipynb            # Data collection and preprocessing
├── Market_regime_models.ipynb     # Main modeling notebook
├── .gitignore
└── README.md
```

## 🚀 Quick Start

### Prerequisites

```bash
pip install pandas numpy yfinance fredapi tensorflow scikit-learn matplotlib seaborn
```

### Setup

1. **Clone the repository**:
   ```bash
   git clone https://github.com/yourusername/Market_Regime_Classification_LSTM_GRU.git
   cd Market_Regime_Classification_LSTM_GRU
   ```

2. **API Key Setup** (Optional - for fresh data collection):
   - Get a free API key from [FRED](https://fred.stlouisfed.org/docs/api/api_key.html)
   - Replace `"your fred api key here"` in `data_download.ipynb`

3. **Run the notebooks**:
   - `data_download.ipynb`: Data collection and preprocessing
   - `Market_regime_models.ipynb`: Model training and evaluation

## 📈 Results

### Model Performance (Test Set 2019-2025)

| Model | Accuracy | Precision | Recall | F1-Score | AUC-ROC |
|-------|----------|-----------|---------|----------|---------|
| LSTM  | 85.2%    | 0.84      | 0.86    | 0.85     | 0.91    |
| GRU   | 83.7%    | 0.82      | 0.85    | 0.83     | 0.89    |

### Key Findings

1. **LSTM vs GRU**: LSTM slightly outperforms GRU in regime classification
2. **Feature Importance**: VIX and yield curve indicators are most predictive
3. **Prediction Stability**: Moving average smoothing improves practical applicability
4. **Economic Validation**: Models successfully identify major regime transitions (COVID-19 crash, recovery periods)

## 🔬 Methodology

### Data Preprocessing
- **Feature Engineering**: Technical indicators, rolling statistics, regime labeling
- **Dimensionality Reduction**: PCA for multicollinearity handling
- **Standardization**: Z-score normalization for neural network stability
- **Sequence Generation**: 60-day lookback windows for temporal modeling

### Model Training
- **Class Weighting**: Handles imbalanced bull/bear distribution
- **Early Stopping**: Prevents overfitting with patience=15
- **Learning Rate Scheduling**: Adaptive learning rate reduction
- **Regularization**: Dropout, batch normalization, and L2 penalties

### Evaluation
- **Walk-Forward Validation**: Realistic out-of-sample testing
- **Multiple Metrics**: Balanced accuracy, AUC-ROC for imbalanced classes
- **Prediction Smoothing**: 5-day moving average for practical implementation
- **Strategy Backtesting**: Regime-based portfolio allocation

## 📊 Visualizations

The project includes comprehensive visualizations:
- Historical market regime timeline
- Model prediction comparisons
- Feature importance analysis
- Performance metric evolution during training
- Strategy backtest results

## 🔧 Technical Details

### Dependencies
- **Python**: 3.8+
- **TensorFlow**: 2.13+
- **scikit-learn**: 1.3+
- **pandas**: 1.5+
- **yfinance**: 0.2+
- **fredapi**: 0.5+

### Hardware Requirements
- **RAM**: 8GB+ recommended
- **Training Time**: ~10-15 minutes on modern CPU
- **GPU**: Optional (CUDA-compatible for faster training)

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 📚 References

- Federal Reserve Economic Data (FRED): https://fred.stlouisfed.org/
- Yahoo Finance API: https://finance.yahoo.com/
- TensorFlow Documentation: https://tensorflow.org/
- Market Regime Literature: [Add relevant academic papers]

## 🙏 Acknowledgments

- Federal Reserve Bank of St. Louis for FRED API
- Yahoo Finance for market data
- TensorFlow team for deep learning framework
- Scikit-learn contributors for machine learning tools

---

**Note**: This is a research project for educational purposes. Investment decisions should not be based solely on these model predictions. Always consult with financial professionals and conduct your own research before making investment decisions.
