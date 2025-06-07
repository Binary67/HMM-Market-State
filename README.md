# HMM Market State Predictor

This project analyzes historical market data using a Gaussian Hidden Markov Model (HMM) to classify market regimes as **Uptrend**, **Downtrend**, or **Sideway**. It provides utilities to engineer technical indicators, evaluate predictions and backtest a simple trading strategy.

## Overview

1. `DataDownloader.py` retrieves OHLCV data from Yahoo Finance.
2. `FeatureEngineering.py` adds a set of technical indicators.
3. `HiddenMarkovModel.py` fits and applies the HMM.
4. `ModelEvaluation.py` computes basic validation metrics.
5. `BacktestingModule.py` simulates a regime-based strategy.
6. `StreamlitInterface.py` exposes the pipeline through an interactive dashboard.

## Requirements

- Python 3.10+
- `pandas`, `numpy`, `yfinance`, `hmmlearn`, `scikit-learn`, `backtesting`, `streamlit`

Install dependencies with:

```bash
pip install pandas numpy yfinance hmmlearn scikit-learn backtesting streamlit
```

## Usage

Run the Streamlit interface to download data, train the model and execute a backtest:

```bash
python main.py
```

A browser window will open allowing you to choose the ticker, date range and backtest parameters.

## Features

- Historical data download from Yahoo Finance
- Modular technical indicator computation
- Hidden Markov Model for regime detection
- Backtesting with risk management and trailing stops
- Interactive dashboard for visualization and metrics
