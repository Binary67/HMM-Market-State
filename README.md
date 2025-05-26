# HMM Market State Predictor

A machine learning project that uses Hidden Markov Models (HMM) to predict market states: **Bullish** 📈, **Bearish** 📉, or **Sideways** ➡️.

## What is This Project?

Think of the stock market like weather patterns. Just like how today's weather gives us clues about tomorrow's weather, today's market behavior can help us predict tomorrow's market state. This project uses a mathematical model called Hidden Markov Model to find these hidden patterns and make predictions.

### Simple Example
- If the market has been going up for several days with high volume → Likely to continue **Bullish**
- If there's been a lot of selling pressure with declining prices → Might turn **Bearish**  
- If prices have been moving in a narrow range → Could stay **Sideways**

## How Hidden Markov Models Work (Simplified)

Imagine you're in a room with someone flipping coins, but you can't see the coins - only hear the results. However, you know there are two different coins:
- **Fair coin**: 50% heads, 50% tails
- **Biased coin**: 80% heads, 20% tails

By listening to the sequence of results (heads, tails, heads, heads...), you can guess which coin is being used and predict the next flip. That's exactly what HMM does with market data!

In our case:
- **Hidden states**: Market conditions (Bullish, Bearish, Sideways)
- **Observable data**: Price changes, volume, technical indicators
- **Prediction**: What market state comes next

## Features

- 📊 **Real-time market data processing**
- 🧠 **HMM model training and prediction**
- 📈 **Visualization of market states and predictions**
- 📱 **Easy-to-use interface**
- 🔄 **Automatic model retraining**
- 📊 **Performance metrics and backtesting**
