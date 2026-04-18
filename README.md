# 🔥 Hybrid Deep Learning for Stock Price Prediction (LSTM • GRU • LSTM+GRU)

Time-series modeling | Deep learning | Quantitative finance | Sequence learning

---

## 🧭 Overview

This project implements and compares **deep learning sequence models** for stock price prediction using:

* Long Short-Term Memory (LSTM)
* Gated Recurrent Units (GRU)
* A hybrid LSTM–GRU architecture

The models are trained to predict the **next-day closing price** based on historical market data and technical indicators.

The implementation is inspired by:

> Hossain, M. A., et al. (2018)
> *Hybrid Deep Learning Model for Stock Price Prediction*
> IEEE SSCI
> DOI: 10.1109/SSCI.2018.8628641

---

## ⚡ Key Features

* 📊 Data source: Yahoo Finance (via `yfinance`)
* 📈 Technical indicators:

  * Moving averages (MA5, MA10, MA25)
  * MACD (EMA-based)
* 🧠 Deep learning models:

  * LSTM (stacked)
  * GRU (stacked)
  * Hybrid LSTM + GRU
* 📉 Evaluation metrics:

  * RMSE
  * MSE
  * MAE
  * MAPE
* 📊 Model comparison:

  * Performance table
  * Metric visualization

---

## 🧮 Feature Engineering

Input features include:

* Closing price
* Moving averages (5, 10, 25 days)
* MACD indicator

All features are normalized using MinMax scaling.

---

## 🧱 Data Representation

A sliding window approach is used:

* Input: past **15 days** of features
* Output: next-day closing price

Unlike tabular approaches, this model preserves **temporal structure**, allowing recurrent networks to capture:

* trend dynamics
* temporal dependencies
* nonlinear sequence behavior

---

## 🧠 Model Architectures

### 🔹 Model 1 — LSTM

* 3 stacked LSTM layers
* Captures long-term dependencies in time series

---

### 🔹 Model 2 — GRU

* 2 stacked GRU layers
* More computationally efficient than LSTM
* Designed to capture temporal dependencies with fewer parameters

---

### 🔹 Model 3 — Hybrid LSTM + GRU

* LSTM layer → GRU layer
* Combines:

  * long-term memory (LSTM)
  * efficient gating (GRU)
* Includes dropout for regularization

---

## 🏗️ Training Configuration

* Optimizer: Adam (learning rate = 1e-4)
* Loss: Mean Squared Error (MSE)
* Epochs: 50
* Batch size: 16
* Train/test split: 70% / 30%
* Validation split: 10% (from training set)
* No shuffling (preserves time order)

---

## 📊 Evaluation Metrics

The models are evaluated using:

* RMSE (Root Mean Squared Error)
* MSE (Mean Squared Error)
* MAE (Mean Absolute Error)
* MAPE (Mean Absolute Percentage Error)

---

## 📈 Results

### 🔹 Price Prediction Example

### 🔹 Model Comparison

* Train vs Test performance
* Metric comparison across models

---

## 🧠 Interpretation

This project highlights:

* The importance of **temporal modeling** in financial data
* Differences between LSTM and GRU architectures
* Benefits of hybrid architectures for capturing complex dynamics

Key observations:

* LSTM captures long-term dependencies effectively
* GRU offers faster convergence
* Hybrid models can balance both

---

## ⚠️ Limitations

* Single asset (AAPL only)
* Limited feature set (no volume, sentiment, etc.)
* No hyperparameter optimization
* No backtesting of trading strategy

---

## 🚀 Future Work

* Extend to multiple assets (stocks, crypto, indices)
* Add attention mechanisms / Transformers
* Feature expansion (volume, volatility, sentiment)
* Hyperparameter optimization
* Trading strategy backtesting

---

## ⚠️ Disclaimer

This project is for research and educational purposes only.
It does not constitute financial advice.

---

## 👤 Author

Martiros Khurshudyan | Physics | Machine Learning | Quantitative Finance


