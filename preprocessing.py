import os
from utils import load_or_create_dataset

# TODO: alter into a formal pipeline - retrieve data, apply features, save data, etc. or at least a function group or class that can be called

# Define tickers from Plan.md
core_tickers = ['GS']  # Goldman Sachs
correlated_tickers = ['JPM', 'MS', 'C']  # Banking peers
market_indices = ['SPY', 'XLF']  # Market context
volatility_tickers = ['VIX']  # Volatility index

# Combine all tickers
all_tickers = core_tickers + correlated_tickers + market_indices + volatility_tickers

print(f"Processing data for tickers: {all_tickers}")
print(f"Date range: 2010-01-01 to 2024-12-31")

# Follow the 4-step workflow: 
# 1) Load raw data or final dataset
# 2) Clean data (if needed)
# 3) Add features (if needed)
# 4) Save final dataset (if needed)
dataset, stock_data = load_or_create_dataset(all_tickers)

# Display dataset info
print(f"\nFinal dataset shape: {dataset.shape}")
print(f"Date range: {dataset['date'].min()} to {dataset['date'].max()}")
print(f"Sample columns: {list(dataset.columns[:10])}")
print(f"Available stock data: {list(stock_data.keys())}")

print("\nDataset ready for analysis and modeling!")