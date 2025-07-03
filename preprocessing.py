import os
from dotenv import load_dotenv
import requests
import time
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from utils import load_or_create_dataset, FourierFeatures, TechnicalIndicators
import matplotlib.pyplot as plt
import pickle

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

# GS-specific analysis and visualization
if 'GS' in stock_data:
    print("\nPerforming GS-specific analysis...")
    gs_data = stock_data['GS']
    
    # Add enhanced features for visualization
    gs_enhanced = FourierFeatures.add_fourier_components(gs_data)
    gs_enhanced = TechnicalIndicators.add_all_indicators(gs_enhanced)
    
    print("Enhanced GS data for visualization:")
    print(f"Shape: {gs_enhanced.shape}")
    print(f"Columns: {list(gs_enhanced.columns)}")
    
    # Visualize Fourier components
    plt.figure(figsize=(14, 7))
    
    # Plot original and Fourier components
    plt.plot(gs_enhanced['date'], gs_enhanced['close'], label='Original Price', alpha=0.7)
    for component in [3, 6, 9, 100]:
        plt.plot(gs_enhanced['date'], gs_enhanced[f'fourier_{component}'], 
                label=f'Fourier {component} components', alpha=0.8)
    
    plt.title('Goldman Sachs Price with Fourier Transform Components')
    plt.xlabel('Date')
    plt.ylabel('Price (USD)')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # create images folder if it doesn't exist
    if not os.path.exists('images'):
        os.makedirs('images')

    plt.savefig('images/gs_fourier.png')

    plt.show()