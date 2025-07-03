import os
from dotenv import load_dotenv
import requests
import time
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from utils import load_data, FourierFeatures, TechnicalIndicators
import matplotlib.pyplot as plt
import pickle


# Define tickers from Plan.md
core_tickers = ['GS']  # Goldman Sachs
correlated_tickers = ['JPM', 'MS', 'C']  # Banking peers
market_indices = ['SPY', 'XLF']  # Market context
volatility_tickers = ['VIX']  # Volatility index

# Combine all tickers
all_tickers = core_tickers + correlated_tickers + market_indices + volatility_tickers

print(f"Fetching data for tickers: {all_tickers}")
print(f"Date range: 2010-01-01 to 2024-12-31")


master_data_clean, stock_data = load_data(all_tickers)

# Apply Fourier features and technical indicators to GS data
if 'GS' in stock_data:
    # First add Fourier features
    gs_enhanced = FourierFeatures.add_fourier_components(stock_data['GS'])
    print("Fourier transform features added")
    print(f"Shape after Fourier: {gs_enhanced.shape}")
    
    # Then add technical indicators
    gs_enhanced = TechnicalIndicators.add_all_indicators(gs_enhanced)
    print("Technical indicators added to GS data")
    print(f"Final shape: {gs_enhanced.shape}")
    print(f"New columns: {list(gs_enhanced.columns)}")
    
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
    plt.show()