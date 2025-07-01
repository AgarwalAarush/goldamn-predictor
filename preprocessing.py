import os
from dotenv import load_dotenv
import requests
import time
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from utils import FourierFeatures, TechnicalIndicators, DataProcessor, PolygonDataPipeline
import matplotlib.pyplot as plt
import pickle

load_dotenv()

POLYGON_API_KEY = os.getenv('POLYGON_API_KEY')
    
# Initialize pipeline
pipeline = PolygonDataPipeline(POLYGON_API_KEY)

# Define tickers from Plan.md
core_tickers = ['GS']  # Goldman Sachs
correlated_tickers = ['JPM', 'MS', 'C']  # Banking peers
market_indices = ['SPY', 'XLF']  # Market context
volatility_tickers = ['VIX']  # Volatility index

# Combine all tickers
all_tickers = core_tickers + correlated_tickers + market_indices + volatility_tickers

print(f"Fetching data for tickers: {all_tickers}")
print(f"Date range: 2010-01-01 to 2024-12-31")

# Initialize processor
processor = DataProcessor()

# Fetch the data
start_date = "2010-01-01"
end_date = "2024-12-31"

# Get all stock data
stock_data = pipeline.get_multiple_stocks(all_tickers, start_date, end_date)

if stock_data:
    print("Creating master dataset...")
    master_data = processor.combine_all_data(stock_data)
    
    if master_data is not None:
        print(f"Master dataset shape: {master_data.shape}")
        print(f"Date range: {master_data['date'].min()} to {master_data['date'].max()}")
        
        # Clean the data
        master_data_clean = processor.clean_data(master_data)
        
        # Save processed data
        processor.save_data(master_data_clean, "data/master_data.pkl")
        
        print("\nColumn overview:")
        print(f"Total columns: {len(master_data_clean.columns)}")
        print(f"Sample columns: {list(master_data_clean.columns[:10])}")
    else:
        print("Failed to create master dataset")

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
    for component in [3, 6, 9]:
        plt.plot(gs_enhanced['date'], gs_enhanced[f'fourier_{component}'], 
                label=f'Fourier {component} components', alpha=0.8)
    
    plt.title('Goldman Sachs Price with Fourier Transform Components')
    plt.xlabel('Date')
    plt.ylabel('Price (USD)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()