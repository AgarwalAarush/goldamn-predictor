# Fourier Transform Features (as specified in Plan.md)
class FourierFeatures:
    @staticmethod
    def add_fourier_components(df, price_col='close', components=[3, 6, 9, 100]):
        """
        Add Fourier transform features for trend analysis
        """
        df = df.copy()
        
        # Get price series
        prices = df[price_col].dropna().values
        
        # Compute FFT
        fft_values = np.fft.fft(prices)
        
        # Extract components for different trends
        for n_components in components:
            # Create filtered FFT
            fft_filtered = np.copy(fft_values)
            fft_filtered[n_components:-n_components] = 0
            
            # Inverse FFT to get filtered signal
            filtered_signal = np.fft.ifft(fft_filtered).real
            
            # Pad with NaN to match original length
            if len(filtered_signal) < len(df):
                padding = len(df) - len(filtered_signal)
                filtered_signal = np.pad(filtered_signal, (padding, 0), mode='constant', constant_values=np.nan)
            
            df[f'fourier_{n_components}'] = filtered_signal[:len(df)]
        
        return df
    
class PolygonDataPipeline:
    def __init__(self, api_key):
        self.api_key = api_key
        self.base_url = "https://api.polygon.io"
        
    def get_stock_data(self, ticker, start_date, end_date, multiplier=1, timespan='day'):
        """
        Fetch OHLCV data from Polygon API
        """
        url = f"{self.base_url}/v2/aggs/ticker/{ticker}/range/{multiplier}/{timespan}/{start_date}/{end_date}"
        params = {
            'adjusted': 'true',  # This handles stock splits and dividends
            'sort': 'asc',
            'apikey': self.api_key
        }
        
        try:
            response = requests.get(url, params=params)
            data = response.json()
            
            if data['status'] == 'OK' and 'results' in data:
                df = pd.DataFrame(data['results'])
                df['date'] = pd.to_datetime(df['t'], unit='ms')
                df = df.rename(columns={
                    'o': 'open',
                    'h': 'high', 
                    'l': 'low',
                    'c': 'close',
                    'v': 'volume',
                    'vw': 'vwap',
                    'n': 'transactions'
                })
                df = df[['date', 'open', 'high', 'low', 'close', 'volume', 'vwap', 'transactions']]
                return df
            else:
                print(f"Error fetching data: {data}")
                return None
        except Exception as e:
            print(f"Error: {e}")
            return None
    
    def get_multiple_stocks(self, tickers, start_date, end_date):
        """
        Fetch data for multiple tickers
        """
        all_data = {}
        for ticker in tickers:
            print(f"Fetching data for {ticker}...")
            data = self.get_stock_data(ticker, start_date, end_date)
            if data is not None:
                all_data[ticker] = data
            time.sleep(0.1)  # Rate limiting
        return all_data

class DataProcessor:
    def __init__(self):
        self.master_data = None
        
    def combine_all_data(self, stock_data_dict):
        """
        Combine all ticker data into master dataset
        """
        master_df = None
        
        for ticker, data in stock_data_dict.items():
            if data is not None and not data.empty:
                # Add technical indicators
                enhanced_data = TechnicalIndicators.add_all_indicators(data)
                
                # Add ticker prefix to columns (except date)
                enhanced_data = enhanced_data.rename(columns=lambda x: f"{ticker}_{x}" if x != 'date' else x)
                
                if master_df is None:
                    master_df = enhanced_data
                else:
                    # Merge on date
                    master_df = pd.merge(master_df, enhanced_data, on='date', how='outer')
        
        return master_df
    
    def clean_data(self, df):
        """
        Clean and validate the dataset
        """
        print(f"Original shape: {df.shape}")
        
        # Sort by date
        df = df.sort_values('date').reset_index(drop=True)
        
        # Forward fill missing values for technical indicators
        df = df.fillna(method='ffill')
        
        # Drop rows with too many missing values
        df = df.dropna(thresh=len(df.columns) * 0.7)  # Keep rows with at least 70% data
        
        print(f"Cleaned shape: {df.shape}")
        return df
    
    def save_data(self, df, filename='processed_data.pkl'):
        """
        Save processed data
        """
        with open(filename, 'wb') as f:
            pickle.dump(df, f)
        print(f"Data saved to {filename}")

# Technical Indicators Calculator (Enhanced)
class TechnicalIndicators:
    @staticmethod
    def add_all_indicators(df, price_col='close'):
        """
        Add comprehensive technical indicators as specified in Plan.md
        """
        df = df.copy()
        
        # Moving Averages
        df['ma_7'] = df[price_col].rolling(window=7).mean()
        df['ma_21'] = df[price_col].rolling(window=21).mean()
        df['ma_50'] = df[price_col].rolling(window=50).mean()
        df['ma_200'] = df[price_col].rolling(window=200).mean()
        
        # MACD (12, 26, 9 parameters)
        df['ema_12'] = df[price_col].ewm(span=12).mean()
        df['ema_26'] = df[price_col].ewm(span=26).mean()
        df['macd'] = df['ema_12'] - df['ema_26']
        df['macd_signal'] = df['macd'].ewm(span=9).mean()
        df['macd_histogram'] = df['macd'] - df['macd_signal']
        
        # Bollinger Bands (20-period, 2 standard deviations)
        df['bb_middle'] = df[price_col].rolling(window=20).mean()
        df['bb_std'] = df[price_col].rolling(window=20).std()
        df['bb_upper'] = df['bb_middle'] + (df['bb_std'] * 2)
        df['bb_lower'] = df['bb_middle'] - (df['bb_std'] * 2)
        
        # RSI (14-period)
        delta = df[price_col].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['rsi'] = 100 - (100 / (1 + rs))
        
        # Momentum indicators
        df['momentum'] = df[price_col] / df[price_col].shift(10) - 1
        df['log_momentum'] = np.log(df[price_col]) - np.log(df[price_col].shift(30))
        
        # Volume-based indicators
        if 'volume' in df.columns:
            # On-Balance Volume (OBV)
            df['obv'] = (np.sign(df[price_col].diff()) * df['volume']).fillna(0).cumsum()
            
            # Volume-Price Trend
            df['vpt'] = ((df[price_col].diff() / df[price_col].shift(1)) * df['volume']).fillna(0).cumsum()
        
        # Volatility measures
        df['log_returns'] = np.log(df[price_col] / df[price_col].shift(1))
        df['volatility_20'] = df['log_returns'].rolling(window=20).std() * np.sqrt(252)
        
        return df