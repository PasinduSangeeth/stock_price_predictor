import yfinance as yf
import pandas as pd
from datetime import datetime
import os

class StockDataCollector:
    """Class to handle stock data collection from Yahoo Finance"""
    
    def __init__(self):
        # Create data directory if it doesn't exist
        self.data_dir = os.path.join('data', 'raw')
        os.makedirs(self.data_dir, exist_ok=True)
    
    def get_stock_data(self, symbol: str, start_date: datetime, end_date: datetime) -> pd.DataFrame:
        """Get historical stock data from Yahoo Finance"""
        try:
            # Verify symbol exists
            ticker = yf.Ticker(symbol)
            info = ticker.info
            if not info:
                raise ValueError(f"Invalid stock symbol: {symbol}")
            
            # Download data
            df = yf.download(symbol, start=start_date, end=end_date)
            if df.empty:
                raise ValueError(f"No data found for symbol {symbol}")
            
            # Verify required columns exist
            required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
            if not all(col in df.columns for col in required_columns):
                raise ValueError(f"Missing required columns for {symbol}")
            
            # Save to CSV
            file_path = os.path.join(self.data_dir, f"{symbol}_data.csv")
            df.to_csv(file_path)
            print(f"Data saved to {file_path}")
            
            return df
            
        except Exception as e:
            print(f"Error downloading data for {symbol}: {str(e)}")
            raise Exception(f"Error fetching data for {symbol}: {str(e)}") 