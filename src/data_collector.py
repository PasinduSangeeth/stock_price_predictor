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
        """
        Download stock data for a given symbol and date range
        
        Args:
            symbol: Stock symbol (e.g., 'AAPL' for Apple)
            start_date: Start date for data collection
            end_date: End date for data collection
            
        Returns:
            DataFrame with stock data
        """
        try:
            print(f"Downloading data for {symbol}...")
            df = yf.download(symbol, start=start_date, end=end_date)
            
            # Save to CSV
            file_path = os.path.join(self.data_dir, f"{symbol}_data.csv")
            df.to_csv(file_path)
            print(f"Data saved to {file_path}")
            
            return df
            
        except Exception as e:
            print(f"Error downloading data for {symbol}: {str(e)}")
            raise 