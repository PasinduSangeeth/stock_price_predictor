import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

class DataPreprocessor:
    """Class to handle data preprocessing for stock price prediction"""
    
    def __init__(self):
        self.scaler = MinMaxScaler(feature_range=(0, 1))
    
    def prepare_data(self, df: pd.DataFrame, target_column: str = 'Close') -> tuple:
        """
        Prepare data for training
        
        Args:
            df: DataFrame with stock data
            target_column: Column to predict (default: 'Close' price)
            
        Returns:
            Tuple of (scaled data, scaler object)
        """
        # Select features (using OHLCV data)
        features = ['Open', 'High', 'Low', 'Close', 'Volume']
        data = df[features].copy()
        
        # Handle missing values
        data = data.dropna()
        
        # Scale the features
        scaled_data = self.scaler.fit_transform(data)
        
        return scaled_data, self.scaler
    
    def create_sequences(self, data: np.ndarray, seq_length: int = 60) -> tuple:
        """
        Create sequences for time series prediction
        
        Args:
            data: Scaled data
            seq_length: Number of time steps to look back
            
        Returns:
            Tuple of (X, y) where X is the input sequences and y is the target values
        """
        X = []
        y = []
        
        for i in range(seq_length, len(data)):
            X.append(data[i-seq_length:i])
            y.append(data[i, 3])  # 3 is the index of Close price
        
        return np.array(X), np.array(y) 