import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

class DataPreprocessor:
    """Class to handle data preprocessing for stock price prediction"""
    
    def __init__(self):
        self.scaler = MinMaxScaler(feature_range=(0, 1))
    
    def prepare_data(self, df: pd.DataFrame) -> tuple:
        """Prepare data for training"""
        features = ['Open', 'High', 'Low', 'Close', 'Volume']
        data = df[features].copy()
        data = data.dropna()
        scaled_data = self.scaler.fit_transform(data)
        return scaled_data, self.scaler
    
    def create_sequences(self, data: np.ndarray, seq_length: int = 60) -> tuple:
        """Create sequences for time series prediction"""
        X = []
        y = []
        for i in range(seq_length, len(data)):
            X.append(data[i-seq_length:i])
            y.append(data[i, 3])  # Index 3 is Close price
        return np.array(X), np.array(y)