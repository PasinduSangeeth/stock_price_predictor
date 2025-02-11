import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam

class StockPricePredictor:
    """Class for creating and training the LSTM model"""
    
    def __init__(self, sequence_length: int = 60, n_features: int = 5):
        self.sequence_length = sequence_length
        self.n_features = n_features
        self.model = self._build_model()
    
    def _build_model(self) -> Sequential:
        """Build LSTM model architecture"""
        model = Sequential([
            LSTM(units=50, return_sequences=True, 
                 input_shape=(self.sequence_length, self.n_features)),
            Dropout(0.2),
            LSTM(units=50, return_sequences=False),
            Dropout(0.2),
            Dense(units=1)
        ])
        
        model.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error')
        return model
    
    def train(self, X: np.ndarray, y: np.ndarray, epochs: int = 10, 
             batch_size: int = 32, validation_split: float = 0.1):
        """Train the model"""
        history = self.model.fit(
            X, y,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=validation_split,
            verbose=1
        )
        return history
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions"""
        return self.model.predict(X) 