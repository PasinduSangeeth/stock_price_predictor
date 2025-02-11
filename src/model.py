import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam

class StockPricePredictor:
    """Class for creating and training the LSTM model"""
    
    def __init__(self, sequence_length: int, n_features: int):
        self.sequence_length = sequence_length
        self.n_features = n_features
        self.model = self._build_model()
        
    def _build_model(self):
        model = keras.Sequential([
            keras.layers.LSTM(50, return_sequences=True, 
                            input_shape=(self.sequence_length, self.n_features)),
            keras.layers.Dropout(0.2),
            keras.layers.LSTM(50, return_sequences=False),
            keras.layers.Dropout(0.2),
            keras.layers.Dense(1)
        ])
        
        model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001),
                     loss='mean_squared_error')
        return model
    
    def train(self, X_train: np.ndarray, y_train: np.ndarray, 
             epochs: int = 50, batch_size: int = 32, 
             validation_split: float = 0.1):
        """
        Train the model
        """
        history = self.model.fit(
            X_train, y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=validation_split,
            verbose=1
        )
        return history
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions
        """
        return self.model.predict(X) 