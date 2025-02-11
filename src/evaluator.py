import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error

class ModelEvaluator:
    """Class for evaluating model performance and making predictions"""
    
    def evaluate_predictions(self, y_true: np.ndarray, y_pred: np.ndarray):
        """Calculate error metrics"""
        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_true, y_pred)
        
        print(f"Mean Squared Error: {mse:.4f}")
        print(f"Root Mean Squared Error: {rmse:.4f}")
        print(f"Mean Absolute Error: {mae:.4f}")
        
        return {'mse': mse, 'rmse': rmse, 'mae': mae}
    
    def plot_predictions(self, y_true: np.ndarray, y_pred: np.ndarray, title: str = "Stock Price Prediction"):
        """Plot actual vs predicted values"""
        plt.figure(figsize=(12, 6))
        plt.plot(y_true, label='Actual')
        plt.plot(y_pred, label='Predicted')
        plt.title(title)
        plt.xlabel('Time')
        plt.ylabel('Price')
        plt.legend()
        plt.show() 