import os
import joblib
from datetime import datetime

class ModelManager:
    """Class to handle model saving and loading"""
    
    def __init__(self):
        self.models_dir = 'models'
        os.makedirs(self.models_dir, exist_ok=True)
    
    def save_model(self, model, scaler, symbol: str):
        """Save model and scaler"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_filename = f"{symbol}_model_{timestamp}.h5"
        scaler_filename = f"{symbol}_scaler_{timestamp}.pkl"
        
        # Save model
        model_path = os.path.join(self.models_dir, model_filename)
        model.model.save(model_path)
        
        # Save scaler
        scaler_path = os.path.join(self.models_dir, scaler_filename)
        joblib.dump(scaler, scaler_path)
        
        print(f"Model saved to {model_path}")
        print(f"Scaler saved to {scaler_path}")
        
        return model_path, scaler_path
    
    def load_model(self, model_path: str, scaler_path: str):
        """Load saved model and scaler"""
        from tensorflow import keras
        from src.model import StockPricePredictor
        
        # Load model
        loaded_model = keras.models.load_model(model_path)
        
        # Create predictor instance
        predictor = StockPricePredictor(sequence_length=60, n_features=5)
        predictor.model = loaded_model
        
        # Load scaler
        scaler = joblib.load(scaler_path)
        
        return predictor, scaler 