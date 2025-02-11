import streamlit as st
import pandas as pd
from datetime import datetime, timedelta
import os
import matplotlib.pyplot as plt
from src.data_collector import StockDataCollector
from src.data_preprocessor import DataPreprocessor
from src.model import StockPricePredictor
from src.model_manager import ModelManager

class StockPredictorApp:
    def __init__(self):
        self.collector = StockDataCollector()
        self.preprocessor = DataPreprocessor()
        self.model_manager = ModelManager()
        
    def run(self):
        st.title("Stock Price Predictor")
        
        # Sidebar
        st.sidebar.header("Parameters")
        symbol = st.sidebar.text_input("Stock Symbol", value="AAPL")
        days = st.sidebar.slider("Days of historical data", 100, 1000, 500)
        
        if st.sidebar.button("Predict"):
            self.make_prediction(symbol, days)
    
    def make_prediction(self, symbol: str, days: int):
        try:
            # Get data
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days)
            
            with st.spinner("Fetching stock data..."):
                df = self.collector.get_stock_data(symbol, start_date, end_date)
            
            # Preprocess data
            with st.spinner("Processing data..."):
                scaled_data, scaler = self.preprocessor.prepare_data(df)
                X, y = self.preprocessor.create_sequences(scaled_data)
            
            # Load or train model
            model_files = self.get_latest_model(symbol)
            if model_files:
                model, scaler = self.model_manager.load_model(*model_files)
                st.success("Loaded existing model")
            else:
                st.warning("Training new model...")
                model = StockPricePredictor(sequence_length=60, n_features=5)
                model.train(X, y, epochs=10)
                self.model_manager.save_model(model, scaler, symbol)
            
            # Make prediction
            last_sequence = X[-1:]
            prediction = model.predict(last_sequence)
            
            # Display results
            st.subheader("Latest Stock Data")
            st.dataframe(df.tail())
            
            st.subheader("Prediction")
            st.write(f"Next day's predicted close price: ${prediction[0][0]:.2f}")
            
            # Plot
            self.plot_stock_data(df, prediction[0][0])
            
        except Exception as e:
            st.error(f"Error occurred: {str(e)}")
    
    def get_latest_model(self, symbol: str):
        """Get the latest model files for the symbol if they exist"""
        try:
            model_dir = self.model_manager.models_dir
            if not os.path.exists(model_dir):
                os.makedirs(model_dir)
                return None, None
            
            # Get all files for this symbol
            all_files = [f for f in os.listdir(model_dir) if f.startswith(symbol)]
            if not all_files:
                return None, None
            
            # Separate model and scaler files
            model_files = [f for f in all_files if f.endswith('.h5')]
            scaler_files = [f for f in all_files if f.endswith('.pkl')]
            
            if model_files and scaler_files:
                latest_model = os.path.join(model_dir, sorted(model_files)[-1])
                latest_scaler = os.path.join(model_dir, sorted(scaler_files)[-1])
                return latest_model, latest_scaler
            
            return None, None
        
        except Exception as e:
            print(f"Error in get_latest_model: {str(e)}")
            return None, None
    
    def plot_stock_data(self, df: pd.DataFrame, prediction: float):
        """Plot stock data with prediction"""
        fig = plt.figure(figsize=(12, 6))
        plt.plot(df['Close'], label='Historical Close Price')
        plt.scatter(len(df), prediction, color='red', label='Prediction')
        plt.title('Stock Price Prediction')
        plt.xlabel('Time')
        plt.ylabel('Price')
        plt.legend()
        st.pyplot(fig)

if __name__ == "__main__":
    app = StockPredictorApp()
    app.run() 