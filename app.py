import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
from datetime import datetime
from src.data_collector import StockDataCollector
from src.data_preprocessor import DataPreprocessor
from src.model import StockPricePredictor
from src.evaluator import ModelEvaluator

# Set the time period for data
start_date = datetime(2015, 1, 1)
end_date = datetime.now()

# Download stock data 
stock_symbol = 'AAPL' # Apple stock
df = yf.download(stock_symbol, start=start_date, end=end_date)

# Save the data
df.to_csv('stock_data.csv')

# Display first few rows of the data
print(df.head())

plt.style.use('fivethirtyeight')
%matplotlib inline

# StockDataCollector object 
collector = StockDataCollector()

# get stock data
symbol = 'AAPL'  # apple stock
start = datetime(2020, 1, 1)  # from 2020
end = datetime.now()  # till now

# get stock data
data = collector.get_stock_data(symbol, start, end)

# print first 5 rows of data
print("\nFirst 5 rows of data:")
print(data.head())

plt.style.use('fivethirtyeight')
%matplotlib inline

start = datetime(2015, 1, 1)
end = datetime(2025, 1, 1)

df = data.DataReader('AAPL', 'yahoo', start, end)

def main():
    try:
        print("Starting stock price prediction...")
        
        # Initialize components
        collector = StockDataCollector()
        preprocessor = DataPreprocessor()
        evaluator = ModelEvaluator()
        
        # Get data
        symbol = 'AAPL'
        start_date = datetime(2020, 1, 1)
        end_date = datetime.now()
        
        print(f"\n1. Collecting data for {symbol}...")
        df = collector.get_stock_data(symbol, start_date, end_date)
        print("✓ Data collection completed")
        
        print("\n2. Preprocessing data...")
        scaled_data, scaler = preprocessor.prepare_data(df)
        X, y = preprocessor.create_sequences(scaled_data)
        print("✓ Data preprocessing completed")
        
        print("\n3. Splitting data and training model...")
        train_size = int(len(X) * 0.8)
        X_train, X_test = X[:train_size], X[train_size:]
        y_train, y_test = y[:train_size], y[train_size:]
        
        model = StockPricePredictor(sequence_length=60, n_features=5)
        history = model.train(X_train, y_train, epochs=10)
        print("✓ Model training completed")
        
        print("\n4. Making predictions and evaluating...")
        y_pred = model.predict(X_test)
        metrics = evaluator.evaluate_predictions(y_test, y_pred)
        evaluator.plot_predictions(y_test, y_pred, title=f"{symbol} Stock Price Prediction")
        print("✓ Evaluation completed")
        
    except Exception as e:
        print(f"\nError occurred: {str(e)}")

if __name__ == "__main__":
    main()


