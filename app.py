import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
from datetime import datetime
from src.data_collector import StockDataCollector
from src.data_preprocessor import DataPreprocessor
from src.model import StockPricePredictor

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
    # Initialize components
    collector = StockDataCollector()
    preprocessor = DataPreprocessor()
    
    # Get data
    symbol = 'AAPL'
    start_date = datetime(2020, 1, 1)
    end_date = datetime.now()
    
    # Collect and preprocess data
    df = collector.get_stock_data(symbol, start_date, end_date)
    scaled_data, scaler = preprocessor.prepare_data(df)
    X, y = preprocessor.create_sequences(scaled_data)
    
    # Create and train model
    model = StockPricePredictor(sequence_length=60, n_features=5)
    history = model.train(X, y, epochs=10)  # Using 10 epochs for testing
    
    print("Model training completed!")

if __name__ == "__main__":
    main()


