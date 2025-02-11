import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
from datetime import datetime
from src.data_collector import StockDataCollector

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

# StockDataCollector object එකක් සාදනවා
collector = StockDataCollector()

# දත්ත ලබා ගැනීමේ parameters
symbol = 'AAPL'  # Apple සමාගම
start = datetime(2020, 1, 1)  # 2020 ජනවාරි 1 සිට
end = datetime.now()  # අද දක්වා

# දත්ත ලබා ගන්නවා
data = collector.get_stock_data(symbol, start, end)

# මුල් පේළි 5 පෙන්වනවා
print("\nFirst 5 rows of data:")
print(data.head())

plt.style.use('fivethirtyeight')
%matplotlib inline

start = datetime(2015, 1, 1)
end = datetime(2025, 1, 1)

df = data.DataReader('AAPL', 'yahoo', start, end)


