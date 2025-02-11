import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
import datetime as dt

# Set the time period for data
start_date = dt.datetime(2015, 1, 1)
end_date = dt.datetime.now()

# Download stock data 
stock_symbol = 'AAPL' # Apple stock
df = yf.download(stock_symbol, start=start_date, end=end_date)

# Save the data
df.to_csv('stock_data.csv')

# Display first few rows of the data
print(df.head())

plt.style.use('fivethirtyeight')
%matplotlib inline

plt.style.use('fivethirtyeight')
%matplotlib inline

start = dt.datetime(2015, 1, 1)
end = dt.datetime(2025, 1, 1)

df = data.DataReader('AAPL', 'yahoo', start, end)


