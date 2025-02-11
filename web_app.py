import streamlit as st
import pandas as pd
from datetime import datetime, timedelta
import os
import matplotlib.pyplot as plt
import numpy as np
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
        
        # Add example stock symbols
        st.sidebar.markdown("""
        **Example Symbols:**
        - AAPL (Apple)
        - MSFT (Microsoft)
        - GOOGL (Google)
        - TSLA (Tesla)
        - AMZN (Amazon)
        """)
        
        symbol = st.sidebar.text_input("Stock Symbol", value="AAPL")
        days = st.sidebar.slider("Days of historical data", 100, 1000, 500)
        
        if st.sidebar.button("Predict"):
            if not symbol:
                st.error("Please enter a stock symbol")
            else:
                self.make_prediction(symbol.upper(), days)
    
    def make_prediction(self, symbol: str, days: int):
        try:
            # Get data
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days)
            
            with st.spinner("Fetching stock data..."):
                df = self.collector.get_stock_data(symbol, start_date, end_date)
            
            # Create models directory if it doesn't exist
            if not os.path.exists('models'):
                os.makedirs('models')
            
            # Preprocess data
            with st.spinner("Processing data..."):
                scaled_data, scaler = self.preprocessor.prepare_data(df)
                X, y = self.preprocessor.create_sequences(scaled_data)
            
            # Always train new model for now (to fix prediction issues)
            st.warning("Training new model...")
            model = StockPricePredictor(sequence_length=60, n_features=5)
            model.train(X, y, epochs=10)
            
            # Save model
            model_path, scaler_path = self.model_manager.save_model(model, scaler, symbol)
            
            # Make prediction
            last_sequence = X[-1:]
            prediction = model.predict(last_sequence)
            
            # Inverse transform prediction
            prediction_reshaped = np.zeros((1, 5))  # 5 features
            prediction_reshaped[0, 3] = prediction[0][0]  # 3 is the index for Close price
            prediction = scaler.inverse_transform(prediction_reshaped)[0, 3]
            
            # Display results
            st.subheader("Latest Stock Data")
            
            # Add indicator explanations
            st.markdown("""
            **Table Indicators Explanation:**
            - **Open**: Opening price of the trading day
            - **High**: Highest price during the trading day
            - **Low**: Lowest price during the trading day
            - **Close**: Closing price of the trading day
            - **Volume**: Number of shares traded
            
            *All prices are in USD ($)*
            """)
            
            # Display the data table
            st.dataframe(df.tail())
            
            # Add trading volume context
            avg_volume = float(df['Volume'].mean())
            last_volume = float(df['Volume'].iloc[-1])
            volume_change = ((last_volume - avg_volume) / avg_volume) * 100
            
            # Format numbers with commas
            st.markdown(f"""
            **Volume Analysis:**
            - Last Trading Volume: {int(last_volume):,} shares
            - Average Volume ({days} days): {int(avg_volume):,} shares
            - Volume Change: {volume_change:+.1f}% compared to average
            """)
            
            st.subheader("Prediction")
            st.write(f"Next day's predicted close price: ${prediction:.2f}")
            
            # Plot
            self.plot_stock_data(df, prediction)
            
        except Exception as e:
            st.error(f"Error occurred: {str(e)}")
            print(f"Error details: {str(e)}")
    
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
        try:
            # Create figure
            fig = plt.figure(figsize=(12, 6))
            
            # Get last 30 days of data for better visualization
            df_plot = df.tail(30).copy()
            dates = pd.to_datetime(df_plot.index)
            last_date = dates[-1]
            next_date = last_date + pd.Timedelta(days=1)
            
            # Plot historical data
            plt.plot(dates, df_plot['Close'], 
                    label='Historical Close Price',
                    color='blue',
                    linewidth=2)
            
            # Plot prediction point
            plt.scatter(next_date, prediction,
                       color='red',
                       s=100,
                       label='Prediction',
                       zorder=5)
            
            # Add last 5 days' close prices as annotations
            for i in range(-5, 0):
                price = float(df_plot['Close'].iloc[i])
                date = dates[i]
                plt.annotate(f'${price:.2f}',
                            (date, price),
                            xytext=(10, 10),
                            textcoords='offset points',
                            fontsize=8)
            
            # Add prediction annotation
            plt.annotate(f'${prediction:.2f}',
                        (next_date, prediction),
                        xytext=(10, -15),
                        textcoords='offset points',
                        color='red',
                        fontsize=10,
                        fontweight='bold')
            
            # Calculate price change percentage
            last_price = float(df_plot['Close'].iloc[-1])
            price_change = ((prediction - last_price) / last_price) * 100
            
            # Add price change to title
            plt.title(f'Stock Price Prediction\nPredicted Change: {price_change:+.2f}%', 
                     fontsize=16, pad=20)
            
            plt.xlabel('Date', fontsize=12)
            plt.ylabel('Price ($)', fontsize=12)
            plt.grid(True, linestyle='--', alpha=0.7)
            plt.legend(loc='upper left', fontsize=10)
            
            # Format x-axis
            plt.gca().xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%Y-%m-%d'))
            plt.xticks(rotation=45)
            
            # Adjust layout
            plt.tight_layout()
            
            # Display plot
            st.pyplot(fig)
            
            # Add price change analysis
            if price_change > 0:
                change_color = "green"
                trend = "increase"
            else:
                change_color = "red"
                trend = "decrease"
                
            st.markdown(f"""
            **Price Change Analysis:**
            - Current Price: ${last_price:.2f}
            - Predicted Price: ${prediction:.2f}
            - Expected Change: <span style='color:{change_color}'>{price_change:+.2f}%</span> ({trend})
            """, unsafe_allow_html=True)
            
        except Exception as e:
            st.error(f"Error in plotting: {str(e)}")
            print(f"Plotting error details: {str(e)}")

if __name__ == "__main__":
    app = StockPredictorApp()
    app.run() 