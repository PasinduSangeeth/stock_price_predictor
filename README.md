# Stock Price Predictor

A machine learning application that predicts stock prices using LSTM neural networks and provides real-time analysis through an interactive web interface.

## Features
- Real-time stock data collection from Yahoo Finance
- Interactive web interface for easy predictions
- Historical price visualization with trend analysis
- Volume trend analysis with comparative metrics
- Price change predictions with confidence metrics
- Support for multiple stock symbols (AAPL, MSFT, GOOGL, TSLA, etc.)

## Tech Stack
- **Backend**: Python 3.11.6
- **Machine Learning**: TensorFlow, LSTM Neural Networks
- **Web Interface**: Streamlit
- **Data Processing**: pandas, numpy
- **Data Collection**: yfinance API
- **Visualization**: matplotlib
- **Version Control**: Git

## Screenshots
![Stock Price Predictor Interface](image.png)
*Stock Price Prediction Interface showing historical data, volume analysis, and price predictions*

## Key Features Demonstrated
1. **Data Analysis**:
   - Historical price trends
   - Volume analysis
   - Price change predictions

2. **Machine Learning**:
   - LSTM neural network implementation
   - Real-time model training
   - Prediction accuracy metrics

3. **User Interface**:
   - Interactive stock symbol input
   - Adjustable historical data range
   - Visual price trend analysis
   - Detailed volume metrics

## Installation & Usage
1. Clone the repository:
```bash
git clone [your-repository-url]
cd stock_price_predictor
```

2. Install requirements:
```bash
pip install -r requirements.txt
```

3. Run the application:
```bash
streamlit run web_app.py
```

## Project Structure
```
stock_price_predictor/
├── src/
│   ├── data_collector.py    # Yahoo Finance data collection
│   ├── data_preprocessor.py # Data preprocessing
│   ├── model.py            # LSTM model implementation
│   └── model_manager.py    # Model saving/loading utilities
├── web_app.py              # Streamlit interface
└── requirements.txt        # Project dependencies
```

## Future Improvements
- Multiple prediction timeframes
- Additional technical indicators
- Portfolio optimization suggestions
- Real-time market news integration
- Enhanced visualization options

## Contributing
Pull requests are welcome. For major changes, please open an issue first.

## License
MIT

## Author
Pasindu Sangeeth

- LinkedIn: [Pasindu Sangeeth](https://www.linkedin.com/in/sangeeth99)
