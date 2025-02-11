# Stock Price Predictor

A machine learning application that predicts stock prices using LSTM neural networks.

## Features
- Historical stock data collection using yfinance
- Data preprocessing and sequence creation
- LSTM model for price prediction
- Model saving and loading
- Web interface using Streamlit
- Visualization of predictions

## Installation
1. Clone the repository:
```bash
git clone https://github.com/PasinduSangeeth/stock_price_predictor.git
cd stock_price_predictor
```

2. Create and activate virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage
1. Run the web application:
```bash
streamlit run src/web_app.py
```

2. Enter a stock symbol (e.g., AAPL for Apple)
3. Adjust the historical data period
4. Click "Predict" to see the results

## Project Structure
```
stock_price_predictor/
├── src/
│   ├── data_collector.py
│   ├── data_preprocessor.py
│   ├── model.py
│   ├── evaluator.py
│   ├── model_manager.py
│   └── web_app.py
├── tests/
│   └── test_data_collector.py
├── models/
├── requirements.txt
└── README.md
```

## Contributing
Pull requests are welcome. For major changes, please open an issue first.

## License
MIT
