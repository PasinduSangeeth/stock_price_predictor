import unittest
from datetime import datetime
from src.data_collector import StockDataCollector

class TestDataCollector(unittest.TestCase):
    def setUp(self):
        self.collector = StockDataCollector()
        
    def test_get_stock_data(self):
        symbol = 'AAPL'
        start = datetime(2023, 1, 1)
        end = datetime(2023, 12, 31)
        
        df = self.collector.get_stock_data(symbol, start, end)
        
        self.assertIsNotNone(df)
        self.assertGreater(len(df), 0)
        self.assertTrue('Close' in df.columns) 