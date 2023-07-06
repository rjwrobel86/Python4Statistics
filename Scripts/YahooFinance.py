#Yahoo Finance API
import yfinance as yf
ticker_symbol = "AAPL" 
stock_data = yf.download(ticker_symbol)
print(stock_data)