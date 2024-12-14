import yfinance as yf
import ta
import pandas as pd
import numpy as np
import plotly.express as px
from pypfopt.efficient_frontier import EfficientFrontier
from pypfopt import risk_models
from pypfopt import expected_returns

class FinancialAnalysisTalib:
    def __init__(self, data):
        self.data = data

    def calculate_moving_average(self, data, window_size):
        return ta.trend.sma_indicator(data, window=window_size)

    def calculate_technical_indicators(self, data):
        data['SMA'] = self.calculate_moving_average(data['Close'], 20)
        data['RSI'] = ta.momentum.RSIIndicator(data['Close'], window=14).rsi()
        data['EMA'] = ta.trend.ema_indicator(data['Close'], window=20)
        macd = ta.trend.MACD(data['Close'])
        data['MACD'] = macd.macd()
        data['MACD_Signal'] = macd.macd_signal()
        return data

    def plot_stock_data(self, data):
        fig = px.line(data, x=data.index, y=['Close', 'SMA'], title='Stock Price with Moving Average')
        fig.show()

    def plot_rsi(self, data):
        fig = px.line(data, x=data.index, y='RSI', title='Relative Strength Index (RSI)')
        fig.show()

    def plot_ema(self, data):
        fig = px.line(data, x=data.index, y=['Close', 'EMA'], title='Stock Price with Exponential Moving Average')
        fig.show()

    def plot_macd(self, data):
        fig = px.line(data, x=data.index, y=['MACD', 'MACD_Signal'], title='Moving Average Convergence Divergence (MACD)')
        fig.show()
    
    def calculate_bollinger_bands(self, data, window_size=20, num_std_dev=2):
        data['SMA'] = self.calculate_moving_average(data['Close'], window_size)
        rolling_std = data['Close'].rolling(window=window_size).std()
        data['Bollinger_High'] = data['SMA'] + (rolling_std * num_std_dev)
        data['Bollinger_Low'] = data['SMA'] - (rolling_std * num_std_dev)
        return data

    def plot_bollinger_bands(self, data):
        fig = px.line(data, x=data.index, y=['Close', 'SMA', 'Bollinger_High', 'Bollinger_Low'],
                      title='Stock Price with Bollinger Bands')
        fig.show()

    # def calculate_portfolio_weights(self, data):
    #     # Replace infinite values with NaN
    #     data = data.replace([float('inf'), float('-inf')], float('nan'))
        
    #     # Drop rows with any NaN values
    #     data = data.dropna()
        
    #     # Ensure there's still data left after cleaning
    #     if data.empty:
    #         raise ValueError("Data is empty after cleaning. Please check your data for NaNs or infinite values.")
        
    #     # Calculate expected annualized returns and risk
    #     mu = expected_returns.mean_historical_return(data)
    #     s = risk_models.sample_cov(data)
        
    #     # Ensure the covariance matrix is symmetric
    #     s = (s + s.T) / 2
    #     print("Covariance matrix:", s)
        
    #     # Make sure the covariance matrix is positive semi-definite
    #     s = np.maximum(s, 0)
    #     print("Adjusted covariance matrix:", s)
        
    #     # Obtain the Efficient Frontier with Maximum Sharpe
    #     ef = EfficientFrontier(mu, s)
    #     try:
    #         weights = ef.max_sharpe()
    #     except Exception as e:
    #         print("Error during optimization:", e)
    #         raise
        
    #     # Get interpretable weights
    #     cleaned_weights = ef.clean_weights()
    #     print("Cleaned weights:", cleaned_weights)
        
    #     # Display portfolio performance
    #     performance = ef.portfolio_performance(verbose=True)
        
    #     # Convert the cleaned weights to a dictionary with tickers as keys
    #     return cleaned_weights

    

    # def calculate_portfolio_performance(self, data):
    #     mu = expected_returns.mean_historical_return(data)
    #     cov = risk_models.sample_cov(data)
    #     ef = EfficientFrontier(mu, cov)
    #     weights = ef.max_sharpe()
    #     portfolio_return, portfolio_volatility, sharpe_ratio = ef.portfolio_performance()
    #     return portfolio_return, portfolio_volatility, sharpe_ratio


    def calculate_portfolio_weights(self, tickers, start_date, end_date):
        data = yf.download(tickers, start=start_date, end=end_date)['Close']
        mu = expected_returns.mean_historical_return(data)
        cov = risk_models.sample_cov(data)
        ef = EfficientFrontier(mu, cov)
        weights = ef.max_sharpe()
        weights = dict(zip(tickers, weights.values()))
        return weights

    def calculate_portfolio_performance(self, tickers, start_date, end_date):
            data = yf.download(tickers, start=start_date, end=end_date)['Close']
            mu = expected_returns.mean_historical_return(data)
            cov = risk_models.sample_cov(data)
            ef = EfficientFrontier(mu, cov)
            weights = ef.max_sharpe()
            portfolio_return, portfolio_volatility, sharpe_ratio = ef.portfolio_performance()
            return portfolio_return, portfolio_volatility, sharpe_ratio