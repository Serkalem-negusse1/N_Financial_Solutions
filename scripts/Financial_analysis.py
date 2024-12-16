import yfinance as yf
import ta
import pandas as pd
import numpy as np
import plotly.express as px
from pypfopt.efficient_frontier import EfficientFrontier
from pypfopt import risk_models
from pypfopt import expected_returns

import matplotlib.pyplot as plt
import matplotlib.dates as mdates  # For date formatting on X-axis

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

    def calculate_bollinger_bands(self, data, window_size=20, num_std_dev=2):
        data['SMA'] = self.calculate_moving_average(data['Close'], window_size)
        rolling_std = data['Close'].rolling(window=window_size).std()
        data['Bollinger_High'] = data['SMA'] + (rolling_std * num_std_dev)
        data['Bollinger_Low'] = data['SMA'] - (rolling_std * num_std_dev)
        return data

    # Unified plotting methods with year formatting
    def _format_x_axis(self, ax):
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))  # Format major ticks as years
        ax.xaxis.set_major_locator(mdates.YearLocator(base=5))    # Place ticks every 5 years
        ax.tick_params(axis='x', which='major', labelsize=10)     # Adjust label size
        ax.xaxis.set_minor_locator(mdates.YearLocator(base=1))    # Minor ticks every year
        plt.setp(ax.get_xticklabels(), rotation=45, ha='right')   # Rotate labels for clarity

    def plot_stock_data(self, data, ax):
        ax.plot(data.index, data['Close'], label='Close Price', color='blue')
        ax.plot(data.index, data['SMA'], label='SMA (20)', color='orange')
        ax.set_title("Stock Price with SMA")
        ax.legend()
        self._format_x_axis(ax)  # Apply date formatting

    def plot_rsi(self, data, ax):
        ax.plot(data.index, data['RSI'], label='RSI', color='purple')
        ax.axhline(70, color='red', linestyle='--', linewidth=1)
        ax.axhline(30, color='green', linestyle='--', linewidth=1)
        ax.set_title("Relative Strength Index (RSI)")
        ax.legend()
        self._format_x_axis(ax)

    def plot_ema(self, data, ax):
        ax.plot(data.index, data['Close'], label='Close Price', color='blue')
        ax.plot(data.index, data['EMA'], label='EMA (20)', color='green')
        ax.set_title("Stock Price with EMA")
        ax.legend()
        self._format_x_axis(ax)

    def plot_macd(self, data, ax):
        ax.plot(data.index, data['MACD'], label='MACD', color='black')
        ax.plot(data.index, data['MACD_Signal'], label='Signal Line', color='red')
        ax.set_title("MACD and Signal Line")
        ax.legend()
        self._format_x_axis(ax)

    def plot_bollinger_bands(self, data, ax):
        ax.plot(data.index, data['Close'], label='Close Price', color='blue')
        ax.plot(data.index, data['Bollinger_High'], label='Bollinger High', color='green')
        ax.plot(data.index, data['Bollinger_Low'], label='Bollinger Low', color='red')
        ax.plot(data.index, data['SMA'], label='SMA (20)', color='orange', linestyle='--')  # Add SMA
        ax.set_title("Bollinger Bands with SMA")
        ax.legend()
        self._format_x_axis(ax)  # Apply X-axis formatting