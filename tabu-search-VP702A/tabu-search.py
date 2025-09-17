#!/usr/bin/env python3

import random
# import matplotlib.pyplot as plt
from collections import deque
from sympy.utilities.iterables import multiset_permutations
import numpy as np
import yfinance as yf
import os.path
import pandas as pd
import logging

np.random.seed(42)

# Objectives:
# 1. Find optimal portfolio allocation
# 2. Minimize risk
# 3. Maximize return
#
# Constraints:
# 1. Atleast 60% in funds
# 2. Atleast 50% of funds in global market
# 3. Atleast 15% of funds in Swedish market

class TabuSearch:
    def __init__(self, jobs, tabu_size=5, max_iterations=100):
        self.jobs = jobs
        self.n = len(jobs)
        self.tabu_size = tabu_size
        self.max_iterations = max_iterations
        self.tabu_list = deque(maxlen=tabu_size)
        self.best_solution = None
        self.best_tardiness = float('inf')

class StockAnalyzer:
    def __init__(self, tickers, period="1y", risk_free_rate=0.02):
        self.tickers = tickers
        self.period = period
        self.risk_free_rate = risk_free_rate
        self.data, self.full_data = self.fetch_data()
        self.returns = self.calculate_returns()
        self.mean_returns = np.mean(self.returns, axis=0)
        self.cov_matrix = np.cov(self.returns, rowvar=False)
        self.trading_days_per_year = self._estimate_trading_days_per_year()

    def fetch_data(self):
        # if os.path.exists("stock_data.csv"):
        #     print("Loading data from CSV...")
        #     data = pd.read_csv("stock_data.csv", index_col=0, parse_dates=True)
        # else:
        print("Fetching data from Yahoo Finance...")
        #     data = yf.download(self.tickers, period=self.period)
        data = yf.download(self.tickers, period=self.period, group_by=None)
        close_data = data.xs(key="Close", axis=1, level=1)
        # Always work with Close prices - handle both single and multiple tickers
        logger.info(close_data)
        return close_data, data

    def _estimate_trading_days_per_year(self):
        """Estimate annual trading days based on actual data length and time period"""
        actual_days = len(self.returns)
        
        # Extract numeric part from period string (e.g., "2y" -> 2, "6mo" -> 0.5)
        if self.period.endswith('y'):
            years = float(self.period[:-1])
        elif self.period.endswith('mo'):
            years = float(self.period[:-2]) / 12
        elif self.period.endswith('d'):
            years = float(self.period[:-1]) / 365
        else:
            # Default fallback
            years = 1
        
        trading_days_per_year = actual_days / years if years > 0 else 252
        
        # Cap between reasonable bounds (account for holidays, weekends)
        trading_days_per_year = max(200, min(trading_days_per_year, 260))
        
        print(f"Estimated trading days per year: {trading_days_per_year:.0f}")
        return trading_days_per_year

    def save_to_csv(self, filepath):
        self.data.to_csv(filepath)

    def calculate_returns(self):
        returns = self.data.pct_change().dropna()
        return returns.values

    def portfolio_performance(self, weights):
        portfolio_return = np.sum(self.mean_returns * weights) * self.trading_days_per_year
        portfolio_stddev = np.sqrt(np.dot(weights.T, np.dot(self.cov_matrix * self.trading_days_per_year, weights)))
        sharpe = (portfolio_return - self.risk_free_rate) / portfolio_stddev
        return portfolio_return, portfolio_stddev, sharpe

    def random_portfolio(self):
        weights = np.random.random(len(self.tickers))
        weights /= np.sum(weights)
        return weights

    def print_info(self):
        print("Tickers:", self.tickers)
        print("Period:", self.period)
        print("Risk-free rate:", self.risk_free_rate)
        print("Data Head:\n", self.data.head())
        print("Data Tail:\n", self.data.tail())
        print("Returns Head:\n", self.returns[:5])
        print("Mean Returns:\n", self.mean_returns)
        print("Covariance Matrix:\n", self.cov_matrix)

if __name__ == "__main__":
    logger = logging.getLogger()
    logging.basicConfig(level=logging.INFO)
    # tickers = ['AAPL', 'GOOGL', 'MSFT', 'TSLA']
    tickers = ['AAPL', 'GOOGL', 'MSFT', '^SPX']
    analyzer = StockAnalyzer(tickers, period="1y", risk_free_rate=0.02)
    print(analyzer.print_info())
    for _ in range(3):
        weights = analyzer.random_portfolio()
        print("Random Portfolio Weights:", weights)
        performance = analyzer.portfolio_performance(weights)
        print("Random Portfolio Performance (Return, StdDev, Sharpe):", performance)
    for weights in multiset_permutations([0, 0, 0, 1]):
        weights = np.array(weights, dtype=float)
        print("Random Portfolio Weights:", weights)
        performance = analyzer.portfolio_performance(weights)
        print("Random Portfolio Performance (Return, StdDev, Sharpe):", performance)

    if not os.path.exists("stock_data.csv"):
        analyzer.save_to_csv("stock_data.csv")
