

import random
import matplotlib.pyplot as plt
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
#
# Psuedo code from https://www.baeldung.com/cs/tabu-search
#
# algorithm TabuSearch(S, maxIter, f, neighborhoods, aspirationCriteria):
#     // INPUT
#     //    S = the search space
#     //    maxIter = the maximal number of iterations
#     //    f = the objective function
#     //    neighborhoods = the definition of neighborhoods
#     //    aspirationCriteria = the aspiration criteria
#     // OUTPUT
#     //    The best solution found
# 
#     Choose the initial candidate solution s in S
#     s* <- s  // Initialize the best-so-far solution
#     k <- 1
# 
#     while k <= maxIter:
#         // Sample the allowed neighbors of s
#         Generate a sample V(s, k) of the allowed solutions in N(s, k)
#         // s' in V(s, k) iff (s' not in T) or (a(k, s') = true)
# 
#         Set s to the best solution in V(s, k)
# 
#         // Update the best-so-far if necessary
#         if f(s) < f(s*):
#             s* <- s
# 
#         Update T and a
# 
#         // Start another iteration
#         k <- k + 1
# 
#     return s*

class TabuSearch:
    """
    Tabu Search for portfolio optimization.
    https://www.baeldung.com/cs/tabu-search
    """
    def __init__(self, stock_analyzer, max_iter=100, tabu_size=10):
        self.stock_analyzer = stock_analyzer
        self.max_iter = max_iter
        self.tabu_size = tabu_size
        self.tabu_list = deque(maxlen=tabu_size)
        self.best_solution = None
        self.best_performance = (-np.inf, np.inf, -np.inf)
        self.performance_history = []
        self.best_performance_history = []

    def initialize_solution(self):
        """
        Initialize a random solution (portfolio weights).
        """
        for _ in range(1000):
            weights = self.stock_analyzer.random_portfolio()
            if self.constraint_satisfied(weights):
                logger.info(f"Initial weights: {weights}")
                return weights
        # Fallback: create equal weights and adjust to satisfy constraints
        weights = np.ones(len(self.stock_analyzer.tickers)) / len(self.stock_analyzer.tickers)
        logger.warning(f"Using fallback equal weights: {weights}")
        return weights

    def random_restart(self):
        """Generate a random feasible portfolio to escape local optima."""
        for _ in range(1000):
            weights = self.stock_analyzer.random_portfolio()
            if self.constraint_satisfied(weights):
                logger.info("Random restart successful.")
                return weights
        return self.initialize_solution()

    def objective_function(self, weights):
        return self.stock_analyzer.portfolio_performance(weights)

    def constraint_satisfied(self, weights):
        """Check if portfolio weights satisfy constraints."""
        # Ensure weights sum to 1 (with small tolerance)
        weights = np.round(weights, 4)
        if not np.isclose(np.sum(weights), 1.0, atol=1e-4):
            logger.debug(f"Weights {weights} don't sum to 1 (sum={np.sum(weights):.6f})")
            return False
        # Ensure all weights are non-negative
        if np.any(weights < 0):
            logger.debug(f"Weights {weights} contain negative values")
            return False
        # Constraint 1: Maximum 40% in a single asset
        if np.any(weights > 0.4):
            logger.debug(f"Weights {weights} violate max 60% constraint")
            return False
        # Constraint 2: At least 5% in at least half the assets
        active_assets = np.sum(weights >= 0.05)
        if active_assets < len(weights) // 2:
            return False
        # Constraint 4: Herfindahl-Hirschman Index (HHI)
        hhi = np.sum(weights ** 2)
        if hhi > 0.20:  # penalize concentration
            logger.debug(f"Weights {weights} violate HHI constraint (HHI={hhi:.4f})")
            return False
        return True


    def search(self):
        current_solution = self.initialize_solution()
        self.best_solution = current_solution
        self.best_performance = self.objective_function(current_solution)
        
        self.performance_history = []
        self.best_performance_history = []
        no_improve_count = 0

        for i in range(self.max_iter):
            logger.info(f"Iteration {i+1}/{self.max_iter}")
            neighbors = self.get_neighbors(current_solution)
            best_neighbor = None
            best_neighbor_performance = (-np.inf, np.inf, -np.inf)

            logger.debug(f"Current Solution: {current_solution}, Performance: {self.objective_function(current_solution)}")
            logger.debug(f"Tabu List: {self.tabu_list}")
            logger.debug(f"Evaluating {len(neighbors)} neighbors: {neighbors}")
            for neighbor, move in neighbors:
                performance = self.objective_function(neighbor)
                logger.debug(f"Evaluating neighbor: {neighbor}, Performance: {performance}, Compared to best: {best_neighbor_performance}")
                # if move in self.tabu_list and performance[2] <= self.best_performance[2]:
                logger.debug(f"Tabu check for move {move} in tabu list: {self.tabu_list}")
                if move in self.tabu_list and move != 'random':
                    logger.debug(f"Neighbor {neighbor} is in Tabu list, skipping.")
                    continue
                if not self.constraint_satisfied(neighbor):
                    logger.debug(f"Neighbor {neighbor} does not satisfy constraints, skipping.")
                    continue
                logger.debug(f"Neighbor weights: {neighbor}, Performance: {performance}")
                if performance[2] > best_neighbor_performance[2]:
                    best_neighbor = neighbor
                    best_neighbor_performance = performance
                    best_move = move
            if best_neighbor is None:
                logger.warning("No valid neighbors found, performing random restart.")
                current_solution = self.random_restart()
                continue

            if best_neighbor is not None:
                # update tabu list and current solution
                current_solution = best_neighbor
                self.tabu_list.append(best_move)

            # Track performance history
            self.performance_history.append(best_neighbor_performance[2])

            # update global best
            if best_neighbor_performance[2] > self.best_performance[2]:
                self.best_solution = current_solution
                self.best_performance = best_neighbor_performance
                logger.info(f"Current Best Weights: {self.best_solution}, With Return: {self.best_performance[0]:.2%}, StdDev: {self.best_performance[1]:.2%}, Sharpe: {self.best_performance[2]:.4f}")
            else:
                no_improve_count += 1

            self.best_performance_history.append(self.best_performance[2])

            if no_improve_count >= 10:
                current_solution = self.random_restart()
                logger.warning(f"No improvement on iteration {i}, restarting with weights: {current_solution}")
                no_improve_count = 0

        logger.debug(f"Tabu List: {self.tabu_list}")
        logger.info(f"Best Solution: {self.best_solution}")
        return self.best_solution, self.best_performance


    def get_neighbors(self, current_solution, step_size=0.05, num_random=2):
        neighbors = []
        num_assets = len(current_solution)

        # Local perturbations
        for i in range(num_assets):
            for delta in [-step_size, step_size]:
                neighbor = current_solution.copy()
                neighbor[i] += delta
                if 0 <= neighbor[i] <= 1:
                    neighbor /= np.sum(neighbor)
                    move = (i, delta)
                    neighbors.append((neighbor, move))

        # Swap moves (shift allocation between two assets)
        for i in range(num_assets):
            for j in range(i+1, num_assets):
                neighbor = current_solution.copy()
                transfer = step_size
                if neighbor[i] >= transfer:
                    neighbor[i] -= transfer
                    neighbor[j] += transfer
                    move = (i, j, transfer)
                    neighbors.append((neighbor, move))

        # Random reallocation moves
        for _ in range(num_random):
            neighbor = self.stock_analyzer.random_portfolio()
            neighbors.append((neighbor, 'random'))

        logger.info(f"Generated {len(neighbors)} neighbors")
        return neighbors

    def plot_optimization_progress(self):
        """Plot the optimization progress."""
        fig, axs = plt.subplots(2, 2, figsize=(15, 15))
        
        # Current performance vs iteration
        axs[0, 0].plot(self.performance_history, 'b-', alpha=0.7, label='Current Sharpe')
        axs[0, 0].plot(self.best_performance_history, 'r-', linewidth=2, label='Best Sharpe')
        axs[0, 0].set_xlabel('Iteration')
        axs[0, 0].set_ylabel('Sharpe Ratio')
        axs[0, 0].set_title('Optimization Progress')
        axs[0, 0].legend()
        axs[0, 0].grid(True, alpha=0.3)
        
        # Distribution of performance values
        axs[0, 1].hist(self.performance_history, bins=20, alpha=0.7, edgecolor='black')
        axs[0, 1].axvline(self.best_performance[2], color='red', linestyle='--', linewidth=2, label=f'Best: {self.best_performance[2]:.4f}')
        axs[0, 1].set_xlabel('Sharpe Ratio')
        axs[0, 1].set_ylabel('Frequency')
        axs[0, 1].set_title('Distribution of Explored Solutions')
        axs[0, 1].legend()
        axs[0, 1].grid(True, alpha=0.3)

        # Efficient frontier comparison
# def plot_efficient_frontier(tickers, best_performance, analyzer, fund_names):
        random_returns = []
        random_risks = []
        better_returns = []
        better_risks = []
        for _ in range(5000):
            # use random restart to ensure feasibility
            w = self.random_restart()
            r, v, s = analyzer.portfolio_performance(w)
            if s > best_performance[2]:
                better_returns.append(r)
                better_risks.append(v)
            else:
                random_returns.append(r)
                random_risks.append(v)

        axs[1, 0].scatter(random_risks, random_returns, c="lightgray", label="Random Portfolios")
        axs[1, 0].scatter(better_risks, better_returns, c="blue", marker="P", label="Random Better Portfolios")
        axs[1, 0].scatter(best_performance[1], best_performance[0], c="red", marker="*", s=150, label="Best Tabu Solution")
        axs[1, 0].set_xlabel("Risk (StdDev)")
        axs[1, 0].set_ylabel("Return")
        axs[1, 0].set_title("Efficient Frontier (Random vs. Tabu Search)")
        axs[1, 0].legend()
        axs[1, 0].grid(True, alpha=0.3)

        # Final allocation pie chart
# def plot_final_allocation(tickers, best_weights, fund_names):
        axs[1, 1].pie(best_weights, labels=[fund_names.get(t, t) for t in tickers], autopct='%1.1f%%', startangle=140)
        axs[1, 1].set_title("Best Portfolio Allocation")
        
        plt.suptitle("Tabu Search Portfolio Optimization Results", fontsize=18, y=1.02)

        plt.tight_layout()
        plt.show()

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
        data = yf.download(self.tickers, period=self.period, group_by='ticker')
        close_data = data.xs(key="Close", axis=1, level=1)
        close_data = close_data.reindex(columns=self.tickers)
        # Always work with Close prices - handle both single and multiple tickers
        logger.info(f"Fetched data shape: {close_data.shape}")
        logger.info(f"Original columns: {list(data.xs('Close', axis=1, level=1).columns) if len(self.tickers) > 1 else 'Single ticker'}")
        logger.info(f"Reordered columns: {list(close_data.columns)}")
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
        weights = np.array(weights)
        portfolio_return = np.dot(self.mean_returns, weights) * self.trading_days_per_year
        portfolio_stddev = np.sqrt(np.dot(weights.T, np.dot(self.cov_matrix * self.trading_days_per_year, weights)))
        sharpe = (portfolio_return - self.risk_free_rate) / portfolio_stddev
        return round(portfolio_return, 4), round(portfolio_stddev, 4), round(sharpe, 4)

    def random_portfolio(self):
        weights = np.random.random(len(self.tickers))
        weights /= np.sum(weights)
        return weights

    def print_info(self):
        print("Tickers:", self.tickers)
        print("Period:", self.period)
        print("Risk-free rate:", self.risk_free_rate)
        print("Trading days per year:", self.trading_days_per_year)
        print("Data shape:", self.data.shape)
        print("Data columns order:", list(self.data.columns))
        print("Returns shape:", self.returns.shape)
        print("Mean Returns (annualized):")
        for i, ticker in enumerate(self.tickers):
            print(f"  {ticker}: {self.mean_returns[i] * self.trading_days_per_year:.2%}")
        print("Column order verification:", list(self.data.columns) == self.tickers)



if __name__ == "__main__":
    logger = logging.getLogger()
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(funcName)s : %(message)s', level=logging.DEBUG, filename='tabu_search.log', filemode='w',)
    # tickers = ['AAPL', 'GOOGL', 'MSFT', '^SPX', 'TSLA', 'AMZN', 'FB', 'NVDA']
    tickers = [
            '0P0001BM0U.ST', '0P0001H4TL.ST', '0P00000LF6.ST',
            '0P00000LEY.ST', '0P00005U1J.ST', '0P0001ECQR.ST',
            '0P0000ULAP.ST', '0P0001BM0Y.ST', '0P0001K2MJ.ST'
               ]
    fund_names = {
        '0P0001BM0U.ST': "Avanza Auto 2",
        '0P0001H4TL.ST': "Avanza Emerging Markets", 
        '0P0001BMN5.ST': "Länsförsäkringar Global Index",
        '0P00000LF6.ST': "Swedbank Robur Small Cap Global A",
        '0P0001QVRY.F': "Swedbank Robur Emerging Europe C", # Note: frankfurt traded fund
        '0P00000LEY.ST': "Swedbank Robur Småbolagsfond Sverige A",
        '0P00005U1J.ST': "Avanza Zero",
        '0P0001ECQR.ST': "Avanza Global",
        '0P0001J6WY.ST': "Avanza Europa",
        '0P0000ULAP.ST': "Spiltan Aktiefond Investmentbolag",
        '0P0001BM0Y.ST': "Avanza Auto 6",
        '0P0000X5RL.ST': "Avanza 75",
        '0P0001K2MJ.ST': "Avanza World Tech by TIN"
    }
    analyzer = StockAnalyzer(tickers, period="2y", risk_free_rate=0.05)
    print(analyzer.print_info())
    # for weights in multiset_permutations([0, 0, 0, 1]):
    #     weights = np.array(weights, dtype=float)
    #     print("Random Portfolio Weights:", weights)
    #     performance = analyzer.portfolio_performance(weights)
    #     print("Random Portfolio Performance (Return, StdDev, Sharpe):", performance)

    for run in range(5):
        print(f"\nRun {run + 1}:")
        print('\n' + "x"*50 + '\n')
        
        tabu_search = TabuSearch(analyzer, max_iter=500, tabu_size=20)
        best_weights, best_performance = tabu_search.search()
        
        print(f"Best Portfolio Return: {best_performance[0]:.2%}")
        print(f"Best Portfolio StdDev: {best_performance[1]:.2%}")  
        print(f"Best Portfolio Sharpe Ratio: {best_performance[2]:.4f}")
        print("\nBest Portfolio Allocation:")
        
        for ticker, weight in zip(tickers, best_weights):
            name = fund_names.get(ticker, ticker)
            print(f"  {name}: {weight:.2%}")
        # for _ in range(3):
        #     weights = analyzer.random_portfolio()
        #     print("Random Portfolio Weights:", weights)
        #     performance = analyzer.portfolio_performance(weights)
        #     print("Random Portfolio Performance (Return, StdDev, Sharpe):", performance)

        tabu_search.plot_optimization_progress()
        np.random.seed(42 + run)
