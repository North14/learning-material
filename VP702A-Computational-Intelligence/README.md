## Tabu Search
- [Tabu Search](#tabu-search/tabu-search.py)

Demonstration of Tabu Search algorithm for finding optimal portfolio allocation.
Uses historical stock data to evaluate portfolio performance.
Objective is defined as the maximization of the Sharpe ratio.

Tabu search is a metaheuristic optimization algorithm that guides a local search procedure.
The implementation uses a tabu list to avoid cycles and encourages exploration of the solution space.
Exploration beyond the local optimum is acheived by the restarting the search from a new random solution after 10 iterations without improvement.
The algorithm iteratively explores neighboring solutions by making small adjustments to the current portfolio allocation.
Additionally, the algorithm explores by swapping allocations between two randomly selected assets.
