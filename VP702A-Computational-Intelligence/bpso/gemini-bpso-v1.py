#!/usr/bin/env python3

# Bio-Physical Swarm Optimization (BPSO) Algorithm
# This document outlines the conceptual design of the Bio-Physical Swarm Optimization (BPSO) algorithm, inspired by the physical communication methods observed in bacteria.



# function BPSO(objective_function, bounds, N, max_iter):
#   // INPUT
#   //    objective_function = the function to minimize
#   //    bounds = the search space boundaries
#   //    N = the number of agents (bacteria)
#   //    max_iter = maximum number of iterations
#   // OUTPUT
#   //    The best solution found

#   // Initialization
#   population = initialize_population(N, bounds)
#   global_best_pos = find_best(population, objective_function)
#   T = initial_temperature
# 
#   for iter from 1 to max_iter:
#     for each agent in population:
#       // Agent Update Sub-Process
#       fitness = objective_function(agent.pos)
#       stiffness = calculate_gradient_norm(agent.pos)
# 
#       if stiffness > STIFFNESS_THRESHOLD or is_near_global_best(agent, global_best_pos):
#         agent.state = EXPLOIT
#         // Refine position with a small random mutation
#         agent.pos += random_vector(small_magnitude)
#       else:
#         agent.state = EXPLORE
#         // Update velocity based on inertia, global pull, and thermal noise
#         inertia = W * agent.velocity
#         global_pull = C1 * rand() * (global_best_pos - agent.pos)
#         random_walk = T * random_vector(unit_magnitude)
#         agent.velocity = inertia + global_pull + random_walk
#         agent.pos += agent.velocity
# 
#       // Enforce bounds
#       agent.pos = clip_to_bounds(agent.pos, bounds)
# 
#     // Update Global Best and Temperature
#     current_best_pos = find_best(population, objective_function)
#     if objective_function(current_best_pos) < objective_function(global_best_pos):
#       global_best_pos = current_best_pos
# 
#     T = T * ANNEALING_RATE
# 
#   return global_best_pos

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

np.random.seed(42) # For reproducibility

# --- Objective Function and its Gradient ---
# Let's use a standard test function: the Rastrigin function.
# It's challenging due to its many local minima.
def rastrigin(x):
    """Rastrigin function for optimization testing."""
    A = 10
    return A * len(x) + np.sum(x**2 - A * np.cos(2 * np.pi * x), axis=0)

def rastrigin_gradient(x):
    """Gradient of the Rastrigin function."""
    return 2 * x + 2 * np.pi * 10 * np.sin(2 * np.pi * x)

class Agent:
    """Represents a single bacterium in the BPSO algorithm."""
    def __init__(self, bounds):
        self.dim = len(bounds[0])
        self.pos = np.random.uniform(bounds[0], bounds[1], self.dim)
        self.velocity = np.random.uniform(-1, 1, self.dim)
        self.fitness = np.inf
        self.state = 'EXPLORE'  # Can be 'EXPLORE' or 'EXPLOIT'

class BPSO_Optimizer:
    """
    Bio-Physical Swarm Optimization (BPSO) implementation.
    """
    def __init__(self, objective_fn, gradient_fn, bounds, n_agents=30,
                 w=0.5, c1=1.5, stiffness_threshold=10.0,
                 initial_temp=5.0, annealing_rate=0.99):
        self.objective_fn = objective_fn
        self.gradient_fn = gradient_fn
        self.bounds_low, self.bounds_high = np.array(bounds[0]), np.array(bounds[1])
        self.dim = len(self.bounds_low)

        # Algorithm parameters
        self.n_agents = n_agents
        self.w = w  # Inertia weight
        self.c1 = c1 # Global best pull
        self.stiffness_threshold = stiffness_threshold
        self.T = initial_temp
        self.annealing_rate = annealing_rate

        # Initialization
        self.population = [Agent(bounds) for _ in range(n_agents)]
        self.quorum_size = max(2, n_agents // 10)
        self.global_best_pos = self.population[0].pos.copy()
        self.global_best_fitness = np.inf
        self._update_global_best()

    def _update_global_best(self):
        """Finds and updates the best solution found so far."""
        for agent in self.population:
            agent.fitness = self.objective_fn(agent.pos)
            if agent.fitness < self.global_best_fitness:
                self.global_best_fitness = agent.fitness
                self.global_best_pos = agent.pos.copy()

    def step(self):
        """Performs one iteration of the BPSO algorithm."""
        for agent in self.population:
            stiffness = np.linalg.norm(self.gradient_fn(agent.pos))

            # --- State Transition Logic ---
            # Agents near the global best or in steep areas will exploit
            dist_to_best = np.linalg.norm(agent.pos - self.global_best_pos)
            if stiffness > self.stiffness_threshold or dist_to_best < 0.5:
                 agent.state = 'EXPLOIT'
            else:
                 agent.state = 'EXPLORE'

            # --- Movement Logic ---
            if agent.state == 'EXPLOIT':
                # Small, random local search
                noise = np.random.normal(0, 0.1, self.dim) # Smaller random vector
                agent.pos += noise
            else: # EXPLORE
                # PSO-like movement with thermal noise
                inertia = self.w * agent.velocity
                global_pull = self.c1 * np.random.rand() * (self.global_best_pos - agent.pos)
                random_walk = self.T * (np.random.rand(self.dim) * 2 - 1)
                agent.velocity = inertia + global_pull + random_walk
                agent.pos += agent.velocity
                print("Explore Move:")
                print(f"Inertia: {inertia}, Global Pull: {global_pull}, Random Walk: {random_walk}")
                print(f"Agent pos: {agent.pos}, Velocity: {agent.velocity}, State: {agent.state}")

            # --- Enforce boundaries ---
            agent.pos = np.clip(agent.pos, self.bounds_low, self.bounds_high)

        self._update_global_best()
        self.T *= self.annealing_rate # Anneal temperature
        
        # Return current state for visualization
        return self.population, self.global_best_pos, self.global_best_fitness

# --- Visualization ---
def visualize_bpso():
    """Runs and visualizes the BPSO algorithm on a 2D Rastrigin function."""
    bounds = ([-5.12], [5.12])
    if len(bounds[0]) != 2: # Forcing 2D for this visualization
        bounds = ([-5.12, -5.12], [5.12, 5.12])
        
    optimizer = BPSO_Optimizer(
        objective_fn=rastrigin,
        gradient_fn=rastrigin_gradient,
        bounds=bounds,
        n_agents=20, # Number of agents, more for better coverage
        initial_temp=5.0, # Starting temperature, lower for less exploration
        stiffness_threshold=15.0, # Gradient norm threshold, higher for more exploration
        annealing_rate=0.99, # Cooling rate, slower for more exploration
        w=0.5, # Inertia weight, smaller for more exploration
        c1=1.5 # Global best pull, stronger pull towards best
    )

    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Create background contour plot
    x = np.linspace(bounds[0][0], bounds[1][0], 200)
    y = np.linspace(bounds[0][1], bounds[1][1], 200)
    X, Y = np.meshgrid(x, y)
    Z = rastrigin(np.array([X, Y]))
    contour = ax.contourf(X, Y, Z, levels=50, cmap='viridis_r', alpha=0.7)
    fig.colorbar(contour, label='Fitness Value (lower is better)')

    # Initialize scatter plots for agents
    explore_scatter = ax.scatter([], [], c='cyan', label='Explore Agents', edgecolors='k', s=50, zorder=3)
    exploit_scatter = ax.scatter([], [], c='magenta', label='Exploit Agents', edgecolors='k', s=80, marker='*', zorder=3)
    best_scatter = ax.scatter([], [], c='red', label='Global Best', marker='X', s=200, edgecolors='k', zorder=4)

    ax.set_xlim(bounds[0][0], bounds[1][0])
    ax.set_ylim(bounds[0][1], bounds[1][1])
    ax.set_xlabel('X coordinate')
    ax.set_ylabel('Y coordinate')
    ax.set_title('Bio-Physical Swarm Optimization (BPSO)')
    ax.legend()
    fig.tight_layout()

    # Text for displaying iteration and fitness
    iter_text = ax.text(0.02, 0.95, '', transform=ax.transAxes, fontsize=12,
                        bbox=dict(boxstyle='round,pad=0.5', fc='wheat', alpha=0.7))


    def update(frame):
        population, best_pos, best_fitness = optimizer.step()

        explore_pos = np.array([p.pos for p in population if p.state == 'EXPLORE'])
        exploit_pos = np.array([p.pos for p in population if p.state == 'EXPLOIT'])

        if explore_pos.size > 0:
            explore_scatter.set_offsets(explore_pos)
        else:
            explore_scatter.set_offsets(np.empty((0, 2)))

        if exploit_pos.size > 0:
            exploit_scatter.set_offsets(exploit_pos)
        else:
            exploit_scatter.set_offsets(np.empty((0, 2)))

        best_scatter.set_offsets(best_pos)
        iter_text.set_text(f'Iteration: {frame+1}\nBest Fitness: {best_fitness:.4f}\nTemp: {optimizer.T:.2f}')

        return explore_scatter, exploit_scatter, best_scatter, iter_text

    ani = FuncAnimation(fig, update, frames=100, blit=True, interval=100, repeat=False)
    plt.show()


if __name__ == '__main__':
    print("Running BPSO visualization...")
    print("This will open a Matplotlib window showing the agents converging on the minimum of the Rastrigin function.")
    print("The optimal solution is at (0,0) with a fitness of 0.")
    visualize_bpso()

