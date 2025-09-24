import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import logging

# --- Setup Logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Multi-Objective Test Problem (ZDT1) ---
def zdt1(x):
    """
    Zitzler-Deb-Thiele's function N. 1 (ZDT1). A standard multi-objective benchmark.
    It has two objectives, both to be minimized.
    The Pareto-optimal front is convex.
    """
    if np.any(x < 0) or np.any(x > 1):
        return [np.inf, np.inf] # Penalize solutions outside the bounds [0, 1]

    f1 = x[0]
    g = 1 + 9 * np.sum(x[1:]) / (len(x) - 1)
    h = 1 - np.sqrt(f1 / g)
    f2 = g * h
    return [f1, f2]

# --- Helper Functions for Multi-Objective Logic ---
def dominates(p_objectives, q_objectives):
    """
    Checks if solution p dominates solution q.
    A solution p dominates q if it is not worse in any objective and strictly better in at least one.
    """
    p, q = np.array(p_objectives), np.array(q_objectives)
    return np.all(p <= q) and np.any(p < q)

def calculate_crowding_distance(front):
    """
    Calculates the crowding distance for each solution in a front.
    This is used to maintain diversity. Solutions in less crowded regions are preferred.
    """
    if len(front) <= 2:
        for agent in front:
            agent.crowding_distance = np.inf
        return

    for agent in front:
        agent.crowding_distance = 0

    num_objectives = len(front[0].objectives)
    for m in range(num_objectives):
        # Sort by the m-th objective
        front.sort(key=lambda agent: agent.objectives[m])
        
        # Assign infinite distance to boundary solutions
        front[0].crowding_distance = np.inf
        front[-1].crowding_distance = np.inf

        f_min = front[0].objectives[m]
        f_max = front[-1].objectives[m]
        
        if f_max == f_min:
            continue

        # Calculate for intermediate solutions
        for i in range(1, len(front) - 1):
            front[i].crowding_distance += (front[i+1].objectives[m] - front[i-1].objectives[m]) / (f_max - f_min)

class Agent:
    """Represents a single bacterium in the MO-BPSO algorithm."""
    def __init__(self, dim, bounds):
        self.dim = dim
        self.bounds_low, self.bounds_high = bounds
        self.pos = np.random.uniform(self.bounds_low, self.bounds_high, self.dim)
        self.velocity = np.zeros(self.dim)
        self.objectives = [np.inf, np.inf]
        self.state = 'EXPLORE'
        self.crowding_distance = 0

class MO_BPSO_Optimizer:
    """
    Multi-Objective Bio-Physical Swarm Optimization (MO-BPSO).
    Adapts BPSO for multi-objective problems by replacing the global best
    with an archive of non-dominated solutions and using crowding distance for diversity.
    """
    def __init__(self, objective_fn, dim=30, bounds=(0, 1), n_agents=100,
                 archive_size=100, w=0.5, c=2.0, initial_temp=1.0, annealing_rate=0.995):
        self.objective_fn = objective_fn
        self.dim = dim
        self.bounds_low, self.bounds_high = bounds
        self.n_agents = n_agents
        
        self.archive_size = archive_size
        self.archive = []

        # Algorithm parameters (Tuned for better exploration)
        self.w = w
        self.c = c
        self.T = initial_temp
        self.annealing_rate = annealing_rate

        # Initialization
        self.population = [Agent(dim, (self.bounds_low, self.bounds_high)) for _ in range(n_agents)]
        for agent in self.population:
            agent.objectives = self.objective_fn(agent.pos)
        self._update_archive()

    def _update_archive(self):
        """Updates the archive with non-dominated solutions from the current population."""
        # Combine current population and archive
        combined = self.population + self.archive
        
        # Find non-dominated solutions in the combined set
        # This is a computationally intensive step (O(M*(N+A)^2)) but necessary.
        non_dominated_front = []
        
        # Use a set of tuples to efficiently check for duplicate positions
        added_positions = set()

        for p in combined:
            is_dominated = False
            for q in combined:
                if p is q: continue
                if dominates(q.objectives, p.objectives):
                    is_dominated = True
                    break
            if not is_dominated:
                pos_tuple = tuple(p.pos)
                if pos_tuple not in added_positions:
                    non_dominated_front.append(p)
                    added_positions.add(pos_tuple)
        
        # If archive is too large, prune it using crowding distance
        if len(non_dominated_front) > self.archive_size:
            calculate_crowding_distance(non_dominated_front)
            non_dominated_front.sort(key=lambda x: x.crowding_distance, reverse=True)
            self.archive = non_dominated_front[:self.archive_size]
        else:
            self.archive = non_dominated_front
        
        # OPTIMIZATION: Calculate crowding distance once for the updated archive
        calculate_crowding_distance(self.archive)


    def _select_leader(self):
        """Selects a leader from the archive for an agent to follow."""
        if not self.archive:
            return self.population[np.random.randint(len(self.population))]
        
        # Tournament selection based on pre-calculated crowding distance to promote spread
        i1, i2 = np.random.randint(0, len(self.archive), 2)
        candidate1 = self.archive[i1]
        candidate2 = self.archive[i2]
        
        # The crowding distance is already calculated in _update_archive
        if candidate1.crowding_distance > candidate2.crowding_distance:
            return candidate1
        return candidate2

    def step(self):
        """Performs one iteration of the MO-BPSO algorithm."""
        for agent in self.population:
            leader = self._select_leader()
            
            # State Switching Logic: Agents close to the front should refine their position
            dist_to_leader = np.linalg.norm(agent.pos - leader.pos)
            if dist_to_leader < 0.1 * self.dim: # Heuristic threshold
                agent.state = 'EXPLOIT'
            else:
                agent.state = 'EXPLORE'
            
            # Movement Logic
            if agent.state == 'EXPLOIT':
                # Local refinement search
                noise = np.random.normal(0, 0.05, self.dim) # small random search
                agent.pos += noise
            else: # EXPLORE
                # Movement towards a diverse leader + thermal noise
                inertia = self.w * agent.velocity
                pull_to_leader = self.c * np.random.rand() * (leader.pos - agent.pos)
                random_walk = self.T * (np.random.rand(self.dim) * 2 - 1) * 0.1 # Scaled down thermal noise
                agent.velocity = inertia + pull_to_leader + random_walk
                agent.pos += agent.velocity
            
            # Enforce boundaries
            agent.pos = np.clip(agent.pos, self.bounds_low, self.bounds_high)
            agent.objectives = self.objective_fn(agent.pos)

        self._update_archive()
        self.T *= self.annealing_rate
        return self.population, self.archive

# --- Visualization ---
def visualize_mo_bpso():
    """Runs and visualizes the MO-BPSO algorithm on the ZDT1 problem."""
    optimizer = MO_BPSO_Optimizer(objective_fn=zdt1)
    
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.set_xlim(0, 1.1)
    ax.set_ylim(0, 1.1)
    ax.set_xlabel('Objective 1 (f1)')
    ax.set_ylabel('Objective 2 (f2)')
    ax.set_title('MO-BPSO: Population and Pareto Front')
    ax.grid(True)
    
    pop_scatter = ax.scatter([], [], c='lightblue', label='Population')
    archive_scatter = ax.scatter([], [], c='red', label='Pareto Front (Archive)', edgecolors='k')

    # Plot the true Pareto front for ZDT1 for reference
    x_true = np.linspace(0, 1, 100)
    y_true = 1 - np.sqrt(x_true)
    ax.plot(x_true, y_true, 'k--', label='True Pareto Front')
    ax.legend()
    
    iter_text = ax.text(0.65, 1.01, '', transform=ax.transAxes)

    def update(frame):
        population, archive = optimizer.step()
        
        pop_obj = np.array([p.objectives for p in population])
        pop_scatter.set_offsets(pop_obj)
        
        if archive:
            archive_obj = np.array([a.objectives for a in archive])
            archive_scatter.set_offsets(archive_obj)
        
        iter_text.set_text(f'Iteration: {frame+1}, Archive Size: {len(archive)}')

        # --- ADDED: Logging to console ---
        if (frame + 1) % 10 == 0 and archive:
            avg_f1 = np.mean([a.objectives[0] for a in archive])
            avg_f2 = np.mean([a.objectives[1] for a in archive])
            logging.info(f"Iter {frame+1:3d} | Archive: {len(archive):3d} | Avg F1: {avg_f1:.4f} | Avg F2: {avg_f2:.4f}")

        return pop_scatter, archive_scatter, iter_text

    # The only change is setting blit=False to ensure the plot renders correctly.
    ani = FuncAnimation(fig, update, frames=100, blit=False, interval=50, repeat=False)
    plt.show()

    # --- ADDED: Print final best solutions ---
    final_archive = optimizer.archive
    if final_archive:
        # Sort by crowding distance to show the most diverse solutions first
        final_archive.sort(key=lambda x: x.crowding_distance, reverse=True)
        print("\n--- Top 10 Solutions from Final Pareto Front ---")
        for i, agent in enumerate(final_archive[:10]):
            print(f"  {i+1:2d}. Objectives: [{agent.objectives[0]:.4f}, {agent.objectives[1]:.4f}] | Crowding Distance: {agent.crowding_distance:.4f}")

if __name__ == '__main__':
    visualize_mo_bpso()


