import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import logging

# Pymoo imports
from pymoo.problems import get_problem
from pymoo.indicators.hv import HV
from pymoo.indicators.gd import GD
from pymoo.indicators.gd_plus import GDPlus
from pymoo.indicators.igd import IGD
from pymoo.indicators.igd_plus import IGDPlus
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.optimize import minimize
from pymoo.termination import get_termination

# --- Setup Logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

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
        
        # Personal best memory for each agent
        self.best_pos = np.copy(self.pos)
        self.best_objectives = [np.inf, np.inf]
        
        self.crowding_distance = 0

class MO_BPSO_Optimizer:
    """
    Multi-Objective Bio-Physical Swarm Optimization (MO-BPSO).
    Now compatible with pymoo problems and performance indicators.
    """
    def __init__(self, problem, n_agents=100, archive_size=100, w=0.5, c1=1.5, c2=1.5, 
                 initial_temp=1.0, annealing_rate=0.995):
        self.problem = problem
        self.dim = problem.n_var
        self.n_obj = problem.n_obj
        
        # Get bounds from problem
        self.bounds_low = problem.xl
        self.bounds_high = problem.xu
        
        self.n_agents = n_agents
        self.archive_size = archive_size

        # Algorithm parameters
        self.w = w   # Inertia
        self.c1 = c1 # Cognitive component
        self.c2 = c2 # Social component
        self.T = initial_temp
        self.annealing_rate = annealing_rate

        # Initialization
        self.population = [Agent(self.dim, (self.bounds_low, self.bounds_high)) for _ in range(n_agents)]
        
        # Evaluate initial population
        for agent in self.population:
            X = agent.pos.reshape(1, -1)
            F = self.problem.evaluate(X)
            agent.objectives = F[0]
            agent.best_objectives = agent.objectives.copy()
        
        self.archive = []
        self._update_archive()
        
        # Performance tracking
        self.history = {
            'hypervolume': [],
            'gd': [],
            'gd_plus': [],
            'igd': [],
            'igd_plus': [],
            'n_archive': []
        }

    def _update_archive(self):
        """Updates the archive with non-dominated solutions from the current population."""
        combined = self.population + self.archive
        
        # Get objectives for non-dominated sorting
        F_combined = np.array([agent.objectives for agent in combined])
        
        # Use pymoo's non-dominated sorting
        nds = NonDominatedSorting()
        fronts = nds.do(F_combined)
        
        # Take only the first front (non-dominated solutions)
        if len(fronts) > 0:
            non_dominated_indices = fronts[0]
            non_dominated_agents = [combined[i] for i in non_dominated_indices]
            
            # Remove duplicates based on position
            unique_agents = []
            added_positions = set()
            for agent in non_dominated_agents:
                pos_tuple = tuple(np.round(agent.pos, decimals=8))  # Round to avoid floating point issues
                if pos_tuple not in added_positions:
                    unique_agents.append(agent)
                    added_positions.add(pos_tuple)
            
            # Truncate archive if necessary
            if len(unique_agents) > self.archive_size:
                calculate_crowding_distance(unique_agents)
                unique_agents.sort(key=lambda x: x.crowding_distance, reverse=True)
                self.archive = unique_agents[:self.archive_size]
            else:
                self.archive = unique_agents
            
            calculate_crowding_distance(self.archive)
        else:
            self.archive = []

    def _select_leader(self):
        """Selects a leader from the archive using tournament selection based on crowding distance."""
        if not self.archive:
            return self.population[np.random.randint(len(self.population))]
        
        i1, i2 = np.random.randint(0, len(self.archive), 2)
        candidate1 = self.archive[i1]
        candidate2 = self.archive[i2]
        
        return candidate1 if candidate1.crowding_distance > candidate2.crowding_distance else candidate2

    def step(self):
        """Performs one iteration of the MO-BPSO algorithm."""
        for agent in self.population:
            leader = self._select_leader()
            
            # PSO-style movement equation
            r1, r2 = np.random.rand(2)
            inertia = self.w * agent.velocity
            cognitive_pull = self.c1 * r1 * (agent.best_pos - agent.pos)
            social_pull = self.c2 * r2 * (leader.pos - agent.pos)
            
            agent.velocity = inertia + cognitive_pull + social_pull
            agent.pos += agent.velocity
            
            # Enforce boundaries
            agent.pos = np.clip(agent.pos, self.bounds_low, self.bounds_high)
            
            # Evaluate new position
            X = agent.pos.reshape(1, -1)
            F = self.problem.evaluate(X)
            agent.objectives = F[0]

            # Update Personal Best
            if dominates(agent.objectives, agent.best_objectives):
                agent.best_pos = np.copy(agent.pos)
                agent.best_objectives = agent.objectives.copy()
            elif not dominates(agent.best_objectives, agent.objectives):
                if np.random.rand() < 0.5:
                    agent.best_pos = np.copy(agent.pos)
                    agent.best_objectives = agent.objectives.copy()

        self._update_archive()
        self.T *= self.annealing_rate
        return self.population, self.archive
    
    def calculate_performance_metrics(self):
        """Calculate various performance indicators using pymoo."""
        if not self.archive:
            return None
            
        # Get current Pareto front
        F_archive = np.array([agent.objectives for agent in self.archive])
        
        # Get true Pareto front for comparison
        pf = self.problem.pareto_front()
        
        if pf is None:
            logging.warning("True Pareto front not available for this problem")
            return None
        
        metrics = {}
        
        try:
            # Hypervolume (higher is better)
            ref_point = np.max(pf, axis=0) * 1.1  # Reference point slightly beyond true PF
            hv = HV(ref_point=ref_point)
            metrics['hypervolume'] = hv(F_archive)
            
            # Generational Distance (lower is better)
            gd = GD(pf)
            metrics['gd'] = gd(F_archive)
            
            # GD+ (lower is better)  
            gd_plus = GDPlus(pf)
            metrics['gd_plus'] = gd_plus(F_archive)
            
            # Inverted Generational Distance (lower is better)
            igd = IGD(pf)
            metrics['igd'] = igd(F_archive)
            
            # IGD+ (lower is better)
            igd_plus = IGDPlus(pf)
            metrics['igd_plus'] = igd_plus(F_archive)
            
            metrics['n_archive'] = len(self.archive)
            
        except Exception as e:
            logging.warning(f"Error calculating metrics: {e}")
            return None
            
        return metrics

# --- Visualization with Performance Tracking ---
def visualize_mo_bpso_with_metrics(problem_name='zdt1', n_iterations=200):
    """
    Runs and visualizes the MO-BPSO algorithm with performance metrics.
    
    Parameters:
    -----------
    problem_name : str
        Name of the pymoo problem to solve (e.g., 'zdt1', 'zdt2', 'dtlz1')
    n_iterations : int
        Number of iterations to run
    """
    
    # Load problem from pymoo
    problem = get_problem(problem_name)
    optimizer = MO_BPSO_Optimizer(problem)
    
    # Setup visualization
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Left plot: Objective space
    ax1.set_xlabel('Objective 1 (f1)')
    ax1.set_ylabel('Objective 2 (f2)')
    ax1.set_title(f'MO-BPSO on {problem_name.upper()}: Pareto Front Evolution')
    ax1.grid(True)
    
    pop_scatter = ax1.scatter([], [], c='lightblue', label='Population', alpha=0.5, s=20)
    archive_scatter = ax1.scatter([], [], c='red', label='Pareto Front (Archive)', 
                                  edgecolors='darkred', s=40, alpha=0.8)

    # Plot true Pareto front if available
    pf = problem.pareto_front()
    if pf is not None:
        ax1.scatter(pf[:, 0], pf[:, 1], c='black', marker='x', 
                   label='True Pareto Front', s=30, alpha=0.7)
    
    ax1.legend()
    iter_text = ax1.text(0.02, 0.98, '', transform=ax1.transAxes, 
                        verticalalignment='top', fontsize=10)
    
    # Right plot: Performance metrics
    ax2.set_xlabel('Iteration')
    ax2.set_ylabel('Metric Value')
    ax2.set_title('Performance Metrics Over Time')
    ax2.grid(True)
    
    # Initialize metric plots
    iterations = []
    hv_line, = ax2.plot([], [], 'b-', label='Hypervolume', linewidth=2)
    gd_line, = ax2.plot([], [], 'r-', label='GD', linewidth=2)
    igd_line, = ax2.plot([], [], 'g-', label='IGD', linewidth=2)
    
    ax2.legend()
    ax2.set_yscale('log')  # Log scale for better visibility of different metrics

    def update(frame):
        population, archive = optimizer.step()
        
        # Update left plot
        if population:
            pop_obj = np.array([p.objectives for p in population])
            pop_scatter.set_offsets(pop_obj)
        
        if archive:
            archive_obj = np.array([a.objectives for a in archive])
            archive_scatter.set_offsets(archive_obj)
            
            # Calculate performance metrics
            metrics = optimizer.calculate_performance_metrics()
            if metrics:
                # Store metrics
                for key, value in metrics.items():
                    optimizer.history[key].append(value)
                
                # Update iteration counter
                iterations.append(frame + 1)
                
                # Update performance plots
                if len(iterations) > 1:
                    hv_line.set_data(iterations, optimizer.history['hypervolume'])
                    gd_line.set_data(iterations, optimizer.history['gd'])
                    igd_line.set_data(iterations, optimizer.history['igd'])
                    
                    # Auto-scale the performance plot
                    ax2.relim()
                    ax2.autoscale_view()
        
        # Update iteration text
        metrics_text = ""
        if archive and optimizer.history['hypervolume']:
            current_metrics = optimizer.calculate_performance_metrics()
            if current_metrics:
                metrics_text = (f"HV: {current_metrics['hypervolume']:.4f} | "
                               f"GD: {current_metrics['gd']:.4f} | "
                               f"IGD: {current_metrics['igd']:.4f}")
        
        iter_text.set_text(f'Iteration: {frame+1} | Archive: {len(archive)}\n{metrics_text}')

        # Console logging
        if (frame + 1) % 20 == 0 and archive:
            logging.info(f"Iter {frame+1:3d} | Archive: {len(archive):3d} | {metrics_text}")

        return pop_scatter, archive_scatter, iter_text, hv_line, gd_line, igd_line

    ani = FuncAnimation(fig, update, frames=n_iterations, blit=False, interval=100, repeat=False)
    plt.tight_layout()
    plt.show()
    
    # Print final performance summary
    print_performance_summary(optimizer, problem_name)
    
    return optimizer

def print_performance_summary(optimizer, problem_name):
    """Print a comprehensive performance summary."""
    if not optimizer.archive or not optimizer.history['hypervolume']:
        print("No performance data available.")
        return
    
    final_metrics = optimizer.calculate_performance_metrics()
    if not final_metrics:
        print("Could not calculate final metrics.")
        return
    
    print(f"\n{'='*60}")
    print(f"FINAL PERFORMANCE SUMMARY - {problem_name.upper()}")
    print(f"{'='*60}")
    print(f"Final Archive Size: {len(optimizer.archive)}")
    print(f"Final Hypervolume: {final_metrics['hypervolume']:.6f}")
    print(f"Final GD: {final_metrics['gd']:.6f}")
    print(f"Final GD+: {final_metrics['gd_plus']:.6f}")
    print(f"Final IGD: {final_metrics['igd']:.6f}")
    print(f"Final IGD+: {final_metrics['igd_plus']:.6f}")
    print(f"{'='*60}")
    
    # Show best and worst performing iterations
    if len(optimizer.history['hypervolume']) > 10:
        best_hv_iter = np.argmax(optimizer.history['hypervolume']) + 1
        best_hv_value = max(optimizer.history['hypervolume'])
        
        best_igd_iter = np.argmin(optimizer.history['igd']) + 1
        best_igd_value = min(optimizer.history['igd'])
        
        print(f"Best Hypervolume: {best_hv_value:.6f} at iteration {best_hv_iter}")
        print(f"Best IGD: {best_igd_value:.6f} at iteration {best_igd_iter}")
        print(f"{'='*60}")

def compare_with_nsga2(problem_name='zdt1', n_evaluations=20000):
    """
    Compare MO-BPSO with NSGA-II on the same problem.
    """
    print(f"\n{'='*60}")
    print(f"COMPARISON: MO-BPSO vs NSGA-II on {problem_name.upper()}")
    print(f"{'='*60}")
    
    problem = get_problem(problem_name)
    
    # Run MO-BPSO
    print("Running MO-BPSO...")
    mo_bpso = MO_BPSO_Optimizer(problem, n_agents=100)
    n_iterations = n_evaluations // 100  # 100 agents per iteration
    
    for i in range(n_iterations):
        mo_bpso.step()
        if (i + 1) % 50 == 0:
            metrics = mo_bpso.calculate_performance_metrics()
            if metrics:
                print(f"  Iter {i+1:3d}: HV={metrics['hypervolume']:.4f}, IGD={metrics['igd']:.4f}")
    
    bpso_metrics = mo_bpso.calculate_performance_metrics()
    
    # Run NSGA-II
    print("\nRunning NSGA-II...")
    algorithm = NSGA2(pop_size=100)
    termination = get_termination("n_eval", n_evaluations)
    
    res = minimize(problem, algorithm, termination, verbose=False)
    
    # Calculate NSGA-II metrics
    pf = problem.pareto_front()
    if pf is not None and bpso_metrics is not None:
        ref_point = np.max(pf, axis=0) * 1.1
        
        hv = HV(ref_point=ref_point)
        nsga2_hv = hv(res.F)
        
        igd = IGD(pf)
        nsga2_igd = igd(res.F)
        
        print(f"\nFINAL COMPARISON RESULTS:")
        print(f"{'Algorithm':<12} {'Hypervolume':<12} {'IGD':<12} {'Archive Size':<12}")
        print(f"{'-'*50}")
        print(f"{'MO-BPSO':<12} {bpso_metrics['hypervolume']:<12.6f} {bpso_metrics['igd']:<12.6f} {len(mo_bpso.archive):<12}")
        print(f"{'NSGA-II':<12} {nsga2_hv:<12.6f} {nsga2_igd:<12.6f} {len(res.F):<12}")
        
        # Determine winner
        bpso_wins = 0
        if bpso_metrics['hypervolume'] > nsga2_hv:
            bpso_wins += 1
        if bpso_metrics['igd'] < nsga2_igd:
            bpso_wins += 1
            
        if bpso_wins >= 1:
            print(f"\n🏆 MO-BPSO shows competitive/better performance!")
        else:
            print(f"\n📊 NSGA-II shows better performance on these metrics.")
    
    return mo_bpso, res

if __name__ == '__main__':
    # Select problem to solve
    # Example list:
    # 'zdt1', 'zdt2', 'zdt3', 'zdt4', 'zdt6' # ZDT family
    # 'dtlz1', 'dtlz2', 'dtlz3', 'dtlz4', 'dtlz5', 'dtlz6', 'dtlz7' # DTLZ family
    # 'wfg1', 'wfg2', 'wfg3', 'wfg4', 'wfg5', 'wfg6', 'wfg7', 'wfg8', 'wfg9' # WFG family
    problem = 'dtlz2'

    # Run visualization with metrics
    optimizer = visualize_mo_bpso_with_metrics(problem, n_iterations=150)
    
    # Optional: Run comparison with NSGA-II
    print("\nWould you like to run a comparison with NSGA-II? This will take a moment...")
    mo_bpso, nsga2_result = compare_with_nsga2(problem, n_evaluations=15000)
