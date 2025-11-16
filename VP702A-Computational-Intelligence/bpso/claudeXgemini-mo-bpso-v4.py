import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from matplotlib.animation import FuncAnimation
import logging
import time
from collections import defaultdict

# Pymoo imports
from pymoo.problems import get_problem
from pymoo.indicators.hv import HV
from pymoo.indicators.gd import GD
from pymoo.indicators.gd_plus import GDPlus
from pymoo.indicators.igd import IGD
from pymoo.indicators.igd_plus import IGDPlus
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.algorithms.moo.nsga3 import NSGA3
from pymoo.algorithms.moo.moead import MOEAD
from pymoo.optimize import minimize
from pymoo.termination import get_termination
from pymoo.util.ref_dirs import get_reference_directions

# --- Setup Logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Helper Functions for Multi-Objective Logic ---
def dominates(p_objectives, q_objectives):
    """Checks if solution p dominates solution q."""
    p, q = np.array(p_objectives), np.array(q_objectives)
    return np.all(p <= q) and np.any(p < q)

def calculate_crowding_distance(front):
    """Calculates the crowding distance for each solution in a front."""
    if len(front) <= 2:
        for agent in front:
            agent.crowding_distance = np.inf
        return

    for agent in front:
        agent.crowding_distance = 0

    num_objectives = len(front[0].objectives)
    for m in range(num_objectives):
        front.sort(key=lambda agent: agent.objectives[m])
        
        front[0].crowding_distance = np.inf
        front[-1].crowding_distance = np.inf

        f_min = front[0].objectives[m]
        f_max = front[-1].objectives[m]
        
        if f_max == f_min:
            continue

        for i in range(1, len(front) - 1):
            front[i].crowding_distance += (front[i+1].objectives[m] - front[i-1].objectives[m]) / (f_max - f_min)

class Agent:
    """Represents a single agent in the MO-BPSO algorithm."""
    def __init__(self, dim, bounds):
        self.dim = dim
        self.bounds_low, self.bounds_high = bounds
        self.pos = np.random.uniform(self.bounds_low, self.bounds_high, self.dim)
        self.velocity = np.zeros(self.dim)
        self.objectives = None
        
        self.best_pos = np.copy(self.pos)
        self.best_objectives = None
        
        self.crowding_distance = 0
        self.stuck_counter = 0  # Track stagnation

class MO_BPSO_Optimizer:
    """Multi-Objective Bio-Physical Swarm Optimization with detailed performance tracking."""
    
    def __init__(self, problem, n_agents=100, archive_size=100, w=0.5, c1=1.5, c2=1.5, 
                 initial_temp=1.0, annealing_rate=0.995, adaptive_params=True):
        self.problem = problem
        self.dim = problem.n_var
        self.n_obj = problem.n_obj
        
        self.bounds_low = problem.xl
        self.bounds_high = problem.xu
        
        self.n_agents = n_agents
        self.archive_size = archive_size

        # Algorithm parameters
        self.w_initial = w
        self.c1_initial = c1
        self.c2_initial = c2
        self.w = w
        self.c1 = c1
        self.c2 = c2
        self.T = initial_temp
        self.annealing_rate = annealing_rate
        self.adaptive_params = adaptive_params

        # Performance tracking
        self.population = [Agent(self.dim, (self.bounds_low, self.bounds_high)) for _ in range(n_agents)]
        
        # Evaluate initial population
        self.evaluations = 0
        for agent in self.population:
            agent.objectives = self._evaluate(agent.pos)
            agent.best_objectives = agent.objectives.copy()
        
        self.archive = []
        self._update_archive()
        
        # Detailed performance tracking
        self.history = {
            'hypervolume': [],
            'gd': [],
            'gd_plus': [],
            'igd': [],
            'igd_plus': [],
            'n_archive': [],
            'convergence_rate': [],
            'diversity_metric': [],
            'stagnation_count': [],
            'parameter_w': [],
            'parameter_c1': [],
            'parameter_c2': [],
            'best_solutions_per_iter': [],
            'evaluation_count': []
        }
        
        self.stagnation_threshold = 10
        self.last_best_hv = 0
        self.stagnation_counter = 0

    def _evaluate(self, pos):
        """Evaluate a solution and increment evaluation counter."""
        X = pos.reshape(1, -1)
        F = self.problem.evaluate(X)
        self.evaluations += 1
        return F[0]

    def _update_archive(self):
        """Updates the archive with non-dominated solutions."""
        combined = self.population + self.archive
        
        F_combined = np.array([agent.objectives for agent in combined])
        
        nds = NonDominatedSorting()
        fronts = nds.do(F_combined)
        
        if len(fronts) > 0:
            non_dominated_indices = fronts[0]
            non_dominated_agents = [combined[i] for i in non_dominated_indices]
            
            # Remove duplicates
            unique_agents = []
            added_positions = set()
            for agent in non_dominated_agents:
                pos_tuple = tuple(np.round(agent.pos, decimals=8))
                if pos_tuple not in added_positions:
                    unique_agents.append(agent)
                    added_positions.add(pos_tuple)
            
            if len(unique_agents) > self.archive_size:
                calculate_crowding_distance(unique_agents)
                unique_agents.sort(key=lambda x: x.crowding_distance, reverse=True)
                self.archive = unique_agents[:self.archive_size]
            else:
                self.archive = unique_agents
            
            calculate_crowding_distance(self.archive)
        else:
            self.archive = []

    def _adaptive_parameter_update(self):
        """Update algorithm parameters based on performance."""
        if not self.adaptive_params or len(self.history['hypervolume']) < 5:
            return
        
        # Check for stagnation
        current_hv = self.history['hypervolume'][-1] if self.history['hypervolume'] else 0
        
        if abs(current_hv - self.last_best_hv) < 1e-6:
            self.stagnation_counter += 1
        else:
            self.stagnation_counter = 0
            self.last_best_hv = current_hv
        
        # Adapt parameters based on stagnation
        if self.stagnation_counter > self.stagnation_threshold:
            # Increase exploration
            self.w = min(0.9, self.w * 1.1)
            self.c1 = min(2.5, self.c1 * 1.05)
            self.c2 = max(0.5, self.c2 * 0.95)
        else:
            # Balance exploration and exploitation
            self.w = max(0.1, self.w * 0.99)
            self.c1 = max(0.5, self.c1 * 0.99)
            self.c2 = min(2.5, self.c2 * 1.01)

    def _select_leader(self):
        """Selects a leader from the archive using tournament selection."""
        if not self.archive:
            return self.population[np.random.randint(len(self.population))]
        
        # Tournament selection with crowding distance
        tournament_size = min(3, len(self.archive))
        candidates = np.random.choice(self.archive, tournament_size, replace=False)
        
        return max(candidates, key=lambda x: x.crowding_distance)

    def step(self):
        """Performs one iteration of the MO-BPSO algorithm."""
        for agent in self.population:
            leader = self._select_leader()
            
            # PSO movement with adaptive parameters
            r1, r2 = np.random.rand(2)
            inertia = self.w * agent.velocity
            cognitive_pull = self.c1 * r1 * (agent.best_pos - agent.pos)
            social_pull = self.c2 * r2 * (leader.pos - agent.pos)
            
            agent.velocity = inertia + cognitive_pull + social_pull
            
            # Velocity clamping based on temperature
            v_max = (self.bounds_high - self.bounds_low) * self.T * 0.1
            agent.velocity = np.clip(agent.velocity, -v_max, v_max)
            
            agent.pos += agent.velocity
            agent.pos = np.clip(agent.pos, self.bounds_low, self.bounds_high)
            
            # Evaluate new position
            agent.objectives = self._evaluate(agent.pos)

            # Update personal best with stagnation tracking
            if agent.best_objectives is None or dominates(agent.objectives, agent.best_objectives):
                agent.best_pos = np.copy(agent.pos)
                agent.best_objectives = agent.objectives.copy()
                agent.stuck_counter = 0
            else:
                agent.stuck_counter += 1
                # Random restart if stuck too long
                if agent.stuck_counter > 20:
                    agent.pos = np.random.uniform(self.bounds_low, self.bounds_high, self.dim)
                    agent.velocity = np.zeros(self.dim)
                    agent.stuck_counter = 0

        self._update_archive()
        self._adaptive_parameter_update()
        self.T *= self.annealing_rate
        
        return self.population, self.archive
    
    def calculate_performance_metrics(self):
        """Calculate comprehensive performance indicators."""
        if not self.archive:
            return None
            
        F_archive = np.array([agent.objectives for agent in self.archive])
        pf = self.problem.pareto_front()
        
        if pf is None:
            return None
        
        metrics = {}
        
        try:
            # Standard metrics
            ref_point = np.max(pf, axis=0) * 1.1
            hv = HV(ref_point=ref_point)
            metrics['hypervolume'] = hv(F_archive)
            
            gd = GD(pf)
            metrics['gd'] = gd(F_archive)
            
            igd = IGD(pf)
            metrics['igd'] = igd(F_archive)
            
            # Additional analysis metrics
            metrics['n_archive'] = len(self.archive)
            metrics['evaluations'] = self.evaluations
            
            # Convergence rate (improvement in HV)
            if len(self.history['hypervolume']) > 0:
                prev_hv = self.history['hypervolume'][-1]
                metrics['convergence_rate'] = metrics['hypervolume'] - prev_hv
            else:
                metrics['convergence_rate'] = 0
            
            # Diversity metric (spread of solutions)
            if len(F_archive) > 1:
                distances = []
                for i in range(len(F_archive)):
                    for j in range(i+1, len(F_archive)):
                        distances.append(np.linalg.norm(F_archive[i] - F_archive[j]))
                metrics['diversity_metric'] = np.mean(distances) if distances else 0
            else:
                metrics['diversity_metric'] = 0
            
            # Stagnation tracking
            metrics['stagnation_count'] = self.stagnation_counter
            
            # Parameter tracking
            metrics['parameter_w'] = self.w
            metrics['parameter_c1'] = self.c1
            metrics['parameter_c2'] = self.c2
            
        except Exception as e:
            logging.warning(f"Error calculating metrics: {e}")
            return None
            
        return metrics

class ComprehensiveBenchmark:
    """Comprehensive benchmark suite for MO-BPSO vs other algorithms."""
    
    def __init__(self):
        self.test_problems = {
            # Convex problems
            'zdt1': {'difficulty': 'easy', 'characteristics': 'convex, continuous'},
            'zdt4': {'difficulty': 'hard', 'characteristics': 'multimodal, 21^9 local fronts'},
            'zdt6': {'difficulty': 'medium', 'characteristics': 'non-uniform, biased'},
            
            # Non-convex problems  
            'zdt2': {'difficulty': 'medium', 'characteristics': 'non-convex, continuous'},
            'zdt3': {'difficulty': 'medium', 'characteristics': 'discontinuous, multi-part'},
            
            # Scalable problems
            'dtlz1': {'difficulty': 'medium', 'characteristics': 'linear, 3-objective'},
            'dtlz2': {'difficulty': 'easy', 'characteristics': 'spherical, 3-objective'},
            'dtlz3': {'difficulty': 'hard', 'characteristics': 'multimodal, 3^k local fronts'},
            'dtlz4': {'difficulty': 'medium', 'characteristics': 'biased, 3-objective'},
        }
        
        self.algorithms = {
            'MO-BPSO': None,  # Will be set during testing
            'NSGA-II': NSGA2,
            'NSGA-III': NSGA3,
            'MOEA/D': MOEAD
        }
        
        self.results = defaultdict(dict)
        self.detailed_results = []

    def run_algorithm(self, algorithm_name, problem_name, n_evaluations=20000, n_runs=5):
        """Run a single algorithm on a problem multiple times."""
        problem = get_problem(problem_name)
        all_metrics = []
        execution_times = []
        
        print(f"  Running {algorithm_name} on {problem_name.upper()}...")
        
        for run in range(n_runs):
            start_time = time.time()
            
            if algorithm_name == 'MO-BPSO':
                # Run MO-BPSO
                optimizer = MO_BPSO_Optimizer(problem, n_agents=100, adaptive_params=True)
                n_iterations = n_evaluations // 100
                
                run_metrics = []
                for i in range(n_iterations):
                    optimizer.step()
                    if (i + 1) % 10 == 0:  # Track every 10 iterations
                        metrics = optimizer.calculate_performance_metrics()
                        if metrics:
                            metrics['iteration'] = i + 1
                            metrics['algorithm'] = algorithm_name
                            metrics['problem'] = problem_name
                            metrics['run'] = run + 1
                            run_metrics.append(metrics.copy())
                
                final_metrics = optimizer.calculate_performance_metrics()
                final_F = np.array([agent.objectives for agent in optimizer.archive])
                
            else:
                # Run pymoo algorithms
                pop_size = 100
                
                if algorithm_name == 'NSGA-III' and problem.n_obj > 2:
                    ref_dirs = get_reference_directions("das-dennis", problem.n_obj, n_partitions=12)
                    algorithm = self.algorithms[algorithm_name](pop_size=pop_size, ref_dirs=ref_dirs)
                else:
                    algorithm = self.algorithms[algorithm_name](pop_size=pop_size)
                
                termination = get_termination("n_eval", n_evaluations)
                res = minimize(problem, algorithm, termination, verbose=False)
                
                final_F = res.F
                
                # Calculate metrics for pymoo algorithms
                pf = problem.pareto_front()
                if pf is not None:
                    ref_point = np.max(pf, axis=0) * 1.1
                    
                    hv = HV(ref_point=ref_point)
                    igd = IGD(pf)
                    gd = GD(pf)
                    
                    final_metrics = {
                        'hypervolume': hv(final_F),
                        'igd': igd(final_F),
                        'gd': gd(final_F),
                        'n_archive': len(final_F),
                        'evaluations': n_evaluations,
                        'algorithm': algorithm_name,
                        'problem': problem_name,
                        'run': run + 1
                    }
                    
                    run_metrics = [final_metrics]  # Only final metrics for pymoo
                else:
                    continue
            
            execution_time = time.time() - start_time
            execution_times.append(execution_time)
            
            if final_metrics:
                final_metrics['execution_time'] = execution_time
                final_metrics['final_archive_size'] = len(final_F)
                all_metrics.extend(run_metrics)
        
        # Calculate statistics across runs
        if all_metrics:
            final_run_metrics = [m for m in all_metrics if 'iteration' not in m or m.get('iteration', 0) >= n_iterations-10]
            
            stats = {
                'mean_hv': np.mean([m['hypervolume'] for m in final_run_metrics if 'hypervolume' in m]),
                'std_hv': np.std([m['hypervolume'] for m in final_run_metrics if 'hypervolume' in m]),
                'mean_igd': np.mean([m['igd'] for m in final_run_metrics if 'igd' in m]),
                'std_igd': np.std([m['igd'] for m in final_run_metrics if 'igd' in m]),
                'mean_time': np.mean(execution_times),
                'std_time': np.std(execution_times),
                'success_rate': len(final_run_metrics) / n_runs,
                'mean_evaluations': np.mean([m.get('evaluations', n_evaluations) for m in final_run_metrics])
            }
            
            self.results[problem_name][algorithm_name] = stats
            self.detailed_results.extend(all_metrics)
            
            return stats, all_metrics
        
        return None, []

    def run_comprehensive_benchmark(self, n_runs=5, n_evaluations=15000):
        """Run comprehensive benchmark comparing all algorithms on all problems."""
        print("="*80)
        print("COMPREHENSIVE MULTI-OBJECTIVE OPTIMIZATION BENCHMARK")
        print("="*80)
        
        total_tests = len(self.test_problems) * len(self.algorithms)
        current_test = 0
        
        for problem_name, problem_info in self.test_problems.items():
            print(f"\nTesting Problem: {problem_name.upper()} - {problem_info['characteristics']}")
            print(f"Difficulty: {problem_info['difficulty']}")
            print("-" * 60)
            
            # Test each algorithm
            for algorithm_name in self.algorithms.keys():
                current_test += 1
                print(f"[{current_test}/{total_tests}] ", end="")
                
                try:
                    stats, detailed = self.run_algorithm(
                        algorithm_name, problem_name, n_evaluations, n_runs
                    )
                    
                    if stats:
                        print(f"    ✓ Completed - HV: {stats['mean_hv']:.4f}±{stats['std_hv']:.4f}")
                    else:
                        print(f"    ✗ Failed")
                        
                except Exception as e:
                    print(f"    ✗ Error: {e}")
        
        self._analyze_results()
        self._generate_detailed_report()
        self._create_visualizations()
        
        return self.results, self.detailed_results

    def _analyze_results(self):
        """Analyze results to identify algorithm strengths and weaknesses."""
        print("\n" + "="*80)
        print("PERFORMANCE ANALYSIS")
        print("="*80)
        
        # Overall performance ranking
        algorithm_scores = defaultdict(list)
        
        for problem_name in self.test_problems.keys():
            if problem_name in self.results:
                problem_results = self.results[problem_name]
                
                # Rank algorithms by hypervolume for this problem
                hv_ranking = sorted(problem_results.items(), 
                                  key=lambda x: x[1].get('mean_hv', 0), reverse=True)
                
                print(f"\n{problem_name.upper()} - Hypervolume Ranking:")
                for rank, (algo, stats) in enumerate(hv_ranking, 1):
                    score = len(hv_ranking) - rank + 1  # Higher rank = higher score
                    algorithm_scores[algo].append(score)
                    print(f"  {rank}. {algo:<12} HV: {stats.get('mean_hv', 0):.6f}±{stats.get('std_hv', 0):.6f} "
                          f"IGD: {stats.get('mean_igd', float('inf')):.6f}±{stats.get('std_igd', 0):.6f}")
        
        # Overall ranking
        overall_ranking = []
        for algo, scores in algorithm_scores.items():
            avg_score = np.mean(scores) if scores else 0
            overall_ranking.append((algo, avg_score, len(scores)))
        
        overall_ranking.sort(key=lambda x: x[1], reverse=True)
        
        print(f"\n{'OVERALL ALGORITHM RANKING'}")
        print("-" * 50)
        for rank, (algo, avg_score, n_problems) in enumerate(overall_ranking, 1):
            print(f"{rank}. {algo:<12} Average Score: {avg_score:.2f} ({n_problems} problems)")

    def _generate_detailed_report(self):
        """Generate detailed performance report with strengths/weaknesses analysis."""
        if not self.detailed_results:
            return
        
        df = pd.DataFrame(self.detailed_results)
        
        print("\n" + "="*80)
        print("DETAILED ALGORITHM ANALYSIS")
        print("="*80)
        
        # Analyze MO-BPSO specifically
        mo_bpso_data = df[df['algorithm'] == 'MO-BPSO']
        
        if not mo_bpso_data.empty:
            print("\nMO-BPSO PERFORMANCE ANALYSIS:")
            print("-" * 40)
            
            # Best performing problems
            problem_performance = mo_bpso_data.groupby('problem').agg({
                'hypervolume': ['mean', 'std'],
                'igd': ['mean', 'std'],
                'convergence_rate': 'mean',
                'diversity_metric': 'mean'
            }).round(6)
            
            print("\n✅ STRENGTHS:")
            # Find problems where MO-BPSO performs well
            for problem in self.test_problems.keys():
                if problem in self.results and 'MO-BPSO' in self.results[problem]:
                    mo_bpso_hv = self.results[problem]['MO-BPSO'].get('mean_hv', 0)
                    
                    # Compare with best competitor
                    competitors = {k: v.get('mean_hv', 0) for k, v in self.results[problem].items() 
                                 if k != 'MO-BPSO'}
                    
                    if competitors:
                        best_competitor_hv = max(competitors.values())
                        
                        if mo_bpso_hv >= best_competitor_hv * 0.95:  # Within 5%
                            improvement = (mo_bpso_hv / best_competitor_hv - 1) * 100
                            problem_char = self.test_problems[problem]['characteristics']
                            print(f"  • {problem.upper()}: {improvement:+.1f}% vs best competitor ({problem_char})")
            
            print("\n❌ WEAKNESSES:")
            for problem in self.test_problems.keys():
                if problem in self.results and 'MO-BPSO' in self.results[problem]:
                    mo_bpso_hv = self.results[problem]['MO-BPSO'].get('mean_hv', 0)
                    
                    competitors = {k: v.get('mean_hv', 0) for k, v in self.results[problem].items() 
                                 if k != 'MO-BPSO'}
                    
                    if competitors:
                        best_competitor_hv = max(competitors.values())
                        
                        if mo_bpso_hv < best_competitor_hv * 0.90:  # More than 10% worse
                            gap = (1 - mo_bpso_hv / best_competitor_hv) * 100
                            problem_char = self.test_problems[problem]['characteristics']
                            print(f"  • {problem.upper()}: -{gap:.1f}% vs best competitor ({problem_char})")
            
            # Parameter adaptation analysis
            if 'parameter_w' in mo_bpso_data.columns:
                print("\n🔧 PARAMETER ADAPTATION INSIGHTS:")
                param_stats = mo_bpso_data.groupby('problem')[['parameter_w', 'parameter_c1', 'parameter_c2']].agg(['mean', 'std'])
                print("  • Parameter evolution shows adaptive behavior")
                print(f"  • Average inertia (w): {mo_bpso_data['parameter_w'].mean():.3f}")
                print(f"  • Average cognitive (c1): {mo_bpso_data['parameter_c1'].mean():.3f}")
                print(f"  • Average social (c2): {mo_bpso_data['parameter_c2'].mean():.3f}")

    def _create_visualizations(self):
        """Create comprehensive visualizations of benchmark results."""
        if not self.detailed_results:
            return
        
        df = pd.DataFrame(self.detailed_results)
        
        # Create subplot figure
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Comprehensive Multi-Objective Algorithm Benchmark Results', fontsize=16)
        
        # 1. Hypervolume comparison by problem
        if 'hypervolume' in df.columns:
            ax = axes[0, 0]
            pivot_hv = df.groupby(['problem', 'algorithm'])['hypervolume'].mean().unstack()
            pivot_hv.plot(kind='bar', ax=ax, width=0.8)
            ax.set_title('Average Hypervolume by Problem')
            ax.set_ylabel('Hypervolume')
            ax.tick_params(axis='x', rotation=45)
            ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        
        # 2. IGD comparison by problem  
        if 'igd' in df.columns:
            ax = axes[0, 1]
            pivot_igd = df.groupby(['problem', 'algorithm'])['igd'].mean().unstack()
            pivot_igd.plot(kind='bar', ax=ax, width=0.8)
            ax.set_title('Average IGD by Problem (Lower is Better)')
            ax.set_ylabel('IGD')
            ax.set_yscale('log')
            ax.tick_params(axis='x', rotation=45)
            ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        
        # 3. Execution time comparison
        if 'execution_time' in df.columns:
            ax = axes[0, 2]
            time_data = df.groupby('algorithm')['execution_time'].mean()
            time_data.plot(kind='bar', ax=ax, color='skyblue')
            ax.set_title('Average Execution Time')
            ax.set_ylabel('Time (seconds)')
            ax.tick_params(axis='x', rotation=45)
        
        # 4. MO-BPSO convergence analysis
        mo_bpso_data = df[df['algorithm'] == 'MO-BPSO']
        if not mo_bpso_data.empty and 'iteration' in mo_bpso_data.columns:
            ax = axes[1, 0]
            for problem in mo_bpso_data['problem'].unique():
                problem_data = mo_bpso_data[mo_bpso_data['problem'] == problem]
                if len(problem_data) > 1:
                    ax.plot(problem_data['iteration'], problem_data['hypervolume'], 
                           marker='o', label=problem.upper(), linewidth=2, markersize=4)
            ax.set_title('MO-BPSO Convergence Curves')
            ax.set_xlabel('Iteration')
            ax.set_ylabel('Hypervolume')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        # 5. Parameter evolution for MO-BPSO
        if not mo_bpso_data.empty and all(col in mo_bpso_data.columns for col in ['parameter_w', 'parameter_c1', 'parameter_c2']):
            ax = axes[1, 1]
            problem = mo_bpso_data['problem'].iloc[0]  # Pick one problem for example
            problem_data = mo_bpso_data[mo_bpso_data['problem'] == problem]
            if 'iteration' in problem_data.columns and len(problem_data) > 1:
                ax.plot(problem_data['iteration'], problem_data['parameter_w'], 'b-', label='Inertia (w)', linewidth=2)
                ax.plot(problem_data['iteration'], problem_data['parameter_c1'], 'r-', label='Cognitive (c1)', linewidth=2)
                ax.plot(problem_data['iteration'], problem_data['parameter_c2'], 'g-', label='Social (c2)', linewidth=2)
                ax.set_title(f'Parameter Evolution - {problem.upper()}')
                ax.set_xlabel('Iteration')
                ax.set_ylabel('Parameter Value')
                ax.legend()
                ax.grid(True, alpha=0.3)
        
        # 6. Success rate and robustness
        ax = axes[1, 2]
        success_rates = []
        algorithms = []
        
        for algo in df['algorithm'].unique():
            algo_data = df[df['algorithm'] == algo]
            # Calculate coefficient of variation for hypervolume (lower = more robust)
            cv_values = []
            for problem in algo_data['problem'].unique():
                problem_data = algo_data[algo_data['problem'] == problem]
                if len(problem_data) > 1:
                    hv_values = problem_data['hypervolume'].dropna()
                    if len(hv_values) > 1 and hv_values.mean() > 0:
                        cv = hv_values.std() / hv_values.mean()
                        cv_values.append(cv)
            
            if cv_values:
                avg_cv = np.mean(cv_values)
                success_rates.append(1 / (1 + avg_cv))  # Convert to success-like metric
                algorithms.append(algo)
        
        if success_rates:
            bars = ax.bar(algorithms, success_rates, color='lightcoral', alpha=0.7)
            ax.set_title('Algorithm Robustness\n(Higher = More Consistent)')
            ax.set_ylabel('Robustness Score')
            ax.tick_params(axis='x', rotation=45)
            
            # Add value labels on bars
            for bar, rate in zip(bars, success_rates):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                       f'{rate:.3f}', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.show()
        
        # Create additional detailed comparison plot
        self._create_detailed_comparison_plot(df)

    def _create_detailed_comparison_plot(self, df):
        """Create detailed comparison focusing on MO-BPSO vs competitors."""
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Detailed MO-BPSO Performance Analysis', fontsize=16)
        
        # 1. Radar chart comparing algorithms across different problem characteristics
        ax = axes[0, 0]
        
        # Group problems by difficulty
        problem_groups = {
            'Easy': [p for p, info in self.test_problems.items() if info['difficulty'] == 'easy'],
            'Medium': [p for p, info in self.test_problems.items() if info['difficulty'] == 'medium'],
            'Hard': [p for p, info in self.test_problems.items() if info['difficulty'] == 'hard']
        }
        
        group_performance = {}
        for group_name, problems in problem_groups.items():
            group_performance[group_name] = {}
            for algo in df['algorithm'].unique():
                algo_data = df[(df['algorithm'] == algo) & (df['problem'].isin(problems))]
                if not algo_data.empty:
                    avg_hv = algo_data['hypervolume'].mean()
                    group_performance[group_name][algo] = avg_hv
        
        # Plot grouped performance
        x_pos = np.arange(len(problem_groups))
        width = 0.2
        
        for i, algo in enumerate(df['algorithm'].unique()):
            values = [group_performance[group].get(algo, 0) for group in problem_groups.keys()]
            ax.bar(x_pos + i * width, values, width, label=algo, alpha=0.8)
        
        ax.set_xlabel('Problem Difficulty')
        ax.set_ylabel('Average Hypervolume')
        ax.set_title('Performance by Problem Difficulty')
        ax.set_xticks(x_pos + width * 1.5)
        ax.set_xticklabels(problem_groups.keys())
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 2. Convergence comparison
        ax = axes[0, 1]
        mo_bpso_data = df[df['algorithm'] == 'MO-BPSO']
        
        if not mo_bpso_data.empty and 'iteration' in mo_bpso_data.columns:
            # Show convergence for different problem types
            colors = ['blue', 'red', 'green', 'orange', 'purple']
            for i, problem in enumerate(mo_bpso_data['problem'].unique()[:5]):
                problem_data = mo_bpso_data[mo_bpso_data['problem'] == problem]
                if len(problem_data) > 1:
                    # Group by iteration and take mean across runs
                    conv_data = problem_data.groupby('iteration')['hypervolume'].mean()
                    ax.plot(conv_data.index, conv_data.values, 
                           color=colors[i % len(colors)], linewidth=2, 
                           label=f'{problem.upper()}', marker='o', markersize=3)
            
            ax.set_xlabel('Iteration')
            ax.set_ylabel('Hypervolume')
            ax.set_title('MO-BPSO Convergence by Problem')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        # 3. Performance distribution
        ax = axes[1, 0]
        
        # Box plot of hypervolume distributions
        hv_data = []
        labels = []
        
        for algo in df['algorithm'].unique():
            algo_hv = df[df['algorithm'] == algo]['hypervolume'].dropna()
            if len(algo_hv) > 0:
                hv_data.append(algo_hv)
                labels.append(algo)
        
        if hv_data:
            bp = ax.boxplot(hv_data, labels=labels, patch_artist=True)
            colors = ['lightblue', 'lightcoral', 'lightgreen', 'lightyellow']
            for patch, color in zip(bp['boxes'], colors):
                patch.set_facecolor(color)
            
            ax.set_title('Hypervolume Distribution Across All Problems')
            ax.set_ylabel('Hypervolume')
            ax.tick_params(axis='x', rotation=45)
            ax.grid(True, alpha=0.3)
        
        # 4. Efficiency analysis (Performance vs Time)
        ax = axes[1, 1]
        
        if 'execution_time' in df.columns:
            for algo in df['algorithm'].unique():
                algo_data = df[df['algorithm'] == algo]
                if len(algo_data) > 0:
                    avg_hv = algo_data['hypervolume'].mean()
                    avg_time = algo_data['execution_time'].mean()
                    std_hv = algo_data['hypervolume'].std()
                    std_time = algo_data['execution_time'].std()
                    
                    ax.errorbar(avg_time, avg_hv, xerr=std_time, yerr=std_hv, 
                               fmt='o', markersize=8, label=algo, capsize=5)
            
            ax.set_xlabel('Execution Time (seconds)')
            ax.set_ylabel('Hypervolume')
            ax.set_title('Performance vs Efficiency')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()

    def generate_recommendations(self):
        """Generate specific recommendations for improving MO-BPSO."""
        print("\n" + "="*80)
        print("ALGORITHM IMPROVEMENT RECOMMENDATIONS")
        print("="*80)
        
        if not self.detailed_results:
            print("No results available for recommendations.")
            return
        
        df = pd.DataFrame(self.detailed_results)
        mo_bpso_data = df[df['algorithm'] == 'MO-BPSO']
        
        if mo_bpso_data.empty:
            print("No MO-BPSO data available for analysis.")
            return
        
        print("\n🎯 SPECIFIC RECOMMENDATIONS FOR MO-BPSO:")
        print("-" * 50)
        
        # Analyze performance patterns
        problem_performance = {}
        for problem in mo_bpso_data['problem'].unique():
            problem_data = mo_bpso_data[mo_bpso_data['problem'] == problem]
            if len(problem_data) > 0:
                avg_hv = problem_data['hypervolume'].mean()
                problem_info = self.test_problems.get(problem, {})
                problem_performance[problem] = {
                    'hv': avg_hv,
                    'characteristics': problem_info.get('characteristics', 'unknown'),
                    'difficulty': problem_info.get('difficulty', 'unknown')
                }
        
        # Find patterns in good vs poor performance
        good_problems = [p for p, data in problem_performance.items() if data['hv'] > 0.5]
        poor_problems = [p for p, data in problem_performance.items() if data['hv'] <= 0.5]
        
        print("1. PROBLEM-SPECIFIC INSIGHTS:")
        if good_problems:
            good_chars = [problem_performance[p]['characteristics'] for p in good_problems]
            print(f"   ✅ Performs well on: {', '.join(good_problems)}")
            print(f"      Common characteristics: {', '.join(set(good_chars))}")
        
        if poor_problems:
            poor_chars = [problem_performance[p]['characteristics'] for p in poor_problems]
            print(f"   ❌ Struggles with: {', '.join(poor_problems)}")
            print(f"      Common characteristics: {', '.join(set(poor_chars))}")
        
        print("\n2. PARAMETER TUNING RECOMMENDATIONS:")
        if all(col in mo_bpso_data.columns for col in ['parameter_w', 'parameter_c1', 'parameter_c2']):
            # Analyze parameter patterns for good performance
            high_perf_data = mo_bpso_data[mo_bpso_data['hypervolume'] > mo_bpso_data['hypervolume'].quantile(0.75)]
            
            if len(high_perf_data) > 0:
                optimal_w = high_perf_data['parameter_w'].mean()
                optimal_c1 = high_perf_data['parameter_c1'].mean()
                optimal_c2 = high_perf_data['parameter_c2'].mean()
                
                print(f"   🔧 Optimal parameter ranges found:")
                print(f"      Inertia (w): {optimal_w:.3f} ± {high_perf_data['parameter_w'].std():.3f}")
                print(f"      Cognitive (c1): {optimal_c1:.3f} ± {high_perf_data['parameter_c1'].std():.3f}")
                print(f"      Social (c2): {optimal_c2:.3f} ± {high_perf_data['parameter_c2'].std():.3f}")
        
        print("\n3. ALGORITHMIC IMPROVEMENTS:")
        
        # Check convergence patterns
        if 'convergence_rate' in mo_bpso_data.columns:
            avg_conv_rate = mo_bpso_data['convergence_rate'].mean()
            if avg_conv_rate < 0.01:
                print("   🐌 Slow convergence detected:")
                print("      - Consider adaptive step size mechanisms")
                print("      - Implement local search hybridization")
                print("      - Use dynamic population sizing")
        
        # Check diversity patterns
        if 'diversity_metric' in mo_bpso_data.columns:
            diversity_trend = mo_bpso_data['diversity_metric'].std()
            if diversity_trend < 0.1:
                print("   📉 Diversity issues detected:")
                print("      - Strengthen crowding distance mechanism")
                print("      - Add mutation operators")
                print("      - Implement niching strategies")
        
        # Check stagnation patterns
        if 'stagnation_count' in mo_bpso_data.columns:
            avg_stagnation = mo_bpso_data['stagnation_count'].mean()
            if avg_stagnation > 5:
                print("   🔄 Stagnation issues detected:")
                print("      - Implement restart mechanisms")
                print("      - Add perturbation strategies")
                print("      - Use dynamic archive management")
        
        print("\n4. IMPLEMENTATION RECOMMENDATIONS:")
        print("   💡 Based on benchmark results:")
        
        # Compare with best performing algorithm
        best_algo = None
        best_hv = 0
        
        for algo in df['algorithm'].unique():
            algo_hv = df[df['algorithm'] == algo]['hypervolume'].mean()
            if algo_hv > best_hv:
                best_hv = algo_hv
                best_algo = algo
        
        if best_algo and best_algo != 'MO-BPSO':
            print(f"      - Study {best_algo} mechanisms for potential hybridization")
            print(f"      - Consider ensemble approaches combining MO-BPSO with {best_algo}")
        
        print("      - Implement problem-specific parameter adaptation")
        print("      - Add early convergence detection and restart")
        print("      - Consider multi-population strategies for hard problems")
        
        print("\n5. TESTING RECOMMENDATIONS:")
        print("   🧪 Additional benchmark suggestions:")
        print("      - Test on many-objective problems (>3 objectives)")
        print("      - Evaluate on constrained optimization problems")
        print("      - Test scalability with different population sizes")
        print("      - Analyze performance on dynamic optimization problems")

def run_comprehensive_analysis():
    """Run the complete benchmark analysis."""
    benchmark = ComprehensiveBenchmark()
    
    print("Starting comprehensive benchmark analysis...")
    print("This will test MO-BPSO against NSGA-II, NSGA-III, and MOEA/D")
    print("on multiple problem types with detailed performance tracking.\n")
    
    # Run benchmark with multiple runs for statistical significance
    results, detailed_results = benchmark.run_comprehensive_benchmark(
        n_runs=3,  # Reduced for faster testing, increase for production
        n_evaluations=10000  # Reduced for faster testing
    )
    
    # Generate improvement recommendations
    benchmark.generate_recommendations()
    
    # Save results for further analysis
    if detailed_results:
        df = pd.DataFrame(detailed_results)
        df.to_csv('mo_bpso_benchmark_results.csv', index=False)
        print(f"\n📊 Results saved to 'mo_bpso_benchmark_results.csv'")
        print(f"Total tests completed: {len(detailed_results)}")
    
    return benchmark

# --- Quick Demo Function ---
def quick_demo():
    """Quick demonstration of MO-BPSO with performance tracking."""
    print("Quick MO-BPSO Demo on ZDT1...")
    
    problem = get_problem('zdt1')
    optimizer = MO_BPSO_Optimizer(problem, n_agents=50, adaptive_params=True)
    
    # Track performance over iterations
    performance_history = []
    
    for i in range(50):
        optimizer.step()
        
        if (i + 1) % 10 == 0:
            metrics = optimizer.calculate_performance_metrics()
            if metrics:
                metrics['iteration'] = i + 1
                performance_history.append(metrics)
                print(f"Iter {i+1:2d}: HV={metrics['hypervolume']:.4f}, "
                      f"IGD={metrics['igd']:.4f}, Archive={metrics['n_archive']}")
    
    # Simple visualization
    if performance_history:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Plot convergence
        iterations = [p['iteration'] for p in performance_history]
        hv_values = [p['hypervolume'] for p in performance_history]
        
        ax1.plot(iterations, hv_values, 'b-o', linewidth=2)
        ax1.set_title('Hypervolume Convergence')
        ax1.set_xlabel('Iteration')
        ax1.set_ylabel('Hypervolume')
        ax1.grid(True)
        
        # Plot final Pareto front
        if optimizer.archive:
            final_F = np.array([agent.objectives for agent in optimizer.archive])
            ax2.scatter(final_F[:, 0], final_F[:, 1], c='red', alpha=0.7, s=50)
            
            # True Pareto front
            pf = problem.pareto_front()
            if pf is not None:
                ax2.plot(pf[:, 0], pf[:, 1], 'k--', alpha=0.5, label='True PF')
            
            ax2.set_title('Final Pareto Front')
            ax2.set_xlabel('f1')
            ax2.set_ylabel('f2')
            ax2.legend()
            ax2.grid(True)
        
        plt.tight_layout()
        plt.show()
    
    return optimizer

if __name__ == '__main__':
    print("MO-BPSO Comprehensive Benchmark Suite")
    print("="*50)
    print("Choose an option:")
    print("1. Quick Demo (fast)")
    print("2. Comprehensive Benchmark (slower, detailed analysis)")
    
    choice = input("\nEnter choice (1 or 2): ").strip()
    
    if choice == '1':
        optimizer = quick_demo()
    else:
        benchmark = run_comprehensive_analysis()
