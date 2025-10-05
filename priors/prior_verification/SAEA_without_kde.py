import numpy as np
import pandas as pd
import warnings
import os
import random
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
from sklearn.preprocessing import StandardScaler
from deap import base, creator, tools, algorithms

# Import PDE modules
from pde1_convection_2d import convection_2d
from pde2_vca_2d import vca_2d
from pde3_heat_2d import heat_2d
from pde4_kdv_2d import kdv_2d
from pde5_ad_2d import ad_2d
from pde6_convection2_2d import convection2_2d

warnings.filterwarnings('ignore')


class SAEA_PINN:
    def __init__(self, max_evals=50, pop_size=30, neighborhood_radius=0.1):
        """
        Surrogate-Assisted Evolutionary Algorithm for PINN Hyperparameter Optimization

        Args:
            max_evals: Maximum number of function evaluations (initial + iterations)
            pop_size: Population size for GA
            neighborhood_radius: Radius for neighborhood sampling
        """
        self.activation_map = {0: 'sin', 1: 'tanh', 2: 'swish', 3: 'sigmoid', 4: 'relu'}
        self.reverse_activation_map = {'sin': 0, 'tanh': 1, 'swish': 2, 'sigmoid': 3, 'relu': 4}

        # Parameter bounds: [n_layers, n_nodes, activation, epochs, grid_size, learning_rate]
        self.bounds = np.array([
            [2, 10],  # n_layers
            [5, 100],  # n_nodes
            [0, 4],  # activation (index)
            [5000, 100000],  # epochs
            [10, 200],  # grid_size
            [1e-5, 0.1]  # learning_rate
        ])

        self.dim = len(self.bounds)
        self.max_evals = max_evals
        self.pop_size = pop_size
        self.neighborhood_radius = neighborhood_radius

        # Storage for evaluated points and function values
        self.X_evaluated = []
        self.y_evaluated = []
        self.eval_count = 0
        self.results = []

        # Scaler and Kriging surrogate model
        self.scaler = StandardScaler()
        kernel = C(1.0) * RBF(length_scale=1.0)
        self.kriging = GaussianProcessRegressor(kernel=kernel, alpha=1e-6, normalize_y=True)

        # Initialize DEAP
        self._setup_deap()

    def _setup_deap(self):
        """Setup DEAP genetic algorithm components"""
        # Clear any existing creators
        if hasattr(creator, "FitnessMin"):
            del creator.FitnessMin
        if hasattr(creator, "Individual"):
            del creator.Individual

        creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
        creator.create("Individual", list, fitness=creator.FitnessMin)

        self.toolbox = base.Toolbox()

        # Individual generation
        for i in range(self.dim):
            self.toolbox.register(f"attr_{i}", random.uniform,
                                  self.bounds[i][0], self.bounds[i][1])

        attrs = [getattr(self.toolbox, f"attr_{i}") for i in range(self.dim)]
        self.toolbox.register("individual", tools.initCycle, creator.Individual, attrs, n=1)
        self.toolbox.register("population", tools.initRepeat, list, self.toolbox.individual)

        # Genetic operations
        self.toolbox.register("mate", self._crossover)
        self.toolbox.register("mutate", self._mutation)
        self.toolbox.register("select", tools.selTournament, tournsize=3)
        self.toolbox.register("evaluate", self._kriging_evaluate)

    def _clip_and_cast(self, x):
        """Clip parameters to bounds and cast to appropriate types"""
        x = np.array(x, dtype=float).flatten()
        x[0] = int(np.clip(np.round(x[0]), 2, 10))  # n_layers
        x[1] = int(np.clip(np.round(x[1]), 5, 100))  # n_nodes
        x[2] = int(np.clip(np.round(x[2]), 0, 4))  # activation index
        x[3] = int(np.clip(np.round(x[3]), 5000, 100000))  # epochs
        x[4] = int(np.clip(np.round(x[4]), 10, 200))  # grid_size
        x[5] = float(np.clip(x[5], 1e-5, 0.1))  # learning_rate
        return x

    def evaluate_pinn(self, params):
        """Evaluate PINN with given parameters"""
        try:
            p = self._clip_and_cast(params)
            result = vca_2d(
                int(p[0]), int(p[1]), self.activation_map[int(p[2])],
                int(p[3]), int(p[4]), float(p[5]),
                f"saea_iter_{self.eval_count + 1}", plotshow=False, device='cuda'
            )
            return float(result['final_l2_re_error'])
        except Exception as e:
            print(f"[evaluate_pinn] Exception while evaluating params {params}: {e}")
            return 1.0

    def _load_initial_samples(self, csv_path='initial_samples2.csv', n_samples=10):
        """Load initial samples from CSV file"""
        print(f"Loading initial samples from {csv_path}...")

        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"Initial sample file not found: {csv_path}")

        initial_data = pd.read_csv(csv_path)

        for i in range(min(n_samples, len(initial_data))):
            row = initial_data.iloc[i]

            # Convert activation name to index if it's a string
            if isinstance(row['activation'], str):
                activation_idx = self.reverse_activation_map.get(row['activation'], 0)
            else:
                activation_idx = int(row['activation'])

            params = np.array([
                row['n_layers'], row['n_nodes'], activation_idx,
                row['epochs'], row['grid_size'], row['learning_rate']
            ])

            # Evaluate with true function
            error = self.evaluate_pinn(params)

            # Store results
            clipped_params = self._clip_and_cast(params)
            self.X_evaluated.append(clipped_params)
            self.y_evaluated.append(error)
            self.eval_count += 1

            # Save to results
            self.results.append({
                'iteration': self.eval_count,
                'n_layers': int(clipped_params[0]),
                'n_nodes': int(clipped_params[1]),
                'activation': self.activation_map[int(clipped_params[2])],
                'epochs': int(clipped_params[3]),
                'grid_size': int(clipped_params[4]),
                'learning_rate': float(clipped_params[5]),
                'fitness': float(error),
                'is_initial': True
            })

            print(f"Initial sample {i + 1}: Error = {error:.6f}")

        # Update Kriging model
        self._update_kriging()

    def _update_kriging(self):
        """Update Kriging surrogate model"""
        if len(self.X_evaluated) > 0:
            X = np.array(self.X_evaluated)
            y = np.array(self.y_evaluated)

            # Scale features
            X_scaled = self.scaler.fit_transform(X)

            # Fit Kriging model
            self.kriging.fit(X_scaled, y)

    def _kriging_evaluate(self, individual):
        """Evaluate individual using Kriging model"""
        x = np.array(individual)
        clipped_x = self._clip_and_cast(x)

        # Scale the point
        x_scaled = self.scaler.transform([clipped_x])
        pred_y, _ = self.kriging.predict(x_scaled, return_std=True)
        return pred_y[0],

    def _crossover(self, ind1, ind2):
        """Blend crossover (BLX-alpha)"""
        alpha = 0.5
        for i in range(len(ind1)):
            if random.random() < 0.5:
                temp = alpha * ind1[i] + (1 - alpha) * ind2[i]
                ind2[i] = alpha * ind2[i] + (1 - alpha) * ind1[i]
                ind1[i] = temp

        # Boundary repair
        self._repair_bounds(ind1)
        self._repair_bounds(ind2)
        return ind1, ind2

    def _mutation(self, individual, indpb=0.2):
        """Polynomial mutation with parameter-specific handling"""
        for i in range(len(individual)):
            if random.random() < indpb:
                if i == 5:  # learning_rate - log scale mutation
                    current_val = max(individual[i], 1e-5)
                    log_val = np.log10(current_val)
                    log_val += random.gauss(0, 0.5)
                    individual[i] = 10 ** log_val
                else:
                    delta = 0.1 * (self.bounds[i][1] - self.bounds[i][0])
                    individual[i] += random.gauss(0, delta)

        self._repair_bounds(individual)
        return individual,

    def _repair_bounds(self, individual):
        """Repair boundary constraints"""
        for i in range(len(individual)):
            individual[i] = np.clip(individual[i],
                                    self.bounds[i][0],
                                    self.bounds[i][1])

    def _is_evaluated(self, x, tol=1e-6):
        """Check if point has been evaluated"""
        clipped_x = self._clip_and_cast(x)
        for eval_x in self.X_evaluated:
            if np.allclose(clipped_x, eval_x, atol=tol):
                return True
        return False

    def _sample_neighborhood(self, center):
        """Sample in neighborhood of center point"""
        attempts = 0
        max_attempts = 100

        while attempts < max_attempts:
            noise = np.random.normal(0, self.neighborhood_radius, self.dim)

            # Apply different noise scales for different parameters
            noise[0] *= 2  # n_layers
            noise[1] *= 10  # n_nodes
            noise[2] *= 1  # activation
            noise[3] *= 5000  # epochs
            noise[4] *= 20  # grid_size
            noise[5] *= center[5] * 0.5  # learning_rate (relative)

            x = center + noise

            # Boundary correction
            x = np.clip(x, self.bounds[:, 0], self.bounds[:, 1])

            if not self._is_evaluated(x):
                return x

            attempts += 1

        # If no new point found, return a random point
        return np.array([np.random.uniform(low, high)
                         for low, high in self.bounds])

    def optimize(self, n_bo_iterations=40):
        """Main optimization loop"""
        print("Starting Surrogate-Assisted Evolutionary Algorithm for PINN optimization...")

        # Load initial samples
        self._load_initial_samples(n_samples=10)

        print("\nStarting SAEA optimization iterations...")

        while self.eval_count < self.max_evals:
            current_iter = self.eval_count + 1
            print(f"\nIteration {current_iter}")

            # GA search on Kriging surrogate
            pop = self.toolbox.population(n=self.pop_size)

            # Evaluate initial population
            fitnesses = list(map(self.toolbox.evaluate, pop))
            for ind, fit in zip(pop, fitnesses):
                ind.fitness.values = fit

            # Evolution process
            for gen in range(40):  # Limited generations for efficiency
                offspring = algorithms.varAnd(pop, self.toolbox,
                                              cxpb=0.7, mutpb=0.3)
                fits = self.toolbox.map(self.toolbox.evaluate, offspring)
                for fit, ind in zip(fits, offspring):
                    ind.fitness.values = fit

                pop = self.toolbox.select(offspring + pop, self.pop_size)

            # Find best point predicted by Kriging
            best_ind = tools.selBest(pop, 1)[0]
            candidate = np.array(best_ind)

            # Sampling strategy
            if self._is_evaluated(candidate):
                # If already evaluated, sample in neighborhood
                new_sample = self._sample_neighborhood(candidate)
                print(f"  Strategy: Neighborhood sampling around best GA point")
            else:
                new_sample = candidate
                print(f"  Strategy: GA optimal point")

            # True function evaluation
            new_y = self.evaluate_pinn(new_sample)

            # Store results
            clipped_sample = self._clip_and_cast(new_sample)
            self.X_evaluated.append(clipped_sample)
            self.y_evaluated.append(new_y)
            self.eval_count += 1

            # Add to results
            self.results.append({
                'iteration': self.eval_count,
                'n_layers': int(clipped_sample[0]),
                'n_nodes': int(clipped_sample[1]),
                'activation': self.activation_map[int(clipped_sample[2])],
                'epochs': int(clipped_sample[3]),
                'grid_size': int(clipped_sample[4]),
                'learning_rate': float(clipped_sample[5]),
                'fitness': float(new_y),
                'is_initial': False
            })

            # Update surrogate model
            self._update_kriging()

            # Progress report
            current_best = min(self.y_evaluated)
            print(f"  Current error: {new_y:.6f}")
            print(f"  Best error so far: {current_best:.6f}")

            # Save intermediate results
            results_df = pd.DataFrame(self.results)
            results_df.to_csv('saea_pde2.csv', index=False)

            if self.eval_count >= self.max_evals:
                break

        # Final results
        best_idx = np.argmin(self.y_evaluated)
        best_x = self.X_evaluated[best_idx]
        best_y = self.y_evaluated[best_idx]

        print(f"\nOptimization completed!")
        print(f"Total evaluations: {self.eval_count}")
        print(f"Best parameters:")
        print(f"  n_layers: {int(best_x[0])}")
        print(f"  n_nodes: {int(best_x[1])}")
        print(f"  activation: {self.activation_map[int(best_x[2])]}")
        print(f"  epochs: {int(best_x[3])}")
        print(f"  grid_size: {int(best_x[4])}")
        print(f"  learning_rate: {best_x[5]:.6f}")
        print(f"Best error: {best_y:.6f}")

        return pd.DataFrame(self.results)

    def save_results(self, filename='saea_pde2.csv'):
        """Save optimization results to CSV"""
        results_df = pd.DataFrame(self.results)
        results_df.to_csv(filename, index=False)
        print(f"Results saved to {filename}")
        return results_df


def main():
    """Main execution function"""
    print("SAEA-based PINN Hyperparameter Optimization")
    print("=" * 50)

    # Create optimizer
    # 10 initial samples + 40 optimization iterations = 50 total evaluations
    optimizer = SAEA_PINN(
        max_evals=50,  # 10 initial + 40 iterations
        pop_size=40,  # Population size for GA
        neighborhood_radius=0.15  # Neighborhood sampling radius
    )

    # Run optimization
    results = optimizer.optimize(n_bo_iterations=40)

    # Save final results
    optimizer.save_results('./result/saea_pde2.csv')

    print("\nOptimization completed successfully!")
    print(f"Results saved to saea_pde2.csv")

    return results


if __name__ == "__main__":
    results = main()