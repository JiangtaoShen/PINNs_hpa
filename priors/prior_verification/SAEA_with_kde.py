import numpy as np
import pandas as pd
import pickle
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, ConstantKernel
from sklearn.preprocessing import StandardScaler
from scipy.stats import norm
import warnings
import os
import random
from scipy.spatial.distance import cdist

from pde1_convection_2d import convection_2d
from pde2_vca_2d import vca_2d
from pde3_heat_2d import heat_2d
from pde4_kdv_2d import kdv_2d
from pde5_ad_2d import ad_2d
from pde6_convection2_2d import convection2_2d

warnings.filterwarnings('ignore')


class GeneticAlgorithm:
    def __init__(self, bounds, pop_size=40, n_generations=40, mutation_rate=0.1, crossover_rate=0.8):
        self.bounds = bounds  # [(min, max), (min, max), ...]
        self.pop_size = pop_size
        self.n_generations = n_generations
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.n_vars = len(bounds)

    def _initialize_population(self):
        """初始化种群"""
        population = []
        for _ in range(self.pop_size):
            individual = []
            for i, (low, high) in enumerate(self.bounds):
                if i in [0, 1, 2, 3, 4]:  # 离散变量
                    individual.append(random.randint(int(low), int(high)))
                else:  # 连续变量 (learning_rate)
                    individual.append(random.uniform(low, high))
            population.append(individual)
        return population

    def _mutate(self, individual):
        """变异操作"""
        mutated = individual.copy()
        for i in range(len(mutated)):
            if random.random() < self.mutation_rate:
                low, high = self.bounds[i]
                if i in [0, 1, 2, 3, 4]:  # 离散变量
                    mutated[i] = random.randint(int(low), int(high))
                else:  # 连续变量
                    mutated[i] = random.uniform(low, high)
        return mutated

    def _crossover(self, parent1, parent2):
        """交叉操作"""
        if random.random() > self.crossover_rate:
            return parent1.copy(), parent2.copy()

        crossover_point = random.randint(1, len(parent1) - 1)
        child1 = parent1[:crossover_point] + parent2[crossover_point:]
        child2 = parent2[:crossover_point] + parent1[crossover_point:]

        return child1, child2

    def _tournament_selection(self, population, fitness_values, tournament_size=3):
        """锦标赛选择"""
        selected_idx = random.sample(range(len(population)), tournament_size)
        selected_fitness = [fitness_values[i] for i in selected_idx]
        winner_idx = selected_idx[np.argmin(selected_fitness)]  # 最小化问题
        return population[winner_idx]

    def optimize(self, objective_function):
        """主优化循环"""
        population = self._initialize_population()

        for generation in range(self.n_generations):
            # 评估种群
            fitness_values = []
            for individual in population:
                fitness = objective_function(individual)
                fitness_values.append(fitness)

            # 选择、交叉、变异产生新种群
            new_population = []

            # 精英保留：保留最好的个体
            best_idx = np.argmin(fitness_values)
            new_population.append(population[best_idx])

            # 生成剩余个体
            while len(new_population) < self.pop_size:
                parent1 = self._tournament_selection(population, fitness_values)
                parent2 = self._tournament_selection(population, fitness_values)

                child1, child2 = self._crossover(parent1, parent2)
                child1 = self._mutate(child1)
                child2 = self._mutate(child2)

                new_population.extend([child1, child2])

            population = new_population[:self.pop_size]

        # 返回最优解
        final_fitness = [objective_function(ind) for ind in population]
        best_idx = np.argmin(final_fitness)
        return population[best_idx], final_fitness[best_idx]


class KrigingGAPriorBO:
    def __init__(self, kde_models_dict, beta=1.0):
        self.kde_models = kde_models_dict.get('all pdes', kde_models_dict)
        self.beta = beta
        self.activation_map = {0: 'sin', 1: 'tanh', 2: 'swish', 3: 'sigmoid', 4: 'relu'}

        # placeholders
        self.results = []
        self.X_obs = []
        self.y_obs = []

        # scaler 和 kriging 后面会在初始化数据后创建/拟合
        self.scaler = None
        self.kriging = None  # 使用Kriging而不是GP

        # 用于KDE值标准化
        self.kde_scaler = None
        self.kde_values = []

    def _clip_and_cast(self, x):
        x = np.array(x, dtype=float).flatten()
        x[0] = int(np.clip(np.round(x[0]), 2, 10))  # n_layers
        x[1] = int(np.clip(np.round(x[1]), 5, 100))  # n_nodes
        x[2] = int(np.clip(np.round(x[2]), 0, 4))  # activation index
        x[3] = int(np.clip(np.round(x[3]), 5000, 100000))  # epochs
        x[4] = int(np.clip(np.round(x[4]), 10, 200))  # grid_size
        x[5] = float(np.clip(x[5], 1e-5, 0.1))  # learning_rate
        return x

    def evaluate_prior(self, params):
        try:
            p = self._clip_and_cast(params)
            log_prob = 0.0
            for i in range(6):
                if i == 5:
                    val = np.log10(p[5])
                    log_prob += self.kde_models[i].score_samples([[val]])[0]
                else:
                    log_prob += self.kde_models[i].score_samples([[p[i]]])[0]

            prior = np.exp(log_prob)
            return max(prior, 1e-300)
        except Exception as e:
            print(f"[evaluate_prior] exception: {e}")
            return 1e-300

    def evaluate_pinn(self, params):
        try:
            p = self._clip_and_cast(params)
            result = ad_2d(
                int(p[0]), int(p[1]), self.activation_map[int(p[2])],
                int(p[3]), int(p[4]), float(p[5]),
                f"iter_{len(self.results) + 1}", plotshow=False, device='cuda'
            )
            return float(result['final_l2_re_error'])
        except Exception as e:
            print(f"[evaluate_pinn] exception while evaluating params {params}: {e}")
            return 1.0

    def _scale_X(self, X):
        if self.scaler is None:
            self.scaler = StandardScaler()
            Xs = self.scaler.fit_transform(X)
        else:
            Xs = self.scaler.transform(X)
        return Xs

    def _scale_point(self, x):
        x = np.array(x, dtype=float).reshape(1, -1)
        if self.scaler is None:
            raise RuntimeError("Scaler not initialized.")
        return self.scaler.transform(x)

    def _normalize_kde_values(self):
        """标准化KDE值"""
        if len(self.kde_values) > 1:
            kde_array = np.array(self.kde_values).reshape(-1, 1)
            if self.kde_scaler is None:
                self.kde_scaler = StandardScaler()
                self.kde_scaler.fit(kde_array)
            return self.kde_scaler.transform(kde_array).flatten()
        else:
            return np.array(self.kde_values)

    def _get_normalized_kde_value(self, params):
        """获取单个参数的标准化KDE值"""
        kde_val = self.evaluate_prior(params)
        self.kde_values.append(kde_val)

        if len(self.kde_values) > 1:
            normalized_values = self._normalize_kde_values()
            return normalized_values[-1]  # 返回最新添加的标准化值
        else:
            return 0.0  # 第一个值无法标准化，返回0

    def surrogate_objective(self, params, iteration):
        """代理模型目标函数，融合先验知识"""
        if len(self.X_obs) == 0 or self.kriging is None:
            return 1.0

        x_orig = self._clip_and_cast(params)

        try:
            x_scaled = self._scale_point(x_orig)
            mu, _ = self.kriging.predict(x_scaled, return_std=True)
            mu = float(mu.ravel()[0])
        except Exception as e:
            print(f"[surrogate_objective] kriging predict failed: {e}")
            return 1.0

        # 获取标准化的先验值
        kde_val = self.evaluate_prior(x_orig)
        temp_kde_values = self.kde_values + [kde_val]
        if len(temp_kde_values) > 1:
            kde_array = np.array(temp_kde_values).reshape(-1, 1)
            if self.kde_scaler is None:
                temp_scaler = StandardScaler()
                temp_scaler.fit(kde_array)
                normalized_kde = temp_scaler.transform([[kde_val]])[0, 0]
            else:
                normalized_kde = self.kde_scaler.transform([[kde_val]])[0, 0]
        else:
            normalized_kde = 0.0

        # 先验融合: f_tilde = f_hat - k * P_hat
        k = self.beta / max(1, iteration)
        f_tilde = mu - k * normalized_kde

        return f_tilde

    def _is_duplicate(self, x_new, tolerance=1e-6):
        """检查是否为重复点"""
        if len(self.X_obs) == 0:
            return False

        X_array = np.array(self.X_obs)
        x_new = np.array(x_new).reshape(1, -1)

        # 计算欧几里得距离
        distances = cdist(x_new, X_array, metric='euclidean')
        min_distance = np.min(distances)

        return min_distance < tolerance

    def _sample_nearby(self, x_center, noise_scale=0.1):
        """在给定点附近随机采样"""
        bounds = [(2, 10), (5, 100), (0, 4), (5000, 100000), (10, 200), (1e-5, 0.1)]

        max_attempts = 50
        for _ in range(max_attempts):
            x_new = x_center.copy()

            # 为每个维度添加噪声
            for i in range(len(x_new)):
                low, high = bounds[i]
                if i in [0, 1, 2, 3, 4]:  # 离散变量
                    noise = int(np.random.normal(0, noise_scale * (high - low)))
                    x_new[i] = np.clip(int(x_center[i]) + noise, int(low), int(high))
                else:  # 连续变量
                    noise = np.random.normal(0, noise_scale * (high - low))
                    x_new[i] = np.clip(x_center[i] + noise, low, high)

            # 检查新点是否重复
            if not self._is_duplicate(x_new):
                return self._clip_and_cast(x_new)

        # 如果仍然找不到不重复的点，返回随机点
        print("Warning: Could not find nearby non-duplicate point, returning random sample")
        return self._clip_and_cast([
            np.random.randint(2, 11),
            np.random.randint(5, 101),
            np.random.randint(0, 5),
            np.random.randint(5000, 100001),
            np.random.randint(10, 201),
            10 ** np.random.uniform(-5, -1)
        ])

    def optimize_acquisition_with_ga(self, iteration):
        """使用遗传算法优化代理模型"""
        bounds = [(2, 10), (5, 100), (0, 4), (5000, 100000), (10, 200), (1e-5, 0.1)]

        ga = GeneticAlgorithm(bounds, pop_size=40, n_generations=40)

        def objective_func(params):
            return self.surrogate_objective(params, iteration)

        best_x, best_val = ga.optimize(objective_func)
        best_x = self._clip_and_cast(best_x)

        # 检查是否为重复点
        if self._is_duplicate(best_x):
            print(f"Duplicate point detected, sampling nearby...")
            best_x = self._sample_nearby(best_x)

        return best_x

    def run_optimization(self, n_bo_iterations=40):
        self.results = []
        self.X_obs = []
        self.y_obs = []
        self.kde_values = []

        csv_path = 'initial_samples_kde2.csv'
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"Initial sample file not found: {csv_path}")

        print("Reading initial samples...")
        initial_data = pd.read_csv(csv_path)

        # 只取前10个样本
        initial_data = initial_data.head(10)

        for i, row in initial_data.iterrows():
            params = [row['n_layers'], row['n_nodes'], row['activation'],
                      row['epochs'], row['grid_size'], row['learning_rate']]
            error = self.evaluate_pinn(params)

            # 记录KDE值用于标准化
            kde_val = self.evaluate_prior(params)
            self.kde_values.append(kde_val)

            self.results.append({
                'iteration': i + 1, 'n_layers': int(params[0]), 'n_nodes': int(params[1]),
                'activation': self.activation_map[int(params[2])], 'epochs': int(params[3]),
                'grid_size': int(params[4]), 'learning_rate': params[5],
                'fitness': error, 'is_initial': True
            })

            self.X_obs.append(self._clip_and_cast(params))
            self.y_obs.append(float(error))

            print(f"Initial {i + 1}: Error = {error:.6f}")

        # 初始化KDE标准化器
        if len(self.kde_values) > 1:
            kde_array = np.array(self.kde_values).reshape(-1, 1)
            self.kde_scaler = StandardScaler()
            self.kde_scaler.fit(kde_array)

        # 保存初始结果
        pd.DataFrame(self.results).to_csv('./result/kriging_ga_pde6.csv', index=False)

        # 使用Kriging代理模型 (这里仍使用GaussianProcessRegressor作为Kriging的实现)
        kernel = ConstantKernel(1.0) * Matern(length_scale=0.5, nu=1.5)
        self.kriging = GaussianProcessRegressor(kernel=kernel, alpha=1e-6, normalize_y=True)

        X_arr = np.array(self.X_obs, dtype=float)
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X_arr)
        self.kriging.fit(X_scaled, np.array(self.y_obs, dtype=float))

        print("\nStarting Kriging-GA optimization...")
        for it in range(n_bo_iterations):
            current_iteration = len(self.results) - len(initial_data) + 1
            print(
                f"Iteration {len(self.results) + 1}, k = β/t = {self.beta}/{current_iteration} = {self.beta / max(1, current_iteration):.6f}")

            x_next = self.optimize_acquisition_with_ga(current_iteration)
            error = self.evaluate_pinn(x_next)

            # 记录新的KDE值
            kde_val = self.evaluate_prior(x_next)
            self.kde_values.append(kde_val)

            # 更新KDE标准化器
            if len(self.kde_values) > 1:
                kde_array = np.array(self.kde_values).reshape(-1, 1)
                self.kde_scaler = StandardScaler()
                self.kde_scaler.fit(kde_array)

            self.results.append({
                'iteration': len(self.results) + 1, 'n_layers': int(x_next[0]),
                'n_nodes': int(x_next[1]), 'activation': self.activation_map[int(x_next[2])],
                'epochs': int(x_next[3]), 'grid_size': int(x_next[4]),
                'learning_rate': float(x_next[5]), 'fitness': float(error),
                'is_initial': False
            })

            self.X_obs.append(self._clip_and_cast(x_next))
            self.y_obs.append(float(error))

            # 重新训练Kriging模型
            X_arr = np.array(self.X_obs, dtype=float)
            self.scaler = StandardScaler()
            X_scaled = self.scaler.fit_transform(X_arr)
            self.kriging.fit(X_scaled, np.array(self.y_obs, dtype=float))

            print(f"Current: {error:.6f}, Best: {min(self.y_obs):.6f}")

            pd.DataFrame(self.results).to_csv('./result/saea_kde_pde5.csv', index=False)

        print(f"\nCompleted! Best error: {min(self.y_obs):.6f}")
        return pd.DataFrame(self.results)


if __name__ == "__main__":
    with open('kde_models_dict_ab_all.pkl', 'rb') as f:
        kde_models_dict = pickle.load(f)

    optimizer = KrigingGAPriorBO(kde_models_dict, beta=1.0)
    results = optimizer.run_optimization(n_bo_iterations=40)