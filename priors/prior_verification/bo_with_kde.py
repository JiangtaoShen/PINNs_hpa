import numpy as np
import pandas as pd
import pickle
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, ConstantKernel
from sklearn.preprocessing import StandardScaler
from scipy.optimize import minimize
from scipy.stats import norm
import warnings
import os

from pde1_convection_2d import convection_2d
from pde2_vca_2d import vca_2d
from pde3_heat_2d import heat_2d
from pde4_kdv_2d import kdv_2d
from pde5_ad_2d import ad_2d
from pde6_convection2_2d import convection2_2d

warnings.filterwarnings('ignore')


class SimplePriorBO:
    def __init__(self, kde_models_dict, beta=1.0):
        self.kde_models = kde_models_dict.get('all pdes', kde_models_dict)
        self.beta = beta
        self.activation_map = {0: 'sin', 1: 'tanh', 2: 'swish', 3: 'sigmoid', 4: 'relu'}

        # placeholders
        self.results = []
        self.X_obs = []
        self.y_obs = []

        # scaler 和 gp 后面会在初始化数据后创建/拟合
        self.scaler = None
        self.gp = None

    def _clip_and_cast(self, x):
        x = np.array(x, dtype=float).flatten()
        x[0] = int(np.clip(np.round(x[0]), 2, 10))          # n_layers
        x[1] = int(np.clip(np.round(x[1]), 5, 100))         # n_nodes
        x[2] = int(np.clip(np.round(x[2]), 0, 4))           # activation index
        x[3] = int(np.clip(np.round(x[3]), 5000, 100000))   # epochs
        x[4] = int(np.clip(np.round(x[4]), 10, 200))        # grid_size
        x[5] = float(np.clip(x[5], 1e-5, 0.1))              # learning_rate
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
            result = convection2_2d(
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

    def acquisition_function(self, x, gamma_n):

        if len(self.X_obs) == 0 or self.gp is None:
            return 1.0


        x_orig = self._clip_and_cast(x)


        try:
            x_scaled = self._scale_point(x_orig)
            mu, sigma = self.gp.predict(x_scaled, return_std=True)
            mu = mu.ravel()[0]
            sigma = float(sigma.ravel()[0])
        except Exception as e:
            print(f"[acquisition_function] gp predict failed: {e}")
            return 0.0

        f_best = np.min(self.y_obs)

        # Expected Improvement for minimization: improvement = f_best - mu
        imp = f_best - mu
        if sigma <= 0:
            Z = 0.0
        else:
            Z = imp / sigma
        ei = imp * norm.cdf(Z) + sigma * norm.pdf(Z)
        if ei < 0:
            ei = 0.0

        # Prior 权重（使用幂次 gamma_n）
        prior = self.evaluate_prior(x_orig) ** gamma_n

        return ei * prior

    def optimize_acquisition(self, gamma_n, n_restarts=20, n_local=6):
        """优化 acquisition：先随机搜索挑选好的起点，再对这些起点做局部 L-BFGS-B"""
        best_x = None
        best_val = -np.inf

        # 1) 随机采样作为候选起点
        candidates = []
        for _ in range(max(200, n_restarts * 10)):
            cand = [
                np.random.randint(2, 11),        # n_layers
                np.random.randint(5, 101),       # n_nodes
                np.random.randint(0, 5),         # activation
                np.random.randint(5000, 100001), # epochs
                np.random.randint(10, 201),      # grid_size
                10 ** np.random.uniform(-5, -1)  # learning_rate
            ]
            val = self.acquisition_function(cand, gamma_n)
            candidates.append((val, cand))

        # 2) 选取 top 起点
        candidates.sort(key=lambda t: -t[0])
        start_points = [c[1] for c in candidates[:n_restarts]]

        # bounds（保持与变量含义一致）
        bounds = [(2, 10), (5, 100), (0, 4), (5000, 100000), (10, 200), (1e-5, 0.1)]

        def neg_acq_for_minimize(x):
            x_local = np.array(x, dtype=float).flatten()
            # clip/round discrete vars for evaluation
            x_clipped = self._clip_and_cast(x_local)
            val = self.acquisition_function(x_clipped, gamma_n)
            return -float(val)

        for sp in start_points[:n_local]:
            try:
                res = minimize(neg_acq_for_minimize, sp, bounds=bounds, method='L-BFGS-B',
                               options={'maxiter': 200})
                if res.success:
                    acq_val = -res.fun
                else:
                    acq_val = self.acquisition_function(sp, gamma_n)

                if acq_val > best_val:
                    best_val = acq_val
                    best_x = res.x if (hasattr(res, 'x') and res.x is not None) else sp
            except Exception as e:
                print(f"[optimize_acquisition] local optimize failed at start {sp}: {e}")
                continue

        if best_x is None:
            best_x = candidates[0][1]

        best_x = self._clip_and_cast(best_x)
        return best_x

    def run_optimization(self, n_bo_iterations=40):
        self.results = []
        self.X_obs = []
        self.y_obs = []

        csv_path = 'initial_samples_kde.csv'
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"Initial sample file not found: {csv_path}")

        print("Reading initial samples...")
        initial_data = pd.read_csv(csv_path)

        for i, row in initial_data.iterrows():
            params = [row['n_layers'], row['n_nodes'], row['activation'],
                      row['epochs'], row['grid_size'], row['learning_rate']]
            error = self.evaluate_pinn(params)

            self.results.append({
                'iteration': i + 1, 'n_layers': int(params[0]), 'n_nodes': int(params[1]),
                'activation': self.activation_map[int(params[2])], 'epochs': int(params[3]),
                'grid_size': int(params[4]), 'learning_rate': params[5],
                'fitness': error, 'is_initial': True
            })

            self.X_obs.append(self._clip_and_cast(params))
            self.y_obs.append(float(error))

            print(f"Initial {i + 1}: Error = {error:.6f}")

            pd.DataFrame(self.results).to_csv('pinn_prior_bo_results.csv', index=False)

        kernel = ConstantKernel(1.0) * Matern(length_scale=0.5, nu=1.5)
        self.gp = GaussianProcessRegressor(kernel=kernel, alpha=1e-6, normalize_y=True)

        X_arr = np.array(self.X_obs, dtype=float)
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X_arr)
        self.gp.fit(X_scaled, np.array(self.y_obs, dtype=float))

        print("\nStarting BO optimization...")
        for it in range(n_bo_iterations):
            gamma_n = self.beta / max(1, len(self.X_obs))
            print(f"Iteration {len(self.results) + 1}, γ_n = {gamma_n:.6f}")

            x_next = self.optimize_acquisition(gamma_n)
            error = self.evaluate_pinn(x_next)

            self.results.append({
                'iteration': len(self.results) + 1, 'n_layers': int(x_next[0]),
                'n_nodes': int(x_next[1]), 'activation': self.activation_map[int(x_next[2])],
                'epochs': int(x_next[3]), 'grid_size': int(x_next[4]),
                'learning_rate': float(x_next[5]), 'fitness': float(error),
                'is_initial': False
            })

            self.X_obs.append(self._clip_and_cast(x_next))
            self.y_obs.append(float(error))

            X_arr = np.array(self.X_obs, dtype=float)
            self.scaler = StandardScaler()
            X_scaled = self.scaler.fit_transform(X_arr)
            self.gp.fit(X_scaled, np.array(self.y_obs, dtype=float))

            print(f"Current: {error:.6f}, Best: {min(self.y_obs):.6f}")

            pd.DataFrame(self.results).to_csv('./result/bo_kde_pde6.csv', index=False)

        print(f"\nCompleted! Best error: {min(self.y_obs):.6f}")
        return pd.DataFrame(self.results)


if __name__ == "__main__":
    with open('kde_models_dict_ab_all.pkl', 'rb') as f:
        kde_models_dict = pickle.load(f)

    optimizer = SimplePriorBO(kde_models_dict, beta=1.0)
    results = optimizer.run_optimization(n_bo_iterations=40)