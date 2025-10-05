import numpy as np
import matplotlib.pyplot as plt
import pickle
import pandas as pd

# Load saved KDE models
with open('kde_models_dict_ab_all.pkl', 'rb') as f:
    kde_models_dict = pickle.load(f)

print("Available KDE keys:", kde_models_dict.keys())

# 改后的参数名称
param_names = ['depth', 'width', 'act_func', 'n_epo', 'n_sam', 'lr']

param_ranges = {
    'depth': [2, 10],
    'width': [5, 100],
    'act_func': [0, 4],
    'n_epo': [5000, 100000],
    'n_sam': [10, 200],
    'lr': [1e-5, 0.1]  # log scale
}

# pde1-8 + all pdes
pde_keys = sorted([k for k in kde_models_dict.keys() if k.startswith("pde")]) + ["all pdes"]

# Load real samples (top 60) for each PDE into a dictionary
real_samples_dict = {}
activation_encoding = {'sin': 0, 'tanh': 1, 'swish': 2, 'sigmoid': 3, 'relu': 4}

# 先收集所有数据，用于 all_data
all_samples_list = []

for pde in pde_keys:
    if pde == "all pdes":
        # 合并前面收集的所有样本
        real_samples_dict[pde] = np.vstack(all_samples_list)
    else:
        i = int(pde.replace('pde', ''))
        df = pd.read_csv(f'./data/pde{i}_results.csv')
        cols = ['final_l2_re_error', 'num_layers', 'num_nodes', 'activation_func',
                'epochs', 'grid_size', 'learning_rate']
        df = df[cols]
        df['activation_func'] = df['activation_func'].map(activation_encoding)
        df_sorted = df.sort_values(by='final_l2_re_error', ascending=True)
        top = df_sorted.head(60)
        # 对应顺序：depth, width, act_func, epochs, samples, lr
        param_matrix = top[['num_layers', 'num_nodes', 'activation_func',
                            'epochs', 'grid_size', 'learning_rate']].to_numpy()
        real_samples_dict[pde] = param_matrix
        all_samples_list.append(param_matrix)

# 绘图
fig, axes = plt.subplots(len(pde_keys), len(param_names), figsize=(24, 20))
fig.subplots_adjust(hspace=0.4, wspace=0.3)

bins_num = 10  # Number of bins for histogram

for row, pde in enumerate(pde_keys):
    for col, param in enumerate(param_names):
        ax = axes[row, col]
        x_min, x_max = param_ranges[param]

        # X-axis values
        if param == 'lr':
            x_vals = np.logspace(np.log10(x_min), np.log10(x_max), 500)[:, None]
            bins = np.logspace(np.log10(x_min), np.log10(x_max), bins_num)
        else:
            x_vals = np.linspace(x_min, x_max, 500)[:, None]
            bins = np.linspace(x_min, x_max, bins_num)

        kde = kde_models_dict[pde][col]

        # Plot histogram of real samples
        samples = real_samples_dict[pde][:, col].copy()
        samples = np.clip(samples, x_min, x_max)
        ax.hist(samples, bins=bins, density=True, alpha=0.3, color='C0')

        # Plot KDE curve
        if param == 'lr':
            log_density = kde.score_samples(np.log10(x_vals))
            density = np.exp(log_density) / (x_vals.flatten() * np.log(10))  # correction for log scale
        else:
            log_density = kde.score_samples(x_vals)
            density = np.exp(log_density)

        ax.plot(x_vals.flatten(), density, color='red')  # KDE curve in red

        # Titles and labels
        if row == 0:
            ax.set_title(param, fontsize=16)
        if col == 0:
            ax.set_ylabel(pde, fontsize=16)

        if param == 'lr':
            ax.set_xscale('log')

plt.savefig("kde_hist_real_samples_all.pdf", format="pdf")
plt.close(fig)

print("Plot saved as kde_hist_real_samples_all.pdf")