import pandas as pd
import numpy as np
import pickle
from sklearn.neighbors import KernelDensity

activation_encoding = {'sin': 0, 'tanh': 1, 'swish': 2, 'sigmoid': 3, 'relu': 4}
param_ranges = {
    'n_layers': [2, 10],
    'n_nodes': [5, 100],
    'activation': [0, 4],
    'epochs': [5000, 100000],
    'grid_size': [10, 200],
    'learning_rate': [1e-5, 0.1]  # log scale
}


def build_kde_models_from_csv(filename, top_k=60, bandwidths=None, use_rank=False):
    df = pd.read_csv(filename)
    cols = ['final_l2_ab_error', 'num_layers', 'num_nodes',
            'activation_func', 'epochs', 'grid_size', 'learning_rate']
    df = df[cols]
    df['activation_func'] = df['activation_func'].map(activation_encoding)

    # Y = final_l2_ab_error, 筛选 <=1.0，并缩放
    Y = df['final_l2_ab_error'].values
    mask = Y <= 1.0
    df = df[mask].copy()
    df['final_l2_ab_error'] = df['final_l2_ab_error'] * 1e3

    # 排序，取 top_k
    df_sorted = df.sort_values(by='final_l2_ab_error', ascending=True)
    top = df_sorted.head(top_k).copy()

    if use_rank:
        # 用排名替代函数值
        top['final_l2_ab_error'] = np.arange(1, len(top) + 1)

    # 参数矩阵
    param_matrix = top[['num_layers', 'num_nodes', 'activation_func',
                        'epochs', 'grid_size', 'learning_rate']].to_numpy()

    if bandwidths is None:
        bandwidths = [1.0, 1.0, 1.0, 1.0, 1.0, 0.3]

    kde_models = []
    for i in range(6):
        data = param_matrix[:, i][:, None]
        bw = bandwidths[i]
        if i == 5:  # learning_rate 对数处理
            data = np.log10(data)
        kde = KernelDensity(kernel='gaussian', bandwidth=bw).fit(data)
        kde_models.append(kde)
    return kde_models, top


# 自定义带宽
custom_bandwidths = [1, 10, 0.5, 10000, 20, 0.2]

kde_models_dict = {}
all_top_data = []

for i in range(1, 9):
    fname = f'./data/pde{i}_results.csv'

    # 单个 PDE 模型（Top60，正常函数值）
    kde_models, top_df = build_kde_models_from_csv(
        fname, top_k=60, bandwidths=custom_bandwidths, use_rank=False
    )
    kde_models_dict[f'pde{i}'] = kde_models

    # 为 all_pde 收集 Top20（排名）
    df = pd.read_csv(fname)
    cols = ['final_l2_ab_error', 'num_layers', 'num_nodes',
            'activation_func', 'epochs', 'grid_size', 'learning_rate']
    df = df[cols]
    df['activation_func'] = df['activation_func'].map(activation_encoding)

    mask = df['final_l2_ab_error'].values <= 1.0
    df = df[mask].copy()
    df['final_l2_ab_error'] = df['final_l2_ab_error'] * 1e3

    df_sorted = df.sort_values(by='final_l2_ab_error', ascending=True)
    top20 = df_sorted.head(20).copy()
    top20['final_l2_ab_error'] = np.arange(1, len(top20) + 1)  # 排名
    all_top_data.append(top20)

# 合并所有 PDE 的 top20
df_all = pd.concat(all_top_data, axis=0)

# 构建 all pdes 模型
param_matrix_all = df_all[['num_layers', 'num_nodes', 'activation_func',
                           'epochs', 'grid_size', 'learning_rate']].to_numpy()
kde_models_all = []
for i in range(6):
    data = param_matrix_all[:, i][:, None]
    bw = custom_bandwidths[i]
    if i == 5:  # learning_rate
        data = np.log10(data)
    kde = KernelDensity(kernel='gaussian', bandwidth=bw).fit(data)
    kde_models_all.append(kde)

# 保存
kde_models_dict['all pdes'] = kde_models_all
with open('kde_models_dict_ab_all.pkl', 'wb') as f:
    pickle.dump(kde_models_dict, f)


# 采样函数
def sample_params(kde_models, n_samples=5):
    samples = np.zeros((n_samples, 6))
    for i in range(6):
        s = kde_models[i].sample(n_samples).flatten()
        if i == 2:  # activation
            s = np.round(s).astype(int)
            s = np.clip(
                s, param_ranges['activation'][0], param_ranges['activation'][1])
        elif i == 5:  # learning_rate
            s = 10**s
            s = np.clip(
                s, param_ranges['learning_rate'][0], param_ranges['learning_rate'][1])
        else:
            s = np.round(s).astype(int)
            key = list(param_ranges.keys())[i]
            s = np.clip(s, param_ranges[key][0], param_ranges[key][1])
        samples[:, i] = s
    return samples


# 例：采样 all pdes 的参数
sampled_all = sample_params(kde_models_dict['all pdes'], n_samples=5)
print("Sampled parameters from all pdes:")
print(sampled_all)