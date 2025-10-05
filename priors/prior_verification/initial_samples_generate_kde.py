import pickle
import pandas as pd
import numpy as np

# 参数范围和采样函数（保持和原代码一致）
param_ranges = {
    'n_layers': [2, 10],
    'n_nodes': [5, 100],
    'activation': [0, 4],
    'epochs': [5000, 100000],
    'grid_size': [10, 200],
    'learning_rate': [1e-5, 0.1]  # log scale
}

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

# 1. 读取保存的 KDE 模型
with open("kde_models_dict_ab_all.pkl", "rb") as f:
    kde_models_dict = pickle.load(f)

# 2. 从 "all pdes" KDE 模型中采样 10 个点
samples = sample_params(kde_models_dict["all pdes"], n_samples=10)

# 3. 保存为 CSV
df = pd.DataFrame(samples, columns=[
    "n_layers", "n_nodes", "activation", "epochs", "grid_size", "learning_rate"
])
df.to_csv("sampled_params.csv", index=False)

print("已保存到 sampled_params.csv")
print(df)
