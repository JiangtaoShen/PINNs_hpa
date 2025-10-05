import pandas as pd
import numpy as np

# 参数范围
param_ranges = {
    'n_layers': [2, 10],
    'n_nodes': [5, 100],
    'activation': [0, 4],          # 整数编码
    'epochs': [5000, 100000],
    'grid_size': [10, 200],
    'learning_rate': [1e-5, 0.1]   # 连续，log scale 可选
}

def uniform_sample_params(n_samples=10, log_lr=True):
    samples = np.zeros((n_samples, 6))
    keys = list(param_ranges.keys())
    for i, key in enumerate(keys):
        low, high = param_ranges[key]
        if key == "activation":
            s = np.random.randint(low, high + 1, size=n_samples)
        elif key == "learning_rate":
            if log_lr:
                s = 10 ** np.random.uniform(np.log10(low), np.log10(high), size=n_samples)
            else:
                s = np.random.uniform(low, high, size=n_samples)
        else:
            s = np.random.randint(low, high + 1, size=n_samples)
        samples[:, i] = s
    return samples

# 采样 10 个参数组合
samples = uniform_sample_params(n_samples=10)

# 保存为 CSV，保留一位小数（和示例一致）
df = pd.DataFrame(samples, columns=[
    "n_layers", "n_nodes", "activation", "epochs", "grid_size", "learning_rate"
])
df = df.astype(float)  # 保证写入时带小数点
df.to_csv("uniform_sampled_params.csv", index=False)

print("已保存到 uniform_sampled_params.csv")
print(df)