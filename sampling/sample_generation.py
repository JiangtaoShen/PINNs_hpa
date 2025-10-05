import numpy as np
import pandas as pd
from scipy.stats import qmc

# 设置随机种子确保可复现性
np.random.seed(42)

# 定义参数范围
param_ranges = {
    'n_layers': [2, 10],  # 整数
    'n_nodes': [5, 100],  # 整数
    'activation': ['tanh', 'relu', 'sigmoid', 'sin', 'swish'],  # 分类
    'epochs': [5000, 100000],  # 整数
    'grid_size': [10, 200],  # 整数
    'learning_rate': [1e-5, 0.1]  # 对数尺度
}

# 生成拉丁超立方样本 (6维)
sampler = qmc.LatinHypercube(d=6)
sample = sampler.random(n=2000)

# 转换样本到实际参数空间
params = []


# 数值型参数的转换函数
def scale_parameter(value, min_val, max_val, is_int=False, log_scale=False):
    if log_scale:
        log_min = np.log10(min_val)
        log_max = np.log10(max_val)
        scaled = 10 ** (value * (log_max - log_min) + log_min)
    else:
        scaled = value * (max_val - min_val) + min_val

    return int(round(scaled)) if is_int else scaled


# 处理每个样本
for s in sample:
    # 1. 网络层数 (整数)
    n_layers = scale_parameter(s[0], *param_ranges['n_layers'], is_int=True)
    # 2. 节点数 (整数)
    n_nodes = scale_parameter(s[1], *param_ranges['n_nodes'], is_int=True)
    # 3. 激活函数 (分类)
    act_idx = int(s[2] * len(param_ranges['activation']))
    activation = param_ranges['activation'][min(act_idx, len(param_ranges['activation']) - 1)]
    # 4. 训练轮数 (整数)
    epochs = scale_parameter(s[3], *param_ranges['epochs'], is_int=True)
    # 5. 网格尺寸 (整数)
    grid_size = scale_parameter(s[4], *param_ranges['grid_size'], is_int=True)
    # 6. 学习率 (对数尺度)
    lr = scale_parameter(s[5], *param_ranges['learning_rate'], log_scale=True)
    params.append([n_layers, n_nodes, activation, epochs, grid_size, lr])

# 转换为DataFrame
df = pd.DataFrame(params, columns=[
    'n_layers', 'n_nodes', 'activation', 'epochs', 'grid_size', 'learning_rate'
])

# 保存到CSV文件
df.to_csv('parameter_samples.csv', index=False)

# 显示前5个样本
print("前5个参数样本:")
print(df.head())