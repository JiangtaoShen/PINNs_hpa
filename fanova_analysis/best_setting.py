import pandas as pd

# 读取 CSV 文件
df = pd.read_csv('./data/pde4_results.csv')

# 找到前5名 (final_l2_re_error 最小)
top5 = df.nsmallest(5, 'final_l2_re_error')

# 指定参数列
params_columns = ['num_layers', 'num_nodes', 'activation_func', 'epochs', 'grid_size', 'learning_rate']

# 输出结果
for i, row in top5.iterrows():
    best_vector = row[params_columns].values
    min_l2_error = row['final_l2_re_error']
    print(f"第 {i+1} 名:")
    print("最优参数向量：", best_vector)
    print("最小 L2 误差：", min_l2_error)
    print("-" * 40)
