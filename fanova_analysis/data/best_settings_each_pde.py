import pandas as pd

# 读取CSV文件
df = pd.read_csv('pde8_results.csv')

# 找到最小final_l2_re_error的行
min_idx = df['final_l2_re_error'].idxmin()
best_row = df.loc[min_idx]
print(best_row)

# 输出结果
print(f"最小 final_l2_re_error: {best_row['final_l2_re_error']}")
print(f"num_layers: {best_row['num_layers']}")
print(f"num_nodes: {best_row['num_nodes']}")
print(f"activation_func: {best_row['activation_func']}")
print(f"epochs: {best_row['epochs']}")
print(f"grid_size: {best_row['grid_size']}")
print(f"learning_rate: {best_row['learning_rate']}")