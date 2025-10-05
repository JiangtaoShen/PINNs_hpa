import pickle
import pandas as pd
import os

folder_path = './pde8_helmholtz_2d'
all_data = []

for i in range(2000):
    file_path = os.path.join(folder_path, f'helmholtz_2d_{i}.pkl')
    if os.path.exists(file_path):
        with open(file_path, 'rb') as f:
            data = pickle.load(f)
        all_data.append(data)
    else:
        print(f"文件不存在: {file_path}")

# 转成 DataFrame
df = pd.DataFrame(all_data)

# 保存为 CSV
output_csv = 'pde8_results.csv'
df.to_csv(output_csv, index=False, encoding='utf-8-sig')

print(f"已保存到 {output_csv}，共 {len(df)} 条记录。")