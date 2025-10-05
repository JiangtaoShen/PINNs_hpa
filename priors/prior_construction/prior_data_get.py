import pandas as pd
import os

# 创建结果列表
all_top_20 = []

# 处理8个PDE文件
for i in range(1, 9):
    file_path = f'./data/pde{i}_results.csv'

    # 检查文件是否存在
    if not os.path.exists(file_path):
        print(f"警告: 文件 {file_path} 不存在，跳过")
        continue

    try:
        # 读取CSV文件
        df = pd.read_csv(file_path)

        # 按final_l2_re_error升序排序，取前20个最小的
        top_20 = df.sort_values('final_l2_re_error').head(20).reset_index(drop=True)

        # 添加fitness列，值为排名（1-20）
        top_20['fitness'] = top_20.index + 1

        # 选择需要的列（不包含pde_id和final_l2_re_error）
        result = top_20[['num_layers', 'num_nodes', 'activation_func', 'epochs',
                         'grid_size', 'learning_rate', 'fitness']]

        # 添加到总列表
        all_top_20.append(result)

        print(f"已处理 pde{i}_results.csv，找到 {len(result)} 个数据点")

    except Exception as e:
        print(f"处理文件 {file_path} 时出错: {e}")

# 合并所有数据
if all_top_20:
    combined_data = pd.concat(all_top_20, ignore_index=True)

    print(f"\n总共合并了 {len(combined_data)} 个数据点")
    print(f"来自 {len(all_top_20)} 个PDE文件")

    # 保存合并结果到CSV文件
    combined_data.to_csv('./data/all_pdes_good.csv', index=False)

    print("合并后的数据已保存到 './data/all_pdes_good.csv'")

    # 打印数据概览
    print("\n数据概览:")
    print(combined_data.info())
    print("\n前10行数据预览:")
    print(combined_data.head(10))

else:
    print("没有找到任何可处理的数据")