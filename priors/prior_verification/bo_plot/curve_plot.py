import pandas as pd
import matplotlib.pyplot as plt
import os


def plot_convergence(pde_id: int, n_iter: int = 50):
    # 构造文件名和对应的标签、样式
    files_info = {
        f"./data/bo_pde{pde_id}.csv": {
            "label": "BO",
            "marker": "s",
            "color": "#2E86AB"
        },
        f"./data/bo_kde_pde{pde_id}.csv": {
            "label": "BO with prior",
            "marker": "o",
            "color": "#F24236"
        },
        f"./data/saea_pde{pde_id}.csv": {
            "label": "SAEA",
            "marker": "^",
            "color": "#228B22"
        },
        f"./data/saea_kde_pde{pde_id}.csv": {
            "label": "SAEA with prior",
            "marker": "D",
            "color": "#FF8C00"
        }
    }

    # 存储成功读取的数据
    data_dict = {}

    # 尝试读取每个文件
    for file_path, style_info in files_info.items():
        if os.path.exists(file_path):
            try:
                df = pd.read_csv(file_path)
                if "fitness" in df.columns and len(df) > 0:
                    # 提取前 n_iter 个 fitness
                    fitness = df["fitness"].head(n_iter).reset_index(drop=True)
                    # 计算收敛曲线（best so far）
                    best_fitness = fitness.cummin()
                    data_dict[file_path] = {
                        "best_fitness": best_fitness,
                        "style": style_info
                    }
                    print(f"成功读取: {file_path} (数据点数: {len(fitness)})")
                else:
                    print(f"警告: {file_path} 中没有 'fitness' 列或数据为空")
            except Exception as e:
                print(f"读取 {file_path} 时出错: {e}")
        else:
            print(f"文件不存在: {file_path}")

    # 检查是否有可用数据
    if not data_dict:
        print(f"错误: 没有找到 PDE{pde_id} 的有效数据文件")
        return

    # 绘图
    plt.figure(figsize=(6, 4))

    # 绘制所有可用的数据
    for file_path, data_info in data_dict.items():
        best_fitness = data_info["best_fitness"]
        style = data_info["style"]

        plt.plot(best_fitness,
                 marker=style["marker"],
                 color=style["color"],
                 markersize=3.5,
                 label=style["label"])

    plt.xlabel("Iteration")
    plt.ylabel("l2 relative error (log)")
    plt.yscale("log")
    plt.title(f"PDE{pde_id}")
    plt.legend()
    plt.grid(True)
    plt.savefig(f'convergence_plot_pde{pde_id}.pdf', format='pdf', bbox_inches='tight', dpi=300)
    plt.show()

    print(f"成功绘制 PDE{pde_id} 的收敛图，包含 {len(data_dict)} 个数据集")


# 使用示例
plot_convergence(4)