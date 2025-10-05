import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# 文件列表
file_paths = [f"./data/pde{i}_results.csv" for i in range(1, 9)]
labels = [f"PDE{i}" for i in range(1, 9)]

all_errors = []

# 读取每个文件的 final_l2_re_error 列
for path in file_paths:
    try:
        df = pd.read_csv(path)
        errors = df["final_l2_re_error"].dropna()  # 去掉可能的 NaN
        all_errors.append(errors)
        print(f"{path}: {len(errors)} rows read")
    except Exception as e:
        print(f"Error reading {path}: {e}")
        all_errors.append([])  # 保持列表长度一致

# 确保每个 PDE 都有数据
if any(len(errors) == 0 for errors in all_errors):
    print("Warning: Some files have no data!")

# 绘制箱型图
plt.figure(figsize=(8, 5))
box = plt.boxplot(all_errors, labels=labels, patch_artist=True, widths=0.4,
                  boxprops=dict(facecolor="lightblue", color="black"),
                  whiskerprops=dict(color="black"),
                  capprops=dict(color="black"),
                  medianprops=dict(color="red", linewidth=2),
                  flierprops=dict(marker="o", markerfacecolor="orange", markersize=6, linestyle="none"))

plt.yscale("log")
plt.ylabel("l2 relative error (log)", fontsize=13)
plt.grid(axis="y", linestyle="--", alpha=0.6, which="both")
plt.tight_layout()

# 保存为 PDF 文件
plt.savefig("./final_l2_re_error_boxplot.pdf")  # 保存为 PDF
plt.show()