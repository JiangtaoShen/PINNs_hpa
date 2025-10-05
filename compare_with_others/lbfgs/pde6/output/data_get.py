import pickle
import os
import pandas as pd

# 文件名前缀和范围
file_prefix = "m"
file_suffix = "_pde6.pkl"
num_files = 7

results = []

for i in range(1, num_files + 1):
    file_name = f"{file_prefix}{i}{file_suffix}"
    if os.path.exists(file_name):
        with open(file_name, "rb") as f:
            data = pickle.load(f)
            train_losses = data["train_losses"]
            training_time = data["training_time"]

            final_loss_dict = train_losses[-1]  # 最后一次 loss
            results.append({
                "pde": "pde6",
                "model": f"m{i}",
                "total_loss": final_loss_dict["total_loss"],
                "final_l2": final_loss_dict["l2_metric"]
            })
    else:
        print(f"⚠️ 文件 {file_name} 不存在，跳过。")

# 保存为 CSV（pde列在最前面）
df = pd.DataFrame(results, columns=["pde", "model", "total_loss", "final_l2"])
output_csv = "summary_results.csv"
df.to_csv(output_csv, index=False)

print(f"✅ 已保存结果到 {output_csv}")
print(df)