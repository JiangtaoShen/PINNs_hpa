import pickle
import os
import matplotlib.pyplot as plt

# -----------------------------
# 配置
# -----------------------------
file_prefix = "m"
file_suffix = "_pde8.pkl"
num_files = 7
plot_step = 20  # 每隔 n 个点绘制一次

# M1-M7模型对应名称
label_map = {
    "m1": "PINN*",
    "m2": "PINN",
    "m3": "Ro-PINN",
    "m4": "FLS",
    "m5": "QRes",
    "m6": "KAN",
    "m7": "PINNsFormer",
}

# -----------------------------
# 读取 L2 曲线
# -----------------------------
l2_curves = {}

for i in range(1, num_files + 1):
    file_name = f"{file_prefix}{i}{file_suffix}"
    if os.path.exists(file_name):
        with open(file_name, "rb") as f:
            data = pickle.load(f)
            train_losses = data["train_losses"]
            l2_metric = [d["l2_metric"] for d in train_losses]
            l2_curves[f"m{i}"] = l2_metric
    else:
        print(f"⚠️ 文件 {file_name} 不存在，跳过。")

# -----------------------------
# 绘图
# -----------------------------
fig, ax = plt.subplots(figsize=(8, 5))

for i in range(1, num_files + 1):
    key = f"m{i}"
    if key in l2_curves:
        label = label_map.get(key, key)
        epochs = range(len(l2_curves[key]))
        y_vals = l2_curves[key]

        # 只绘制间隔 plot_step 的点
        epochs_sampled = epochs[::plot_step]
        y_sampled = y_vals[::plot_step]

        if key == "m1":  # PINN* 黑色
            ax.plot(epochs_sampled, y_sampled, label=label, color="black")
        else:
            ax.plot(epochs_sampled, y_sampled, label=label, alpha=0.8)

ax.set_yscale("log")
ax.set_xlabel("Epoch")
ax.set_ylabel("L2 Relative Error")
ax.grid(True)
ax.legend(loc='best')

plt.tight_layout()
plt.savefig("l2_convergence.pdf", dpi=300)
plt.show()