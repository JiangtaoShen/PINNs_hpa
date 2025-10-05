import pickle
import os
import matplotlib.pyplot as plt
import numpy as np

# -----------------------------
# 配置
# -----------------------------
file_prefix = "m"
file_suffix = "_pde2.pkl"
num_files = 7

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
# 读取L2曲线
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
# Adam和LBFGS的epoch数
# -----------------------------
adam_epochs = len(l2_curves.get("m1", []))
lbfgs_epochs = len(l2_curves.get("m2", [])) if "m2" in l2_curves else 1000

def scale_epochs(curve, target_len):
    """把LBFGS曲线横坐标缩放到与Adam一致"""
    n = len(curve)
    x = np.linspace(0, target_len, n)
    return x, curve

# -----------------------------
# 绘图
# -----------------------------
fig, ax1 = plt.subplots(figsize=(8, 5))

n = 20  # 每隔 n 个点取一个，可以自行调整

# 绘制曲线
for i in range(1, num_files + 1):
    key = f"m{i}"
    if key in l2_curves:
        label = label_map.get(key, key)
        if i == 1:  # M1 使用 Adam，下采样
            x_vals = np.arange(adam_epochs)[::n]
            y_vals = l2_curves[key][::n]
            ax1.plot(x_vals, y_vals, label=label, color="black")
        else:       # M2-M7 使用 LBFGS
            x, y = scale_epochs(l2_curves[key], adam_epochs)
            ax1.plot(x, y, label=label)

ax1.set_yscale("log")
ax1.set_xlabel("Epoch (Adam)")
ax1.set_ylabel("L2 Retalive Error")
ax1.grid(True)
ax1.legend(loc='best')

# 上方坐标轴 (LBFGS)
def adam_to_lbfgs(x):
    return x / adam_epochs * lbfgs_epochs
def lbfgs_to_adam(x):
    return x / lbfgs_epochs * adam_epochs

ax2 = ax1.secondary_xaxis('top', functions=(adam_to_lbfgs, lbfgs_to_adam))
ax2.set_xlabel("Epoch (LBFGS)")

plt.tight_layout()
plt.savefig("l2_dual_axis.pdf", dpi=300)
plt.show()