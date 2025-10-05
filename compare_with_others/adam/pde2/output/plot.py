import pickle
import os
import matplotlib.pyplot as plt

# -----------------------------
# 配置
# -----------------------------
file_name = "m7_pde2.pkl"
model_label = "PINNsFormer"

# -----------------------------
# 读取 L2 曲线
# -----------------------------
l2_curve = []

if os.path.exists(file_name):
    with open(file_name, "rb") as f:
        data = pickle.load(f)
        train_losses = data["train_losses"]
        l2_curve = [d["l2_metric"] for d in train_losses]
    print(f"加载成功: {file_name}, 共 {len(l2_curve)} 个点")
else:
    raise FileNotFoundError(f"⚠️ 文件 {file_name} 不存在，请确认路径")

# -----------------------------
# 绘图
# -----------------------------
fig, ax = plt.subplots(figsize=(8, 5))

ax.plot(range(len(l2_curve)), l2_curve, label=model_label, color="blue", linewidth=2)

ax.set_yscale("log")
ax.set_xlabel("Epoch")
ax.set_ylabel("L2 Relative Error")
ax.grid(True)
ax.legend(loc='best')

plt.tight_layout()
# plt.savefig("m7_l2_convergence.pdf", dpi=300)
plt.show()
