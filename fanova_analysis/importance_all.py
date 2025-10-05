import os
import pickle
import numpy as np
import pandas as pd
from fanova import fANOVA
from ConfigSpace import ConfigurationSpace
from ConfigSpace.hyperparameters import UniformFloatHyperparameter
import fanova.visualizer
import matplotlib.pyplot as plt
from plot_func import plot_fanova_importances


# load parameter samples
df_x = pd.read_csv('./data/parameter_samples.csv')
activation_map = {act: i for i, act in enumerate(df_x['activation'].unique())}
df_x['activation'] = df_x['activation'].map(activation_map)
X = df_x.to_numpy()
X = np.tile(X, (8, 1))
print("Activation function encoding: ", activation_map)

# # 用循环读取 8 个 pde 文件
# Y_list = []
# for i in range(1, 9):
#     df_y = pd.read_csv(f'./data/pde{i}_results.csv')
#     Y_list.append(df_y['final_l2_ab_error'].values)
#
# # 拼接
# Y = np.concatenate(Y_list, axis=0)
#
# # log10 + 归一化到 [0,100]
# mask = Y <= 1.0
# X = X[mask]
# Y = Y[mask]
# Y = Y*1e3
#
# cs = ConfigurationSpace()
# cs.add(UniformFloatHyperparameter("depth", lower=2, upper=10))
# cs.add(UniformFloatHyperparameter("width", lower=5, upper=100))
# cs.add(UniformFloatHyperparameter("act_func", lower=0, upper=4))
# cs.add(UniformFloatHyperparameter("n_epo", lower=5000, upper=100000))
# cs.add(UniformFloatHyperparameter("n_sam", lower=10, upper=200))
# cs.add(UniformFloatHyperparameter("lr", lower=1e-5, upper=0.1, log=True))
#
# X = pd.DataFrame(X, columns=["depth", "width", "act_func", "n_epo", "n_sam", "lr"])
#
# # modeling
# f = fANOVA(X, Y, config_space=cs)
#
# # plot single landscape
# vis = fanova.visualizer.Visualizer(f, cs, f"./plots/all_data")
# for param in X.columns:
#     print(f"Plotting marginal for {param}...")
#     vis.plot_marginal(param, show=False)
#     plt.savefig(f"./plots/all_data/marginal_{param}.pdf", format='pdf', dpi=300, pad_inches=0)
#     plt.close()
#
# # calculate all single and pairs importance
# plot_fanova_importances(f, X, top_k_interactions=5, save_path=f"./plots/all_data")
#
# # plot pairs landscape
# pairs = [("depth", "width"), ("n_epo", "lr")]
# for p1, p2 in pairs:
#     figs = vis.plot_pairwise_marginal((p1, p2), show=False)
#     for idx, fig in enumerate(figs):
#         fig.savefig(f"./plots/all_data/pairwise_{p1}_{p2}_{idx+2}d.pdf",
#                     format='pdf', dpi=300, bbox_inches='tight', pad_inches=0.1)
#         plt.close(fig)
#
# print(f"All plots saved in ./plots/all_data")