import numpy as np
import matplotlib.pyplot as plt
import pickle
import seaborn as sns

# Load saved KDE models
with open('kde_models_dict_ab_all.pkl', 'rb') as f:
    kde_models_dict = pickle.load(f)

pde_keys = sorted(kde_models_dict.keys())
param_names = ['depth', 'width', 'act_func', 'n_epo', 'n_sam', 'lr']

param_ranges = {
    'depth': [2, 10],
    'width': [5, 100],
    'act_func': [0, 4],
    'n_epo': [5000, 100000],
    'n_sam': [10, 200],
    'lr': [1e-5, 0.1]  # log scale
}

def compute_kl_divergence(kde_p, kde_q, x_vals, param_name):
    """
    Compute KL divergence KL(P || Q) where P and Q are KDE distributions.
    x_vals: grid points for evaluation
    For lr, density correction for log scale is applied.
    """
    if param_name == 'lr':
        log_p = kde_p.score_samples(np.log10(x_vals))
        p = np.exp(log_p) / (x_vals.flatten() * np.log(10))  # correction for log scale

        log_q = kde_q.score_samples(np.log10(x_vals))
        q = np.exp(log_q) / (x_vals.flatten() * np.log(10))
    else:
        log_p = kde_p.score_samples(x_vals)
        p = np.exp(log_p)

        log_q = kde_q.score_samples(x_vals)
        q = np.exp(log_q)

    # Avoid zeros for numerical stability
    epsilon = 1e-12
    p = np.clip(p, epsilon, None)
    q = np.clip(q, epsilon, None)

    dx = (x_vals[1] - x_vals[0]).flatten()[0]
    kl = np.sum(p * np.log(p / q)) * dx

    kl = max(kl, 0.0)
    return kl

sns.set(style="whitegrid", font_scale=1.2)

for param_idx, param_name in enumerate(param_names):
    print(f"Computing KL divergence matrix for parameter: {param_name}")

    # Prepare evaluation grid for this parameter
    x_min, x_max = param_ranges[param_name]
    if param_name == 'lr':
        x_vals = np.logspace(np.log10(x_min), np.log10(x_max), 2000)[:, None]
    else:
        x_vals = np.linspace(x_min, x_max, 2000)[:, None]

    # Move "all pdes" to the last position
    pde_keys_sorted = [k for k in pde_keys if k != "all pdes"] + ["all pdes"]

    n = len(pde_keys_sorted)
    kl_matrix = np.zeros((n, n))

    # Compute KL divergence for each PDE pair
    for i in range(n):
        for j in range(n):
            kde_p = kde_models_dict[pde_keys_sorted[i]][param_idx]
            kde_q = kde_models_dict[pde_keys_sorted[j]][param_idx]
            kl = compute_kl_divergence(kde_p, kde_q, x_vals, param_name)
            kl_matrix[i, j] = kl

    plt.figure(figsize=(9, 7))
    ax = sns.heatmap(
        kl_matrix,
        annot=True,
        fmt=".3f",
        cmap="rainbow",
        xticklabels=pde_keys_sorted,
        yticklabels=pde_keys_sorted,
        linecolor='white',
        linewidth=0.5,
        vmin=0,
        vmax=0.5
    )
    ax.set_title(f"KL Divergence for {param_name}", fontsize=16, pad=16)
    plt.xticks(rotation=45)
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(f"kl_divergence_{param_name}.pdf")
    plt.close()
    print(f"Saved beautified KL divergence heatmap for {param_name} as kl_divergence_{param_name}.pdf")