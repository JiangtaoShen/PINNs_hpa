import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm
import seaborn as sns
from collections import OrderedDict
import os
from itertools import combinations


def plot_fanova_importances(f, X, top_k_interactions=3, save_path="./plots/all_data"):
    param_names = [param.name for param in f.cs_params]
    n_params = len(param_names)

    print("Computing individual parameter importances...")
    individual_tree_data = OrderedDict()

    for i, param_name in enumerate(tqdm(param_names, desc="Individual params")):
        try:
            f._fANOVA__compute_marginals((i,))
            tree_importances = []
            for t in range(f.n_trees):
                if f.trees_total_variance[t] > 0:
                    tree_imp = f.V_U_individual[(i,)][t] / f.trees_total_variance[t]
                    tree_importances.append(max(0, tree_imp))  # Ensure non-negative
            individual_tree_data[param_name] = tree_importances
        except Exception as e:
            print(f"Warning: Could not compute importance for {param_name}: {e}")
            individual_tree_data[param_name] = [0.0]

    print("Computing pairwise interaction importances...")
    pairwise_tree_data = OrderedDict()
    pairs = list(combinations(range(n_params), 2))

    for i, j in tqdm(pairs, desc="Pairwise interactions"):
        param1, param2 = param_names[i], param_names[j]
        try:
            f._fANOVA__compute_marginals((i, j))
            tree_importances = []
            for t in range(f.n_trees):
                if f.trees_total_variance[t] > 0:
                    tree_imp = f.V_U_individual[(i, j)][t] / f.trees_total_variance[t]
                    tree_importances.append(max(0, tree_imp))
            pairwise_tree_data[f"{param1}×{param2}"] = tree_importances
        except Exception as e:
            print(f"Warning: Could not compute interaction for {param1}, {param2}: {e}")
            pairwise_tree_data[f"{param1}×{param2}"] = [0.0]

    print(f"Selecting top {top_k_interactions} interactions...")
    top_interactions = sorted(
        pairwise_tree_data.items(),
        key=lambda x: np.mean(x[1]),
        reverse=True
    )[:top_k_interactions]

    all_labels = []
    all_data = []
    colors = []

    for param_name, data in individual_tree_data.items():
        all_labels.append(param_name)
        all_data.append(data)
        colors.append('skyblue')

    for interaction_name, data in top_interactions:
        all_labels.append(interaction_name)
        all_data.append(data)
        colors.append('salmon')

    plt.figure(figsize=(9, 6))

    plt.rcParams.update({'font.size': 14})

    boxplot = plt.boxplot(
        all_data,
        labels=all_labels,
        patch_artist=True,
        showmeans=True,
        meanline=True,
        widths=0.4
    )

    for patch, color in zip(boxplot['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    plt.ylabel('Variance Contribution', fontsize=14)
    plt.xticks(rotation=45, ha='right', fontsize=12)
    plt.yticks(fontsize=12)
    plt.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()

    # Create directory if it doesn't exist
    os.makedirs(save_path, exist_ok=True)

    # Generate full file paths
    plot_file_path = os.path.join(save_path, "fanova_importances.pdf")
    csv_file_path = os.path.join(save_path, "all_importance_data.csv")

    plt.savefig(plot_file_path, dpi=300, bbox_inches='tight')
    plt.show()

    importance_records = []
    for param_name, data in individual_tree_data.items():
        importance_records.append({
            'Type': 'Individual',
            'Name': param_name,
            'Mean Importance': np.mean(data)
        })
    for interaction_name, data in pairwise_tree_data.items():
        importance_records.append({
            'Type': 'Pairwise',
            'Name': interaction_name,
            'Mean Importance': np.mean(data)
        })

    importance_df = pd.DataFrame(importance_records)

    total_importance = importance_df['Mean Importance'].sum()
    total_row = pd.DataFrame({
        'Type': ['Total'],
        'Name': ['Sum of All Importances'],
        'Mean Importance': [total_importance]
    })
    importance_df = pd.concat([importance_df, total_row], ignore_index=True)

    importance_df.to_csv(csv_file_path, index=False)
    print(f"Plot saved to: {plot_file_path}")
    print(f"All importances saved to: {csv_file_path}")
    print(f"Total importance sum: {total_importance:.4f}")

    return {
        'individual_importances': {k: np.mean(v) for k, v in individual_tree_data.items()},
        'pairwise_importances': {k: np.mean(v) for k, v in pairwise_tree_data.items()},
        'top_interactions': {k: np.mean(v) for k, v in top_interactions}
    }