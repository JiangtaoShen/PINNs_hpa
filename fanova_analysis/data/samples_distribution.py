import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({
    'font.size': 14,
    'axes.titlesize': 14,
    'axes.labelsize': 14,
    'xtick.labelsize': 14,
    'ytick.labelsize': 14
})

# Load data
df = pd.read_csv('../data/parameter_samples.csv')

# 1. Depth (n_layers)
plt.figure(figsize=(8, 6))
plt.hist(df['n_layers'], bins=range(2, 12), alpha=0.7, color='skyblue', edgecolor='black')
plt.title('Depth Distribution')
plt.xlabel('Number of Layers')
plt.ylabel('Count')
plt.xticks(range(2, 11))
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('./Depth_Distribution.pdf')
plt.close()

# 2. Width (n_nodes)
plt.figure(figsize=(8, 6))
plt.hist(df['n_nodes'], bins=20, alpha=0.7, color='skyblue', edgecolor='black')
plt.title('Width Distribution')
plt.xlabel('Number of Nodes')
plt.ylabel('Count')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('./Width_Distribution.pdf')
plt.close()

# 3. Activation Function
plt.figure(figsize=(8, 6))
act_counts = df['activation'].value_counts()
plt.bar(act_counts.index, act_counts.values, alpha=0.7, color='skyblue', edgecolor='black')
plt.title('Activation Function Distribution')
plt.xlabel('Activation Function')
plt.ylabel('Count')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('./Activation_Distribution.pdf')
plt.close()

# 4. Epochs
plt.figure(figsize=(8, 6))
plt.hist(df['epochs'], bins=20, alpha=0.7, color='skyblue', edgecolor='black')
plt.title('Epochs Distribution')
plt.xlabel('Epochs')
plt.ylabel('Count')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('./Epochs_Distribution.pdf')
plt.close()

# 5. Samples (grid_size)
plt.figure(figsize=(8, 6))
plt.hist(df['grid_size'], bins=20, alpha=0.7, color='skyblue', edgecolor='black')
plt.title('Sample Size Distribution')
plt.xlabel('Grid Size')
plt.ylabel('Count')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('./GridSize_Distribution.pdf')
plt.close()

# 6. Learning Rate (binned statistics)
plt.figure(figsize=(8, 6))
lr = df['learning_rate']
bins = [1e-5, 1e-4, 1e-3, 1e-2, 1e-1]
labels = ['[1e-5,1e-4)', '[1e-4,1e-3)', '[1e-3,1e-2)', '[1e-2,1e-1]']
counts, _ = np.histogram(lr, bins=bins)
plt.bar(labels, counts, color='skyblue', alpha=0.8, edgecolor='black')
plt.title('Learning Rate Distribution')
plt.xlabel('Learning Rate')
plt.ylabel('Count')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('./LearningRate_Distribution.pdf')
plt.close()