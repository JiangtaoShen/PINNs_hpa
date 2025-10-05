import pickle
import numpy as np
import pandas as pd
from tqdm import tqdm

# Initialize empty array
l2_re_errors = np.zeros((2000, 1))

for i in tqdm(range(2000), desc="Processing files"):
    file_path = f'../data/pde5_burgers_2d/burgers_2d_{i}.pkl'
    with open(file_path, 'rb') as f:
        data = pickle.load(f)
    # l2_re_errors[i, 0] = data['final_l2_re_error']
    l2_re_errors[i, 0] = data['runtime']

# Save to CSV
# output_path = '../data/pde5_l2_re_errors.csv'
output_path = '../data/pde5_runtime.csv'
pd.DataFrame(l2_re_errors).to_csv(output_path, index=False, header=False)
print(f"\nSuccessfully saved L2 relative errors to {output_path}")