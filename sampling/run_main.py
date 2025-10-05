import pandas as pd
import pickle
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm
from pathlib import Path
from pde8_helmholtz_2d import helmholtz_2d


def run_simulation(args):
    i, sample_i, device = args
    # Define the output file path
    output_file = Path(f'./output/pde8_helmholtz_2d/helmholtz_2d_{i}.pkl')

    # Skip if the file already exists
    if output_file.exists():
        return i

    # Run the simulation
    result = helmholtz_2d(
        num_layers=sample_i[0],
        num_nodes=sample_i[1],
        activation_func=sample_i[2],
        epochs=sample_i[3],
        grid_size=sample_i[4],
        learning_rate=sample_i[5],
        test_name=f'helmholtz_2d_{i}',
        plotshow=False,
        device=device
    )

    # Save the result
    with open(output_file, 'wb') as f:
        pickle.dump(result, f)
    return i


if __name__ == '__main__':
    # Load parameter samples
    samples = pd.read_csv('parameter_samples.csv').values
    total_samples = len(samples)

    # Set the number of available GPUs
    # Adjust this based on your system configuration
    num_gpus = 4
    devices = [f'cuda:{i}' for i in range(num_gpus)]

    # Check for already completed simulations
    output_dir = Path('./output/pde8_helmholtz_2d')
    existing_files = []
    if output_dir.exists():
        existing_files = list(output_dir.glob('helmholtz_2d_*.pkl'))

    completed_indices = set()
    for file in existing_files:
        try:
            # Extract the index from the filename
            index = int(file.stem.split('_')[-1])
            completed_indices.add(index)
        except (ValueError, IndexError):
            # Ignore files with improperly formatted names
            continue

    print(f"Found {len(completed_indices)} existing results.")

    # Prepare the list of tasks, skipping completed ones
    tasks = []
    for i in range(total_samples):
        if i not in completed_indices:
            # Assign a GPU to the task using a round-robin strategy
            device = devices[i % num_gpus]
            tasks.append((i, samples[i], device))

    # Execute the remaining tasks
    if not tasks:
        print("All simulations have already been completed!")
    else:
        print(f"Starting {len(tasks)} remaining simulations across {num_gpus} GPU(s).")

        # Use a process pool to run simulations in parallel
        with ProcessPoolExecutor(max_workers=num_gpus) as executor:
            # Use tqdm for a progress bar
            results = list(tqdm(executor.map(run_simulation, tasks), total=len(tasks)))
        print("All simulations completed!")