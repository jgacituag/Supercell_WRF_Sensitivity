"""
Step 1: Generate Sobol samples for sensitivity analysis
Creates parameter samples and saves them for later use.

Usage:
    python src/scripts/step1_generate_samples.py --config configs/experiment_config.yaml
"""

import numpy as np
import pickle
import argparse
import os
import yaml
from SALib.sample import saltelli
import warnings
warnings.filterwarnings('ignore')

def load_config(config_path):
    """Loads and returns the configuration dictionary from a YAML file."""
    if not os.path.exists(config_path):
        print(f"ERROR: Config file not found at {config_path}. Exiting.")
        return None 
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def generate_samples(n_samples, problem, seed=None):
    """
    Generate Saltelli samples for Sobol analysis.
    
    Parameters
    ----------
    n_samples : int
        Number of base samples. Total samples will be N(2D+2)
        where D is the number of parameters.
    problem : dict
        SALib problem definition
    seed : int, optional
        Random seed for reproducibility
        
    Returns
    -------
    param_values : ndarray
        Array of parameter combinations (N(2D+2) x D)
    """
    
    if seed is not None:
        np.random.seed(seed)
    
    # Generate Saltelli samples
    param_values = saltelli.sample(problem, n_samples, calc_second_order=True)
    
    return param_values


def save_experiment_setup(problem, param_values, output_dir, metadata=None):
    """
    Save experiment configuration and samples.
    
    Parameters
    ----------
    problem : dict
        SALib problem definition
    param_values : ndarray
        Generated parameter samples
    output_dir : str
        Output directory
    metadata : dict, optional
        Additional metadata to save
    """
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Save problem definition
    with open(f'{output_dir}/problem.pkl', 'wb') as f:
        pickle.dump(problem, f)
    
    # Save parameter values
    np.save(f'{output_dir}/param_values.npy', param_values)
    
    # Save metadata
    if metadata is None:
        metadata = {}
    
    metadata.update({
        'n_samples_base': len(param_values) // (2 * problem['num_vars'] + 2),
        'n_samples_total': len(param_values),
        'num_vars': problem['num_vars'],
        'var_names': problem['names']
    })
    
    with open(f'{output_dir}/metadata.pkl', 'wb') as f:
        pickle.dump(metadata, f)
    
    # Create human-readable summary
    with open(f'{output_dir}/experiment_summary.txt', 'w') as f:
        f.write("="*60 + "\n")
        f.write("SOBOL SENSITIVITY ANALYSIS - EXPERIMENT SETUP\n")
        f.write("="*60 + "\n\n")
        f.write(f"Base samples (N): {metadata['n_samples_base']}\n")
        f.write(f"Total samples: {metadata['n_samples_total']}\n")
        f.write(f"Number of parameters: {metadata['num_vars']}\n")
        f.write(f"Formula: N(2D+2) = {metadata['n_samples_base']} × "
                f"(2×{metadata['num_vars']}+2) = {metadata['n_samples_total']}\n\n")
        
        f.write("Parameters and Bounds:\n")
        f.write("-"*60 + "\n")
        for i, (name, bounds) in enumerate(zip(problem['names'], problem['bounds'])):
            f.write(f"{i+1:2d}. {name:20s}: [{bounds[0]:8.2f}, {bounds[1]:8.2f}]\n")
        
        f.write("\n" + "="*60 + "\n")
        f.write("Files created:\n")
        f.write("  - problem.pkl: SALib problem definition\n")
        f.write("  - param_values.npy: All parameter combinations\n")
        f.write("  - metadata.pkl: Experiment metadata\n")
        f.write("="*60 + "\n")
    
    print(f"\nExperiment setup saved to: {output_dir}/")
    print(f"  Total samples to run: {metadata['n_samples_total']}")


def create_batch_files(n_samples_total, output_dir, samples_per_batch=100):
    """
    Create batch index files for parallel processing.
    
    This creates small text files indicating which samples each
    job should process, useful for array jobs on clusters.
    
    Parameters
    ----------
    n_samples_total : int
        Total number of samples
    output_dir : str
        Output directory
    samples_per_batch : int
        Number of samples per batch/job
    """
    
    batch_dir = f'{output_dir}/batch_indices'
    os.makedirs(batch_dir, exist_ok=True)
    
    n_batches = int(np.ceil(n_samples_total / samples_per_batch))
    
    for batch_id in range(n_batches):
        start_idx = batch_id * samples_per_batch
        end_idx = min((batch_id + 1) * samples_per_batch, n_samples_total)
        
        with open(f'{batch_dir}/batch_{batch_id:04d}.txt', 'w') as f:
            f.write(f"{start_idx}\n{end_idx}\n")
    
    print(f"\nCreated {n_batches} batch files in {batch_dir}/")
    print(f"  Samples per batch: {samples_per_batch}")
    print(f"  Use these for array jobs on your cluster")


def main():
    parser = argparse.ArgumentParser(
        description='Generate Sobol samples for WRF sensitivity analysis')
    parser.add_argument('--config', type=str, default='configs/default_config.yaml',
                            help='Path to the configuration file.')
    args = parser.parse_args()
    
    if args.config == 'configs/default_config.yaml':
        print("INFO: Using default configuration file: configs/default_config.yaml")
        print("      To create a new experiment, please copy this file, edit it,")
        print("      and specify it using: --config configs/your_new_config.yaml")
    else:
        print(f"INFO: Loading configuration from: {args.config}")
    config = load_config(args.config)
    if config is None:
        return

    print("="*60)
    print("STEP 1: GENERATING SOBOL SAMPLES")
    print("="*60)

    n_samples = config['SOBOL']['N_SAMPLES_BASE']
    seed= config['SOBOL']['RANDOM_SEED']
    samples_per_batch = config['SOBOL']['SAMPLES_PER_BATCH']
    output_root = config['PATHS']['OUTPUT_ROOT']
    exp_name = config['PATHS']['EXPERIMENT_NAME']
    output_dir=f"{output_root}/{exp_name}"

    # Define problem
    print("\nDefining problem...")
    param_dict = config['SOBOL']['PROBLEM_DEFINITION']
    problem = {
        'num_vars': len(param_dict),
        'names': list(param_dict.keys()),
        'bounds': list(param_dict.values())
    }
    
    # Generate samples
    print(f"\nGenerating samples with N={n_samples} and D={len(param_dict)}...")
    total_samples_calc = n_samples * (2 * problem['num_vars'] + 2)
    print(f"This will create {total_samples_calc} total samples")
    
    param_values = generate_samples(n_samples, problem, seed=seed)
    
    # Save everything
    print("\nSaving experiment setup...")
    metadata = {
        'seed': seed,
        'samples_per_batch': samples_per_batch,
        'config_file_path': args.config
    }
    save_experiment_setup(problem, param_values, output_dir, metadata)
    
    # Create batch files
    create_batch_files(
        len(param_values), 
        output_dir, 
        samples_per_batch
    )
    
    print("\n" + "="*60)
    print("STEP 1 COMPLETE!")
    print("="*60)
    print("\nNext step: python src/scripts/step2_generate_soundings.py --config configs/experiment_config.yaml")


if __name__ == "__main__":
    main()