"""
Step 1: Generate Sobol samples for sensitivity analysis
Creates parameter samples and saves them for later use.

Usage:
    python step1_generate_samples.py --n_samples 500 --output_dir sobol_exp
"""

import numpy as np
import pickle
import argparse
import os
from SALib.sample import saltelli
import warnings
warnings.filterwarnings('ignore')

def define_problem():
    """
    Define the problem for Sobol sensitivity analysis.
    
    Modify parameter bounds HERE.
    """
    
    problem = {
        'num_vars': 9,
        'names': [
            'low_level_rh',       # Boundary layer RH (%)
            'mid_level_rh',       # Mid-level RH (%)
            'upper_level_rh',     # Upper troposphere RH (%)
            'surface_theta',      # Surface potential temp (K)
            'low_level_lapse',    # BL lapse rate (K/km)
            'mid_level_lapse',    # Mid-level lapse rate (K/km)
            'shear_depth',        # Total Shear height (m)
            'shear_magnitude',    # 0 to shear_depth bulk shear (m/s)
            'shear_curvature',    # Hodograph curvature (0-1)
            #'shear_direction',    # Shear vector direction (deg)
            #'low_level_jet'       # LLJ strength (m/s)
        ],
        'bounds': [
            [70, 95],         # Low RH: moist BL
            [30, 70],         # Mid RH: dry to moist
            [10, 40],         # Upper RH: dry upper levels
            [295, 305],       # Surface theta Kelvin: cool to warm
            [7.5, 9.5],       # Low lapse: stable to unstable
            [6.0, 8.0],       # Mid lapse: typical tropospheric
            [4000, 8000],     # Total Shear Depth
            [1, 35],          # Shear: weak to strong
            [0, 1],           # Curvature: linear to semi-circular
            #[0, 360]         # Direction: all azimuths
            #[0, 15]           # LLJ: none to strong
        ]
    }
    
    return problem


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
        description='Generate Sobol samples for WRF sensitivity analysis'
    )
    parser.add_argument('--n_samples', type=int, default=25,
                       help='Number of base samples (default: 500)')
    parser.add_argument('--output_dir', type=str, default='outputs/sobol_exp_500',
                       help='Output directory (default: outputs/sobol_exp_500)')
    parser.add_argument('--seed', type=int, default=591946,
                       help='Random seed for reproducibility')
    parser.add_argument('--samples_per_batch', type=int, default=100,
                       help='Samples per batch for parallel processing (default: 100)')
    
    args = parser.parse_args()
    
    print("="*60)
    print("STEP 1: GENERATING SOBOL SAMPLES")
    print("="*60)
    
    # Define problem
    print("\nDefining problem...")
    problem = define_problem()
    
    # Generate samples
    print(f"\nGenerating samples with N={args.n_samples}...")
    print(f"This will create {args.n_samples * (2 * problem['num_vars'] + 2)} total samples")
    
    param_values = generate_samples(args.n_samples, problem, seed=args.seed)
    
    # Save everything
    print("\nSaving experiment setup...")
    metadata = {
        'seed': args.seed,
        'samples_per_batch': args.samples_per_batch
    }
    save_experiment_setup(problem, param_values, args.output_dir, metadata)
    
    # Create batch files
    create_batch_files(
        len(param_values), 
        args.output_dir, 
        args.samples_per_batch
    )
    
    print("\n" + "="*60)
    print("STEP 1 COMPLETE!")
    print("="*60)
    print(f"\nNext step: python src/step2_generate_soundings.py --input_dir {args.output_dir}")


if __name__ == "__main__":
    main()