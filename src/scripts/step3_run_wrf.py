"""
Step 3: Run WRF simulations

For single simulation:
    python step3_run_wrf.py --input_dir sobol_experiment --sample_id 0

For batch processing:
    python step3_run_wrf.py --input_dir sobol_experiment --batch_id 0

For cluster array jobs, use the provided SLURM script template.
"""

import numpy as np
import pickle
import argparse
import os
import sys
import shutil
import wrf_module as wrfm


def load_config_template(template_file='wrf_config_template.pkl'):
    """Load WRF configuration template."""
    
    if os.path.exists(template_file):
        with open(template_file, 'rb') as f:
            config = pickle.load(f)
    else:
        # Default configuration
        config = {
            'modelpath': '/path/to/wrf/model',
            'run_model': True,
            'plot_exp': False,
            'nthreads': 4,
            'model_dt': 6,
            'model_dt_fract_num': 0,
            'model_dt_fract_den': 1,
            'model_dx': 500,
            'model_dy': 500,
            'model_nx': 160,
            'model_ny': 160,
            'model_nz': 100,
            # Sensitivity function settings
            'sf_type': 'upward_vertical_velocity_percentile',
            'sf_percentile': 99,
            'sf_inilev': 0,
            'sf_endlev': -1,
            'sf_initime': 0,
            'sf_endtime': -1
        }
        
        # Save template for future use
        with open(template_file, 'wb') as f:
            pickle.dump(config, f)
        
        print(f"Created default config template: {template_file}")
        print("Edit this file to customize WRF settings")
    
    return config


def setup_wrf_run(sample_id, input_dir, base_config):
    """
    Setup configuration for a single WRF run.
    
    Parameters
    ----------
    sample_id : int
        Sample index to run
    input_dir : str
        Experiment input directory
    base_config : dict
        Base WRF configuration
        
    Returns
    -------
    config : dict
        Complete configuration for this run
    sounding_file : str
        Path to input sounding
    """
    
    # Create run-specific config
    config = base_config.copy()
    config['expname'] = 'sobol_wrf'
    config['run_num'] = sample_id
    config['datapath'] = f'{input_dir}/wrf_output'
    
    # Locate sounding file
    sounding_file = f'{input_dir}/soundings/input_sounding_{sample_id:05d}'
    
    if not os.path.exists(sounding_file):
        raise FileNotFoundError(f"Sounding not found: {sounding_file}")
    
    return config, sounding_file


def run_single_simulation(sample_id, input_dir, base_config):
    """Run WRF for a single sample."""
    
    print(f"\n{'='*60}")
    print(f"Running WRF simulation for sample {sample_id}")
    print(f"{'='*60}\n")
    
    try:
        # Setup configuration
        config, sounding_file = setup_wrf_run(sample_id, input_dir, base_config)
        
        # Copy sounding to WRF working directory
        wrf_workdir = config['modelpath']
        shutil.copy(sounding_file, f'{wrf_workdir}/input_sounding')
        
        # Run WRF
        wrfm.run_wrf(config)
        
        print(f"\n{'='*60}")
        print(f"Sample {sample_id} completed successfully")
        print(f"{'='*60}\n")
        
        return True
        
    except Exception as e:
        print(f"\n{'='*60}")
        print(f"ERROR in sample {sample_id}: {e}")
        print(f"{'='*60}\n")
        
        # Log error
        error_dir = f'{input_dir}/wrf_output/errors'
        os.makedirs(error_dir, exist_ok=True)
        
        with open(f'{error_dir}/sample_{sample_id:05d}.txt', 'w') as f:
            f.write(f"Sample ID: {sample_id}\n")
            f.write(f"Error: {str(e)}\n")
        
        return False


def run_batch(batch_id, input_dir, base_config):
    """Run WRF for all samples in a batch."""
    
    # Get batch indices
    batch_file = f'{input_dir}/batch_indices/batch_{batch_id:04d}.txt'
    with open(batch_file, 'r') as f:
        lines = f.readlines()
        start_idx = int(lines[0].strip())
        end_idx = int(lines[1].strip())
    
    print(f"\n{'='*60}")
    print(f"Processing batch {batch_id}")
    print(f"Samples {start_idx} to {end_idx-1}")
    print(f"{'='*60}\n")
    
    success_count = 0
    fail_count = 0
    
    for sample_id in range(start_idx, end_idx):
        success = run_single_simulation(sample_id, input_dir, base_config)
        
        if success:
            success_count += 1
        else:
            fail_count += 1
    
    # Save batch summary
    summary_dir = f'{input_dir}/wrf_output/batch_summaries'
    os.makedirs(summary_dir, exist_ok=True)
    
    with open(f'{summary_dir}/batch_{batch_id:04d}.txt', 'w') as f:
        f.write(f"Batch ID: {batch_id}\n")
        f.write(f"Sample range: {start_idx} to {end_idx-1}\n")
        f.write(f"Total samples: {end_idx - start_idx}\n")
        f.write(f"Successful: {success_count}\n")
        f.write(f"Failed: {fail_count}\n")
    
    print(f"\n{'='*60}")
    print(f"Batch {batch_id} complete!")
    print(f"Successful: {success_count}, Failed: {fail_count}")
    print(f"{'='*60}\n")


def main():
    parser = argparse.ArgumentParser(
        description='Run WRF simulations for Sobol samples'
    )
    parser.add_argument('--input_dir', type=str, required=True,
                       help='Experiment input directory')
    parser.add_argument('--sample_id', type=int, default=None,
                       help='Single sample ID to run')
    parser.add_argument('--batch_id', type=int, default=None,
                       help='Batch ID for batch processing')
    parser.add_argument('--config_file', type=str, 
                       default='wrf_config_template.pkl',
                       help='WRF configuration template file')
    
    args = parser.parse_args()
    
    if args.sample_id is None and args.batch_id is None:
        parser.error("Must specify either --sample_id or --batch_id")
    
    if args.sample_id is not None and args.batch_id is not None:
        parser.error("Cannot specify both --sample_id and --batch_id")
    
    # Load configuration
    print("Loading WRF configuration...")
    base_config = load_config_template(args.config_file)
    
    # Run simulation(s)
    if args.sample_id is not None:
        run_single_simulation(args.sample_id, args.input_dir, base_config)
    else:
        run_batch(args.batch_id, args.input_dir, base_config)


if __name__ == "__main__":
    main()