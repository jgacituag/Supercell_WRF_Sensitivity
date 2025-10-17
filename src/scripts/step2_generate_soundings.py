"""
Step 2: Generate WRF input soundings for all Sobol samples

Usage:
    python src/scripts/step2_generate_soundings.py --config configs/experiment_config.yaml
"""

import numpy as np
import pickle
import argparse
import os
import yaml
import sys
import warnings
import multiprocessing
from functools import partial
from tqdm import tqdm

warnings.filterwarnings('ignore')

# Add src directory to path to find core modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
try:
    from core.sounding_generator import (generate_sounding, 
                                         write_input_sounding,
                                         calculate_diagnostics)
except ImportError:
    print("ERROR: Could not import 'sounding_generator'.")
    print("Make sure 'sounding_generator.py' exists in the 'src/core/' directory.")
    sys.exit(1)


def load_config(config_path):
    """Loads and returns the configuration dictionary from a YAML file."""
    if not os.path.exists(config_path):
        print(f"ERROR: Config file not found at {config_path}. Exiting.")
        return None 
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def load_experiment(input_dir):
    """Load experiment configuration and parameter samples."""
    with open(f'{input_dir}/problem.pkl', 'rb') as f:
        problem = pickle.load(f)
    param_values = np.load(f'{input_dir}/param_values.npy')
    with open(f'{input_dir}/metadata.pkl', 'rb') as f:
        metadata = pickle.load(f)
    return problem, param_values, metadata


def _init_diagnostics_dict(skip_diagnostics=False):
    """Initializes the nested dictionary for storing results."""
    if not skip_diagnostics:
        diag_dict = {
            'sample_id': [], 'mucape': [], 'mucin': [], 'sbcape': [],
            'sbcin': [], 'pwat': [], 'lifted_index': [],
            'shear_0_1km': [], 'shear_0_3km': [], 'shear_0_6km': [],
            'srh_0_3km': [], 'surface_theta': [], 'surface_qv': [],
            'surface_p': [], 'surface_t': [],
            'theta_corrections': [], 'qv_corrections': []
        }
    else:
        diag_dict = {
            'sample_id': [],
            'theta_corrections': [], 'qv_corrections': []
        }
    return diag_dict


def _stitch_results(results_list, skip_diagnostics=False):
    """
    Combines results from parallel workers into the final diagnostics dict
    and failed samples list.
    """
    diagnostics = _init_diagnostics_dict(skip_diagnostics)
    failed_samples = []
    
    # Sort results by sample_id to ensure order
    results_list.sort(key=lambda x: x['sample_id'])

    for res in results_list:
        if res['status'] == 'success':
            diagnostics['sample_id'].append(res['sample_id'])
            diagnostics['theta_corrections'].append(res['corrections']['theta_n_levels'])
            diagnostics['qv_corrections'].append(res['corrections']['qv_n_levels'])
            
            if not skip_diagnostics:
                for key, val in res['diagnostics'].items():
                    if key in diagnostics:
                        diagnostics[key].append(val)
        
        else:
            print(f"  ERROR: Sample {res['sample_id']} failed: {res['error']}")
            failed_samples.append(res['sample_id'])
            
            # Store NaN for failed cases
            diagnostics['sample_id'].append(res['sample_id'])
            for key in diagnostics.keys():
                if key not in ['sample_id']:
                    diagnostics[key].append(np.nan)
                    
    return diagnostics, failed_samples


def _process_sample_worker(sample_id, problem, param_values, base_sounding, output_dir, skip_diagnostics):
    """
    Core processing logic for a single sample.
    This function is called by both sequential and parallel loops.
    """
    try:
        param_dict = dict(zip(problem['names'], param_values[sample_id]))
        
        # 1. Generate sounding
        sounding = generate_sounding(param_dict, base_sounding_file=base_sounding)
        
        # 2. Calculate diagnostics
        diag_results = {}
        if not skip_diagnostics:
            diag_results = calculate_diagnostics(sounding)
        
        # 3. Write sounding file
        output_file = f'{output_dir}/input_sounding_{sample_id:05d}'
        write_input_sounding(output_file, sounding)
        
        # 4. Get correction info
        correction_results = sounding['corrections']
        
        return {
            'status': 'success',
            'sample_id': sample_id,
            'diagnostics': diag_results,
            'corrections': correction_results
        }
        
    except Exception as e:
        # import traceback # Uncomment for deep debugging
        # traceback.print_exc() # Uncomment for deep debugging
        return {
            'status': 'failed',
            'sample_id': sample_id,
            'error': str(e)
        }


def generate_soundings_parallel(param_values, problem, base_sounding, output_dir, 
                                n_processors, start_idx=None, end_idx=None, 
                                skip_diagnostics=False):
    """
    Generate soundings in parallel using multiprocessing.Pool.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    if start_idx is None: start_idx = 0
    if end_idx is None: end_idx = len(param_values)
    n_process = end_idx - start_idx
    
    if n_processors is None:
        n_processors = os.cpu_count()
        print(f"INFO: N_PROCESSORS not set, auto-detecting: {n_processors} cores")
    
    print(f"\nProcessing samples {start_idx} to {end_idx-1} in PARALLEL using {n_processors} workers...")

    # Use functools.partial to "freeze" arguments that are the same for all workers
    worker_func = partial(_process_sample_worker,
                          problem=problem,
                          param_values=param_values,
                          base_sounding=base_sounding,
                          output_dir=output_dir,
                          skip_diagnostics=skip_diagnostics)
    
    sample_indices = range(start_idx, end_idx)
    results_list = []

    with multiprocessing.Pool(processes=n_processors) as pool:
        # Use imap to get results lazily, and wrap with tqdm for a progress bar
        results_list = list(tqdm(pool.imap(worker_func, sample_indices), 
                                total=n_process, 
                                desc="Parallel Processing"))
    
    print("\nParallel processing complete. Collating results...")
    return _stitch_results(results_list, skip_diagnostics)


def generate_soundings(param_values, problem, base_sounding, output_dir, 
                       start_idx=None, end_idx=None, skip_diagnostics=False):
    """
    Generate soundings sequentially (single-core).
    """
    os.makedirs(output_dir, exist_ok=True)
    
    if start_idx is None: start_idx = 0
    if end_idx is None: end_idx = len(param_values)
    n_process = end_idx - start_idx
    
    print(f"\nProcessing samples {start_idx} to {end_idx-1} SEQUENTIALLY...")
    if skip_diagnostics:
        print("Skipping diagnostic calculations (faster)")
    else:
        print("Using MetPy for diagnostic calculations")

    results_list = []
    for i in tqdm(range(start_idx, end_idx), desc="Sequential Processing"):
        result = _process_sample_worker(i, problem, param_values, base_sounding, 
                                        output_dir, skip_diagnostics)
        results_list.append(result)
    
    return _stitch_results(results_list, skip_diagnostics)


def save_diagnostics(diagnostics, failed_samples, output_dir, skip_diagnostics=False):
    """Save diagnostic information."""
    
    # Save diagnostics
    diag_file = f'{output_dir}/diagnostics.pkl'
    with open(diag_file, 'wb') as f:
        pickle.dump(diagnostics, f)
    print(f"\nDiagnostics saved to: {diag_file}")
    
    # Save failed samples list
    if failed_samples:
        failed_file = f'{output_dir}/failed_samples.txt'
        with open(failed_file, 'w') as f:
            for idx in failed_samples:
                f.write(f"{idx}\n")
        print(f"Warning: {len(failed_samples)} samples failed!")
        print(f"Failed sample IDs saved to: {failed_file}")
    
    # Create summary
    summary_file = f'{output_dir}/summary.txt'
    with open(summary_file, 'w') as f:
        f.write("="*60 + "\n")
        f.write("SOUNDING GENERATION SUMMARY\n")
        f.write("="*60 + "\n\n")
        
        n_total = len(diagnostics['sample_id'])
        n_failed = len(failed_samples)
        n_valid = n_total - n_failed
        
        f.write(f"Total samples processed: {n_total}\n")
        f.write(f"Valid soundings: {n_valid} ({n_valid/n_total*100:.1f}%)\n")
        f.write(f"Failed samples: {n_failed} ({n_failed/n_total*100:.1f}%)\n\n")
        
        if not skip_diagnostics and n_valid > 0:
            f.write("Diagnostic Statistics (valid samples only):\n")
            f.write("-"*60 + "\n")
            
            for key in ['mucape', 'mucin', 'sbcape', 'sbcin',
                       'shear_0_1km', 'shear_0_3km', 'shear_0_6km', 
                       'pwat', 'surface_theta', 'surface_qv']:
                if key in diagnostics:
                    values = np.array(diagnostics[key])
                    valid_mask = ~np.isnan(values)
                    if valid_mask.sum() > 0:
                        values_valid = values[valid_mask]
                        
                        f.write(f"{key:20s}: ")
                        f.write(f"min={np.min(values_valid):8.2f}, ")
                        f.write(f"mean={np.mean(values_valid):8.2f}, ")
                        f.write(f"max={np.max(values_valid):8.2f}\n")
    
    print(f"Summary saved to: {summary_file}")


def main():
    parser = argparse.ArgumentParser(
        description='Generate WRF input soundings from Sobol samples'
    )
    # <-- MODIFIED: Changed default config path
    parser.add_argument('--config', type=str, default='configs/default_config.yaml',
                            help='Path to the configuration file.')
    parser.add_argument('--skip_diagnostics', action='store_true',
                       help='Override config and skip diagnostic calculations (faster)')
    parser.add_argument('--start_idx', type=int, default=None,
                       help='Starting sample index (optional, for manual batching)')
    parser.add_argument('--end_idx', type=int, default=None,
                       help='Ending sample index (optional, for manual batching)')
    
    args = parser.parse_args()
    
    if args.config == 'configs/default_config.yaml':
        print("INFO: Using default configuration file: configs/default_config.yaml")
        print("      To use your experiment settings, specify --config configs/experiment_config.yaml")
    else:
        print(f"INFO: Loading configuration from: {args.config}")
        
    config = load_config(args.config)
    if config is None:
        return

    print("="*60)
    print("STEP 2: GENERATING SOUNDINGS")
    print("="*60)

    # Load paths and settings from config
    output_root = config['PATHS']['OUTPUT_ROOT']
    exp_name = config['PATHS']['EXPERIMENT_NAME']
    input_dir = f"{output_root}/{exp_name}"
    base_sounding = config['PATHS']['BASE_SOUNDING_TEMPLATE']
    
    # Allow command-line to override config
    skip_diagnostics = config['SOUNDINGS'].get('SKIP_DIAGNOSTICS', False)
    if args.skip_diagnostics:
        skip_diagnostics = True
        print("INFO: --skip_diagnostics flag set, overriding config.")
    

    run_parallel = config['SOUNDINGS'].get('RUN_PARALLEL', False)
    n_processors = config['SOUNDINGS'].get('N_PROCESSORS', None)
    
    # Load experiment
    print(f"\nLoading experiment configuration from: {input_dir}")
    try:
        problem, param_values, metadata = load_experiment(input_dir)
    except FileNotFoundError:
        print(f"ERROR: Experiment files not found in {input_dir}")
        print("      Please run step1_generate_samples.py first.")
        return
    
    print(f"Total samples in experiment: {len(param_values)}")
    
    # Determine output directory
    output_dir = f'{input_dir}/soundings'
    

    if run_parallel:
        diagnostics, failed_samples = generate_soundings_parallel(
            param_values,
            problem,
            base_sounding,
            output_dir,
            n_processors, # <-- New argument
            start_idx=args.start_idx,
            end_idx=args.end_idx,
            skip_diagnostics=skip_diagnostics
        )
    else:
        diagnostics, failed_samples = generate_soundings(
            param_values,
            problem,
            base_sounding,
            output_dir,
            start_idx=args.start_idx,
            end_idx=args.end_idx,
            skip_diagnostics=skip_diagnostics
        )
    
    if 'qv_corrections' in diagnostics and len(diagnostics['qv_corrections']) > 0:
        corrections = np.array(diagnostics['qv_corrections'])
        corrections_valid = corrections[~np.isnan(corrections)]
        if len(corrections_valid) > 0:
            print(f"Soundings needing qv correction: {(corrections_valid > 0).sum()}/{len(corrections_valid)}")
            if (corrections_valid > 0).sum() > 0:
                print(f"Mean levels corrected: {corrections_valid[corrections_valid > 0].mean():.1f}")
        
    # Save results
    print("\nSaving diagnostics...")
    save_diagnostics(diagnostics, failed_samples, output_dir, 
                    skip_diagnostics=skip_diagnostics)
    
    print("\n" + "="*60)
    print("STEP 2 COMPLETE!")
    print("="*60)
    print("\nNext step(Optional): python src/scripts/step2b_filter_soundings.py --config configs/experiment_config.yaml")


if __name__ == "__main__":

    multiprocessing.freeze_support()
    main()