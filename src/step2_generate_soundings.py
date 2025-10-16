"""
Step 2: Generate WRF input soundings for all Sobol samples

Usage:
    python step2_generate_soundings.py --input_dir sobol_experiment
"""

import numpy as np
import pickle
import argparse
import os
import sys
from sounding_generator import (generate_sounding, 
                                write_input_sounding,
                                calculate_cape_cin,
                                calculate_diagnostics)
import warnings
warnings.filterwarnings('ignore')

def load_experiment(input_dir):
    """Load experiment configuration and parameter samples."""
    
    with open(f'{input_dir}/problem.pkl', 'rb') as f:
        problem = pickle.load(f)
    
    param_values = np.load(f'{input_dir}/param_values.npy')
    
    with open(f'{input_dir}/metadata.pkl', 'rb') as f:
        metadata = pickle.load(f)
    
    return problem, param_values, metadata


def generate_soundings(param_values, problem, base_sounding, output_dir, 
                      start_idx=None, end_idx=None, skip_diagnostics=False):
    """
    Generate soundings for a range of samples.
    
    Parameters
    ----------
    param_values : ndarray
        All parameter combinations
    problem : dict
        SALib problem definition
    base_sounding : str
        Path to base input_sounding file
    output_dir : str
        Directory to store soundings
    start_idx : int, optional
        Starting index (if None, start from 0)
    end_idx : int, optional
        Ending index (if None, process all)
    skip_diagnostics : bool
        If True, skip diagnostic calculations (faster)
        
    Returns
    -------
    diagnostics : dict
        Diagnostic information for generated soundings
    failed_samples : list
        List of sample indices that failed
    """
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Determine range
    if start_idx is None:
        start_idx = 0
    if end_idx is None:
        end_idx = len(param_values)
    
    n_process = end_idx - start_idx
    
    # Initialize diagnostics dictionary
    if not skip_diagnostics:
        diagnostics = {
            'sample_id': [],
            'mucape': [],
            'mucin': [],
            'sbcape': [],
            'sbcin': [],
            'pwat': [],
            'lifted_index': [],
            'shear_0_1km': [],
            'shear_0_3km': [],
            'shear_0_6km': [],
            'srh_0_3km': [],
            'surface_theta': [],
            'surface_qv': [],
            'surface_p': [],
            'surface_t': [],
            'theta_corrections': [],
            'qv_corrections': []
    }
    else:
        diagnostics = {'sample_id': []}
    
    failed_samples = []
    
    print(f"\nProcessing samples {start_idx} to {end_idx-1} ({n_process} total)")
    if skip_diagnostics:
        print("Skipping diagnostic calculations (faster)")
    else:
        print("Using MetPy for diagnostic calculations")
    
    for i in range(start_idx, end_idx):
        param_dict = dict(zip(problem['names'], param_values[i]))
        
        try:
            # Generate sounding
            sounding = generate_sounding(param_dict, 
                                        base_sounding_file=base_sounding)
            
            # Store sample ID
            diagnostics['sample_id'].append(i)
            
            # Calculate diagnostics if requested
            if not skip_diagnostics:
                diag = calculate_diagnostics(sounding)
                for key in diag.keys():
                    if key in diagnostics:
                        diagnostics[key].append(diag[key])
            
            # Write sounding file
            output_file = f'{output_dir}/input_sounding_{i:05d}'
            write_input_sounding(output_file, sounding)
            
            # Progress update
            if (i - start_idx + 1) % 10 == 0:
                progress = (i - start_idx + 1) / n_process * 100
                print(f"  Progress: {i-start_idx+1}/{n_process} ({progress:.1f}%)")

            diagnostics['theta_corrections'].append(
                sounding['corrections']['theta_n_levels']
            )
            diagnostics['qv_corrections'].append(
                sounding['corrections']['qv_n_levels']
            )
        except Exception as e:
            print(f"  ERROR: Sample {i} failed: {e}")
            import traceback
            traceback.print_exc()
            failed_samples.append(i)
            
            # Store NaN for failed cases
            diagnostics['sample_id'].append(i)
            if not skip_diagnostics:
                for key in diagnostics.keys():
                    if key != 'sample_id':
                        diagnostics[key].append(np.nan)
    
    return diagnostics, failed_samples


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
            
            # Use mucape/mucin instead of cape/cin
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
    parser.add_argument('--input_dir', type=str, required=True,
                       help='Input directory containing Sobol samples')
    parser.add_argument('--base_sounding', type=str, default='src/input_sounding',
                       help='Base input_sounding file (default: input_sounding)')
    parser.add_argument('--skip_diagnostics', action='store_true',
                       help='Skip diagnostic calculations (faster)')
    parser.add_argument('--start_idx', type=int, default=None,
                       help='Starting sample index (optional, for manual batching)')
    parser.add_argument('--end_idx', type=int, default=None,
                       help='Ending sample index (optional, for manual batching)')
    
    args = parser.parse_args()
    
    print("="*60)
    print("STEP 2: GENERATING SOUNDINGS")
    print("="*60)
    
    # Load experiment
    print("\nLoading experiment configuration...")
    problem, param_values, metadata = load_experiment(args.input_dir)
    
    print(f"Total samples in experiment: {len(param_values)}")
    
    # Determine output directory
    output_dir = f'{args.input_dir}/soundings'
    
    # Generate soundings
    print("\nGenerating soundings...")
    diagnostics, failed_samples = generate_soundings(
        param_values,
        problem,
        args.base_sounding,
        output_dir,
        start_idx=args.start_idx,
        end_idx=args.end_idx,
        skip_diagnostics=args.skip_diagnostics
    )
    corrections = np.array(diagnostics['qv_corrections'])
    print(f"Soundings needing qv correction: {(corrections > 0).sum()}/{len(corrections)}")
    print(f"Mean levels corrected: {corrections[corrections > 0].mean():.1f}")
    # Save results
    print("\nSaving diagnostics...")
    save_diagnostics(diagnostics, failed_samples, output_dir, 
                    skip_diagnostics=args.skip_diagnostics)
    print("\n" + "="*60)
    print("STEP 2 COMPLETE!")
    print("="*60)
    print(f"\nNext step: Run WRF simulations")


if __name__ == "__main__":
    main()