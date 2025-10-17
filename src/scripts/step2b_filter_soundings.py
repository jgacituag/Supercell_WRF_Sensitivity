"""
Step 2b: Filter soundings based on convective viability

Usage:
    python src/scripts/step2b_filter_soundings.py --config configs/experiment_config.yaml
"""

import numpy as np
import pickle
import argparse
import os
import yaml

def load_config(config_path):
    """Loads and returns the configuration dictionary from a YAML file."""
    if not os.path.exists(config_path):
        print(f"ERROR: Config file not found at {config_path}. Exiting.")
        return None 
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def load_diagnostics(input_dir):
    """Load diagnostic information."""
    
    diag_path = f'{input_dir}/soundings/diagnostics.pkl'
    if not os.path.exists(diag_path):
        print(f"ERROR: Diagnostics file not found: {diag_path}")
        print("Please run step2_generate_soundings.py first.")
        return None
        
    with open(diag_path, 'rb') as f:
        diagnostics = pickle.load(f)
    
    return diagnostics


def filter_soundings(diagnostics, criteria):
    """
    Filter soundings based on a flexible set of criteria.
    
    Parameters
    ----------
    diagnostics : dict
        Diagnostic information from step 2
    criteria : dict
        Dictionary of filtering criteria (e.g., MUCAPE_MIN, MUCAPE_MAX)
        
    Returns
    -------
    viable_mask : ndarray
        Boolean mask indicating viable soundings
    filter_stats : dict
        Statistics about filtering
    """
    
    # --- 1. Load all diagnostic arrays ---
    n_total = len(diagnostics['sample_id'])
    diag_arrays = {
        'mucape': np.array(diagnostics['mucape']),
        'mucin': np.array(diagnostics['mucin']),
        'shear_0_6km': np.array(diagnostics['shear_0_6km']),
        'pwat': np.array(diagnostics['pwat'])
    }
    
    # --- 2. Define the filters ---
    # (config_key, diagnostic_key, operator, failure_key)
    # ADD HERE OTHER FILTERING OPTIONS
    filter_definitions = [
        ('MUCAPE_MIN',    'mucape',      np.greater_equal, 'insufficient_cape'),
        ('MUCAPE_MAX',    'mucape',      np.less_equal,    'excessive_cape'),
        ('MUCIN_MIN',     'mucin',       np.greater_equal, 'excessive_cin'),
        ('MUCIN_MAX',     'mucin',       np.less_equal,    'insufficient_cin'),
        ('SHEAR_0_6KM_MIN', 'shear_0_6km', np.greater_equal, 'insufficient_shear'),
        ('SHEAR_0_6KM_MAX', 'shear_0_6km', np.less_equal,    'excessive_shear'),
        ('PWAT_MIN',      'pwat',        np.greater_equal, 'insufficient_pwat'),
        ('PWAT_MAX',      'pwat',        np.less_equal,    'excessive_pwat'),
    ]

    # --- 3. Initialize masks and stats ---
    total_mask = np.ones(n_total, dtype=bool)
    filter_stats = {
        'n_total': n_total,
        'criteria': {},
        'failures': {}
    }
    
    # --- 4. Handle non-finite values ---
    # Soundings with NaN for *any* of the variables are invalid
    finite_mask = np.ones(n_total, dtype=bool)
    for key, arr in diag_arrays.items():
        finite_mask = finite_mask & np.isfinite(arr)
        
    total_mask = total_mask & finite_mask
    filter_stats['failures']['non_finite'] = (~finite_mask).sum()

    # --- 5. Apply all defined filters ---
    print("\nApplying filtering criteria...")
    
    for config_key, diag_key, op_func, failure_key in filter_definitions:
        threshold = criteria.get(config_key, None)
        
        # Only apply if threshold is not None
        if threshold is not None:
            print(f"  - Applying {config_key}: {threshold}")
            diag_array = diag_arrays[diag_key]
            
            # Apply the operation (e.g., mucape >= 200)
            mask_i = op_func(diag_array, threshold)
            
            # Store statistics
            # We count failures only for samples that were otherwise finite
            filter_stats['failures'][failure_key] = (~mask_i & finite_mask).sum()
            filter_stats['criteria'][config_key] = threshold
            
            # Add to the total mask
            total_mask = total_mask & mask_i
            
    # --- 6. Final statistics ---
    filter_stats['n_viable'] = total_mask.sum()
    filter_stats['n_filtered'] = n_total - filter_stats['n_viable']
    filter_stats['viable_fraction'] = filter_stats['n_viable'] / n_total if n_total > 0 else 0
    
    return total_mask, filter_stats


def save_filter_results(viable_mask, filter_stats, diagnostics, output_dir):
    """Save filtering results."""
    
    all_sample_ids = np.array(diagnostics['sample_id'])
    viable_sample_ids = all_sample_ids[viable_mask]
    
    save_path = f'{output_dir}/viable_sample_ids.npy'
    np.save(save_path, viable_sample_ids)
    
    # --- Create human-readable summary ---
    with open(f'{output_dir}/filter_summary.txt', 'w') as f:
        f.write("="*60 + "\n")
        f.write("SOUNDING FILTERING SUMMARY\n")
        f.write("="*60 + "\n\n")
        
        f.write(f"Total soundings: {filter_stats['n_total']}\n")
        f.write(f"Viable soundings: {filter_stats['n_viable']} "
                f"({filter_stats['viable_fraction']*100:.1f}%)\n")
        f.write(f"Filtered out: {filter_stats['n_filtered']} "
                f"({(1-filter_stats['viable_fraction'])*100:.1f}%)\n\n")
        
        f.write("Filtering Criteria Applied:\n")
        f.write("-"*60 + "\n")
        if not filter_stats['criteria']:
            f.write("  No filters applied (all 'null' in config).\n")
        else:
            for key, value in filter_stats['criteria'].items():
                f.write(f"  - {key:15s}: {value}\n")
        
        f.write("\nReason for Filtering (non-exclusive):\n")
        f.write("-"*60 + "\n")
        for key, value in filter_stats['failures'].items():
            pct = value / filter_stats['n_total'] * 100 if filter_stats['n_total'] > 0 else 0
            f.write(f"  - {key:25s}: {value:4d} ({pct:5.1f}%)\n")
        
        f.write("\n" + "="*60 + "\n")
        f.write(f"Viable sample IDs saved to: {save_path}\n")
        f.write("="*60 + "\n")
    
    print(f"\nFiltering complete!")
    print(f"  Viable: {filter_stats['n_viable']}/{filter_stats['n_total']} "
          f"({filter_stats['viable_fraction']*100:.1f}%)")
    print(f"  Results saved to: {output_dir}/")
    print(f"  Viable IDs saved to: {save_path}")


def main():
    parser = argparse.ArgumentParser(
        description='Filter soundings based on convective viability'
    )
    parser.add_argument('--config', type=str, default='configs/default_config.yaml',
                            help='Path to the configuration file.')
    
    args = parser.parse_args()
    
    if args.config == 'configs/default_config.yaml':
        print("INFO: Using default configuration file: configs/default_config.yaml")
        print("      To filter your experiment, specify --config configsExample: configs/experiment_config.yaml")
    else:
        print(f"INFO: Loading configuration from: {args.config}")
        
    config = load_config(args.config)
    if config is None:
        return

    print("="*60)
    print("STEP 2b: FILTERING SOUNDINGS")
    print("="*60)

    # Load paths and settings from config
    output_root = config['PATHS']['OUTPUT_ROOT']
    exp_name = config['PATHS']['EXPERIMENT_NAME']
    input_dir = f"{output_root}/{exp_name}"
    
    # <-- MODIFIED: 'criteria' is now the entire FILTER dictionary
    criteria = config.get('FILTER', {}) 
    
    # Load diagnostics
    print(f"\nLoading diagnostics from: {input_dir}/soundings/")
    diagnostics = load_diagnostics(input_dir)
    if diagnostics is None:
        return
    
    # Filter soundings
    viable_mask, filter_stats = filter_soundings(diagnostics, criteria)
    
    # Save results
    output_dir = f'{input_dir}/soundings'
    save_filter_results(viable_mask, filter_stats, diagnostics, output_dir)
    
    print("\n" + "="*60)
    print("STEP 2b COMPLETE!")
    print("="*60)
    print(f"\nNext step: Run WRF only on viable soundings")
    print(f"  Use the sample IDs in: {output_dir}/viable_sample_ids.npy")


if __name__ == "__main__":
    main()