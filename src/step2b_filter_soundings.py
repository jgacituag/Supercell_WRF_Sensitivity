"""
Step 2b: Filter soundings based on convective viability

Usage:
    python step2b_filter_soundings.py --input_dir sobol_experiment
"""

import numpy as np
import pickle
import argparse
import os

def load_diagnostics(input_dir):
    """Load diagnostic information."""
    
    with open(f'{input_dir}/soundings/diagnostics.pkl', 'rb') as f:
        diagnostics = pickle.load(f)
    
    return diagnostics


def filter_soundings(diagnostics, criteria=None):
    """
    Filter soundings based on convective viability criteria.
    
    Parameters
    ----------
    diagnostics : dict
        Diagnostic information from step 2
    criteria : dict, optional
        Filtering criteria. Defaults:
        - MUCAPE > 200 J/kg
        - MUCIN > -200 J/kg (not too strong a cap)
        
    Returns
    -------
    viable_mask : ndarray
        Boolean mask indicating viable soundings
    filter_stats : dict
        Statistics about filtering
    """
    
    if criteria is None:
        criteria = {
            'mucape_min': 200,      # Minimum CAPE for organized convection
            'mucin_min': -200,      # CIN must be weaker (less negative) than -200 J/kg
        }
    
    n_total = len(diagnostics['sample_id'])
    
    # Convert to arrays
    mucape = np.array(diagnostics['mucape'])
    mucin = np.array(diagnostics['mucin'])

    
    # Apply filters
    mask_cape = mucape >= criteria['mucape_min']
    mask_cin = mucin >= criteria['mucin_min']  # SIMPLIFIED: just one CIN constraint
    mask_finite = np.isfinite(mucape) & np.isfinite(mucin)
    
    # Combine all criteria
    viable_mask = mask_cape & mask_cin & mask_finite
    
    # Calculate statistics
    filter_stats = {
        'n_total': n_total,
        'n_viable': viable_mask.sum(),
        'n_filtered': n_total - viable_mask.sum(),
        'viable_fraction': viable_mask.sum() / n_total,
        'criteria': criteria,
        'failures': {
            'insufficient_cape': (~mask_cape).sum(),
            'excessive_cin': (~mask_cin).sum(),
            'non_finite': (~mask_finite).sum()
        }
    }
    
    return viable_mask, filter_stats


def save_filter_results(viable_mask, filter_stats, diagnostics, output_dir):
    """Save filtering results."""
    
    # Create human-readable summary
    with open(f'{output_dir}/filter_summary.txt', 'w') as f:
        f.write("="*60 + "\n")
        f.write("SOUNDING FILTERING SUMMARY\n")
        f.write("="*60 + "\n\n")
        
        f.write(f"Total soundings: {filter_stats['n_total']}\n")
        f.write(f"Viable soundings: {filter_stats['n_viable']} "
                f"({filter_stats['viable_fraction']*100:.1f}%)\n")
        f.write(f"Filtered out: {filter_stats['n_filtered']} "
                f"({(1-filter_stats['viable_fraction'])*100:.1f}%)\n\n")
        
        f.write("Filtering Criteria:\n")
        f.write("-"*60 + "\n")
        f.write(f"  MUCAPE >= {filter_stats['criteria']['mucape_min']} J/kg\n")
        f.write(f"  MUCIN >= {filter_stats['criteria']['mucin_min']} J/kg "
                f"(i.e., |CIN| <= {-filter_stats['criteria']['mucin_min']} J/kg)\n")
        
        f.write("\nReason for Filtering:\n")
        f.write("-"*60 + "\n")
        for key, value in filter_stats['failures'].items():
            pct = value / filter_stats['n_total'] * 100
            f.write(f"  {key:25s}: {value:4d} ({pct:5.1f}%)\n")
        
        f.write("\n" + "="*60 + "\n")
        f.write("Viable sample IDs saved to: viable_sample_ids.npy\n")
        f.write("="*60 + "\n")
    
    print(f"\nFiltering complete!")
    print(f"  Viable: {filter_stats['n_viable']}/{filter_stats['n_total']} "
          f"({filter_stats['viable_fraction']*100:.1f}%)")
    print(f"  Results saved to: {output_dir}/")


def main():
    parser = argparse.ArgumentParser(
        description='Filter soundings based on convective viability'
    )
    parser.add_argument('--input_dir', type=str, required=True,
                       help='Input directory containing soundings')
    parser.add_argument('--mucape_min', type=float, default=200,
                       help='Minimum MUCAPE (J/kg)')
    parser.add_argument('--mucin_min', type=float, default=-200,
                       help='Minimum MUCIN - CIN must be weaker than this (J/kg)')
    
    args = parser.parse_args()
    
    print("="*60)
    print("STEP 2b: FILTERING SOUNDINGS")
    print("="*60)
    
    # Load diagnostics
    print("\nLoading diagnostics...")
    diagnostics = load_diagnostics(args.input_dir)
    
    # Define criteria
    criteria = {
        'mucape_min': args.mucape_min,
        'mucin_min': args.mucin_min,
    }
    
    # Filter soundings
    print("\nApplying filtering criteria...")
    viable_mask, filter_stats = filter_soundings(diagnostics, criteria)
    
    # Save results
    output_dir = f'{args.input_dir}/soundings'
    save_filter_results(viable_mask, filter_stats, diagnostics, output_dir)
    
    print("\n" + "="*60)
    print("STEP 2b COMPLETE!")
    print("="*60)
    print(f"\nNext step: Run WRF only on viable soundings")
    print(f"  Use the sample IDs in: {output_dir}/viable_sample_ids.npy")


if __name__ == "__main__":
    main()