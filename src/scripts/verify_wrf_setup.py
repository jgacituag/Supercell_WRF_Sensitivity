#!/usr/bin/env python
"""
WRF Setup Verification Script

Run this before starting your experiments to check:
- All required files exist
- Configuration is correct
- Paths are valid
- WRF executables work

Usage:
    python verify_wrf_setup.py [--config configs/experiment_config.yaml]
"""

import os
import sys
import yaml
from pathlib import Path
import subprocess


def check_file(filepath, description):
    """Check if file exists and print result."""
    if Path(filepath).exists():
        print(f"  ✓ {description}: {filepath}")
        return True
    else:
        print(f"  ✗ {description} NOT FOUND: {filepath}")
        return False


def check_executable(filepath, description):
    """Check if file exists and is executable."""
    path = Path(filepath)
    if path.exists() and os.access(path, os.X_OK):
        print(f"  ✓ {description}: {filepath}")
        return True
    elif path.exists():
        print(f"  ⚠ {description} exists but NOT EXECUTABLE: {filepath}")
        return False
    else:
        print(f"  ✗ {description} NOT FOUND: {filepath}")
        return False


def test_wrf_execution(wrf_path):
    """Try running WRF executables with --help to check they work."""
    print("\nTesting WRF executables...")
    
    ideal_exe = wrf_path / 'ideal.exe'
    if ideal_exe.exists():
        try:
            # Just check it can be executed (will fail but that's OK)
            result = subprocess.run(
                [str(ideal_exe)],
                capture_output=True,
                timeout=5
            )
            print(f"  ✓ ideal.exe can be executed")
        except Exception as e:
            print(f"  ⚠ ideal.exe execution test failed: {e}")
    
    wrf_exe = wrf_path / 'wrf.exe'
    if wrf_exe.exists():
        try:
            result = subprocess.run(
                [str(wrf_exe)],
                capture_output=True,
                timeout=5
            )
            print(f"  ✓ wrf.exe can be executed")
        except Exception as e:
            print(f"  ⚠ wrf.exe execution test failed: {e}")


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Verify WRF setup')
    parser.add_argument('--config', default='configs/experiment_config.yaml',
                      help='Path to experiment config file')
    args = parser.parse_args()
    
    # Get repository root (2 levels up from src/scripts/)
    repo_root = Path(__file__).resolve().parents[2]
    config_file = repo_root / args.config
    
    print("="*60)
    print("WRF Setup Verification")
    print("="*60)
    
    all_good = True
    
    # 1. Check config file
    print("\n1. Configuration File")
    if not check_file(config_file, "Config file"):
        print("\nERROR: Config file not found. Create it first!")
        sys.exit(1)
    
    # 2. Load and check config
    print("\n2. Loading Configuration...")
    try:
        with open(config_file, 'r') as f:
            config = yaml.safe_load(f)
        print("  ✓ Config file loaded successfully")
    except Exception as e:
        print(f"  ✗ Failed to load config: {e}")
        sys.exit(1)
    
    # 3. Check WRF section exists
    print("\n3. Checking WRF Configuration...")
    if 'WRF' not in config:
        print("  ✗ WRF section missing from config!")
        print("    Add the WRF section to your config file.")
        all_good = False
    else:
        print("  ✓ WRF section found")
        
        required_keys = ['MODEL_PATH', 'MODEL_DT', 'MODEL_DX', 'MODEL_NX', 
                        'MODEL_NY', 'MODEL_NZ']
        for key in required_keys:
            if key in config['WRF']:
                print(f"    ✓ {key}: {config['WRF'][key]}")
            else:
                print(f"    ✗ {key}: NOT SET")
                all_good = False
    
    if 'WRF' not in config or 'MODEL_PATH' not in config['WRF']:
        print("\nCannot continue without MODEL_PATH. Exiting.")
        sys.exit(1)
    
    # 4. Check WRF model path
    wrf_path = Path(config['WRF']['MODEL_PATH'])
    print(f"\n4. Checking WRF Model Path: {wrf_path}")
    
    if not wrf_path.exists():
        print(f"  ✗ WRF path does not exist!")
        all_good = False
    else:
        print(f"  ✓ WRF path exists")
        
        # Check executables
        if not check_executable(wrf_path / 'ideal.exe', "ideal.exe"):
            all_good = False
        if not check_executable(wrf_path / 'wrf.exe', "wrf.exe"):
            all_good = False
        
        # Check base namelist
        if not check_file(wrf_path / 'namelist.input', "Base namelist.input"):
            print("    WARNING: Base namelist will be needed for runs")
            all_good = False
        
        # Check some required data files
        required_files = ['LANDUSE.TBL', 'VEGPARM.TBL', 'RRTM_DATA']
        for fname in required_files:
            if not check_file(wrf_path / fname, fname):
                all_good = False
    
    # 5. Check experiment directory
    print("\n5. Checking Experiment Directory...")
    if 'PATHS' in config and 'EXPERIMENT_NAME' in config['PATHS']:
        exp_name = config['PATHS']['EXPERIMENT_NAME']
        exp_dir = repo_root / 'outputs' / exp_name
        
        if exp_dir.exists():
            print(f"  ✓ Experiment directory exists: {exp_dir}")
            
            # Check soundings
            soundings_dir = exp_dir / 'soundings'
            if soundings_dir.exists():
                print(f"  ✓ Soundings directory exists")
                
                # Count soundings
                sounding_files = list(soundings_dir.glob('input_sounding_*'))
                print(f"    → {len(sounding_files)} soundings generated")
                
                # Check viable samples
                viable_file = soundings_dir / 'viable_sample_ids.npy'
                if viable_file.exists():
                    import numpy as np
                    viable = np.load(viable_file)
                    print(f"    → {len(viable)} viable samples (after filtering)")
                else:
                    print(f"    ⚠ No viable_sample_ids.npy - run step2b first")
            else:
                print(f"  ⚠ Soundings not generated yet")
                print(f"    Run: python src/scripts/step1_generate_samples.py")
                print(f"         python src/scripts/step2_generate_soundings.py")
        else:
            print(f"  ⚠ Experiment directory doesn't exist yet: {exp_dir}")
            print(f"    Will be created when running experiments")
    
    # 6. Check Python environment
    print("\n6. Checking Python Environment...")
    try:
        import netCDF4
        print("  ✓ netCDF4 module")
    except ImportError:
        print("  ✗ netCDF4 module not found")
        all_good = False
    
    try:
        import wrf
        print("  ✓ wrf-python module")
    except ImportError:
        print("  ✗ wrf-python module not found")
        all_good = False
    
    try:
        import numpy
        print("  ✓ numpy module")
    except ImportError:
        print("  ✗ numpy module not found")
        all_good = False
    
    # 7. Test WRF execution (if requested)
    if all_good and wrf_path.exists():
        test_wrf_execution(wrf_path)
    
    # 8. Summary
    print("\n" + "="*60)
    if all_good:
        print("✓ ALL CHECKS PASSED!")
        print("="*60)
        print("\nYou're ready to run WRF experiments!")
        print("\nNext steps:")
        print("  1. Test one sample:")
        print("     python src/scripts/step3_run_wrf.py --start 0 --end 1")
        print("  2. Generate PBS jobs:")
        print("     python src/scripts/submit_wrf_jobs.py --batch-size 50")
        print("  3. Submit jobs:")
        print("     python src/scripts/submit_wrf_jobs.py --batch-size 50 --submit")
    else:
        print("⚠ SOME CHECKS FAILED")
        print("="*60)
        print("\nPlease fix the issues above before running experiments.")
        print("See the setup guide for details.")
    print()


if __name__ == '__main__':
    main()
