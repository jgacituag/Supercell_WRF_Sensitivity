#!/usr/bin/env python
"""
Step 3: Run WRF simulations for Sobol samples

This script executes WRF ideal.exe simulations for each viable sounding
generated in Steps 1-2, extracts key metrics, and saves results for
sensitivity analysis.

Usage:
    python step3_run_wrf.py [--start INDEX] [--end INDEX] [--batch-size N]
"""

import os
import sys
import yaml
import argparse
import pickle
import numpy as np
from pathlib import Path
from datetime import datetime
import multiprocessing as mp
import logging

# Add src to path
REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(REPO_ROOT / 'src'))
sys.path.append(str(REPO_ROOT / 'src/core'))

from core.wrf_module import WRFRun

def _install_global_exception_logger(logger):
    def _excepthook(exc_type, exc, tb):
        logger.critical("Uncaught exception", exc_info=(exc_type, exc, tb))
    sys.excepthook = _excepthook

def _log_pbs_context(logger):
    jid = os.getenv("PBS_JOBID", "N/A")
    nodefile = os.getenv("PBS_NODEFILE")
    nodes = None
    if nodefile and Path(nodefile).exists():
        try:
            nodes = Path(nodefile).read_text().strip().splitlines()
        except Exception:
            nodes = []
    logger.info("PBS context: JOBID=%s NODES=%s", jid, len(nodes) if nodes else 0)
    if nodes:
        logger.debug("PBS nodes:\n%s", "\n".join(nodes[:16] + (["..."] if len(nodes) > 16 else [])))

def load_experiment_config(config_file):
    """Load experiment configuration from YAML."""
    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)
    return config


def get_viable_samples(experiment_dir):
    """
    Load list of viable sample IDs.
    
    Returns
    -------
    viable_ids : np.ndarray
        Array of sample IDs that passed filtering
    """
    viable_file = experiment_dir / 'soundings' / 'viable_sample_ids.npy'
    
    if not viable_file.exists():
        raise FileNotFoundError(
            f"Viable samples file not found: {viable_file}\n"
            "Please run step2b_filter_soundings.py first."
        )
    
    viable_ids = np.load(viable_file)
    print(f"Loaded {len(viable_ids)} viable samples from {viable_file}")
    
    return viable_ids


def run_wrf_sample(args):
    """
    Run WRF for a single sample (for parallel execution).
    
    Parameters
    ----------
    args : tuple
        (sample_id, sounding_file, config_dict, nthreads)
    
    Returns
    -------
    metrics : dict or None
        Extracted metrics if successful
    """
    sample_id, sounding_file, config_dict, nthreads = args
    
    # Convert config_dict to format expected by WRFRun
    wrf_config = {
        'wrf': {
            'model_path': config_dict['WRF']['MODEL_PATH'],
            'model_dt': config_dict['WRF']['MODEL_DT'],
            'model_dx': config_dict['WRF']['MODEL_DX'],
            'model_dy': config_dict['WRF'].get('MODEL_DY', config_dict['WRF']['MODEL_DX']),
            'model_nx': config_dict['WRF']['MODEL_NX'],
            'model_ny': config_dict['WRF']['MODEL_NY'],
            'model_nz': config_dict['WRF']['MODEL_NZ'],
            'temp_dir': config_dict['WRF'].get('TEMP_DIR', '/tmp/wrf_runs'),
        },
        'paths': {
            'wrf_output_dir': config_dict['PATHS']['WRF_OUTPUT_DIR']
        }
    }
    
    # Initialize WRF runner
    wrf_runner = WRFRun(config=wrf_config)

    # Run simulation
    try:
        metrics = wrf_runner.run_single_experiment(
            sample_id=sample_id,
            sounding_file=sounding_file,
            nthreads=nthreads,
            cleanup_after=config_dict['WRF'].get('CLEANUP_AFTER_RUN', True),
            keep_wrfout=config_dict['WRF'].get('KEEP_WRFOUT', False)
        )
        return metrics
    
    except Exception as e:
        print(f"ERROR: Sample {sample_id:05d} failed with exception: {e}")
        return None


def main():
    logs_dir = REPO_ROOT / 'logs'
    logs_dir.mkdir(parents=True, exist_ok=True)
    logfile = logs_dir / 'step3_debug.log'

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(logfile, mode='a')  # append
        ]
    )
    logger = logging.getLogger("step3")
    _install_global_exception_logger(logger)
    _log_pbs_context(logger)
    logger.info("Starting Step 3: Run WRF simulations")

    parser = argparse.ArgumentParser(
        description='Run WRF simulations for Sobol sensitivity experiments'
    )
    parser.add_argument(
        '--config',
        type=str,
        default='configs/experiment_config.yaml',
        help='Path to experiment configuration file'
    )
    parser.add_argument(
        '--start',
        type=int,
        default=0,
        help='Start index for batch processing'
    )
    parser.add_argument(
        '--end',
        type=int,
        default=None,
        help='End index for batch processing (default: all samples)'
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=None,
        help='Number of samples to process in this batch'
    )

    parser.add_argument(
        '--parallel',
        type=int,
        default=1,
        help='Number of WRF runs to execute in parallel'
    )
    parser.add_argument(
        '--resume',
        action='store_true',
        help='Skip samples that already have metrics files'
    )
    parser.add_argument(
        '--nthreads',
        type=int,
        default=None,
        help='Number of MPI ranks for wrf.exe'
    )
    args = parser.parse_args()
    
    # Load configuration
    config_file = REPO_ROOT / args.config
    if not config_file.exists():
        print(f"ERROR: Config file not found: {config_file}")
        sys.exit(1)
    
    config = load_experiment_config(config_file)
    
    # Setup paths
    experiment_name = config['PATHS']['EXPERIMENT_NAME']
    experiment_dir = REPO_ROOT / 'outputs' / experiment_name
    soundings_dir = experiment_dir / 'soundings'
    wrf_output_dir = REPO_ROOT / config['PATHS']['WRF_OUTPUT_DIR']
    
    # Create output directory
    wrf_output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load viable samples
    viable_ids = get_viable_samples(experiment_dir)
    
    # Determine batch range
    start_idx = args.start
    end_idx = args.end if args.end is not None else len(viable_ids)
    
    if args.batch_size:
        end_idx = min(start_idx + args.batch_size, len(viable_ids))
    
    batch_ids = viable_ids[start_idx:end_idx]
    
    print(f"\n{'='*60}")
    print(f"WRF Execution - Step 3")
    print(f"{'='*60}")
    print(f"Experiment:     {experiment_name}")
    print(f"Total viable:   {len(viable_ids)}")
    print(f"Batch range:    {start_idx} to {end_idx}")
    print(f"Batch size:     {len(batch_ids)}")
    print(f"Parallel runs:  {args.parallel}")
    print(f"{'='*60}\n")
    
    # Get number of threads
    nthreads = args.nthreads or config['WRF'].get('MPI_PROCS', 1)
    
    # Prepare arguments for each sample
    run_args = []
    for sample_id in batch_ids:
        logger.info("Sample %d: starting", sample_id)
        sounding_file = soundings_dir / f'input_sounding_{sample_id:05d}'
        
        # Skip if resuming and metrics already exist
        if args.resume:
            metrics_file = wrf_output_dir / f'sample_{sample_id:05d}' / 'metrics.pkl'
            if metrics_file.exists():
                print(f"Skipping sample {sample_id:05d} (metrics already exist)")
                continue
        
        if not sounding_file.exists():
            print(f"WARNING: Sounding file not found for sample {sample_id:05d}")
            continue
        
        run_args.append((sample_id, str(sounding_file), config, nthreads))
        logger.info("Sample %d: done", sample_id)

    if not run_args:
        print("No samples to process!")
        return
    
    print(f"Processing {len(run_args)} samples...")
    
    # Run simulations
    if args.parallel > 1:
        # Parallel execution
        print(f"Running {args.parallel} simulations in parallel\n")
        with mp.Pool(args.parallel) as pool:
            results = pool.map(run_wrf_sample, run_args)
    else:
        # Serial execution
        print("Running simulations serially\n")
        results = [run_wrf_sample(arg) for arg in run_args]
    
    # Summary
    print(f"\n{'='*60}")
    print("Execution Summary")
    print(f"{'='*60}")
    
    successful = [r for r in results if r is not None and r.get('success', False)]
    failed = [r for r in results if r is None or not r.get('success', False)]
    
    print(f"Total processed: {len(results)}")
    print(f"Successful:      {len(successful)}")
    print(f"Failed:          {len(failed)}")
    
    if successful:
        # Print some statistics
        max_updrafts = [r['max_updraft'] for r in successful if 'max_updraft' in r]
        if max_updrafts:
            print(f"\nUpdraft statistics (m/s):")
            print(f"  Min:    {np.min(max_updrafts):.1f}")
            print(f"  Mean:   {np.mean(max_updrafts):.1f}")
            print(f"  Max:    {np.max(max_updrafts):.1f}")
    
    # Save batch summary
    batch_summary = {
        'batch_range': (start_idx, end_idx),
        'processed': len(results),
        'successful': len(successful),
        'failed': len(failed),
        'timestamp': datetime.now().isoformat()
    }
    
    summary_file = wrf_output_dir / f'batch_{start_idx:05d}_{end_idx:05d}_summary.pkl'
    with open(summary_file, 'wb') as f:
        pickle.dump(batch_summary, f)
    
    print(f"\nBatch summary saved to: {summary_file}")
    print(f"{'='*60}\n")


if __name__ == '__main__':
    main()