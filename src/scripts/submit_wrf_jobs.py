#!/usr/bin/env python
"""
Helper script to generate and submit PBS jobs for WRF executions.

This creates multiple PBS job scripts from a template, each handling
a batch of Sobol samples, and optionally submits them to the queue.

Usage:
    python submit_wrf_jobs.py --batch-size 50 --submit
"""

import os
import sys
import yaml
import argparse
import numpy as np
from pathlib import Path


def load_config(config_file):
    """Load experiment configuration."""
    with open(config_file, 'r') as f:
        return yaml.safe_load(f)


def get_viable_count(experiment_dir):
    """Get number of viable samples."""
    viable_file = experiment_dir / 'soundings' / 'viable_sample_ids.npy'
    if not viable_file.exists():
        raise FileNotFoundError(
            f"Viable samples not found. Run step2b_filter_soundings.py first."
        )
    viable_ids = np.load(viable_file)
    return len(viable_ids)


def create_job_script(template_file, output_file, replacements):
    """
    Create PBS job script from template with replacements.
    
    Parameters
    ----------
    template_file : Path
        Template PBS script
    output_file : Path
        Output job script path
    replacements : dict
        Dictionary of {{KEY}}: value replacements
    """
    with open(template_file, 'r') as f:
        content = f.read()
    
    for key, value in replacements.items():
        content = content.replace(f'{{{{{key}}}}}', str(value))
    
    with open(output_file, 'w') as f:
        f.write(content)
    
    # Make executable
    os.chmod(output_file, 0o755)
    
    print(f"Created: {output_file}")


def main():
    parser = argparse.ArgumentParser(
        description='Generate and submit PBS jobs for WRF Sobol experiments'
    )
    parser.add_argument(
        '--config',
        type=str,
        default='configs/experiment_config.yaml',
        help='Path to experiment configuration'
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        required=True,
        help='Number of samples per PBS job'
    )
    parser.add_argument(
        '--template',
        type=str,
        default='pbs_jobs/templates/queue_wrf.sh',
        help='PBS job template file'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='pbs_jobs/generated',
        help='Directory for generated job scripts'
    )
    parser.add_argument(
        '--submit',
        action='store_true',
        help='Submit jobs to queue after creating them'
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Show what would be done without creating files'
    )
    
    args = parser.parse_args()
    
    # Paths
    repo_root = Path(__file__).resolve().parents[2]
    config_file = repo_root / args.config
    template_file = repo_root / args.template
    output_dir = repo_root / args.output_dir
    
    # Load config
    if not config_file.exists():
        print(f"ERROR: Config file not found: {config_file}")
        sys.exit(1)
    
    config = load_config(config_file)
    
    # Check template
    if not template_file.exists():
        print(f"ERROR: Template not found: {template_file}")
        print(f"Create it at: {template_file}")
        sys.exit(1)
    
    # Get experiment info
    experiment_name = config['PATHS']['EXPERIMENT_NAME']
    experiment_dir = repo_root / 'outputs' / experiment_name
    
    try:
        n_viable = get_viable_count(experiment_dir)
    except FileNotFoundError as e:
        print(f"ERROR: {e}")
        sys.exit(1)
    
    # Calculate batches
    n_batches = int(np.ceil(n_viable / args.batch_size))
    
    print(f"\n{'='*60}")
    print(f"PBS Job Generation for WRF Sobol Experiments")
    print(f"{'='*60}")
    print(f"Experiment:       {experiment_name}")
    print(f"Viable samples:   {n_viable}")
    print(f"Batch size:       {args.batch_size}")
    print(f"Number of jobs:   {n_batches}")
    print(f"Template:         {template_file.name}")
    print(f"{'='*60}\n")
    
    if args.dry_run:
        print("DRY RUN - No files will be created\n")
    
    # Create output directory
    if not args.dry_run:
        output_dir.mkdir(parents=True, exist_ok=True)
        logs_dir = repo_root / 'logs'
        logs_dir.mkdir(exist_ok=True)
    
    # Get WRF settings from config
    wrf_config = config.get('WRF', {})
    queue = wrf_config.get('PBS_QUEUE', 'larga')
    ncores = wrf_config.get('PBS_NCORES', 48)
    nthreads = wrf_config.get('MPI_PROCS', 48)
    parallel = wrf_config.get('PARALLEL_RUNS', 1)
    walltime = wrf_config.get('PBS_WALLTIME', '24:00:00')
    
    # Conda settings
    conda_path = wrf_config.get('CONDA_PATH', str(Path.home() / 'miniconda3'))
    conda_env = wrf_config.get('CONDA_ENV', 'wrf_python')
    
    # Generate job scripts
    job_files = []
    
    for batch_id in range(n_batches):
        start_idx = batch_id * args.batch_size
        end_idx = min((batch_id + 1) * args.batch_size, n_viable)
        
        # Replacements for template
        replacements = {
            'BATCH_ID': batch_id,
            'QUEUE': queue,
            'NCORES': ncores,
            'NTHREADS': nthreads,
            'PARALLEL': parallel,
            'WALLTIME': walltime,
            'REPO_PATH': str(repo_root),
            'CONFIG_FILE': args.config,
            'START_IDX': start_idx,
            'END_IDX': end_idx,
            'CONDA_PATH': conda_path,
            'CONDA_ENV': conda_env
        }
        
        job_file = output_dir / f'wrf_batch_{batch_id:03d}.sh'
        
        print(f"Batch {batch_id:03d}: samples {start_idx:4d}-{end_idx:4d} → {job_file.name}")
        
        if not args.dry_run:
            create_job_script(template_file, job_file, replacements)
            job_files.append(job_file)
    
    # Submit jobs
    if args.submit and not args.dry_run:
        print(f"\n{'='*60}")
        print("Submitting jobs to PBS queue...")
        print(f"{'='*60}\n")
        
        for job_file in job_files:
            cmd = f"qsub {job_file}"
            print(f"Submitting: {job_file.name}")
            result = os.system(cmd)
            if result != 0:
                print(f"  WARNING: Submission may have failed (exit code {result})")
        
        print(f"\n{len(job_files)} jobs submitted!")
        print(f"Monitor with: qstat -u $USER")
    
    elif args.submit and args.dry_run:
        print("\n[DRY RUN] Would submit jobs with: qsub <job_file>")
    
    else:
        print(f"\nJob scripts created in: {output_dir}")
        print(f"Submit manually with: qsub <job_file>")
        print(f"Or re-run with --submit flag")
    
    log_file = repo_root / 'logs' / f'wrf_batch_{batch_id:03d}.pbs.log'
    replacements.update({
        'LOG_FILE': str(log_file),
    })   


if __name__ == '__main__':
    main()
