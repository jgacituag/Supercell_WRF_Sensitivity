#!/usr/bin/env python
"""
WRF execution module for Sobol sensitivity analysis.

This module handles:
- Setting up WRF ideal.exe runs for each sounding
- Executing WRF simulations
- Extracting key output metrics
- Managing temporary and output directories
"""

import os
import sys
import shutil
import logging
import subprocess
import numpy as np
import netCDF4 as nc
from pathlib import Path
from datetime import datetime


class WRFRun:
    """
    Manage WRF ideal.exe simulations for sensitivity analysis.
    
    Parameters
    ----------
    config : dict
        Configuration dictionary with structure:
        {
            'wrf': {
                'model_path': '/path/to/em_quarter_ss',
                'model_dt': 12,
                'model_dx': 2000.0,
                'model_dy': 2000.0,
                'model_nx': 80,
                'model_ny': 80,
                'model_nz': 51,
                'temp_dir': '/tmp/wrf_runs'
            },
            'paths': {
                'wrf_output_dir': 'outputs/wrf_results'
            }
        }
    logger : logging.Logger, optional
        Logger instance
    """
    
    def __init__(self, config: dict, logger=None):
        """Initialize WRF runner with configuration."""
        self.config = config
        self.logger = logger or logging.getLogger(__name__)
        self.mpi_launcher = config['wrf'].get('mpi_launcher', 'mpirun')
        # WRF paths
        self.wrf_model_path = Path(config['wrf']['model_path'])
        self.namelist_template = self.wrf_model_path / 'namelist.input'
        
        if not self.wrf_model_path.exists():
            raise FileNotFoundError(f"WRF model path not found: {self.wrf_model_path}")
        
        if not self.namelist_template.exists():
            raise FileNotFoundError(f"Namelist template not found: {self.namelist_template}")
        
        # Output paths
        self.output_base = Path(config['paths']['wrf_output_dir'])
        self.temp_base = Path(config['wrf'].get('temp_dir', '/tmp/wrf_runs'))
        
        # Create directories
        self.output_base.mkdir(parents=True, exist_ok=True)
        self.temp_base.mkdir(parents=True, exist_ok=True)
        
        self.logger.info(f"WRF runner initialized:")
        self.logger.info(f"  Model path: {self.wrf_model_path}")
        self.logger.info(f"  Output dir: {self.output_base}")
        self.logger.info(f"  Temp dir:   {self.temp_base}")
    
    
    def run_single_experiment(self, sample_id: int, sounding_file: str,
                             nthreads: int = 1, 
                             cleanup_after: bool = True,
                             keep_wrfout: bool = False):
        """
        Run WRF simulation for a single Sobol sample.
        
        Parameters
        ----------
        sample_id : int
            Sample identifier
        sounding_file : str
            Path to input_sounding file
        nthreads : int, optional
            Number of ,pi threads
        cleanup_after : bool, optional
            Remove temporary run directory after completion
        keep_wrfout : bool, optional
            Keep wrfout files (can be large!)
        
        Returns
        -------
        metrics : dict
            Extracted metrics from simulation, or None if failed
        """
        self.logger.info(f"Starting WRF run for sample {sample_id:05d}")
        
        # Create run directory
        run_dir = self.temp_base / f"wrf_run_{sample_id:05d}"
        run_dir.mkdir(parents=True, exist_ok=True)
        
        try:
            # Setup run environment
            self._setup_run_directory(run_dir, sounding_file)
            
            # Prepare namelist
            self._prepare_namelist(run_dir, sounding_file)
            
            # Run ideal.exe
            self.logger.info(f"  Running ideal.exe...")
            self._run_ideal(run_dir, nthreads)
            
            # Run wrf.exe
            self.logger.info(f"  Running wrf.exe...")
            self._run_wrf(run_dir, nthreads)
            
            # Extract metrics
            self.logger.info(f"  Extracting metrics...")
            metrics = self._extract_metrics(run_dir, sample_id)
            
            # Save outputs
            output_dir = self.output_base / f"sample_{sample_id:05d}"
            output_dir.mkdir(parents=True, exist_ok=True)
            
            self._save_metrics(metrics, output_dir)
            
            if keep_wrfout:
                self._save_wrfout(run_dir, output_dir)
            
            self.logger.info(f"Sample {sample_id:05d} completed successfully")
            
            return metrics
        
        except Exception as e:
            self.logger.error(f"Sample {sample_id:05d} FAILED: {e}")
            return None
        
        finally:
            if cleanup_after and run_dir.exists():
                self.logger.debug(f"Cleaning up: {run_dir}")
                shutil.rmtree(run_dir)
    
    
    def _setup_run_directory(self, run_dir: Path, sounding_file: str):
        """
        Setup WRF run directory with required files.
        
        Parameters
        ----------
        run_dir : Path
            Target run directory
        sounding_file : str
            Path to input_sounding file
        """
        # Copy sounding file
        sounding_dest = run_dir / "input_sounding"
        shutil.copy(sounding_file, sounding_dest)
        
        # Link WRF executables
        ideal_exe = self.wrf_model_path / "ideal.exe"
        wrf_exe = self.wrf_model_path / "wrf.exe"
        
        if not ideal_exe.exists():
            raise FileNotFoundError(f"ideal.exe not found: {ideal_exe}")
        if not wrf_exe.exists():
            raise FileNotFoundError(f"wrf.exe not found: {wrf_exe}")
        
        (run_dir / "ideal.exe").symlink_to(ideal_exe)
        (run_dir / "wrf.exe").symlink_to(wrf_exe)
        
        self.logger.debug(f"Run directory setup complete: {run_dir}")
    
    
    def _prepare_namelist(self, run_dir: Path, sounding_path: Path):
        """
        Prepare namelist.input with values from CONFIG.
        
        Parameters
        ----------
        run_dir : Path
            WRF run directory
        sounding_path : Path
            Path to sounding file (unused, kept for compatibility)
        """
        namelist_path = run_dir / "namelist.input"
        
        # Get values from CONFIG
        wrf_config = self.config['wrf']
        
        # Calculate dt fractions (default to no fraction)
        dt = wrf_config['model_dt']
        dt_fract_num = 0
        dt_fract_den = 1
        
        # Create replacements dictionary
        replacements = {
            '@@NX@@': wrf_config['model_nx'],
            '@@NY@@': wrf_config['model_ny'],
            '@@NZ@@': wrf_config['model_nz'],
            '@@DX@@': wrf_config['model_dx'],
            '@@DY@@': wrf_config.get('model_dy', wrf_config['model_dx']),
            '@@DT@@': dt,
            '@@DT_FRACT_NUM@@': dt_fract_num,
            '@@DT_FRACT_DEN@@': dt_fract_den
        }
        
        # Read template
        with open(self.namelist_template, 'r') as f:
            content = f.read()
        
        # Replace all placeholders
        for placeholder, value in replacements.items():
            content = content.replace(placeholder, str(value))
        
        # Write modified namelist
        with open(namelist_path, 'w') as f:
            f.write(content)
        
        self.logger.debug(f"Prepared namelist: NX={replacements['@@NX@@']}, "
                         f"NY={replacements['@@NY@@']}, NZ={replacements['@@NZ@@']}, "
                         f"DX={replacements['@@DX@@']}, DT={replacements['@@DT@@']}")
    
    
    def _run_ideal(self, run_dir: Path, nthreads: int):
        """
        Execute ideal.exe to generate initial conditions.
        
        Parameters
        ----------
        run_dir : Path
            WRF run directory
        nthreads : int
            Number of OpenMP threads
        """
        cmd = ['./ideal.exe']
        
        result = subprocess.run(
            cmd,
            cwd=run_dir,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        
        if result.returncode != 0:
            self.logger.error(f"ideal.exe failed with code {result.returncode}")
            self.logger.error(f"STDERR: {result.stderr}")
            raise RuntimeError("ideal.exe execution failed")
        
        # Check for wrfinput file
        wrfinput = run_dir / "wrfinput_d01"
        if not wrfinput.exists():
            raise RuntimeError("ideal.exe did not produce wrfinput_d01")
    
    
    def _run_wrf(self, run_dir: Path, nthreads: int):
        """
        Run wrf.exe with MPI.
        
        Parameters
        ----------
        run_dir : Path
            WRF run directory
        nthreads : int
            Number of MPI processes
        """
        cmd = [self.mpi_launcher, '-np', str(nthreads), './wrf.exe']
        self.logger.info(f"Running MPI command: {' '.join(cmd)}")
        
        with open(run_dir/'wrf.stdout', 'w') as so, open(run_dir/'wrf.stderr', 'w') as se:
            result = subprocess.run(
                cmd,
                cwd=run_dir,
                stdout=so,              # write to file
                stderr=se,              # write to file
                text=True
            )
        
        if result.returncode != 0:
            # Also echo small tail snippets into the main logger for convenience
            try:
                tail_err = (run_dir/'wrf.stderr').read_text()[-2000:]
            except Exception:
                tail_err = "(failed to read wrf.stderr)"
            self.logger.error(f"wrf.exe failed with code {result.returncode}")
            self.logger.error(f"stderr (tail):\n{tail_err}")
            raise RuntimeError("wrf.exe execution failed")

        wrfout = list(run_dir.glob("wrfout_d01_*"))
        if not wrfout:
            raise RuntimeError("wrf.exe did not produce wrfout file")
        self.logger.info(f"WRF completed successfully. Output: {wrfout[0]}")
    
    
    def _extract_metrics(self, run_dir: Path, sample_id: int):
        """
        Extract key metrics from WRF output.
        
        Parameters
        ----------
        run_dir : Path
            WRF run directory
        sample_id : int
            Sample identifier
        
        Returns
        -------
        metrics : dict
            Dictionary of extracted metrics
        """
        # Find wrfout file
        wrfout_files = sorted(run_dir.glob("wrfout_d01_*"))
        if not wrfout_files:
            raise FileNotFoundError("No wrfout files found")
        
        wrfout = wrfout_files[-1]  # Use last output time
        
        metrics = {
            'sample_id': sample_id,
            'success': True,
            'timestamp': datetime.now().isoformat()
        }
        
        try:
            with nc.Dataset(wrfout, 'r') as ds:
                # Get dimensions
                nt = ds.dimensions['Time'].size
                nz = ds.dimensions['bottom_top'].size
                ny = ds.dimensions['south_north'].size
                nx = ds.dimensions['west_east'].size
                
                metrics['dimensions'] = {'nt': nt, 'nz': nz, 'ny': ny, 'nx': nx}
                
                # Extract vertical velocity (W)
                if 'W' in ds.variables:
                    w = ds.variables['W'][:]  # Shape: (nt, nz+1, ny, nx)
                    
                    # Maximum updraft
                    metrics['max_updraft'] = float(np.max(w))
                    metrics['min_downdraft'] = float(np.min(w))
                    
                    # Updraft statistics at mid-levels (e.g., 3-8 km)
                    # Approximate height levels (assuming dz ~ 400m near surface)
                    z_low = max(0, nz // 4)
                    z_high = min(nz, 3 * nz // 4)
                    w_mid = w[:, z_low:z_high, :, :]
                    
                    metrics['max_updraft_midlevel'] = float(np.max(w_mid))
                    metrics['mean_updraft_midlevel'] = float(np.mean(w_mid[w_mid > 1.0]))
                

                if 'QRAIN' in ds.variables:
                    qrain = ds.variables['QRAIN'][:]
                    metrics['max_qrain'] = float(np.max(qrain))
                
                if 'QGRAUP' in ds.variables:
                    qgraup = ds.variables['QGRAUP'][:]
                    metrics['max_qgraup'] = float(np.max(qgraup))
                
                # Maximum perturbation pressure
                if 'P' in ds.variables:
                    p_pert = ds.variables['P'][:]
                    metrics['max_pressure_pert'] = float(np.max(p_pert))
                    metrics['min_pressure_pert'] = float(np.min(p_pert))
        
        except Exception as e:
            self.logger.warning(f"Error extracting some metrics: {e}")
            metrics['extraction_warning'] = str(e)
        
        return metrics
    
    
    def _save_metrics(self, metrics: dict, output_dir: Path):
        """Save extracted metrics to file."""
        import pickle
        
        metrics_file = output_dir / "metrics.pkl"
        with open(metrics_file, 'wb') as f:
            pickle.dump(metrics, f)
        
        self.logger.debug(f"Metrics saved: {metrics_file}")
    
    
    def _save_wrfout(self, run_dir: Path, output_dir: Path):
        """Copy wrfout files to output directory."""
        wrfout_files = list(run_dir.glob("wrfout_d01_*"))
        
        for wrfout in wrfout_files:
            dest = output_dir / wrfout.name
            shutil.copy(wrfout, dest)
            self.logger.debug(f"Saved wrfout: {dest}")


def main():
    """Example/test usage of WRFRun class."""
    
    # Example configuration
    config = {
        'wrf': {
            'model_path': '/home/user/WRF/em_quarter_ss',
            'model_dt': 12,
            'model_dx': 2000.0,
            'model_dy': 2000.0,
            'model_nx': 80,
            'model_ny': 80,
            'model_nz': 51,
            'temp_dir': '/tmp/wrf_runs'
        },
        'paths': {
            'wrf_output_dir': 'outputs/wrf_results'
        }
    }
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Initialize runner
    wrf = WRFRun(config)
    
    # Run test sample
    sounding_file = "data/soundings/input_sounding_00000"
    metrics = wrf.run_single_experiment(
        sample_id=0,
        sounding_file=sounding_file,
        nthreads=4,
        cleanup_after=False,
        keep_wrfout=True
    )
    
    print("\n=== Extracted Metrics ===")
    for key, value in metrics.items():
        print(f"{key:30s}: {value}")


if __name__ == '__main__':
    main()