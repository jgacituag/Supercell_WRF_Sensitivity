#!/bin/bash
#PBS -N wrf_metrics
#PBS -l nodes=1:ppn=48
#PBS -l walltime=06:00:00
#PBS -j oe
#PBS -o logs/metrics_batch.log

echo "========================================"
echo "WRF Sobol Sensitivity - METRICS"
echo "========================================"
echo "Job started at: $(date)"
echo "Running on host: $(hostname)"
echo "PBS Job ID: $PBS_JOBID"
echo "Working directory: $PBS_O_WORKDIR"
echo "Current dir: $(pwd)"
echo "========================================" 
# Print environment info
echo "--- Environment ---"
echo "USER: $USER"
echo "PATH: $PATH"
echo "LD_LIBRARY_PATH: $LD_LIBRARY_PATH"
echo "-------------------"
# Load required modules
source /opt/load-libs.sh 1

# Change to repository directory
cd /nfsmounts/storage/scratch/jorge.gacitua/Supercell_WRF_Sensitivity
echo "Working in: $(pwd)"

# Activate conda environment
source /home/jorge.gacitua/salidas/miniconda3/etc/profile.d/conda.sh
conda activate wrf-sensitivity

cd "$PBS_O_WORKDIR" || exit 1

python python src/scripts/step4_compute_metrics.py --config configs/experiment_config.yaml




