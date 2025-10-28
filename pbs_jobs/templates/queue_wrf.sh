#!/bin/bash
#PBS -N wrf_sobol_{{BATCH_ID}}
#PBS -q {{QUEUE}}
#PBS -l nodes=1:ppn={{NCORES}}
#PBS -j oe
#PBS -o {{REPO_PATH}}/logs/wrf_batch_{{BATCH_ID}}.pbs.log
#PBS -e {{REPO_PATH}}/logs/wrf_batch_{{BATCH_ID}}.pbs.log
#PBS -l walltime={{WALLTIME}}

# This is a template PBS script for running WRF Sobol experiments
# Variables in {{BRACKETS}} will be replaced by the submission script

echo "========================================"
echo "WRF Sobol Sensitivity - Batch {{BATCH_ID}}"
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
cd {{REPO_PATH}}
echo "Working in: $(pwd)"

# Activate conda environment
source {{CONDA_PATH}}/etc/profile.d/conda.sh
conda activate {{CONDA_ENV}}

# Verify environment
echo "--- Python Environment ---"
echo "Python: $(which python)"
echo "Python version: $(python --version)"
python -c "import netCDF4, wrf, yaml; print('Required packages OK')"
echo "--------------------------"
# Create logs directory if it doesn't exist
mkdir -p {{REPO_PATH}}/logs

cd "$PBS_O_WORKDIR" || exit 1

echo "JOBID=$PBS_JOBID"          >&2
echo "WORKDIR=$PWD"              >&2
echo "OUTPUT=$(readlink -f {{REPO_PATH}}/logs/wrf_batch_{{BATCH_ID}}.pbs.log)" >&2
echo "NODEFILE=$PBS_NODEFILE"    >&2

# Run WRF simulations for this batch
echo "========================================" 
echo "Starting WRF simulations..."
echo "Batch range: {{START_IDX}} to {{END_IDX}}"
echo "Threads per run: {{NTHREADS}}"
echo "Parallel runs: {{PARALLEL}}"
echo "========================================" 

python -u src/scripts/step3_run_wrf.py \
    --config {{CONFIG_FILE}} \
    --start {{START_IDX}} \
    --end {{END_IDX}} \
    --nthreads {{NTHREADS}}\
    --parallel 1 \
    --resume

EXIT_CODE=$?

echo "========================================"
echo "Job finished at: $(date)"
echo "Exit code: $EXIT_CODE"
echo "========================================"

exit $EXIT_CODE
