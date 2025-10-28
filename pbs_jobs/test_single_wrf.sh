#!/bin/bash
#PBS -N wrf_test_single
#PBS -q larga
#PBS -l nodes=1:ppn=48
#PBS -j oe
#PBS -o /home/jorge.gacitua/salidas/Supercell_WRF_Sensitivity/logs/test_single_out.log
#PBS -e /home/jorge.gacitua/salidas/Supercell_WRF_Sensitivity/logs/test_single_err.log
#PBS -l walltime=02:00:00

# Test script for running ONE WRF sample
# Usage: qsub src/test_single_wrf.sh

# Load required modules
source /opt/load-libs.sh 1

# Change to repository directory
cd /home/jorge.gacitua/salidas/Supercell_WRF_Sensitivity

# Activate conda environment
source /home/jorge.gacitua/salidas/miniconda3/etc/profile.d/conda.sh
conda activate wrf-sensitivity

# Verify environment
echo "Python: $(which python)"
echo "Python version: $(python --version)"

# Create logs directory
#mkdir -p /home/jorge.gacitua/salidas/Supercell_WRF_Sensitivity/logs
#chmod 755 /home/jorge.gacitua/salidas/Supercell_WRF_Sensitivity/logs

# Run ONE WRF simulation
echo "========================================" 
echo "Starting WRF test simulation..."
echo "Testing sample 0"
echo "========================================" 

python -u src/scripts/step3_run_wrf.py \
    --start 0 \
    --end 1 \
    --nthreads 48\
    --parallel 1

EXIT_CODE=$?

echo "========================================"
echo "Job finished at: $(date)"
echo "Exit code: $EXIT_CODE"
echo "========================================"

# Show results if successful
if [ $EXIT_CODE -eq 0 ]; then
    echo ""
    echo "SUCCESS! Check results at:"
    echo "outputs/sobol_exp_default/wrf_results/sample_00000/metrics.pkl"
    echo ""
    echo "To view metrics:"
    echo "python -c \"import pickle; print(pickle.load(open('outputs/sobol_exp_default/wrf_results/sample_00000/metrics.pkl','rb')))\""
fi

exit $EXIT_CODE