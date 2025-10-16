# WRF Sobol Sensitivity Analysis

Modular workflow for Sobol sensitivity analysis on WRF convective idealized simulations.

## Quick Start
conda env create -f environment.yml
conda activate wrf-sensitivity

### 2. Initial default Run (500 samples)

```bash
# Step 1: Generate Sobol samples
python src/step1_generate_samples.py --output_dir outputs/sobol_exp_500

# Step 2: Generate soundings
python src/step2_generate_soundings.py --input_dir outputs/sobol_exp_500
python src/step2_generate_soundings.py --input_dir sobol_exp

# Skip diagnostics (fast - recommended for large runs)
python src/step2_generate_soundings.py --input_dir sobol_exp --skip_diagnostics

# Parallel with MetPy
python src/step2_generate_soundings.py --input_dir sobol_exp --batch_id 0

# Step 3: Configure and run WRF (edit wrf_config_template.pkl first)
sbatch run_wrf_array.slurm

# Step 4: Merge results
python merge_results.py --input_dir sobol_exp --type all

# Step 5: Check if you need more samples
# (You'll need to create step4 and step5 based on your WRF diagnostics)
```

## Key Features

- **Thermodynamically consistent** soundings (no excessive CIN)
- **Southern Hemisphere** wind profiles (clockwise hodographs)
- **Cluster-optimized** for parallel execution
- **Convergence analysis** to determine required samples
- **Modular design** for easy customization

## File Structure

```
├── sounding_generator.py      # Core sounding generation functions
├── step1_generate_samples.py  # Generate Sobol samples
├── step2_generate_soundings.py # Create WRF input files
├── step3_run_wrf.py           # Run WRF simulations
├── run_wrf_array.slurm        # SLURM array job template
├── merge_results.py           # Merge parallel results
├── extract_scripts.py         # Utility to extract all files
└── extract_scripts.ps1        # Windows extraction script
```

## Parameters (Adjustable in step1)

1. **cape_target** (500-4000 J/kg): Target CAPE value
2. **low_level_rh** (70-95%): Boundary layer moisture
3. **mid_level_rh** (30-70%): Mid-level moisture
4. **upper_level_rh** (10-40%): Upper troposphere moisture
5. **surface_theta** (295-305 K): Surface potential temperature
6. **low_level_lapse** (6-8 K/km): Low-level lapse rate
7. **mid_level_lapse** (5-7 K/km): Mid-level lapse rate
8. **shear_magnitude** (10-40 m/s): 0-6km bulk shear
9. **shear_curvature** (0-1): Hodograph shape (0=linear, 1=circular)
10. **shear_direction** (0-360°): Wind shear direction
11. **low_level_jet** (0-15 m/s): LLJ strength

## Workflow

### Phase 1: Initial Exploration
```bash
# 500 base samples → ~12,000 total simulations
python step1_generate_samples.py --n_samples 500
python step2_generate_soundings.py --input_dir sobol_exp
# Submit to cluster
sbatch run_wrf_array.slurm
# Analyze convergence (you'll need to implement this)
```

### Phase 2: Determine if Sufficient
- If converged: Compute Sobol indices and done!
- If not converged: Generate additional samples as recommended

## Customization

### Add New Parameters
Edit `define_problem()` in `step1_generate_samples.py`:
```python
problem = {
    'num_vars': 12,  # Add 1
    'names': [..., 'your_new_param'],
    'bounds': [..., [min_val, max_val]]
}
```

Then modify `generate_realistic_sounding()` in `sounding_generator.py`.

### Cluster Settings
Edit `run_wrf_array.slurm`:
```bash
#SBATCH --array=0-119          # Number of batches
#SBATCH --cpus-per-task=4      # Cores per simulation
#SBATCH --mem=16G              # Memory per job
#SBATCH --time=02:00:00        # Time limit
```

## Requirements

```bash
pip install numpy scipy SALib netCDF4 wrf-python matplotlib
```

## License
MIT — see [LICENSE](LICENSE).
