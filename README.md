# WRF Sobol Sensitivity Analysis

A modular workflow for Sobol sensitivity analysis on idealized WRF supercell simulations. This project uses a configuration-driven approach to generate, process, and filter a parameter space for a WRF sensitivity experiment.

---

## Quick start

```bash
# 1) create env
conda env create -f environment.yml
conda activate wrf-sensitivity

# 2) copy & edit your experiment config
cp configs/default_config.yaml configs/experiment_config.yaml
#   - set PATHS.EXPERIMENT_DIR (e.g., outputs/sobol_exp_default)
#   - define parameters (problem) and Saltelli options
#   - tune filters (optional)
#   - set metrics_job and sobol_job blocks

# 3) Step 1: sampling
python src/scripts/step1_generate_samples.py --config configs/experiment_config.yaml

# 4) Step 2: generate soundings
python src/scripts/step2_generate_soundings.py --config configs/experiment_config.yaml

# 5) (optional) Step 2b: filter
python src/scripts/step2b_filter_soundings.py --config configs/experiment_config.yaml
#     This creates soundings/viable_sample_ids.npy
#     If you want to run ALL samples, create a viable list covering 0..N-1.

# 6) Step 3: run WRF on HPC in batches
#     Templates live in pbs_jobs/templates/
#     Generated job files go to pbs_jobs/generated/ (not tracked by git)
python src/scripts/submit_wrf_jobs.py --config configs/experiment_config.yaml --batch-size 500 --submit

# 7) Step 4: compute metrics (config-driven; supports --skip-existing)
python src/scripts/step4_compute_metrics.py --config configs/experiment_config.yaml --skip-existing

# 8) Step 5: Sobol indices + CIs + convergence check
python src/scripts/step5_compute_sobol.py --config configs/experiment_config.yaml
## Repository Structure

```
.
├── configs/                    # default + your editable experiment config
│    ├── default_config.yaml      # Default template, do not edit
├── notebooks/                  # exploration only (not required for the pipeline)
├── outputs/                    # heavy results (ignored by git)
│   └── <EXPERIMENT_DIR>/
│       ├── soundings/          # input_sounding_XXXXX, viable_sample_ids.npy
│       ├── wrf_results/        # sample_00000/ (wrfout_d0*, metrics.pkl, etc.)
│       ├── sobol/              # Sobol indices results per metric
│       └── (metadata.pkl, problem.pkl, param_values.npy, experiment_summary.txt)
├── pbs_jobs/
│   ├── templates/              # queue_wrf.sh (template under version control)
│   ├── generated/              # wrf_batch_*.sh (auto; ignored by git)
│   └── metrics_batch.sh        # optional batch for step4 on the queue
├── src/
│   ├── core/                   # plotting, WRF helpers, sounding generator
│   └── scripts/                # step1..step5 + submitter + verifiers
├── README.md                    # This file
├── requirements.txt             # Pip requirements file
├── environment.yml              # Conda environment
└── LICENSE
```

## License

MIT — see [LICENSE](https://www.google.com/search?q=LICENSE).

## Contributors & Acknowledgements

This project was made possible by the valuable contributions and guidance from:


* **Juan Jose Ruiz:** WRF simulation and high-performance computing.
* **Rafael Bellester-Ripoll:** Sobol sensitivity analysis.
* **Paola Salio:** Convective behavior analysis.
* **Esteban Semino:** Classification of convection.
* **Jorge Gacitúa Gutiérrez([@jgacituag](https://github.com/jgacituag)):** Project integration.
