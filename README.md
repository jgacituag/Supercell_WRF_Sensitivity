# WRF Sobol Sensitivity Analysis

A modular workflow for Sobol sensitivity analysis on idealized WRF supercell simulations. This project uses a configuration-driven approach to generate, process, and filter a parameter space for a WRF sensitivity experiment.

## Installation

1.  Clone the repository:
    ```bash
    git clone git@github.com:jgacituag/Supercell_WRF_Sensitivity.git
    cd Supercell_WRF_Sensitivity
    ```

2.  Create the Conda environment:
    ```bash
    conda env create -f environment.yml
    conda activate wrf-sensitivity
    ```
    *(Alternatively, if you prefer pip: `pip install -r requirements.txt`)*

## Workflow

This project is driven by a central configuration file. The main workflow is to **copy**, **edit**, and **run** based on this file.

### Configure Your Experiment

Copy the default configuration to create your new experiment file.

```bash
cp configs/default_config.yaml configs/experiment_config.yaml
````

Now, **edit `configs/experiment_config.yaml`** to define your experiment:

  * `PATHS.EXPERIMENT_NAME`: Change this from `"sobol_exp_default"` to a unique name (e.g., `"sobol_exp_650"`).
  * `SOBOL.N_SAMPLES_BASE`: Set your desired number of base samples N ( total samples = N(2*D+2) e.g., `25` gives `25 * (2*12 + 2) = 650` total samples).
  * `SOBOL.PROBLEM_DEFINITION`: Define the D parameters and their bounds.
  * `SOUNDINGS.RUN_PARALLEL`: Set to `true` to use multiprocessing for step 2.
  * `FILTER`: Adjust the `_MIN` and `_MAX` thresholds for any diagnostic variable (`MUCAPE`, `MUCIN`, etc.). Set a value to `null` to disable that specific filter.
  * Full Options inside the `configs/experiment_config.yaml` file

### Step 1: Generate Samples

This script reads your config file, defines the Sobol problem, and generates the `param_values.npy` file.

```bash
python src/scripts/step1_generate_samples.py --config configs/experiment_config.yaml
```

### Step 2: Generate Soundings

This script reads the `param_values.npy` file and generates an `input_sounding_XXXXX` file for *every* sample. It will run in parallel if `RUN_PARALLEL: true` is set in your config.

```bash
python src/scripts/step2_generate_soundings.py --config configs/experiment_config.yaml
```

### 4\. Step 2b: Filter Soundings

This script reads the `diagnostics.pkl` file (created in Step 2), applies the flexible filters you defined in `configs/experiment_config.yaml`, and saves the final list of "good" simulations to run.

```bash
python src/scripts/step2b_filter_soundings.py --config configs/experiment_config.yaml
```

### 5\. Run WRF

You are now ready to run WRF. The key outputs you need are:

  * **Viable Job List:** `outputs/experiment_config/soundings/viable_sample_ids.npy`
    This file contains the *only* sample IDs you need to run (e.g., `[0, 2, 3, 5, 8, ...]`).
  * **Sounding Files:** `outputs/experiment_config/soundings/input_sounding_XXXXX`
    These are the individual `input_sounding` files for each viable WRF simulation.

You can now use this information to launch your HPC/SLURM array jobs.

## Repository Structure

```
.
├── configs/
│    ├── default_config.yaml      # Default template, do not edit
│    └── experiment_config.yaml   # Your experiment's configuration (edit this)
├── environment.yml              # Conda environment
├── LICENSE
├── notebooks/                   # Jupyter notebooks for exploration and plotting
│    ├── 01_sobol_exploration.ipynb
│    └── 02_plot_soundings_envelope.ipynb
├── outputs/
│    └── (Empty)           # default experiment output directory
├── README.md                    # This file
├── requirements.txt             # Pip requirements file
└── src/
    ├── core/                    # Core Python modules
    │    └── sounding_generator.py # The main sounding generation logic
    ├── input_sounding           # The base WRF idealized input_sounding template
    └── scripts/                 # Main executable scripts
         ├── step1_generate_samples.py
         ├── step2_generate_soundings.py
         └── step2b_filter_soundings.py
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
