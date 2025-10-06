# Supercell_WRF_Sensitivity

**generate, prescreen, and run WRF idealized supercell experiments**
with **Sobol/Saltelli** sampling. 

## Highlights
- Analytic environment generator (`src/envgen_analytic.py`) with smooth, piecewise profiles:
  surface temperature, tropospheric lapse, tropopause height, surface RH, moist-layer depth,
  shear magnitude/depth, hodograph curvature, optional LLJ.
- Saltelli driver (`src/saltelli_driver.py`) to create soundings + diagnostics (MLCAPE/MLCIN, PW, 0–1/0–6 km shear).
- Pre-screen utility (`src/preselect_envs.py`) to ensure coverage (e.g., **low-CAPE/high-shear** corner) before running WRF.
- Plot utility (`src/plot_soundings_envelope.py`) to show **base vs. envelope** of modified soundings (thick line + fill-between).
- WRF-friendly `input_sounding` writer.

## Quickstart
```bash
# 1) Create environment
conda env create -f environment.yml
conda activate wrf-sensitivity

# 2) Generate Saltelli set, diagnostics, and WRF input_sounding files
python src/saltelli_driver.py --n_base 512 --out outputs/env_catalog.csv --write_soundings outputs/soundings

# 3) Pre-screen coverage (optional; e.g., pick 20 low-CAPE/high-shear indices)
python src/preselect_envs.py --csv outputs/env_catalog.csv --modelpath wrf_case --out outputs/env_prescreen.csv   --want_lowcape_hishear 20 --lowcape_th 1000 --hishear_th 25

# 4) Plot envelope against your base sounding
python src/plot_soundings_envelope.py --csv outputs/env_catalog.csv --modelpath wrf_case   --base_input input_sounding --outfig outputs/soundings_envelope.png
```

## Repository layout
```
Supercell_WRF_Sensitivity/
  ├─ src/
  │   ├─ envgen_analytic.py          # analytic soundings + diagnostics + writer
  │   ├─ saltelli_driver.py          # Saltelli sampling + generation
  │   ├─ preselect_envs.py           # prescreen CAPE/CIN/shear coverage
  │   ├─ plot_soundings_envelope.py  # base vs envelope plots
  │   └─ mapper_u01_to_conf.py       # optional mapper for existing modify_sounding workflows
  ├─ wrf_case/
  │   └─ input_sounding              # put your base sounding here (for plotting/compat)
  ├─ scripts/
  │   ├─ run_generate_soundings.sh   # example: generate & store soundings
  │   └─ run_prescreen.sh            # example: prescreen coverage
  ├─ configs/
  │   └─ problem_u01.yaml            # example problem definition for record-keeping
  ├─ notebooks/                      # analysis notebooks (empty)
  ├─ data/                           # place for small inputs (avoid committing WRF outputs)
  ├─ outputs/                        # (gitignored) results, catalogs, figures, soundings
  ├─ environment.yml
  ├─ requirements.txt
  ├─ LICENSE
  └─ .github/workflows/ci.yml
```

## License
MIT — see [LICENSE](LICENSE).
