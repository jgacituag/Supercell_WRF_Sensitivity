#!/usr/bin/env bash
set -euo pipefail
mkdir -p outputs
# Example thresholds for low-CAPE & high-shear
python src/preselect_envs.py --csv outputs/env_catalog.csv --modelpath wrf_case   --out outputs/env_prescreen.csv --want_lowcape_hishear 20 --lowcape_th 1000 --hishear_th 25
