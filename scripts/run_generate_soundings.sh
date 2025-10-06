#!/usr/bin/env bash
set -euo pipefail
mkdir -p outputs
python src/saltelli_driver.py --n_base 512 --out outputs/env_catalog.csv --write_soundings outputs/soundings
