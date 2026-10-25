#!/usr/bin/env bash
# Runs the whole analysis from the repository root, in order.
# Plots are written to results/plots/ without opening any window.
set -euo pipefail
export MPLBACKEND=Agg

for script in src/01_generate_data.py \
              src/02_naive_estimation.py \
              src/03_causal_forest.py \
              src/04_uplift_modeling.py \
              src/05_statistical_validation.py \
              src/06_roi_targeting.py \
              src/07_robustness.py; do
    echo
    echo ">>> Running $script"
    python "$script"
done
