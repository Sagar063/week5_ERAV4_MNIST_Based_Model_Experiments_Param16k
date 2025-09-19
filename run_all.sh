#!/usr/bin/env bash
set -e
python train.py --mode grid
python update_readme.py
echo "All done. See results/results.csv and README.cmd (with plots)."
