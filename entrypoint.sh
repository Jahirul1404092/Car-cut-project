#!/bin/bash --login
# The --login ensures the bash configuration is loaded,
# enabling Conda.
. /opt/conda/etc/profile.d/conda.sh
echo "Running: CarCut Processing server"
ls
conda activate carCut && python -W ignore main.py

