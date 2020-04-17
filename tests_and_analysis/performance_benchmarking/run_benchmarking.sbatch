#!/bin/bash
#SBATCH -C scarf18
#SBATCH -p devel
#SBATCH -c 24
#SBATCH -n 1
#SBATCH -t 10:00
#SBATCH --job-name=euphonic_perform
#SBATCH --exclusive
#SBATCH -v
#SBATCH -o %j.log
#SBATCH -e %j.err

module load python/3.6.5-gcc-4.8.5
python -m pip install numpy
python -m pip install -e .[phonopy]
python run_benchmark_tests.py
