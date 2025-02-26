#!/bin/bash
#SBATCH --ntasks=1
#SBATCH -t 00:10:00
#SBATCH --mem-per-cpu=32768
#SBATCH --job-name='fix_plotting'
#SBATCH --error='fix_plotting.err'

source /home/vllgsbr2/.bashrc
source activate burnscar

cd /scratch/zt1/project/vllgsbr2-prj/git_burn_scar_composite_dev_and_analysis
python fix_plotting.py
