#!/bin/bash
#SBATCH --ntasks=1
#SBATCH -t 00:30:00
#SBATCH --mem-per-cpu=8192
#SBATCH --job-name='build_burn_scar_VIIRS_database_OPERATIONAL'
#SBATCH --error='build_burn_scar_VIIRS_database_OPERATIONAL.err'

source /home/vllgsbr2/.bashrc
source activate burnscar

cd /scratch/zt1/project/vllgsbr2-prj/git_burn_scar_composite_dev_and_analysis
python build_burn_scar_VIIRS_database.py OPERATIONAL

