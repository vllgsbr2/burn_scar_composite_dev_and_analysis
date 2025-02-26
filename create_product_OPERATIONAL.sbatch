#!/bin/bash
#SBATCH --ntasks=1
#SBATCH -t 03:00:00
#SBATCH --mem-per-cpu=16384
#SBATCH --job-name='create_product'
#SBATCH --error='create_product.err'

source /home/vllgsbr2/.bashrc
source activate burnscar

cd /scratch/zt1/project/vllgsbr2-prj/git_burn_scar_composite_dev_and_analysis
python create_product.py OPERATIONAL

