#!/bin/bash
#SBATCH --job-name=pathgan
#SBATCH --output=./outputs/%x_%j.out
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu
#SBATCH --mem=60000

# Module load
module load cuda/10.0.130/intel-19.0.3.199

# Activate anaconda environment code
source activate $WORKDIR/miniconda3/envs/pytorch

# execution
python main.py ${SLURM_JOBID} ./config/config_camelyon.yml
python main.py ${SLURM_JOBID} ./config/config_crc.yml
