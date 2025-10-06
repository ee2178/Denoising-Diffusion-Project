#!/bin/bash

#SBATCH --nodes=1
#SBATCH --account=pr_198_general
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:rtx8000:1
#SBATCH --cpus-per-task=1
#SBATCH --mem=32GB
#SBATCH --time=00:01:00
#SBATCH --job-name=test
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=ee2178@nyu.edu
#SBATCH --output=slurm_%j.out
#SBATCH --error=slurm_%j.err
# Activate Conda environment

module load cuda/11.6.2

nvidia-smi

source ~/.bashrc                   # Ensure conda is available
conda activate env     

cd ~/vast/ee2178/Denoising-Diffusion-Project/my_CDLNet       # Replace with the actual path
python3 test.py
