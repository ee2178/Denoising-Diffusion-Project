#!/bin/bash
#SBATCH --nodes=1
#SBATCH --account=pr_198_general
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=1
#SBATCH --mem=32GB
#SBATCH --time=24:00:00
#SBATCH --job-name=LPDSNet-FastMRI
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=ee2178@nyu.edu
#SBATCH --output=slurm_%j.out
#SBATCH --error=slurm_%j.err

module load cuda/11.6.2

# Activate Conda environment

source ~/.bashrc                   # Ensure conda is available
conda activate env      # Replace with your actual env name

# Navigate to the directory containing train.py

cd ~/vast/ee2178/Denoising-Diffusion-Project/my_CDLNet       # Replace with the actual path
python3 train.py config.json
