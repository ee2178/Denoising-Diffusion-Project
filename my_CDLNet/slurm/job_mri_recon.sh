#!/bin/bash
#SBATCH --nodes=1
#SBATCH --account=torch_pr_89_general
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --constraint=l40s
#SBATCH --cpus-per-task=4
#SBATCH --mem=32GB
#SBATCH --time=48:00:00
#SBATCH --job-name=LPDSNet-FastMRI-MRI-Reconstruction-IMMAP2p5
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=ee2178@nyu.edu
#SBATCH --output=logs/slurm_%j.out
#SBATCH --error=logs/slurm_%j.err

# Activate Conda environment
source ~/.bashrc                   # Ensure conda is available
conda activate env      # Replace with your actual env name

# Navigate to the directory containing train.py
cd ~/scratch/ee2178/Denoising-Diffusion-Project/my_CDLNet       # Replace with the actual path
python3 train_mri_recon.py mri_config.json
