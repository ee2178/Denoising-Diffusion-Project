#!/bin/bash
#SBATCH --nodes=1
#SBATCH --account=torch_pr_89_general
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64GB
#SBATCH --time=12:00:00
#SBATCH --job-name=knee_fastMRI_coil_combine
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=ee2178@nyu.edu
#SBATCH --output=logs/slurm_%j.out
#SBATCH --error=logs/slurm_%j.err

# ----------------------------
# Environment
# ----------------------------
source ~/.bashrc
conda activate env   # same env you used for ESPIRiT

# ----------------------------
# Working directory
# ----------------------------
cd ~/scratch/ee2178/Denoising-Diffusion-Project/my_CDLNet

# ----------------------------
# Run coil combination
# ----------------------------
python3 coil_combine.py \
  --train /home/ee2178/scratch/ee2178/datasets/fastmri/knee/multicoil_train \
  --val   /home/ee2178/scratch/ee2178/datasets/fastmri/knee/multicoil_val \
  --test  /home/ee2178/scratch/ee2178/datasets/fastmri/knee/multicoil_test_v2 \
  --target /home/ee2178/scratch/ee2178/datasets/fastmri_preprocessed/knee_coil_combined \
  --device cuda:0
