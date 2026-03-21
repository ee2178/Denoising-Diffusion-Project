#!/bin/bash
#SBATCH --nodes=1
#SBATCH --account=torch_pr_89_tandon_advanced
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64GB
#SBATCH --time=48:00:00
#SBATCH --job-name=knee_fastMRI_espirit
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=ee2178@nyu.edu
#SBATCH --output=logs/slurm_%j.out
#SBATCH --error=logs/slurm_%j.err

# module load cuda/11.6.2

# Activate Conda environment

source ~/.bashrc                   # Ensure conda is available
conda activate env      # Replace with your actual env name

# Navigate to the directory containing train.py
cd ~/scratch/ee2178/Denoising-Diffusion-Project/my_CDLNet       # Replace with the actual path

python3 smap_est.py --train /home/ee2178/scratch/ee2178/datasets/fastmri/knee/multicoil_train --val /home/ee2178/scratch/ee2178/datasets/fastmri/knee/multicoil_val --test /home/ee2178/scratch/ee2178/datasets/fastmri/knee/multicoil_test_v2 --target /home/ee2178/scratch/ee2178/datasets/fastmri_preprocessed/knee_coil_combined
