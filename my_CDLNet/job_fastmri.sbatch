#!/bin/bash
#SBATCH --nodes=1
#SBATCH --account=pr_198_general
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=1
#SBATCH --mem=48GB
#SBATCH --time=15:00:00
#SBATCH --job-name=fastMRI_Coil_Combination
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

python3 mri_data.py --train /home/ee2178/vast/ee2178/datasets/fastmri/brain/multicoil_train --val /home/ee2178/vast/ee2178/datasets/fastmri/brain/multicoil_val --test /home/ee2178/vast/ee2178/datasets/fastmri/brain/multicoil_test --target /home/ee2178/vast/ee2178/datasets/fastmri_preprocessed/brain_T2W_coil_combined
