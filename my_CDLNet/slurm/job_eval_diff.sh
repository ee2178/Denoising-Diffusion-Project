#!/bin/bash
#SBATCH --nodes=1
#SBATCH --account=pr_198_general
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=1
#SBATCH --mem=32GB
#SBATCH --time=4:00:00
#SBATCH --job-name=ImMAP2_Eval_LPDSNet
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
python3 eval_diff.py eval_config.json --kspace_path=../../datasets/fastmri/brain/multicoil_val --smap_path=../../datasets/fastmri_preprocessed/brain_T2W_coil_combined/val/ --noise_level=0.05 --save_name="eval_results/lpds_e2e.txt" --eval_e2e=True
