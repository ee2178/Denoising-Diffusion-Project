#!/bin/bash
#SBATCH --nodes=1
#SBATCH --account=torch_pr_89_general
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --constraint=l40s
#SBATCH --cpus-per-task=1
#SBATCH --mem=32GB
#SBATCH --time=24:00:00
#SBATCH --job-name=ImMAP2_Array_Eval_LPDSNet
#SBATCH --array=0-3
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=ee2178@nyu.edu
#SBATCH --output=logs/slurm_%x_%A_%a.out
#SBATCH --error=logs/slurm_%x_%A_%a.err

source ~/.bashrc
conda activate env

cd ~/scratch/ee2178/Denoising-Diffusion-Project/my_CDLNet

# Sweep modes
MODES=(1 2 2.5 4)
IMMAP_MODE="${MODES[$SLURM_ARRAY_TASK_ID]}"

# Reasonable save name (one file per mode)
SAVE_NAME="eval_results/accel_10/immap_mode_${IMMAP_MODE}.txt"

python3 eval_diff.py configs/eval_config.json \
  --kspace_path=../../datasets/fastmri/brain/multicoil_val \
  --smap_path=../../datasets/fastmri_preprocessed/brain_T2W_coil_combined/val/ \
  --noise_level=0.0 \
  --save_name="${SAVE_NAME}" \
  --immap_mode="${IMMAP_MODE}" \
  --accel=10
