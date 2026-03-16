#!/bin/bash
#SBATCH --nodes=1
#SBATCH --account=torch_pr_89_tandon_advanced
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --constraint=a100
#SBATCH --cpus-per-task=16
#SBATCH --mem=64GB
#SBATCH --time=48:00:00
#SBATCH --job-name=LPDSNet-Knee
#SBATCH --array=0-4
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=ee2178@nyu.edu
#SBATCH --output=logs/slurm_%A_%a.out
#SBATCH --error=logs/slurm_%A_%a.err

# Activate Conda
source ~/.bashrc
conda activate env

# Move to project
cd ~/scratch/ee2178/Denoising-Diffusion-Project/my_CDLNet

# -------------------------------
# Config lists
# -------------------------------

configs=(
"configs/knee/config.json"
"configs/knee/immap2p5_R10_config.json"
"configs/knee/immap2p5_R6_config.json"
"configs/knee/mri_R10_config.json"
"configs/knee/mri_R6_config.json"
)

config=${configs[$SLURM_ARRAY_TASK_ID]}

echo "Running job $SLURM_ARRAY_TASK_ID with config $config"

# -------------------------------
# Run correct training script
# -------------------------------

if [ "$SLURM_ARRAY_TASK_ID" -eq 0 ]; then
    python3 train.py $config
else
    python3 train_mri_recon.py $config
fi
