#!/bin/bash
#SBATCH --nodes=1
#SBATCH --account=torch_pr_89_tandon_advanced
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --constraint=a100
#SBATCH --cpus-per-task=16
#SBATCH --mem=64GB
#SBATCH --time=48:00:00
#SBATCH --job-name=LPDSNet-Knee
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=ee2178@nyu.edu
#SBATCH --output=logs/slurm_%A_%a.out
#SBATCH --error=logs/slurm_%A_%a.err
#SBATCH --array=0-1

# Activate environment
source ~/.bashrc
conda activate env

cd ~/scratch/ee2178/Denoising-Diffusion-Project/my_CDLNet

BASE_CONFIG=configs/knee/immap2p5_R6_config.json
TMP_CONFIG=/tmp/config_${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}.json

# Create modified config using Python
python3 - <<EOF
import json

with open("$BASE_CONFIG", "r") as f:
    cfg = json.load(f)

if $SLURM_ARRAY_TASK_ID == 0:
    print("Running WITHOUT blur")
    cfg["train"]["fit"]["blur"] = False
    cfg["paths"]["save"] = "trained_nets/knee/LPDS_noBlur_R6"
else:
    print("Running WITH blur")
    cfg["train"]["fit"]["blur"] = True
    cfg["paths"]["save"] = "trained_nets/knee/LPDS_Blur_R6"

with open("$TMP_CONFIG", "w") as f:
    json.dump(cfg, f, indent=4)
EOF

# Run training
python3 train_mri_recon.py $TMP_CONFIG
