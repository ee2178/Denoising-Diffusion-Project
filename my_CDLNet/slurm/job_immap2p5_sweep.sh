#!/bin/bash
#SBATCH --nodes=1
#SBATCH --account=torch_pr_89_tandon_advanced
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --constraint=a100
#SBATCH --cpus-per-task=16
#SBATCH --mem=64GB
#SBATCH --time=48:00:00
#SBATCH --job-name=LPDS-Knee-SSDU-Sweep-R6
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=ee2178@nyu.edu
#SBATCH --output=logs/slurm_%A_%a.out
#SBATCH --error=logs/slurm_%A_%a.err
#SBATCH --array=0-7

# -----------------------
# Environment
# -----------------------
source ~/.bashrc
conda activate env

cd ~/scratch/ee2178/Denoising-Diffusion-Project/my_CDLNet

# -----------------------
# Base settings
# -----------------------
ACCEL=6
BASE_CONFIG=configs/knee/immap2p5_R${ACCEL}_config.json
TMP_CONFIG=/tmp/config_${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}.json

# -----------------------
# Build config
# -----------------------
python3 - <<EOF
import json

with open("$BASE_CONFIG", "r") as f:
    cfg = json.load(f)

task_id = $SLURM_ARRAY_TASK_ID
accel = "$ACCEL"

# -----------------------
# Sweep values
# -----------------------
ssdu_base_accel_vals = [1, 2]
sigma_scaling_vals = [False, True]
noise_dist_vals = ["log", "uniform"]

# Decode task_id into sweep indices
i_ssdu = task_id // 4
i_sigma = (task_id % 4) // 2
i_noise = task_id % 2

ssdu_base_accel = ssdu_base_accel_vals[i_ssdu]
sigma_scaling = sigma_scaling_vals[i_sigma]
noise_dist = noise_dist_vals[i_noise]

# -----------------------
# Apply config updates
# -----------------------
cfg["train"]["fit"]["ssdu_base_accel"] = ssdu_base_accel
cfg["train"]["fit"]["sigma_scaling"] = sigma_scaling
cfg["train"]["fit"]["noise_dist"] = noise_dist

# -----------------------
# Save directory name
# -----------------------
sigma_str = "sigmaScaleON" if sigma_scaling else "sigmaScaleOFF"
noise_str = noise_dist.upper()

save_name = (
    f"LPDS_ImMAP2p5_SSDU_R{accel}"
    f"_baseAccel{ssdu_base_accel}x"
    f"_{sigma_str}"
    f"_noise{noise_str}"
)

cfg["paths"]["save"] = f"trained_nets/knee/{save_name}"

print("=" * 60)
print(f"Task ID            : {task_id}")
print(f"Acceleration       : R={accel}")
print(f"SSDU base accel    : {ssdu_base_accel}")
print(f"Sigma scaling      : {sigma_scaling}")
print(f"Noise distribution : {noise_dist}")
print(f"Save path          : {cfg['paths']['save']}")
print("=" * 60)

with open("$TMP_CONFIG", "w") as f:
    json.dump(cfg, f, indent=4)
EOF

# -----------------------
# Run training
# -----------------------
python3 train_mri_recon.py "$TMP_CONFIG"
