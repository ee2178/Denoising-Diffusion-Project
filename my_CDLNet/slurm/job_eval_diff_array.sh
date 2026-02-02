#!/bin/bash
#SBATCH --nodes=1
#SBATCH --account=pr_198_general
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=1
#SBATCH --mem=32GB
#SBATCH --time=4:00:00

#SBATCH --array=0-3
#SBATCH --job-name=ImMAP_Eval
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=ee2178@nyu.edu
#SBATCH --output=logs/slurm_%x_%A_%a.out
#SBATCH --error=logs/slurm_%x_%A_%a.err

module load cuda/11.6.2

# Make sure conda is available
source ~/.bashrc
conda activate evrt-detr

# -----------------------------
# Map array index → immap_mode
# -----------------------------
IMMAP_MODES=(1 2 2.5 3)
IMMAP_MODE=${IMMAP_MODES[$SLURM_ARRAY_TASK_ID]}

# Make filename-safe version (2.5 → 2p5)
IMMAP_TAG=$(echo "${IMMAP_MODE}" | sed 's/\./p/')

# -----------------------------
# Paths
# -----------------------------
cd ~/vast/ee2178/Denoising-Diffusion-Project/my_CDLNet || exit 1

SAVE_NAME="eval_results/immap_${IMMAP_TAG}_mask.txt"

echo "Running ImMAP mode = ${IMMAP_MODE}"
echo "Saving to ${SAVE_NAME}"

# -----------------------------
# Run evaluation
# -----------------------------
python3 eval_diff.py eval_config.json \
  --kspace_path=../../datasets/fastmri/brain/multicoil_val \
  --smap_path=../../datasets/fastmri_preprocessed/brain_T2W_coil_combined/val/ \
  --noise_level=0.05 \
  --save_name="${SAVE_NAME}" \
  --eval_e2e=False \
  --immap_mode="${IMMAP_MODE}"

