#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=4GB
#SBATCH --time=9:00:00
#SBATCH --job-name=CDLNet-Basic
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=ee2178@nyu.edu
#SBATCH --output=slurm_%j.out
#SBATCH --error=slurm_%j.err

cd ~/vast/ee2178/Denoising_Diffusion_Project/my_CDLNet
python3 train.py config.json