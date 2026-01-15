#!/bin/bash
#SBATCH --job-name=lejepa_train
#SBATCH --output=logs/%x_%j.log
#SBATCH --error=logs/%x_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=40G
#SBATCH --time=24:00:00
#SBATCH --partition=compsci-gpu
#SBATCH --gres=gpu:rtx_pro_6000:1


# Fail fast
set -e

# Activate environment
source /home/users/aho13/jepa_tests/venv/bin/activate

# Optional: debugging
echo "Running on $(hostname)"
echo "Running on partition: $SLURM_JOB_PARTITION"
echo "Job ID: $SLURM_JOB_ID"
echo "Node list: $SLURM_NODELIST"
echo "GPUs: $SLURM_GPUS"

nvidia-smi
which python
python --version

# Run training
export HYDRA_FULL_ERROR=1

./run_inf.sh python eval/views_resnet.py \
  +lamb=0.05 \
  +V=4 \
  +proj_dim=2048 \
  +lr=0.05 \
  +bs=256 \
  +epochs=100 \ 
  +num_workers=8 \
  +device="cuda" 
  