#!/bin/bash
#SBATCH --job-name=ddp_train
#SBATCH --output=logs/%x_%j.log
#SBATCH --error=logs/%x_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2
#SBATCH --cpus-per-task=8
#SBATCH --mem=50G
#SBATCH --time=24:00:00
#SBATCH --partition=compsci-gpu
#SBATCH --gres=gpu:a6000:2

# Fail fast
set -e

# Threading config (split across 2 GPUs)
export OMP_NUM_THREADS=6
export MKL_NUM_THREADS=6

# Activate environment
source /home/users/aho13/jepa_tests/env/bin/activate

# Optional: debugging
echo "Running on $(hostname)"
echo "Running on partition: $SLURM_JOB_PARTITION"
echo "Job ID: $SLURM_JOB_ID"
echo "Node list: $SLURM_NODELIST"
echo "GPUs: $SLURM_GPUS"

nvidia-smi
which python
python --version

# Run training with Lightning DDP (no manual srun needed)
export HYDRA_FULL_ERROR=1

# Lightning handles DDP automatically when distributed=True and world_size>1
srun python eval/run_training_loop.py \
  +lamb=0.05 \
  +V_global=2 \
  +V_local=4 \
  +V_mixed=0 \
  +model_name=vit_base_patch16_224.dino \
  +global_img_size=224 \
  +local_img_size=96 \
  +proj_dim=256 \
  +lr=5e-4 \
  +bs=256 \
  +grad_accum=1 \
  +epochs=100\
  +num_workers=5 \
  +device=cuda \
  +prefetch_factor=4 \
  +dataset=inet100 \
  +reg=SimCLR \
  +distributed=True \
  +seed=0 \
  +world_size=2 \
  +log_interval=20 \
