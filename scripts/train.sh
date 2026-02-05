#!/bin/bash
#SBATCH --job-name=train
#SBATCH --output=logs/%x_%j.log
#SBATCH --error=logs/%x_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --partition=compsci-gpu
#SBATCH --gres=gpu:a5000:1


# Fail fast
set -e

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

# Run training
export HYDRA_FULL_ERROR=1

# # SimCLR training configuration:
srun python eval/run_training_loop.py \
  +lamb=0.05 \
  +V_global=2 \
  +V_local=2 \
  +V_mixed=2\
  +model_name=vit_base_patch16_224.dino \
  +global_img_size=224 \
  +local_img_size=96 \
  +proj_dim=256 \
  +lr=5e-4 \
  +bs=256 \
  +grad_accum=1 \
  +epochs=100\
  +num_workers=8\
  +device=cuda \
  +prefetch_factor=4 \
  +dataset=inet100 \
  +reg=SimCLR \
  +distributed=False \
  +seed=0 \
  +world_size=1 \
  +log_interval=20 \
  # +cl_cheating=False \

