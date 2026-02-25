#!/bin/bash
#SBATCH --job-name=train
#SBATCH --output=logs/%x_%j.log
#SBATCH --error=logs/%x_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=45G
#SBATCH --time=48:00:00
#SBATCH --partition=compsci-gpu
#SBATCH --gres=gpu:rtx_pro_6000:1


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
srun python src/run_training_loop.py \
  +lamb=0.05\
  +V_global=2 \
  +V_local=6 \
  +V_mixed=2 \
  +model_name=vit_large_patch16_dinov3.sat493m \
  +global_img_size=224 \
  +local_img_size=96 \
  +proj_dim=512 \
  +lr=5e-4 \
  +bs=512 \
  +grad_accum=1 \
  +epochs=100\
  +num_workers=8\
  +device=cuda \
  +prefetch_factor=2 \
  +dataset=imagenet-1k \
  +distributed=False \
  +seed=0 \
  +world_size=1 \
  +log_interval=40 \
  +reg=hybrid \
  # +invariance_weight=25.0 \
  # +rdm_reg_weight=125.0 \
  # +lp_norm_parameter=2.0 \
  # +mean_shift_value=0.0 \
  # +target_distribution=rectified_lp_distribution \

