#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --gpus-per-node=1
#SBATCH -p compute
#SBATCH -A def-blaschow
#SBATCH --time=24:00:00
#SBATCH --output=/scratch/abhinn/slurm/simclr-%j.out
#SBATCH --exclude=trig0062

set -euo pipefail

module --force purge
module load StdEnv/2023 gcc/12.3 cuda/12.2
module load opencv/4.11.0 2>/dev/null || true

# avoid matplotlib/font cache permission spam
export MPLCONFIGDIR=/scratch/abhinn/.mplconfig
export XDG_CACHE_HOME=/scratch/abhinn/.cache
mkdir -p "$MPLCONFIGDIR" "$XDG_CACHE_HOME"

cd /scratch/abhinn/projects/Computational_Neuroscience_-25--26
source /scratch/abhinn/venvs/simclr/bin/activate
export PYTHONPATH=$PWD

#hyperparams
spectral_loss_coeff="${1-0.005}"
batch_size="${2-512}"
more_epochs="${3-50}"   # how many epochs to add each run when resuming
lr="${4-0.01}"
grad_clip="${5-1.0}"
warmup_epochs="${6-10}"
tau="${7-0.4}"
eval_every="${8-10}"
lp_epochs="${9-50}"
save_dir="/scratch/abhinn/runs/simclr_debiased_tau0.4_lr0.01_bs512/start_20260216-162406"

ckpt="/scratch/abhinn/runs/simclr_debiased_tau0.4_lr0.01_bs512/start_20260216-162406/ckpts/simclr/last.pt"
echo "save_dir=$save_dir"
echo "bs=$batch_size tau=$tau lr=$lr warmup=$warmup_epochs grad_clip=$grad_clip eval_every=$eval_every lp_epochs=$lp_epochs"
echo "spectral_loss_coeff=$spectral_loss_coeff"

COMMON_ARGS=(
  --imagenet_root /scratch/abhinn/datasets/imagenet_small_80_20
  --batch_size "$batch_size"
  --tau "$tau"
  --spectral_loss_warmup_epochs 30
  --lr "$lr"
  --wd 1e-6
  --workers 8
  --warmup_epochs "$warmup_epochs"
  --grad_clip "$grad_clip"
  --eval_every "$eval_every"
  --lp_epochs "$lp_epochs"
  --lp_lr 0.3
  --amp
  --seed $((1000 + SLURM_ARRAY_TASK_ID)) \
  --neural_data_dir /scratch/abhinn/projects/Computational_Neuroscience_-25--26/src/REVERSE_PRED_FINAL/majajhong_cache