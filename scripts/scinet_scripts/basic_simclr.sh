#!/bin/bash
#SBATCH --job-name=simclr_basic
#SBATCH --output=/scratch/CompNeuro/job_logs/%x_%j.out
#SBATCH --error=/scratch/CompNeuro/job_logs/%x_%j.err
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8

JOBID=$SLURM_JOB_ID

export PYTHONPATH=$PWD

source ~/CompNeuro/Computational_Neuroscience_-25--26/.venv/bin/activate
export PYTHONPATH=$PWD


module load postgresql/16.0
module load StdEnv/2023  gcc/12.3  cuda/12.2
module load mpi4py/3.1.4
module load opencv/4.11.0

python -u "CompNeuro/Computational_Neuroscience_-25--26/scripts/train_simclr.py" --imagenet_root CompNeuro/Computational_Neuroscience_-25--26/split_data --batch_size 32 --epochs 1 --save_dir new_logs --skip_knn_metric True --skip_spectrum_metric True --spectral_loss_coeff 0.1 --neural_ev True --limit_train 32 --limit_val 32

echo "train_simclr.py launched"