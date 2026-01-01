#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time 24:00:00
#SBATCH --output=basic_simclr_output_%j.txt
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --mail-type=BEGIN,END,FAIL,ALL
#SBATCH --mail-user=samuel.kostousov@gmail.com

# Load required modules
module load postgresql/16.0
module load StdEnv/2023  gcc/12.3  cuda/12.2
module load mpi4py/3.1.4
module load opencv/4.11.0

# Activate virtual environment if needed
source $HOME/CompNeuro/Computational_Neuroscience_-25--26/.venv/bin/activate
export PYTHONPATH=$PWD
export PYTHONPATH="/home/kostouso/CompNeuro/Computational_Neuroscience_-25--26":$PYTHONPATH

spectral_loss_coeff="${1-0.0}"
batch_size="${2-32}"
epochs="${3-200}"

# Run the Python script
python -u "/home/kostouso/CompNeuro/Computational_Neuroscience_-25--26/scripts/train_simclr.py" --imagenet_root /home/kostouso/CompNeuro/Computational_Neuroscience_-25--26/split_data --batch_size $batch_size --epochs $epochs --save_dir "/scratch/kostouso/CompNeuro/Computational_Neuroscience_-25--26/sk_logs_spec_loss_$spectral_loss_coeff" --tau 0.2 --lr 0.3 --wd 1e-6 --workers 16 --warmup_epochs 10 --eval_every 5 --lp_epochs 5 --lp_lr 0.1 --amp --seed 0 --neural_data_dir "/home/kostouso/CompNeuro/Computational_Neuroscience_-25--26/src/metrics/neural_data" --spectral_loss_coeff $spectral_loss_coeff