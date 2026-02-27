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

spectral_loss_coeff="${1-0.02}"
batch_size="${2-512}"
epochs="${3-30}"
lr="${4-0.3}"
grad_clip="${5-0.0}"
warmup_epochs="${6-10}"
tau="${7-0.2}"
more_args="${8-}"
echo "Spectral Loss Coefficient: $spectral_loss_coeff"
echo "Batch Size: $batch_size"
echo "Epochs: $epochs"
echo "Learning Rate: $lr"
echo "Gradient Clipping: $grad_clip"
echo "Warmup Epochs: $warmup_epochs"
echo "Tau: $tau"
echo "Additional Arguments: $more_args"

# Run the Python script
python -u "/home/kostouso/CompNeuro/Computational_Neuroscience_-25--26/scripts/train_simclr.py" --imagenet_root /home/kostouso/CompNeuro/Computational_Neuroscience_-25--26/split_data --batch_size $batch_size --epochs $epochs --save_dir "/scratch/kostouso/CompNeuro/Computational_Neuroscience_-25--26/Feb_2_launch/sk_logs_spec_loss_${spectral_loss_coeff}_warmup_${warmup_epochs}_tau_${tau}" --tau $tau --lr $lr --wd 1e-6 --workers 16 --spectral_loss_warmup_epochs $warmup_epochs --grad_clip $grad_clip --eval_every 0 --lp_epochs 5 --lp_lr 0.1 --amp --seed 0 --tag warmup --neural_data_dir "/home/kostouso/CompNeuro/Computational_Neuroscience_-25--26/src/REVERSE_PRED_FINAL/majajhong_cache" --spectral_loss_coeff $spectral_loss_coeff $more_args
python -u "/home/kostouso/CompNeuro/Computational_Neuroscience_-25--26/scripts/train_simclr.py" --imagenet_root /home/kostouso/CompNeuro/Computational_Neuroscience_-25--26/split_data --batch_size $batch_size --epochs $epochs --save_dir "/scratch/kostouso/CompNeuro/Computational_Neuroscience_-25--26/Feb_2_launch/sk_logs_spec_loss_${spectral_loss_coeff}_warmup_${warmup_epochs}_tau_${tau}" --tau $tau --lr $lr --wd 1e-6 --workers 16 --spectral_loss_warmup_epochs 0 --grad_clip $grad_clip --eval_every 0 --lp_epochs 5 --lp_lr 0.1 --amp --seed 0 --tag no_warmup--neural_data_dir "/home/kostouso/CompNeuro/Computational_Neuroscience_-25--26/src/REVERSE_PRED_FINAL/majajhong_cache" --spectral_loss_coeff $spectral_loss_coeff $more_args