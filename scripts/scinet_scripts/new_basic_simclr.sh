#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time 02:00:00
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


# Run the Python script
python -u "/home/kostouso/CompNeuro/Computational_Neuroscience_-25--26/scripts/train_simclr.py" --imagenet_root /home/kostouso/CompNeuro/Computational_Neuroscience_-25--26/split_data --batch_size 32 --epochs 1 --save_dir /home/kostouso/CompNeuro/Computational_Neuroscience_-25--26/new_logs --skip_knn_metric True --skip_spectrum_metric True --spectral_loss_coeff 0.1 --neural_ev False --limit_train 32 --limit_val 32