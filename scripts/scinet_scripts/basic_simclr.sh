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
epochs="${2-100}"
batch_size="${3-32}"
neural_ev="${4-False}"
log_folder_name ="${5-basic_simclr_logs}"
if [ "$neural_ev" == "True" ]; then
    neural_ev_arg="--neural_ev True"
else
    neural_ev_arg=""
fi

echo "Arguments: spectral_loss_coeff=$spectral_loss_coeff epochs=$epochs batch_size=$batch_size neural_ev=$neural_ev"

# Run the Python script
python -u "/home/kostouso/CompNeuro/Computational_Neuroscience_-25--26/scripts/train_simclr.py" --imagenet_root /home/kostouso/CompNeuro/Computational_Neuroscience_-25--26/split_data --batch_size $batch_size --epochs $epochs --save_dir /scratch/kostouso/CompNeuro/Computational_Neuroscience_-25--26/$log_folder_name --skip_knn_metric True --skip_spectrum_metric True --spectral_loss_coeff $spectral_loss_coeff $neural_ev_arg