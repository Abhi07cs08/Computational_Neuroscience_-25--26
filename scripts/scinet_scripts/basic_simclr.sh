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
log_folder_name="${5-basic_simclr_logs}"
neural_data_dir="${6-/home/kostouso/CompNeuro/Computational_Neuroscience_-25--26/src/metrics/neural_data}"
skip_knn_metric="${7-False}"
skip_spectrum_metric="${8-False}"
skip_alpha_metric="${9-True}"
if [ "$neural_ev" == "True" ]; then
    neural_ev_arg="--neural_ev True --neural_data_dir $neural_data_dir"
else
    neural_ev_arg=""
fi

if [ "$skip_knn_metric" == "True" ]; then
    skip_knn_arg="--skip_knn_metric True"
else
    skip_knn_arg=""
fi

if [ "$skip_spectrum_metric" == "True" ]; then
    skip_spectrum_arg="--skip_spectrum_metric True"
else
    skip_spectrum_arg=""
fi

if [ "$skip_alpha_metric" == "True" ]; then
    skip_alpha_arg="--skip_alpha_metric True"
else
    skip_alpha_arg=""
fi

echo "Arguments: spectral_loss_coeff=$spectral_loss_coeff epochs=$epochs batch_size=$batch_size neural_ev=$neural_ev neural_data_dir=$neural_data_dir log_folder_name=$log_folder_name"

# Run the Python script
python -u "/home/kostouso/CompNeuro/Computational_Neuroscience_-25--26/scripts/train_simclr.py" --imagenet_root /home/kostouso/CompNeuro/Computational_Neuroscience_-25--26/split_data --batch_size $batch_size --epochs $epochs --save_dir "/scratch/kostouso/CompNeuro/Computational_Neuroscience_-25--26/$log_folder_name" $skip_knn_arg $skip_spectrum_arg $skip_alpha_arg --spectral_loss_coeff $spectral_loss_coeff $neural_ev_arg