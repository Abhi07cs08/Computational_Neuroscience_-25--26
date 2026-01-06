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

imagenet_root="${1-/home/kostouso/CompNeuro/Computational_Neuroscience_-25--26/split_data}"
ckpt_path="${2-/scratch/kostouso/CompNeuro/Computational_Neuroscience_-25--26/all_model_logs/sk_logs_spec_loss_0.0/start_20260101-221806/simclr/last.pt}"
more_epochs="${3-1}"

echo "Arguments: more_epochs=$more_epochs ckpt_path=$ckpt_path imagenet_root=$imagenet_root"

# Run the Python script
python -u "/home/kostouso/CompNeuro/Computational_Neuroscience_-25--26/scripts/train_simclr.py" --imagenet_root $imagenet_root ckpt --ckpt_path $ckpt_path --more_epochs $more_epochs