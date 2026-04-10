#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time 24:00:00
#SBATCH --output=ev_calculator_%j.txt
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

args=${1-""}

python -u "/home/kostouso/CompNeuro/Computational_Neuroscience_-25--26/src/utils/calculate_all_ev_3.py" $args