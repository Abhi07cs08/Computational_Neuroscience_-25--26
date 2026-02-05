#!/bin/bash
#SBATCH --array 1-4%2
#SBATCH --job-name=hpo_optuna_simclr
#SBATCH --time 01:00:00

#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --output=optuna_test_simclr_output_%A_%a.txt
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


OTPUNA_STUDY_NAME="simclr_spec_reg_hpo_study"

OPTUNA_DB=$SCRATCH/${OTPUNA_STUDY_NAME}.db

python -u "/home/kostouso/CompNeuro/Computational_Neuroscience_-25--26/scripts/hpo_train.py" --optuna_db $OPTUNA_DB --optuna_study_name $OTPUNA_STUDY_NAME --imagenet_root /home/kostouso/CompNeuro/Computational_Neuroscience_-25--26/split_data --epochs 1 --batch_size 512 --save_dir "/scratch/kostouso/CompNeuro/Computational_Neuroscience_-25--26/Feb_2_launch/sk_logs_spec_loss_optuna_test" --tau 0.2 --lr 0.1 --wd 1e-6 --workers 16 --eval_every 30 --lp_epochs 5 --lp_lr 0.1 --amp --seed 0 --neural_data_dir "/home/kostouso/CompNeuro/Computational_Neuroscience_-25--26/src/REVERSE_PRED_FINAL/majajhong_cache"
