#!/bin/bash
#SBATCH --array 1-30%30
#SBATCH --job-name=hpo_optuna_grid_pi
#SBATCH --time 20:00:00
#SBATCH --account=def-blaschow
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --output=optuna_grid_output_%A_%a.txt
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --mail-type=BEGIN,END,FAIL,ALL
#SBATCH --mail-user=samuel.kostousov@gmail.com

module load postgresql/16.0
module load StdEnv/2023  gcc/12.3  cuda/12.2
module load mpi4py/3.1.4
module load opencv/4.11.0

source $HOME/CompNeuro/Computational_Neuroscience_-25--26/.venv/bin/activate
export PYTHONPATH=$PWD
export PYTHONPATH="/home/kostouso/CompNeuro/Computational_Neuroscience_-25--26":$PYTHONPATH



OTPUNA_STUDY_NAME=${1-"grid_search_alpha_bands"}
more_args=${2-""}

OPTUNA_DB=$SCRATCH/${OTPUNA_STUDY_NAME}.db

python -u "/home/kostouso/CompNeuro/Computational_Neuroscience_-25--26/scripts/hpo_train_grid.py" --optuna_db $OPTUNA_DB --optuna_study_name $OTPUNA_STUDY_NAME --imagenet_root /home/kostouso/CompNeuro/Computational_Neuroscience_-25--26/split_data --epochs 200 --batch_size 512 --save_dir "/scratch/kostouso/CompNeuro/Computational_Neuroscience_-25--26/Mar_31_launch/sk_logs_spec_loss_optuna_grid" --tau 0.2 --lr 0.1 --eval_every 0 --spectral_loss_warmup_epochs 10 --wd 1e-6 --workers 16 --lp_epochs 5 --lp_lr 0.1 --amp --seed 0  --neural_data_dir "/home/kostouso/CompNeuro/Computational_Neuroscience_-25--26/src/latest_neural_data/majajhong_cache" $more_args
