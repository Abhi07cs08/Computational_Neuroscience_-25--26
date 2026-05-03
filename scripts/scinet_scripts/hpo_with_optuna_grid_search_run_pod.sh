source /workspace/Computational_Neuroscience_-25--26/.venv/bin/activate
export PYTHONPATH=$PWD
export PYTHONPATH="/workspace/Computational_Neuroscience_-25--26":$PYTHONPATH



OTPUNA_STUDY_NAME=${1-"grid_search_alpha_bands_retry"}
workers=${2-16}
more_args=${3-"--skip_ssl_val"}

OPTUNA_DB=$SCRATCH/${OTPUNA_STUDY_NAME}.db

python -u "/workspace/Computational_Neuroscience_-25--26/scripts/hpo_train_grid.py" --optuna_db $OPTUNA_DB --optuna_study_name $OTPUNA_STUDY_NAME --imagenet_root "/workspace/Computational_Neuroscience_-25--26/imagenet" --epochs 200 --batch_size 512 --save_dir "/workspace/Computational_Neuroscience_-25--26/optuna_results/optuna_grid_30042026" --tau 0.2 --lr 0.1 --eval_every 0 --spectral_loss_warmup_epochs 10 --wd 1e-6 --workers $workers --lp_epochs 5 --lp_lr 0.1 --amp --seed 0  --neural_data_dir "/workspace/Computational_Neuroscience_-25--26/src/latest_neural_data/majajhong_cache" $more_args
