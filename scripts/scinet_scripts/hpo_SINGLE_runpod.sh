source /workspace/Computational_Neuroscience_-25--26/.venv/bin/activate
export PYTHONPATH=$PWD
export PYTHONPATH="/workspace/Computational_Neuroscience_-25--26":$PYTHONPATH


target_alpha="${1-1.0}"
spectral_loss_coeff="${2-2.0}"
echo "Target Alpha: $target_alpha"
echo "Spectral Loss Coefficient: $spectral_loss_coeff"
more_args=${3-"--parallel --skip_ssl_val"}
spec_loss_warmup_epochs=${4-10}

OPTUNA_DB=$SCRATCH/${OTPUNA_STUDY_NAME}.db

python -u "/workspace/Computational_Neuroscience_-25--26/scripts/train_simclr.py" --imagenet_root /workspace/Computational_Neuroscience_-25--26/imagenet --epochs 200 --batch_size 512 --save_dir "/workspace/Computational_Neuroscience_-25--26/optuna_results/optuna_grid_01052026" --tau 0.2 --lr 0.1 --eval_every 0 --spectral_loss_warmup_epochs $spec_loss_warmup_epochs --target_alpha $target_alpha --spectral_loss_coeff $spectral_loss_coeff --wd 1e-6 --workers 16 --lp_epochs 5 --lp_lr 0.1 --amp --seed 0  --neural_data_dir "/workspace/Computational_Neuroscience_-25--26/src/latest_neural_data/majajhong_cache" $more_args