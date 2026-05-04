source /workspace/Computational_Neuroscience_-25--26/.venv/bin/activate
export PYTHONPATH=$PWD
export PYTHONPATH="/workspace/Computational_Neuroscience_-25--26":$PYTHONPATH

ckpt_path="${1-/scratch/kostouso/CompNeuro/Computational_Neuroscience_-25--26/all_model_logs/sk_logs_spec_loss_0.0/start_20260101-221806/simclr/last.pt}"
more_epochs="${2-1}"

echo "Arguments: more_epochs=$more_epochs ckpt_path=$ckpt_path"

# Run the Python script
python -u "/workspace/Computational_Neuroscience_-25--26/scripts/train_simclr.py" --imagenet_root "/workspace/Computational_Neuroscience_-25--26/imagenet" ckpt --ckpt_path $ckpt_path --more_epochs $more_epochs