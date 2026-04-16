#!/usr/bin/env bash

# Remote settings
REMOTE_HOST="kostouso@trillium-gpu.scinet.utoronto.ca"
# REMOTE_PATH="/scratch/kostouso/CompNeuro/Computational_Neuroscience_-25--26/"
REMOTE_PATH="/scratch/kostouso/CompNeuro/Computational_Neuroscience_-25--26/Mar_31_launch/"
# Local destination (WSL -> Windows D:)
LOCAL_PATH="/mnt/d/CompNeuroTrials/"

# SSH settings
SSH_OPTS="-F /ssh-keys/config -i /ssh-keys/id_ed25519"

echo "Ensuring local directory exists..."
mkdir -p "$LOCAL_PATH"

echo "Starting rsync transfer..."
rsync -avhPz \
  --exclude='*e[0-9]*.pt' \
  -e "ssh $SSH_OPTS" \
  "$REMOTE_HOST:$REMOTE_PATH" \
  --exclude='*e[0-9]*.pt' \
  --include='*last.pt' \
  --exclude='*.pt' \
  "$LOCAL_PATH"

echo "Done."