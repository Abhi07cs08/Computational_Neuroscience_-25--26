#!/bin/bash
#SBATCH --job-name=simclr_basic
#SBATCH --output=logs/simclr_basic_%j.out
#SBATCH --error=logs/simclr_basic_%j.err
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1

