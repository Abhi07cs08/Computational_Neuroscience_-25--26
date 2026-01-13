# Computational Neuroscience Research
## Improving Bidirectional Predicitivity via Spectrum Modulation

**BOLD** denotes tasks to be prioritized

### 30/10/2025-13/11/2025
#### TODO
1. ~~calculate alpha value function~~
2. ~~calculate alpha of neural data~~
3. ~~create diff spectral loss~~
4. ~~train basic model with spectral loss~~

#### Log
- encountering problems with brainscore
- trained basic CNN on fashionmnist
- confirmed alpha-value of neural data is around 1
- created differentiable spectral loss
- confirmed spectral losses's effectivity in bringing spectrum to 1, although the differnce is not major, presumably baecause the model's natural spectrum is already around higher than 1 (for such a simple model). However, spectrum at val/train divergence is significantly lower than when spectrum is not modulated.
- cifar100 is slow but will possibly try

![alt text](figures/neural_data_spectrum_13112025.png)
![alt text](figures/fashionmnist_spectralloss_13112025.png)

### 13/11/2025-27/11/2025
#### TODO 
1. ~~implement F&R EV~~ (18/11/2025)
2. integrate spectral loss with SimCLR
3. **F&R EV for SimCLR**
4. ~~Setup working envo with repo and transfer work~~ (19/11/2025)

#### Log
##### 18/11/2025
- created functions to compute F, R and BPI EV of model for a given brainscore benchmark
- useful commands
    - run ```source .venv/bin/activate``` to activate virtual env
    - run ```export PYTHONPATH="$PWD"```to add src dir to discoverable packages

##### 19/11/2025
- reinstalling venv with python 3.11 and the installing brainscore repo in root directory (by cloning, not using package). doesnt work otherwise
- made brainscore and my notebook compatible with our repo#####

##### 20/11/2025
- FR EV & R2 score negative. it used to be that accuracy on initial trained data neared 100 from the ridge regression. now its at around 23%, and for unseen data after cross val it is negative. cannot make it positive. normalizing doesnt seem to help. 23% agnostic to alpha parameter. 5 fold stratified split. averaged model activations of 4 dim to get 2 dim and make it smaller



ssh -F /ssh-keys/config -i /ssh-keys/id_ed25519 -Y kostouso@trillium-gpu.scinet.utoronto.ca

module load postgresql/16.0
module load StdEnv/2023  gcc/12.3  cuda/12.2
module load mpi4py/3.1.4
module load opencv/4.11.0

sbatch ~/CompNeuro/Computational_Neuroscience_-25--26/scripts/scinet_scripts/basic_simclr.sh 0.1 400 32 True "16_12_2025_neural_ev"

python scripts/train_simclr.py --imagenet_root train_val --batch_size 14 --epochs 1 --limit_train 40 --limit_val 40

sbatch ~/CompNeuro/Computational_Neuroscience_-25--26/scripts/scinet_scripts/basic_simclr.sh 0.1 1 32 True "short_tests"

scontrol update JobID=172457 TimeLimit=01:00:00

python scripts/train_simclr.py --imagenet_root train_val --epochs 1 --batch_size 12 --eval_every 1 --lp_epochs 1 --amp --seed 0 --save_dir runs/simclr


sbatch ~/CompNeuro/Computational_Neuroscience_-25--26/scripts/scinet_scripts/ak_simclr_request.sh

![python -u "/home/kostouso/CompNeuro/Computational_Neuroscience_-25--26/scripts/train_simclr.py" --imagenet_root /home/kostouso/CompNeuro/Computational_Neuroscience_-25--26/split_data --batch_size 32 --epochs 200 --save_dir "/scratch/kostouso/CompNeuro/Computational_Neuroscience_-25--26/ak_logs" --tau 0.2 --lr 0.3 --wd 1e-6 --workers 16 --warmup_epochs 10 --eval_every 5 --lp_epochs 5 --lp_lr 0.1 --amp --seed 0 --neural_data_dir "/home/kostouso/CompNeuro/Computational_Neuroscience_-25--26/src/metrics/neural_data" --spectral_loss_coeff 0.0:](figures/simclr_run30122025.png)


source scripts/fetch_data_from_scinet.sh /scratch/kostouso/CompNeuro/Computational_Neuroscience_-25--26/all_model_logs/* figures/model_logs --exclude-ext .pt

 python scripts/train_simclr.py ckpt --imagenet_root
 train_val --ckpt_path figures/model_logs/sk_logs_spec_loss_0.5/start_20260101-002021 --more_epochs 1s

 start_20260101-215812

 kostouso@trig-login01:/scratch/kostouso$ sbatch ~/CompNeuro/Computational_Neuroscience_-25--26/scripts/scinet_scripts/continue_training.sh /scratch/kostouso/CompNeuro/Computational_Neuroscience_-25--26/all_model_logs/sk_logs_spec_loss_0.0/start_20260101-215812/ckpts/simclr/last.pt 1

 sbatch ~/CompNeuro/Computational_Neuroscience_-25--26/scripts/scinet_scripts/continue_training.sh /scratch/kostouso/CompNeuro/Computational_Neuroscience_-25--26/all_model_logs/sk_logs_spec_loss_0.0/start_20260101-215812/ckpts/simclr/last.pt 300

 sbatch ~/CompNeuro/Computational_Neuroscience_-25--26/scripts/scinet_scripts/continue_training.sh /scratch/kostouso/CompNeuro/Computational_Neuroscience_-25--26/all_model_logs/sk_logs_spec_loss_0.0/start_20260101-215812/ckpts/simclr/last.pt 300

 source scripts/fetch_data_from_scinet.sh /scratch/kostouso/CompNeuro/Computational_Neuroscience_-25--26/all_model_logs/sk_logs_spec_loss_0.0/start_20260101-215812/ckpts/simclr/* figures/model_logs/sk_logs_spec_loss_0.0/start_20260101-215812/ckpts/simclr/