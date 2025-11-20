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
4. ~~Setup working envo with repo and transfer work~~

#### Log
##### 18/11/2025
- created functions to compute F, R and BPI EV of model for a given brainscore benchmark
- useful commands
    - run ```source .venv/bin/activate``` to activate virtual env
    - run ```export PYTHONPATH="$PWD"```to add src dir to discoverable packages

##### 19/11/2025
- reinstalling venv with python 3.11 and the installing brainscore repo in root directory (by cloning, not using package). doesnt work otherwise
- made brainscore and my notebook compatible with our repo