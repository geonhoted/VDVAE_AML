# AML Team Project: Very Deep VAE

We reproduce and extend Very Deep Variational Autoencoder. Under image is reproduced reference paper's result with 75-layer hierarchical architecture trained in epoch 300 on CIFAR-10 dataset.  
<p align="center">
  <img src="https://github.com/geonhoted/VDVAE_AML/blob/main/result_of_baseline_epoch300.png?raw=true" alt="VDVAE epoch300 result" />
</p>  
Here is our reference. "Very Deep VAEs Generalize Autoregressive Models and Can Outperform Them on Images" (https://arxiv.org/abs/2011.10650)  

**Our model variants:**
- Baseline: Vanilla VDVAE reimplemented based on paper
- KL-Annealing: Custom schedule for gradually increasing KL weight
- Attention: Self-attention added to the encoder block

## 1. How to use Baseline Model

### Setup

1) Generate elice environment and create folder named 'content'.  
2) Download the `vdvae_baseline.ipynb` and upload it under 'content' directory.
3) Download the `enviornment.yml` and upload this file (not in the content folder).
```
## Run this command on bash

conda env create -f environment.yml
conda activate vdvae
```

### Training Baseline Model

1) Set the interpreter as `vdvae`.
2) Run the `vdvae_baseline.ipynb` code sequentially. You can train from scratch at first, and you can train further from trained model if you have checkpoint. 
    - **Training from Scratch**: You need to change the hyperparameters for model architecture and training. The hyperparameters for model architecture is managed on `model_hps.py`. And hyperparameters for training is managed on `hps.py`. (*Note*: If you define hyperparameters in `cifar10 = Hyperparams()`, those hyperparameters will not updated by under `--add_agrument` part.) When you run the `train.py`, the training will start. 
    - **Training from checkpoint**: If you successfully trained the model, you will get checkpoints in `saved_models/test/`. You can train further from this point by setting `hps.py` like under:  
    ```
    parser.add_argument('--restore_path', type=str, default='./saved_models/test/latest')
    parser.add_argument('--restore_ema_path', type=str, default='./saved_models/test/latest')
    parser.add_argument('--restore_log_path', type=str, default='./saved_models/test/latest-log.jsonl')
    parser.add_argument('--restore_optimizer_path', type=str, default='./saved_models/test/latest-opt.th')
    ```
3) Check the result images consists of reconstructed images and sampled images.
4) Check the `log.txt` file to check your training process is stable or not.

## 2. How to use KL-Annealing added Model

### Setup

1) Download the `vdvae_kl_annealing.ipynb` and upload it under 'content' directory.
2) Other things are same as Baseline.

### Training KL-Annealing added Model

The only difference KL-Annealing added Model is you can set **KL-Annealing Schedule**.  
1) Set kl-annealing schedule in `hps.py`. (*Note*: 10000 means that KL term affects completely when training iteration is same as 10,000.)
    ```
    ## Change the kl_anneal_iters value with your preference.

    parser.add_argument('--kl_anneal_iters', type=int, default=10000)
    ```
2) Other things are same in Baseline. You can train from scratch or checkpoints.
3) Check the result images consists of reconstructed images and sampled images.
4) Check the `log.txt` file to check your training process is stable or not.

## 3. How to use Attention added Model

### Setup

Download the vdvae_self_attention.ipynb and upload it under the 'content' directory.

Other dependencies and setup steps (environment, directory structure, interpreter setting) are the same as the Baseline.

### Training Attention added Model
This version adds self-attention to the encoder blocks to enhance the model's representational capacity, especially in capturing global dependencies.

1. The architecture changes are already reflected in the source files, including block.py and encoder.py. Make sure use_attn=True and use_gated_residual=True are properly set when instantiating the encoder blocks.

2. You can train the model:

- **From scratch**: Start fresh using your own configuration for hyperparameters and training steps.

- **From checkpoint**: Same as the Baseline, you can resume training from saved weights by modifying the checkpoint paths in hps.py.

3. Check the output:

    Reconstructed images and sampled images will be shown in the notebook after each epoch.

    Inspect the training log (log.txt) to monitor stability and performance.

4. You may compare attention-added results with the baseline to analyze improvements in sample quality or log-likelihood. 
