# UAFT
PyTorch implementation of unsupervised adversarial fine-tuning on Conv-TasNet

## Dependencies
1. Python and packages   

   This code was tested on Python 3.6 with PyTorch 1.2.0. Other packages can be installed by:
   
   `pip install -r requirements.txt`
   
## Dataset
1. source domain datasets:VCTK
2. target domain darasets:ST-CMDS,THCHS-30
   
## Pre-training
   We use [PyTorch implementation of Conv-TasNet](https://github.com/JusperLee/Dual-Path-RNN-Pytorch) to get well-trained model on VCTK datasets.
   And we provide our checkpoint.
   
## adversarial fine-tuning
   After generator mixture speech of source domain and target domain for training, and create the scp file of them.
   You can edit `train.yml` with corresponding file path and run the following command to start trainging.
   `python train_Tasnet.py --opt config/Conv_Tasnet/train.yml`
