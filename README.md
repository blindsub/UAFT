# UAFT
PyTorch implementation of unsupervised adversarial fine-tuning on Conv-TasNet

## Dependencies
1. Python and packages   

   This code was tested on Python 3.6 with PyTorch 1.2.0. Other packages can be installed by:
   
   `pip install -r requirements.txt`
   
## Dataset
1. Source domain datasets: [VCTK](https://datashare.is.ed.ac.uk/handle/10283/3443)
2. Target domain darasets: [ST-CMDS](https://www.openslr.org/38/),[THCHS-30](https://www.openslr.org/18/)
   
## Pre-training
   We use [PyTorch implementation of Conv-TasNet](https://github.com/JusperLee/Dual-Path-RNN-Pytorch) to get well-trained model on VCTK datasets.
   And we provide our checkpoint.
   
## adversarial fine-tuning
   Generate mixture speech of source domain and target domain for training, and create the scp file of them.
   Then you can edit `train.yml` with corresponding file path and run the following command to start training.
   
   `python train_Tasnet.py --opt config/Conv_Tasnet/train.yml`
