#!/bin/bash

#SBATCH --job-name LPIPS-TIMM
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-gpu=8
#SBATCH --mem-per-gpu=29G
#SBATCH --time 1-0
#SBATCH --partition batch_ugrad
#SBATCH -o slurm/%A-%x.out
#SBATCH -w ariel-v9

# tar -xvf /data/lo_rem7024/repos/datasets/twoafc.tar -C /local_datasets/


python train.py -b ./configs/vgg.yaml


exit 0