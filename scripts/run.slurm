#!/bin/bash
#SBATCH --partition=gpu-test                    # the DGX only belongs in the 'gpu'  partition
#SBATCH --qos=gpu-test                           # need to select 'gpu' QoS
#SBATCH --job-name=python-gpu
#SBATCH --output=../tr_output/udp_all.out
#SBATCH --error=../tr_output/udp_all.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1                 # up to 128; 
#SBATCH --gres=gpu:A100.80gb:4              # up to 8; only request what you need
#SBATCH --mem-per-cpu=35500M                 # memory per CORE; total memory is 1 TB (1,000,000 MB)
#SBATCH --export=ALL 
#SBATCH --time=3-01:00:00                   # set to 1hr; please choose carefully



# to see ID and state of GPUs assigned
nvidia-smi


./install.sh mlm_udp_all_train_eval
