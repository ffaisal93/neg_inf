#!/bin/bash
#SBATCH --partition=gpuq                    # the DGX only belongs in the 'gpu'  partition
#SBATCH --qos=gpu                           # need to select 'gpu' QoS
#SBATCH --job-name=python-gpu
##SBATCH --output=../tr_output/udp_all.out
##SBATCH --error=../tr_output/udp_all.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1                 # up to 128; 
#SBATCH --gres=gpu:A100.40gb:1              # up to 8; only request what you need
#SBATCH --mem-per-cpu=35500M                 # memory per CORE; total memory is 1 TB (1,000,000 MB)
#SBATCH --export=ALL 
#SBATCH --time=3-01:00:00                   # set to 1hr; please choose carefully


task=$1
lang=$2
lang2=$3
lang3=$4
model=$5

# to see ID and state of GPUs assigned
nvidia-smi
echo ${task} ${lang} ${lang2} ${lang3}

bash install.sh --task ${task} --lang ${lang} --lang2 ${lang2} --lang3 ${lang3} --MODEL_NAME ${model}


# bash install.sh --task eval_udp_adapters_3 --lang yor --lang2 jap --lang3 fre
