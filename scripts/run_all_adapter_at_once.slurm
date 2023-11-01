#!/bin/bash
#SBATCH --partition=gpuq                    # the DGX only belongs in the 'gpu'  partition
#SBATCH --qos=gpu                           # need to select 'gpu' QoS
#SBATCH --job-name=python-gpu
#SBATCH --output=../tr_output/anli_all.out
#SBATCH --error=../tr_output/anli_all.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1                 # up to 128; 
#SBATCH --gres=gpu:A100.80gb:1              # up to 8; only request what you need
#SBATCH --mem-per-cpu=35500M                 # memory per CORE; total memory is 1 TB (1,000,000 MB)
#SBATCH --export=ALL 
#SBATCH --time=3-01:00:00                   # set to 1hr; please choose carefully


# task=$1
# lang=$2
# lang2=$3
# lang3=$4

# to see ID and state of GPUs assigned
nvidia-smi
# echo ${task} ${lang} ${lang2} ${lang3}

# bash install.sh --task ${task} --lang ${lang} --lang2 ${lang2} --lang3 ${lang3}


declare -a arr=("gub" "est" "bre" "eng" "ben" "kmr" "spa" "bul" "pms" "gle" "nep" "cym" "fin" "hye" "mya" "hin" "tel" "tam" "kor" "ell" "hun" "heb" "zho" "ara" "swe" "jap" "fre" "deu" "rus" "bam" "ewe" "hau" "ibo" "kin" "mos" "pcm" "wol" "yor")


task='eval_xnli_adapters_1'
# declare -a arr=("yor")
for ind1 in "${!arr[@]}";do
	file1=${arr[$ind1]}
	echo $file1
	bash install.sh --task ${task} --lang ${file1} --MODEL_NAME xlmr
	# bash install.sh --task ${task} --lang ${file1} --MODEL_NAME bert
done