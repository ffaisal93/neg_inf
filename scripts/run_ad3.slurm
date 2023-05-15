#!/bin/bash
#SBATCH --partition=gpuq                    # the DGX only belongs in the 'gpu'  partition
#SBATCH --qos=gpu                           # need to select 'gpu' QoS
#SBATCH --job-name=python-gpu
##SBATCH --output=../tr_output/udp_all.out
##SBATCH --error=../tr_output/udp_all.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1                 # up to 128; 
#SBATCH --gres=gpu:A100.40gb:1              # up to 8; only request what you need
##SBATCH --gres=gpu:3g.40gb:1 
#SBATCH --mem-per-cpu=35500M                 # memory per CORE; total memory is 1 TB (1,000,000 MB)
#SBATCH --export=ALL 
#SBATCH --time=3-01:00:00                   # set to 1hr; please choose carefully


# task=$1
# lang=$2
# lang2=$3
# lang3=$4

task=$1
ind1=$2

# to see ID and state of GPUs assigned
nvidia-smi

# echo ${task} ${lang} ${lang2} ${lang3}

# bash install.sh --task ${task} --lang ${lang} --lang2 ${lang2} --lang3 ${lang3}


# bash install.sh --task eval_udp_adapters_3 --lang yor --lang2 jap --lang3 fre


declare -a arr=("gub" "est" "bre" "eng" "ben" "kmr" "spa" "bul" "pms" "gle" "nep" "cym" "fin" "hye" "mya" "hin" "tel" "tam" "kor" "ell" "hun" "heb" "zho" "ara" "swe" "jap" "fre" "deu" "rus" "bam" "ewe" "hau" "ibo" "kin" "mos" "pcm" "wol" "yor")

count=0
start=0
end=9000

for ind2 in "${!arr[@]}";do
	for ind3 in "${!arr[@]}";do
		if [ $ind2 -gt $ind1 ] && [ $ind3 -gt $ind2 ]; then
			if [ "$count" -ge $start ] && [ "$count" -lt $end ]; then
				lang1=${arr[$ind1]}
				lang2=${arr[$ind2]}
				lang3=${arr[$ind3]}
				echo $ind1 $ind2 $ind3
				echo $task $lang1 $lang2 $lang3 $count
				bash install.sh --task ${task} --lang ${lang1} --lang2 ${lang2} --lang3 ${lang3}
			fi
			count=$((count+1))
		fi
	done
done