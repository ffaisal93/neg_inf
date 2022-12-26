#!/bin/bash
declare -a arr=("gub" "est" "bre" "eng" "ben" "kmr" "spa" "bul" "pms" "gle" "nep" "cym" "fin" "hye" "mya" "hin" "tel" "tam" "kor" "ell" "hun" "heb" "zho" "ara" "swe" "jap" "fre" "deu" "rus" "bam" "ewe" "hau" "ibo" "kin" "mos" "pcm" "wol" "yor")

# declare -a arr=("yor")

# declare -a arr=("est")


###all udp-pos evaluate
for file in "${arr[@]}";do
	echo ../tr_output/${file}_udp.out
	echo ../tr_output/${file}_udp.err
	sbatch -o ../tr_output/${file}_udp.out -e ../tr_output/${file}_udp.err run_single.slurm mlm_udp_single_train_eval ${file}
done 

# ##all ner evaluate
# for file in "${arr[@]}";do
# 	echo ../tr_output/${file}_ner.out
# 	echo ../tr_output/${file}_ner.err
# 	sbatch -o ../tr_output/${file}_ner.out -e ../tr_output/${file}_ner.err run_single.slurm mlm_ner_single_train_eval ${file}
# done 

#all udp freeze adapter evaluate
# for file in "${arr[@]}";do
# 	echo ../tr_output/${file}_udp.out
# 	echo ../tr_output/${file}_udp.err
# 	sbatch -o ../tr_output/${file}_udp.out -e ../tr_output/${file}_udp.err run_single.slurm freeze_mlm_udp_single_train_eval ${file}
# done 

##all-pos adapter train
# file=pos
# echo ../tr_output/${file}_udp.out
# echo ../tr_output/${file}_udp.err
# sbatch -o ../tr_output/${file}_udp.out -e ../tr_output/${file}_udp.err run_single.slurm train_pos_all ${file}


# ##all-ner adapter train
# file=ner
# echo ../tr_output/${file}_udp.out
# echo ../tr_output/${file}_udp.err
# sbatch -o ../tr_output/${file}_udp.out -e ../tr_output/${file}_udp.err run_single.slurm train_ner_all ${file}

# ##all-m2ner adapter train
# file=m2ner
# echo ../tr_output/${file}_udp.out
# echo ../tr_output/${file}_udp.err
# sbatch -o ../tr_output/${file}_udp.out -e ../tr_output/${file}_udp.err run_single.slurm train_m2ner_all ${file}
