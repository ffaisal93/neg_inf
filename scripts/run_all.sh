#!/bin/bash
declare -a arr=("gub" "est" "bre" "eng" "ben" "kmr" "spa" "bul" "pms" "gle" "nep" "cym" "fin" "hye" "mya" "hin" "tel" "tam" "kor" "ell" "hun" "heb" "zho" "ara" "swe" "jap" "fre" "deu" "rus" "bam" "ewe" "hau" "ibo" "kin" "mos" "pcm" "wol" "yor")

# declare -a arr=("yor")

# declare -a arr=("gub" "est" "bre" "eng" "ben")


# ###all udp-pos evaluate
# for file in "${arr[@]}";do
# 	echo ../tr_output/${file}_udp.out
# 	echo ../tr_output/${file}_udp.err
# 	sbatch -o ../tr_output/${file}_udp.out -e ../tr_output/${file}_udp.err run_single.slurm mlm_udp_single_train_eval ${file}
# done 

# ###all udp-pos evaluate
# for file in "${arr[@]}";do
# 	echo ../tr_output/${file}_adp.out
# 	echo ../tr_output/${file}_adp.err
# 	sbatch -o ../tr_output/${file}_adp.out -e ../tr_output/${file}_adp.err run_single.slurm train_lang_adapters ${file}
# done 

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


# ##all udp adapter evaluate : 1 set
# for ind1 in "${!arr[@]}";do
# 	file1=${arr[$ind1]}
# 	echo $file1
# 	outfile="../tr_output/${file1}_ner.out"
# 	errorfile="../tr_output/${file1}_ner.err"
# 	echo ${outfile}
# 	echo ${errorfile}
# 	echo
# 	sbatch -o ${outfile} -e ${errorfile} run_single.slurm eval_ner_adapters_1 ${file1}
# done 


# ##all udp adapter evaluate : 2 set
# for ind1 in "${!arr[@]}";do
# 	for ind2 in "${!arr[@]}";do
# 		if [ $ind2 -gt $ind1 ]; then
# 			file1=${arr[$ind1]}
# 			file2=${arr[$ind2]}
# 			echo $file1 $file2
# 			outfile="../tr_output/${file1}_${file2}_ner.out"
# 			errorfile="../tr_output/${file1}_${file2}_ner.err"
# 			echo ${outfile}
# 			echo ${errorfile}
# 			echo
# 			sbatch -o ${outfile} -e ${errorfile} run_single.slurm eval_ner_adapters_2 ${file1} ${file2}
# 		fi
# 	done
# done 

# #all udp adapter evaluate : 3set
# count=0
# start=815
# end=915
# for ind1 in "${!arr[@]}";do
# 	for ind2 in "${!arr[@]}";do
# 		for ind3 in "${!arr[@]}";do
# 			if [ $ind2 -gt $ind1 ] && [ $ind3 -gt $ind2 ]; then
# 				if [ "$count" -ge $start ] && [ "$count" -lt $end ]; then
# 					file1=${arr[$ind1]}
# 					file2=${arr[$ind2]}
# 					file3=${arr[$ind3]}
# 					echo $file1 $file2 $file3 $count
# 					outfile="../tr_output/${file1}_${file2}_${file3}_udp.out"
# 					errorfile="../tr_output/${file1}_${file2}_${file3}_udp.err"
# 					echo ${outfile}
# 					echo ${errorfile}
# 					echo
# 					sbatch -o ${outfile} -e ${errorfile} run_single.slurm eval_udp_adapters_3 ${file1} ${file2} ${file3}
#   				fi
# 				count=$((count+1))
# 			fi
# 		done
# 	done
# done 


# #all-udp adapter train
# file=est
# echo ../tr_output/${file}_udp.out
# echo ../tr_output/${file}_udp.err
# sbatch -o ../tr_output/${file}_udp.out -e ../tr_output/${file}_udp.err run_single.slurm eval_udp_adapters ${file}

# #all-udp adapter train
# file=udp
# echo ../tr_output/${file}_udp.out
# echo ../tr_output/${file}_udp.err
# sbatch -o ../tr_output/${file}_udp.out -e ../tr_output/${file}_udp.err run_single.slurm train_udp_all ${file}


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
