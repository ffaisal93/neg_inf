#!/bin/bash
task=${1:-udp}

declare -a arr=("gub" "est" "bre" "eng" "ben" "kmr" "spa" "bul" "pms" "gle" "nep" "cym" "fin" "hye" "mya" "hin" "tel" "tam" "kor" "ell" "hun" "heb" "zho" "ara" "swe" "jap" "fre" "deu" "rus" "bam" "ewe" "hau" "ibo" "kin" "mos" "pcm" "wol" "yor")

# declare -a arr=("est")

if [[ "$task" = "udp" ]]; then

	#all udp adapter evaluate : 3set
	for ind1 in "${!arr[@]}";do
		file1=${arr[$ind1]}
		echo $file1
		outfile="../tr_output/${file1}_3_udp.out"
		errorfile="../tr_output/${file1}_3_udp.err"
		echo ${outfile}
		echo ${errorfile}
		echo
		sbatch -o ${outfile} -e ${errorfile} run_ad3.slurm eval_udp_adapters_3 ${ind1}
	done	
fi

if [[ "$task" = "pos" ]]; then

	#all pos adapter evaluate : 3set
	for ind1 in "${!arr[@]}";do
		file1=${arr[$ind1]}
		echo $file1
		outfile="../tr_output/${file1}_3_pos.out"
		errorfile="../tr_output/${file1}_3_pos.err"
		echo ${outfile}
		echo ${errorfile}
		echo
		sbatch -o ${outfile} -e ${errorfile} run_ad3.slurm eval_pos_adapters_3 ${ind1}
	done	
fi

if [[ "$task" = "ner" ]]; then

	#all ner adapter evaluate : 3set
	for ind1 in "${!arr[@]}";do
		file1=${arr[$ind1]}
		echo $file1
		outfile="../tr_output/${file1}_3_ner.out"
		errorfile="../tr_output/${file1}_3_ner.err"
		echo ${outfile}
		echo ${errorfile}
		echo
		sbatch -o ${outfile} -e ${errorfile} run_ad3.slurm eval_ner_adapters_3 ${ind1}
	done	
fi



