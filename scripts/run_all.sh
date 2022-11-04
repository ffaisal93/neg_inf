declare -a arr=("gub" "est" "bre" "eng" "ben" "kmr" "spa" "bul")

for file in "${arr[@]}";do
	echo ../tr_output/${file}_udp.out
	echo ../tr_output/${file}_udp.err
	sbatch -o ../tr_output/${file}_udp.out -e ../tr_output/${file}_udp.err run_single.slurm mlm_udp_single_train_eval ${file}
done 