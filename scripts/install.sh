task=${task:-none}
lang=${lang:-eng}

while [ $# -gt 0 ]; do

   if [[ $1 == *"--"* ]]; then
        param="${1/--/}"
        declare $param="$2"
        echo $1 $2 #Optional to see the parameter:value result
   fi

  shift
done

echo ${task}
echo ${lang}

module load python/3.8.6-ff
cd /scratch/ffaisal/neg_inf


# echo "------------------------------Install adapter latest------------------------------"
# module load python/3.8.6-ff
# rm -rf adapter-transformers-l
# rm -rf vnv/vnv-adp-l
# python -m venv vnv/vnv-adp-l
# source vnv/vnv-adp-l/bin/activate
# wget -O adapters3.1.0.tar.gz https://github.com/adapter-hub/adapter-transformers/archive/refs/tags/adapters3.1.0.tar.gz
# tar -xf adapters3.1.0.tar.gz
# rm adapters3.1.0.tar.gz
# mv adapter-transformers-adapters3.1.0 adapter-transformers-l
# cd adapter-transformers-l
##cp ../scripts/ad_l_trans_trainer.py src/transformers/trainer.py
# pip install adapter-transformers
# ../vnv/vnv-adp-l/bin/python -m pip install --upgrade pip
# cd ..
# pip install --upgrade pip
# pip3 install -r requirements.txt
# ##for A100 gpu
# deactivate




# echo "------------------------------Install transformer latest------------------------------"
# module load python/3.8.6-ff
# rm -rf transformers-orig
# rm -rf venv vnv/vnv-trns
# module load python/3.8.6-ff
# python -m venv vnv/vnv-trns
# python -m venv vnv/vnv-trns
# source vnv/vnv-trns/bin/activate
# wget -O transformersv4.21.1.tar.gz "https://github.com/huggingface/transformers/archive/refs/tags/v4.21.1.tar.gz"
# tar -xf transformersv4.21.1.tar.gz
# rm transformersv4.21.1.tar.gz
# mv transformers-4.21.1 transformers-orig
# cd transformers-orig
# pip install .
# pip install --upgrade pip
# cd ..
# pip install -r requirements.txt
# deactivate


# if [[ "$task" = "download_udp_train" || "$task" = "all" ]]; then
# 	echo "-------------------------------Download UDP all train data-----------------------------"
# 	cd data
# 	wget -O udp_all_train.zip https://gmuedu-my.sharepoint.com/:u:/g/personal/ffaisal_gmu_edu/EZRJRItvJ9ZCsykKm8MZlwcBuMF8Va3kShpHcg4JqT3yxg?download=1
# 	module load openjdk/11.0.2-qg
# 	jar -xf udp_all_train.zip
# 	rm udp_all_train.zip
# 	module unload openjdk/11.0.2-qg
# 	cd ..
# fi


if [[ "$task" = "train_udp_all" || "$task" = "all" ]]; then

	echo "------------------------------Train UDP combined adapter------------------------------"
	source vnv/vnv-adp-l/bin/activate

	export TASK_NAME="udp_all"

	python scripts/run_udp.py \
	    --model_name_or_path bert-base-cased \
	    --do_train \
	    --task_name $TASK_NAME \
	    --per_device_train_batch_size 12 \
	    --learning_rate 5e-4 \
	    --num_train_epochs 5 \
	    --max_seq_length 256 \
	    --cache_dir /scratch/ffaisal/hug_cache/datasets \
	    --output_dir experiments/$TASK_NAME \
	    --overwrite_output_dir \
	    --store_best_model \
	    --evaluation_strategy epoch \
	    --metric_score las \
	    --train_adapter
	deactivate
fi

if [[ "$task" = "download_mlm_train" || "$task" = "all" ]]; then
	echo "------------------------------Download mlm all train data-------------------------------"
	wget -O mlm_data.zip https://gmuedu-my.sharepoint.com/:u:/g/personal/ffaisal_gmu_edu/EWB3UQCJefFMjFAPasc8BkoByY2aK9AQDRsM1yTwt3HK8w?download=1
	module load openjdk/11.0.2-qg
	jar -xf mlm_data.zip
	rm mlm_data.zip
	rm -rf __MACOSX
	module unload openjdk/11.0.2-qg

	echo "------------------------------Create mlm all train 10k data-------------------------------"
	mkdir same_length
	cd mlm_data

	for file in ./*
	do
		base=$(basename $file .txt)
		if [[ "$base" != "lang_meta.json" ]]; then
		    linec=($(wc -l < $file))
		    echo $file $base
		    if [[ $linec -gt 10000  ]]; then
		    	head -n 10000 $file > ../same_length/$file

			else
				cat $file $file $file $file $file $file $file $file $file $file > ${base}_1.txt
				head -n 10000 ${base}_1.txt > ../same_length/$file

			fi
		linea=($(wc -l < ../same_length/${base}.txt))
		echo $linec $linea
			
		fi
	    
	done
	cd ..
	cp mlm_data/lang_meta.json same_length/lang_meta.json	
	rm -rf mlm_data
	mkdir mlm_data
	mv same_length/* mlm_data/
	rm -rf same_length
	cd ..

fi


if [[ "$task" = "mlm_train_data_1k" || "$task" = "all" ]]; then
	echo "------------------------------Create mlm all train 1k data-------------------------------"
	cd data/mlm_data
	mkdir ../mlm_data_1k

	for file in ./*
	do
		base=$(basename $file .txt)
		if [[ "$base" != "lang_meta.json" ]]; then
		    linec=($(wc -l < $file))
		    echo $file $base
		    head -n 1000 $file > ../mlm_data_1k/$file
		linea=($(wc -l < ../mlm_data_1k/$file))
		echo $linec $linea
			
		fi
	    
	done
	cd ..
	cp mlm_data/lang_meta.json mlm_data_1k/lang_meta.json	
	cd ..

fi


# if [[ "$task" = "mlm_udp_train_eval" || "$task" = "all" ]]; then

# 	# declare -a arr=("gub" "est" "bre" "eng" "ben" "kmr" "spa" "bul")
# 	declare -a arr=("eng")
# 	# Set comma as delimiter

# 	## now loop through the above array
# 	for file in "${arr[@]}"
# 	do
# 		output_dir=tmp/test-mlm
# 		echo "------------------------------Train mlm ${file}--------------------------------------"
# 		source vnv/vnv-trns/bin/activate

# 		rm -rf output_dir

# 		filename=${file}.txt
# 		python scripts/run_mlm_no_trainer.py \
# 	    --model_name_or_path bert-base-multilingual-cased \
# 	    --train_file data/mlm_data/${filename} \
# 	    --per_device_train_batch_size 8 \
# 	    --per_device_eval_batch_size 8 \
# 	    --num_train_epochs 1 \
# 	    --gradient_accumulation_steps 1 \
# 	    --max_train_steps 10 \
# 	    --max_seq_length 512 \
# 	    --output_dir ${output_dir}

# 	    deactivate


# 	    echo "------------------------------evaluate mlm ${file} +udp------------------------------"
# 		source vnv/vnv-adp-l/bin/activate

# 		export TASK_NAME="en_ewt"

# 		python scripts/run_udp.py \
# 		    --model_name_or_path ${output_dir} \
# 		    --do_eval False \
# 		    --do_predict \
# 		    --lang_config data/mlm_data/lang_meta.json \
# 		    --family_name germanic \
# 		    --task_name $TASK_NAME \
# 		    --per_device_train_batch_size 12 \
# 		    --learning_rate 5e-4 \
# 		    --num_train_epochs 5 \
# 		    --max_seq_length 256 \
# 		    --cache_dir /scratch/ffaisal/hug_cache/datasets \
# 		    --output_dir experiments/$TASK_NAME/ud_$TASK_NAME \
# 		    --overwrite_output_dir \
# 		    --store_best_model \
# 		    --evaluation_strategy epoch \
# 		    --metric_score las 
# 		deactivate



# 	done
# fi



if [[ "$task" = "mlm_udp_single_train_eval" ]]; then
	# declare -a arr=("gub" "est" "bre" "eng" "ben" "kmr" "spa" "bul")
	declare -a families=("tupian" "uralic" "celtic" "germanic" "indic" "iranian" "romance" "slavic")
	file=${lang}
	datafile=data/udp_all_train
	rm experiments/udp_test_results_all_${file}.txt
	result_file=experiments/udp_test_results_all_${file}.txt
	declare -a max_steps=(1 10 100)
	for step in "${max_steps[@]}"; do
		for run in {1..10}; do
			output_dir=tmp/test-mlm-all-${file}
			echo "------------------------------Train mlm ${file}--------------------------------------"
			source vnv/vnv-trns/bin/activate

			rm -rf output_dir

			filename=${file}.txt
			python scripts/run_mlm_no_trainer.py \
		    --model_name_or_path bert-base-multilingual-cased \
		    --train_file data/mlm_data/${filename} \
		    --per_device_train_batch_size 8 \
		    --per_device_eval_batch_size 8 \
		    --num_train_epochs 1 \
		    --gradient_accumulation_steps 1 \
		    --max_train_steps ${step} \
		    --max_seq_length 512 \
		    --output_dir ${output_dir}

		    deactivate

			for fam in "${families[@]}";do
			    echo "------------------------------evaluate mlm ${file} +udp------------------------------"
				source vnv/vnv-adp-l/bin/activate
				echo ${file}_${fam}_${step}_${run}
				text=${file}_${fam}_${step}_${run}

				export TASK_NAME="udp_all"

				python scripts/run_udp_updated.py \
				    --model_name_or_path ${output_dir} \
				    --do_eval False \
				    --do_predict \
				    --prefix ${text} \
				    --lang_config data/mlm_data/lang_meta.json \
				    --family_name ${fam} \
				    --task_name $TASK_NAME \
				    --per_device_train_batch_size 12 \
				    --learning_rate 5e-4 \
				    --num_train_epochs 5 \
				    --max_seq_length 256 \
				    --cache_dir /scratch/ffaisal/hug_cache/datasets \
				    --output_dir experiments/$TASK_NAME/ud_$TASK_NAME \
				    --data_file ${datafile} \
				    --result_file ${result_file} \
				    --overwrite_output_dir \
				    --store_best_model \
				    --evaluation_strategy epoch \
				    --metric_score las 

				if [[ "$run" = 1 ]]; then
					echo mbert-${file}_${fam}_${step}_${run}
					text=mbert-${file}_${fam}_${step}_${run}

					python scripts/run_udp_updated.py \
					    --model_name_or_path bert-base-multilingual-cased \
					    --do_eval False \
					    --do_predict \
					    --prefix ${text} \
					    --lang_config data/mlm_data/lang_meta.json \
					    --family_name ${fam} \
					    --task_name $TASK_NAME \
					    --per_device_train_batch_size 12 \
					    --learning_rate 5e-4 \
					    --num_train_epochs 5 \
					    --max_seq_length 256 \
					    --cache_dir /scratch/ffaisal/hug_cache/datasets \
					    --output_dir experiments/$TASK_NAME/ud_$TASK_NAME \
					    --data_file ${datafile} \
				    	--result_file ${result_file} \
					    --overwrite_output_dir \
					    --store_best_model \
					    --evaluation_strategy epoch \
					    --metric_score las 
				fi

				deactivate
			done
		done
	done
fi





# source vnv/vnv-adp-l/bin/activate

# export TASK_NAME="udp_all"

# python scripts/run_udp_copy.py \
# --model_name_or_path bert-base-multilingual-cased \
# --do_eval False \
# --do_train \
# --prefix "mix_all" \
# --lang_config data/mlm_data/lang_meta.json \
# --family_name germanic \
# --task_name $TASK_NAME \
# --per_device_train_batch_size 48 \
# --learning_rate 5e-4 \
# --num_train_epochs 10 \
# --max_seq_length 256 \
# --cache_dir /scratch/ffaisal/hug_cache/datasets \
# --output_dir experiments/${TASK_NAME} \
# --overwrite_output_dir \
# --store_best_model \
# --evaluation_strategy epoch \
# --metric_score las \
# --train_adapter


# python scripts/run_udp_copy.py \
# --model_name_or_path bert-base-multilingual-cased \
# --do_eval False \
# --do_predict \
# --prefix "mix_all" \
# --lang_config data/mlm_data/lang_meta.json \
# --family_name germanic \
# --task_name $TASK_NAME \
# --per_device_train_batch_size 48 \
# --learning_rate 5e-4 \
# --num_train_epochs 5 \
# --max_seq_length 256 \
# --cache_dir /scratch/ffaisal/hug_cache/datasets \
# --output_dir experiments/${TASK_NAME}_mix_all/ud_udp_all \
# --overwrite_output_dir \
# --store_best_model \
# --evaluation_strategy epoch \
# --metric_score las 



# deactivate


# export TASK_NAME="en_ewt"
# result_file=experiments/udp_test_results_temp.txt
# python scripts/run_udp_updated.py \
# --model_name_or_path bert-base-multilingual-cased \
# --do_eval False \
# --do_train \
# --do_predict \
# --prefix "cc" \
# --lang_config data/mlm_data/lang_meta.json \
# --family_name germanic \
# --task_name  $TASK_NAME \
# --per_device_train_batch_size 32 \
# --learning_rate 5e-4 \
# --num_train_epochs 1 \
# --max_seq_length 256 \
# --cache_dir /scratch/ffaisal/hug_cache/datasets \
# --output_dir experiments/$TASK_NAME/ud_$TASK_NAME \
# --result_file ${result_file} \
# --overwrite_output_dir \
# --store_best_model \
# --evaluation_strategy epoch \
# --metric_score las \
# --train_adapter



# export TASK_NAME="en_ewt"
# result_file=experiments/udp_test_results_temp.txt
# python scripts/run_udp_updated.py \
# --model_name_or_path bert-base-multilingual-cased \
# --do_eval False \
# --do_train \
# --prefix "cc" \
# --lang_config data/mlm_data/lang_meta.json \
# --family_name germanic \
# --task_name  $TASK_NAME \
# --per_device_train_batch_size 32 \
# --learning_rate 5e-4 \
# --num_train_epochs 1 \
# --max_seq_length 256 \
# --cache_dir /scratch/ffaisal/hug_cache/datasets \
# --output_dir experiments/$TASK_NAME/ud_$TASK_NAME \
# --result_file ${result_file} \
# --overwrite_output_dir \
# --store_best_model \
# --evaluation_strategy epoch \
# --metric_score las
