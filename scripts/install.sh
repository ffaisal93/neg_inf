#!/bin/bash
task=${task:-none}
lang=${lang:-eng}
lang2=${lang2:-eng}
lang3=${lang3:-eng}
MODEL_NAME=${MODEL_NAME:-bert}
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

if [[ "$MODEL_NAME" = "bert" ]]; then
	MODEL_PATH='bert-base-multilingual-cased'
fi

if [[ "$MODEL_NAME" = "xlmr" ]]; then
	MODEL_PATH='xlm-roberta-base'
fi

if [[ "$task" = "install_adapter" || "$task" = "all" ]]; then
	echo "------------------------------Install adapter latest------------------------------"
	module load python/3.8.6-ff
	rm -rf adapter-transformers-l
	rm -rf vnv/vnv-adp-l
	python -m venv vnv/vnv-adp-l
	source vnv/vnv-adp-l/bin/activate
	wget -O adapters3.1.0.tar.gz https://github.com/adapter-hub/adapter-transformers/archive/refs/tags/adapters3.1.0.tar.gz
	tar -xf adapters3.1.0.tar.gz
	rm adapters3.1.0.tar.gz
	mv adapter-transformers-adapters3.1.0 adapter-transformers-l
	cd adapter-transformers-l
	cp ../scripts/ad_l_trans_trainer.py src/transformers/trainer.py
	pip install .
	../vnv/vnv-adp-l/bin/python -m pip install --upgrade pip
	cd ..
	pip install --upgrade pip
	pip3 install -r requirements.txt
	##for A100 gpu
	deactivate
fi



if [[ "$task" = "install_transformer" || "$task" = "all" ]]; then
	echo "------------------------------Install transformer latest------------------------------"
	module load python/3.8.6-ff
	
	rm -rf transformers-orig
	rm -rf vnv/vnv-trns
	module load python/3.8.6-ff
	python -m venv vnv/vnv-trns
	source vnv/vnv-trns/bin/activate
	wget -O transformersv4.21.1.tar.gz "https://github.com/huggingface/transformers/archive/refs/tags/v4.21.1.tar.gz"
	tar -xf transformersv4.21.1.tar.gz
	rm transformersv4.21.1.tar.gz
	mv transformers-4.21.1 transformers-orig
	cd transformers-orig
	pip install .
	pip install --upgrade pip
	cd ..
	pip install -r requirements.txt
	deactivate
fi

if [[ "$task" = "create_dir" || "$task" = "all" ]]; then
	echo "-------------------------------create data dir-----------------------------"
 	mkdir data
  	mkdir experiments
   mkdir results
   mkdir tr_output
fi


if [[ "$task" = "download_udp_train" || "$task" = "all" ]]; then
	echo "-------------------------------Download UDP all train data-----------------------------"
	cd data
	wget -O udp_all_train.zip https://gmuedu-my.sharepoint.com/:u:/g/personal/ffaisal_gmu_edu/EZRJRItvJ9ZCsykKm8MZlwcBuMF8Va3kShpHcg4JqT3yxg?download=1
	module load openjdk/11.0.2-qg
	jar -xf udp_all_train.zip
	rm udp_all_train.zip
	module unload openjdk/11.0.2-qg
	cd ..
fi

if [[ "$task" = "download_tydiqa_train" || "$task" = "all" ]]; then
	echo "-------------------------------Download Tydiqa all train data-----------------------------"
	cd data
	wget -O tydiqa-gold.zip https://gmuedu-my.sharepoint.com/:u:/g/personal/ffaisal_gmu_edu/EbQsqd4reOdMg-xlivmeoDQB6cyNyyx7ieD-DpElKBHIww?download=1
	module load openjdk/11.0.2-qg
	jar -xf tydiqa-gold.zip
	rm tydiqa-gold.zip
	module unload openjdk/11.0.2-qg
	cd ..
fi


if [[ "$task" = "download_xnli_train" || "$task" = "all" ]]; then
	echo "-------------------------------Download UDP all train data-----------------------------"
	cd data
	wget -O xnli_all_train.zip https://gmuedu-my.sharepoint.com/:u:/g/personal/ffaisal_gmu_edu/EdzCjtAwVbZNjAs9NddORnIBjaoJIuhllGYvySQ-05nqsw?download=1
	module load openjdk/11.0.2-qg
	jar -xf xnli_all_train.zip
	rm xnli_all_train.zip
	module unload openjdk/11.0.2-qg
	cd ..
fi

if [[ "$task" = "download_ner_train" || "$task" = "all" ]]; then
	echo "-------------------------------Download NER all train data-----------------------------"
	cd data
	wget -O ner_all_train.zip https://gmuedu-my.sharepoint.com/:u:/g/personal/ffaisal_gmu_edu/EVidvsVS7s1CuY2EcwpQ9bABip-n0Trjc2bfgmS7QlOlOg?download=1
	wget -O m2ner_all_train.zip https://gmuedu-my.sharepoint.com/:u:/g/personal/ffaisal_gmu_edu/EU1D43O4S1ZInHRTAygJpP8BEf08x9X0Kj7TTbPMvsr6vw?download=1
	module load openjdk/11.0.2-qg
	jar -xf ner_all_train.zip
	jar -xf m2ner_all_train.zip
	rm ner_all_train.zip
	rm m2ner_all_train.zip
	module unload openjdk/11.0.2-qg
	cd ..
fi


if [[ "$task" = "train_xnli_all" || "$task" = "all" ]]; then

	echo "------------------------------Train xnli combined adapter------------------------------"
	source vnv/vnv-adp-l/bin/activate

	export TASK_NAME="xnli_all"

	python scripts/run_xnli.py \
		--model_name_or_path $MODEL_PATH \
		--train_language all \
		--dataset_name xnli \
		--task_name $TASK_NAME \
		--data_file data/xnli_all_train\
		--do_train \
		--train_adapter \
		--do_eval \
		--per_device_train_batch_size 32 \
		--learning_rate 5e-5 \
		--num_train_epochs 5 \
		--max_seq_length 128 \
		--overwrite_output_dir \
		--output_dir experiments/${MODEL_NAME}_${TASK_NAME} \
		--cache_dir /scratch/ffaisal/hug_cache/datasets \
		--save_steps -1
	deactivate
fi

if [[ "$task" = "train_qa_all" || "$task" = "all" ]]; then

	echo "------------------------------Train qa combined adapter------------------------------"
	source vnv/vnv-adp-l/bin/activate

	export TASK_NAME="tydiqa_all"

	python scripts/run_qa.py \
		--model_name_or_path $MODEL_PATH \
		--dataset_name tydiqa \
		--data_file data/tydiqa-gold \
		--task_name $TASK_NAME \
		--data_file data/tydiqa-gold \
		--do_train \
		--train_adapter \
		--per_device_train_batch_size 32 \
		--learning_rate 3e-5 \
		--num_train_epochs 5 \
		--max_seq_length 384 \
		--doc_stride 128 \
		--overwrite_output_dir \
		--output_dir experiments/${MODEL_NAME}_${TASK_NAME} \
		--cache_dir /scratch/ffaisal/hug_cache/datasets 
	deactivate
fi

if [[ "$task" = "eval_qa_all" || "$task" = "all" ]]; then

	echo "------------------------------Train qa combined adapter------------------------------"
	source vnv/vnv-adp-l/bin/activate

	export TASK_NAME="tydiqa_all"
	text='temp'
	result_file='tmp/tmp.txt'
	python scripts/run_qa.py \
		--model_name_or_path ${MODEL_PATH} \
		--data_file data/tydiqa-gold \
		--do_predict \
		--prefix ${text} \
		--dataset_name tydiqa \
		--task_name $TASK_NAME \
		--output_dir experiments/${MODEL_NAME}_${TASK_NAME}/$TASK_NAME \
		--result_file ${result_file} \
		--cache_dir /scratch/ffaisal/hug_cache/datasets/$TASK_NAME \
		--per_device_train_batch_size 12 \
		--learning_rate 3e-5 \
		--num_train_epochs 5 \
		--max_seq_length 384 \
		--doc_stride 128 \
		--overwrite_output_dir \
		--cache_dir /scratch/ffaisal/hug_cache/datasets
	deactivate
fi



if [[ "$task" = "train_udp_all" || "$task" = "all" ]]; then

	echo "------------------------------Train UDP combined adapter------------------------------"
	source vnv/vnv-adp-l/bin/activate

	export TASK_NAME="udp_all"

	python scripts/run_udp_updated.py \
	    --model_name_or_path ${MODEL_PATH} \
	    --data_file data/udp_all_train \
	    --do_train \
	    --task_name $TASK_NAME \
	    --per_device_train_batch_size 12 \
	    --learning_rate 5e-4 \
	    --num_train_epochs 5 \
	    --max_seq_length 256 \
	    --cache_dir /scratch/ffaisal/hug_cache/datasets \
	    --output_dir /projects/antonis/fahim/neg_inf/experiments/${MODEL_NAME}/$TASK_NAME \
	    --overwrite_output_dir \
	    --store_best_model \
	    --evaluation_strategy epoch \
	    --metric_score las \
	    --train_adapter \
	    --save_steps 3000
	deactivate
fi

if [[ "$task" = "train_pos_all" || "$task" = "all" ]]; then

	echo "------------------------------Train POS combined adapter------------------------------"
	source vnv/vnv-adp-l/bin/activate

	export TASK_NAME="pos_all"

	python scripts/run_pos_updated.py \
		--model_name_or_path ${MODEL_PATH} \
		--dataset_name universal_dependencies \
		--data_file data/udp_all_train \
		--do_train \
		--task_name $TASK_NAME \
		--label_column_name upos \
		--per_device_train_batch_size 36 \
		--learning_rate 5e-4 \
		--num_train_epochs 5 \
		--max_seq_length 256 \
		--cache_dir /scratch/ffaisal/hug_cache/datasets \
		--output_dir /projects/antonis/fahim/neg_inf/experiments/${MODEL_NAME}/$TASK_NAME \
		--overwrite_output_dir \
		--evaluation_strategy epoch \
		--train_adapter
	deactivate

fi

if [[ "$task" = "train_ner_all" || "$task" = "all" ]]; then

	echo "------------------------------Train NER combined adapter------------------------------"
	source vnv/vnv-adp-l/bin/activate

	export label_column_name="ner_tags"
	export TASK_NAME="ner_all"
		# --max_steps 50 \

	export TASK_NAME="ner_all"
	python scripts/run_ner_updated.py \
		--model_name_or_path ${MODEL_PATH} \
		--dataset_name wikiann \
		--data_file data/ner_all_train \
		--do_train \
		--task_name $TASK_NAME \
		--label_column_name ${label_column_name} \
		--per_device_train_batch_size 36 \
		--learning_rate 5e-4 \
		--num_train_epochs 5 \
		--max_seq_length 256 \
		--cache_dir /scratch/ffaisal/hug_cache/datasets \
		--output_dir /projects/antonis/fahim/neg_inf/experiments/${MODEL_NAME}/$TASK_NAME \
		--overwrite_output_dir \
		--evaluation_strategy epoch \
		--train_adapter
	deactivate

fi

if [[ "$task" = "train_m2ner_all" || "$task" = "all" ]]; then

	echo "------------------------------Train M2NER combined adapter------------------------------"
	source vnv/vnv-adp-l/bin/activate

	export label_column_name="ner_tags"

	export TASK_NAME="m2ner_all"
	python scripts/run_ner_updated.py \
		--model_name_or_path bert-base-multilingual-cased \
		--dataset_name wikiann \
		--data_file data/m2ner_all_train \
		--do_train \
		--task_name $TASK_NAME \
		--label_column_name ${label_column_name} \
		--per_device_train_batch_size 36 \
		--learning_rate 5e-4 \
		--num_train_epochs 5 \
		--max_seq_length 256 \
		--cache_dir /scratch/ffaisal/hug_cache/datasets \
		--output_dir experiments/$TASK_NAME \
		--overwrite_output_dir \
		--evaluation_strategy epoch \
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
	rm -rf data/mlm_data
	mv mlm_data data/
	cd ..

fi


if [[ "$task" = "mlm_train_data_1k" || "$task" = "all" ]]; then
	echo "------------------------------Create mlm all train 1k data-------------------------------"
	cd data/mlm_data
	rm -rf ../mlm_data_1k
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
	# declare -a families=("tupian" "uralic" "celtic" "germanic" "indic" "iranian" "romance" "slavic")
	# declare -a families=("uralic")
	file=${lang}
	datafile=data/udp_all_train
	TASK_ADAPTER_DIR="/projects/antonis/fahim/neg_inf/experiments/${MODEL_NAME}"
	result_fold='/projects/antonis/fahim/neg_inf'
	udp_result_file="${result_fold}/experiments/${MODEL_NAME}_udp_test_results_all_${file}.txt"
	pos_result_file="${result_fold}/experiments/${MODEL_NAME}_pos_test_results_all_${file}.txt"
	rm ${udp_result_file}
	rm ${pos_result_file}

	declare -a max_steps=(1 10 100 1000)
	for step in "${max_steps[@]}"; do
		for run in {1..10}; do
			dseed=${run}
			output_dir=tmp/${MODEL_NAME}-udp-test-mlm-all-${file}
			echo "------------------------------Train mlm ${file}--------------------------------------"
			echo "data seed:"${dseed}
			source vnv/vnv-trns/bin/activate

			rm -rf output_dir

			filename=${file}.txt
			python scripts/run_mlm_no_trainer.py \
		    --model_name_or_path ${MODEL_PATH} \
		    --train_file data/mlm_data/${filename} \
		    --per_device_train_batch_size 8 \
		    --per_device_eval_batch_size 8 \
		    --num_train_epochs 5 \
		    --max_train_steps ${step} \
		    --max_seq_length 512 \
		    --output_dir ${output_dir}

		   deactivate

		   echo "------------------------------evaluate mlm ${file} +udp------------------------------"
			source vnv/vnv-adp-l/bin/activate
			echo ${file}_${step}_${run}
			text=${file}_${step}_${run}

			export TASK_NAME="udp_all"
			echo ${TASK_NAME}------------------------------------

			python scripts/run_udp_updated.py \
			    --model_name_or_path ${output_dir} \
			    --do_eval False \
			    --do_predict \
			    --prefix ${text} \
			    --ds_script_name scripts/universal_dependencies.py \
			    --lang_config metadata/udp_metadata.json \
			    --task_name $TASK_NAME \
			    --per_device_train_batch_size 12 \
			    --learning_rate 5e-4 \
			    --num_train_epochs 5 \
			    --max_seq_length 256 \
			    --cache_dir /scratch/ffaisal/hug_cache/datasets/$TASK_NAME \
			    --output_dir ${TASK_ADAPTER_DIR}/$TASK_NAME/ud_$TASK_NAME \
			    --data_file ${datafile} \
			    --result_file ${udp_result_file} \
			    --overwrite_output_dir \
			    --store_best_model \
			    --evaluation_strategy epoch \
			    --metric_score las 

			if [[ "$run" = 1 ]]; then
				echo mbert-${file}_${step}_${run}
				text=mbert-${file}_${step}_${run}

				python scripts/run_udp_updated.py \
				    --model_name_or_path ${MODEL_PATH} \
				    --do_eval False \
				    --do_predict \
				    --prefix ${text} \
				    --ds_script_name scripts/universal_dependencies.py \
				    --lang_config metadata/udp_metadata.json \
				    --task_name $TASK_NAME \
				    --per_device_train_batch_size 12 \
				    --learning_rate 5e-4 \
				    --num_train_epochs 5 \
				    --max_seq_length 256 \
				    --cache_dir /scratch/ffaisal/hug_cache/datasets/$TASK_NAME \
				    --output_dir ${TASK_ADAPTER_DIR}/$TASK_NAME/ud_$TASK_NAME \
				    --data_file ${datafile} \
			    	 --result_file ${udp_result_file} \
				    --overwrite_output_dir \
				    --store_best_model \
				    --evaluation_strategy epoch \
				    --metric_score las 
			fi


			#pos
			echo ${file}_${step}_${run}
			text=${file}_${step}_${run}
			export TASK_NAME="pos_all"
			echo ${TASK_NAME}------------------------------------

			python scripts/run_pos_updated.py \
				--model_name_or_path ${output_dir} \
				--do_predict \
				--prefix ${text} \
				--dataset_name universal_dependencies \
				--dataset_config_name en_ewt \
				--ds_script_name scripts/universal_dependencies.py \
				--data_file data/udp_all_train \
				--lang_config metadata/udp_metadata.json \
				--task_name $TASK_NAME \
				--label_column_name upos \
				--output_dir ${TASK_ADAPTER_DIR}/$TASK_NAME/$TASK_NAME \
				--result_file ${pos_result_file} \
				--cache_dir /scratch/ffaisal/hug_cache/datasets/$TASK_NAME \
				--per_device_train_batch_size 12 \
				--learning_rate 5e-4 \
				--num_train_epochs 5 \
				--max_seq_length 256 \
				--overwrite_output_dir

			if [[ "$run" = 1 ]]; then
				echo mbert-${file}_${step}_${run}
				text=mbert-${file}_${step}_${run}
				python scripts/run_pos_updated.py \
					--model_name_or_path ${MODEL_PATH} \
					--do_predict \
					--prefix ${text} \
					--dataset_name universal_dependencies \
					--ds_script_name scripts/universal_dependencies.py \
					--dataset_config_name en_ewt \
					--data_file data/udp_all_train \
					--lang_config metadata/udp_metadata.json \
					--task_name $TASK_NAME \
					--label_column_name upos \
					--output_dir ${TASK_ADAPTER_DIR}/$TASK_NAME/$TASK_NAME \
					--result_file ${pos_result_file} \
					--cache_dir /scratch/ffaisal/hug_cache/datasets/$TASK_NAME \
					--per_device_train_batch_size 12 \
					--learning_rate 5e-4 \
					--num_train_epochs 5 \
					--max_seq_length 256 \
					--overwrite_output_dir 
			fi			

			deactivate
		done
	done
fi




if [[ "$task" = "mlm_ner_single_train_eval" ]]; then
	# declare -a arr=("gub" "est" "bre" "eng" "ben" "kmr" "spa" "bul")
	# declare -a families=("tupian" "uralic" "celtic" "germanic" "indic" "iranian" "romance" "slavic")
	# declare -a families=("uralic")
	file=${lang}
	TASK_ADAPTER_DIR="/projects/antonis/fahim/neg_inf/experiments/${MODEL_NAME}"
	result_fold='/projects/antonis/fahim/neg_inf'
	result_file="${result_fold}/experiments/${MODEL_NAME}_ner_test_results_all_${file}.txt"
	rm ${result_file}

	m2_result_file="${result_fold}/experiments/${MODEL_NAME}_m2ner_test_results_all_${file}.txt"
	rm ${m2_result_file}

	declare -a max_steps=(1 10 100 1000)
	for step in "${max_steps[@]}"; do
		for run in {1..10}; do
			dseed=${run}
			output_dir=tmp/${MODEL_NAME}-ner-test-mlm-all-${file}
			echo "------------------------------Train mlm ${file}--------------------------------------"
			echo "data seed:"${dseed}
			source vnv/vnv-trns/bin/activate

			rm -rf output_dir

			filename=${file}.txt
			python scripts/run_mlm_no_trainer.py \
		    --model_name_or_path ${MODEL_PATH} \
		    --train_file data/mlm_data/${filename} \
		    --per_device_train_batch_size 8 \
		    --per_device_eval_batch_size 8 \
		    --num_train_epochs 5 \
		    --max_train_steps ${step} \
		    --max_seq_length 512 \
		    --output_dir ${output_dir}

		    deactivate

		   echo "------------------------------evaluate mlm ${file} +ner------------------------------"
			source vnv/vnv-adp-l/bin/activate


			#ner
			echo ${file}_${step}_${run}
			text=${file}_${step}_${run}			
			label_column_name="ner_tags"
			export TASK_NAME="ner_all"
			echo ${TASK_NAME}------------------------------------

			python scripts/run_ner_updated.py \
				--model_name_or_path ${output_dir} \
				--do_predict \
				--prefix ${text} \
				--dataset_name wikiann \
				--ds_script_name wikiann \
				--data_file data/ner_all_train \
				--lang_config metadata/ner_metadata.json \
				--task_name $TASK_NAME \
				--label_column_name ${label_column_name} \
				--output_dir ${TASK_ADAPTER_DIR}/$TASK_NAME/$TASK_NAME \
				--result_file ${result_file} \
				--cache_dir /scratch/ffaisal/hug_cache/datasets/$TASK_NAME \
				--per_device_train_batch_size 12 \
				--learning_rate 5e-4 \
				--num_train_epochs 5 \
				--max_seq_length 256 \
				--overwrite_output_dir



			if [[ "$run" = 1 ]]; then
				echo mbert-${file}_${step}_${run}
				text=mbert-${file}_${step}_${run}
				python scripts/run_ner_updated.py \
					--model_name_or_path ${MODEL_PATH} \
					--do_predict \
					--prefix ${text} \
					--dataset_name wikiann \
					--ds_script_name wikiann \
					--data_file data/ner_all_train \
					--lang_config metadata/ner_metadata.json \
					--task_name $TASK_NAME \
					--label_column_name ${label_column_name} \
					--output_dir ${TASK_ADAPTER_DIR}/$TASK_NAME/$TASK_NAME \
					--result_file ${result_file} \
					--cache_dir /scratch/ffaisal/hug_cache/datasets/$TASK_NAME \
					--per_device_train_batch_size 12 \
					--learning_rate 5e-4 \
					--num_train_epochs 5 \
					--max_seq_length 256 \
					--overwrite_output_dir 
			fi

			# echo ${file}_${step}_${run}++++++++++++++++
			# text=${file}_${step}_${run}
			# result_file=experiments/m2ner_test_results_all_${file}.txt
			# label_column_name="ner_tags"
			# export TASK_NAME="m2ner_all"
			# echo ${TASK_NAME}------------------------------------

			# python scripts/run_ner_updated.py \
			# 	--model_name_or_path ${output_dir} \
			# 	--do_predict \
			# 	--prefix ${text} \
			# 	--dataset_name masakhaner2 \
			# 	--ds_script_name masakhane/masakhaner2 \
			# 	--data_file data/m2ner_all_train \
			# 	--lang_config metadata/ner_m2_metadata.json \
			# 	--task_name $TASK_NAME \
			# 	--label_column_name ${label_column_name} \
			# 	--output_dir experiments/$TASK_NAME/$TASK_NAME \
			# 	--result_file ${result_file} \
			# 	--cache_dir /scratch/ffaisal/hug_cache/datasets/$TASK_NAME \
			# 	--per_device_train_batch_size 12 \
			# 	--learning_rate 5e-4 \
			# 	--num_train_epochs 5 \
			# 	--max_seq_length 256 \
			# 	--overwrite_output_dir



			# if [[ "$run" = 1 ]]; then
			# 	echo mbert-${file}_${step}_${run}
			# 	text=mbert-${file}_${step}_${run}
			# 	python scripts/run_ner_updated.py \
			# 		--model_name_or_path ${MODEL_PATH} \
			# 		--do_predict \
			# 		--prefix ${text} \
			# 		--dataset_name masakhaner2 \
			# 		--ds_script_name masakhane/masakhaner2 \
			# 		--data_file data/m2ner_all_train \
			# 		--lang_config metadata/ner_m2_metadata.json \
			# 		--task_name $TASK_NAME \
			# 		--label_column_name ${label_column_name} \
			# 		--output_dir experiments/$TASK_NAME/$TASK_NAME \
			# 		--result_file ${result_file} \
			# 		--cache_dir /scratch/ffaisal/hug_cache/datasets/$TASK_NAME \
			# 		--per_device_train_batch_size 12 \
			# 		--learning_rate 5e-4 \
			# 		--num_train_epochs 5 \
			# 		--max_seq_length 256 \
			# 		--overwrite_output_dir
			# fi			

			deactivate
		done
	done
fi


if [[ "$task" = "mlm_nli_single_train_eval" ]]; then
	# declare -a arr=("gub" "est" "bre" "eng" "ben" "kmr" "spa" "bul")
	# declare -a families=("tupian" "uralic" "celtic" "germanic" "indic" "iranian" "romance" "slavic")
	# declare -a families=("uralic")
	file=${lang}
	result_fold='/projects/antonis/fahim/neg_inf'
	rm ${result_fold}/experiments/${MODEL_NAME}_xnli_test_results_all_${file}.txt
	rm ${result_fold}/experiments/${MODEL_NAME}_anli_test_results_all_${file}.txt

	declare -a max_steps=(1 10 100 1000)
	for step in "${max_steps[@]}"; do
		for run in {1..10}; do
			dseed=${run}
			output_dir=tmp/${MODEL_NAME}-test-mlm-all-${file}
			echo "------------------------------Train mlm ${file}--------------------------------------"
			echo "data seed:"${dseed}
			source vnv/vnv-trns/bin/activate

			rm -rf output_dir

			filename=${file}.txt
			python scripts/run_mlm_no_trainer.py \
		    --model_name_or_path ${MODEL_PATH} \
		    --train_file data/mlm_data/${filename} \
		    --per_device_train_batch_size 8 \
		    --per_device_eval_batch_size 8 \
		    --num_train_epochs 5 \
		    --max_train_steps ${step} \
		    --max_seq_length 512 \
		    --output_dir ${output_dir}

		    deactivate

		   echo "------------------------------evaluate mlm ${file} +nli------------------------------"
			source vnv/vnv-adp-l/bin/activate


			#xnli
			echo ${file}_${step}_${run}
			text=${file}_${step}_${run}			
			result_file=${result_fold}/experiments/${MODEL_NAME}_xnli_test_results_all_${file}.txt
			export TASK_NAME="xnli_all"
			echo ${TASK_NAME}------------------------------------

			python scripts/run_xnli.py \
				--model_name_or_path ${output_dir} \
				--do_predict \
				--train_language en \
				--prefix ${text} \
				--dataset_name xnli \
				--lang_config metadata/xnli_metadata.json \
				--task_name $TASK_NAME \
				--output_dir experiments/${MODEL_NAME}_${TASK_NAME}/$TASK_NAME \
				--cache_dir /scratch/ffaisal/hug_cache/datasets/$TASK_NAME \
				--per_device_train_batch_size 32 \
				--result_file ${result_file} \
				--learning_rate 5e-5 \
				--num_train_epochs 5 \
				--max_seq_length 128 \
				--overwrite_output_dir \
				--cache_dir /scratch/ffaisal/hug_cache/datasets



			if [[ "$run" = 1 ]]; then
				echo mbert-${file}_${step}_${run}
				text=mbert-${file}_${step}_${run}
				python scripts/run_xnli.py \
					--model_name_or_path ${MODEL_PATH} \
					--do_predict \
					--train_language en \
					--prefix ${text} \
					--dataset_name xnli \
					--lang_config metadata/xnli_metadata.json \
					--task_name $TASK_NAME \
					--output_dir experiments/${MODEL_NAME}_${TASK_NAME}/$TASK_NAME \
					--cache_dir /scratch/ffaisal/hug_cache/datasets/$TASK_NAME \
					--per_device_train_batch_size 32 \
					--result_file ${result_file} \
					--learning_rate 5e-5 \
					--num_train_epochs 5 \
					--max_seq_length 128 \
					--overwrite_output_dir \
					--cache_dir /scratch/ffaisal/hug_cache/datasets
			fi

			echo ${file}_${step}_${run}++++++++++++++++
			text=${file}_${step}_${run}
			result_file=${result_fold}/experiments/${MODEL_NAME}_anli_test_results_all_${file}.txt
			export TASK_NAME="xnli_all"
			echo ${TASK_NAME}------------------------------------

			python scripts/run_xnli.py \
				--model_name_or_path ${output_dir} \
				--do_predict \
				--train_language bzd \
				--prefix ${text} \
				--dataset_name americas_nli \
				--lang_config metadata/anli_metadata.json \
				--task_name $TASK_NAME \
				--output_dir experiments/${MODEL_NAME}_${TASK_NAME}/$TASK_NAME \
				--result_file ${result_file} \
				--cache_dir /scratch/ffaisal/hug_cache/datasets/$TASK_NAME \
				--per_device_train_batch_size 32 \
				--learning_rate 5e-5 \
				--num_train_epochs 5 \
				--max_seq_length 128 \
				--overwrite_output_dir \
				--cache_dir /scratch/ffaisal/hug_cache/datasets



			if [[ "$run" = 1 ]]; then
				echo mbert-${file}_${step}_${run}
				text=mbert-${file}_${step}_${run}
				python scripts/run_xnli.py \
				--model_name_or_path ${MODEL_PATH} \
				--do_predict \
				--train_language bzd \
				--prefix ${text} \
				--dataset_name americas_nli \
				--lang_config metadata/anli_metadata.json \
				--task_name $TASK_NAME \
				--output_dir experiments/${MODEL_NAME}_${TASK_NAME}/$TASK_NAME \
				--result_file ${result_file} \
				--cache_dir /scratch/ffaisal/hug_cache/datasets/$TASK_NAME \
				--per_device_train_batch_size 32 \
				--learning_rate 5e-5 \
				--num_train_epochs 5 \
				--max_seq_length 128 \
				--overwrite_output_dir \
				--cache_dir /scratch/ffaisal/hug_cache/datasets
			fi			

			deactivate
		done
	done
fi

if [[ "$task" = "mlm_qa_single_train_eval" ]]; then
	# declare -a arr=("gub" "est" "bre" "eng" "ben" "kmr" "spa" "bul")
	# declare -a families=("tupian" "uralic" "celtic" "germanic" "indic" "iranian" "romance" "slavic")
	# declare -a families=("uralic")
	file=${lang}
	result_fold='/projects/antonis/fahim/neg_inf'
	rm ${result_fold}/experiments/${MODEL_NAME}_tydiqa_test_results_all_${file}.txt

	declare -a max_steps=(1 10 100 1000)
	for step in "${max_steps[@]}"; do
		for run in {1..10}; do
			dseed=${run}
			output_dir=tmp/${MODEL_NAME}-test-mlm-all-${file}
			echo "------------------------------Train mlm ${file}--------------------------------------"
			echo "data seed:"${dseed}
			source vnv/vnv-trns/bin/activate

			rm -rf output_dir

			filename=${file}.txt
			python scripts/run_mlm_no_trainer.py \
		    --model_name_or_path ${MODEL_PATH} \
		    --train_file data/mlm_data/${filename} \
		    --per_device_train_batch_size 8 \
		    --per_device_eval_batch_size 8 \
		    --num_train_epochs 5 \
		    --max_train_steps ${step} \
		    --max_seq_length 512 \
		    --output_dir ${output_dir}

		    deactivate

		   echo "------------------------------evaluate mlm ${file} +nli------------------------------"
			source vnv/vnv-adp-l/bin/activate


			#xnli
			echo ${file}_${step}_${run}
			text=${file}_${step}_${run}			
			result_file=${result_fold}/experiments/${MODEL_NAME}_tydiqa_test_results_all_${file}.txt
			export TASK_NAME="tydiqa_all"
			echo ${TASK_NAME}------------------------------------

			python scripts/run_qa.py \
				--model_name_or_path ${output_dir} \
				--data_file data/tydiqa-gold \
				--do_predict \
				--prefix ${text} \
				--dataset_name tydiqa \
				--task_name $TASK_NAME \
				--output_dir experiments/${MODEL_NAME}_${TASK_NAME}/$TASK_NAME \
				--result_file ${result_file} \
				--cache_dir /scratch/ffaisal/hug_cache/datasets/$TASK_NAME \
				--per_device_train_batch_size 12 \
				--learning_rate 3e-5 \
				--num_train_epochs 5 \
				--max_seq_length 384 \
				--doc_stride 128 \
				--overwrite_output_dir \
				--cache_dir /scratch/ffaisal/hug_cache/datasets



			if [[ "$run" = 1 ]]; then
				echo mbert-${file}_${step}_${run}
				text=mbert-${file}_${step}_${run}
				python scripts/run_qa.py \
					--model_name_or_path ${MODEL_PATH} \
					--data_file data/tydiqa-gold \
					--do_predict \
					--prefix ${text} \
					--dataset_name tydiqa \
					--task_name $TASK_NAME \
					--output_dir experiments/${MODEL_NAME}_${TASK_NAME}/$TASK_NAME \
					--result_file ${result_file} \
					--cache_dir /scratch/ffaisal/hug_cache/datasets/$TASK_NAME \
					--per_device_train_batch_size 12 \
					--learning_rate 3e-5 \
					--num_train_epochs 5 \
					--max_seq_length 384 \
					--doc_stride 128 \
					--overwrite_output_dir \
					--cache_dir /scratch/ffaisal/hug_cache/datasets
			fi
		
			deactivate
		done
	done
fi

if [[ "$task" = "test_nli" ]]; then
	source vnv/vnv-adp-l/bin/activate
	file='en'
	rm -rf /projects/antonis/fahim/neg_inf/experiments/xnli_all
	mkdir /projects/antonis/fahim/neg_inf/experiments/xnli_all
	result_file=/projects/antonis/fahim/neg_inf/experiments/xnli_all/${MODEL_NAME}_xnli_test_results_all_${file}.txt
	export TASK_NAME="xnli_all"
	text="temp"
	python scripts/run_xnli.py \
		--model_name_or_path ${MODEL_PATH} \
		--do_predict \
		--train_language en \
		--prefix ${text} \
		--dataset_name xnli \
		--lang_config metadata/xnli_metadata.json \
		--task_name $TASK_NAME \
		--output_dir experiments/${MODEL_NAME}_${TASK_NAME}/$TASK_NAME \
		--cache_dir /scratch/ffaisal/hug_cache/datasets/$TASK_NAME \
		--per_device_train_batch_size 32 \
		--learning_rate 5e-5 \
		--num_train_epochs 5 \
		--max_seq_length 128 \
		--overwrite_output_dir \
		--cache_dir /scratch/ffaisal/hug_cache/datasets


	python scripts/run_xnli.py \
		--model_name_or_path ${MODEL_PATH} \
		--do_predict \
		--train_language bzd \
		--prefix ${text} \
		--dataset_name americas_nli \
		--lang_config metadata/anli_metadata.json \
		--task_name $TASK_NAME \
		--output_dir experiments/${MODEL_NAME}_${TASK_NAME}/$TASK_NAME \
		--result_file ${result_file} \
		--cache_dir /scratch/ffaisal/hug_cache/datasets/$TASK_NAME \
		--per_device_train_batch_size 32 \
		--learning_rate 5e-5 \
		--num_train_epochs 5 \
		--max_seq_length 128 \
		--overwrite_output_dir \
		--cache_dir /scratch/ffaisal/hug_cache/datasets
	deactivate
fi


if [[ "$task" = "freeze_mlm_udp_single_train_eval" ]]; then
	# declare -a arr=("gub" "est" "bre" "eng" "ben" "kmr" "spa" "bul")
	declare -a families=("tupian" "uralic" "celtic" "germanic" "indic" "iranian" "romance" "slavic")
	file=${lang}
	datafile=data/udp_all_train
	rm experiments/udp_test_results_all_frz_${file}.txt
	result_file=experiments/udp_test_results_all_frz_${file}.txt
	declare -a max_steps=(1 10 100)
	for step in "${max_steps[@]}"; do
		for run in {1..10}; do
			dseed=${run}
			output_dir=tmp/test-frz-mlm-all-${file}
			echo "------------------------------Train mlm ${file}--------------------------------------"
			echo "data seed:"${dseed}
			source vnv/vnv-adp-l/bin/activate

			rm -rf output_dir

			filename=${file}.txt
			python scripts/run_mlm_freeze_adapter.py \
		    --model_name_or_path bert-base-multilingual-cased \
		    --train_file data/mlm_data/${filename} \
		    --load_adapter /scratch/ffaisal/neg_inf/experiments/udp_all/ud_udp_all \
		    --per_device_train_batch_size 8 \
		    --per_device_eval_batch_size 8 \
		    --do_train \
		    --freeze_adapter true \
		    --data_seed ${dseed} \
		    --max_steps ${step} \
		    --max_seq_length 512 \
		    --overwrite_output_dir \
		    --output_dir ${output_dir}

		    deactivate

			for fam in "${families[@]}";do
			    echo "------------------------------evaluate mlm ${file} +udp + data-seed ${dseed}------------------------------"
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

if [[ "$task" = "train_lang_adapters" ]]; then
	echo "------------------------------Train Lang Adapters--------------------------------------"
	file=${lang}
	filename=${file}.txt
	output_dir="/projects/antonis/fahim/neg_inf/adapters/${MODEL_NAME}/${file}"
	source vnv/vnv-adp-l/bin/activate
	python scripts/run_mlm_adapters.py \
		--model_name_or_path xlm-roberta-base \
		--train_file data/mlm_data/${filename} \
		--validation_file data/mlm_data_1k/${filename} \
		--per_device_train_batch_size 8 \
		--per_device_eval_batch_size 8 \
		--do_train \
		--do_eval \
		--output_dir ${output_dir} \
		--overwrite_output_dir \
		--train_adapter
	rm -rf ${output_dir}/checkpoint*
	deactivate
fi


if [[ "$task" = "eval_udp_adapters_1" ]]; then
	echo "------------------------------Evaluate UDP Adapters--------------------------------------"
	file=${lang}
	filename=${file}.txt
	datafile="data/udp_all_train"
	output_dir="tmp/adapters/${file}"
	text="${lang}"
	export TASK_NAME="udp_all"
	echo ${TASK_NAME}------------------------------------
	rm "experiments/adapter/udp_1lang_test_results_${lang}.txt"
	result_file="experiments/adapter/udp_1lang_test_results_${lang}.txt"
	source vnv/vnv-adp-l/bin/activate

	python scripts/run_udp_adapter.py \
		--model_name_or_path bert-base-multilingual-cased \
		--do_eval False \
		--do_predict \
		--do_predict_adapter \
		--lang_1 ${lang} \
		--lang_adapter_path tmp/adapters \
		--prefix ${text} \
		--ds_script_name scripts/universal_dependencies.py \
		--lang_config metadata/udp_metadata.json \
		--task_name $TASK_NAME \
		--per_device_train_batch_size 12 \
		--learning_rate 5e-4 \
		--num_train_epochs 5 \
		--max_seq_length 256 \
		--cache_dir /scratch/ffaisal/hug_cache/datasets/$TASK_NAME \
		--output_dir experiments/$TASK_NAME/ud_$TASK_NAME \
		--data_file ${datafile} \
		--result_file ${result_file} \
		--overwrite_output_dir \
		--store_best_model \
		--evaluation_strategy epoch \
		--metric_score las

	deactivate
fi

if [[ "$task" = "eval_pos_adapters_1" ]]; then
	echo "------------------------------Evaluate POS Adapters--------------------------------------"
	file=${lang}
	filename=${file}.txt
	datafile="data/udp_all_train"
	output_dir="tmp/adapters/${file}"
	text="${lang}"
	export TASK_NAME="pos_all"
	echo ${TASK_NAME}------------------------------------
	rm "experiments/adapter/pos_1lang_test_results_${lang}.txt"
	result_file="experiments/adapter/pos_1lang_test_results_${lang}.txt"
	source vnv/vnv-adp-l/bin/activate

	python scripts/run_pos_adapter.py \
		--model_name_or_path bert-base-multilingual-cased \
		--do_eval False \
		--do_predict \
		--do_predict_adapter \
		--lang_1 ${lang} \
		--lang_adapter_path tmp/adapters \
		--prefix ${text} \
		--ds_script_name scripts/universal_dependencies.py \
		--dataset_name universal_dependencies \
		--lang_config metadata/udp_metadata.json \
		--task_name $TASK_NAME \
		--label_column_name upos \
		--per_device_train_batch_size 12 \
		--learning_rate 5e-4 \
		--num_train_epochs 5 \
		--max_seq_length 256 \
		--cache_dir /scratch/ffaisal/hug_cache/datasets/$TASK_NAME \
		--output_dir experiments/$TASK_NAME/$TASK_NAME \
		--data_file ${datafile} \
		--result_file ${result_file} \
		--overwrite_output_dir

	deactivate
fi

if [[ "$task" = "eval_ner_adapters_1" ]]; then
	echo "------------------------------Evaluate UDP Adapters--------------------------------------"
	file=${lang}
	filename=${file}.txt
	label_column_name="ner_tags"
	datafile="data/ner_all_train"
	output_dir="tmp/adapters/${file}"
	text="${lang}"
	export TASK_NAME="ner_all"
	echo ${TASK_NAME}------------------------------------
	rm "experiments/adapter/ner_1lang_test_results_${lang}.txt"
	result_file="experiments/adapter/ner_1lang_test_results_${lang}.txt"
	source vnv/vnv-adp-l/bin/activate

	python scripts/run_ner_adapter.py \
		--model_name_or_path bert-base-multilingual-cased \
		--do_eval False \
		--do_predict \
		--do_predict_adapter \
		--lang_1 ${lang} \
		--lang_adapter_path tmp/adapters \
		--prefix ${text} \
		--ds_script_name wikiann \
		--dataset_name wikiann \
		--data_file data/ner_all_train \
		--lang_config metadata/ner_metadata.json \
		--task_name $TASK_NAME \
		--label_column_name ${label_column_name} \
		--per_device_train_batch_size 12 \
		--learning_rate 5e-4 \
		--num_train_epochs 5 \
		--max_seq_length 256 \
		--cache_dir /scratch/ffaisal/hug_cache/datasets/$TASK_NAME \
		--output_dir experiments/$TASK_NAME/$TASK_NAME \
		--result_file ${result_file} \
		--overwrite_output_dir

	deactivate
fi




if [[ "$task" = "eval_udp_adapters_2" ]]; then
	echo "------------------------------Evaluate UDP Adapters--------------------------------------"
	file=${lang}
	filename=${file}.txt
	datafile="data/udp_all_train"
	output_dir="tmp/adapters/${file}"
	text="${lang}_${lang2}"
	export TASK_NAME="udp_all"
	echo ${TASK_NAME}------------------------------------
	rm "experiments/adapter/udp_2lang_test_results_${lang}_${lang2}.txt"
	result_file="experiments/adapter/udp_2lang_test_results_${lang}_${lang2}.txt"
	source vnv/vnv-adp-l/bin/activate

	python scripts/run_udp_adapter.py \
		--model_name_or_path bert-base-multilingual-cased \
		--do_eval False \
		--do_predict \
		--do_predict_adapter \
		--lang_1 ${lang} \
		--lang_2 ${lang2} \
		--lang_adapter_path tmp/adapters \
		--prefix ${text} \
		--ds_script_name scripts/universal_dependencies.py \
		--lang_config metadata/udp_metadata.json \
		--task_name $TASK_NAME \
		--per_device_train_batch_size 12 \
		--learning_rate 5e-4 \
		--num_train_epochs 5 \
		--max_seq_length 256 \
		--cache_dir /scratch/ffaisal/hug_cache/datasets/$TASK_NAME \
		--output_dir experiments/$TASK_NAME/ud_$TASK_NAME \
		--data_file ${datafile} \
		--result_file ${result_file} \
		--overwrite_output_dir \
		--store_best_model \
		--evaluation_strategy epoch \
		--metric_score las

	deactivate
fi

if [[ "$task" = "eval_pos_adapters_2" ]]; then
	echo "------------------------------Evaluate POS Adapters--------------------------------------"
	file=${lang}
	filename=${file}.txt
	datafile="data/udp_all_train"
	output_dir="tmp/adapters/${file}"
	text="${lang}_${lang2}"
	export TASK_NAME="pos_all"
	echo ${TASK_NAME}------------------------------------
	rm "experiments/adapter/pos_2lang_test_results_${lang}_${lang2}.txt"
	result_file="experiments/adapter/pos_2lang_test_results_${lang}_${lang2}.txt"
	source vnv/vnv-adp-l/bin/activate

	python scripts/run_pos_adapter.py \
		--model_name_or_path bert-base-multilingual-cased \
		--do_eval False \
		--do_predict \
		--do_predict_adapter \
		--lang_1 ${lang} \
		--lang_2 ${lang2} \
		--lang_adapter_path tmp/adapters \
		--prefix ${text} \
		--ds_script_name scripts/universal_dependencies.py \
		--dataset_name universal_dependencies \
		--lang_config metadata/udp_metadata.json \
		--task_name $TASK_NAME \
		--label_column_name upos \
		--per_device_train_batch_size 12 \
		--learning_rate 5e-4 \
		--num_train_epochs 5 \
		--max_seq_length 256 \
		--cache_dir /scratch/ffaisal/hug_cache/datasets/$TASK_NAME \
		--output_dir experiments/$TASK_NAME/$TASK_NAME \
		--data_file ${datafile} \
		--result_file ${result_file} \
		--overwrite_output_dir

	deactivate
fi

if [[ "$task" = "eval_ner_adapters_2" ]]; then
	echo "------------------------------Evaluate UDP Adapters--------------------------------------"
	file=${lang}
	filename=${file}.txt
	label_column_name="ner_tags"
	datafile="data/ner_all_train"
	output_dir="tmp/adapters/${file}"
	text="${lang}_${lang2}"
	export TASK_NAME="ner_all"
	echo ${TASK_NAME}------------------------------------
	rm "experiments/adapter/ner_2lang_test_results_${lang}_${lang2}.txt"
	result_file="experiments/adapter/ner_2lang_test_results_${lang}_${lang2}.txt"
	source vnv/vnv-adp-l/bin/activate

	python scripts/run_ner_adapter.py \
		--model_name_or_path bert-base-multilingual-cased \
		--do_eval False \
		--do_predict \
		--do_predict_adapter \
		--lang_1 ${lang} \
		--lang_2 ${lang2} \
		--lang_adapter_path tmp/adapters \
		--prefix ${text} \
		--ds_script_name wikiann \
		--dataset_name wikiann \
		--data_file data/ner_all_train \
		--lang_config metadata/ner_metadata.json \
		--task_name $TASK_NAME \
		--label_column_name ${label_column_name} \
		--per_device_train_batch_size 12 \
		--learning_rate 5e-4 \
		--num_train_epochs 5 \
		--max_seq_length 256 \
		--cache_dir /scratch/ffaisal/hug_cache/datasets/$TASK_NAME \
		--output_dir experiments/$TASK_NAME/$TASK_NAME \
		--result_file ${result_file} \
		--overwrite_output_dir

	deactivate
fi

if [[ "$task" = "eval_udp_adapters_3" ]]; then
	echo "------------------------------Evaluate UDP Adapters--------------------------------------"
	file=${lang}
	filename=${file}.txt
	datafile="data/udp_all_train"
	output_dir="tmp/adapters/${file}"
	text="${lang}_${lang2}_${lang3}"
	export TASK_NAME="udp_all"
	echo ${TASK_NAME}------------------------------------
	rm "experiments/adapter/udp_3lang_test_results_${lang}_${lang2}_${lang3}.txt"
	result_file="experiments/adapter/udp_3lang_test_results_${lang}_${lang2}_${lang3}.txt"
	source vnv/vnv-adp-l/bin/activate

	python scripts/run_udp_adapter.py \
		--model_name_or_path bert-base-multilingual-cased \
		--do_eval False \
		--do_predict \
		--do_predict_adapter \
		--lang_1 ${lang} \
		--lang_2 ${lang2} \
		--lang_3 ${lang3} \
		--lang_adapter_path tmp/adapters \
		--prefix ${text} \
		--ds_script_name scripts/universal_dependencies.py \
		--lang_config metadata/udp_metadata.json \
		--task_name $TASK_NAME \
		--per_device_train_batch_size 12 \
		--learning_rate 5e-4 \
		--num_train_epochs 5 \
		--max_seq_length 256 \
		--cache_dir /scratch/ffaisal/hug_cache/datasets/$TASK_NAME \
		--output_dir experiments/$TASK_NAME/ud_$TASK_NAME \
		--data_file ${datafile} \
		--result_file ${result_file} \
		--overwrite_output_dir \
		--store_best_model \
		--evaluation_strategy epoch \
		--metric_score las

	deactivate
fi

if [[ "$task" = "eval_pos_adapters_3" ]]; then
	echo "------------------------------Evaluate POS Adapters--------------------------------------"
	file=${lang}
	filename=${file}.txt
	datafile="data/udp_all_train"
	output_dir="tmp/adapters/${file}"
	text="${lang}_${lang2}_${lang3}"
	export TASK_NAME="pos_all"
	echo ${TASK_NAME}------------------------------------
	rm "experiments/adapter/pos_3lang_test_results_${lang}_${lang2}_${lang3}.txt"
	result_file="experiments/adapter/pos_3lang_test_results_${lang}_${lang2}_${lang3}.txt"
	source vnv/vnv-adp-l/bin/activate

	python scripts/run_pos_adapter.py \
		--model_name_or_path bert-base-multilingual-cased \
		--do_eval False \
		--do_predict \
		--do_predict_adapter \
		--lang_1 ${lang} \
		--lang_2 ${lang2} \
		--lang_3 ${lang3} \
		--lang_adapter_path tmp/adapters \
		--prefix ${text} \
		--ds_script_name scripts/universal_dependencies.py \
		--dataset_name universal_dependencies \
		--lang_config metadata/udp_metadata.json \
		--task_name $TASK_NAME \
		--label_column_name upos \
		--per_device_train_batch_size 12 \
		--learning_rate 5e-4 \
		--num_train_epochs 5 \
		--max_seq_length 256 \
		--cache_dir /scratch/ffaisal/hug_cache/datasets/$TASK_NAME \
		--output_dir experiments/$TASK_NAME/$TASK_NAME \
		--data_file ${datafile} \
		--result_file ${result_file} \
		--overwrite_output_dir

	deactivate
fi

if [[ "$task" = "eval_ner_adapters_3" ]]; then
	echo "------------------------------Evaluate UDP Adapters--------------------------------------"
	file=${lang}
	filename=${file}.txt
	label_column_name="ner_tags"
	datafile="data/ner_all_train"
	output_dir="tmp/adapters/${file}"
	text="${lang}_${lang2}_${lang3}"
	export TASK_NAME="ner_all"
	echo ${TASK_NAME}------------------------------------
	rm "experiments/adapter/ner_3lang_test_results_${lang}_${lang2}_${lang3}.txt"
	result_file="experiments/adapter/ner_3lang_test_results_${lang}_${lang2}_${lang3}.txt"
	source vnv/vnv-adp-l/bin/activate

	python scripts/run_ner_adapter.py \
		--model_name_or_path bert-base-multilingual-cased \
		--do_eval False \
		--do_predict \
		--do_predict_adapter \
		--lang_1 ${lang} \
		--lang_2 ${lang2} \
		--lang_3 ${lang3} \
		--lang_adapter_path tmp/adapters \
		--prefix ${text} \
		--ds_script_name wikiann \
		--dataset_name wikiann \
		--data_file data/ner_all_train \
		--lang_config metadata/ner_metadata.json \
		--task_name $TASK_NAME \
		--label_column_name ${label_column_name} \
		--per_device_train_batch_size 12 \
		--learning_rate 5e-4 \
		--num_train_epochs 5 \
		--max_seq_length 256 \
		--cache_dir /scratch/ffaisal/hug_cache/datasets/$TASK_NAME \
		--output_dir experiments/$TASK_NAME/$TASK_NAME \
		--result_file ${result_file} \
		--overwrite_output_dir

	deactivate
fi


if [[ "$task" = "eval_xnli_adapters_1" ]]; then
	echo "------------------------------Evaluate UDP Adapters--------------------------------------"
	file=${lang}
	filename=${file}.txt
	text="${lang}"
	export TASK_NAME="xnli_all"
	echo ${TASK_NAME}------------------------------------
	result_fold='/projects/antonis/fahim/neg_inf'
	result_file=${result_fold}/experiments/adapter/${MODEL_NAME}_anli_1lang_test_results_${lang}.txt
	rm ${result_file}

	source vnv/vnv-adp-l/bin/activate

	python scripts/run_xnli.py \
		--model_name_or_path ${MODEL_PATH} \
		--do_predict \
		--do_predict_adapter \
		--train_language bzd \
		--lang_1 ${lang} \
		--lang_adapter_path /projects/antonis/fahim/neg_inf/adapters/${MODEL_NAME} \
		--prefix ${text} \
		--dataset_name americas_nli \
		--lang_config metadata/anli_metadata.json \
		--task_name $TASK_NAME \
		--output_dir experiments/${MODEL_NAME}_${TASK_NAME}/$TASK_NAME \
		--cache_dir /scratch/ffaisal/hug_cache/datasets/$TASK_NAME \
		--per_device_train_batch_size 32 \
		--result_file ${result_file} \
		--learning_rate 5e-5 \
		--num_train_epochs 5 \
		--max_seq_length 128 \
		--overwrite_output_dir \
		--cache_dir /scratch/ffaisal/hug_cache/datasets

	deactivate
fi


if [[ "$task" = "eval_xnli_adapters_2" ]]; then
	echo "------------------------------Evaluate UDP Adapters--------------------------------------"
	file=${lang}
	filename=${file}.txt
	text="${lang}_${lang2}"
	export TASK_NAME="xnli_all"
	echo ${TASK_NAME}------------------------------------
	result_fold='/projects/antonis/fahim/neg_inf'
	result_file=${result_fold}/experiments/adapter/${MODEL_NAME}_anli_2lang_test_results_${lang}_${lang2}.txt
	rm ${result_file}
	source vnv/vnv-adp-l/bin/activate

	python scripts/run_xnli.py \
		--model_name_or_path ${MODEL_PATH} \
		--do_predict \
		--do_predict_adapter \
		--train_language bzd \
		--lang_1 ${lang} \
		--lang_2 ${lang2} \
		--lang_adapter_path /projects/antonis/fahim/neg_inf/adapters/${MODEL_NAME} \
		--prefix ${text} \
		--dataset_name americas_nli \
		--lang_config metadata/anli_metadata.json \
		--task_name $TASK_NAME \
		--output_dir experiments/${MODEL_NAME}_${TASK_NAME}/$TASK_NAME \
		--cache_dir /scratch/ffaisal/hug_cache/datasets/$TASK_NAME \
		--per_device_train_batch_size 32 \
		--result_file ${result_file} \
		--learning_rate 5e-5 \
		--num_train_epochs 5 \
		--max_seq_length 128 \
		--overwrite_output_dir \
		--cache_dir /scratch/ffaisal/hug_cache/datasets


	deactivate
fi

if [[ "$task" = "eval_xnli_adapters_3" ]]; then
	echo "------------------------------Evaluate UDP Adapters--------------------------------------"
	file=${lang}
	filename=${file}.txt
	label_column_name="ner_tags"
	datafile="data/ner_all_train"
	output_dir="tmp/adapters/${file}"
	text="${lang}_${lang2}_${lang3}"
	export TASK_NAME="ner_all"
	echo ${TASK_NAME}------------------------------------
	rm "experiments/adapter/ner_3lang_test_results_${lang}_${lang2}_${lang3}.txt"
	result_file="experiments/adapter/ner_3lang_test_results_${lang}_${lang2}_${lang3}.txt"
	source vnv/vnv-adp-l/bin/activate

	python scripts/run_xnli.py \
		--model_name_or_path ${MODEL_PATH} \
		--do_predict \
		--do_predict_adapter \
		--train_language bzd \
		--lang_1 ${lang} \
		--lang_2 ${lang2} \
		--lang_3 ${lang3} \
		--lang_adapter_path tmp/adapters \
		--prefix ${text} \
		--dataset_name xnli \
		--lang_config metadata/xnli_metadata.json \
		--task_name $TASK_NAME \
		--output_dir experiments/${MODEL_NAME}_${TASK_NAME}/$TASK_NAME \
		--cache_dir /scratch/ffaisal/hug_cache/datasets/$TASK_NAME \
		--per_device_train_batch_size 32 \
		--result_file ${result_file} \
		--learning_rate 5e-5 \
		--num_train_epochs 5 \
		--max_seq_length 128 \
		--overwrite_output_dir \
		--cache_dir /scratch/ffaisal/hug_cache/datasets

	deactivate
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
