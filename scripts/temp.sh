python scripts/run_mlm_no_trainer.py \
    --model_name_or_path bert-base-multilingual-cased \
    --train_file data/mlm_data/eng.txt \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 8 \
    --num_train_epochs 5 \
    --max_train_steps 10 \
    --max_seq_length 512 \
    --output_dir tmp/tmp


universal_dependencies

file="est"
TASK_NAME='ner_all'
result_file=experiments/ner_temp_test_results_all_${file}.txt
label_column_name="ner_tags"
text="t"
python scripts/run_ner_updated.py \
	--model_name_or_path bert-base-multilingual-cased \
	--do_predict \
	--prefix ${text} \
	--dataset_name wikiann \
	--ds_script_name wikiann \
	--data_file data/ner_all_train \
	--lang_config metadata/ner_metadata.json \
	--task_name $TASK_NAME \
	--label_column_name ${label_column_name} \
	--output_dir experiments/${TASK_NAME}/${TASK_NAME} \
	--result_file ${result_file} \
	--cache_dir /scratch/ffaisal/hug_cache/datasets \
	--max_steps 20 \
	--per_device_train_batch_size 12 \
	--learning_rate 5e-4 \
	--num_train_epochs 5 \
	--max_seq_length 256 \
	--overwrite_output_dir


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


python scripts/run_ner_updated.py \
	--model_name_or_path bert-base-multilingual-cased \
	--dataset_name universal_dependencies \
	--data_file data/udp_all_train \
	--do_train \
	--task_name $TASK_NAME \
	--label_column_name upos \
	--per_device_train_batch_size 12 \
	--learning_rate 5e-4 \
	--num_train_epochs 5 \
	--max_seq_length 256 \
	--cache_dir /scratch/ffaisal/hug_cache/datasets \
	--output_dir experiments/$TASK_NAME \
	--overwrite_output_dir \
	--evaluation_strategy epoch \
	--max_steps 20 \
	--train_adapter

  # module load python/3.8.6-ff
#training on task, while task adapter freezed

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


echo "-------------------------------Download UDP all train data-----------------------------"
cd data
wget -O udp_all_train.zip https://gmuedu-my.sharepoint.com/:u:/g/personal/ffaisal_gmu_edu/EZRJRItvJ9ZCsykKm8MZlwcBuMF8Va3kShpHcg4JqT3yxg?download=1
module load openjdk/11.0.2-qg
jar -xf udp_all_train.zip
rm udp_all_train.zip
module unload openjdk/11.0.2-qg
cd ..

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
#cp ../scripts/ad_l_trans_trainer.py src/transformers/trainer.py
pip install adapter-transformers
../vnv/vnv-adp-l/bin/python -m pip install --upgrade pip
cd ..
pip install --upgrade pip
pip3 install -r requirements.txt
##for A100 gpu
deactivate