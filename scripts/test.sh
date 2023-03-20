#!/bin/bash
ope=$1

#opes:
#mb-en
#mb-ud
#en-mb-en
#en-mb-fr_en-en
#en-mb-fr_ud-en
#en-mb-ud
#en-mb-fr_en-ud
#en-mb-fr_ud-ud


# on en-ewt: to match the performance of full finetuning to an acceptable level, the learning rate has to differ:
# -full-finetune: 1e-4 (90/87)
# -adapter: 10e-4 (88/85)

# language modeling:
# if I continue running, it decreases
# 	- step 10: same 88/85
# 	- step 1 epoch: 88/84
# 	- step 100: 87/83

cd /scratch/ffaisal/neg_inf/
source deactivate
source vnv/vnv-adp-l/bin/activate

cd /scratch/ffaisal/neg_inf/adapter-transformers-l/examples/pytorch


TASK_NAME="en_ewt"

if [ "$ope" == "mb" ]; then

	cd dependency-parsing
	python run_udp.py \
    --model_name_or_path bert-base-multilingual-cased \
    --do_train \
    --do_eval \
    --do_predict \
    --task_name $TASK_NAME \
    --per_device_train_batch_size 64 \
    --learning_rate 3e-4 \
    --num_train_epochs 1 \
    --max_seq_length 256 \
    --output_dir experiments/$ope \
    --overwrite_output_dir \
    --store_best_model \
    --evaluation_strategy epoch \
    --metric_score uas 
    cd ..

elif [ "$ope" == "mb-en" ]; then

	cd dependency-parsing
	python run_udp.py \
    --model_name_or_path bert-base-multilingual-cased \
    --load_adapter experiments/$TASK_NAME-mbert/ud_en_ewt \
    --do_predict \
    --task_name $TASK_NAME \
    --per_device_train_batch_size 64 \
    --learning_rate 10e-4 \
    --num_train_epochs 1 \
    --max_seq_length 256 \
    --output_dir experiments/$TASK_NAME-mbert/ \
    --overwrite_output_dir \
    --store_best_model \
    --evaluation_strategy epoch \
    --metric_score uas \
    --train_adapter
    cd ..


elif [ "$ope" == "mb-ud" ]; then

	cd dependency-parsing
	python run_udp.py \
    --model_name_or_path bert-base-multilingual-cased \
    --load_adapter /scratch/ffaisal/neg_inf/experiments/udp_all/ud_udp_all \
    --do_predict \
    --task_name $TASK_NAME \
    --per_device_train_batch_size 64 \
    --learning_rate 1e-4 \
    --num_train_epochs 1 \
    --max_seq_length 256 \
    --max_steps 1 \
    --output_dir experiments/$ope \
    --overwrite_output_dir \
    --store_best_model \
    --evaluation_strategy epoch \
    --metric_score uas \
    --train_adapter
    cd ..

elif [ "$ope" == "en-mb-en" ]; then

	mfile="a-wo"
	cd language-modeling
	python run_mlm.py \
    --model_name_or_path bert-base-multilingual-cased \
    --train_file /scratch/ffaisal/neg_inf/data/mlm_data_1k/eng.txt \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 8 \
    --do_train \
    --do_eval \
    --num_train_epochs 1 \
    --max_steps 1 \
    --overwrite_output_dir \
    --output_dir tmp/a-wo
    cd ..

    cd dependency-parsing
	python run_udp.py \
    --model_name_or_path /scratch/ffaisal/neg_inf/adapter-transformers-l/examples/pytorch/language-modeling/tmp/${mfile} \
    --load_adapter experiments/$TASK_NAME-mbert/ud_en_ewt \
    --do_predict \
    --task_name $TASK_NAME \
    --per_device_train_batch_size 64 \
    --learning_rate 10e-4 \
    --num_train_epochs 1 \
    --max_seq_length 256 \
    --output_dir experiments/$TASK_NAME-${mfile} \
    --overwrite_output_dir \
    --store_best_model \
    --evaluation_strategy epoch \
    --metric_score uas \
    --train_adapter
    cd ..

elif [ "$ope" == "en-mb-ufr_en-en" ]; then

	mfile="a-uf"
	cd language-modeling
	python run_mlm.py \
    --model_name_or_path bert-base-multilingual-cased \
    --train_file /scratch/ffaisal/neg_inf/data/mlm_data_1k/eng.txt \
    --load_adapter ../dependency-parsing/experiments/$TASK_NAME-mbert/ud_en_ewt \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 8 \
    --do_train \
    --do_eval \
    --num_train_epochs 1 \
    --max_steps 1 \
    --overwrite_output_dir \
    --output_dir tmp/${mfile}
    cd ..

    cd dependency-parsing
	python run_udp.py \
    --model_name_or_path /scratch/ffaisal/neg_inf/adapter-transformers-l/examples/pytorch/language-modeling/tmp/${mfile} \
    --load_adapter experiments/$TASK_NAME-mbert/ud_en_ewt \
    --do_predict \
    --task_name $TASK_NAME \
    --per_device_train_batch_size 64 \
    --learning_rate 10e-4 \
    --num_train_epochs 1 \
    --max_seq_length 256 \
    --output_dir experiments/$TASK_NAME-${mfile} \
    --overwrite_output_dir \
    --store_best_model \
    --evaluation_strategy epoch \
    --metric_score uas \
    --train_adapter
    cd ..

elif [ "$ope" == "en-mb-fr_en-en" ]; then

	mfile="a-f"
	cd language-modeling
	python run_mlm.py \
    --model_name_or_path bert-base-multilingual-cased \
    --train_file /scratch/ffaisal/neg_inf/data/mlm_data_1k/eng.txt \
    --load_adapter ../dependency-parsing/experiments/$TASK_NAME-mbert/ud_en_ewt \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 8 \
    --do_train \
    --do_eval \
    --num_train_epochs 1 \
    --freeze_adapter true \
    --max_steps 1 \
    --overwrite_output_dir \
    --output_dir tmp/${mfile}
    cd ..

    cd dependency-parsing
	python run_udp.py \
    --model_name_or_path /scratch/ffaisal/neg_inf/adapter-transformers-l/examples/pytorch/language-modeling/tmp/${mfile} \
    --load_adapter experiments/$TASK_NAME-mbert/ud_en_ewt \
    --do_predict \
    --task_name $TASK_NAME \
    --per_device_train_batch_size 64 \
    --learning_rate 10e-4 \
    --num_train_epochs 1 \
    --max_seq_length 256 \
    --output_dir experiments/$TASK_NAME-${mfile} \
    --overwrite_output_dir \
    --store_best_model \
    --evaluation_strategy epoch \
    --metric_score uas \
    --train_adapter
    cd ..


elif [ "$ope" == "en-mb-fr_ud-ud" ]; then

    mfile="a-f-ud"
    cd language-modeling
    python run_mlm.py \
    --model_name_or_path bert-base-multilingual-cased \
    --train_file /scratch/ffaisal/neg_inf/data/mlm_data_1k/eng.txt \
    --load_adapter /scratch/ffaisal/neg_inf/experiments/udp_all/ud_udp_all \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 8 \
    --do_train \
    --do_eval \
    --num_train_epochs 1 \
    --freeze_adapter true \
    --max_steps 1 \
    --overwrite_output_dir \
    --output_dir tmp/${mfile}
    cd ..

    cd dependency-parsing
    python run_udp.py \
    --model_name_or_path /scratch/ffaisal/neg_inf/adapter-transformers-l/examples/pytorch/language-modeling/tmp/${mfile} \
    --load_adapter /scratch/ffaisal/neg_inf/experiments/udp_all/ud_udp_all \
    --do_predict \
    --task_name $TASK_NAME \
    --per_device_train_batch_size 64 \
    --learning_rate 10e-4 \
    --num_train_epochs 1 \
    --max_seq_length 256 \
    --output_dir experiments/$TASK_NAME-${mfile} \
    --overwrite_output_dir \
    --store_best_model \
    --evaluation_strategy epoch \
    --metric_score uas \
    --train_adapter
    cd ..


elif [ "$ope" == "en-mb-fr_ud-en" ]; then

    mfile="a-f-ud-en"
    cd language-modeling
    python run_mlm.py \
    --model_name_or_path bert-base-multilingual-cased \
    --train_file /scratch/ffaisal/neg_inf/data/mlm_data_1k/eng.txt \
    --load_adapter /scratch/ffaisal/neg_inf/experiments/udp_all/ud_udp_all \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 8 \
    --do_train \
    --do_eval \
    --num_train_epochs 1 \
    --freeze_adapter true \
    --max_steps 1 \
    --overwrite_output_dir \
    --output_dir tmp/${mfile}
    cd ..

    cd dependency-parsing
    python run_udp.py \
    --model_name_or_path /scratch/ffaisal/neg_inf/adapter-transformers-l/examples/pytorch/language-modeling/tmp/${mfile} \
    --load_adapter experiments/$TASK_NAME-mbert/ud_en_ewt \
    --do_predict \
    --task_name $TASK_NAME \
    --per_device_train_batch_size 64 \
    --learning_rate 10e-4 \
    --num_train_epochs 1 \
    --max_seq_length 256 \
    --output_dir experiments/$TASK_NAME-${mfile} \
    --overwrite_output_dir \
    --store_best_model \
    --evaluation_strategy epoch \
    --metric_score uas \
    --train_adapter
    cd ..

elif [ "$ope" == "en-mb-ufr_ud-en" ]; then

    mfile="a-uf-ud"
    cd language-modeling
    python run_mlm.py \
    --model_name_or_path bert-base-multilingual-cased \
    --train_file /scratch/ffaisal/neg_inf/data/mlm_data_1k/eng.txt \
    --load_adapter /scratch/ffaisal/neg_inf/experiments/udp_all/ud_udp_all \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 8 \
    --do_train \
    --do_eval \
    --num_train_epochs 1 \
    --max_steps 1 \
    --overwrite_output_dir \
    --output_dir tmp/${mfile}
    cd ..

    cd dependency-parsing
    python run_udp.py \
    --model_name_or_path /scratch/ffaisal/neg_inf/adapter-transformers-l/examples/pytorch/language-modeling/tmp/${mfile} \
    --load_adapter experiments/$TASK_NAME-mbert/ud_en_ewt \
    --do_predict \
    --task_name $TASK_NAME \
    --per_device_train_batch_size 64 \
    --learning_rate 10e-4 \
    --num_train_epochs 1 \
    --max_seq_length 256 \
    --output_dir experiments/$TASK_NAME-${mfile} \
    --overwrite_output_dir \
    --store_best_model \
    --evaluation_strategy epoch \
    --metric_score uas \
    --train_adapter
    cd ..


elif [ "$ope" == "ner_temp" ]; then

    mfile="a-uf-ud"

    cd /scratch/ffaisal/neg_inf
    output_dir="tmp/test"

    deactivate
    source vnv/vnv-trns/bin/activate

    python scripts/run_mlm_no_trainer.py \
        --model_name_or_path xlm-roberta-base \
        --train_file data/mlm_data/eng.txt \
        --per_device_train_batch_size 8 \
        --per_device_eval_batch_size 8 \
        --num_train_epochs 5 \
        --max_train_steps 100 \
        --max_seq_length 512 \
        --output_dir ${output_dir}
    deactivate


    source vnv/vnv-adp-l/bin/activate
    
    text="ner_temp"         
    result_file=tmp/ner_test.txt
    label_column_name="ner_tags"
    export TASK_NAME="ner"


    # python scripts/run_ner_updated.py \
    #     --model_name_or_path ${output_dir} \
    #     --do_predict \
    #     --prefix ${text} \
    #     --dataset_name wikiann \
    #     --ds_script_name wikiann \
    #     --data_file data/ner_all_train \
    #     --lang_config metadata/ner_metadata.json \
    #     --task_name $TASK_NAME \
    #     --label_column_name ${label_column_name} \
    #     --output_dir experiments/$TASK_NAME/$TASK_NAME \
    #     --result_file ${result_file} \
    #     --cache_dir /scratch/ffaisal/hug_cache/datasets/$TASK_NAME \
    #     --per_device_train_batch_size 12 \
    #     --learning_rate 5e-4 \
    #     --num_train_epochs 5 \
    #     --max_seq_length 256 \
    #     --overwrite_output_dir

    # python scripts/run_ner_updated.py \
    #     --model_name_or_path bert-base-multilingual-cased \
    #     --do_predict \
    #     --prefix ${text} \
    #     --dataset_name wikiann \
    #     --ds_script_name wikiann \
    #     --data_file data/ner_all_train \
    #     --lang_config metadata/ner_metadata.json \
    #     --task_name $TASK_NAME \
    #     --label_column_name ${label_column_name} \
    #     --output_dir experiments/$TASK_NAME/$TASK_NAME \
    #     --result_file ${result_file} \
    #     --cache_dir /scratch/ffaisal/hug_cache/datasets/$TASK_NAME \
    #     --per_device_train_batch_size 12 \
    #     --learning_rate 5e-4 \
    #     --num_train_epochs 5 \
    #     --max_seq_length 256 \
    #     --overwrite_output_dir

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
        --output_dir experiments/$TASK_NAME/$TASK_NAME/$TASK_NAME \
        --result_file ${result_file} \
        --cache_dir /scratch/ffaisal/hug_cache/datasets/$TASK_NAME \
        --per_device_train_batch_size 32 \
        --learning_rate 5e-4 \
        --num_train_epochs 1 \
        --max_seq_length 256 \
        --overwrite_output_dir \
        --dataset_config_name en \
        --do_train \
        --train_adapter

    # python scripts/run_ner_updated.py \
    #     --model_name_or_path bert-base-multilingual-uncased \
    #     --do_predict \
    #     --prefix ${text} \
    #     --dataset_name wikiann \
    #     --ds_script_name wikiann \
    #     --data_file data/ner_all_train \
    #     --lang_config metadata/ner_metadata.json \
    #     --task_name $TASK_NAME \
    #     --label_column_name ${label_column_name} \
    #     --output_dir experiments/$TASK_NAME/$TASK_NAME \
    #     --result_file ${result_file} \
    #     --cache_dir /scratch/ffaisal/hug_cache/datasets/$TASK_NAME \
    #     --per_device_train_batch_size 12 \
    #     --learning_rate 5e-4 \
    #     --num_train_epochs 5 \
    #     --max_seq_length 256 \
    #     --overwrite_output_dir

    deactivate


fi

#./test.sh mb-en
#./test.sh mb-ud

