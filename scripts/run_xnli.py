#!/usr/bin/env python
# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" Finetuning multi-lingual models on XNLI (e.g. Bert, DistilBERT, XLM).
    Adapted from `examples/text-classification/run_glue.py`"""

import logging
import os
import random
import sys
from dataclasses import dataclass, field
from typing import Optional
import evaluate
import datasets
import numpy as np
from datasets import load_dataset, load_metric, load_from_disk, DatasetDict

import transformers

import transformers.adapters.composition as ac
from transformers import (
    AdapterConfig,
    AdapterTrainer,
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    EvalPrediction,
    HfArgumentParser,
    MultiLingAdapterArguments,
    PretrainedConfig,
    Trainer,
    TrainingArguments,
    default_data_collator,
    set_seed,
)

from transformers.trainer_utils import get_last_checkpoint
from transformers.utils import check_min_version
from transformers.utils.versions import require_version


# Will error if the minimal version of Transformers is not installed. Remove at your own risks.
check_min_version("4.12.0")

require_version("datasets>=1.8.0", "To fix: pip install -r examples/pytorch/text-classification/requirements.txt")

logger = logging.getLogger(__name__)

task_to_keys = {
    "cola": ("sentence", None),
    "mnli": ("premise", "hypothesis"),
    "xnli": ("premise", "hypothesis"),
    "mrpc": ("sentence1", "sentence2"),
    "qnli": ("question", "sentence"),
    "qqp": ("question1", "question2"),
    "rte": ("sentence1", "sentence2"),
    "sst2": ("sentence", None),
    "stsb": ("sentence1", "sentence2"),
    "wnli": ("sentence1", "sentence2"),

}


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.

    Using `HfArgumentParser` we can turn this class
    into argparse arguments to be able to specify them on
    the command line.
    """
    data_file: str = field(default=False,metadata={"help": "The identifier of the Universal Dependencies dataset to train on."})
    lang_config: str = field(default=False,metadata={"help": "The identifier of the Universal Dependencies dataset to train on."})
    max_seq_length: Optional[int] = field(
        default=128,
        metadata={
            "help": (
                "The maximum total input sequence length after tokenization. Sequences longer "
                "than this will be truncated, sequences shorter will be padded."
            )
        },
    )
    prefix: str = field(default=False,metadata={"help": "The identifier of the Universal Dependencies dataset to train on."})
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached preprocessed datasets or not."}
    )
    dataset_name: Optional[str] = field(
        default=None, metadata={"help": "The name of the dataset to use (via the datasets library)."}
    )
    task_name: Optional[str] = field(
        default=None,
        metadata={"help": "The name of the task to train on: " + ", ".join(task_to_keys.keys())},
    )
    result_file: str = field(default=False,metadata={"help": "The identifier of the Universal Dependencies dataset to train on."})
    pad_to_max_length: bool = field(
        default=True,
        metadata={
            "help": (
                "Whether to pad all samples to `max_seq_length`. "
                "If False, will pad the samples dynamically when batching to the maximum length in the batch."
            )
        },
    )
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of training examples to this "
                "value if set."
            )
        },
    )
    max_eval_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of evaluation examples to this "
                "value if set."
            )
        },
    )
    max_predict_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of prediction examples to this "
                "value if set."
            )
        },
    )
    do_predict_adapter: bool = field(
        default=False,
        metadata={"help": "Overwrite the cached training and evaluation sets."},
    )
    server_ip: Optional[str] = field(default=None, metadata={"help": "For distant debugging."})
    server_port: Optional[str] = field(default=None, metadata={"help": "For distant debugging."})
    def __post_init__(self):
        if self.task_name is not None:
            self.task_name = self.task_name.lower()
            # if self.task_name not in task_to_keys.keys():
            #     raise ValueError("Unknown task, you should pick one in " + ",".join(task_to_keys.keys()))


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    model_name_or_path: str = field(
        default=None, metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    # language: str = field(
    #     default=None, metadata={"help": "Evaluation language. Also train language if `train_language` is set to None."}
    # )
    train_language: Optional[str] = field(
        default=None, metadata={"help": "Train language if it is different from the evaluation language."}
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where do you want to store the pretrained models downloaded from huggingface.co"},
    )
    do_lower_case: Optional[bool] = field(
        default=False,
        metadata={"help": "arg to indicate if tokenizer should do lower case in AutoTokenizer.from_pretrained()"},
    )
    use_fast_tokenizer: bool = field(
        default=True,
        metadata={"help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."},
    )
    model_revision: str = field(
        default="main",
        metadata={"help": "The specific model version to use (can be a branch name, tag name or commit id)."},
    )
    use_auth_token: bool = field(
        default=False,
        metadata={
            "help": (
                "Will use the token generated when running `huggingface-cli login` (necessary to use this script "
                "with private models)."
            )
        },
    )
    lang_adapter_path: str = field( 
            default=None,   
            metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}    
        ) 
    lang_1: str = field(
        default=None,
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    lang_2: str = field(
        default=None,
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    lang_3: str = field(
        default=None,
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    ignore_mismatched_sizes: bool = field(
        default=False,
        metadata={"help": "Will enable to load a pretrained model whose head dimensions are different."},
    )


def main():
    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.

    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments, MultiLingAdapterArguments))
    model_args, data_args, training_args, adapter_args = parser.parse_args_into_dataclasses()

    # Sending telemetry. Tracking the example usage helps us better allocate resources to maintain them. The
    # information sent is the one passed as arguments along with your Python/PyTorch versions.
    # send_example_telemetry("run_xnli", model_args)

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    logger.info(f"Training/evaluation parameters {training_args}")

    # Detecting last checkpoint.
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir) and training_args.do_train and not training_args.overwrite_output_dir:
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
        elif last_checkpoint is not None:
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )

    # Set seed before initializing model.
    set_seed(training_args.seed)

    # In distributed training, the load_dataset function guarantees that only one local process can concurrently
    # download the dataset.
    # Downloading and loading xnli dataset from the hub.
    task_name = data_args.task_name 
    language = adapter_args.language
    if task_name!='xnli_all' and training_args.do_train:
        if model_args.train_language is None:
            train_dataset = load_dataset(
                data_args.dataset_name,
                model_args.language,
                split="train",
                cache_dir=model_args.cache_dir,
                use_auth_token=True if model_args.use_auth_token else None,
            )
        else:
            train_dataset = load_dataset(
                data_args.dataset_name,
                model_args.train_language,
                split="train",
                cache_dir=model_args.cache_dir,
                use_auth_token=True if model_args.use_auth_token else None,
            )
        label_list = train_dataset.features["label"].names
    elif task_name=='xnli_all' and training_args.do_train:
        dataset = load_from_disk(data_args.data_file)
        dataset=dataset.train_test_split(test_size=0.2)
        dataset = DatasetDict({"train":dataset['train'],"validation":dataset['test']})
        train_dataset=dataset['train']
        eval_dataset=dataset['validation']
        label_list = train_dataset.features["label"].names
        print(label_list)

    # if training_args.do_eval:
    #     eval_dataset = load_dataset(
    #         data_args.dataset_name,
    #         model_args.language,
    #         split="validation",
    #         cache_dir=model_args.cache_dir,
    #         use_auth_token=True if model_args.use_auth_token else None,
    #     )
    #     label_list = eval_dataset.features["label"].names

    if training_args.do_predict:
        print(model_args.train_language)
        predict_dataset = load_dataset(
            data_args.dataset_name,
            model_args.train_language,
            split="test",
            cache_dir=model_args.cache_dir,
            use_auth_token=True if model_args.use_auth_token else None,
        )
        label_list = predict_dataset.features["label"].names

    # Labels
    num_labels = len(label_list)

    # Load pretrained model and tokenizer
    # In distributed training, the .from_pretrained methods guarantee that only one local process can concurrently
    # download model & vocab.
    config = AutoConfig.from_pretrained(
        model_args.config_name if model_args.config_name else model_args.model_name_or_path,
        num_labels=num_labels,
        finetuning_task="xnli",
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
        do_lower_case=model_args.do_lower_case,
        cache_dir=model_args.cache_dir,
        use_fast=model_args.use_fast_tokenizer,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
    )
    model = AutoModelForSequenceClassification.from_pretrained(
        model_args.model_name_or_path,
        from_tf=bool(".ckpt" in model_args.model_name_or_path),
        config=config,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
        ignore_mismatched_sizes=model_args.ignore_mismatched_sizes,
    )

    # Setup adapters
    if adapter_args.train_adapter:
        # check if adapter already exists, otherwise add it
        if task_name not in model.config.adapters:
            # resolve the adapter config
            adapter_config = AdapterConfig.load(
                adapter_args.adapter_config,
                non_linearity=adapter_args.adapter_non_linearity,
                reduction_factor=adapter_args.adapter_reduction_factor,
            )
            # load a pre-trained from Hub if specified
            if adapter_args.load_adapter:
                model.load_adapter(
                    adapter_args.load_adapter,
                    config=adapter_config,
                    load_as=task_name,
                )
            # otherwise, add a fresh adapter
            else:
                model.add_adapter(task_name, config=adapter_config)
        model.train_adapter([task_name])

    # Preprocessing the datasets
    # Padding strategy
    if data_args.pad_to_max_length:
        padding = "max_length"
    else:
        # We will pad later, dynamically at batch creation, to the max sequence length in each batch
        padding = False

    def preprocess_function(examples):
        # Tokenize the texts
        return tokenizer(
            examples["premise"],
            examples["hypothesis"],
            padding=padding,
            max_length=data_args.max_seq_length,
            truncation=True,
        )

    if training_args.do_train:
        if data_args.max_train_samples is not None:
            max_train_samples = min(len(train_dataset), data_args.max_train_samples)
            train_dataset = train_dataset.select(range(max_train_samples))
        with training_args.main_process_first(desc="train dataset map pre-processing"):
            train_dataset = train_dataset.map(
                preprocess_function,
                batched=True,
                load_from_cache_file=not data_args.overwrite_cache,
                desc="Running tokenizer on train dataset",
            )
        # Log a few random samples from the training set:
        for index in random.sample(range(len(train_dataset)), 3):
            logger.info(f"Sample {index} of the training set: {train_dataset[index]}.")

    if training_args.do_eval:
        if data_args.max_eval_samples is not None:
            max_eval_samples = min(len(eval_dataset), data_args.max_eval_samples)
            eval_dataset = eval_dataset.select(range(max_eval_samples))
        with training_args.main_process_first(desc="validation dataset map pre-processing"):
            eval_dataset = eval_dataset.map(
                preprocess_function,
                batched=True,
                load_from_cache_file=not data_args.overwrite_cache,
                desc="Running tokenizer on validation dataset",
            )

    if training_args.do_predict:
        if data_args.max_predict_samples is not None:
            max_predict_samples = min(len(predict_dataset), data_args.max_predict_samples)
            predict_dataset = predict_dataset.select(range(max_predict_samples))
        with training_args.main_process_first(desc="prediction dataset map pre-processing"):
            predict_dataset = predict_dataset.map(
                preprocess_function,
                batched=True,
                load_from_cache_file=not data_args.overwrite_cache,
                desc="Running tokenizer on prediction dataset",
            )

    # Get the metric function
    metric = evaluate.load("xnli")

    # You can define your custom compute_metrics function. It takes an `EvalPrediction` object (a namedtuple with a
    # predictions and label_ids field) and has to return a dictionary string to float.
    def compute_metrics(p: EvalPrediction):
        preds = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
        preds = np.argmax(preds, axis=1)
        return metric.compute(predictions=preds, references=p.label_ids)

    # Data collator will default to DataCollatorWithPadding, so we change it if we already did the padding.
    if data_args.pad_to_max_length:
        data_collator = default_data_collator
    elif training_args.fp16:
        data_collator = DataCollatorWithPadding(tokenizer, pad_to_multiple_of=8)
    else:
        data_collator = None

    ## Initialize our Trainer
    trainer_class = AdapterTrainer if adapter_args.train_adapter else Trainer
    trainer = trainer_class(
        model=model,
        args=training_args,
        train_dataset=train_dataset if training_args.do_train else None,
        eval_dataset=eval_dataset if training_args.do_eval else None,
        compute_metrics=compute_metrics,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    # Training
    if training_args.do_train:
        checkpoint = None
        if training_args.resume_from_checkpoint is not None:
            checkpoint = training_args.resume_from_checkpoint
        elif last_checkpoint is not None:
            checkpoint = last_checkpoint
        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        metrics = train_result.metrics
        max_train_samples = (
            data_args.max_train_samples if data_args.max_train_samples is not None else len(train_dataset)
        )
        metrics["train_samples"] = min(max_train_samples, len(train_dataset))

        trainer.save_model()  # Saves the tokenizer too for easy upload

        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()

    # Evaluation
    if training_args.do_eval:
        logger.info("*** Evaluate ***")
        metrics = trainer.evaluate(eval_dataset=eval_dataset)

        max_eval_samples = data_args.max_eval_samples if data_args.max_eval_samples is not None else len(eval_dataset)
        metrics["eval_samples"] = min(max_eval_samples, len(eval_dataset))

        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

    # Prediction
    if training_args.do_predict:
        logging.info("*** Test ***")

        def load_ladapters(lad, lang):
            lang_adapter_config = AdapterConfig.load(
                            os.path.join(lad,'adapter_config.json'),
                            non_linearity="gelu",

                        )

            lang_adapter_name = trainer.model.load_adapter(
                lad,
                config=lang_adapter_config,
                load_as=lang,

            )
            return lang_adapter_name

        def load_tadapters(tad, tname):
            task_adapter_config = AdapterConfig.load(
                            config="pfeiffer", non_linearity="gelu", reduction_factor=16
                        )
            task_adapter_name = trainer.model.load_adapter(
                tad,
                config=task_adapter_config,
                load_as=tname,
            )
            return task_adapter_name


        # if adapter_args.train_adapter:
        if data_args.do_predict_adapter:
            lang_1_path = os.path.join(model_args.lang_adapter_path ,model_args.lang_1,'mlm')
            logger.info('lang_1_path:{}'.format(lang_1_path))
            lang_1_name=load_ladapters(lang_1_path, 'lang_1')
            adp_list=[lang_1_name]

            if model_args.lang_2 is not None:
                lang_2_path = os.path.join(model_args.lang_adapter_path ,model_args.lang_2,'mlm')
                logger.info('lang_2_path:{}'.format(lang_2_path))
                lang_2_name=load_ladapters(lang_2_path, 'lang_2')
                adp_list.append(lang_2_name)

            if model_args.lang_3 is not None:
                lang_3_path = os.path.join(model_args.lang_adapter_path ,model_args.lang_3,'mlm')
                logger.info('lang_3_path:{}'.format(lang_3_path))
                lang_3_name=load_ladapters(lang_3_path, 'lang_3')
                adp_list.append(lang_3_name)

            task_path = os.path.join(training_args.output_dir)
            logger.info('task_path:{}'.format(task_path))
            task_adapter_name=load_tadapters(task_path, 'task_j')

            model.add_adapter_fusion(adp_list)
            if model_args.lang_3 is not None and model_args.lang_2 is not None:
                model.active_adapters = ac.Stack(ac.Fuse(lang_1_name, lang_2_name, lang_3_name), task_adapter_name)
            elif model_args.lang_3 is None and model_args.lang_2 is not None:
                model.active_adapters = ac.Stack(ac.Fuse(lang_1_name, lang_2_name), task_adapter_name)
            else:
                model.active_adapters = ac.Stack(lang_1_name, task_adapter_name)
            trainer.model = model.to(training_args.device)

        else:
            if language:
                lang_adapter_config = AdapterConfig.load(
                    config="pfeiffer", non_linearity="gelu", reduction_factor=2, leave_out=leave_out
                )
                model.load_adapter(
                    os.path.join(training_args.output_dir, "best_model", language)
                    if training_args.do_train
                    else adapter_args.load_lang_adapter,
                    config=lang_adapter_config,
                    load_as=language,
                    leave_out=leave_out,
                )
            task_adapter_config = AdapterConfig.load(
                config="pfeiffer", non_linearity="gelu", reduction_factor=16, 
            )
            model.load_adapter(
                os.path.join(training_args.output_dir),
                config=task_adapter_config,
                load_as=task_name
            )
            if language:
                model.set_active_adapters(ac.Stack(lang_adapter_name, task_name))
            else:
                model.set_active_adapters(task_name)
            trainer.model = model.to(training_args.device)

        def get_dataset(data_lang):
            predict_dataset = load_dataset(
            data_args.dataset_name,
            data_lang,
            split="test",
            cache_dir=model_args.cache_dir,
            use_auth_token=True if model_args.use_auth_token else None,)
            print(len(predict_dataset), data_lang,data_args.dataset_name,"==============================")

            if data_args.max_predict_samples is not None:
                max_predict_samples = min(len(predict_dataset), data_args.max_predict_samples)
                predict_dataset = predict_dataset.select(range(max_predict_samples))
            with training_args.main_process_first(desc="prediction dataset map pre-processing"):
                predict_dataset = predict_dataset.map(
                    preprocess_function,
                    batched=True,
                    load_from_cache_file=not data_args.overwrite_cache,
                    desc="Running tokenizer on prediction dataset",
                )

            return predict_dataset

        import json
        with open(data_args.lang_config) as json_file:
            adapter_info = json.load(json_file)
        # adapter_info = ad_data[data_args.family_name]

        output_test_results_file = data_args.result_file
        if trainer.is_world_process_zero():
            writer = open(output_test_results_file, "a")

        text='x'
        count=0

        if model_args.lang_3 is not None:
            langs_to_consider=[model_args.lang_1, model_args.lang_2, model_args.lang_3]
            langs_to_consider=['fra' if item=='fre' else item for item in langs_to_consider]
            langs_to_consider=['jpn' if item=='jap' else item for item in langs_to_consider]
            ad_info={}
            for i,v in adapter_info.items():
                if 'iso-3' in v and v['iso-3'] in langs_to_consider:
                    ad_info[i]=adapter_info[i]
            adapter_info=ad_info
            print('langs to consider: {}'.format(langs_to_consider))
            print('adapter_info: {}'.format(adapter_info))

        for lang, info in adapter_info.items():
            count+=1
            # if count>3:
            #     break
            print(lang, info, count)
            try:
                dataset=get_dataset(lang)

                # predictions, _, metrics = trainer.predict(dataset["test"])

                predictions, labels, metrics = trainer.predict(dataset, metric_key_prefix="predict")

                max_predict_samples = (
                    data_args.max_predict_samples if data_args.max_predict_samples is not None else len(predict_dataset)
                )
                # metrics["predict_samples"] = min(max_predict_samples, len(predict_dataset))

                # trainer.log_metrics("predict", metrics)
                # trainer.save_metrics("predict", metrics)

                # predictions = np.argmax(predictions, axis=1)
                # output_predict_file = os.path.join(training_args.output_dir, "predictions.txt")
                # if trainer.is_world_process_zero():
                #     with open(output_predict_file, "w") as writer:
                #         writer.write("index\tprediction\n")
                #         for index, item in enumerate(predictions):
                #             item = label_list[item]
                #             writer.write(f"{index}\t{item}\n")

                if trainer.is_world_process_zero():
                    logger.info("%s,%s,%s,%s,%s\n" % (data_args.prefix,task_name, 
                        lang, 
                        metrics['predict_accuracy'], 
                        metrics['predict_accuracy']))
                    writer.write("%s,%s,%s,%s,%s\n" % (data_args.prefix,task_name,
                        lang, 
                        metrics['predict_accuracy'], 
                        metrics['predict_accuracy']))
                    print("saved in {}".format(output_test_results_file),"++++++++++++__________+++++++++++__________++++++++")
            except:
                logger.info("#########------------------------error happened in %s----------------########" %(lang))
                writer.write("%s,%s,%s,%s,%s\n" % (data_args.prefix,task_name,  lang, 0, 0))




if __name__ == "__main__":
    main()