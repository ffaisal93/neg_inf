import logging
import os
import sys
from dataclasses import dataclass, field
from typing import Dict, Optional
import torch

from datasets import load_dataset, load_from_disk, DatasetDict

import transformers.adapters.composition as ac
from preprocessing import preprocess_dataset
from transformers import (
    AdapterConfig,
    AutoAdapterModel,
    AutoConfig,
    AutoTokenizer,
    HfArgumentParser,
    MultiLingAdapterArguments,
    set_seed,
)
logger = logging.getLogger(__name__)


all_lang = ["gub", "est", "bre", "eng", "ben", "kmr", "spa", "bul", "pms", "gle", "nep", "cym", "fin", "hye", "mya", "hin","tel", "tam", "kor", "ell", "hun" ,"heb", "zho", "ara", "swe", "jap", "fre", "deu","rus", "bam", "ewe", "hau", "ibo", "kin", "mos", "pcm", "wol", "yor"]

# all_lang = ["gub", "est", "bre", "eng", "ben", "kmr"]


def load_model():
    model_name = "bert-base-multilingual-cased"

    config = AutoConfig.from_pretrained(
            model_name
        )


    model = AutoAdapterModel.from_pretrained(
            model_name,
            config=config
            )
    return model

def main():

    def load_ladapters(lad, lang, model):
        lang_adapter_config = AdapterConfig.load(
                        os.path.join(lad,'adapter_config.json'),
                        non_linearity="gelu"
                    )

        lang_adapter_name = model.load_adapter(
            lad,
            config=lang_adapter_config,
            load_as=lang
        )
        return lang_adapter_name



    # cos=torch.nn.CosineSimilarity(dim=0)
    # model.freeze_model(True)
    # model.set_active_adapters(adapter_name)

    adapter_path = "../tmp/adapters"
    output_test_results_file="../results/adapter_similarity.txt"
    writer = open(output_test_results_file, "w")

    cos=torch.nn.CosineSimilarity(dim=0)
    pdist = torch.nn.PairwiseDistance(p=2)
    for ind1,l1 in enumerate(all_lang):
        for ind2,l2 in enumerate(all_lang):
          if ind1>ind2:
            model1 = load_model()
            model2 = load_model()
            
            print(l1, l2)
            lang_1_path = os.path.join(adapter_path, l1,'mlm')
            logger.info('lang_1_path:{}'.format(lang_1_path))
            adapter_name1=load_ladapters(lang_1_path, l1, model1)
            model1.train_adapter([adapter_name1])

            lang_2_path = os.path.join(adapter_path, l2,'mlm')
            logger.info('lang_1_path:{}'.format(lang_1_path))
            adapter_name2=load_ladapters(lang_2_path, l2, model2)
            model2.train_adapter([adapter_name2])

            x1=torch.tensor(())
            for name,param in model1.named_parameters():
                if param.requires_grad==True:
                    # print(name)
                    x1 = torch.cat((x1,param.data.flatten()))

            x2=torch.tensor(())
            for name,param in model2.named_parameters():
                if param.requires_grad==True:
                    # print(name)
                    x2 = torch.cat((x2,param.data.flatten()))
            cos_dist = 1-cos(x1,x2)
            pdist_val = pdist(torch.unsqueeze(x1, 0),torch.unsqueeze(x2, 0))

            print(cos_dist, pdist_val)
            writer.write("%s,%s,%s, %s\n" % (l1,l2, cos_dist.item(), pdist_val.item()))

            model1.delete_adapter(adapter_name1)
            model2.delete_adapter(adapter_name2)
            # model2=0



if __name__ == "__main__":
    main()




