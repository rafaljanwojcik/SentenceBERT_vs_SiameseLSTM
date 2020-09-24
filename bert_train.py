import os
from pathlib import Path
import logging
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from time import time
from datetime import date
import argparse
import matplotlib.pyplot as plt
from transformers import BertForSequenceClassification, Trainer, TrainingArguments, BertTokenizer

from modules.data import ImportData
from modules.models import SiameseBERT2
from modules.utils import collate_fn_bert, setup_logger, compute_metrics, compute_metrics_siamBERT, get_quora_huggingface
from modules.train import CustomTrainer

import transformers
transformers.logging.set_verbosity_info()

path = Path('./logs/data/')
if not (path/'dataset.csv').exists():
    get_quora_huggingface(path)

today = str(date.today())
path = Path(f'./logs/train_job_{today}/')
emb_path = Path('./logs/embeddings')
data_path = Path('./logs/data')

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-model_name", "--model_name", type=str, help="Name of trained model. Needed only for correct logs output", default='bert')  
    parser.add_argument("-log", "--logdir", type=str, help="Directory to save all downloaded files, and model checkpoints.", default=path)  
    parser.add_argument("-df", "--data_file", type=str, help="Path to dataset.", default=data_path/"dataset.csv")
    parser.add_argument("-s", "--split_seed", type=int, help="Seed for splitting the dataset.", default=44)
    parser.add_argument("-b", "--batch_size", type=int, help="Batch Size.", default=16)
    parser.add_argument("-epo", "--n_epoch", type=int, help="Number of epochs.", default=4)
    parser.add_argument("-bert_cls", "--bert_cls", type=str, help="Type of BERT trained (classificator, siamese).", default='siamese')
    parser.add_argument("-bert_backbone", "--bert_backbone", type=str, help="Either path to the model, or name of the BERT model that should be used, compatible with HuggingFace Transformers.", default='bert-base-uncased')

    args = parser.parse_args('')
    args.logdir = args.logdir/(args.bert_cls+'_'+args.model_name)
    model_path = args.logdir/'best_model/'
    if not args.logdir.exists():
        os.makedirs(args.logdir)

    logger = setup_logger(str(args.logdir/'logs.log'))
    logger.info("Begining job. All files and logs will be saved at: {}".format(args.logdir))


    logger.info('Reading Dataset and splitting into train and test datasets with seed: {}'.format(args.split_seed))
    data = ImportData(str(args.data_file))
    data.train_test_split(seed=args.split_seed)


    logger.info('')
    logger.info('Number of training samples        :{}'.format(len(data.train)))
    logger.info('Number of validation samples      :{}'.format(len(data.test)))
    logger.info('')

    model = SiameseBERT2.from_pretrained("bert-base-uncased") if args.bert_cls=='siamese' else BertForSequenceClassification.from_pretrained('bert-base-uncased')
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    logging_num = data.train.shape[0]/(torch.cuda.device_count() * args.batch_size)


    training_args = TrainingArguments(
        output_dir=str(args.logdir/'results'),          # output directory
        overwrite_output_dir = True,
        do_train=True,
        do_eval=True,
        save_steps= logging_num//2, #logging_num/4,
        save_total_limit = 8,
        eval_steps = 500,#logging_num//10,
        logging_steps = 500,#logging_num//10,
        evaluation_strategy="steps",
        logging_first_step = True,
        num_train_epochs=4,              # total # of training epochs
        per_device_train_batch_size=args.batch_size,  # batch size per device during training
        per_device_eval_batch_size=64,   # batch size for evaluation
        warmup_steps=500,                # number of warmup steps for learning rate scheduler
        weight_decay=0.01, # strength of weight decay
        logging_dir=str(args.logdir/'logs'),            # directory for storing logs
    )

    trainer_class = CustomTrainer if args.bert_cls == 'siamese' else Trainer
    trainer_args = {'model':model, 'args':training_args, 
                    'data_collator':lambda x: collate_fn_bert(x, tokenizer, args.bert_cls),
                    'train_dataset':data.train.values,
                    'eval_dataset':data.test.values, 
                    'compute_metrics':compute_metrics_siamBERT if args.bert_cls == 'siamese' else compute_metrics}
    if args.bert_cls == 'siamese':
        trainer_args['logger'] = logger
    trainer = trainer_class(**trainer_args)