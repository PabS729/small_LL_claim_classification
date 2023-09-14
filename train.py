import numpy as np
import pandas as pd
from sklearn import metrics
import transformers
import torch
from torch.optim import AdamW
from torch.utils.data import Dataset, DataLoader, RandomSampler, SequentialSampler
import os
import logging
import argparse
import time
from io import open
import torch
import torch.distributed as dist
from torch.nn.functional import softmax
from torch.utils.data import DataLoader, SequentialSampler
from configs import *
from utils import * 
from misc import *
from transformers import (RobertaConfig, RobertaForSequenceClassification, RobertaTokenizer,
                          T5Config, T5ForConditionalGeneration, T5Tokenizer,
                          BertConfig, BertForSequenceClassification, BertTokenizer, )


MODEL_CLASSES = {'roberta': (RobertaConfig, RobertaForSequenceClassification, RobertaTokenizer),
                 't5': (T5Config, T5ForConditionalGeneration, T5Tokenizer),
                 'bert': (BertConfig, BertForSequenceClassification, BertTokenizer),
                 'muppet': (RobertaConfig, RobertaForSequenceClassification, RobertaTokenizer), }
NUM_LABELS = 2
MAX_LEN = 200
TRAIN_BATCH_SIZE = 8
VALID_BATCH_SIZE = 4
EPOCHS = 1
LEARNING_RATE = 1e-05

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)

from torch import cuda
device = 'cuda' if cuda.is_available() else 'cpu'


def loss_fn(outputs, targets):
    return torch.nn.BCEWithLogitsLoss()(outputs, targets)


def main():
    parser = argparse.ArgumentParser()
    args = add_args(parser)
    if args.local_rank in [-1, 0]:
        t0 = time.time()
        logger.info(args)
    set_dist(args)
    if args.local_rank == -1 or args.no_cuda:
        logging.getLogger().setLevel(logging.INFO)
    else:
        logging.getLogger().setLevel(logging.INFO if dist.get_rank() in [-1, 0] else logging.WARN)
    set_seed(args)

    df = pd.read_csv("./data/train.csv")
    df['list'] = df[df.columns[2:]].values.tolist()
    new_df = df[['comment_text', 'list']].copy()
    new_df.head()

    config_class, model_class, tokenizer_class = MODEL_CLASSES[args.model_type]
    config = config_class.from_pretrained(args.config_name if args.config_name else args.model_name_or_path,
                                          num_labels=NUM_LABELS)
    model = model_class.from_pretrained(args.model_name_or_path, config=config, cache_dir='cache',
                                        local_files_only=True)
    tokenizer = tokenizer_class.from_pretrained(args.model_name_or_path, cache_dir='cache', local_files_only=True)

            # prepare training dataloader
    train_dataset = CustomDataset(df, tokenizer, MAX_LEN)
    logger.info("Training Data Counts: " + str(len(train_dataset)))
    train_example_num = len(train_dataset)

    eval_dataset = CustomDataset(df, tokenizer, MAX_LEN)
    logger.info("Evaluation Data Counts: " + str(len(eval_dataset)))

    train_sampler = RandomSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.train_batch_size,
                                          num_workers=4, pin_memory=True)
    eval_sampler = RandomSampler(eval_dataset)
    eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args.train_batch_size,
                                          num_workers=4, pin_memory=True)

    save_steps = max(len(train_dataloader), 1)

        # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
            {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
             'weight_decay': args.weight_decay},
            {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
             'weight_decay': 0.0}
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    num_train_optimization_steps = args.num_train_epochs * len(train_dataloader)
    scheduler = get_linear_schedule_with_warmup(optimizer,
                                                    num_warmup_steps=args.warmup_steps,
                                                    num_training_steps=num_train_optimization_steps)

    if args.cont:
        if args.local_rank == -1:
            map_location = "cpu"
        else:
            map_location = "cuda:%d" % args.local_rank
        optimizer_state = torch.load(os.path.join(args.output_dir, 'checkpoint-last/optimizer.pt'),
                                     map_location=map_location)
        scheduler_state = torch.load(os.path.join(args.output_dir, 'checkpoint-last/scheduler.pt'),
                                         map_location=map_location)
        optimizer.load_state_dict(optimizer_state)
        scheduler.load_state_dict(scheduler_state)

            
    # Start training
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", train_example_num)
    logger.info("  Batch size = %d", args.train_batch_size)
    logger.info("  Batch num = %d", math.ceil(train_example_num / args.train_batch_size))
    logger.info("  Num epoch = %d", args.num_train_epochs)

    if args.cont:
        with open(os.path.join(args.output_dir, "checkpoint-last/training_state.json"), 'r') as f:
            training_state = json.load(f)
    else:
        training_state = {}
        training_state['global_step'] = 0
        training_state['best_acc'] = 0
        training_state['best_macro_f1'] = 0
        training_state['not_acc_inc_cnt'] = 0
        training_state['tr_loss'] = 0
        training_state['is_early_stop'] = False
        training_state['epoch'] = 0
        training_state['dev_acc'] = []
        training_state['dev_macro_f1'] = []

    start_epoch = training_state['epoch']
    slm = SLMClass(model)
    train_eval(start_epoch, EPOCHS, slm, train_dataloader, device, optimizer, loss_fn, logger, save_steps, eval_dataloader)




for epoch in range(EPOCHS):
    outputs, targets = validation(epoch)
    outputs = np.array(outputs) >= 0.5
    accuracy = metrics.accuracy_score(targets, outputs)
    f1_score_micro = metrics.f1_score(targets, outputs, average='micro')
    f1_score_macro = metrics.f1_score(targets, outputs, average='macro')
    print(f"Accuracy Score = {accuracy}")
    print(f"F1 Score (Micro) = {f1_score_micro}")
    print(f"F1 Score (Macro) = {f1_score_macro}")