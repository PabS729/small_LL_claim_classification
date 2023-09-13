import os
import logging
import argparse
import time
from io import open
from tqdm import tqdm
import torch
import torch.distributed as dist
from torch.nn.functional import softmax
from torch.utils.data import DataLoader, SequentialSampler
from transformers import (RobertaConfig, RobertaForSequenceClassification, RobertaTokenizer,
                          T5Config, T5ForConditionalGeneration, T5Tokenizer,
                          BertConfig, BertForSequenceClassification, BertTokenizer, )
from configs import add_args, set_dist, cleanup, set_seed
from utils import load_nyt_data, load_speeches_sentences_data,  load_sci_data, save_sci_result, SequentialDistributedSampler, save_nyt_result, save_speeches_sentences_result


MODEL_CLASSES = {'roberta': (RobertaConfig, RobertaForSequenceClassification, RobertaTokenizer),
                 't5': (T5Config, T5ForConditionalGeneration, T5Tokenizer),
                 'bert': (BertConfig, BertForSequenceClassification, BertTokenizer),
                 'muppet': (RobertaConfig, RobertaForSequenceClassification, RobertaTokenizer), }
NUM_LABELS = {
    'claimbuster': 3,
    'claim-rank': 2,
    'clef_2022_worth': 2,
    'nyt': 2,
    'sci_abstract': 2,
    'speeches_sentences': 2
}

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)


from datasets import load_dataset
dataset = load_dataset("sem_eval_2018_task_1", "subtask5.english")

labels = [label for label in dataset['train'].features.keys() if label not in ['ID', 'Tweet']]
id2label = {idx:label for idx, label in enumerate(labels)}
label2id = {label:idx for idx, label in enumerate(labels)}

from transformers import AutoTokenizer
import numpy as np

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

def preprocess_data(examples):
  # take a batch of texts
  text = examples["Tweet"]
  # encode them
  encoding = tokenizer(text, padding="max_length", truncation=True, max_length=128)
  # add labels
  labels_batch = {k: examples[k] for k in examples.keys() if k in labels}
  # create numpy array of shape (batch_size, num_labels)
  labels_matrix = np.zeros((len(text), len(labels)))
  # fill numpy array
  for idx, label in enumerate(labels):
    labels_matrix[:, idx] = labels_batch[label]

  encoding["labels"] = labels_matrix.tolist()
  
  return encoding

encoded_dataset = dataset.map(preprocess_data, batched=True, remove_columns=dataset['train'].column_names)

example = encoded_dataset['train'][0]
print(example.keys())
encoded_dataset.set_format("torch")

batch_size = 8
metric_name = "f1"

from transformers import TrainingArguments, Trainer

args = TrainingArguments(
    f"bert-finetuned-sem_eval-english",
    evaluation_strategy = "epoch",
    save_strategy = "epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    num_train_epochs=5,
    weight_decay=0.01,
    load_best_model_at_end=True,
    metric_for_best_model=metric_name,
    #push_to_hub=True,
)