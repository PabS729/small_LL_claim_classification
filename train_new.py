import transformers
from transformers.utils import send_example_telemetry
from datasets import load_dataset, load_metric
import numpy as np
import datasets
import random
import pandas as pd
from misc import *
from IPython.display import display, HTML
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer

task_dict = {
    "clef_2021": "tweet-text",
    "clef_2022": "tweet-text",
    "claimbuster":"Text",
    "lesa":"text",
}

dataset_locs = {
    "clef_2021": "./data/CLEF_2021",
    "clef_2022": "./data/CLEF_2022_ALAM",
    "claimbuster":"./data/claimbuster",
    "lesa":"./data/lesa-twitter",
}


task = "lesa"
model_checkpoint = "bert-base-uncased"
batch_size = 16
max_len = 256

actual_task = "mnli" if task == "mnli-mm" else task
dataset = load_dataset("glue", actual_task)
metric = load_metric('glue', actual_task)
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint, use_fast=True)

dataset_file_train = pd.DataFrame.from_csv(dataset_locs[task], separator = "\t")
dataset_file_eval = pd.DataFrame.from_csv(dataset_locs[task], separator = "\t")
dataset_file_test = pd.DataFrame.from_csv(dataset_locs[task], separator = "\t")
train_dataset = CustomDataset(dataset_file_train, tokenizer, max_len)
eval_dataset = CustomDataset(dataset_file_eval, tokenizer, max_len)
test_dataset = CustomDataset(dataset_file_test, tokenizer, max_len)


def preprocess_function(examples):
    return tokenizer(examples[task_dict[task]], truncation=True)

encoded_dataset = dataset.map(preprocess_function, batched=True)

num_labels = 1
model = AutoModelForSequenceClassification.from_pretrained(model_checkpoint, num_labels=num_labels)

metric_name = "accuracy"
model_name = model_checkpoint.split("/")[-1]

args = TrainingArguments(
    f"{model_name}-finetuned-{task}",
    evaluation_strategy = "epoch",
    save_strategy = "epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    num_train_epochs=5,
    weight_decay=0.01,
    load_best_model_at_end=True,
    metric_for_best_model=metric_name,
    push_to_hub=True,
)

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    if task != "stsb":
        predictions = np.argmax(predictions, axis=1)
    else:
        predictions = predictions[:, 0]
    return metric.compute(predictions=predictions, references=labels)

validation_key = "validation_mismatched" if task == "mnli-mm" else "validation_matched" if task == "mnli" else "validation"
trainer = Trainer(
    model,
    args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics
)

trainer.train()
trainer.evaluate()