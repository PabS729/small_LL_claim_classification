import transformers
from transformers.utils import send_example_telemetry
from torch.utils.data import random_split
import numpy as np
import evaluate
import random
import pandas as pd
import os
from misc import *
from IPython.display import display, HTML
from transformers import AutoTokenizer, DistilBertForSequenceClassification, Trainer, TrainingArguments, AutoModelForSequenceClassification

dataset_locs = {
    "clef_2021": "./data/sub1/",
    "clef_2022": "./data/CLEF_2022_ALAM",
    "claimbuster":"./data/claimbuster",
    "lesa":"./data/lesa-twitter",
    'speeches':'./data/gold',
    'silver_speech':'./data/silver+bronze'
}

token = open("../token.txt")
tok = token.readline()
task = "clef_2021"
#model_checkpoint = "distilbert-base-uncased"
model_checkpoint = "roberta-base"
batch_size = 16
max_len = 512
metric = evaluate.load("accuracy")

ls_gold = os.listdir(dataset_locs['speeches'])
ls_silver = os.listdir(dataset_locs['silver_speech'])
# actual_task = "mnli" if task == "mnli-mm" else task
# dataset = load_dataset("glue", actual_task)
# metric = load_metric('glue', "cola")
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)


dataset_locs[task] + "dataset_dev_v1_english.tsv"

# concatenated_sent = "[CLS]" + sent1 + "[SEP]" + sent2 + "[SEP]" + sent3
# tokenized_sent = tokenizer(concatenated_sent, truncate=True, pading=True)
trains = []
train_labs = []
silver_tr = []
silver_label = []
for i in range(0, 4):
    new_train, train_labels = read_data(dataset_locs['speeches'] + '/' + ls_gold[i])
    trains += new_train
    train_labs += list(train_labels)

print(len(trains), len(train_labs))
for j in range(10):
    print(trains[j], train_labs[j])
for k in range(0, 2):
    silver_st, silver_labs = preprocess_silver_label(dataset_locs['silver_speech'] + '/' + ls_silver[k])
    silver_tr += silver_st
    silver_label += list(silver_labs)

# new_test, test_labels = read_test_data("./data/clef-2022_labeling_golden.xlsx")
new_test, test_labels = read_test_data("./data/human_eval_merged.xlsx")
train_encodings = tokenizer(trains, truncation=True, padding=True)
test_encodings = tokenizer(new_test, truncation=True, padding=True)


mixed_st = trains + silver_tr
mixed_labs = train_labs + silver_label
full_dataset = newDataset(train_encodings, train_labs)

train_size = int(0.8 * len(full_dataset))
val_size = len(full_dataset) - train_size
train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
test_dataset = newDataset(test_encodings, test_labels)
print("loaded datasets")

num_labels = 1
model = AutoModelForSequenceClassification.from_pretrained(model_checkpoint)
metric_name = "accuracy"
model_name = model_checkpoint.split("/")[-1]

args = TrainingArguments(
    f"output_dir/{model_name}-finetuned-{task}-silver",
    evaluation_strategy = "epoch",
    save_strategy = "epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    num_train_epochs=10,
    weight_decay=0.01,
    load_best_model_at_end=True,
    metric_for_best_model=metric_name,
    push_to_hub=False,
)

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    if task != "stsb":
        predictions = np.argmax(predictions, axis=1)
    else:
        predictions = predictions[:, 0]
    return metric.compute(predictions=predictions, references=labels)
trainer = Trainer(
    model,
    args,
    train_dataset=full_dataset,
    eval_dataset=test_dataset,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
)

trainer.train()
trainer.evaluate()
token.close()