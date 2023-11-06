import transformers
from transformers.utils import send_example_telemetry
from torch.utils.data import random_split
import numpy as np
import evaluate
from configs import set_seed
import argparse
import logging
import random
import pandas as pd
from torch import inf 
import os
from misc import *
from IPython.display import display, HTML
from transformers import AutoTokenizer, DistilBertForSequenceClassification, Trainer, TrainingArguments, AutoModelForSequenceClassification
import wandb
dataset_locs = {
    "clef_2021": "./data/sub1/",
    "clef_2022": "./data/CLEF_2022_ALAM",
    "claimbuster":"./data/claimbuster",
    "lesa":"./data/lesa-twitter",
    'speeches':'./data/gold',
    'silver_speech':'./data/silver+bronze'
}


parser = argparse.ArgumentParser()
parser.add_argument("--use_gold", default=4)
parser.add_argument("--use_silver", default=0)
parser.add_argument("--use_bronze", default=0)
parser.add_argument("--epochs", default=5)
parser.add_argument("--batch_size", default=16)
parser.add_argument("--seed", default=123)
parser.add_argument("--model", default="distilbert-base-uncased")

args = parser.parse_args()
set_seed(args)
logging.basicConfig(
    filemode='a',
    filename="log_gold_%d_silver_%d_bronze_%d_epochs_%d_seed_%d",
    format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)
logger.info(args)
# token = open("../token.txt")
# tok = token.readline()
task = "clef_2021"
#model_checkpoint = "distilbert-base-uncased"
model_checkpoint = args.model
batch_size = args.batch_size
max_len = 512
metric = evaluate.load("accuracy")

ls_gold = os.listdir(dataset_locs['speeches'])

# actual_task = "mnli" if task == "mnli-mm" else task
# dataset = load_dataset("glue", actual_task)
# metric = load_metric('glue', "cola")
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)

dataset_locs["clef_2021"] + "dataset_dev_v1_english.tsv"


# concatenated_sent = "[CLS]" + sent1 + "[SEP]" + sent2 + "[SEP]" + sent3
# tokenized_sent = tokenizer(concatenated_sent, truncate=True, pading=True)
trains = []
train_labs = []
silver_tr = []
silver_label = []
bronze_tr = []
bronze_label = []

silv_df = pd.read_excel("silver_labels.xlsx")
broz_df = pd.read_excel("bronze_labels.xlsx")

silver_tr = list(silv_df["SENTENCES"])
silver_label = list(silv_df["Golden"])

bronze_tr = list(broz_df["SENTENCES"])
bronze_label = list(broz_df["Golden"])

for i in range(0, len(ls_gold)):
    new_train, train_labels = read_data(dataset_locs['speeches'] + '/' + ls_gold[i])
    trains += new_train
    train_labs += train_labels


gold_len = args.use_gold
if gold_len != 0:
    gold_len = len(trains)

gold_st = trains[:gold_len]
gold_label = train_labs[:gold_len]

silver_st = silver_tr[:int(args.use_silver)]
silver_label = silver_label[:int(args.use_silver)]

print(silver_label[:10])
bronze_st = bronze_tr[:int(args.use_bronze)]
bronze_label = bronze_label[:int(args.use_bronze)]

# new_test, test_labels = read_test_data("./data/clef-2022_labeling_golden.xlsx")
new_test, test_labels = read_test_data("./data/human_eval_merged.xlsx")



mixed_st = gold_st + silver_st
mixed_labs = gold_label + silver_label
mixed_encodings = tokenizer(mixed_st, truncation=True, padding=True)
test_encodings = tokenizer(new_test, truncation=True, padding=True)
full_dataset = newDataset(mixed_encodings, mixed_labs)

# train_size = int(0.8 * len(full_dataset))
# val_size = len(full_dataset) - train_size
# train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
test_dataset = newDataset(test_encodings, test_labels)
print("loaded datasets")

num_labels = 1
model = AutoModelForSequenceClassification.from_pretrained(model_checkpoint)
metric_name = "accuracy"
model_name = model_checkpoint.split("/")[-1]

args_train = TrainingArguments(
    # f"output_dir/{model_name}",
    f"output_dir/{model_name}-gold-{gold_len}-silver-{args.use_silver}-bronze-{args.use_bronze}-epochs-{args.epochs}-seeds-{args.seed}",
    evaluation_strategy = "epoch",
    save_strategy = "epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    num_train_epochs=args.epochs,
    weight_decay=0.01,
    load_best_model_at_end=True,
    metric_for_best_model=metric_name,
    push_to_hub=False,
    seed = args.seed
)
print("okk")
def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    if task != "stsb":
        predictions = np.argmax(predictions, axis=1)
    else:
        predictions = predictions[:, 0]
    return metric.compute(predictions=predictions, references=labels)
trainer = Trainer(
    model,
    args_train,
    train_dataset=full_dataset,
    eval_dataset=test_dataset,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
)

print("start training")
trainer.train()
trainer.evaluate()