import os
import logging
import argparse
import math
from contextlib import nullcontext
import time
import json
import numpy as np
from io import open
from tqdm import tqdm
import torch
import torch.distributed as dist
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader, Dataset, SequentialSampler, RandomSampler, TensorDataset
from torch.utils.data.distributed import DistributedSampler
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from torch.optim import AdamW
from transformers import (get_linear_schedule_with_warmup,
                          RobertaConfig, RobertaForSequenceClassification, RobertaTokenizer,
                          T5Config, T5ForConditionalGeneration, T5Tokenizer,
                          BertConfig, BertForSequenceClassification, BertTokenizer, )
from configs import add_args, set_dist, cleanup, set_seed
from utils import get_elapse_time, load_general_data, load_detection_data, load_claimrank_data, load_claimbuster_data, \
    save_checkpoint, SequentialDistributedSampler, load_retro_data

MODEL_CLASSES = {
    'roberta': (RobertaConfig, RobertaForSequenceClassification, RobertaTokenizer),
    't5': (T5Config, T5ForConditionalGeneration, T5Tokenizer),
    'bert': (BertConfig, BertForSequenceClassification, BertTokenizer),
}
NUM_LABELS = {
    'claimbuster': 3,
    'claim-rank': 2,
    'clef_2022_worth': 2,
    'clef_2022_detect': 2,
    'mt': 2,
    'oc': 2,
    'pe': 2,
    'vg': 2,
    'wd': 2,
    'wtp': 2,
    'lesa-twitter': 2,
    'mix_detection': 2,
    'retro_mix_detection': 2,
}

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)




def distributed_concat(tensor, num_total_examples):
    output_tensors = [tensor.clone() for _ in range(torch.distributed.get_world_size())]
    torch.distributed.all_gather(output_tensors, tensor)
    concat = torch.cat(output_tensors, dim=0)
    # truncate the dummy elements added by SequentialDistributedSampler
    return concat[:num_total_examples]


def evaluate(args, model, eval_examples, eval_data, tokenizer, compute_f1=False, test=False):
    if args.local_rank == -1:
        eval_sampler = SequentialSampler(eval_data)
        eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=args.eval_batch_size,
                                     num_workers=4, pin_memory=True)
    else:
        eval_sampler = SequentialDistributedSampler(eval_data, batch_size=args.eval_batch_size)
        eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=args.eval_batch_size,
                                     num_workers=0, pin_memory=True, drop_last=False)

    # Eval!
    logger.info("***** Running evaluation *****")
    logger.info("  Num examples = %d", len(eval_examples))
    logger.info("  Num batches = %d", len(eval_dataloader))
    logger.info("  Batch size = %d", args.eval_batch_size)
    nb_eval_steps = 0
    model.eval()
    pred_ids = []
    labels = []
    for idx, batch in enumerate(tqdm(eval_dataloader, total=len(eval_dataloader), desc="Evaluating")):
        inputs = {k: v.to(args.device) for k, v in batch.items()}
        with torch.no_grad():
            if args.model_type in ['t5']:
                model_to_generate = model.module if hasattr(model, 'module') else model
                predictions = model_to_generate.generate(
                    inputs['input_ids'],
                    attention_mask=inputs['input_ids'].ne(tokenizer.pad_token_id),
                    use_cache=True,
                    max_new_tokens=5,
                    num_beams=5,
                )
                pred_ids.append(predictions)
                labels.append(inputs['labels'])
            else:
                logit = model(**inputs).logits
                pred_ids.append(logit)
                labels.append(inputs['labels'])
        nb_eval_steps += 1

    if args.local_rank == -1 or args.no_cuda:
        pred_ids = torch.concat(pred_ids, dim=0)
        labels = torch.concat(labels, dim=0)
    else:
        pred_ids = distributed_concat(torch.concat(pred_ids, dim=0), len(eval_sampler.dataset))
        labels = distributed_concat(torch.concat(labels, dim=0), len(eval_sampler.dataset))
    if args.model_type in ['t5']:
        pred_ids = list(pred_ids.cpu().numpy())
        labels = list(labels.cpu().numpy())
        pred_nls = [tokenizer.decode(id, skip_special_tokens=True, clean_up_tokenization_spaces=False).strip() for id in
                    pred_ids]
        with tokenizer.as_target_tokenizer():
            label_nls = [tokenizer.decode(id, skip_special_tokens=True, clean_up_tokenization_spaces=False) for id in
                         labels]
        logger.info("Hypothesis example 1: " + pred_nls[0])
        logger.info("Reference example 1: " + label_nls[0])
        eval_acc = np.mean([int(p == g) for p, g in zip(pred_nls, label_nls)])
        if compute_f1:
            # pred_nums = [int(p) for p in pred_nls]
            # label_nums = [int(l) for l in label_nls]
            eval_macro_f1 = f1_score(label_nls, pred_nls, average='macro')
            if test:
                label_set = set(label_nls)
                metrics_for_labels = {}
                for l in label_set:
                    metrics_for_labels[l] = {
                        'f1': round(f1_score(label_nls, pred_nls, average='macro', labels=[l]), 4),
                        'precision': round(precision_score(label_nls, pred_nls, average='macro', labels=[l]), 4),
                        'recall': round(recall_score(label_nls, pred_nls, average='macro', labels=[l]), 4),
                    }
    else:
        pred_ids = pred_ids.cpu().numpy()
        labels = labels.cpu().numpy()
        preds = pred_ids.argmax(axis=1)
        eval_acc = np.mean(labels == preds)
        if compute_f1:
            eval_macro_f1 = f1_score(labels, preds, average='macro')
            if test:
                label_set = set(labels)
                metrics_for_labels = {}
                for l in label_set:
                    metrics_for_labels[l] = {
                        'f1': round(f1_score(labels, preds, average='macro', labels=[l]), 4),
                        'precision': round(precision_score(labels, preds, average='macro', labels=[l]), 4),
                        'recall': round(recall_score(labels, preds, average='macro', labels=[l]), 4),
                    }

    result = {
        "eval_acc": round(eval_acc, 4),
    }
    if compute_f1:
        result["eval_macro_f1"] = round(eval_macro_f1, 4)
        if test:
            result["eval_metrics_for_labels"] = metrics_for_labels
    logger.info("***** Eval results *****")
    for key in sorted(result.keys()):
        if key != "eval_metrics_for_labels":
            logger.info("  %s = %s", key, str(round(result[key], 4)))

    return result


def load_and_cache_data(args, tokenizer, split_tag):
    func_dict = {
        'claimbuster': load_claimbuster_data,
        'claim-rank': load_claimrank_data,
        'clef_2022_worth': load_general_data,
        'mt': load_general_data,
        'oc': load_general_data,
        'pe': load_general_data,
        'vg': load_general_data,
        'wd': load_general_data,
        'wtp': load_general_data,
        'lesa-twitter': load_general_data,
        'mix_detection': load_detection_data,
        'retro_mix_detection': load_retro_data,
    }
    return func_dict[args.task](args, tokenizer, split_tag)


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

    # Build model
    config_class, model_class, tokenizer_class = MODEL_CLASSES[args.model_type]
    config = config_class.from_pretrained(args.config_name if args.config_name else args.model_name_or_path,
                                          num_labels=NUM_LABELS[args.task])
    model = model_class.from_pretrained(args.model_name_or_path, config=config, cache_dir='cache',
                                        local_files_only=True)
    tokenizer = tokenizer_class.from_pretrained(args.model_name_or_path, cache_dir='cache', local_files_only=True)
    if args.cont:
        model_file = os.path.join(args.output_dir, "checkpoint-last/pytorch_model.bin")
        model.load_state_dict(torch.load(model_file))
    model.to(args.device)
    if args.local_rank == -1 and args.n_gpu > 1:
        # for DataParallel
        model = torch.nn.DataParallel(model)
    if args.local_rank != -1 and args.n_gpu > 1:
        # for DistributedDataParallel
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank],
                                                          output_device=args.local_rank)
    if args.local_rank in [-1, 0]:
        fa = open(os.path.join(args.output_dir, 'summary.log'), 'a+')

    if args.do_train:
        if args.local_rank in [-1, 0]:
            summary_fn = './tensorboard/{}'.format('/'.join(args.output_dir.split('/')[1:]))
            tb_writer = SummaryWriter(summary_fn)

        # prepare training dataloader
        train_examples, train_dataset = load_and_cache_data(args, tokenizer, split_tag='train')
        logger.info("Training Data Counts: " + str(len(train_dataset)))
        train_example_num = len(train_dataset)

        if args.local_rank == -1:
            train_sampler = RandomSampler(train_dataset)
            train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.train_batch_size,
                                          num_workers=4, pin_memory=True)
        else:
            train_sampler = DistributedSampler(train_dataset)
            train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.train_batch_size,
                                          num_workers=0, pin_memory=True, drop_last=False)
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
        for cur_epoch in range(start_epoch, int(args.num_train_epochs)):
            if args.local_rank != -1:
                train_dataloader.sampler.set_epoch(cur_epoch)
            bar = tqdm(train_dataloader, total=len(train_dataloader), desc="Training")
            nb_tr_steps = training_state['global_step'] * args.gradient_accumulation_steps
            nb_tr_examples = 0
            model.train()
            for step, batch in enumerate(bar):
                inputs = {k: v.to(args.device) for k, v in batch.items()}
                mcontext = model.no_sync if args.local_rank != -1 and nb_tr_steps % args.gradient_accumulation_steps != 0 else nullcontext
                with mcontext():
                    loss = model(**inputs).loss

                    if args.gradient_accumulation_steps > 1:
                        loss = loss / args.gradient_accumulation_steps
                    training_state['tr_loss'] += loss.item()

                    nb_tr_examples += inputs['input_ids'].size(0)
                    nb_tr_steps += 1
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

                if nb_tr_steps % args.gradient_accumulation_steps == 0:
                    # Update parameters
                    optimizer.step()
                    optimizer.zero_grad()
                    scheduler.step()
                    training_state['global_step'] += 1
                    train_loss = round(training_state['tr_loss'] * args.gradient_accumulation_steps / nb_tr_steps, 4)
                    bar.set_description("[{}] Train loss {}".format(cur_epoch, round(train_loss, 3)))

                if (step + 1) % save_steps == 0 and args.do_eval:
                    logger.info("***** CUDA.empty_cache() *****")
                    torch.cuda.empty_cache()

                    eval_examples, eval_data = load_and_cache_data(args, tokenizer, 'dev')

                    result = evaluate(args, model, eval_examples, eval_data, tokenizer, compute_f1=True)
                    eval_acc = result['eval_acc']
                    eval_macro_f1 = result['eval_macro_f1']

                    training_state['dev_acc'].append({'epoch': training_state['epoch'], 'dev_acc': eval_acc})
                    training_state['dev_macro_f1'].append(
                        {'epoch': training_state['epoch'], 'dev_macro_f1': eval_macro_f1})
                    if args.local_rank in [-1, 0]:
                        tb_writer.add_scalar('dev_acc', round(eval_acc, 4), cur_epoch)
                        tb_writer.add_scalar('dev_macro_f1', round(eval_macro_f1, 4), cur_epoch)

                    if eval_macro_f1 > training_state['best_macro_f1']:
                        logger.info("  Best ma-f1: %s", round(eval_macro_f1, 4))
                        logger.info("  " + "*" * 20)
                        if args.local_rank in [-1, 0]:
                            fa.write("[%d] Best ma-f1 changed into %.4f\n" % (cur_epoch, round(eval_macro_f1, 4)))
                        training_state['best_macro_f1'] = eval_macro_f1
                        output_dir = os.path.join(args.output_dir, 'checkpoint-best-ma-f1')
                        if args.local_rank in [-1, 0]:
                            if not os.path.exists(output_dir):
                                os.makedirs(output_dir)
                            model_to_save = model.module if hasattr(model, 'module') else model
                            output_model_file = os.path.join(output_dir, "pytorch_model.bin")
                            torch.save(model_to_save.state_dict(), output_model_file)
                            logger.info("Save the best ma-f1 model into %s", output_model_file)

                    if eval_acc > training_state['best_acc']:
                        training_state['not_acc_inc_cnt'] = 0
                        logger.info("  Best acc: %s", round(eval_acc, 4))
                        logger.info("  " + "*" * 20)
                        if args.local_rank in [-1, 0]:
                            fa.write("[%d] Best acc changed into %.4f\n" % (cur_epoch, round(eval_acc, 4)))
                        training_state['best_acc'] = eval_acc
                        # Save best checkpoint for best accuracy
                        output_dir = os.path.join(args.output_dir, 'checkpoint-best-acc')
                        if args.local_rank in [-1, 0]:
                            if not os.path.exists(output_dir):
                                os.makedirs(output_dir)
                            model_to_save = model.module if hasattr(model, 'module') else model
                            output_model_file = os.path.join(output_dir, "pytorch_model.bin")
                            torch.save(model_to_save.state_dict(), output_model_file)
                            logger.info("Save the best acc model into %s", output_model_file)
                    else:
                        training_state['not_acc_inc_cnt'] += 1
                        logger.info("acc does not increase for %d epochs", training_state['not_acc_inc_cnt'])
                        if training_state['not_acc_inc_cnt'] > args.patience:
                            if args.local_rank in [-1, 0]:
                                logger.info("Early stop as acc do not increase for %d times",
                                            training_state['not_acc_inc_cnt'])
                            if args.local_rank in [-1, 0]:
                                fa.write("[%d] Early stop as not_acc_inc_cnt=%d\n" % (
                                    cur_epoch, training_state['not_acc_inc_cnt']))
                            training_state['is_early_stop'] = True
                            break

                    # save last checkpoint
                    last_output_dir = os.path.join(args.output_dir, 'checkpoint-last')

                    if args.save_last_checkpoints and args.local_rank in [-1, 0]:
                        if not os.path.exists(last_output_dir):
                            os.makedirs(last_output_dir)
                        model_to_save = model.module if hasattr(model, 'module') else model
                        output_model_file = os.path.join(last_output_dir, "pytorch_model.bin")
                        output_optimizer_file = os.path.join(last_output_dir, "optimizer.pt")
                        output_scheduler_file = os.path.join(last_output_dir, "scheduler.pt")
                        save_checkpoint(args.local_rank, training_state, optimizer, scheduler, model_to_save,
                                        output_model_file,
                                        output_optimizer_file, output_scheduler_file, last_output_dir)
                        logger.info("Save the last model into %s", output_model_file)

                model.train()
            if training_state['is_early_stop']:
                break

            logger.info("***** CUDA.empty_cache() *****")
            torch.cuda.empty_cache()
            training_state['epoch'] += 1

        if args.local_rank in [-1, 0]:
            tb_writer.close()

    if args.do_test:
        logger.info("  " + "***** Testing *****")
        logger.info("  Batch size = %d", args.eval_batch_size)

        for criteria in ['best-acc', 'best-ma-f1']:
            file = os.path.join(args.output_dir, 'checkpoint-{}/pytorch_model.bin'.format(criteria))
            logger.info("Reload model from {}".format(file))
            if hasattr(model, 'module'):
                model.module.load_state_dict(torch.load(file))
            else:
                model.load_state_dict(torch.load(file))

            model.to(args.device)

            if args.task == 'mix_detection':
                testsets = {'0': 'claimbuster', '3': 'clef_2022', '4': 'newsclaim', '5': 'lesa-twitter'}
                for test_data_id in args.test_set.split(','):
                    test_examples, test_data = load_detection_data(args, tokenizer, 'test', test_data=test_data_id)
                    result = evaluate(args, model, test_examples, test_data, tokenizer, compute_f1=True, test=True)
                    logger.info("{} test_acc=%.4f test_macro_f1=%.4f".format(testsets[test_data_id]),
                                result['eval_acc'], result['eval_macro_f1'])
                    logger.info("metrics for each label:\n {}".format(str(result['eval_metrics_for_labels'])))
                    logger.info("  " + "*" * 20)

                    if args.local_rank in [-1, 0]:
                        fa.write("[%s] %s test-acc: %.4f test_macro_f1=%.4f\n" % (
                            criteria, testsets[test_data_id], result['eval_acc'], result['eval_macro_f1'],
                            ))
                        fa.write("metrics for each label:\n {}\n".format(str(result['eval_metrics_for_labels'])))
                    if args.res_fn and args.local_rank in [-1, 0]:
                        with open(args.res_fn, 'a+') as f:
                            f.write('[Time: {}] {}\n'.format(get_elapse_time(t0), file))
                            f.write("[%s] %s acc: %.4f macro-f1: %.4f\n\n" % (
                                criteria, testsets[test_data_id], result['eval_acc'], result['eval_macro_f1']))
                            f.write("metrics for each label:\n {}\n".format(str(result['eval_metrics_for_labels'])))
            elif args.task == 'retro_mix_detection':
                testsets = ['clef_2022_detection', 'claimbuster']
                for test_set in testsets:
                    test_examples, test_data = load_retro_data(args, tokenizer, 'test', task=test_set)
                    result = evaluate(args, model, test_examples, test_data, tokenizer, compute_f1=True, test=True)
                    logger.info("{} test_acc=%.4f test_macro_f1=%.4f".format(test_set),
                                result['eval_acc'], result['eval_macro_f1'])
                    logger.info("metrics for each label:\n {}".format(str(result['eval_metrics_for_labels'])))
                    logger.info("  " + "*" * 20)

                    if args.local_rank in [-1, 0]:
                        fa.write("[%s] %s test-acc: %.4f test_macro_f1=%.4f\n" % (
                            criteria, test_set, result['eval_acc'], result['eval_macro_f1'],
                        ))
                        fa.write("metrics for each label:\n {}\n".format(str(result['eval_metrics_for_labels'])))
                    if args.res_fn and args.local_rank in [-1, 0]:
                        with open(args.res_fn, 'a+') as f:
                            f.write('[Time: {}] {}\n'.format(get_elapse_time(t0), file))
                            f.write("[%s] %s acc: %.4f macro-f1: %.4f\n\n" % (
                                criteria, test_set, result['eval_acc'], result['eval_macro_f1']))
                            f.write("metrics for each label:\n {}\n".format(str(result['eval_metrics_for_labels'])))
            else:
                test_examples, test_data = load_and_cache_data(args, tokenizer, 'test')
                result = evaluate(args, model, test_examples, test_data, tokenizer, compute_f1=True, test=True)
                logger.info("test_acc=%.4f test_macro_f1=%.4f",
                            result['eval_acc'], result['eval_macro_f1'])
                logger.info("metrics for each label:\n {}".format(str(result['eval_metrics_for_labels'])))
                logger.info("  " + "*" * 20)

                if args.local_rank in [-1, 0]:
                    fa.write("[%s] test-acc: %.4f test_macro_f1=%.4f\n" % (
                        criteria, result['eval_acc'], result['eval_macro_f1'],
                    ))
                    fa.write("metrics for each label:\n {}\n".format(str(result['eval_metrics_for_labels'])))
                if args.res_fn and args.local_rank in [-1, 0]:
                    with open(args.res_fn, 'a+') as f:
                        f.write('[Time: {}] {}\n'.format(get_elapse_time(t0), file))
                        f.write("[%s] acc: %.4f macro-f1: %.4f\n\n" % (
                            criteria, result['eval_acc'], result['eval_macro_f1']))
                        f.write("metrics for each label:\n {}\n".format(str(result['eval_metrics_for_labels'])))
    if args.local_rank in [-1, 0]:
        fa.close()
    cleanup(args)


if __name__ == "__main__":
    main()
