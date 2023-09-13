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


def distributed_concat(tensor, num_total_examples):
    output_tensors = [tensor.clone()
                      for _ in range(torch.distributed.get_world_size())]
    torch.distributed.all_gather(output_tensors, tensor)
    concat = torch.cat(output_tensors, dim=0)
    # truncate the dummy elements added by SequentialDistributedSampler
    return concat[:num_total_examples]


def evaluate(args, model, eval_examples, eval_data, tokenizer, apply_softmax=True):
    if args.local_rank == -1:
        eval_sampler = SequentialSampler(eval_data)
        eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=args.eval_batch_size,
                                     num_workers=4, pin_memory=True)
    else:
        eval_sampler = SequentialDistributedSampler(
            eval_data, batch_size=args.eval_batch_size)
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

    for idx, batch in enumerate(tqdm(eval_dataloader, total=len(eval_dataloader), desc="Evaluating")):
        inputs = {k: v.to(args.device) for k, v in batch.items()}
        with torch.no_grad():
            if args.model_type in ['t5']:
                model_to_generate = model.module if hasattr(
                    model, 'module') else model
                predictions = model_to_generate.generate(
                    inputs['input_ids'],
                    attention_mask=inputs['input_ids'].ne(
                        tokenizer.pad_token_id),
                    use_cache=True,
                    max_new_tokens=5,
                    num_beams=5,
                )
                pred_ids.append(predictions)
            else:
                logit = model(**inputs).logits
                if apply_softmax:
                    logit = softmax(logit, dim=1)
                pred_ids.append(logit)
        nb_eval_steps += 1

    if args.local_rank == -1 or args.no_cuda:
        pred_ids = torch.concat(pred_ids, dim=0)
    else:
        pred_ids = distributed_concat(torch.concat(
            pred_ids, dim=0), len(eval_sampler.dataset))
    if args.model_type in ['t5']:
        pred_ids = list(pred_ids.cpu().numpy())
    else:
        pred_ids = pred_ids.cpu().numpy()

    logger.info("***** Eval Ended *****")
    return pred_ids


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
        logging.getLogger().setLevel(logging.INFO if dist.get_rank()
                                     in [-1, 0] else logging.WARN)
    set_seed(args)

    # Build model
    config_class, model_class, tokenizer_class = MODEL_CLASSES[args.model_type]
    config = config_class.from_pretrained(args.config_name if args.config_name else args.model_name_or_path,
                                          num_labels=NUM_LABELS[args.task])
    model = model_class.from_pretrained(args.model_name_or_path, config=config, cache_dir='cache',
                                        local_files_only=True)
    tokenizer = tokenizer_class.from_pretrained(
        args.model_name_or_path, cache_dir='cache', local_files_only=True)
    if args.cont:
        model_file = os.path.join(
            args.output_dir, "checkpoint-last/pytorch_model.bin")
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

    if args.do_test:
        logger.info("  " + "***** Testing *****")
        logger.info("  Batch size = %d", args.eval_batch_size)

        for criteria in ['best-acc']:
            file = os.path.join(
                args.output_dir, 'checkpoint-{}/pytorch_model.bin'.format(criteria))
            logger.info("Load model from {}".format(file))
            if hasattr(model, 'module'):
                model.module.load_state_dict(torch.load(file))
            else:
                model.load_state_dict(torch.load(file))

            model.to(args.device)

            # load data
            logger.info("Load {} data".format(args.task))
            if args.task == "nyt":
                test_examples, test_data = load_nyt_data(
                    args, tokenizer, 'test')
            elif args.task == "sci_abstract":
                test_examples, test_data = load_sci_data(
                    args, tokenizer, 'test')
            elif args.task == "speeches_sentences":
                test_examples, test_data = load_speeches_sentences_data(
                    args, tokenizer, 'dict')

            # evaluation
            result = evaluate(args, model, test_examples, test_data, tokenizer)

            # save result
            if args.task == "nyt":
                save_nyt_result(args, result, test_examples)
            elif args.task == "sci_abstract":
                save_sci_result(args, result, test_examples)
            elif args.task == "speeches_sentences":
                save_speeches_sentences_result(args, result, test_examples)

    if args.local_rank in [-1, 0]:
        fa.close()
    cleanup(args)


if __name__ == "__main__":
    main()
