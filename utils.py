import pandas as pd
import time
import os
import logging
import torch
import numpy as np
import json
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, Sampler
import torch.distributed as dist
import math
from tqdm import tqdm
import stanza
import datetime

argument_tasks = ['mt', 'oc', 'pe', 'vg', 'wd', 'wtp']
logger = logging.getLogger(__name__)


def get_file_name(data_root, task, split_tag=None):
    if task == 'claimbuster':
        data_dir = "{}/claimbuster".format(data_root)
        train_fn = "{}/crowdsourced.csv".format(data_dir)
        test_fn = "{}/groundtruth.csv".format(data_dir)
        dev_fn = "{}/crowdsourced.csv".format(data_dir)
    elif task == 'clef_2022_worth':
        data_dir = "{}/CLEF_2022".format(data_root)
        train_fn = "{}/CT22_english_1A_checkworthy_train.tsv".format(data_dir)
        test_fn = "{}/CT22_english_1A_checkworthy_dev_test.tsv".format(data_dir)
        dev_fn = "{}/CT22_english_1A_checkworthy_dev.tsv".format(data_dir)
    elif task == 'clef_2022_detection':
        data_dir = "{}/CLEF_2022".format(data_root)
        train_fn = "{}/CT22_english_1B_claim_train.tsv".format(data_dir)
        test_fn = "{}/CT22_english_1B_claim_dev_test.tsv".format(data_dir)
        dev_fn = "{}/CT22_english_1B_claim_dev.tsv".format(data_dir)
    elif task == 'claim-rank':
        data_dir = "{}/claim-rank".format(data_root)
        train_fn = ""
        test_fn = ""
        dev_fn = ""
    elif task == 'clef_2019_worth':
        data_dir = "{}/CLEF_2019".format(data_root)
        train_fn = "{}/training".format(data_dir)
        test_fn = "{}/test".format(data_dir)
        dev_fn = "{}/training".format(data_dir)
    elif task == 'news_claim':
        assert split_tag in ["test", "dev"], "split_tag not found in ['test', 'dev']"
        data_dir = "{}/newsclaim".format(data_root)
        train_fn = ""
        test_fn = "{}/test.json".format(data_dir)
        dev_fn = "{}/dev.json".format(data_dir)
    elif task == 'nyt_corpus':
        assert split_tag in ["test"], "split_tag not found in ['test']"
        data_dir = "{}/nyt_corpus".format(data_root)
        train_fn = ""
        test_fn = "{}/NYT_all_full_text.csv".format(data_dir)
        dev_fn = ""
    elif task in argument_tasks:
        data_dir = "{}/".format(data_root) + task.upper()
        train_fn = "{}/merged.train".format(data_dir)
        test_fn = "{}/merged.test".format(data_dir)
        dev_fn = "{}/merged.dev".format(data_dir)
    elif task == 'lesa-twitter':
        data_dir = "{}/lesa-twitter".format(data_root)
        train_fn = "{}/twitter_train.csv".format(data_dir)
        test_fn = "{}/twitter_test.csv".format(data_dir)
        dev_fn = "{}/twitter_dev.csv".format(data_dir)
    elif task == 'sci_abstract':
        assert split_tag in ["test"], "split_tag not found in ['test']"
        data_dir = "{}/sci".format(data_root)
        train_fn = ""
        test_fn = "{}/sci_abstract.json".format(data_dir)
        # test_fn = "{}/sci_120000_20000.json".format(data_dir)
        dev_fn = ""
    elif task == 'speeches_sentences':
        # assert split_tag in ["test"], "split_tag not found in ['test']"
        data_dir = "{}/speeches_sentences".format(data_root)
        train_fn = ""
        test_fn = "" # return data_dir because there are lots of files
        dev_fn = ""
    else:
        raise ValueError("Task not found in ['claimbuster', 'claim-rank', 'clef_2022_worth', 'clef_2019_worth', "
                         "'clef_2022_detection','nyt_corpus','sci_abstract','speeches_sentences']")
    if split_tag == 'train':
        return train_fn
    elif split_tag == 'test':
        return test_fn
    elif split_tag == 'dev':
        return dev_fn
    else:
        return data_dir, train_fn, test_fn


def read_examples(filename, task):
    func_dict = {
        'claimbuster': read_claimbuster_examples,
        'clef_2022_worth': read_clef_2022_worth_examples,
        'claim-rank': read_claim_rank_examples,
        'clef_2019_worth': read_clef_2019_worth_examples,
        'clef_2022_detection': read_clef_2022_worth_examples,
        'lesa-twitter': read_lesa_twitter_examples,
    }
    if task in func_dict.keys():
        return func_dict[task](filename)
    elif task in ['mt', 'oc', 'pe', 'vg', 'wd', 'wtp']:
        return read_argument_examples(filename)
    else:
        raise ValueError("Task {} not defined.".format(task))


def read_argument_examples(filename):
    examples = []
    with open(filename, 'r') as f:
        lines = f.readlines()
        texts = [l.strip().split('\t')[0] for l in lines]
        labels = [l.strip().split('\t')[1] for l in lines]
    for idx, (text, label) in enumerate(zip(texts, labels)):
        examples.append(
            Example(idx=idx, source=text, target=label)
        )
    return examples


def read_lesa_twitter_examples(file_name):
    examples = []
    df = pd.read_csv(file_name)
    texts, labels = df.text.to_list(), df.claim.to_list()
    for idx, (text, label) in enumerate(zip(texts, labels)):
        examples.append(
            Example(idx=idx, source=text, target=label)
        )
    return examples


def read_speeches_sentences_examples(file_dir, cache_path):
    examples = []
    filename_list=os.listdir(file_dir)
    file_num=len(filename_list)
    idx = 0
    for i in tqdm(range(file_num)):
        with open("{}/{}".format(file_dir,filename_list[i]),"r",encoding="utf-8") as f:
            texts = f.readlines()
            for text in texts:
                examples.append(
                    Example_ss(idx=idx, source=text.strip(), target=1, filename=filename_list[i])
                )
                idx += 1
    return examples


def read_sci_examples(filename, cache_path):
    examples = []
    nlp = stanza.Pipeline('en', processors='tokenize', dir=cache_path)
    idx = 0
    logger.info("open json file")
    with open(filename,'r',encoding='utf8')as f:
        for json_data in tqdm(f.readlines()):
            try:
                j = json.loads(json_data)
                texts = j["contents"]
                doc = nlp(texts)
                for s in doc.sentences:
                    if s.text[0]=='©':
                        origin=s.text
                for s in doc.sentences:
                    if s.text[0]!='©':
                        examples.append(
                            Example_sci(idx=idx, source=s.text, target=1, id=j["id"], copyright=origin)
                        )
                idx += 1
            except:
                logger.info("cannot loads No."+str(idx)+" json. Continue")
    return examples
        

def read_nyt_examples(filename, n, nsamples, seed, cache_path):
    examples = []
    df = pd.read_csv(filename, nrows=n, encoding="utf-8")
    df_samples = df.sample(nsamples, replace=False, random_state=seed)
    nlp = stanza.Pipeline('en', processors='tokenize', dir=cache_path)
    idx = 0
    for i in range(nsamples):
        texts = df_samples["article_text"][i]
        doc = nlp(texts)
        for s in doc.sentences:
            examples.append(
                Example_nyt(idx=idx, source=s.text, target=1,
                            pub_date=df_samples["pub_date"][i], main=df_samples["main"][i])
            )
            idx += 1
    return examples


def read_clef_2019_worth_examples(dir_name):
    examples = []
    dfs = []
    for filename in os.listdir(dir_name):
        df = pd.read_csv(os.path.join(dir_name, filename), delimiter="\t", usecols=[2, 3], names=['texts', 'labels'])
        dfs.append(df)
    df = pd.concat(dfs, axis=0)
    texts, labels = df.texts.tolist(), df.labels.tolist()
    for idx, (text, label) in enumerate(zip(texts, labels)):
        examples.append(
            Example(idx=idx, source=text, target=label)
        )
    return examples


def read_claimbuster_examples(filename):
    examples = []
    df = pd.read_csv(filename)
    df.Verdict += 1  # these are [-1, 0, 1], but we want [0,1,2] for training BERT models
    texts, labels = df.Text.tolist(), df.Verdict.tolist()
    for idx, (text, label) in enumerate(zip(texts, labels)):
        examples.append(
            Example(idx=idx, source=text, target=label)
        )
    return examples


def read_clef_2022_worth_examples(filename):
    examples = []
    df = pd.read_csv(filename, delimiter="\t")
    texts, labels = df.tweet_text.tolist(), df.class_label.tolist()
    for idx, (text, label) in enumerate(zip(texts, labels)):
        examples.append(
            Example(idx=idx, source=text, target=label)
        )
    return examples


def read_news_claim_examples(filename, data_dir):
    examples = []
    with open('{}/all_sents.json'.format(data_dir), 'r', encoding='utf-8') as f:
        all_sents = json.load(f)
    with open(filename, 'r', encoding='utf-8') as f:
        claim_sents = json.load(f)

    for article in all_sents.keys():
        if article in claim_sents.keys():
            sents_in_article = [dic['sentence'] for k, dic in all_sents[article].items()]
            claims_in_article = [dic['sentence'] for k, dic in claim_sents[article].items()]
            for idx, sent in enumerate(sents_in_article):
                if sent in claims_in_article:
                    examples.append(Example(idx=idx, source=sent, target=1))
                else:
                    examples.append(Example(idx=idx, source=sent, target=0))
    return examples


def read_claim_rank_examples(dir_name):
    examples = []
    dfs = []
    for filename in os.listdir(dir_name):
        df = pd.read_csv(os.path.join(dir_name, filename), delimiter="\t")
        dfs.append(df)
    df = pd.concat(dfs, axis=0)
    df.ALL = df.ALL.apply(lambda x: 0 if x == 0 else 1)  # if at least one source labels it as a claim, it is a claim
    texts, labels = df.Text.tolist(), df.ALL.tolist()
    for idx, (text, label) in enumerate(zip(texts, labels)):
        examples.append(
            Example(idx=idx, source=text, target=label)
        )
    return examples


def convert_claim_examples_to_features(args, examples, task, tokenizer):
    if args.model_type in ['t5']:
        sources = []
        targets = []
        for example in examples:
            if task == 'claimbuster':
                if args.add_task_prefix and args.add_prompt:
                    source_str = "{}: {}: {}".format(task, 'Is the following sentence a non-claim (0),'
                                                           ' a not-check-worthy claim (1), or a check-worthy claim (2)?'
                                                     , example.source)
                elif args.add_task_prefix and not args.add_prompt:
                    source_str = "{}: {}".format(task, example.source)
                else:
                    source_str = example.source
                if example.target in [0, 1, 2]:
                    target_str = str(example.target)
                else:
                    raise ValueError("Label not defined in " + task)
            else:
                if args.add_task_prefix and args.add_prompt:
                    source_str = "{}: {}: {}".format(task, 'Is the following sentence a non-claim (0) or a claim (1)?'
                                                     , example.source)
                elif args.add_task_prefix and not args.add_prompt:
                    source_str = "{}: {}".format(task, example.source)
                else:
                    source_str = example.source
                if example.target in [0, 1]:
                    target_str = str(example.target)
                else:
                    raise ValueError("Label not defined in " + task)
            sources.append(source_str)
            targets.append(target_str)
        source_ids = tokenizer(sources, truncation=True, max_length=512, padding=True).input_ids
        with tokenizer.as_target_tokenizer():
            target_ids = tokenizer(targets, truncation=True, max_length=20, padding=True).input_ids
        return T5Dataset(source_ids, target_ids)
    elif args.model_type in ['bert', 'roberta']:
        sources = []
        targets = []
        for example in examples:
            sources.append(example.source)
            targets.append(example.target)
        source_encodings = tokenizer(sources, truncation=True, max_length=512, padding=True)
        return BertDataset(source_encodings, targets)
    else:
        raise ValueError("Model type not defined.")


def convert_retro_examples_to_features(args, examples, task, tokenizer):
    if args.model_type in ['t5']:
        source_as = []
        source_bs = []
        targets = []
        for example in examples:
            if args.add_task_prefix:
                source_b_str = "{}: {}".format(task, example.source_b)
            else:
                source_b_str = example.source_b
            source_a_str = example.source_a
            if example.target in [0, 1]:
                target_str = str(example.target)
            else:
                raise ValueError("Label not defined in " + task)
            source_as.append(source_a_str)
            source_bs.append(source_b_str)
            targets.append(target_str)
        source_ids = tokenizer(source_bs, source_as, max_length=512, padding=True, truncation="only_second").input_ids
        with tokenizer.as_target_tokenizer():
            target_ids = tokenizer(targets, truncation=True, max_length=20, padding=True).input_ids
        return T5Dataset(source_ids, target_ids)
    elif args.model_type in ['bert', 'roberta']:
        source_as = []
        source_bs = []
        targets = []
        for example in examples:
            source_as.append(example.source_a)
            source_bs.append(example.source_b)
            targets.append(example.target)
        source_encodings = tokenizer(source_bs, source_as, max_length=512, padding=True, truncation="only_second")
        return BertDataset(source_encodings, targets)
    else:
        raise ValueError("Model type not defined.")


def read_retro_examples(filename):
    with open(filename, 'r') as f:
        examples = [json.loads(i) for i in f]
    all_text_a = []
    for i in examples:
        text_a = []
        for j in i["text_a"]:
            if j[1] == 1:
                claim_label = "claim"
            else:
                claim_label = "no claim"
            text_a.append("text: " + j[0] + " label: " + claim_label)
        text_a = "\n".join(text_a)
        all_text_a.append(text_a)
    all_text_b = [i["text_b"] for i in examples]
    labels = [i["label"] for i in examples]
    retro_examples = []
    for idx in range(len(all_text_a)):
        retro_examples.append(
            RetroExample(idx=idx, source_a=all_text_a[idx], source_b=all_text_b[idx], target=labels[idx])
        )
    return retro_examples


def load_retro_data(args, tokenizer, split_tag, task='clef_2022_detection'):
    if split_tag != 'test':
        cache_fn = os.path.join(args.cache_path, split_tag)
    else:
        cache_fn = os.path.join(args.cache_path, split_tag + '_' + task)
    if split_tag == 'train':
        examples = read_retro_examples("{}/retro_data/train_retro_jingwei_replication.jsonl".format(args.data_dir))
    elif split_tag == 'dev':
        examples = read_retro_examples("{}/retro_data/dev_retro_jingwei_replication.jsonl".format(args.data_dir))
    elif split_tag == 'test' and task == 'clef_2022_detection':
        examples = read_retro_examples("{}/retro_data/test_CLEF.jsonl".format(args.data_dir))
    elif split_tag == 'test' and task == 'claimbuster':
        examples = read_retro_examples("{}/retro_data/test_claimbuster.jsonl".format(args.data_dir))
    if args.local_rank != -1:
        dist.barrier()
    if os.path.exists(cache_fn):
        logger.info("Load cache data from %s", cache_fn)
        data = torch.load(cache_fn)
    else:
        logger.info("Create cache data into %s", cache_fn)
        data = convert_retro_examples_to_features(args, examples, 'retro_mix_detection', tokenizer)

        if args.local_rank in [-1, 0]:
            torch.save(data, cache_fn)

    return examples, data


def load_nyt_data(args, tokenizer, split_tag):
    filename = get_file_name(args.data_dir, 'nyt_corpus', split_tag)
    cache_fn = os.path.join(args.cache_path, str(args.seed))
    examples = read_nyt_examples(
        filename, args.nrows, args.nsamples, args.sample_seed, args.cache_path)
    calc_stats(examples, tokenizer, is_tokenize=True)
    if args.local_rank != -1:
        dist.barrier()
    if os.path.exists(cache_fn):
        logger.info("Load cache data from %s", cache_fn)
        data = torch.load(cache_fn)
    else:
        logger.info("Create cache data into %s", cache_fn)
        data = convert_claim_examples_to_features(
            args, examples, 'nyt_corpus', tokenizer)

        if args.local_rank in [-1, 0]:
            torch.save(data, cache_fn)

    return examples, data


def load_speeches_sentences_data(args, tokenizer, split_tag):
    file_fir,_,_ = get_file_name(args.data_dir, 'speeches_sentences', split_tag) # split_tag="dict"
    cache_fn = os.path.join(args.cache_path, str(args.seed))
    examples = read_speeches_sentences_examples(file_fir, args.cache_path)
    calc_stats(examples, tokenizer, is_tokenize=True)
    if args.local_rank != -1:
        dist.barrier()
    if os.path.exists(cache_fn):
        logger.info("Load cache data from %s", cache_fn)
        data = torch.load(cache_fn)
    else:
        logger.info("Create cache data into %s", cache_fn)
        data = convert_claim_examples_to_features(
            args, examples, 'speeches_sentences', tokenizer)

        if args.local_rank in [-1, 0]:
            torch.save(data, cache_fn)

    return examples, data



def load_sci_data(args, tokenizer, split_tag):
    filename = get_file_name(args.data_dir, 'sci_abstract', split_tag)
    cache_fn = os.path.join(args.cache_path, str(args.seed))
    examples = read_sci_examples(filename, args.cache_path)
    # calc_stats(examples, tokenizer, is_tokenize=True)
    if args.local_rank != -1:
        dist.barrier()
    if os.path.exists(cache_fn):
        logger.info("Load cache data from %s", cache_fn)
        data = torch.load(cache_fn)
    else:
        logger.info("Create cache data into %s", cache_fn)
        data = convert_claim_examples_to_features(
            args, examples, 'sci_abstract', tokenizer)

        if args.local_rank in [-1, 0]:
            torch.save(data, cache_fn)

    return examples, data


def load_claimbuster_data(args, tokenizer, split_tag):
    filename = get_file_name(args.data_dir, 'claimbuster', split_tag)
    cache_fn = os.path.join(args.cache_path, split_tag)
    examples = read_claimbuster_examples(filename)
    if split_tag == 'dev':
        _, dev_examples = train_test_split(examples, test_size=0.1, random_state=args.split_seed)
        examples = dev_examples
    elif split_tag == 'train':
        train_examples, _ = train_test_split(examples, test_size=0.1, random_state=args.split_seed)
        examples = train_examples
    calc_stats(examples, tokenizer, is_tokenize=True)
    if args.local_rank != -1:
        dist.barrier()
    if os.path.exists(cache_fn):
        logger.info("Load cache data from %s", cache_fn)
        data = torch.load(cache_fn)
    else:
        logger.info("Create cache data into %s", cache_fn)
        data = convert_claim_examples_to_features(args, examples, 'claimbuster', tokenizer)

        if args.local_rank in [-1, 0]:
            torch.save(data, cache_fn)

    return examples, data


def load_general_data(args, tokenizer, split_tag):
    filename = get_file_name(args.data_dir, args.task, split_tag)
    cache_fn = os.path.join(args.cache_path, split_tag)
    examples = read_examples(filename, args.task)
    calc_stats(examples, tokenizer, is_tokenize=True)
    if args.local_rank != -1:
        dist.barrier()
    if os.path.exists(cache_fn):
        logger.info("Load cache data from %s", cache_fn)
        data = torch.load(cache_fn)
    else:
        logger.info("Create cache data into %s", cache_fn)
        data = convert_claim_examples_to_features(args, examples, args.task, tokenizer)

        if args.local_rank in [-1, 0]:
            torch.save(data, cache_fn)
    return examples, data


def load_claimrank_data(args, tokenizer, split_tag):
    data_dir, _, _ = get_file_name(args.data_dir, 'claim-rank')
    cache_fn = os.path.join(args.cache_path, split_tag)
    examples = read_claim_rank_examples(data_dir)
    train_examples, test_examples = train_test_split(examples, test_size=1550, random_state=args.split_seed)
    train_examples, dev_examples = train_test_split(train_examples, test_size=775, random_state=args.split_seed)
    if split_tag == 'dev':
        examples = dev_examples
    elif split_tag == 'train':
        examples = train_examples
    else:
        examples = test_examples
    calc_stats(examples, tokenizer, is_tokenize=True)
    if args.local_rank != -1:
        dist.barrier()
    if os.path.exists(cache_fn):
        logger.info("Load cache data from %s", cache_fn)
        data = torch.load(cache_fn)
    else:
        logger.info("Create cache data into %s", cache_fn)
        data = convert_claim_examples_to_features(args, examples, 'claim-rank', tokenizer)

        if args.local_rank in [-1, 0]:
            torch.save(data, cache_fn)

    return examples, data


# Train a claim identification model unifying non-noisy data from different datasets
def load_detection_data(args, tokenizer, split_tag, test_data='0'):
    if split_tag != 'test':
        cache_fn = os.path.join(args.cache_path, split_tag)
    else:
        cache_fn = os.path.join(args.cache_path, split_tag + test_data)
    all_examples = []
    all_texts = []
    if split_tag == 'test':
        data_mix = test_data
    else:
        data_mix = args.data_mix
    if '0' in data_mix:
        # load corresponding splits from claimbuster
        filename = get_file_name(args.data_dir, 'claimbuster', split_tag)
        examples = read_claimbuster_examples(filename)
        texts = []
        for example in examples:
            texts.append(example.source)
            if example.target == 2:
                example.target = 1
        all_examples.extend(examples)
        all_texts.extend(texts)
    if '1' in data_mix:
        # add all positive samples from claim-rank
        data_dir, _, _ = get_file_name(args.data_dir, 'claim-rank')
        examples = read_claim_rank_examples(data_dir)
        to_add_examples = []
        to_add_texts = []
        for example in examples:
            if example.source not in all_texts and example.target == 1:
                to_add_examples.append(example)
                to_add_texts.append(example.source)
        logger.info("Add {} {} data from claim-rank".format(len(to_add_examples),
                                                            'test' if split_tag == 'test' else 'train/dev'))
        all_examples.extend(to_add_examples)
        all_texts.extend(to_add_texts)
    if '2' in data_mix:
        # add all positive samples from clef_2019_worth
        train_dir = get_file_name(args.data_dir, 'clef_2019_worth', 'train')
        test_dir = get_file_name(args.data_dir, 'clef_2019_worth', 'test')
        train_examples = read_clef_2019_worth_examples(train_dir)
        test_examples = read_clef_2019_worth_examples(test_dir)
        examples = train_examples + test_examples
        to_add_examples = []
        to_add_texts = []
        for example in examples:
            if example.source not in all_texts and example.target == 1:
                to_add_examples.append(example)
                to_add_texts.append(example.source)
        logger.info("Add {} {} data from clef_2019_worth".format(len(to_add_examples),
                                                                 'test' if split_tag == 'test' else 'train/dev'))
        all_examples.extend(to_add_examples)
        all_texts.extend(to_add_texts)
    if '3' in data_mix:
        if split_tag != 'test':
            train_fn = get_file_name(args.data_dir, 'clef_2022_detection', 'train')
            dev_fn = get_file_name(args.data_dir, 'clef_2022_detection', 'dev')
            train_examples = read_clef_2022_worth_examples(train_fn)
            dev_examples = read_clef_2022_worth_examples(dev_fn)
            examples = train_examples + dev_examples
        else:
            test_fn = get_file_name(args.data_dir, 'clef_2022_detection', 'test')
            examples = read_clef_2022_worth_examples(test_fn)
        logger.info("Add {} {} data from clef_2022_detection".format(len(examples),
                                                                     'test' if split_tag == 'test' else 'train/dev'))
        all_examples.extend(examples)
    if '4' in data_mix:
        assert(split_tag == 'test') # newsclaim has only test set
        test_fn = get_file_name(args.data_dir, 'news_claim', 'test')
        examples = read_news_claim_examples(test_fn, '{}/newsclaim'.format(args.data_dir))
        logger.info("Add {} test data from newsclaim".format(len(examples)))
        all_examples.extend(examples)
    if '5' in data_mix:
        if split_tag != 'test':
            train_fn = get_file_name(args.data_dir, 'lesa-twitter', 'train')
            dev_fn = get_file_name(args.data_dir, 'lesa-twitter', 'dev')
            train_examples = read_lesa_twitter_examples(train_fn)
            dev_examples = read_lesa_twitter_examples(dev_fn)
            examples = train_examples + dev_examples
        else:
            test_fn = get_file_name(args.data_dir, 'lesa-twitter', 'test')
            examples = read_lesa_twitter_examples(test_fn)
        logger.info("Add {} {} data from lesa-twitter".format(len(examples),
                                                              'test' if split_tag == 'test' else 'train/dev'))
        all_examples.extend(examples)

    if split_tag == 'train':
        train_examples, _ = train_test_split(all_examples, test_size=0.1, random_state=args.split_seed)
        all_examples = train_examples
    elif split_tag == 'dev':
        _, dev_examples = train_test_split(all_examples, test_size=0.1, random_state=args.split_seed)
        all_examples = dev_examples

    calc_stats(all_examples, tokenizer, is_tokenize=True)
    if args.local_rank != -1:
        dist.barrier()
    if os.path.exists(cache_fn):
        logger.info("Load cache data from %s", cache_fn)
        data = torch.load(cache_fn)
    else:
        logger.info("Create cache data into %s", cache_fn)
        data = convert_claim_examples_to_features(args, all_examples, 'mix_detection', tokenizer)

        if args.local_rank in [-1, 0]:
            torch.save(data, cache_fn)

    return all_examples, data


def calc_stats(examples, tokenizer=None, is_tokenize=False):
    avg_src_len = []
    avg_trg_len = []
    avg_src_len_tokenize = []
    avg_trg_len_tokenize = []
    for ex in examples:
        if is_tokenize:
            avg_src_len.append(len(ex.source.split()))
            avg_trg_len.append(len(str(ex.target).split()))
            avg_src_len_tokenize.append(len(tokenizer.tokenize(ex.source)))
            avg_trg_len_tokenize.append(len(tokenizer.tokenize(str(ex.target))))
        else:
            avg_src_len.append(len(ex.source.split()))
            avg_trg_len.append(len(str(ex.target).split()))
    if is_tokenize:
        logger.info("Read %d examples, avg src len: %d, avg trg len: %d, max src len: %d, max trg len: %d",
                    len(examples), np.mean(avg_src_len), np.mean(avg_trg_len), max(avg_src_len), max(avg_trg_len))
        logger.info("[TOKENIZE] avg src len: %d, avg trg len: %d, max src len: %d, max trg len: %d",
                    np.mean(avg_src_len_tokenize), np.mean(avg_trg_len_tokenize), max(avg_src_len_tokenize),
                    max(avg_trg_len_tokenize))
    else:
        logger.info("Read %d examples, avg src len: %d, avg trg len: %d, max src len: %d, max trg len: %d",
                    len(examples), np.mean(avg_src_len), np.mean(avg_trg_len), max(avg_src_len), max(avg_trg_len))


def save_speeches_sentences_result(args, pred_ids, examples):
    ex_source = []
    ex_filename = []
    for ex in examples:
        ex_source.append(ex.source)
        ex_filename.append(ex.filename)
    data_df = pd.DataFrame(pred_ids)
    data_df["text"] = ex_source
    data_df["filename"] = ex_filename
    save_path = "./results/" + datetime.datetime.strftime(datetime.datetime.now(), '%Y-%m-%d_%H:%M:%S') + "_" + \
        str(args.task) + ".csv"
    data_df.to_csv(save_path, index=0, encoding="utf-8")
    

def save_nyt_result(args, pred_ids, examples):
    ex_source = []
    ex_pub_date = []
    ex_main=[]
    for ex in examples:
        ex_source.append(ex.source)
        ex_pub_date.append(ex.pub_date)
        ex_main.append(ex.main)
    data_df = pd.DataFrame(pred_ids)
    data_df["text"] = ex_source
    data_df["pub_date"] = ex_pub_date
    data_df["main"] = ex_main
    save_path = "./results/" + datetime.datetime.strftime(datetime.datetime.now(), '%Y-%m-%d_%H:%M:%S') + "_" + \
        str(args.nrows) + "_"  + str(args.nsamples)  + "_" + str(args.sample_seed) + ".csv"
    data_df.to_csv(save_path, index=0, encoding="utf-8")


def save_sci_result(args, pred_ids, examples):
    ex_source = []
    ex_ids = []
    ex_cps = []
    for ex in examples:
        ex_source.append(ex.source)
        ex_ids.append(ex.id)
        ex_cps.append(ex.copyright)
    data_df = pd.DataFrame(pred_ids)
    data_df["text"] = ex_source
    data_df["id"] = ex_ids
    data_df["copyright"] = ex_cps
    save_path = "./results/" + datetime.datetime.strftime(datetime.datetime.now(), '%Y-%m-%d_%H:%M:%S') + "_" + \
        str(args.nrows) + "_"  + str(args.nsamples)  + "_" + str(args.sample_seed) + ".csv"
    data_df.to_csv(save_path, index=0, encoding="utf-8")


class Example(object):
    """A single training/test example."""

    def __init__(self,
                 idx,
                 source,
                 target,
                 ):
        self.idx = idx
        self.source = source
        self.target = target


class RetroExample(object):
    def __init__(self,
                 idx,
                 source_a,
                 source_b,
                 target,
                 ):
        self.idx = idx
        self.source_a = source_a
        self.source_b = source_b
        self.target = target


class Example_ss(object):
    def __init__(self,
                 idx,
                 source,
                 target,
                 filename,
                 ):
        self.idx = idx
        self.source = source
        self.target = target
        self.filename = filename


class Example_nyt(object):
    """A single training/test example."""

    def __init__(self,
                 idx,
                 source,
                 target,
                 pub_date,
                 main
                 ):
        self.idx = idx
        self.source = source
        self.target = target
        self.pub_date = pub_date
        self.main = main


class Example_sci(object):
    """A single training/test example."""

    def __init__(self,
                 idx,
                 source,
                 target,
                 id,
                 copyright
                 ):
        self.idx = idx
        self.source = source
        self.target = target
        self.id = id
        self.copyright = copyright


class BertDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)


class T5Dataset(torch.utils.data.Dataset):
    def __init__(self, input_ids, labels):
        self.input_ids = input_ids
        self.labels = labels

    def __getitem__(self, idx):
        item = {'input_ids': torch.tensor(self.input_ids[idx]), 'labels': torch.tensor(self.labels[idx])}
        return item

    def __len__(self):
        return len(self.labels)


def get_elapse_time(t0):
    elapse_time = time.time() - t0
    if elapse_time > 3600:
        hour = int(elapse_time // 3600)
        minute = int((elapse_time % 3600) // 60)
        return "{}h{}m".format(hour, minute)
    else:
        minute = int((elapse_time % 3600) // 60)
        return "{}m".format(minute)


def save_checkpoint(rank, training_state, optimizer, scheduler, model_to_save, output_model_file,
                    output_optimizer_file, output_scheduler_file, last_output_dir):
    if rank in [-1, 0]:
        torch.save(model_to_save.state_dict(), output_model_file)
        torch.save(optimizer.state_dict(), output_optimizer_file)
        torch.save(scheduler.state_dict(), output_scheduler_file)
        with open(os.path.join(last_output_dir, "training_state.json"), 'w') as f:
            json.dump(training_state, f)


# copied from transformers.trainer_pt_utils, removing warning.
class SequentialDistributedSampler(torch.utils.data.sampler.Sampler):
    """
    Distributed Sampler that subsamples indices sequentially, making it easier to collate all results at the end.

    Even though we only use this sampler for eval and predict (no training), which means that the model params won't
    have to be synced (i.e. will not hang for synchronization even if varied number of forward passes), we still add
    extra samples to the sampler to make it evenly divisible (like in `DistributedSampler`) to make it easy to `gather`
    or `reduce` resulting tensors at the end of the loop.
    """

    def __init__(self, dataset, num_replicas=None, rank=None, batch_size=None):
        if num_replicas is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            num_replicas = dist.get_world_size()
        if rank is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            rank = dist.get_rank()
        self.dataset = dataset
        self.num_replicas = num_replicas
        self.rank = rank
        num_samples = len(self.dataset)
        # Add extra samples to make num_samples a multiple of batch_size if passed
        if batch_size is not None:
            self.num_samples = int(math.ceil(num_samples / (batch_size * num_replicas))) * batch_size
        else:
            self.num_samples = int(math.ceil(num_samples / num_replicas))
        self.total_size = self.num_samples * self.num_replicas
        self.batch_size = batch_size

    def __iter__(self):
        indices = list(range(len(self.dataset)))

        # add extra samples to make it evenly divisible
        indices += indices[: (self.total_size - len(indices))]
        assert (
                len(indices) == self.total_size
        ), f"Indices length {len(indices)} and total size {self.total_size} mismatched"

        # subsample
        indices = indices[self.rank * self.num_samples: (self.rank + 1) * self.num_samples]
        assert (
                len(indices) == self.num_samples
        ), f"Indices length {len(indices)} and sample number {self.num_samples} mismatched"

        return iter(indices)

    def __len__(self):
        return self.num_samples
