import random
import os
import torch
import torch.distributed as dist
import logging
import numpy as np
import multiprocessing

logger = logging.getLogger(__name__)


def add_args(parser):
    parser.add_argument("--model_type", default="bert", type=str,
                        choices=['roberta', 't5', 'bert'])
    parser.add_argument("--num_train_epochs", default=40, type=int)
    parser.add_argument("--cache_path", type=str, required=True)
    parser.add_argument("--task", type=str, required=True, choices=[
                        'clef_2022_worth', 'claimbuster', 'claim-rank', 'mix_detection', 'nyt',
                        'lesa-twitter', 'mt', 'oc', 'pe', 'vg', 'wd', 'wtp', 'retro_mix_detection', 'sci_abstract',
                        'speeches_sentences'])
    parser.add_argument("--data_mix", type=str, default='0',
                        help="0 for claimbuster, 1 for claim_rank, 2 for clef_2019_worth, 3 for clef_2022_detection, "
                             "4 for newsclaim, 5 for lesa-twitter")
    parser.add_argument("--test_set", type=str, default='0,3,4,5')
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--res_dir", type=str, required=True)
    parser.add_argument("--res_fn", type=str, default='')
    parser.add_argument("--add_task_prefix", action='store_true', help="Whether to add task prefix for t5")
    parser.add_argument("--add_prompt", action='store_true', help="Whether to add prompt for t5")
    parser.add_argument("--save_last_checkpoints", action='store_true')
    parser.add_argument("--always_save_model", action='store_true')
    parser.add_argument("--model_name_or_path", default="roberta-base", type=str,
                        help="Path to pre-trained model: e.g. roberta-base")
    parser.add_argument("--output_dir", default=None, type=str, required=True,
                        help="The output directory where the model predictions and checkpoints will be written.")
    parser.add_argument("--load_model_path", default=None, type=str,
                        help="Path to trained model: Should contain the .bin files")
    parser.add_argument("--config_name", default="", type=str,
                        help="Pretrained config name or path if not the same as model_name")
    parser.add_argument("--tokenizer_name", default="roberta-base", type=str,
                        help="Pretrained tokenizer name or path if not the same as model_name")
    parser.add_argument("--do_train", action='store_true',
                        help="Whether to run eval on the train set.")
    parser.add_argument("--do_eval", action='store_true',
                        help="Whether to run eval on the dev set.")
    parser.add_argument("--do_test", action='store_true',
                        help="Whether to run eval on the dev set.")
    parser.add_argument("--no_cuda", action='store_true',
                        help="Avoid using CUDA when available")

    parser.add_argument("--train_batch_size", default=32, type=int,
                        help="Batch size per GPU/CPU for training.")
    parser.add_argument("--eval_batch_size", default=8, type=int,
                        help="Batch size per GPU/CPU for evaluation.")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument("--learning_rate", default=5e-5, type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--beam_size", default=10, type=int,
                        help="beam size for beam search")
    parser.add_argument("--weight_decay", default=0.0, type=float,
                        help="Weight deay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float,
                        help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float,
                        help="Max gradient norm.")
    parser.add_argument("--patience", default=5, type=int)

    parser.add_argument("--save_steps", default=-1, type=int, )
    parser.add_argument("--log_steps", default=-1, type=int, )
    parser.add_argument("--max_steps", default=-1, type=int,
                        help="If > 0: set total number of training steps to perform. Override num_train_epochs.")
    parser.add_argument("--eval_steps", default=-1, type=int,
                        help="")
    parser.add_argument("--train_steps", default=-1, type=int,
                        help="")
    parser.add_argument("--warmup_steps", default=100, type=int,
                        help="Linear warmup over warmup_steps.")
    parser.add_argument("--local_rank", type=int, default=os.getenv('LOCAL_RANK', -1),
                        help="For distributed training: local_rank")
    parser.add_argument('--seed', type=int, default=1234,
                        help="random seed for initialization")
    parser.add_argument('--split_seed', type=int, default=42,
                        help="random seed for data splitting")
    parser.add_argument("--cont", default=False, action='store_true', help="continue training or not")
    # nyt prediction
    parser.add_argument('--nrows', type=int, default=10000,
                        help="Number of rows of file to read. Useful for reading pieces of large files")
    parser.add_argument('--nsamples', type=int, default=50,
                        help="Number of sample articles of nyt_corpus with nrows")
    parser.add_argument('--sample_seed', type=int, default=42,
                        help="random seed for sampling")
    args = parser.parse_args()
    return args


def set_dist(args):
    # Setup CUDA, GPU & distributed training
    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        args.n_gpu = torch.cuda.device_count()
    else:
        # Setup for distributed data parallel
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        dist.init_process_group(backend='nccl', init_method='env://')
        args.n_gpu = torch.cuda.device_count()
    cpu_cont = multiprocessing.cpu_count()
    logger.warning("Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, cpu count: %d",
                   args.local_rank, device, args.n_gpu, bool(args.local_rank != -1), cpu_cont)
    args.device = device
    args.cpu_cont = cpu_cont


def cleanup(args):
    if args.local_rank != -1:
        dist.destroy_process_group()


def set_seed(args):
    """set random seed."""
    seed = args.seed + args.local_rank
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(seed)


