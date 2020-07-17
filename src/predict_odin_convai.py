#  Copyright (c) Microsoft Corporation.
#  Licensed under the MIT license.
"""
 * @Desc: train GPT2 from scratch/ fine tuning.
          Modified based on Huggingface GPT-2 implementation
"""

import argparse
import datetime
import json
import logging
import os
import pickle
import sys
import time
from os.path import join

import numpy as np
import torch
from sqlitedict import SqliteDict
from torch.distributed import get_rank, get_world_size
from tqdm import tqdm

from data_loader import (BucketingDataLoader, DistributedBucketingDataLoader,
                         DynamicBatchingLoader)
from gpt2_training.distributed import (all_gather_list,
                                       all_reduce_and_rescale_tensors)
from gpt2_training.eval_utils import eval_model_loss
from gpt2_training.train_utils import (boolean_string,
                                       get_eval_list_same_length, load_model,
                                       set_lr)
from lsp_model import (GPT2Config, GPT2LMHeadModel, GPT2LMHeadModelBackground,
                       GPT2LMHeadModelOdin, GPT2LMHeadModelOdinFix,
                       GPT2Tokenizer)
from sqlitedict_compress import my_decode, my_encode

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

INF = 100000000
CACHE_EMPTY_STEP = 10000
EVAL_STEP = 100000

#########################################################################
# Prepare Parser
##########################################################################

parser = argparse.ArgumentParser()
parser.add_argument(
    "--model_name_or_path",
    type=str,
    help="pretrained model name or path to local checkpoint",
)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--max_seq_length", type=int, default=128)

parser.add_argument(
    "--skip_eval", action="store_true", help="If true, skip evaluation."
)
parser.add_argument("--init_checkpoint", type=str)
parser.add_argument("--train_input_file", type=str)
parser.add_argument("--eval_input_file", type=str)

parser.add_argument("--eval_batch_size", type=int, default=1)
parser.add_argument("--learning_rate", type=float, default=1e-5)
parser.add_argument(
    "--num_optim_steps",
    type=int,
    default=1000000,
    help="new API specifies num update steps",
)
parser.add_argument(
    "--valid_step",
    type=int,
    default=10000,
    help="how many optim steps between validations",
)
parser.add_argument("--warmup_proportion", type=float, default=0.1)
parser.add_argument("--warmup_steps", type=int, default=16000)

parser.add_argument("--normalize_data", type=boolean_string, default=True)
parser.add_argument("--fp16", type=boolean_string, default=False)
parser.add_argument(
    "--lr_schedule",
    type=str,
    choices=["noam", "noamwd", "BERT", "None"],
    default="noam",
)
parser.add_argument("--loss_scale", type=float, default=0)
parser.add_argument("--no_token_id", type=boolean_string, default=True)

parser.add_argument("--output_dir", type=str)
parser.add_argument("--log_dir", type=str)
parser.add_argument(
    "--pbar", type=boolean_string, default=True, help="turn on progress bar"
)

# distributed
parser.add_argument("--local_rank", type=int, default=-1, help="for torch.distributed")
parser.add_argument("--config", help="JSON config file")


# do normal parsing
args = parser.parse_args()

if args.config is not None:
    # override argparse defaults by config JSON
    opts = json.load(open(args.config))
    for k, v in opts.items():
        if isinstance(v, str):
            # PHILLY ENV special cases
            if "PHILLY_JOB_DIRECTORY" in v:
                v = v.replace(
                    "PHILLY_JOB_DIRECTORY", os.environ["PHILLY_JOB_DIRECTORY"]
                )
            elif "PHILLY_LOG_DIRECTORY" in v:
                v = v.replace(
                    "PHILLY_LOG_DIRECTORY", os.environ["PHILLY_LOG_DIRECTORY"]
                )
        setattr(args, k, v)

    # command line should override config JSON
    argv = sys.argv[1:]
    overrides, _ = parser.parse_known_args(argv)
    for k, v in vars(overrides).items():
        if f"--{k}" in argv:
            setattr(args, k, v)
    setattr(args, "local_rank", overrides.local_rank)


if args.local_rank == -1:
    logger.info("CUDA available? {}".format(str(torch.cuda.is_available())))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_gpu = torch.cuda.device_count()
    args.device, args.n_gpu = device, n_gpu
else:
    # distributed training
    torch.cuda.set_device(args.local_rank)
    device = torch.device("cuda", args.local_rank)
    # Initializes the distributed backend which will take care of
    # sychronizing nodes/GPUs
    torch.distributed.init_process_group(backend="nccl")
    n_gpu = torch.distributed.get_world_size()
    args.device, args.n_gpu = device, 1
    logger.info(
        "device: {} n_gpu: {}, distributed training: {}, "
        "16-bits training: {}".format(
            device, n_gpu, bool(args.local_rank != -1), args.fp16
        )
    )

np.random.seed(args.seed)
torch.random.manual_seed(args.seed)
# torch.cuda.manual_seed(args.seed)
# if n_gpu > 0:
# torch.cuda.manual_seed_all(args.seed)

timestamp = datetime.datetime.now().strftime("%Y-%m-%d%H%M%S")
output_dir = join(
    args.output_dir,
    "{}_{}".format(
        args.eval_input_file.split("/")[-1], args.init_checkpoint.split("/")[-1]
    ),
)
log_dir = (
    args.log_dir if args.log_dir is not None and len(args.log_dir) > 0 else output_dir
)
if args.local_rank == -1 or get_rank() == 0:
    os.makedirs(output_dir, exist_ok=True)

logger.info("Input Argument Information")
args_dict = vars(args)
for a in args_dict:
    logger.info("%-28s  %s" % (a, args_dict[a]))


#########################################################################
# Prepare Data Set
##########################################################################
enc = GPT2Tokenizer.from_pretrained(args.model_name_or_path)

config = GPT2Config.from_json_file(join(args.model_name_or_path, "config.json"))


eval_dataloader_loss = DynamicBatchingLoader(
    args.eval_input_file,
    enc,
    args.normalize_data,
    args.eval_batch_size,
    args.max_seq_length,
)

eval_dataloader_gen = get_eval_list_same_length(
    args.eval_input_file, enc, args.eval_batch_size, True
)


#########################################################################
# Prepare Model and Optimizer
##########################################################################
model = load_model(
    GPT2LMHeadModelOdinFix(config), args.init_checkpoint, args, verbose=True
)


#########################################################################
# Inference !
##########################################################################


model = model.eval()
losses = SqliteDict(
    join(output_dir, "losses.sqlite"), encode=my_encode, decode=my_decode
)
hs = SqliteDict(join(output_dir, "hs.sqlite"), encode=my_encode, decode=my_decode)
gs = SqliteDict(join(output_dir, "gs.sqlite"), encode=my_encode, decode=my_decode)
with torch.no_grad():
    for item_idx, batch in enumerate(tqdm(eval_dataloader_loss)):
        try:
            batch = tuple(t.to(args.device) for t in batch)
            input_ids, position_ids, token_ids, label_ids, src_len, _ = batch
            if args.no_token_id:
                token_ids = None
            loss, ppl, h_prod, g_logits = model(
                input_ids, position_ids, token_ids, label_ids
            )
            losses[item_idx] = loss.detach().cpu().squeeze().numpy()
            hs[item_idx] = h_prod.detach().cpu().squeeze().max(dim=-1)[0].numpy()
            gs[item_idx] = g_logits.detach().cpu().squeeze().numpy()
        except Exception as ex:
            print(ex)
            losses[item_idx] = None
            hs[item_idx] = None
            gs[item_idx] = None
        finally:
            if item_idx % 100 == 0:
                losses.commit()
                hs.commit()
                gs.commit()


losses.commit()
hs.commit()
gs.commit()
losses.close()
hs.close()
gs.close()
