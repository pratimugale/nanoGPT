"""
This training script can be run both on a single gpu in debug mode,
and also in a larger training run with distributed data parallel (ddp).

To run on a single GPU, example:
$ python train.py --batch_size=32 --compile=False

To run with DDP on 4 gpus on 1 node, example:
$ torchrun --standalone --nproc_per_node=4 train.py

To run with DDP on 4 gpus across 2 nodes, example:
- Run on the first (master) node with example IP 123.456.123.456:
$ torchrun --nproc_per_node=8 --nnodes=2 --node_rank=0 --master_addr=123.456.123.456 --master_port=1234 train.py
- Run on the worker node:
$ torchrun --nproc_per_node=8 --nnodes=2 --node_rank=1 --master_addr=123.456.123.456 --master_port=1234 train.py
(If your cluster does not have Infiniband interconnect prepend NCCL_IB_DISABLE=1)
"""

import os
import time
import math
import pickle
from contextlib import nullcontext
from tqdm import tqdm
import pandas as pd
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(current_dir, "../src/qaoa-gpt"))


from datetime import datetime
import numpy as np
import torch
torch.set_float32_matmul_precision('high')
import json
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group

from nanoGPT.model_pad_gemb import GPTConfig as GPTConfig_gemb
from nanoGPT.model_pad_gemb import GPT as GPT_gemb

from nanoGPT.model_pad import GPTConfig as GPTConfig_nogemb
from nanoGPT.model_pad import GPT as GPT_nogemb

from util import generate_circ_from_df, eval_adapt_gpt_circ_jl

# val_sampled_df = pd.read_pickle('data/qaoa_n10w_012325_v7/test_run_df.pkl')
# val_graph_emb_np = np.load(
#     'data/qaoa_n10w_012325_v7/feather_emb_d500.npy'
# )
# val_meta = pd.read_pickle("data/qaoa_n10w_012325_v7/meta.pkl")
# val_emb_graph_id_to_idx_dict = val_meta['emb_graph_id_to_idx_dict']



n_epochs = 20 # (approximately)
eval_ar_every = 1000

# -----------------------------------------------------------------------------
# default config values designed to train a gpt2 (124M) on OpenWebText
# I/O
out_dir = 'out'
eval_interval = 20_000
log_interval = 1
eval_iters = 200
eval_only = False # if True, script exits right after the first eval
always_save_checkpoint = True # if True, always save a checkpoint after each eval
init_from = 'scratch' # 'scratch' or 'resume' or 'gpt2*'
# wandb logging
wandb_log = True 
wandb_project = 'qaoa-gpt-sat'
wandb_run_name = 'baseline'
# data
# model
n_layer = 12
n_head = 12
n_embd = 768
dropout = 0.0 # for pretraining 0 is good, for finetuning try 0.1+
bias = False # do we use bias inside LayerNorm and Linear layers?
# adamw optimizer
learning_rate = 6e-4 # max learning rate
max_iters = 600000 # total number of training iterations
weight_decay = 1e-1
beta1 = 0.9
beta2 = 0.95
grad_clip = 1.0 # clip gradients at this value, or disable if == 0.0
# learning rate decay settings
decay_lr = True # whether to decay the learning rate
warmup_iters = 2000 # how many steps to warm up for
lr_decay_iters = 600000 # should be ~= max_iters per Chinchilla
min_lr = 6e-5 # minimum learning rate, should be ~= learning_rate/10 per Chinchilla
# DDP settings
backend = 'nccl' # 'nccl', 'gloo', etc.
# system
device = 'cuda' # examples: 'cpu', 'cuda', 'cuda:0', 'cuda:1' etc., or try 'mps' on macbooks
dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16' # 'float32', 'bfloat16', or 'float16', the latter will auto implement a GradScaler
compile = True # use PyTorch 2.0 to compile the model to be faster
# -----------------------------------------------------------------------------
# Configuration and Defaults
dataset = 'n12_standard_mixer'
out_dir = 'out'
batch_size = 64
gradient_accumulation_steps = 40
block_size = 512
eval_interval = 250
eval_ar_every = 250
eval_ar_locally = False
pool_type = "qaoa_mixer"
use_graph_emb = False
graph_emb_dim = 500
n_nodes = 12
rounding_digits = 2
mask_formula_loss = True
# -----------------------------------------------------------------------------
# configurator.py is in the same directory as this script
script_dir = os.path.dirname(os.path.abspath(__file__))
configurator_path = os.path.join(script_dir, 'configurator.py')
exec(open(configurator_path).read()) # overrides from command line or config file

# Collect ALL configuration keys after overrides for logging
config_keys = [k for k,v in globals().items() if not k.startswith('_') and isinstance(v, (int, float, bool, str))]
config = {k: globals()[k] for k in config_keys} # will be useful for logging
# -----------------------------------------------------------------------------


if use_graph_emb:
    print("Training model with graph embeddings")
    model_suf = 'gemb'
else:
    print("Training model with NO graph embeddings")
    model_suf = 'nogemb'

# various inits, derived attributes, I/O setup
ddp = int(os.environ.get('RANK', -1)) != -1 # is this a ddp run?
if ddp:
    init_process_group(backend=backend)
    ddp_rank = int(os.environ['RANK'])
    ddp_local_rank = int(os.environ['LOCAL_RANK'])
    ddp_world_size = int(os.environ['WORLD_SIZE'])
    device = f'cuda:{ddp_local_rank}'
    torch.cuda.set_device(device)
    master_process = ddp_rank == 0 # this process will do logging, checkpointing etc.
    seed_offset = ddp_rank # each process gets a different seed
    # world_size number of processes will be training simultaneously, so we can scale
    # down the desired gradient accumulation iterations per process proportionally
    assert gradient_accumulation_steps % ddp_world_size == 0
    gradient_accumulation_steps //= ddp_world_size
else:
    # if not ddp, we are running on a single gpu, and one process
    master_process = True
    seed_offset = 0
    ddp_world_size = 1
tokens_per_iter = gradient_accumulation_steps * ddp_world_size * batch_size * block_size
print(f"tokens per iteration will be: {tokens_per_iter:,}")

if master_process:
    os.makedirs(out_dir, exist_ok=True)
torch.manual_seed(1337 + seed_offset)
torch.backends.cuda.matmul.allow_tf32 = True # allow tf32 on matmul
torch.backends.cudnn.allow_tf32 = True # allow tf32 on cudnn
device_type = 'cuda' if 'cuda' in device else 'cpu' # for later use in torch.autocast
# note: float16 data type will automatically use a GradScaler
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

# poor man's data loader
data_dir = globals().get('data_dir', os.path.join('data', dataset))

mmap='r'
mmap=None
print(f'Opening data (mmap mode: {mmap})...')
train_data = np.load(
    os.path.join(data_dir, 'train.npy'), mmap_mode=mmap
)
val_data = np.load(
    os.path.join(data_dir, 'val.npy'), mmap_mode=mmap
)
if use_graph_emb:
    graph_emb_np = np.load(
        os.path.join(data_dir, 'feather_emb_d500.npy'), mmap_mode=mmap
    )
    emb_dim = graph_emb_np.shape[1]
else:
    graph_emb_np = None
    emb_dim = 0

logging_json_file = os.path.join(out_dir, 'train_log.json')
logging_list = []

def get_batch(split):

    if split == 'train':
        data = train_data
        emb_idx_data = train_data_formula_idx_list
    else:
        data = val_data
        emb_idx_data = val_data_formula_idx_list
    ix = np.random.randint(low=0, high=data.shape[0]-1, size=batch_size)
    data_batch_np = data[ix]
    if use_graph_emb:
        graph_emb_data = torch.from_numpy(graph_emb_np[emb_idx_data[ix]].astype(np.float32))
    else:
        graph_emb_data = None

    #print(f"Get batch graph_emb_data shape: {graph_emb_data.shape}, {graph_emb_data.dtype}")
    x = torch.from_numpy(data_batch_np[:, :1, :].astype(np.int64)).flatten(1)
    y = torch.from_numpy(data_batch_np[:, 1:2, :].astype(np.int64)).flatten(1)
    if device_type == 'cuda':
        # pin arrays x,y, which allows us to move them to GPU asynchronously (non_blocking=True)
        x, y = x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device, non_blocking=True)
        if use_graph_emb:
            graph_emb_data = graph_emb_data.pin_memory().to(device, non_blocking=True).to(torch.bfloat16)
    else:
        x, y = x.to(device), y.to(device)
        if use_graph_emb:
            graph_emb_data = graph_emb_data.to(device)
    #print(f"graph_emb_data dtype: {graph_emb_data.dtype}\n\n\n")

    if mask_formula_loss:
        # Find index of end_of_formula token for each sequence in the batch
        # stoi is available globally after meta is loaded
        eof_token_id = meta['stoi']['end_of_formula']
        for b in range(x.shape[0]):
            eof_indices = (x[b] == eof_token_id).nonzero(as_tuple=True)[0]
            if len(eof_indices) > 0:
                eof_idx = eof_indices[0].item()
                # Mask y from index 0 to eof_idx - 1 (the formula part)
                # Setting to 0 relies on ignore_index=0 in the model's loss function
                y[b, :eof_idx] = 0

    return x, y, graph_emb_data


# ADAPT GPT-specific code
#-------------------------
#-------------------------
#-------------------------

def get_test_energies_df():
    model.eval()
    print("Generating circuits with current state of the model (sequential)")
    
    all_samples_for_eval = []
    
    # Process sequentially for now
    for sample_i, (idx, row) in enumerate(val_sampled_df.iterrows()):
        formula_id = row['formula_id']
        full_target_seq = row['token_seq_round_d2']
        if isinstance(full_target_seq, str):
            import ast
            full_target_seq = ast.literal_eval(full_target_seq)
            
        eof_idx = full_target_seq.index('end_of_formula')
        formula_tokens_raw = full_target_seq[:eof_idx + 1]
        formula_tokens_ids = [meta['stoi'][t] for t in formula_tokens_raw]
        
        x_input = torch.tensor(formula_tokens_ids, dtype=torch.long, device=device).unsqueeze(0)
        
        if use_graph_emb:
            formula_idx = val_emb_formula_id_to_idx_dict[formula_id]
            cur_graph_emb = torch.from_numpy(val_graph_emb_np[formula_idx].astype(np.float32)).to(device).unsqueeze(0)
            if device_type == 'cuda':
                cur_graph_emb = cur_graph_emb.to(torch.bfloat16)
        else:
            cur_graph_emb = None
            
        q_circs = []
        for c_i in range(5): # num_samples=5
            with torch.no_grad():
                with ctx:
                    if use_graph_emb:
                        y_out = model.generate(
                            x_input, 
                            cur_graph_emb, 
                            max_new_tokens=150, 
                            temperature=0.1, 
                            top_k=200, 
                            eos_id=meta['stoi']['eos']
                        )
                    else:
                        y_out = model.generate(
                            x_input,
                            150,  # max_new_tokens is positional in model_pad.py GPT
                            temperature=0.1,
                            top_k=200,
                            eos_id=meta['stoi']['eos']
                        )
            
            generated_only = y_out[0].tolist()[len(formula_tokens_ids):]
            circuit_tokens = [str(meta['itos'][t]) for t in generated_only if t != meta['stoi']['pad'] and t != meta['stoi']['eos']]
            q_circs.append(circuit_tokens)
            
        gt_circuit_raw = full_target_seq[eof_idx + 1:]
        
        def clean_circ(tokens):
            cleaned = []
            for t in tokens:
                t_str = str(t)
                if t_str in ['bos', 'eos', 'pad', 'new_layer_p', 'end_of_formula']:
                    if t_str == 'new_layer_p':
                        cleaned.append(t_str)
                    continue
                if t_str.startswith('op_'):
                    try:
                        cleaned.append(int(t_str.split('_')[1]))
                    except:
                        cleaned.append(t_str)
                else:
                    try:
                        cleaned.append(float(t_str))
                    except:
                        cleaned.append(t_str)
            return cleaned

        filename = str(row['filename'])
        formula_type = 'balanced' if 'balanced' in filename.lower() else 'random'
        formula_list_val = row['formula_list']
        if isinstance(formula_list_val, str):
            import ast
            formula_list_val = ast.literal_eval(formula_list_val)

        sample_for_eval = {
            'graph_prefix': f"EVAL_ITER_{iter_num}__{sample_i}",
            'type': formula_type,
            'formula_jl': formula_list_val,
            'energy_gurobi': float(row['ground_truth_energy']),
            'adapt_circuit': clean_circ(gt_circuit_raw), 
            'q_circuits': [clean_circ(c) for c in q_circs]
        }
        all_samples_for_eval.append(sample_for_eval)

    ## Evaluating energies with ADAPT.jl
    print("Evaluating energies with ADAPT.jl")
    root_dir = os.path.dirname(current_dir)
    temp_dir = os.path.join(root_dir, 'temp_data')
    os.makedirs(temp_dir, exist_ok=True)
    
    in_file = os.path.join(temp_dir, 'val_eval_input_wandb.json')
    out_file = os.path.join(temp_dir, 'val_eval_output_wandb.json')
    
    import json
    with open(in_file, 'w') as f:
        json.dump(all_samples_for_eval, f, indent=4)
        
    julia_script = 'src/qaoa-gpt/adapt_gpt_eval_energy.jl'
    cmd = f"julia -t 4 --project=ADAPT.jl {julia_script} {in_file} {out_file} {val_n_nodes} {pool_type}"
    
    import subprocess
    subprocess.run(cmd, shell=True, check=True)
    
    energies_jl_gc_df = pd.read_json(out_file)
    return energies_jl_gc_df

def eval_model_ar():

    print("Model evaluation...")
    test_energies_df = get_test_energies_df()


    sum_valid_best = 0
    sum_valid_avg = 0.0
    sum_ar_best = 0.0
    sum_ar_avg = 0.0
    
    for idx, row in test_energies_df.iterrows():
        res = row['result_quality']
        sample_svr = res.get('sample_svr', 0.0)
        
        sum_valid_avg += sample_svr
        if sample_svr > 0:
            sum_valid_best += 1
            sum_ar_best += res.get('best_ar_qaoa_gpt', 0.0)
            sum_ar_avg += res.get('avg_ar_qaoa_gpt', 0.0)
            
    total = len(test_energies_df)
    
    best_valid_rate = sum_valid_best / total if total > 0 else 0.0
    avg_valid_rate = sum_valid_avg / total if total > 0 else 0.0
    
    best_ar = sum_ar_best / sum_valid_best if sum_valid_best > 0 else 0.0
    avg_ar = sum_ar_avg / sum_valid_best if sum_valid_best > 0 else 0.0

    return test_energies_df, best_ar, avg_ar, 1.0 - best_valid_rate, 1.0 - avg_valid_rate

#-------------------------
#-------------------------
#-------------------------

# init these up here, can override if init_from='resume' (i.e. from a checkpoint)
iter_num = 0
best_val_loss = 1e9

# attempt to derive vocab_size from the dataset
meta_path = os.path.join(data_dir, 'meta.pkl')
meta_vocab_size = None
if os.path.exists(meta_path):
    with open(meta_path, 'rb') as f:
        meta = pickle.load(f)
    meta_vocab_size = meta['vocab_size']
    print(f"found vocab_size = {meta_vocab_size} (inside {meta_path})")

# For formula lcg graph embeddings
emb_formula_id_to_idx_dict = meta.get('emb_formula_id_to_idx_dict')
emb_formula_idx_to_id_dict = meta.get('emb_formula_idx_to_id_dict')
train_data_formula_idx_list = np.array(meta.get('train_data_formula_idx_list'))
val_data_formula_idx_list = np.array(meta.get('val_data_formula_idx_list'))


# For AR validation
##########################################
pkl_path = os.path.join(data_dir, 'combined_res_tok_shf_val_df.pkl')
csv_path = os.path.join(data_dir, 'combined_res_tok_shf_val_df.csv')

if os.path.exists(pkl_path):
    print(f"Loading validation data from Pickle: {pkl_path}")
    val_sampled_df = pd.read_pickle(pkl_path)
elif os.path.exists(csv_path):
    print(f"Loading validation data from CSV: {csv_path}")
    val_sampled_df = pd.read_csv(csv_path)
    # Ensure types are correct for CSV
    if 'has_emb' in val_sampled_df.columns:
        val_sampled_df['has_emb'] = val_sampled_df['has_emb'].map({'True': True, 'False': False, True: True, False: False})
    if 'n_nodes' in val_sampled_df.columns:
        val_sampled_df['n_nodes'] = pd.to_numeric(val_sampled_df['n_nodes'], errors='coerce')

    # Convert strings to actual lists/dicts if loaded from CSV
    import ast
    for col in [f'token_seq_round_d{rounding_digits}', 'formula_list']:
        if col in val_sampled_df.columns:
            print(f"\tConverting {col} from string to list...")
            val_sampled_df[col] = val_sampled_df[col].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)
else:
    raise FileNotFoundError(f"Could not find validation data at {pkl_path} or {csv_path}")

val_sampled_df = val_sampled_df[
    val_sampled_df['has_emb']
]

_eval_ar_n_samples = globals().get('eval_ar_n_samples', 50)
if len(val_sampled_df) > _eval_ar_n_samples:
    print(f"Subsampling validation set from {len(val_sampled_df)} down to {_eval_ar_n_samples} formulas for efficient AR evaluation.")
    val_sampled_df = val_sampled_df.sample(n=_eval_ar_n_samples, random_state=42)

val_n_nodes = int(val_sampled_df['n_nodes'].max())
val_graph_emb_np = graph_emb_np
val_meta = meta
val_emb_formula_id_to_idx_dict = val_meta.get('emb_formula_id_to_idx_dict')

##########################################


# model init
model_args = dict(n_layer=n_layer, n_head=n_head, n_embd=n_embd, block_size=block_size,
                  bias=bias, vocab_size=None, dropout=dropout) # start with model_args from command line
if init_from == 'scratch':
    # init a new model from scratch
    print("Initializing a new model from scratch")
    # determine the vocab size we'll use for from-scratch training
    if meta_vocab_size is None:
        print("defaulting to vocab_size of GPT-2 to 50304 (50257 rounded up for efficiency)")
    model_args['vocab_size'] = meta_vocab_size if meta_vocab_size is not None else 50304
    if use_graph_emb:
        gptconf = GPTConfig_gemb(**model_args)
        model = GPT_gemb(gptconf)
    else:
        gptconf = GPTConfig_nogemb(**model_args)
        model = GPT_nogemb(gptconf)
elif init_from == 'resume':
    print(f"Resuming training from {out_dir}")
    # resume training from a checkpoint.
    ckpt_path = os.path.join(out_dir, 'ckpt.pt')
    checkpoint = torch.load(ckpt_path, map_location=device)
    checkpoint_model_args = checkpoint['model_args']
    # force these config attributes to be equal otherwise we can't even resume training
    # the rest of the attributes (e.g. dropout) can stay as desired from command line
    for k in ['n_layer', 'n_head', 'n_embd', 'block_size', 'bias', 'vocab_size']:
        model_args[k] = checkpoint_model_args[k]
    # create the model
    if use_graph_emb:
        gptconf = GPTConfig_gemb(**model_args)
        model = GPT_gemb(gptconf)
    else:
        gptconf = GPTConfig_nogemb(**model_args)
        model = GPT_nogemb(gptconf)
    state_dict = checkpoint['model']
    # fix the keys of the state dictionary :(
    # honestly no idea how checkpoints sometimes get this prefix, have to debug more
    unwanted_prefix = '_orig_mod.'
    for k,v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
    model.load_state_dict(state_dict)
    iter_num = checkpoint['iter_num']
    best_val_loss = checkpoint['best_val_loss']
elif init_from.startswith('gpt2'):
    print(f"Initializing from OpenAI GPT-2 weights: {init_from}")
    # initialize from OpenAI GPT-2 weights
    override_args = dict(dropout=dropout)
    model = GPT.from_pretrained(init_from, override_args)
    # read off the created config params, so we can store them into checkpoint correctly
    for k in ['n_layer', 'n_head', 'n_embd', 'block_size', 'bias', 'vocab_size']:
        model_args[k] = getattr(model.config, k)
# crop down the model block size if desired, using model surgery
if block_size < model.config.block_size:
    model.crop_block_size(block_size)
    model_args['block_size'] = block_size # so that the checkpoint will have the right value
model.to(device)

# initialize a GradScaler. If enabled=False scaler is a no-op
scaler = torch.cuda.amp.GradScaler(enabled=(dtype == 'float16'))

# optimizer
optimizer = model.configure_optimizers(weight_decay, learning_rate, (beta1, beta2), device_type)
if init_from == 'resume':
    optimizer.load_state_dict(checkpoint['optimizer'])
checkpoint = None # free up memory

# compile the model
if compile:
    print("compiling the model... (takes a ~minute)")
    unoptimized_model = model
    model = torch.compile(model) # requires PyTorch 2.0

# wrap model into DDP container
if ddp:
    model = DDP(model, device_ids=[ddp_local_rank])

# helps estimate an arbitrarily accurate loss over either split using many batches
@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y, cur_graph_emb = get_batch(split)
            with ctx:
                if use_graph_emb:
                    logits, loss = model(X, cur_graph_emb, Y)
                else:
                    logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

# learning rate decay scheduler (cosine with warmup)
def get_lr(it):
    # 1) linear warmup for warmup_iters steps
    if it < warmup_iters:
        return learning_rate * (it + 1) / (warmup_iters + 1)
    # 2) if it > lr_decay_iters, return min learning rate
    if it > lr_decay_iters:
        return min_lr
    # 3) in between, use cosine decay down to min learning rate
    decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio)) # coeff ranges 0..1
    return min_lr + coeff * (learning_rate - min_lr)

# logging
if wandb_log and master_process:
    import wandb
    wandb.init(project=wandb_project, name=wandb_run_name, config=config)
    
    # Define 'iter' as the default x-axis for train and val metrics
    wandb.define_metric("iter")
    wandb.define_metric("train/*", step_metric="iter")
    wandb.define_metric("val/*", step_metric="iter")

# training loop
X, Y, cur_graph_emb = get_batch('train') # fetch the very first batch
#print(f"From training loop cur_graph_emb: {cur_graph_emb.shape}")
t0 = time.time()
local_iter_num = 0 # number of iterations in the lifetime of this process
raw_model = model.module if ddp else model # unwrap DDP container if needed
running_mfu = -1.0
#while True:


cur_best_ar = None
cur_avg_ar = None
cur_best_er = None
cur_avg_er = None

dataset_n_batches = train_data.shape[0]//batch_size
pbar = tqdm(list(range(n_epochs * dataset_n_batches)))

for i in pbar:
    
    # determine and set the learning rate for this iteration
    lr = get_lr(iter_num) if decay_lr else learning_rate
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    
    # evaluate the loss on train/val sets and write checkpoints
    if iter_num % eval_interval == 0 and master_process:
        losses = estimate_loss()
        saving_model_name = f'ckpt_{i}_{model_suf}.pt'
        if iter_num % eval_ar_every == 0 and iter_num > 0:
            print("\tEvaluating model ER and AR...")
            
            # Temporarily disable Dropout for generation
            model.eval()
            cur_test_energies_df, cur_best_ar, cur_avg_ar, cur_best_er, cur_avg_er = eval_model_ar()
            model.train()
            
            print(f"\tCurrent avg_ar: {cur_avg_ar}, avg_er: {cur_avg_er}\n\n")
            cur_ar_str = str(cur_avg_ar).replace('.', '_')
            cur_er_str = str(cur_avg_er).replace('.', '_')
            saving_model_name = f'ckpt_{i}_{model_suf}__ar_{cur_ar_str}__er_{cur_er_str}.pt'

            logging_list.append(
                {
                    'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                    'model_dir': out_dir,
                    'iter_num': iter_num,
                    'cur_gpt_loss_train': losses['train'].item(),
                    'cur_gpt_loss_val': losses['val'].item(),
                    'cur_best_ar_val': cur_best_ar,
                    'cur_avg_ar_val': cur_avg_ar,
                    'cur_best_er_val': cur_best_er,
                    'cur_avg_er_val': cur_avg_er,
                    'cur_val_df': cur_test_energies_df.to_json(),
                }
            )
            with open(logging_json_file, 'w') as f:
                json.dump(logging_list, f)
            
        
        #print(f"step {iter_num}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
        pbar.set_description(f"train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
        if wandb_log:
            log_dict = {
                "iter": iter_num,
                "train/loss": losses['train'],
                "val/loss": losses['val'],
                "lr": lr,
                "mfu": running_mfu*100, # convert to percentage
            }
            if cur_best_ar is not None:
                log_dict["val/best_ar"] = cur_best_ar
                log_dict["val/avg_ar"] = cur_avg_ar
                
                log_dict["val/best_error_rate"] = cur_best_er
                log_dict["val/avg_error_rate"] = cur_avg_er
                
                log_dict["val/best_svr"] = 1.0 - cur_best_er
                log_dict["val/avg_svr"] = 1.0 - cur_avg_er
            wandb.log(log_dict)
        #if losses['val'] < best_val_loss or always_save_checkpoint:
        if losses['val'] < best_val_loss:
            best_val_loss = losses['val']
        if iter_num > 0:
            checkpoint = {
                'model': raw_model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'model_args': model_args,
                'iter_num': iter_num,
                'best_val_loss': best_val_loss,
                'config': config,
            }
            print(f"saving checkpoint to {out_dir}")
            torch.save(checkpoint, os.path.join(out_dir, saving_model_name))
        #torch.save(checkpoint, os.path.join(out_dir, 'ckpt_overfit.pt'))
        
    if iter_num == 0 and eval_only:
        break

    # forward backward update, with optional gradient accumulation to simulate larger batch size
    # and using the GradScaler if data type is float16
    for micro_step in range(gradient_accumulation_steps):
        if ddp:
            # in DDP training we only need to sync gradients at the last micro step.
            # the official way to do this is with model.no_sync() context manager, but
            # I really dislike that this bloats the code and forces us to repeat code
            # looking at the source of that context manager, it just toggles this variable
            model.require_backward_grad_sync = (micro_step == gradient_accumulation_steps - 1)
        with ctx:
            if use_graph_emb:
                logits, loss = model(X, cur_graph_emb, Y)
            else:
                logits, loss = model(X, Y)
            loss = loss / gradient_accumulation_steps # scale the loss to account for gradient accumulation
        # immediately async prefetch next batch while model is doing the forward pass on the GPU
        X, Y, cur_graph_emb = get_batch('train')
        # backward pass, with gradient scaling if training in fp16
        scaler.scale(loss).backward()
    # clip the gradient
    if grad_clip != 0.0:
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
    # step the optimizer and scaler if training in fp16
    scaler.step(optimizer)
    scaler.update()
    # flush the gradients as soon as we can, no need for this memory anymore
    optimizer.zero_grad(set_to_none=True)

    # timing and logging
    t1 = time.time()
    dt = t1 - t0
    t0 = t1
    if iter_num % log_interval == 0 and master_process:
        # get loss as float. note: this is a CPU-GPU sync point
        # scale up to undo the division above, approximating the true total loss (exact would have been a sum)
        lossf = loss.item() * gradient_accumulation_steps
        if local_iter_num >= 5: # let the training loop settle a bit
            mfu = raw_model.estimate_mfu(batch_size * gradient_accumulation_steps, dt)
            running_mfu = mfu if running_mfu == -1.0 else 0.9*running_mfu + 0.1*mfu
        #print(f"iter {iter_num}: loss {lossf:.4f}, time {dt*1000:.2f}ms, mfu {running_mfu*100:.2f}%")
    iter_num += 1
    local_iter_num += 1

    # termination conditions
    if iter_num > max_iters:
        break

if ddp:
    destroy_process_group()
