# config for training on synthetic data
# launch as the following (e.g. in a screen session)
# $ torchrun --standalone --nproc_per_node=8 train.py config/train_synthetic_data.py

wandb_log = True
wandb_project = 'hhmm_scaling'
wandb_run_name='hhmm-6-4-1M'

# training setting
device = 'mps'
compile = False

# dataset stuff
L = 6
S = 4
n = 1000000
dataset = f"hhmm_uni_{L}_{S}_{n}"
data_dir = "data"

# total batch size: 
# batch_size * block_size * gradient_accumulation_steps * num_gpus

# setting for 100k data total
batch_size = 16
block_size = 64 # context length
vocab_size = S**(L-1) # S**(L-1), S=L=4
bias = False # use bias terms in LayerNorm and Linear layers
gradient_accumulation_steps = 1

# model stuff
n_layer = 4
n_head = 4
n_embd = 128 #n_embd NOT n_embed
dropout = 0.0

# these are just guesses
learning_rate = 3e-3
min_lr = 3e-5

max_iters = 3000 
warmup_iters = max_iters // 20 # warm up for 5% of training
warmdown_ratio = 0.8 # warmdown over the last 80% of training
lr_decay_iters = max_iters # decay over whole training

# eval stuff
eval_interval = 100 # also controls logs to wandb
eval_iters = 50
log_interval = 10

# weight decay
weight_decay = 1e-2
