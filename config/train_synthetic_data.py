# config for training on synthetic data
# launch as the following (e.g. in a screen session)
# $ torchrun --standalone --nproc_per_node=8 train.py config/train_synthetic_data.py

wandb_log = True
wandb_project = 'hhmm_scaling'
wandb_run_name='hhmm-100k-tiny'

# training setting
device = 'mps'
compile = False

# dataset stuff
L = 12
S = 2
n = 100000
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
n_embed = 128
dropout = 0.0

# these are just guesses
learning_rate = 3e-3
min_lr = 3e-5

max_iters = 500 # ~23 epoch for 100k data
lr_decay_iters = max_iters

# eval stuff
eval_interval = 100
eval_iters = 50
log_interval = 10

# weight decay
weight_decay = 1e-2
