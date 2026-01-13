import subprocess
from pathlib import Path

OUT_DIR = Path("runs/datascale_single_epoch")

'''
Run experiments varying dataset size and number of iterations to keep total epochs constant.
(really, just want one epoch to understand the effect of data size)
'''

DATASET_SIZES = [100000, 300000, 1000000, 3000000, 10000000] # up to 10M tokens total
PARENT_SIZE = 10000000 # 10M tokens; the other sizes are truncated versions of this
MAX_ITERS_LIST = [100, 300, 1000, 3000, 10000] #up to 10M tokens total
# this is hardcoded ^^  for now, but should probably be computed based on DATASET_SIZES and total tokens
SEEDS = [0]#, 1, 2, 3, 4] # multiple seeds for averaging

run_name = "hhmm-6-4-10M-se-" # include parent size in run name
# append a timestamp to run_name to make it unique
import time
run_name += str(int(time.time()))

def run_experiment(dataset_size, seed):    

    # get the truncated dataset
    dataset = f"hhmm_uni_6_4_{PARENT_SIZE}"
    if dataset_size != PARENT_SIZE:
        dataset += f"_subset_{dataset_size}"
        
    # get max iters corresponding to dataset size
    max_iters = dataset_size // 1000 # HARDCODED FOR NOW: 1k tokens per iter: JE

    cmd = [
        "python", "train.py",
        "config/train_synthetic_data.py",      # base config
        #f"--out_dir={out_dir}",          # nanoGPT supports this override
        f"--seed={seed}", # set seed for reproducibility 
        "--dataset=" + dataset, # override dataset
        f"--max_iters={max_iters}", # override max iters
        f"--wandb_run_name={run_name}-{max_iters}-{seed}", # unique wandb name
        f"--warmup_iters={max_iters // 20}", # override warmup iters
        f"--lr_decay_iters={max_iters}", # override lr decay iters
    ]
    print("Running:", " ".join(map(str, cmd)))
    subprocess.run(cmd, check=True)

if __name__ == "__main__":
    for dataset_size in DATASET_SIZES:
        for seed in SEEDS:
            run_experiment(dataset_size, seed)