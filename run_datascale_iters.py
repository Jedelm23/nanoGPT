import subprocess
from pathlib import Path

OUT_DIR = Path("runs/datascale_iters")

MAX_ITERS_LIST = [1000]#, 300, 1000, 3000, 10000] #up to 10M tokens total
SEEDS = [1]#[1, 2, 3, 4] # ignore for now; going to do multiple seeds later

run_name = "hhmm-6-4-1M-"
# append a timestamp to run_name to make it unique
import time
run_name += str(int(time.time()))

def run_experiment(max_iters, seed):    
    tag = f"maxsteps_{max_iters}_seed_{seed}"
    out_dir = OUT_DIR / tag
    out_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        "python", "train.py",
        "config/train_synthetic_data.py",      # base config
        #f"--out_dir={out_dir}",          # nanoGPT supports this override
        f"--seed={seed}", # set seed for reproducibility 
        f"--max_iters={max_iters}", # override max iters
        f"--wandb_run_name={run_name}-{max_iters}-{seed}", # unique wandb name
        f"--warmup_iters={max_iters // 20}", # override warmup iters
        f"--lr_decay_iters={max_iters}", # override lr decay iters
    ]
    print("Running:", " ".join(map(str, cmd)))
    subprocess.run(cmd, check=True)

if __name__ == "__main__":
    for max_iters in MAX_ITERS_LIST:
        for seed in SEEDS:
            run_experiment(max_iters, seed)