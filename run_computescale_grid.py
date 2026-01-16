import subprocess
from pathlib import Path
import time

OUT_DIR = Path("runs/computescale_grid")

"Notes: make sure that the dataset is large enough so that the maximum total tokens seen is less than one epoch "

MAX_ITERS_LIST = [50, 100, 300, 1000, 3000] # control D, tokens seen (max_iters * batch_size * block_size * gradient_accumulation_steps * ddp_world_size)
N_LAYERS_LIST = [2, 4, 8, 12]    # control N, model capacity
N_EMBD_FACTOR = 32 # factor to multiply n_embd to scale with n_layers, n_head scales with n_layers
SEEDS = [0] # repeat for multiple data points

run_name = "hhmm-6-4-10M-"

# append a timestamp to run_name to make it unique
run_name += str(int(time.time()))
# append another stamp to indicate computescale
run_name += "-cs"

def run_experiment(max_iters, n_layers, n_heads, n_embd, seed):    
    #tag = f"maxsteps_{max_iters}_layers_{n_layers}_seed_{seed}"
    #out_dir = OUT_DIR / tag
    #out_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        "python", "train.py",
        "config/train_synthetic_data.py",      # base config
        f"--seed={seed}", # set seed for reproducibility 
        f"--max_iters={max_iters}", # override max iters
        f"--n_layer={n_layers}", # override number of layers
        f"--n_head={n_heads}", # set n_head = n_layer
        f"--n_embd={n_embd}", # scale n_embd with n_layers
        f"--wandb_run_name={run_name}-{max_iters}-{n_layers}-{seed}", # unique wandb name
        f"--warmup_iters={max_iters // 20}", # override warmup iters
        f"--lr_decay_iters={max_iters}", # override lr decay iters
    ]
    print("Running:", " ".join(map(str, cmd)))
    subprocess.run(cmd, check=True)

if __name__ == "__main__":
    for max_iters in MAX_ITERS_LIST:
        for n_layers in N_LAYERS_LIST:
            n_heads = n_layers # set n_head = n_layer
            n_embd = n_layers * N_EMBD_FACTOR # scale n_embd with n_layers
            for seed in SEEDS:
                run_experiment(max_iters, n_layers, n_heads, n_embd, seed)