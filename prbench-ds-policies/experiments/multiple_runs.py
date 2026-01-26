import subprocess

def run_experiment(env_name, seed):
    subprocess.run([
        "python",
        "experiments/collect_demos_ds.py",
        f"env={env_name}",
        f"seed={seed}",
    ])

if __name__ == "__main__":
    num_experiments = 110
    env_name = "transport3d-o2"
    for seed in range(num_experiments):
        run_experiment(env_name, seed)