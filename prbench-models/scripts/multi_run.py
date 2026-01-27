import subprocess

def run_experiment(env_name, seed):
    subprocess.run([
        "python",
        "scripts/planning_data_dynamics3d_prbench.py",
        f"seed={seed}",
    ])

if __name__ == "__main__":
    num_experiments = 100
    for seed in range(num_experiments):
        run_experiment(seed)