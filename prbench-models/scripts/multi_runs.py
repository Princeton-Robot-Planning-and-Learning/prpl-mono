import subprocess

def run_experiment(seed):
    subprocess.run([
        "python",
        "scripts/planning_data_dynamics3d_prbench.py",
        "--seed",
        str(seed),
    ])

if __name__ == "__main__":
    num_experiments = 50
    for seed in range(220, 220 + num_experiments):
        run_experiment(seed)