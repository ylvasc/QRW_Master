import os
import time
import numpy as np
from tqdm import tqdm
import uuid
import subprocess

results_base_dir = "results1"
os.makedirs(results_base_dir, exist_ok=True)

alpha_values = np.linspace(0.1, 2.0, 25)  # or 50 steps???
beta_values = np.linspace(0.1, 2.0, 25)

# alpha_values = np.linspace(0.01, 1.0, 25)    #also test
# beta_values = np.linspace(0.01, 1.0, 25)

n_sim = 1000  # Number of states
N_iter = 50  # Number of iterations per (alpha, beta) pair
N = 200       # Number of Grover iterations
max_jobs = 50  # Limit on concurrent jobs
active_processes = []

for a in alpha_values:
    for b in beta_values:
        unique_id = str(uuid.uuid4())[:8]
        file_name = f"results_{unique_id}.pkl"
        save_path = os.path.join(results_base_dir, file_name)

        # delete this after test
        print(f"Running with alpha={a}, beta={b}, n={n_sim}, N_iter={N_iter}, N={N}, save_path={save_path}")

        process = subprocess.Popen(["python", "simulator.py", str(a), str(b), str(n_sim), str(N_iter), str(N), save_path])
        active_processes.append(process)
        time.sleep(60)

# Ensure all jobs finish
for process in active_processes:
    process.wait()