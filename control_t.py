import os
import time
import numpy as np
from tqdm import tqdm
import uuid

results_base_dir = "results3"
os.makedirs(results_base_dir, exist_ok=True)

alpha_values = np.array([0.1, 1.0, 2.0])
beta_values = np.array([0.1, 1.0, 2.0])
t_values = np.array([0.1, 1.0])

n_sim = 100  # Number of states
N_iter = 10  # Number of iterations per (alpha, beta) pair
N = 200       # Number of Grover iterations
max_jobs = 50  # Limit on concurrent jobs
active_pids = []



for t in t_values:
    for a in alpha_values:
        for b in beta_values:
            unique_id = str(uuid.uuid4())[:8]
            file_name = f"results_{unique_id}.pkl"  
            save_path = os.path.join(results_base_dir, file_name)  

            # Delete this after test
            print(f"Running with t={t}, alpha={a}, beta={b}, n={n_sim}, N_iter={N_iter}, N={N}, save_path={save_path}")

            pid = os.spawnlp(os.P_NOWAIT, "python3", "python3", "simulator.py", 
                            str(a), str(b), str(t), str(n_sim), str(N_iter), str(N), save_path)
            active_pids.append(pid)
        

# Ensure all jobs finish
for pid in active_pids:
    os.waitpid(pid, 0)