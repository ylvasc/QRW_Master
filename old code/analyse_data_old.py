import os
import pickle as pkl
import jax.numpy as jnp
import numpy as np
import scipy.signal
import scipy.interpolate
import matplotlib.pyplot as plt
import seaborn as sns

results_base_dir = "results2"
data_dir = "data"
os.makedirs(data_dir, exist_ok=True)

alpha_values = []
beta_values = []
all_estimates = []
all_variances = []
all_percentiles = []
percentiles_var = []
best_probabilities = []
best_probabilities_var = []
best_probabilities_sum = []
opt_iter = []
opt_iter_var = []


sampling_rate = 1000  # for fft

# Loop over the result files and process them
for file_name in os.listdir(results_base_dir):
    if not file_name.endswith(".pkl"):
        continue  # Skip non-PKL files

    file_path = os.path.join(results_base_dir, file_name)

    try:
        with open(file_path, "rb") as f:
            data = pkl.load(f)
    except (pkl.UnpicklingError, EOFError) as e:
        print(f"Warning: Skipping corrupted file {file_name} ({e})")
        continue

    required_keys = {"alpha", "beta", "loss_funcs", "results"}
    if not required_keys.issubset(data.keys()):
        print(f"Warning: Skipping file {file_name} (missing required keys)")
        continue

    alpha, beta = data["alpha"], data["beta"]
    loss_funcs, results = data["loss_funcs"], data["results"]
    N_iter, n, N = results.shape

    if np.isnan(results).any() or np.isnan(loss_funcs).any():
        print(f"Warning: Skipping file {file_name} (contains NaN values)")
        continue

    estimates = []
    percentile_ranks = []
    opt_period = []
    best_prob = []
    best_prob_sum = []
    
    cost_exp = np.zeros((N_iter, N))

    for i in range(N_iter):
        prob_distr = results[i]
        loss_func = loss_funcs[i]

        best_indices = jnp.argsort(prob_distr.max(axis=1))[::-1]
        best_index = best_indices[0]
        worst_index = best_indices[-1]

        loss_sorted = jnp.sort(loss_func)
        estimate = (loss_sorted[best_index] - loss_sorted[0]) / (loss_sorted[-1] - loss_sorted[0])
        estimates.append(float(estimate))

        cost_exp[i] = np.dot(prob_distr.T, loss_sorted)

        percentile = 100 * (jnp.searchsorted(loss_sorted, loss_sorted[best_index]) / len(loss_sorted))
        percentile_ranks.append(float(percentile))

        best_prob.append(float(prob_distr[best_index].max()))

        prob_sum=0
        best_max_index = prob_distr[best_index].argmax()

        for i in range(10):
            prob_sum += prob_distr[best_indices[i]][best_max_index]
        best_prob_sum.append(float(prob_sum))    
        peaks_prob, _ = scipy.signal.find_peaks(prob_distr[best_index])  
        peak_distances_prob = np.diff(peaks_prob)  
        avg_dist_prob = np.mean(peak_distances_prob[1:]) if len(peak_distances_prob) > 0 else 0
        opt_period.append(float(avg_dist_prob))

    alpha_values.append(alpha)
    beta_values.append(beta)
    all_estimates.append(np.mean(estimates))
    all_variances.append(np.var(estimates))
    all_percentiles.append(np.mean(percentile_ranks))
    percentiles_var.append(np.var(percentile_ranks))
    best_probabilities.append(np.mean(best_prob))
    best_probabilities_var.append(np.var(best_prob))
    best_probabilities_sum.append(np.mean(best_prob_sum))
    opt_iter.append(np.mean(opt_period))
    opt_iter_var.append(np.var(opt_period))

alpha_values = np.array(alpha_values)
beta_values = np.array(beta_values)
all_estimates = np.array(all_estimates)
all_variances = np.array(all_variances)
all_percentiles = np.array(all_percentiles)
percentiles_var = np.array(percentiles_var)
best_probabilities = np.array(best_probabilities)
best_probabilities_var = np.array(best_probabilities_var)
best_probabilities_sum =np.array(best_probabilities_sum)
opt_iter=np.array(opt_iter)
opt_iter_var=np.array(opt_iter_var)

data_to_save = {
    "N":N,
    "N_iter": N_iter,
    "n": n,
    "alpha_values": alpha_values,
    "beta_values": beta_values,
    "all_estimates": all_estimates,
    "all_variances": all_variances,
    "all_percentiles": all_percentiles,
    "percentiles_var": percentiles_var,
    "best_probabilities": best_probabilities,
    "best_probabilities_var": best_probabilities_var,
    "best_probabilities_sum": best_probabilities_sum,
    "opt_iter": opt_iter,
    "opt_iter_var": opt_iter_var
}

with open(os.path.join(data_dir, "analysis_results.pkl"), "wb") as f:
    pkl.dump(data_to_save, f)

