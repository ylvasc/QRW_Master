import os
import pickle as pkl
import jax.numpy as jnp
import numpy as np
import scipy.signal

results_base_dir = "results2"
data_dir = "data"
os.makedirs(data_dir, exist_ok=True)

alpha_values, beta_values = [], []
all_estimates, all_variances = [], []
all_percentiles, percentiles_var = [], []
best_probabilities, best_probabilities_var = [], []
best_probabilities_sum = []
opt_iter, opt_iter_var = [], []


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

    if np.isnan(results).any() or np.isnan(loss_funcs).any():
        print(f"Warning: Skipping file {file_name} (contains NaN values)")
        continue

    N_iter, n, N = results.shape

    estimates = np.zeros(N_iter)
    percentile_ranks = np.zeros(N_iter)
    opt_period = np.zeros(N_iter)
    best_prob = np.zeros(N_iter)
    best_prob_sum = np.zeros(N_iter)

    for i in range(N_iter):
        prob_distr = results[i]
        loss_func = loss_funcs[i]

        loss_sorted = np.sort(loss_func)
        best_indices = np.argsort(prob_distr.max(axis=1))[::-1]
        best_index = best_indices[0]

        # approx ratio
        estimates[i] = (loss_sorted[best_index] - loss_sorted[0]) / (loss_sorted[-1] - loss_sorted[0])

        # percentile rank
        percentile_ranks[i] = 100 * (np.searchsorted(loss_sorted, loss_sorted[best_index]) / len(loss_sorted))

        #best probability
        best_prob[i] = prob_distr[best_index].max()

        #probability sum over top 10
        best_max_index = prob_distr[best_index].argmax()
        best_prob_sum[i] = np.sum(prob_distr[best_indices[:10], best_max_index])

        #peak distances for periodicity
        peaks_prob, _ = scipy.signal.find_peaks(prob_distr[best_index])
        peak_distances_prob = np.diff(peaks_prob)
        opt_period[i] = np.mean(peak_distances_prob) if peak_distances_prob.size > 0 else 0

    
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

data_to_save = {
    "N": N,
    "N_iter": N_iter,
    "n": n,
    "alpha_values": np.array(alpha_values),
    "beta_values": np.array(beta_values),
    "all_estimates": np.array(all_estimates),
    "all_variances": np.array(all_variances),
    "all_percentiles": np.array(all_percentiles),
    "percentiles_var": np.array(percentiles_var),
    "best_probabilities": np.array(best_probabilities),
    "best_probabilities_var": np.array(best_probabilities_var),
    "best_probabilities_sum": np.array(best_probabilities_sum),
    "opt_iter": np.array(opt_iter),
    "opt_iter_var": np.array(opt_iter_var)
}

with open(os.path.join(data_dir, "analysis_results.pkl"), "wb") as f:
    pkl.dump(data_to_save, f)
