import os
import pickle as pkl
import jax.numpy as jnp
import numpy as np
import scipy.signal

results_base_dir = "results_t_test"  # Change later
output_dir = "data_t_test"
os.makedirs(output_dir, exist_ok=True)

t1_values = np.array([0.1, 1.0])
t2_values = np.array([0.1, 1.0])
analysis_results = {}

for t1 in t1_values:
    for t2 in t2_values:
        alpha_values, beta_values = [], []
        estimates, variances = [], []
        percentiles, percentiles_var = [], []
        best_probabilities, best_probabilities_var = [], []
        best_probabilities_sum = []
        opt_iter, opt_iter_var = [], []

        N, n, N_iter = None, None, None

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

            required_keys = {"alpha", "beta", "t1", "t2","loss_funcs", "results"}
            if not required_keys.issubset(data.keys()):
                print(f"Warning: Skipping file {file_name} (missing required keys)")
                continue

            if not np.isclose(data["t1"], t1):
                continue
            if not np.isclose(data["t2"], t2):
                continue

            alpha, beta = data["alpha"], data["beta"]
            loss_funcs, results = data["loss_funcs"], data["results"]
            
            if results.ndim != 3 or loss_funcs.ndim != 2:
                print(f"Warning: Skipping file {file_name} (unexpected shape)")
                continue

            if np.isnan(results).any() or np.isnan(loss_funcs).any():
                print(f"Warning: Skipping file {file_name} (contains NaN values)")
                continue

            N_iter, n, N = results.shape
            if loss_funcs.shape != (N_iter, n):
                print(f"Warning: Skipping file {file_name} (loss_funcs shape mismatch)")
                continue

            estimates_run = []
            percentiles_run = []
            best_probs_run = []
            best_prob_sum_run = []
            opt_period_run = []

            for i in range(N_iter):
                prob_distr = results[i]
                loss_func = loss_funcs[i]
                
                best_index = jnp.argmax(prob_distr.max(axis=1))
                loss_sorted = jnp.sort(loss_func)
                estimate = (loss_sorted[best_index] - loss_sorted[0]) / (loss_sorted[-1] - loss_sorted[0])
                estimates_run.append(float(estimate))
                
                percentile = 100 * (jnp.searchsorted(loss_sorted, loss_sorted[best_index]) / len(loss_sorted))
                percentiles_run.append(float(percentile))
                best_probs_run.append(float(prob_distr[best_index].max()))

                # Probability sum over top 10
                best_indices = np.argsort(prob_distr.max(axis=1))[::-1]
                best_max_index = prob_distr[best_index].argmax()
                best_prob_sum_run.append(np.sum(prob_distr[best_indices[:10], best_max_index]))

                peaks_prob, _ = scipy.signal.find_peaks(prob_distr[best_index])
                peak_distances_prob = np.diff(peaks_prob)
                opt_period_run.append(np.mean(peak_distances_prob) if peak_distances_prob.size > 0 else 0)

            alpha_values.append(alpha)
            beta_values.append(beta)
            estimates.append(np.mean(estimates_run))
            variances.append(np.var(estimates_run))
            percentiles.append(np.mean(percentiles_run))
            percentiles_var.append(np.var(percentiles_run))
            best_probabilities.append(np.mean(best_probs_run))
            best_probabilities_var.append(np.var(best_probs_run))
            best_probabilities_sum.append(np.mean(best_prob_sum_run))
            opt_iter.append(np.mean(opt_period_run))
            opt_iter_var.append(np.var(opt_period_run))
        
        analysis_results[(t1,t2)] = {
            "alpha_values": np.array(alpha_values),
            "beta_values": np.array(beta_values),
            "estimates": np.array(estimates),
            "variances": np.array(variances),
            "percentiles": np.array(percentiles),
            "percentiles_var": np.array(percentiles_var),
            "best_probabilities": np.array(best_probabilities),
            "best_probabilities_var": np.array(best_probabilities_var),
            "best_probabilities_sum": np.array(best_probabilities_sum),
            "opt_iter": np.array(opt_iter),
            "opt_iter_var": np.array(opt_iter_var),
            "N": N,
            "n": n,
            "N_iter": N_iter,
        }

with open(os.path.join(output_dir, "t_analysis_results.pkl"), "wb") as f:
    pkl.dump(analysis_results, f)
