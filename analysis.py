import os
import pickle as pkl
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
import uuid

results_base_dir = "results"
plots_dir = "plots"

os.makedirs(plots_dir, exist_ok=True)

alpha_values = []
beta_values = []
all_estimates = []
all_variances = []

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
    
    # Ensure required keys exist
    required_keys = {"alpha", "beta", "loss_funcs", "results"}
    if not required_keys.issubset(data.keys()):
        print(f"Warning: Skipping file {file_name} (missing required keys)")
        continue

    alpha, beta = data["alpha"], data["beta"]
    loss_funcs, results = data["loss_funcs"], data["results"]

    # Check for NaNs and correct shapes
    if not isinstance(loss_funcs, np.ndarray) or not isinstance(results, np.ndarray):
        print(f"Warning: Skipping file {file_name} (data is not NumPy arrays)")
        continue
    
    if np.isnan(results).any() or np.isnan(loss_funcs).any():
        print(f"Warning: Skipping file {file_name} (contains NaN values)")
        continue
    
    if results.ndim != 3 or loss_funcs.ndim != 2:
        print(f"Warning: Skipping file {file_name} (unexpected shape)")
        continue

    N_iter, n, N = results.shape
    if loss_funcs.shape != (N_iter, n):
        print(f"Warning: Skipping file {file_name} (loss_funcs shape mismatch)")
        continue

    estimates = []
    cost_exp = np.zeros((N_iter, N))

    for i in range(N_iter):
        prob_distr = results[i]
        loss_func = loss_funcs[i]

        best_indices = jnp.argsort(prob_distr.max(axis=1))[::-1]
        best_index = best_indices[0]

        loss_sorted = jnp.sort(loss_func)
        estimate = (loss_sorted[best_index] - loss_sorted[0]) / (loss_sorted[-1] - loss_sorted[0])
        estimates.append(float(estimate))

        cost_exp[i] = np.dot(prob_distr.T, loss_sorted)

    alpha_values.append(alpha)
    beta_values.append(beta)
    all_estimates.append(np.mean(estimates))
    all_variances.append(np.var(estimates))

alpha_values = np.array(alpha_values)
beta_values = np.array(beta_values)
all_estimates = np.array(all_estimates)
all_variances = np.array(all_variances)



if alpha_values.size == 0:
    print("No valid data to plot.") #check
else:
    fig, axs = plt.subplots(1, 2, figsize=(14, 6))

    contour1 = axs[0].tricontourf(alpha_values, beta_values, all_estimates, levels=20, cmap="viridis")
    fig.colorbar(contour1, ax=axs[0], label=f"Mean Performance over {N_iter} runs")
    axs[0].set_xlabel(r"$\alpha$")
    axs[0].set_ylabel(r"$\beta$")
    axs[0].set_title(f"Performance (for {n} states, {N} Grover reflections)")

    contour2 = axs[1].tricontourf(alpha_values, beta_values, all_variances, levels=20, cmap="bone")
    fig.colorbar(contour2, ax=axs[1], label=f"Variance over {N_iter} runs")
    axs[1].set_xlabel(r"$\alpha$")
    axs[1].set_ylabel(r"$\beta$")
    axs[1].set_title(f"Performance Variance (for {n} states, {N} Grover reflections)")

    plt.tight_layout()
    unique_id = str(uuid.uuid4())[:8]
    plot_filename = os.path.join(plots_dir, f"plot_{unique_id}_n={n}_N={N}.png")  
    plt.savefig(plot_filename, dpi=300, bbox_inches="tight")

    plt.close(fig)
    plt.show()