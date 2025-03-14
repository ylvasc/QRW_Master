import os
import pickle as pkl
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
import uuid
import seaborn as sns

results_base_dir = "results3" #change later
plots_dir = "plots"

os.makedirs(plots_dir, exist_ok=True)

alpha_values = []
beta_values = []
t_values = np.array([0.1, 1.0])  
all_estimates = []
all_variances = []

fig, axs = plt.subplots(len(t_values), 2, figsize=(14, 6 * len(t_values)))  # Adjust number of rows based on len(t_values)


for t_idx, t in enumerate(t_values):
    alpha_values_t = []
    beta_values_t = []
    estimates_t = []
    variances_t = []

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
        required_keys = {"alpha", "beta", "t", "loss_funcs", "results"}
        if not required_keys.issubset(data.keys()):
            print(f"Warning: Skipping file {file_name} (missing required keys)")
            continue

        alpha, beta = data["alpha"], data["beta"]
        loss_funcs, results = data["loss_funcs"], data["results"]
        t_in_data = data["t"]

        # Proceed only if this file corresponds to the current t value
        if not np.isclose(t_in_data, t):
            continue
        #if t_in_data == t:
            #print(f"Processing file {file_name} for t={t}")
            #print(f"Unique (alpha, beta) pairs for t={t}: {set(zip(alpha_values_t, beta_values_t))}")

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

        alpha_values_t.append(alpha)
        beta_values_t.append(beta)
        estimates_t.append(np.mean(estimates))
        variances_t.append(np.var(estimates))

    alpha_values_t = np.array(alpha_values_t)
    beta_values_t = np.array(beta_values_t)
    estimates_t = np.array(estimates_t)
    variances_t = np.array(variances_t)

    print(f"t={t}: Estimates shape: {estimates_t.shape}, Variances shape: {variances_t.shape}")
    print("alpha: ", alpha_values_t)
    print("beta: ", beta_values_t)
    print("estimates: ", estimates_t)
    print(f"NaNs in estimates_t for t={t}: {np.isnan(estimates_t).any()}")
    print(f"NaNs in variances_t for t={t}: {np.isnan(variances_t).any()}")
    print("Min alpha:", min(alpha_values_t), "Max alpha:", max(alpha_values_t))
    print("Min beta:", min(beta_values_t), "Max beta:", max(beta_values_t))
    print("Min estimate:", min(estimates_t), "Max estimate:", max(estimates_t))

    if alpha_values_t.size == 0:
        print("No valid data to plot.")  # Check if no valid data
    else:
        contour1 = axs[t_idx, 0].tricontourf(alpha_values_t, beta_values_t, estimates_t, levels=50, cmap="viridis")
        fig.colorbar(contour1, ax=axs[t_idx, 0], label=f"Mean Performance over {N_iter} runs")
        axs[t_idx, 0].set_xlabel(r"$\alpha$")
        axs[t_idx, 0].set_ylabel(r"$\beta$")
        axs[t_idx, 0].set_title(f"Performance (for t={t}, {n} states, {N} Grover reflections)")

        contour2 = axs[t_idx, 1].tricontourf(alpha_values_t, beta_values_t, variances_t, levels=20, cmap="bone")
        fig.colorbar(contour2, ax=axs[t_idx, 1], label=f"Variance over {N_iter} runs")
        axs[t_idx, 1].set_xlabel(r"$\alpha$")
        axs[t_idx, 1].set_ylabel(r"$\beta$")
        axs[t_idx, 1].set_title(f"Variance (for t={t}, {n} states, {N} Grover reflections)")

plt.tight_layout()


unique_id = str(uuid.uuid4())[:8]
plot_filename = os.path.join(plots_dir, f"t_plot_{unique_id}_n={n}_N={N}.png")
plt.savefig(plot_filename, dpi=300, bbox_inches="tight")

plt.close(fig)
plt.show()

