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


fig, axs = plt.subplots(5, len(t_values), figsize=(24, 28))  # 5 rows for each subplot


for t_idx, t in enumerate(t_values):
    alpha_values_t = []
    beta_values_t = []
    estimates_t = []
    variances_t = []
    percentiles_t = []
    percentiles_var_t = []
    best_probabilities_t = []

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
        percentile_ranks = []

        for i in range(N_iter):
            prob_distr = results[i]

            


            loss_func = loss_funcs[i]

            best_indices = jnp.argsort(prob_distr.max(axis=1))[::-1]
            best_index = best_indices[0]

            loss_sorted = jnp.sort(loss_func)
            estimate = (loss_sorted[best_index] - loss_sorted[0]) / (loss_sorted[-1] - loss_sorted[0])
            estimates.append(float(estimate))

            cost_exp[i] = np.dot(prob_distr.T, loss_sorted)
            percentile = 100 * (jnp.searchsorted(loss_sorted, loss_sorted[best_index]) / len(loss_sorted))
            percentile_ranks.append(float(percentile))

            best_prob = prob_distr[best_index]

        alpha_values_t.append(alpha)
        beta_values_t.append(beta)
        estimates_t.append(np.mean(estimates))
        variances_t.append(np.var(estimates))
        percentiles_t.append(np.mean(percentile_ranks))
        percentiles_var_t.append(np.var(percentile_ranks))
        best_probabilities_t.append(float(best_prob.max()))

    alpha_values_t = np.array(alpha_values_t)
    beta_values_t = np.array(beta_values_t)
    estimates_t = np.array(estimates_t)
    variances_t = np.array(variances_t)
    percentiles_t = np.array(percentiles_t)
    percentiles_var_t = np.array(percentiles_var_t)
    best_probabilities_t = np.array(best_probabilities_t)

    print(f"t={t}: percentiles:",percentiles_t)
    print(f"t={t}: est:",estimates_t)
    if alpha_values_t.size == 0:
        print("No valid data to plot.")  # Check if no valid data
    else:
        contour1 = axs[0, t_idx].tricontourf(alpha_values_t, beta_values_t, estimates_t, levels=50, cmap="viridis")
        fig.colorbar(contour1, ax=axs[0, t_idx], label=f"Mean Performance over {N_iter} runs")
        axs[0, t_idx].set_xlabel(r"$\alpha$")
        axs[0, t_idx].set_ylabel(r"$\beta$")
        axs[0, t_idx].set_title(f"Performance (for t={t}, {n} states, {N} Grover reflections)", pad=20)
        axs[0, t_idx].set_aspect('equal')

        # Plot the second subplot for "Performance Variance"
        contour2 = axs[1, t_idx].tricontourf(alpha_values_t, beta_values_t, variances_t, levels=20, cmap="bone")
        fig.colorbar(contour2, ax=axs[1, t_idx], label=f"Variance over {N_iter} runs")
        axs[1, t_idx].set_xlabel(r"$\alpha$")
        axs[1, t_idx].set_ylabel(r"$\beta$")
        axs[1, t_idx].set_title(f"Performance Variance (for t={t}, {n} states, {N} Grover reflections)", pad=20)
        axs[1, t_idx].set_aspect('equal')

        # Plot the third subplot for "Mean Percentiles"
        contour3 = axs[2, t_idx].tricontourf(alpha_values_t, beta_values_t, percentiles_t, levels=50, cmap="viridis")
        fig.colorbar(contour3, ax=axs[2, t_idx], label=f"Mean Percentile over {N_iter} runs")
        axs[2, t_idx].set_xlabel(r"$\alpha$")
        axs[2, t_idx].set_ylabel(r"$\beta$")
        axs[2, t_idx].set_title(f"Mean Percentile (for t={t}, {n} states, {N} Grover reflections)", pad=20)
        axs[2, t_idx].set_aspect('equal')

        # Plot the fourth subplot for "Variance of Percentiles"
        contour4 = axs[3, t_idx].tricontourf(alpha_values_t, beta_values_t, percentiles_var_t, levels=20, cmap="bone")
        fig.colorbar(contour4, ax=axs[3, t_idx], label=f"Variance of Percentile over {N_iter} runs")
        axs[3, t_idx].set_xlabel(r"$\alpha$")
        axs[3, t_idx].set_ylabel(r"$\beta$")
        axs[3, t_idx].set_title(f"Percentile Variance (for t={t}, {n} states, {N} Grover reflections)", pad=20)
        axs[3, t_idx].set_aspect('equal')

        # Plot the fifth subplot for "Best Probabilities"
        contour5 = axs[4, t_idx].tricontourf(alpha_values_t, beta_values_t, best_probabilities_t, levels=20, cmap="hot")
        fig.colorbar(contour5, ax=axs[4, t_idx], label=f"Best Probabilities over {N_iter} runs")
        axs[4, t_idx].set_xlabel(r"$\alpha$")
        axs[4, t_idx].set_ylabel(r"$\beta$")
        axs[4, t_idx].set_title(f"Top Probabilities for most amplified state (for {n} states, {N} Grover reflections)", pad=20)
        axs[4, t_idx].set_aspect('equal')

for t_idx, t in enumerate(t_values):
    fig.text(0.35 + t_idx / len(t_values), 0.94, f"t={t}", ha="center", va="bottom", fontsize=14, fontweight='bold')

# Add a main title for the entire figure
fig.suptitle(f"Analysis of t-values for {n} states and {N} Grover reflections", fontsize=16, fontweight='bold')

# Adjust layout for spacing
fig.tight_layout(rect=[0, 0, 1, 0.93])  # Reduced the rect space to prevent overlap


# Save the plot   





unique_id = str(uuid.uuid4())[:8]
plot_filename = os.path.join(plots_dir, f"t_plot_{unique_id}_n={n}_N={N}.png")
plt.savefig(plot_filename, dpi=300, bbox_inches="tight")

plt.close(fig)
plt.show()

