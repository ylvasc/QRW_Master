import pickle as pkl
import numpy as np
import matplotlib.pyplot as plt
import sys
import os
import uuid
import matplotlib.tri as tri
import scipy.interpolate


input_dir = "data_t"  
plots_dir = "plots_t"
os.makedirs(plots_dir, exist_ok=True)

with open(os.path.join(input_dir, "t_analysis_results.pkl"), "rb") as f:
    analysis_results = pkl.load(f)


#generate plots for each t
for t, data in analysis_results.items():
    alpha_values = data["alpha_values"]
    beta_values = data["beta_values"]
    estimates = data["estimates"]
    variances = data["variances"]
    percentiles = data["percentiles"]
    percentiles_var = data["percentiles_var"]
    best_probabilities = data["best_probabilities"]
    best_probabilities_var = data["best_probabilities_var"]
    best_probabilities_sum = data["best_probabilities_sum"]
    opt_iter = data["opt_iter"]
    opt_iter_var = data["opt_iter_var"]
    N = data["N"]
    n = data["n"]
    N_iter = data["N_iter"]

    grid_alpha, grid_beta = np.meshgrid(
    np.linspace(min(alpha_values), max(alpha_values), len(alpha_values)),
    np.linspace(min(beta_values), max(beta_values), len(beta_values))
)
    def save_plot(fig, plot_type):
        unique_id = str(uuid.uuid4())[:8]
        filename = os.path.join(plots_dir, f"plot_{unique_id}_{plot_type}_n={n}_N={N}.png")
        fig.savefig(filename, dpi=300, bbox_inches="tight")
        plt.close(fig)

    def generate_plots(values, title, cmap, label, plot_type):
        fig = plt.figure(figsize=(7, 6))
        contourf = plt.tricontourf(alpha_values, beta_values, values, levels=50, cmap=cmap)
        contour = plt.tricontour(alpha_values, beta_values, values, levels=10, colors='black', linewidths=0.5)
        plt.clabel(contour, inline=True, fontsize=8)
        plt.colorbar(contourf, label=label)
        plt.xlabel(r"$\alpha$")
        plt.ylabel(r"$\beta$")
        plt.title(title, pad=10)
        save_plot(fig, plot_type)

        fig = plt.figure(figsize=(7, 6))
        grid_values = scipy.interpolate.griddata((alpha_values, beta_values), values, (grid_alpha, grid_beta), method='nearest')
        plt.imshow(grid_values, extent=(min(alpha_values), max(alpha_values), min(beta_values), max(beta_values)),
                origin='lower', cmap=cmap, aspect='auto')
        plt.colorbar(label=label)
        plt.xlabel(r"$\alpha$")
        plt.ylabel(r"$\beta$")
        plt.title(title, pad=10)
        save_plot(fig, plot_type + "_imshow")

    #approx ratio
    generate_plots(estimates, 
                   f"Approximation Ratio (t={t}, {n} states, {N} Grover reflections)", 
                   "coolwarm", "Mean Approx ratio", 
                   f"performance_t_{t}")

    #approx ratio var
    generate_plots(variances, 
                   f"Approximation Ratio Variance (t={t}, {n} states, {N} Grover reflections)", 
                   "viridis", "Variance", 
                   f"variance_t_{t}")

    #percentiles
    generate_plots(percentiles, 
                   f"Mean Percentile (t={t}, {n} states, {N} Grover reflections)", 
                   "coolwarm", "Mean Percentile", 
                   f"percentile_t_{t}")

    #percentiles var
    generate_plots(percentiles_var, 
                   f"Percentile Variance (t={t}, {n} states, {N} Grover reflections)", 
                   "viridis", "Variance of Percentiles", 
                   f"percentile_variance_t_{t}")

    #best prob
    generate_plots(best_probabilities, 
                   f"Top Probabilities (t={t}, {n} states, {N} Grover reflections)", 
                   "hot", "Best Probabilities", 
                   f"best_probabilities_t_{t}")

    #best prob var
    generate_plots(best_probabilities_var, 
                   f"Best Probabilities Variance (t={t}, {n} states, {N} Grover reflections)", 
                   "viridis", "Variance of Best Probabilities", 
                   f"best_probabilities_var_t_{t}")

    #best Probabilities Sum (Top 10)
    generate_plots(best_probabilities_sum, 
                   f"Best Probabilities Sum (Top 10) (t={t}, {n} states, {N} Grover reflections)", 
                   "plasma", "Sum of Best Probabilities (Top 10)", 
                   f"best_probabilities_sum_t_{t}")

    # optimal iteration number
    generate_plots(opt_iter, 
                   f"Optimal Iterations (t={t}, {n} states, {N} Grover reflections)", 
                   "inferno", "Optimal Iterations", 
                   f"opt_iter_t_{t}")

    #opt iter var
    generate_plots(opt_iter_var, 
                   f"Variance in Optimal Iterations (t={t}, {n} states, {N} Grover reflections)", 
                   "viridis", "Variance of Optimal Iterations", 
                   f"opt_iter_var_t_{t}")


