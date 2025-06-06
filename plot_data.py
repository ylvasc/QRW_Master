import os
import pickle as pkl
import matplotlib.pyplot as plt
import matplotlib.tri as tri
import scipy.interpolate
import uuid
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages

data_dir = "data_new3"
with open(os.path.join(data_dir, "analysis_results.pkl"), "rb") as f:
    data = pkl.load(f)

N = data["N"]
N_iter = data["N_iter"]
n = data["n"]
alpha_values = data["alpha_values"]
beta_values = data["beta_values"]
all_estimates = data["all_estimates"]
all_variances = data["all_variances"]
all_percentiles = data["all_percentiles"]
percentiles_var = data["percentiles_var"]
best_probabilities = data["best_probabilities"]
best_probabilities_var = data["best_probabilities_var"]
best_probabilities_sum = data["best_probabilities_sum"]
opt_iter = data["opt_iter"]
opt_iter_var = data["opt_iter_var"]

plots_dir = "plots_new3_"
os.makedirs(plots_dir, exist_ok=True)

grid_alpha, grid_beta = np.meshgrid(
    np.linspace(min(alpha_values), max(alpha_values), len(alpha_values)),
    np.linspace(min(beta_values), max(beta_values), len(beta_values))
)
pdf_filename = os.path.join(plots_dir, f"plots_{n}_states_{N}_reflections.pdf")
with PdfPages(pdf_filename) as pdf:

    def save_plot(fig, plot_type):
        # Save the plot as a page in the PDF
        pdf.savefig(fig)
        plt.close(fig)

    def generate_plots(values, title, cmap, label, plot_type):
        fig = plt.figure(figsize=(7, 6))
        grid_values = scipy.interpolate.griddata((alpha_values, beta_values), values, (grid_alpha, grid_beta), method='nearest')
        plt.imshow(grid_values, extent=(min(alpha_values), max(alpha_values), min(beta_values), max(beta_values)),
                   origin='lower', cmap=cmap, aspect='auto')
        plt.colorbar(label=label)
        plt.xlabel(r"$\alpha$")
        plt.ylabel(r"$\beta$")
        plt.title(title, pad=10)
        save_plot(fig, plot_type)

    # Generate and save plots
    generate_plots(all_estimates, f"Approx ratio Performance (for {n} states, {N} Grover reflections)", "coolwarm", f"Mean Performance over {N_iter} runs", "approx_ratio")
    generate_plots(all_variances, f"Variance of Approx ratio (for {n} states, {N} Grover reflections)", "viridis", f"Variance over {N_iter} runs", "variance")
    generate_plots(all_percentiles, f"Mean Percentile (for {n} states, {N} Grover reflections)", "coolwarm", f"Mean Percentile over {N_iter} runs", "percentiles")
    generate_plots(percentiles_var, f"Variance over percentiles (for {n} states, {N} Grover reflections)", "viridis", f"Variance of Percentile over {N_iter} runs", "percentile_variance")
    generate_plots(best_probabilities, f"Top Probabilities for most amplified state (for {n} states, {N} Grover reflections)", "hot", f"Best Probabilities over {N_iter} runs", "best_probabilities")
    generate_plots(best_probabilities/0.001, f"Probability Amplification P_max/P_init (for {n} states, {N} Grover reflections)", "magma", f"Best Probabilities over {N_iter} runs", "probabilities_ampl")
    generate_plots(opt_iter, f"Optimal period (for {n} states, {N} Grover reflections)", "inferno", f"Optimal period over {N_iter} runs", "optimal_period")
    generate_plots(opt_iter_var, f"Variance over Optimal period (for {n} states, {N} Grover reflections)", "viridis", f"Variance Optimal period over {N_iter} runs", "optimal_period_variance")
    generate_plots(best_probabilities_var, f"Variance top probabilities for most amplified state (for {n} states, {N} Grover reflections)", "viridis", f"Variance best probabilities", "best_prob_var")