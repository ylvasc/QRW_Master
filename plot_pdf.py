import os
import pickle as pkl
import matplotlib.pyplot as plt
import matplotlib.tri as tri
import scipy.interpolate
import uuid
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages

# Load the data
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

# Set up LaTeX-friendly font
plt.rcParams.update({
    "text.usetex": False,  # Disable LaTeX rendering
    "font.family": "serif",  # Use serif font (DejaVu Serif as default)
    "axes.labelsize": 12,
    "axes.titlesize": 14,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "figure.figsize": (7, 6),
    "axes.unicode_minus": False
})

def save_plot(fig, plot_type):
    # Save the plot as a page in a separate PDF file
    pdf_filename = os.path.join(plots_dir, f"{plot_type}.pdf")
    with PdfPages(pdf_filename) as pdf:
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
    plt.title(title, fontsize=14, pad=20, loc='center', wrap=True)

    save_plot(fig, plot_type)

# Generate and save plots (each plot will be saved in a separate PDF)
generate_plots(all_estimates, f"Mean approximation ratio (for {n} states, {N} iterations)", "coolwarm", f"Mean Performance over {N_iter} runs", "approx_ratio")
generate_plots(all_variances, f"Mean approximation ratio - Variance (for {n} states, {N} iterations)", "viridis", f"Variance over {N_iter} runs", "variance")
generate_plots(all_percentiles, f"Mean percentiles (for {n} states, {N} iterations)", "coolwarm", f"Mean Percentile over {N_iter} runs", "percentiles")
generate_plots(percentiles_var, f"Mean percentiles - Variance (for {n} states, {N} iterations)", "viridis", f"Variance of Percentile over {N_iter} runs", "percentile_variance")
generate_plots(best_probabilities, f"Best probabilities for most amplified state (for {n} states, {N} iterations)", "hot", f"Best Probabilities over {N_iter} runs", "best_probabilities")
generate_plots(best_probabilities/0.001, 
               f"Probability amplification (for {n} states, {N} iterations)", 
               "magma", 
               f"Best Probabilities over {N_iter} runs", 
               "probabilities_ampl")
generate_plots(opt_iter, f"Optimal period (for {n} states, {N} iterations)", "inferno", f"Optimal period over {N_iter} runs", "optimal_period")
generate_plots(opt_iter_var, f"Optimal period - Variance (for {n} states, {N} iterations)", "viridis", f"Variance Optimal period over {N_iter} runs", "optimal_period_variance")
generate_plots(best_probabilities_var, f"Best probabilities for most amplified state - Variance (for {n} states, {N} iterations)", "viridis", f"Variance best probabilities", "best_prob_var")
generate_plots(best_probabilities_var/0.000001, f"Probability amplification - Variance (for {n} states, {N} iterations)", "viridis", f"Variance best probabilities", "prob_ampl_var")