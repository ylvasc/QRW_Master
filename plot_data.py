import os
import pickle as pkl
import matplotlib.pyplot as plt
import matplotlib.tri as tri
import scipy.interpolate
import uuid
import numpy as np

data_dir = "data"
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
best_probabilities_sum = data["best_probabilities_sum"]
opt_iter = data["opt_iter"]
opt_iter_var = data["opt_iter_var"]

plots_dir = "plots"
os.makedirs(plots_dir, exist_ok=True)



fig = plt.figure(figsize=(7, 6))
contourf = plt.tricontourf(alpha_values, beta_values, all_estimates, levels=50, cmap="coolwarm")
contour = plt.tricontour(alpha_values, beta_values, all_estimates, levels=10, colors='black', linewidths=0.5)
plt.clabel(contour, inline=True, fontsize=8)  
plt.colorbar(contourf, label=f"Mean Performance over {N_iter} runs")
plt.xlabel(r"$\alpha$")
plt.ylabel(r"$\beta$")
plt.title(f"Approx ratio Performance (for {n} states, {N} Grover reflections)", pad=10)
unique_id1 = str(uuid.uuid4())[:8]
plot_filename1 = os.path.join('plots', f"plot_{unique_id1}_approxratio_n={n}_N={N}.png")
plt.savefig(plot_filename1, dpi=300, bbox_inches="tight")
plt.close(fig) 


grid_alpha, grid_beta = np.meshgrid(np.linspace(min(alpha_values), max(alpha_values), len(alpha_values)),
                                    np.linspace(min(beta_values), max(beta_values), len(beta_values)))
grid_estimates = scipy.interpolate.griddata((alpha_values, beta_values), all_estimates, (grid_alpha, grid_beta), method='nearest')
fig = plt.figure(figsize=(7, 6))
im = plt.imshow(grid_estimates, extent=(min(alpha_values), max(alpha_values), min(beta_values), max(beta_values)),
                origin='lower', cmap="coolwarm", aspect='auto')
contour = plt.contour(grid_alpha, grid_beta, grid_estimates, levels=10, colors='black', linewidths=0.5)
plt.clabel(contour, inline=True, fontsize=8)
plt.colorbar(im, label=f"Mean Performance over {N_iter} runs")
plt.xlabel(r"$\alpha$")
plt.ylabel(r"$\beta$")
plt.title(f"Approx ratio Performance (for {n} states, {N} Grover reflections)", pad=10)
unique_id1 = str(uuid.uuid4())[:8]
plot_filename1 = os.path.join('plots', f"plot_{unique_id1}_TESTapproxratio_n={n}_N={N}.png")
plt.savefig(plot_filename1, dpi=300, bbox_inches="tight")
plt.close(fig)

threshold = 0.5
#Filter data for values > 0.5
mask_above = all_estimates > threshold
alpha_above = alpha_values[mask_above]
beta_above = beta_values[mask_above]
all_estimates_above = all_estimates[mask_above]

# Filter data for values < 0.5
mask_below = all_estimates < threshold
alpha_below = alpha_values[mask_below]
beta_below = beta_values[mask_below]
all_estimates_below = all_estimates[mask_below]
triang_above = tri.Triangulation(alpha_above, beta_above)
triang_below = tri.Triangulation(alpha_below, beta_below)

# Plot values above 0.5
plt.figure(figsize=(7, 6))
contourf = plt.tricontourf(triang_above, all_estimates_above, levels=50, cmap="Reds")
contour = plt.tricontour(triang_above, all_estimates_above, levels=10, colors='black', linewidths=0.5)
plt.clabel(contour, inline=True, fontsize=8)  
plt.colorbar(contourf, label=f"Approx ratio Performance over {N_iter} runs")
plt.xlabel(r"$\alpha$")
plt.ylabel(r"$\beta$")
plt.title(f"Upper Approx ratio performance (for {n} states, {N} Grover reflections)", pad=10)
unique_id_1 = str(uuid.uuid4())[:8]
plot_filename1 = os.path.join('plots', f"plot_{unique_id_1}_upper_approxratio_n={n}_N={N}.png")
plt.savefig(plot_filename1, dpi=300, bbox_inches="tight")
plt.close(fig)  

# Plot values below 0.5
plt.figure(figsize=(7, 6))
contourf = plt.tricontourf(triang_below, all_estimates_below, levels=50, cmap="Blues_r")
contour = plt.tricontour(triang_below, all_estimates_below, levels=10, colors='black', linewidths=0.5)
plt.clabel(contour, inline=True, fontsize=8)  
plt.colorbar(contourf, label=f"Approx ratio Performance over {N_iter} runs")
plt.xlabel(r"$\alpha$")
plt.ylabel(r"$\beta$")
plt.title(f"Lower Approx ratio performance (for {n} states, {N} Grover reflections)", pad=10)
unique_id_2 = str(uuid.uuid4())[:8]
plot_filename2 = os.path.join('plots', f"plot_{unique_id_2}_lower_approxratio_n={n}_N={N}.png")
plt.savefig(plot_filename2, dpi=300, bbox_inches="tight")
plt.close(fig) 


# Performance Variance Plot
fig = plt.figure(figsize=(7, 6))
contourf = plt.tricontourf(alpha_values, beta_values, all_variances, levels=50, cmap="viridis")
contour = plt.tricontour(alpha_values, beta_values, all_variances, levels=10, colors='black', linewidths=0.5)
plt.clabel(contour, inline=True, fontsize=8)  
plt.colorbar(contourf, label=f"Variance over {N_iter} runs")
plt.xlabel(r"$\alpha$")
plt.ylabel(r"$\beta$")
plt.title(f"Variance of Approx ratio (for {n} states, {N} Grover reflections)", pad=10)
unique_id2 = str(uuid.uuid4())[:8]
plot_filename2 = os.path.join('plots', f"plot_{unique_id2}_variance_approxratio_n={n}_N={N}.png")
plt.savefig(plot_filename2, dpi=300, bbox_inches="tight")
plt.close(fig)  

# Mean Percentiles Plot
fig = plt.figure(figsize=(7, 6))
contourf = plt.tricontourf(alpha_values, beta_values, all_percentiles, levels=50, cmap="coolwarm")
contour = plt.tricontour(alpha_values, beta_values, all_percentiles, levels=10, colors='black', linewidths=0.5)
plt.clabel(contour, inline=True, fontsize=8) 
plt.colorbar(contourf, label=f"Mean Percentile over {N_iter} runs")
plt.xlabel(r"$\alpha$")
plt.ylabel(r"$\beta$")
plt.title(f"Mean Percentile (for {n} states, {N} Grover reflections)", pad=10)
unique_id3 = str(uuid.uuid4())[:8]
plot_filename3 = os.path.join('plots', f"plot_{unique_id3}_percentiles_n={n}_N={N}.png")
plt.savefig(plot_filename3, dpi=300, bbox_inches="tight")
plt.close(fig)  

threshold = 50
# Filter data for values > 50
mask_above = all_percentiles > threshold
alpha_above = alpha_values[mask_above]
beta_above = beta_values[mask_above]
percentiles_above = all_percentiles[mask_above]

# Filter data for values < 50
mask_below = all_percentiles < threshold
alpha_below = alpha_values[mask_below]
beta_below = beta_values[mask_below]
percentiles_below = all_percentiles[mask_below]
triang_above2 = tri.Triangulation(alpha_above, beta_above)
triang_below2 = tri.Triangulation(alpha_below, beta_below)

# Plot values above 50
plt.figure(figsize=(7, 6))
contourf = plt.tricontourf(triang_above2, percentiles_above, levels=50, cmap="Reds")
contour = plt.tricontour(triang_above2, percentiles_above, levels=10, colors='black', linewidths=0.5)
plt.clabel(contour, inline=True, fontsize=8)  
plt.colorbar(contourf, label=f"Percentile Performance over {N_iter} runs")
plt.xlabel(r"$\alpha$")
plt.ylabel(r"$\beta$")
plt.title(f"Upper Percentiles performance (for {n} states, {N} Grover reflections)", pad=10)
unique_id_1 = str(uuid.uuid4())[:8]
plot_filename1 = os.path.join('plots', f"plot_{unique_id_1}_upper_percentiles_n={n}_N={N}.png")
plt.savefig(plot_filename1, dpi=300, bbox_inches="tight")
plt.close(fig) 

# Plot values below 50
plt.figure(figsize=(7, 6))
contourf = plt.tricontourf(triang_below2, percentiles_below, levels=50, cmap="Blues_r")
contour = plt.tricontour(triang_below2, percentiles_below, levels=10, colors='black', linewidths=0.5)
plt.clabel(contour, inline=True, fontsize=8) 
plt.colorbar(contourf, label=f"Percentile Performance over {N_iter} runs")
plt.xlabel(r"$\alpha$")
plt.ylabel(r"$\beta$")
plt.title(f"Lower percentile performance (for {n} states, {N} Grover reflections)", pad=10)
unique_id_2 = str(uuid.uuid4())[:8]
plot_filename2 = os.path.join('plots', f"plot_{unique_id_2}_lower_percentiles_n={n}_N={N}.png")
plt.savefig(plot_filename2, dpi=300, bbox_inches="tight")
plt.close(fig)  



# Variance of Percentiles Plot
fig = plt.figure(figsize=(7, 6))
contourf = plt.tricontourf(alpha_values, beta_values, percentiles_var, levels=50, cmap="viridis")
contour = plt.tricontour(alpha_values, beta_values, percentiles_var, levels=10, colors='black', linewidths=0.5)
plt.clabel(contour, inline=True, fontsize=8) 
plt.colorbar(contourf, label=f"Variance of Percentile over {N_iter} runs")
plt.xlabel(r"$\alpha$")
plt.ylabel(r"$\beta$")
plt.title(f"Variance over percentiles (for {n} states, {N} Grover reflections)", pad=10)
unique_id4 = str(uuid.uuid4())[:8]
plot_filename4 = os.path.join('plots', f"plot_{unique_id4}_percentile_variance_n={n}_N={N}.png")
plt.savefig(plot_filename4, dpi=300, bbox_inches="tight")
plt.close(fig) 


# Best Probabilities Plot
fig = plt.figure(figsize=(7, 6))
plt.tricontourf(alpha_values, beta_values, best_probabilities, levels=50, cmap="hot")
plt.colorbar(label=f"Best Probabilities over {N_iter} runs")
plt.xlabel(r"$\alpha$")
plt.ylabel(r"$\beta$")
plt.title(f"Top Probabilities for most amplified state (for {n} states, {N} Grover reflections)", pad=10)
unique_id5 = str(uuid.uuid4())[:8]
plot_filename5 = os.path.join('plots', f"plot_{unique_id5}_best_probabilities_n={n}_N={N}.png")
plt.savefig(plot_filename5, dpi=300, bbox_inches="tight")
plt.close(fig)  

# Best Probabilities Sum Plot
fig = plt.figure(figsize=(7, 6))
plt.tricontourf(alpha_values, beta_values, best_probabilities_sum, levels=50, cmap="hot")
plt.colorbar(label=f"Summed Probabilities over {N_iter} runs")
plt.xlabel(r"$\alpha$")
plt.ylabel(r"$\beta$")
plt.title(f"Top Probabilities for most amplified state (for {n} states, {N} Grover reflections)", pad=10)
unique_id5 = str(uuid.uuid4())[:8]
plot_filename5 = os.path.join('plots', f"plot_{unique_id5}_sum_best_probabilities_n={n}_N={N}.png")
plt.savefig(plot_filename5, dpi=300, bbox_inches="tight")
plt.close(fig)  

#optimal period plot
fig = plt.figure(figsize=(7, 6))
plt.tricontourf(alpha_values, beta_values, opt_iter, levels=50, cmap="inferno")
plt.colorbar(label=f"Optimal period over {N_iter} runs")
plt.xlabel(r"$\alpha$")
plt.ylabel(r"$\beta$")
plt.title(f"Optimal period (for {n} states, {N} Grover reflections)", pad=10)
unique_id6 = str(uuid.uuid4())[:8]
plot_filename6 = os.path.join('plots', f"plot_{unique_id6}_testopt_period_n={n}_N={N}.png")
plt.savefig(plot_filename6, dpi=300, bbox_inches="tight")
plt.close(fig)  


# optimal period variance
fig = plt.figure(figsize=(7, 6))
plt.tricontourf(alpha_values, beta_values, opt_iter_var, levels=50, cmap="inferno")
plt.colorbar(label=f"Variance Optimal period over {N_iter} runs")
plt.xlabel(r"$\alpha$")
plt.ylabel(r"$\beta$")
plt.title(f"Variance over Optimal period (for {n} states, {N} Grover reflections)", pad=10)
unique_id7 = str(uuid.uuid4())[:8]
plot_filename7 = os.path.join('plots', f"plot_{unique_id7}_testoptvar_period_n={n}_N={N}.png")
plt.savefig(plot_filename7, dpi=300, bbox_inches="tight")
plt.close(fig) 

