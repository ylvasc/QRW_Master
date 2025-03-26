import os
import pickle as pkl
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
import uuid
from scipy.fft import fft, fftfreq
import scipy.signal
import matplotlib.colors as mcolors
import matplotlib.tri as tri
import scipy.interpolate

results_base_dir = "results2"
plots_dir = "plots"

os.makedirs(plots_dir, exist_ok=True)

alpha_values = []
beta_values = []
all_estimates = []
all_variances = []
all_percentiles = []
percentiles_var = []
best_probabilities = []
opt_iter_best = []   #optimal number of iterations to amplify the actual optimum
opt_iter_found = []  #optimal number of rotations to amplify the state that was found by the algorithm
test2=[]
test2_var=[]
sampling_rate = 1000  # for fft

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
    N_iter, n, N = results.shape
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
    percentile_ranks = []
    opt_period =[]
    best_prob=[]
    test=[]
    cost_exp = np.zeros((N_iter, N))

    for i in range(N_iter):
        prob_distr = results[i]
        loss_func = loss_funcs[i]

        best_indices = jnp.argsort(prob_distr.max(axis=1))[::-1]
        best_index = best_indices[0]
        worst_index= best_indices[-1]

        loss_sorted = jnp.sort(loss_func)
        estimate = (loss_sorted[best_index] - loss_sorted[0]) / (loss_sorted[-1] - loss_sorted[0])
        estimates.append(float(estimate))

        cost_exp[i] = np.dot(prob_distr.T, loss_sorted)

        percentile = 100 * (jnp.searchsorted(loss_sorted, loss_sorted[best_index]) / len(loss_sorted))
        percentile_ranks.append(float(percentile))

        best_prob.append(float(prob_distr[best_index].max()))

 
        
        # fft_values = np.abs(fft(prob_distr[best_index]))
        # frequencies = fftfreq(n)
        # positive_frequencies = frequencies[:n // 2] #x-vals
        # fft_magnitude = fft_values[:n // 2]   #y-vals
        # fft_magnitude[0] = 0
        # dominant_frequency = positive_frequencies[np.argmax(fft_magnitude)]
        # opt_period.append(float(1/dominant_frequency))

        #peaks_prob, _ = scipy.signal.find_peaks(fft_magnitude)
        #print(peaks_prob)
        #print(dominant_frequency)
        #print(1/dominant_frequency)

        peaks_prob, _ = scipy.signal.find_peaks(prob_distr[best_index])  
        peak_distances_prob = np.diff(peaks_prob)  
        avg_dist_prob = np.mean(peak_distances_prob[1:]) if len(peak_distances_prob) > 0 else 0
        test.append(float(avg_dist_prob))

    alpha_values.append(alpha)
    beta_values.append(beta)
    all_estimates.append(np.mean(estimates))
    all_variances.append(np.var(estimates))
    all_percentiles.append(np.mean(percentile_ranks))
    percentiles_var.append(np.var(percentile_ranks))
    best_probabilities.append(np.mean(best_prob))
    #opt_iter_found.append(np.mean(opt_period))
    test2.append(np.mean(test))
    test2_var.append(np.var(test))
    #print(np.diff(np.array(opt_iter_found)- np.array(test2)))
    

alpha_values = np.array(alpha_values)
beta_values = np.array(beta_values)
all_estimates = np.array(all_estimates)
all_variances = np.array(all_variances)
all_percentiles = np.array(all_percentiles)
percentiles_var = np.array(percentiles_var)
best_probabilities = np.array(best_probabilities)
opt_iter_found = np.array(opt_iter_found)


if alpha_values.size == 0:
    print("No valid data to plot.") #check
else:
    #approx ratio plot
    fig = plt.figure(figsize=(7, 6))
    contourf = plt.tricontourf(alpha_values, beta_values, all_estimates, levels=50, cmap="coolwarm")
    contour = plt.tricontour(alpha_values, beta_values, all_estimates, levels=10, colors='black', linewidths=0.5)
    plt.clabel(contour, inline=True, fontsize=8)  # Add labels to contour lines
    plt.colorbar(contourf, label=f"Mean Performance over {N_iter} runs")
    plt.xlabel(r"$\alpha$")
    plt.ylabel(r"$\beta$")
    plt.title(f"Approx ratio Performance (for {n} states, {N} Grover reflections)", pad=10)
    unique_id1 = str(uuid.uuid4())[:8]
    plot_filename1 = os.path.join('plots', f"plot_{unique_id1}_approxratio_n={n}_N={N}.png")
    plt.savefig(plot_filename1, dpi=300, bbox_inches="tight")
    plt.close(fig) 

    # Create a 2D grid for alpha and beta using np.meshgrid
    grid_alpha, grid_beta = np.meshgrid(np.linspace(min(alpha_values), max(alpha_values), len(alpha_values)),
                                        np.linspace(min(beta_values), max(beta_values), len(beta_values)))

    # Interpolate the all_estimates values onto the grid
    grid_estimates = scipy.interpolate.griddata((alpha_values, beta_values), all_estimates, (grid_alpha, grid_beta), method='nearest')

    # Create the plot
    fig = plt.figure(figsize=(7, 6))

    # Plot using imshow to mimic the filled contour plot
    im = plt.imshow(grid_estimates, extent=(min(alpha_values), max(alpha_values), min(beta_values), max(beta_values)),
                    origin='lower', cmap="coolwarm", aspect='auto')

    # Plot contour lines on top of the image (mimicking tricontour)
    contour = plt.contour(grid_alpha, grid_beta, grid_estimates, levels=10, colors='black', linewidths=0.5)

    # Add labels to contour lines (same as plt.clabel for tricontour)
    plt.clabel(contour, inline=True, fontsize=8)

    # Add a color bar to the plot
    plt.colorbar(im, label=f"Mean Performance over {N_iter} runs")

    # Set labels and title
    plt.xlabel(r"$\alpha$")
    plt.ylabel(r"$\beta$")
    plt.title(f"Approx ratio Performance (for {n} states, {N} Grover reflections)", pad=10)

    # Save the plot
    unique_id1 = str(uuid.uuid4())[:8]
    plot_filename1 = os.path.join('plots', f"plot_{unique_id1}_TESTapproxratio_n={n}_N={N}.png")
    plt.savefig(plot_filename1, dpi=300, bbox_inches="tight")

    # Close the plot
    plt.close(fig)

    threshold = 0.5
    # Filter data for values > 0.5
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
    plt.clabel(contour, inline=True, fontsize=8)  # Add labels to contour lines
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
    plt.clabel(contour, inline=True, fontsize=8)  # Add labels to contour lines
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
    plt.clabel(contour, inline=True, fontsize=8)  # Add labels to contour lines
    plt.colorbar(contourf, label=f"Variance over {N_iter} runs")
    plt.xlabel(r"$\alpha$")
    plt.ylabel(r"$\beta$")
    plt.title(f"Variance of Approx ratio (for {n} states, {N} Grover reflections)", pad=10)
    unique_id2 = str(uuid.uuid4())[:8]
    plot_filename2 = os.path.join('plots', f"plot_{unique_id2}_variance_approxratio_n={n}_N={N}.png")
    plt.savefig(plot_filename2, dpi=300, bbox_inches="tight")
    plt.close(fig)  # Close the plot to save memory

    # Mean Percentiles Plot
    fig = plt.figure(figsize=(7, 6))
    contourf = plt.tricontourf(alpha_values, beta_values, all_percentiles, levels=50, cmap="coolwarm")
    contour = plt.tricontour(alpha_values, beta_values, all_percentiles, levels=10, colors='black', linewidths=0.5)
    plt.clabel(contour, inline=True, fontsize=8)  # Add labels to contour lines
    plt.colorbar(contourf, label=f"Mean Percentile over {N_iter} runs")
    plt.xlabel(r"$\alpha$")
    plt.ylabel(r"$\beta$")
    plt.title(f"Mean Percentile (for {n} states, {N} Grover reflections)", pad=10)
    unique_id3 = str(uuid.uuid4())[:8]
    plot_filename3 = os.path.join('plots', f"plot_{unique_id3}_percentiles_n={n}_N={N}.png")
    plt.savefig(plot_filename3, dpi=300, bbox_inches="tight")
    plt.close(fig)  # Close the plot to save memory

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
    plt.clabel(contour, inline=True, fontsize=8)  # Add labels to contour lines
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
    plt.clabel(contour, inline=True, fontsize=8)  # Add labels to contour lines
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
    plt.clabel(contour, inline=True, fontsize=8)  # Add labels to contour lines
    plt.colorbar(contourf, label=f"Variance of Percentile over {N_iter} runs")
    plt.xlabel(r"$\alpha$")
    plt.ylabel(r"$\beta$")
    plt.title(f"Variance over percentiles (for {n} states, {N} Grover reflections)", pad=10)
    unique_id4 = str(uuid.uuid4())[:8]
    plot_filename4 = os.path.join('plots', f"plot_{unique_id4}_percentile_variance_n={n}_N={N}.png")
    plt.savefig(plot_filename4, dpi=300, bbox_inches="tight")
    plt.close(fig)  # Close the plot to save memory


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
    plt.close(fig)  # Close the plot to save memory


    #optimal period plot
    fig = plt.figure(figsize=(7, 6))
    plt.tricontourf(alpha_values, beta_values, test2, levels=50, cmap="inferno")
    plt.colorbar(label=f"Optimal period over {N_iter} runs")
    plt.xlabel(r"$\alpha$")
    plt.ylabel(r"$\beta$")
    plt.title(f"Optimal period (for {n} states, {N} Grover reflections)", pad=10)
    unique_id6 = str(uuid.uuid4())[:8]
    plot_filename6 = os.path.join('plots', f"plot_{unique_id6}_testopt_period_n={n}_N={N}.png")
    plt.savefig(plot_filename6, dpi=300, bbox_inches="tight")
    plt.close(fig)  # Close the plot to save memory


    # optimal period variance
    fig = plt.figure(figsize=(7, 6))
    plt.tricontourf(alpha_values, beta_values, test2_var, levels=50, cmap="inferno")
    plt.colorbar(label=f"Variance Optimal period over {N_iter} runs")
    plt.xlabel(r"$\alpha$")
    plt.ylabel(r"$\beta$")
    plt.title(f"Variance over Optimal period (for {n} states, {N} Grover reflections)", pad=10)
    unique_id7 = str(uuid.uuid4())[:8]
    plot_filename7 = os.path.join('plots', f"plot_{unique_id7}_testoptvar_period_n={n}_N={N}.png")
    plt.savefig(plot_filename7, dpi=300, bbox_inches="tight")
    plt.close(fig)  # Close the plot to save memory