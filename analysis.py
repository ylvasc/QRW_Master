import os
import pickle as pkl
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt

results_base_dir = "results"


alpha_values = []
beta_values = []
all_estimates = []

for file_name in os.listdir(results_base_dir):
    if file_name.endswith(".pkl"):
        file_path = os.path.join(results_base_dir, file_name)

        with open(file_path, "rb") as f:
            data = pkl.load(f)

        alpha = data["alpha"]
        beta = data["beta"]
        loss_funcs = data["loss_funcs"]  # Shape: (N_iter, n)
        results = data["results"]  # Shape: (N_iter, n, N)

        N_iter=results.shape[0]
        n=results.shape[1]
        N=results.shape[2]

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



            ##############plot loss func with best value
            #plt.plot(loss_sorted)
            #plt.scatter(best_indices[0], loss_sorted[best_indices[0]], color='red', s=100, marker='x', zorder=5)
            #plt.title(f"Loss function with best value (Alpha={alpha}, Beta={beta}), estimate{estimate}")
            #plt.grid()
            #plt.show()
            
            cost_exp[i]=np.dot(prob_distr.T, loss_sorted)
    

        alpha_values.append(alpha)
        beta_values.append(beta)
        all_estimates.append(np.mean(estimates)) 


        #############plot cost expectation value
        #plt.plot(np.mean(cost_exp, axis=0))
        #x=np.mean(cost_exp, axis=0)
        #plt.title(f"Cost Expectation Value, estimate{x}")
        #plt.show()


alpha_values = np.array(alpha_values)
beta_values = np.array(beta_values)
all_estimates = np.array(all_estimates)



# Create a contour plot
plt.figure(figsize=(8, 6))
plt.tricontourf(alpha_values, beta_values, all_estimates, levels=20, cmap="viridis")
plt.colorbar(label=f"Mean Performance over {N_iter} runs")
plt.xlabel(r"$\alpha$")
plt.ylabel(r"$\beta$")
plt.title(f"Performance (for {n} states, {N} Grover reflections)")
plt.show()
