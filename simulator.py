import numpy as np
import pickle as pkl
import os
import argparse
import jax
import jax.numpy as jnp
import sys


# LOSS FUNCTION
def generate_loss_func(alpha, beta, n, scaling=1):
    loss_func = np.random.beta(alpha, beta, size=n)
    return loss_func * scaling

# GROVER
@jax.jit
def grover_step(U, psi):
    psi_new = jnp.dot(U, psi)
    return psi_new, jnp.abs(psi_new) ** 2  

def grover_iter(n, N, loss_func):
    loss_func_sort_idx = jnp.argsort(loss_func)
    loss_func_sort = loss_func[loss_func_sort_idx]

    oracle = jnp.diag(jnp.exp(1j * jnp.pi * loss_func_sort))
    psi_uniform = jnp.ones(n, dtype=jnp.complex64) / jnp.sqrt(n)
    D = 2 * jnp.outer(psi_uniform, psi_uniform) - jnp.eye(n, dtype=jnp.complex64)
    U = D @ oracle  

    psi_init = jax.device_put(jnp.array(psi_uniform, dtype=jnp.complex64))
    psi_final, prob_distr = jax.lax.scan(lambda psi, _: grover_step(U, psi), psi_init, None, length=N)

    return jnp.array(prob_distr).T  


if __name__ == "__main__":
 
    alpha = float(sys.argv[1])  
    beta = float(sys.argv[2]) 
    n= int(sys.argv[3])        
    N_iter = int(sys.argv[4])   
    N = int(sys.argv[5])        
    save_path = sys.argv[6]     

    

    loss_funcs = np.zeros((N_iter, n))  
    all_results = np.zeros((N_iter, n, N))  

    for i in range(N_iter):
        loss_func = generate_loss_func(alpha, beta, n)
        loss_funcs[i] = loss_func  
        loss_func_jax = jax.device_put(loss_func)
        all_results[i] = grover_iter(n, N, loss_func_jax)  

    save_data = {
        "alpha": alpha,
        "beta": beta,
        "loss_funcs": loss_funcs,
        "results": all_results
    }
    
    with open(save_path, 'wb') as f:
        pkl.dump(save_data, f)