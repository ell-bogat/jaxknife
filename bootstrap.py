import jax
import jax.numpy as jnp
import numpy as np

def bootstrap(data,statistic_func,n_resamples=1000,seed=0):
    """
    Bootstrap method using numpy and a for loop

    Input: 
    data            (array-like)
    statistic_func  (function)
    n_samples       (int)
    seed            (int)
    """

    est_arr = np.zeros(n_resamples)
    np.random.seed(seed)
 
    for i in range(n_resamples):
        idx = np.random.randint(len(data), size=len(data))
        resample = data[idx]
        
        est_arr[i] = statistic_func(resample)

    return est_arr

def jaxstrap(data,statistic_func,n_resamples=1000,seed=0):
    """
    Bootstrap method using JAX arrays
    Input: 
    data            (array-like)
    statistic_func  (function)
    n_samples       (int)
    seed            (int)
    """

    key = jax.random.PRNGKey(seed)

    resamples = jax.random.choice(key,data,(n_resamples,*data.shape),replace=True)
        
    statistic_vectorized = jax.vmap(statistic_func)
    est_arr = statistic_vectorized(resamples)

    return est_arr