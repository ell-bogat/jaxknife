import jax
import jax.numpy as jnp
import numpy as np

def bootstrap(data,estimator_func,n_resamples=1000,seed=0):
    """
    Bootstrap method using numpy and a for loop

    Input: 
    data        (array-like)
    estimator_func   (function)
    n_samples   (int)
    seed        (int)
    """

    est_arr = np.zeros(n_resamples)
    np.random.seed(seed)
 
    for i in range(n_resamples):
        idx = np.random.randint(len(data), size=len(data))
        resample = data[idx]
        
        est_arr[i] = estimator_func(resample)

    return est_arr

def jaxstrap(data,estimator_func,n_resamples=1000,seed=0):
    """
    Bootstrap method using JAX arrays
    Input: 
    data        (array-like)
    estimator   (function)
    n_samples   (int)
    seed        (int)
    """

    # TO DO: change this to one giant array so we can do it in JAX -Ell

    key = jax.random.PRNGKey(seed)

    resamples = jax.random.choice(key,data,(n_resamples,*data.shape),replace=True)
        
    estimator_vectorized = jax.vmap(estimator_func)
    est_arr = estimator_vectorized(resamples)

    return est_arr