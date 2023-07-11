import jax
import jax.numpy as jnp
import numpy as np

def bootstrap(data,estimator,n_resamples=1000,seed=0):
    """
    Input: 
    data        (array-like)
    estimator   (function)
    n_samples   (int)
    seed        (int)
    """

    est_arr = np.zeros(n_resamples)
 
    for i in range(n_resamples):
        np.random.seed(seed)
        resample = np.random.choice(data,(len(data),),replace=True,axis=0)
        
        est_arr[i] = estimator(resample)

    return est_arr

def jaxstrap(data,estimator,n_resamples=1000,seed=0):
    """
    Input: 
    data        (array-like)
    estimator   (function)
    n_samples   (int)
    seed        (int)
    """

    # TO DO: change this to one giant array so we can do it in JAX -Ell

    key = jax.random.PRNGKey(seed)

    est_arr = np.zeros(n_resamples)
 
    key, subkey = jax.random.split(key)
    resamples = jax.random.choice(subkey,data,(n_resamples,*data.shape),replace=True)
        
    est_arr[i] = estimator(resamples)

    return est_arr