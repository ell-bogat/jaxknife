import jax
import numpy as np

def bootstrap(data,estimator,n_resamples=1000,seed=0):
    """
    Input: 
    data        (array-like)
    estimator   (function)
    n_samples   (int)
    seed        (int)
    """

    key = jax.random.PRNGKey(seed)

    est_arr = np.zeros(n_resamples)
 
    for i in range(n_resamples):
        key, subkey = jax.random.split(key)
        resample = jax.random.choice(subkey,data,(len(data),),replace=True,axis=0)
        
        est_arr[i] = estimator(resample)

    return est_arr