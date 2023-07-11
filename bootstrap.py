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


def jaxknife_func(data,estimator_func):
    """
    Jackknife method using JAX arrays
    # NOTE: systematically leaves out one sample.

    Input: 
    data        (array-like)
    estimator   (function)
    """

    resamples = jackknife_resamples(data)

    estimator_vectorized = jax.vmap(estimator_func)
    est_arr = estimator_vectorized(resamples)

    return est_arr

def jackknife_resamples(data):
    redata = jnp.repeat(jnp.array([data]), data.shape[0], axis=0)
    mask = jnp.repeat(~jnp.eye(data.shape[0], dtype=bool).reshape(data.shape[0], data.shape[0], -1), data.shape[-1], axis=2)
    return redata[mask].reshape(data.shape[0], data.shape[0]-1, -1)
