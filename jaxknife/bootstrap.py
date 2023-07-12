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

def jaxstrapknife(data, statistic_func, n_resamples=None, seed=0):
    """
    Bootstrap method using JAX arrays
    Input: 
    data            (array-like)
    statistic_func  (function)
    n_samples       (int)
    seed            (int)
    """
    if n_resamples is None:
        resamples = jackknife_resamples(data)
    else:
        resamples = bootstrap_resamples(data, n_resamples, seed)
    statistic_vectorized = jax.vmap(statistic_func)
    est_arr = statistic_vectorized(resamples)

    return est_arr


def bootstrap_resamples(data, n_resamples, seed):
    key = jax.random.PRNGKey(seed)
    resamples = jax.random.choice(key,data,(n_resamples,*data.shape),replace=True)
    return resamples


def jackknife_resamples(data):
    redata = jnp.repeat(jnp.array([data]), data.shape[0], axis=0)
    mask = jnp.repeat(~jnp.eye(data.shape[0], dtype=bool).reshape(data.shape[0], data.shape[0], -1), data.shape[-1], axis=2)
    return redata[mask].reshape(data.shape[0], data.shape[0]-1, -1)
