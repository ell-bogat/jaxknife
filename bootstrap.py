import jax
import jax.numpy as jnp
import numpy as np

def bootstrap_loop(data,estimator,n_resamples=1000,seed=0):
    """
    Bootstrap method using numpy and a for loop

    Input: 
    data        (array-like)
    estimator   (function)
    n_samples   (int)
    seed        (int)
    """

    est_arr = np.zeros(n_resamples)
    key = jax.random.PRNGKey(seed)

 
    for i in range(n_resamples):
        np.random.seed(seed)
        key, subkey = jax.random.split(key)
        resample = jax.random.choice(subkey,data,(len(data),),replace=True,axis=0)
        est_arr[i] = estimator(resample)

    return est_arr

def bootstrap_nparray(data,estimator,n_resamples=1000,seed=0):
    """
    Bootstrap method using one large numpy array
    ~~UNFINISHED~~

    Input: 
    data        (array-like)
    estimator   (function)
    n_samples   (int)
    seed        (int)
    """

    est_arr = np.zeros(n_resamples)

 
    np.random.seed(seed)
    #resample = jax.random.choice(subkey,data,(len(data),),replace=True,axis=0)
    #est_arr[i] = estimator(resample)

    return 

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
