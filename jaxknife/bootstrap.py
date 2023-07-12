import jax
import jax.numpy as jnp
import numpy as np

def bootstrap(data,statistic_func,n_resamples=1000,seed=0) -> np.ndarray:
    """Slow Bootstrap.

    Use numpy and a for loop to generate bootstrap errors on a given statistic.

    Args:
        data (array-like): Input data, first axis must a sequence of the realizations of the random variable.
        statistic_func (callable): Function to calculate the desired statistic over the given data.
        n_resamples (int, optional): Number of bootstrap resamples. Defaults to 1000.
        seed (int, optional): RNG seed. Defaults to 0.

    Returns:
        array: Values of the statistic calculated for each resample.
    """

    est_arr = np.zeros(n_resamples)
    np.random.seed(seed)
 
    for i in range(n_resamples):
        idx = np.random.randint(len(data), size=len(data))
        resample = data[idx]
        
        est_arr[i] = statistic_func(resample)

    return est_arr

def jaxstrapknife(data, statistic_func, n_resamples=None, seed=0):

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
