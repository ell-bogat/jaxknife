import jax
import jax.numpy as jnp
import numpy as np

def bootstrap(data, statistic_func, n_resamples, seed):
    resamples = bootstrap_resamples(data, n_resamples, seed)
    est_arr = _apply_vect_statistic(statistic_func, resamples)
    return est_arr

def jackknife(data, statistic_func):
    resamples = jackknife_resamples(data)
    est_arr = _apply_vect_statistic(statistic_func, resamples)
    return est_arr

def _apply_vect_statistic(statistic_func, resamples):
    statistic_vectorized = jax.vmap(statistic_func)
    return statistic_vectorized(resamples)


# def jaxstrapknife(data, statistic_func, n_resamples=None, seed=0):
#     """
#     Bootstrap method using JAX arrays
#     Input: 
#     data            (array-like)
#     statistic_func  (function)
#     n_samples       (int)
#     seed            (int)
#     """
#     if n_resamples is None:
#         resamples = jackknife_resamples(data)
#     else:
#         resamples = bootstrap_resamples(data, n_resamples, seed)
#     statistic_vectorized = jax.vmap(statistic_func)
#     est_arr = statistic_vectorized(resamples)

    # return est_arr


def bootstrap_resamples(data, n_resamples, seed):
    key = jax.random.PRNGKey(seed)
    resamples = jax.random.choice(key,data,(n_resamples,*data.shape),replace=True)
    return resamples


def jackknife_resamples(data):
    redata = jnp.repeat(jnp.array([data]), data.shape[0], axis=0)
    mask = jnp.repeat(~jnp.eye(data.shape[0], dtype=bool).reshape(data.shape[0], data.shape[0], -1), data.shape[-1], axis=2)
    return redata[mask].reshape(data.shape[0], data.shape[0]-1, -1)
