import jax
import jax.numpy as jnp


def bootstrap(data, statistic_func, n_resamples, seed):
    """Bootstrap.

    Args:
        data (array): Input data with shape (N, ...) where N is the number of realizations of the observable.
        statistic_func (callable): Function to calculate a statistic on one set of realizations of the observable.
        n_resamples (int): Number of times to resample the data.
        seed (int): Seed for RNG.

    Returns:
        jax array: The value of the statistic for each resample.
    """
    resamples = bootstrap_resamples(data, n_resamples, seed)
    est_arr = _apply_vect_statistic(statistic_func, resamples)
    return est_arr


def jackknife(data, statistic_func):
    """Jackknife.

    Args:
        data (array): Input data with shape (N, ...) where N is the number of realizations of the observable.
        statistic_func (callable): Function to calculate a statistic on one set of realizations of the observable.

    Returns:
        jax array: The value of the statistic for each resample.
    """
    resamples = jackknife_resamples(data)
    est_arr = _apply_vect_statistic(statistic_func, resamples)
    return est_arr


def _apply_vect_statistic(statistic_func, resamples):
    """Vectorize and apply statistic function.

    Args:
        statistic_func (callable): Function to calculate a statistic on one set of realizations of the observable.
        resamples (jax array): Resampled data with shape (n_resamples, N, ...)

    Returns:
        jax array: The value of the statistic for each resample.
    """
    statistic_vectorized = jax.vmap(statistic_func)
    return statistic_vectorized(resamples)


def bootstrap_resamples(data, n_resamples, seed):
    """Generate bootstrap resamples.

    Args:
        data (array): Input data with shape (N, ...) where N is the number of realizations of the observable.
        n_resamples (int): Number of times to resample the data.
        seed (int): Seed for RNG.

    Returns:
        jax array: Resampled data with shape (n_resamples, N, ...)
    """
    key = jax.random.PRNGKey(seed)
    resamples_size = (n_resamples, data.shape[0])
    resamples = jax.random.choice(key, data, resamples_size, replace=True)
    return resamples


def jackknife_resamples(data):
    """Generate jackknife resamples.

    Args:
        data (array): Input data with shape (N, ...) where N is the number of realizations of the observable.

    Returns:
        jax array: Resampled data with shape (N, N, ...)
    """
    redata = jnp.repeat(jnp.array([data]), data.shape[0], axis=0)
    mask = jnp.repeat(
        ~jnp.eye(data.shape[0], dtype=bool).reshape(data.shape[0], data.shape[0], -1),
        data.shape[-1],
        axis=2,
    )
    return redata[mask].reshape(data.shape[0], data.shape[0] - 1, -1)


if __name__ == "__main__":
    pass
