import jax.numpy as jnp


def com_1D(positions, masses=None, axis=0) -> float:
    """Center of mass in a single axis.

    Args:
        positions (array): Array with 3D positions.
        masses (array, optional): Mass at each position point. If None, all masses are unitary. Defaults to None.
        axis (int, optional): Axis along which to compute the center of mass of. Defaults to 0 (first).

    Raises:
        Exception: Data is not a 2D array.

    Returns:
        float: Center of mass.
    """

    if masses is None:
        masses = jnp.ones((len(positions)))

    if positions.ndim == 2:
        xs = positions[:, axis]
    else:
        raise Exception("Only implemented for 2d data")

    return jnp.sum(xs * masses) / len(xs)


def vel_disp_1D(velocities, axis=0) -> float:
    """Velocity dispersion

    Args:
        velocities (array): Array with 3D velocities and positions.
        axis (int, optional): Axis along which to compute the center of mass of. Defaults to 0 (first).

    Raises:
        Exception: Data is not a 3D array.

    Returns:
        float: Center of mass.
    """
    if velocities.ndim == 3:
        vxs = velocities[:, 1, axis]
    else:
        raise Exception("Only implemented for 3d data")

    return jnp.std(vxs)


if __name__ == "__main__":
    pass
