import numpy as np


def make_pos_data(size=100, seed=0):
    """Generate random 3D positions.

    Args:
        size (int, optional): Number of points. Defaults to 100.
        seed (int, optional): Seed. Defaults to 0.

    Returns:
        array:
    """
    np.random.seed(seed)
    return np.random.uniform(-1, 1, size=(size, 3))


def make_posvel_data(size=100, seed=0):
    """Generate random velocities and positions.

    Args:
        size (int, optional): Number of points. Defaults to 100.
        seed (int, optional): Seed. Defaults to 0.

    Returns:
        array:
    """
    np.random.seed(seed)
    return np.random.uniform(-1, 1, size=(size, 2, 3))


if __name__ == "__main__":
    pass
