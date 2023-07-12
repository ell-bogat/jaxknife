import numpy as np

def make_pos_data(seed=0):
    np.random.seed(seed)
    return np.random.uniform(-1, 1, size=(100, 3))

def make_posvel_data(seed=0):
    np.random.seed(seed)
    return np.random.uniform(-1, 1, size=(100, 2, 3))

if __name__=='__main__':
    pass