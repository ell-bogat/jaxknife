import numpy as np

def make_pos_data(seed=0):
    np.random.seed(seed)
    return np.random.uniform(-1, 1, size=(100, 3))

def make_posvel_data(seed=0):
    np.random.seed(seed)
    return np.random.uniform(-1, 1, size=(100, 2, 3))

def np_bootstrap(data,statistic_func,n_resamples=1000,seed=0):
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

if __name__=='__main__':
    pass