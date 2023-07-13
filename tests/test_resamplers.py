from jaxknife.bootstrap import bootstrap_resamples, jackknife_resamples
from make_testdata import *


def test_bootstrap_resamples():

    # Usual Test Case
    n_resamples = 100
    size = 100
    seed = 0
    testdata1D = make_pos_data(size=size,seed=seed)

    resamples = bootstrap_resamples(testdata1D,n_resamples=n_resamples,seed=seed)
    
    assert resamples.shape == (n_resamples,size,3), "bootstrap_resamples() returned the wrong shape in 1D data case."
    assert resamples[0][0] in testdata1D, "bootstrap_resamples() invented new data in 1D data case."

    testdata2D = make_posvel_data(size=size,seed=seed)

    resamples = bootstrap_resamples(testdata2D,n_resamples=n_resamples,seed=seed)
    
    assert resamples.shape == (n_resamples,size,2,3), "bootstrap_resamples() returned the wrong shap in ND data case."
    assert resamples[0][0] in testdata2D, "bootstrap_resamples() invented new data in ND data case."