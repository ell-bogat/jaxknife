from jaxknife.bootstrap import bootstrap
from make_testdata import *
from jaxknife.stats_funcs import com_1D
import pytest

def test_bootstrap():
    
    n_resamples = 1000
    size = 100
    seed = 0

    pos_data = make_pos_data(size=size,seed=seed)
    com_arr = bootstrap(pos_data,com_1D,n_resamples=n_resamples,seed=seed)
    
    assert np.shape(com_arr) == (n_resamples,), 'bootstrap() returned the wrong shape of stats array.'
    assert np.median(com_arr) == pytest.approx(0,abs=0.1), 'bootstrap() resulted in a weird center of mass.'