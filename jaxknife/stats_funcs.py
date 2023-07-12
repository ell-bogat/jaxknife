import jax.numpy as jnp

def com_1D(positions, masses=None, axis=0):

    if masses is None:
        masses = jnp.ones((len(positions)))

    if positions.ndim == 2:
        xs = positions[:,axis]
    else:
        raise Exception('Only implemented for 2d data')

    return jnp.sum(xs * masses) / len(xs)

def vel_disp_1D(velocities, axis =0):
    
    if velocities.ndim == 3:
        vxs = velocities[:,1,axis]
    else:
        raise Exception('Only implemented for 3d data')
    
    return jnp.std(vxs)

