import numpy as np
import mps
import peps

def unit_emps(nsites, pdim):
    emps = mps.zeros(nsites,pdim*pdim,1)

    for i in range(nsites):
        emps[i] = np.reshape(np.eye(pdim*pdim), emps[0].shape)
    return emps

def gauge(emps, gauge):
    nsites = len(emps)

    emps0 = emps[:]
    for i in range(emps):
        pass

    
    
