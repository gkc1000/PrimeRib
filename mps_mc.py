import numpy as np
import numpy.random

import mps
import mpo
import spins
import scipy.optimize
import autograd

pdim = 2
nsite = 4
bond = 2

Sz = .5*np.array([[1.,0.],[0.,-1.]])
Sm = np.array([[0.,1.],[0.,0.]])
Sp = Sm.T
I = np.eye(2)

print Sz
print Sm
print Sp

def make_heis1d(nsite):
    """
    make Heisenberg hamiltonian
    """
    ham = [None] * nsite
    # see http://itensor.org/docs.cgi?page=tutorials/MPO
    ham[0] = np.zeros([1, pdim, pdim, 5])
    ham[0][0,:,:,1]= .5 * Sm
    ham[0][0,:,:,2]= .5 * Sp
    ham[0][0,:,:,3]= Sz
    ham[0][0,:,:,4]= I

    ham[-1] = np.zeros([5, pdim, pdim, 1])
    ham[-1][0,:,:,0]= I
    ham[-1][1,:,:,0]= Sp
    ham[-1][2,:,:,0]= Sm
    ham[-1][3,:,:,0]= Sz

    for i in range(1,nsite-1):
        ham[i] = np.zeros([5,pdim,pdim,5])
        ham[i][0,:,:,0]= I
        ham[i][1,:,:,0]= Sp
        ham[i][2,:,:,0]= Sm
        ham[i][3,:,:,0]= Sz
        ham[i][4,:,:,1]= .5*Sm
        ham[i][4,:,:,2]= .5*Sp
        ham[i][4,:,:,3]= Sz
        ham[i][4,:,:,4]= I

    return ham

def test_spins():
    configs = spins.configs(nsite, pdim)
    print configs

def energy(ham, m):
    return mps.dot(m, mpo.mapply(ham, m)) / mps.dot(m,m) 

def mps_from_vec(vec, nsite, pdim, bond):
    m=[None]*nsite
    ptr=0
    m[0] = np.reshape(vec[ptr:ptr+pdim*bond], [1,pdim,bond])
    ptr+=pdim*bond
    for i in range(1,nsite-1):
        m[i] = np.reshape(vec[ptr:ptr+pdim*bond*bond], [bond,pdim,bond])
        ptr+=pdim*bond*bond
    m[-1] = np.reshape(vec[ptr:ptr+pdim*bond], [bond,pdim,1])
    return m

def test_min():
    ham = make_heis1d(nsite)
        
    nparam = 2*pdim*bond+(nsite-2)*pdim*bond*bond
    vec = numpy.random.random([nparam])
    
    def energy_fn(vec):
        m = mps_from_vec(vec, nsite, pdim, bond)
        en = energy(ham, m)
        print en
        return en

    deriv =autograd.grad(energy_fn)

    d= deriv(vec)
    print d.shape
    print "Derivative", d

    def nderiv(vec):
        num_deriv = np.zeros(vec.shape)
        eps = 1.e-6
        for i in xrange(vec.shape[0]):
            vec_eps = vec.copy()
            vec_eps[i] +=eps
            print np.linalg.norm(vec_eps - vec), vec.dtype
            #print "energy", energy_fn(vec_eps) , "energy 2", energy_fn(vec)
            num_deriv[i] = (energy_fn(vec_eps) - energy_fn(vec))/eps

        return num_deriv
    
    num_deriv = nderiv(vec)
    print num_deriv.shape
    print "N derivative", num_deriv
    
    print np.linalg.norm(num_deriv-d)
    
    result = scipy.optimize.minimize(energy_fn, jac=deriv, x0=vec)
    mfin = mps_from_vec(result.x, nsite, pdim, bond)

    # for config in spins.configs(nsite,pdim):
    #     print config, mps.ceval(mfin,config)
    print energy(ham,mfin)
    

def test_mc():
    configs = spins.configs(nsite, pdim)
    
    m = mps.random(nsite, pdim, bond)

    tot = 0.
    for config in configs:
        val = mps.ceval(m, config)
        tot+=val*val

    print tot
    print mps.dot(m,m)
        
    ham = make_heis1d(nsite)

    print 
