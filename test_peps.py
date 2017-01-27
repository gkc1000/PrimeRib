import numpy as np
import numpy.linalg
import numpy.random
import peps
import peps_h
import spins
import autograd
import scipy.optimize

def test():
    pdim =2
    bond=2
    numpy.random.seed(5)

    nr=4
    nc=4
    configs = spins.configs(nr*nc,pdim)
    zpeps = peps.random([nr,nc],pdim,bond)
    print zpeps

    for i in range(nr):
        for j in range(nc):
            print np.linalg.norm(zpeps[i,j])

    #auxbond = 2
    # for config in configs:
    #     print config
    #     cpeps = peps.cpeps(zpeps,config)
    #     print "norms", np.linalg.norm(cpeps[0,0]), np.linalg.norm(cpeps[0,1]), np.linalg.norm(cpeps[1,0]), np.linalg.norm(cpeps[1,1])
    #     print peps.ceval(zpeps,config,auxbond)

    php = peps_h.eval_heish(zpeps, None)
    pp = peps.dot(zpeps,zpeps,None)
    print pp
    print numpy.random.random([1])
    print "Expectation value", php,pp, php/pp
    # 
    # for i in range(nr):
    #     for j in range(nc):
    #         print np.linalg.norm(cpeps[i,j])

def test_min():
    numpy.random.seed(5)
    nr=6
    nc=4

    auxbond = 8
    def energy_fn(vec, pdim,bond):
        P = peps.aspeps(vec, (nr,nc), pdim, bond)
        PHP = peps_h.eval_heish(P, auxbond)
        PP = peps.dot(P,P,auxbond)
        print PHP,PP,PHP/PP
        return PHP/PP

    afma=np.zeros([nr*nc])

    configa = np.zeros([nr,nc], dtype=np.int)
    configb = np.zeros([nr,nc], dtype=np.int)
    
    for i in range(nr):
        for j in range(nc):
            configa[i,j] = (i + j) % 2
            configb[i,j] = (i + j + 1) % 2

    #print afma
    #print afmb

    pdim=2
    pepsa = peps.create((nr,nc),pdim,configa)
    pepsb = peps.create((nr,nc),pdim,configb)

    print peps.dot(pepsa,pepsa,None)
    print peps.dot(pepsb,pepsb,None)
    print peps.dot(pepsa,pepsb,None)
    
    PHP = peps_h.eval_heish(pepsa, None)
    #print "config", configa[0,0], configa[0,1]
    
    print "PHP energy", PHP
    print "PP", peps.dot(pepsa,pepsa,None)
    # flata = peps.flatten(pepsa)
    # pepsA = peps.aspeps(flata, (nr,nc), pdim, bond=1)
    # print "reconstituted dot", peps.dot(pepsA,pepsa,None)
    # print pepsa
    # print pepsA
    
    print "A energy", energy_fn(peps.flatten(pepsa), pdim, bond=1)
    print "B energy", energy_fn(peps.flatten(pepsb), pdim, bond=1)

    print "---start opt-----"
    pdim=2
    bond=2

    peps0 = peps.add(pepsa,pepsb) # this has bond=2
    pepsc = peps.zeros(pepsa.shape, pdim, bond-2) # add some zeros to make up full bond
    peps0 = peps.add(peps0, pepsc)

    peps0 = add_noise(peps0, 0.1)

    print energy_fn(peps.flatten(peps0), pdim, bond)
    
    #vec = 0.3*np.random.random(nparam)
    vec = peps.flatten(peps0)
    #0.3*np.random.random(nparam)

    def bound_energy_fn(vec):
        return energy_fn(vec, pdim, bond)
    
    deriv = autograd.grad(bound_energy_fn)
    d = deriv(vec)

    print bound_energy_fn(vec)
    
    result = scipy.optimize.minimize(bound_energy_fn, jac=deriv, x0=vec)
    print "max value", np.max(result.x)
    P0 = peps.aspeps(result.x, (nr,nc), pdim, bond)

    
def add_noise(peps0, noise):
    ret_peps = peps0.copy()
    
    for i in range(peps0.shape[0]):
        for j in range(peps0.shape[1]):
            ret_peps[i,j] += noise*numpy.random.random(ret_peps[i,j].shape)

    return ret_peps
