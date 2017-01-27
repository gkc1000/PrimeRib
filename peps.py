#import numpy as np
import mps
import mpo
import autograd
import autograd.numpy as np
einsum=autograd.numpy.einsum

def zeros(shape,pdim,bond):
    peps = empty(shape,pdim,bond)
    for i in range(peps.shape[0]):
        for j in range(peps.shape[1]):
            peps[i,j] = np.zeros_like(peps[i,j])
    return peps

def random(shape,pdim,bond):
    peps = empty(shape,pdim,bond)
    for i in range(peps.shape[0]):
        for j in range(peps.shape[1]):
            peps[i,j] = np.random.random(peps[i,j].shape)
    return peps
    
def empty(shape,pdim,bond):
    peps = np.zeros(shape, dtype=np.object)

    # dimension of bonds, ludr
    ldims=np.ones(shape[1])*bond
    ldims[0]=1
    rdims=np.ones(shape[1])*bond
    rdims[-1]=1
    ddims=np.ones(shape[0])*bond
    ddims[0]=1
    udims=np.ones(shape[0])*bond
    udims[-1]=1

    for i in range(shape[0]):
        for j in range(shape[1]):
            peps[i,j] = np.empty([pdim,ldims[j],udims[i],ddims[i],rdims[j]])
    return peps

def ceval(peps,config,auxbond):
    # this is a 2d config
    #peps_config = np.reshape(config, peps.shape)
    shape = peps.shape
    cpeps=np.zeros(shape, dtype=np.object)
    for i in range(shape[0]):
        for j in range(shape[1]):
            cpeps[i,j]=peps[i,j][config[i,j],:,:,:,:]
    return contract_cpeps(cpeps,auxbond)

def cpeps(peps,config):
    shape = peps.shape
    cpeps=np.empty(shape, dtype=np.object)
    peps_config = np.reshape(config, peps.shape)
    for i in range(shape[0]):
        for j in range(shape[1]):
            cpeps[i,j]=peps[i,j][peps_config[i,j],:,:,:,:]
    return cpeps

def epeps(pepsa,pepsb):
    shape = pepsa.shape
    epeps = np.empty(shape, dtype=np.object)
    for i in range(shape[0]):
        for j in range(shape[1]):
            epeps[i,j]=einsum("pludr,pLUDR->lLuUdDrR",pepsa[i,j],pepsb[i,j])
            eshape=epeps[i,j].shape
            epeps[i,j]=np.reshape(epeps[i,j],(eshape[0]*eshape[1],
                                              eshape[2]*eshape[3],
                                              eshape[4]*eshape[5],
                                              eshape[6]*eshape[7]))
    return epeps

def add(pepsa,pepsb):
    pdim = pepsa[0,0].shape[0]
    bonda = pepsa[0,0].shape[-1] # right bond
    bondb = pepsb[0,0].shape[-1] # right bond
    shape = pepsa.shape
    pepsc = zeros(shape,pdim,bonda+bondb)

    for i in range(shape[0]):
        for j in range(shape[1]):
            pepsc[i,j][:,:bonda,:bonda,:bonda,:bonda]=pepsa[i,j][:,:,:,:,:]
            pepsc[i,j][:,bonda:,bonda:,bonda:,bonda:]=pepsb[i,j][:,:,:,:,:]
    return pepsc
    
def create(shape,pdim,config):
    peps_config = np.reshape(config, shape)
    peps0 = zeros(shape,pdim,1)
    for i in range(shape[0]):
        for j in range(shape[1]):
            peps0[i,j][peps_config[i,j],0,0,0,0]=1.
    return peps0

      
def dot(pepsa,pepsb,auxbond):
    epeps0 = epeps(pepsa,pepsb)
    return contract_cpeps(epeps0, auxbond)

def size(peps):
    size=0
    for i in range(peps.shape[0]):
        for j in range(peps.shape[1]):
            size+=peps[i,j].size
    return size
    
def flatten(peps):
    vec=np.empty((0))
    for i in range(peps.shape[0]):
        for j in range(peps.shape[1]):
            vec = np.append(vec,np.ravel(peps[i,j]))

    return vec

def aspeps(vec,shape,pdim,bond):
    peps0 = empty(shape,pdim,bond)
    assert vec.size == size(peps0)
    ptr=0
    for i in range(shape[0]):
        for j in range(shape[1]):
            nelem = peps0[i,j].size
            peps0[i,j] = np.reshape(vec[ptr:ptr+nelem],
                                    peps0[i,j].shape)
            ptr += nelem
    return peps0


    
    
def contract_cpeps(cpeps,auxbond):
    cmps0 = [None] * cpeps.shape[1]
    for i in range(cpeps.shape[1]):
        l,u,d,r = cpeps[0,i].shape
        cmps0[i] = np.reshape(cpeps[0,i], (l,u*d,r))

    for i in range(1,cpeps.shape[0]):
        cmpo = [None] * cpeps.shape[1]
        for j in range(cpeps.shape[1]):
            l,u,d,r = cpeps[i,j].shape
            cmpo[j] = cpeps[i,j]

        cmps0 = mpo.mapply(cmpo,cmps0)

        if auxbond is not None: # compress
            #print "compressing"
            cmps0 = mps.compress(cmps0,"l",auxbond)
            
    return mps.ceval(cmps0, [0]*cpeps.shape[1])


