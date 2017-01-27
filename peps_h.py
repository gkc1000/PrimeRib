import numpy as np
import autograd
import autograd.numpy
einsum=autograd.numpy.einsum
import peps
import copy

I = np.eye(2)
#Sz = I
#Sm = I
#Sp = I
Sz = .5*np.array([[1.,0.],[0.,-1.]])
Sm = np.array([[0.,1.],[0.,0.]])
Sp = Sm.T

def eval_hbond(pepsa, i, j, auxbond):
    pepsb = pepsa.copy()
    pepsb[i,j] = einsum("pq,qludr->pludr",Sz, pepsa[i,j])
    pepsb[i,j+1] = einsum("pq,qludr->pludr",Sz, pepsa[i,j+1])

    valzz = peps.dot(pepsb,pepsa,auxbond)

    pepsb = pepsa.copy()
    pepsb[i,j] = einsum("pq,qludr->pludr",Sp, pepsa[i,j])
    pepsb[i,j+1] = einsum("pq,qludr->pludr",Sm, pepsa[i,j+1])

    valpm = peps.dot(pepsb,pepsa,auxbond)

    pepsb = pepsa.copy()
    pepsb[i,j] = einsum("pq,qludr->pludr",Sm, pepsa[i,j])
    pepsb[i,j+1] = einsum("pq,qludr->pludr",Sp, pepsa[i,j+1])

    valmp = peps.dot(pepsb,pepsa,auxbond)

    return valzz + .5*(valpm+valmp)
#return valzz


def eval_vbond(pepsa, i, j,auxbond):
    pepsb = pepsa.copy()
    pepsb[i,j] = einsum("pq,qludr->pludr",Sz, pepsa[i,j])
    pepsb[i+1,j] = einsum("pq,qludr->pludr",Sz, pepsa[i+1,j])
    
    valzz = peps.dot(pepsb,pepsa,auxbond)

    pepsb = pepsa.copy()
    pepsb[i,j] = einsum("pq,qludr->pludr",Sp, pepsa[i,j])
    pepsb[i+1,j] = einsum("pq,qludr->pludr",Sm, pepsa[i+1,j])

    valpm = peps.dot(pepsb,pepsa,auxbond)
    
    pepsb = pepsa.copy()
    pepsb[i,j] = einsum("pq,qludr->pludr",Sm, pepsa[i,j])
    pepsb[i+1,j] = einsum("pq,qludr->pludr",Sp, pepsa[i+1,j])

    valmp = peps.dot(pepsb,pepsa,auxbond)

    return valzz + .5*(valpm+valmp)
#return valzz

def eval_heish(pepsa, auxbond):
    shape = pepsa.shape
    nr,nc=shape
    val=0.
    for i in range(nr):
        for j in range(nc-1):
            val += eval_hbond(pepsa,i,j,auxbond)
    
    for i in range(nr-1):
        for j in range(nc):
            val += eval_vbond(pepsa,i,j,auxbond)

    return val
