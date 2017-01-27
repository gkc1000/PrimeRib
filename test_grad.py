import autograd
import autograd.numpy as np
import autograd.numpy.random as npr

a = npr.random([2,2,2,2])

def fn(a, b):
    bs = np.reshape(b, [2,2,2])
    c = np.einsum("apqb,cqd->acpbd", a, bs)
    return np.sum(c)

def test():
    b = npr.random([8])
    def f(b):
        return fn(a, b)
    deriv = autograd.grad(f)
    print f(b)
    print deriv(b)
                    
