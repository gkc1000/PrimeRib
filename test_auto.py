import numpy as np
import numpy.random
import autograd.numpy.linalg
svd = autograd.numpy.linalg.svd
def test():

    def func(mat):
        u,s,vt=svd(mat, full_matrices=False)
        print s
        return np.trace(np.diag(s*s))

    deriv = autograd.grad(func)

    print deriv(np.random.random((3,3)))
