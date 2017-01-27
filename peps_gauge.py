
def dm(emps, site):
    nsites = len(emps)
    pdim = int(np.sqrt(emps[0].shape[1]))
    assert pdim**2 == emps[0].shape[1]

    mt = emps[0]
    for i in range(site):
        shape = mt.shape
        ei = np.reshape(mt, (shape[0],pdim,pdim,shape[2]))
        ei = np.einsum("lppr->lr", ei)
        mt = np.einsum(ei, emps[i+1])

    for i in range(-1,site+1,-1):
        dddd
        
def gauge_emps(emps0):
    nsites = len(emps0)


            
    for i in range(nsites):
        
    
def gauge_peps(peps0,auxbond):
    epeps0 = epeps(peps0,peps0)

    emps0 = [None]
