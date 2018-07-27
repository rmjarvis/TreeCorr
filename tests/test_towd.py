import numpy as np


def vcorr2d(x, y, kappa, kappa_err=None,
            rmax=1., bins=513):

    hrange = [ [-rmax,rmax], [-rmax,rmax] ]
    
    ind = np.linspace(0,len(x)-1,len(x)).astype(int)
    i1, i2 = np.meshgrid(ind,ind)
    Filtre = (i1 != i2) #& (indx>indy))
    i1 = i1.reshape(len(x)**2)
    i2 = i2.reshape(len(x)**2)
    #Filtre = Filtre.reshape(len(x)**2)
    
    #i1 = i1[Filtre]
    #i2 = i2[Filtre]
    #del Filtre
    
    yshift = y[i2]-y[i1]
    xshift = x[i2]-x[i1]
    if kappa_err is not None:
        weight = 1. / kappa_err**2
        ww = weight[i1] * weight[i2]
    else:
        ww = None
        
    counts = np.histogram2d(xshift, yshift, bins=bins, range=hrange, weights=ww)[0]

    vv = kappa[i1] * kappa[i2]
    if ww is not None:
        vv *= ww
        
    xi, X, Y = np.histogram2d(xshift, yshift, bins=bins, range=hrange, weights=vv)
    xi /= counts
    
    x = copy.deepcopy(X[:-1]) + (X[1] - X[0])/2.
    y = copy.deepcopy(Y[:-1]) + (Y[1] - Y[0])/2.
    x , y = np.meshgrid(x,y)
    
    return xi.T, x, y
