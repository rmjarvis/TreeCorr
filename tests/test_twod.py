import numpy as np
import treecorr
from scipy.spatial.distance import pdist, squareform

def get_correlation_length_matrix(size, e1, e2):

    if abs(e1)>1:
        e1 = 0
    if abs(e2)>1:
        e2 = 0
    e = np.sqrt(e1**2 + e2**2)
    q = (1-e) / (1+e)
    phi = 0.5 * np.arctan2(e2,e1)
    rot = np.array([[np.cos(phi), np.sin(phi)],
                    [-np.sin(phi), np.cos(phi)]])
    ell = np.array([[size**2, 0],
                    [0, (size * q)**2]])
    L = np.dot(rot.T, ell.dot(rot))
    return L

def corr2d(x, y, kappa, kappa_err=None,
           rmax=1., bins=513):

    hrange = [ [-rmax,rmax], [-rmax,rmax] ]
    
    ind = np.linspace(0,len(x)-1,len(x)).astype(int)
    i1, i2 = np.meshgrid(ind,ind)
    #Filtre = (i1 != i2)
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

    mask = (np.abs(xshift) < 4) & (np.abs(yshift) < 4) & (abs(xshift) + abs(yshift) > 0)
    #print('xshift = ',xshift[mask])
    #print('yshift = ',yshift[mask])
    #print('bins = ',bins)
    #print('range = ',hrange)

    counts = np.histogram2d(xshift, yshift, bins=bins, range=hrange, weights=ww)[0]

    vv = kappa[i1] * kappa[i2]
    if ww is not None:
        vv *= ww
    #print('vv = ',vv[mask])

    xi, X, Y = np.histogram2d(xshift, yshift, bins=bins, range=hrange, weights=vv)
    #print('counts = ',counts[7:14,7:14])
    #print('xi = ',xi[7:14,7:14])
    xi /= counts
    #print('xi => ',xi[7:14,7:14])

    x = X[:-1] + (X[1] - X[0])/2.
    y = Y[:-1] + (Y[1] - Y[0])/2.
    x , y = np.meshgrid(x,y)

    return xi.T, x, y

def test_twod():

    np.random.seed(42)
    N = 1000
    x = np.random.uniform(-10, 10, N)
    y = np.random.uniform(-10, 10, N)
    
    size = 2.
    e1 = 0.
    e2 = 0.
    A = 2.
    
    L = get_correlation_length_matrix(size, e1, e2)
    invL = np.linalg.inv(L)
    
    dists = pdist(np.array([x,y]).T, metric='mahalanobis', VI=invL)
    
    K = np.exp(-0.5 * dists**2)
    K = squareform(K)
    np.fill_diagonal(K, 1.)

    kappa = np.random.multivariate_normal(np.zeros(N), K*(A**2))
    kappa += np.random.normal(scale=A/10., size=N)
    kappa_err = np.ones_like(kappa) * (A/10.)
    #print('kappa = ',kappa)

    cat = treecorr.Catalog(x=x, y=y, k=kappa, w=1./kappa_err**2)
    #print('cat = ',cat)
    kk = treecorr.KKCorrelation(min_sep=0., max_sep=10., nbins=10, metric='TwoD', bin_slop=0)
    #print('kk = ',kk)
    kk.process(cat, cat, metric='TwoD')
    #print('kk.xi = ',kk.xi)
    #print('kk.dx = ',kk.dx)
    #print('kk.dy = ',kk.dy)

    xi_brut, xi_x_brut, xi_y_brut = corr2d(x, y, kappa, kappa_err=None,
                                           rmax=10.5, bins=21)
    #print('shape = ',xi_brut.shape)
    #print('xi_brut = ',xi_brut[7:14,7:14])
    #print('kk.xi = ',kk.xi.reshape(xi_brut.shape)[7:14,7:14])
    #print('diff = ',xi_brut[7:14,7:14] - kk.xi.reshape(xi_brut.shape)[7:14,7:14])

    mask = kk.meanr < 9.
    #print('r = ',kk.rnom.reshape(xi_brut.shape)[7:14,7:14])
    #print('meanr = ',kk.meanr.reshape(xi_brut.shape)[7:14,7:14])
    #print('dx = ',kk.dx.reshape(xi_brut.shape)[7:14,7:14])
    #print('dy = ',kk.dy.reshape(xi_brut.shape)[7:14,7:14])
    #print('xi_x_brut = ',xi_x_brut[7:14,7:14])
    #print('xi_y_brut = ',xi_y_brut[7:14,7:14])

    #print('kk.xi.mask = ',kk.xi[mask])
    #print('xi_brut.mask = ',xi_brut.flatten()[mask])
    #print('diff = ',kk.xi[mask] - xi_brut.flatten()[mask])
    print('max abs diff = ',np.max(np.abs(kk.xi[mask] - xi_brut.flatten()[mask])))
    print('max rel diff = ',np.max(np.abs(kk.xi[mask] - xi_brut.flatten()[mask])/np.abs(kk.xi[mask])))
    np.testing.assert_allclose(kk.xi[mask], xi_brut.flatten()[mask], atol=1.e-7)

    
if __name__ == '__main__':
    test_twod()
