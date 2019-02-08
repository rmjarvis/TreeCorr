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

def corr2d(x, y, kappa, w=None, rmax=1., bins=513):

    hrange = [ [-rmax,rmax], [-rmax,rmax] ]
    
    ind = np.linspace(0,len(x)-1,len(x)).astype(int)
    i1, i2 = np.meshgrid(ind,ind)
    i1 = i1.reshape(len(x)**2)
    i2 = i2.reshape(len(x)**2)

    yshift = y[i2]-y[i1]
    xshift = x[i2]-x[i1]
    if w is not None:
        ww = w[i1] * w[i2]
    else:
        ww = None

    mask = (np.abs(xshift) < 4) & (np.abs(yshift) < 4) & (abs(xshift) + abs(yshift) > 0)
    counts = np.histogram2d(xshift, yshift, bins=bins, range=hrange, weights=ww)[0]

    vv = kappa[i1] * kappa[i2]
    if ww is not None: vv *= w

    xi = np.histogram2d(xshift, yshift, bins=bins, range=hrange, weights=vv)[0]
    xi /= counts
    return xi.T


def test_twod():

    # N random points in 2 dimensions
    np.random.seed(42)
    N = 1000
    x = np.random.uniform(-10, 10, N)
    y = np.random.uniform(-10, 10, N)
    
    # Give the points a multivariate Gaussian random field for kappa and gamma
    L1 = [[0.33, 0.09], [-0.01, 0.26]]  # Some arbitrary correlation matrix
    invL1 = np.linalg.inv(L1)
    dists = pdist(np.array([x,y]).T, metric='mahalanobis', VI=invL1)
    K = np.exp(-0.5 * dists**2)
    K = squareform(K)
    np.fill_diagonal(K, 1.)

    A = 2.3
    kappa = np.random.multivariate_normal(np.zeros(N), K*(A**2))

    # Add some noise
    sigma = A/10.
    kappa += np.random.normal(scale=sigma, size=N)
    kappa_err = np.ones_like(kappa) * sigma

    # Calculate the 2D correlation using brute force
    max_sep = 10.
    nbins = 21
    xi_brut = corr2d(x, y, kappa, w=None, rmax=max_sep, bins=nbins)

    cat = treecorr.Catalog(x=x, y=y, k=kappa, w=1./kappa_err**2)
    kk = treecorr.KKCorrelation(min_sep=0., max_sep=max_sep, nbins=nbins, metric='TwoD', bin_slop=0)

    # First the simplest case to get right: cross correlation of the catalog with itself.
    kk.process(cat, cat, metric='TwoD')

    mask = kk.meanr < 9.
    #print('kk.xi.mask = ',kk.xi)
    #print('xi_brut.mask = ',xi_brut)
    #print('diff = ',kk.xi[mask] - xi_brut[mask])
    #print('max abs diff = ',np.max(np.abs(kk.xi[mask] - xi_brut[mask])))
    #print('max rel diff = ',np.max(np.abs(kk.xi[mask] - xi_brut[mask])/np.abs(kk.xi[mask])))
    np.testing.assert_allclose(kk.xi[mask], xi_brut[mask], atol=1.e-7)

    
if __name__ == '__main__':
    test_twod()
