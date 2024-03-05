# Copyright (c) 2003-2024 by Mike Jarvis
#
# TreeCorr is free software: redistribution and use in source and binary forms,
# with or without modification, are permitted provided that the following
# conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions, and the disclaimer given in the accompanying LICENSE
#    file.
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions, and the disclaimer given in the documentation
#    and/or other materials provided with the distribution.

import numpy as np
import treecorr
import time

from test_helper import assert_raises, timer

# This is Pierre-Francois's original code for calculating the two-d correlation function.
def corr2d(x, y, kappa1, kappa2, w=None, rmax=1., bins=513, return_counts=False):

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

    counts = np.histogram2d(xshift, yshift, bins=bins, range=hrange, weights=ww)[0]

    vv = np.real(kappa1[i1] * kappa2[i2])
    if ww is not None: vv *= ww

    xi = np.histogram2d(xshift, yshift, bins=bins, range=hrange, weights=vv)[0]

    # Note: This calculation includes the "pairs" where both objects are the same as part
    # of the zero lag bin.  We don't want that, so subtract it off.
    mid = bins // 2
    if w is None:
        auto = np.real(np.sum(kappa1 * kappa2))
        sumww = len(kappa1)
    else:
        auto = np.real(np.sum(kappa1 * kappa2 * w**2))
        sumww = np.sum(w**2)
    xi[mid,mid] -= auto
    counts[mid,mid] -= sumww

    xi /= counts
    if return_counts:
        return xi.T, counts.T
    else:
        return xi.T


@timer
def test_twod():
    try:
        from scipy.spatial.distance import pdist, squareform
    except ImportError:
        print('Skipping test_twod, since uses scipy, and scipy is not installed.')
        return

    # N random points in 2 dimensions
    rng = np.random.RandomState(8675309)
    N = 200
    x = rng.uniform(-20, 20, N)
    y = rng.uniform(-20, 20, N)

    # Give the points a multivariate Gaussian random field for kappa and gamma
    L1 = [[0.33, 0.09], [-0.01, 0.26]]  # Some arbitrary correlation matrix
    invL1 = np.linalg.inv(L1)
    dists = pdist(np.array([x,y]).T, metric='mahalanobis', VI=invL1)
    K = np.exp(-0.5 * dists**2)
    K = squareform(K)
    np.fill_diagonal(K, 1.)

    A = 2.3
    kappa = rng.multivariate_normal(np.zeros(N), K*(A**2))

    # Add some noise
    sigma = A/10.
    kappa += rng.normal(scale=sigma, size=N)
    kappa_err = np.ones_like(kappa) * sigma

    # Make gamma too
    gamma1 = rng.multivariate_normal(np.zeros(N), K*(A**2))
    gamma1 += rng.normal(scale=sigma, size=N)
    gamma2 = rng.multivariate_normal(np.zeros(N), K*(A**2))
    gamma2 += rng.normal(scale=sigma, size=N)
    gamma = gamma1 + 1j * gamma2

    # Calculate the 2D correlation using brute force
    max_sep = 21.
    nbins = 21
    xi_brut = corr2d(x, y, kappa, kappa, w=None, rmax=max_sep, bins=nbins)

    cat1 = treecorr.Catalog(x=x, y=y, k=kappa, g1=gamma1, g2=gamma2)
    kk = treecorr.KKCorrelation(min_sep=0., max_sep=max_sep, nbins=nbins, bin_type='TwoD',
                                brute=True)

    # Check the dx, dy and related properies
    bin_size = 2*max_sep / nbins
    dx = np.linspace(-max_sep+bin_size/2, max_sep-bin_size/2, nbins)
    dx, dy = np.meshgrid(dx, dx)

    np.testing.assert_allclose(kk.left_edges, dx - bin_size/2)
    np.testing.assert_allclose(kk.right_edges, dx + bin_size/2)
    np.testing.assert_allclose(kk.top_edges, dy + bin_size/2)
    np.testing.assert_allclose(kk.bottom_edges, dy - bin_size/2)
    np.testing.assert_allclose(kk.dxnom, dx)
    np.testing.assert_allclose(kk.dynom, dy)
    np.testing.assert_allclose(kk.rnom, np.hypot(dx,dy))
    np.testing.assert_allclose(np.exp(kk.logr), np.hypot(dx,dy))

    # Non-TwoD Correlation objects don't have some of these.
    kk2 = treecorr.KKCorrelation(min_sep=0., max_sep=max_sep, nbins=nbins, bin_type='Linear')
    with assert_raises(AttributeError):
        kk2.top_edges
    with assert_raises(AttributeError):
        kk2.bottom_edges
    with assert_raises(AttributeError):
        kk2.dxnom
    with assert_raises(AttributeError):
        kk2.dynom

    # First the simplest case to get right: cross correlation of the catalog with itself.
    kk.process(cat1, cat1)

    print('max abs diff = ',np.max(np.abs(kk.xi - xi_brut)))
    print('max rel diff = ',np.max(np.abs(kk.xi - xi_brut)/np.abs(kk.xi)))
    np.testing.assert_allclose(kk.xi, xi_brut, atol=2.e-7)

    # Auto-correlation should do the same thing.
    kk.process(cat1)
    print('max abs diff = ',np.max(np.abs(kk.xi - xi_brut)))
    print('max rel diff = ',np.max(np.abs(kk.xi - xi_brut)/np.abs(kk.xi)))
    np.testing.assert_allclose(kk.xi, xi_brut, atol=2.e-7)

    # Repeat with weights.
    xi_brut = corr2d(x, y, kappa, kappa, w=1./kappa_err**2, rmax=max_sep, bins=nbins)
    cat2 = treecorr.Catalog(x=x, y=y, k=kappa, g1=gamma1, g2=gamma2, w=1./kappa_err**2)
    # NB. Testing that min_sep = 0 is default
    kk = treecorr.KKCorrelation(max_sep=max_sep, nbins=nbins, bin_type='TwoD', brute=True)
    kk.process(cat2, cat2)
    print('max abs diff = ',np.max(np.abs(kk.xi - xi_brut)))
    print('max rel diff = ',np.max(np.abs(kk.xi - xi_brut)/np.abs(kk.xi)))
    np.testing.assert_allclose(kk.xi, xi_brut, atol=2.e-7)

    kk.process(cat2)
    np.testing.assert_allclose(kk.xi, xi_brut, atol=2.e-7)

    # Check GG
    xi_brut = corr2d(x, y, gamma, np.conj(gamma), rmax=max_sep, bins=nbins)
    # Equivalent bin_size = 2.  Check omitting nbins
    gg = treecorr.GGCorrelation(max_sep=max_sep, bin_size=2., bin_type='TwoD', brute=True)
    gg.process(cat1)
    print('max abs diff = ',np.max(np.abs(gg.xip - xi_brut)))
    print('max rel diff = ',np.max(np.abs(gg.xip - xi_brut)/np.abs(gg.xip)))
    np.testing.assert_allclose(gg.xip, xi_brut, atol=2.e-7)

    xi_brut = corr2d(x, y, gamma, np.conj(gamma), w=1./kappa_err**2, rmax=max_sep, bins=nbins)
    # Check omitting max_sep
    gg = treecorr.GGCorrelation(bin_size=2, nbins=nbins, bin_type='TwoD', brute=True)
    gg.process(cat2)
    print('max abs diff = ',np.max(np.abs(gg.xip - xi_brut)))
    print('max rel diff = ',np.max(np.abs(gg.xip - xi_brut)/np.abs(gg.xip)))
    np.testing.assert_allclose(gg.xip, xi_brut, atol=2.e-7)

    # Check NK
    xi_brut = corr2d(x, y, np.ones_like(kappa), kappa, rmax=max_sep, bins=nbins)
    # Check slightly larger bin_size gets rounded down
    nk = treecorr.NKCorrelation(max_sep=max_sep, bin_size=2.05, bin_type='TwoD', brute=True)
    nk.process(cat1, cat1)
    print('max abs diff = ',np.max(np.abs(nk.xi - xi_brut)))
    print('max rel diff = ',np.max(np.abs(nk.xi - xi_brut)/np.abs(nk.xi)))
    np.testing.assert_allclose(nk.xi, xi_brut, atol=1.e-7)

    xi_brut = corr2d(x, y, np.ones_like(kappa), kappa, w=1./kappa_err**2, rmax=max_sep, bins=nbins)
    # Check very small, but non-zeo min_sep
    nk = treecorr.NKCorrelation(min_sep=1.e-6, max_sep=max_sep, nbins=nbins, bin_type='TwoD',
                                brute=True)
    nk.process(cat2, cat2)
    print('max abs diff = ',np.max(np.abs(nk.xi - xi_brut)))
    print('max rel diff = ',np.max(np.abs(nk.xi - xi_brut)/np.abs(nk.xi)))
    np.testing.assert_allclose(nk.xi, xi_brut, atol=1.e-7)

    # Check NN
    xi_brut, counts = corr2d(x, y, np.ones_like(kappa), np.ones_like(kappa),
                             rmax=max_sep, bins=nbins, return_counts=True)
    nn = treecorr.NNCorrelation(max_sep=max_sep, nbins=nbins, bin_type='TwoD', brute=True)
    nn.process(cat1)
    print('max abs diff = ',np.max(np.abs(nn.npairs - counts)))
    print('max rel diff = ',np.max(np.abs(nn.npairs - counts)/np.abs(nn.npairs)))
    np.testing.assert_allclose(nn.npairs, counts, atol=1.e-7)

    nn.process(cat1, cat1)
    print('max abs diff = ',np.max(np.abs(nn.npairs - counts)))
    print('max rel diff = ',np.max(np.abs(nn.npairs - counts)/np.abs(nn.npairs)))
    np.testing.assert_allclose(nn.npairs, counts, atol=1.e-7)

    xi_brut, counts = corr2d(x, y, np.ones_like(kappa), np.ones_like(kappa),
                             w=1./kappa_err**2, rmax=max_sep, bins=nbins, return_counts=True)
    nn = treecorr.NNCorrelation(max_sep=max_sep, nbins=nbins, bin_type='TwoD', brute=True)
    nn.process(cat2)
    print('max abs diff = ',np.max(np.abs(nn.weight - counts)))
    print('max rel diff = ',np.max(np.abs(nn.weight - counts)/np.abs(nn.weight)))
    np.testing.assert_allclose(nn.weight, counts, atol=1.e-7)

    nn.process(cat2, cat2)
    print('max abs diff = ',np.max(np.abs(nn.weight - counts)))
    print('max rel diff = ',np.max(np.abs(nn.weight - counts)/np.abs(nn.weight)))
    np.testing.assert_allclose(nn.weight, counts, atol=1.e-7)

    # Check I/O
    try:
        import fitsio
    except ImportError:
        pass
    else:
        fits_name = 'output/gg_twod.fits'
        gg.write(fits_name)
        gg2 = treecorr.GGCorrelation.from_file(fits_name)
        np.testing.assert_allclose(gg2.npairs, gg.npairs)
        np.testing.assert_allclose(gg2.weight, gg.weight)
        np.testing.assert_allclose(gg2.meanr, gg.meanr)
        np.testing.assert_allclose(gg2.meanlogr, gg.meanlogr)
        np.testing.assert_allclose(gg2.xip, gg.xip)
        np.testing.assert_allclose(gg2.xip_im, gg.xip_im)
        np.testing.assert_allclose(gg2.xim, gg.xim)
        np.testing.assert_allclose(gg2.xim_im, gg.xim_im)

    ascii_name = 'output/gg_twod.txt'
    gg.write(ascii_name, precision=16)
    gg3 = treecorr.GGCorrelation.from_file(ascii_name)
    np.testing.assert_allclose(gg3.npairs, gg.npairs)
    np.testing.assert_allclose(gg3.weight, gg.weight)
    np.testing.assert_allclose(gg3.meanr, gg.meanr)
    np.testing.assert_allclose(gg3.meanlogr, gg.meanlogr)
    np.testing.assert_allclose(gg3.xip, gg.xip)
    np.testing.assert_allclose(gg3.xip_im, gg.xip_im)
    np.testing.assert_allclose(gg3.xim, gg.xim)
    np.testing.assert_allclose(gg3.xim_im, gg.xim_im)

    # The other two, NG and KG can't really be checked with the brute force
    # calculator we have here, so we're counting on the above being a sufficient
    # test of all aspects of the twod binning.  I think that it is sufficient, but I
    # admit I would prefer if we had a real test of these other two pairs, along with
    # xi- for GG.

    # Check invalid constructors
    assert_raises(TypeError, treecorr.NNCorrelation, max_sep=max_sep, nbins=nbins, bin_size=2,
                  bin_type='TwoD')
    assert_raises(TypeError, treecorr.NNCorrelation, nbins=nbins, bin_type='TwoD')
    assert_raises(TypeError, treecorr.NNCorrelation, bin_size=2, bin_type='TwoD')
    assert_raises(TypeError, treecorr.NNCorrelation, max_sep=max_sep, bin_type='TwoD')


@timer
def test_twod_singlebin():

    # Test the singleBin function for TwoD bintype.

    # N random points in 2 dimensions
    rng = np.random.RandomState(8675309)
    N = 5000
    x = rng.uniform(-20, 20, N)
    y = rng.uniform(-20, 20, N)
    g1 = rng.uniform(-0.2, 0.2, N)
    g2 = rng.uniform(-0.2, 0.2, N)
    cat = treecorr.Catalog(x=x, y=y, g1=g1, g2=g2, k=10.*np.ones_like(x))

    max_sep = 21.
    nbins = 5  # Use very chunky bins, so more pairs of non-leaf cells can fall in single bin.

    # First use brute force for reference
    gg0 = treecorr.GGCorrelation(max_sep=max_sep, nbins=nbins, bin_type='TwoD', brute=True)
    t0 = time.time()
    gg0.process(cat)
    t1 = time.time()
    print('t for bs=0 = ',t1-t0)

    # Check top/bottom edges
    print('left = ',gg0.left_edges)
    print('right = ',gg0.right_edges)
    print('bottom = ',gg0.bottom_edges)
    print('top = ',gg0.top_edges)
    print('left = ',gg0.left_edges)
    np.testing.assert_allclose(gg0.left_edges[:,0], -max_sep)
    np.testing.assert_allclose(gg0.right_edges[:,4], max_sep)
    np.testing.assert_allclose(gg0.bottom_edges[0,:], -max_sep)
    np.testing.assert_allclose(gg0.top_edges[4,:], max_sep)
    np.testing.assert_allclose(gg0.left_edges[:,1:], gg0.right_edges[:,:-1])
    np.testing.assert_allclose(gg0.bottom_edges[1:,:], gg0.top_edges[:-1,:])

    # Now do bin_slop = 0
    gg1 = treecorr.GGCorrelation(max_sep=max_sep, nbins=nbins, bin_type='TwoD', bin_slop=0)
    t0 = time.time()
    gg1.process(cat)
    t1 = time.time()
    print('t for bs=1.e-10 = ',t1-t0)
    print('max abs diff xip = ',np.max(np.abs(gg1.xip - gg0.xip)))
    print('max abs diff xim = ',np.max(np.abs(gg1.xim - gg0.xim)))
    np.testing.assert_array_equal(gg1.npairs, gg0.npairs)
    np.testing.assert_allclose(gg1.xip, gg0.xip, atol=1.e-10)
    np.testing.assert_allclose(gg1.xim, gg0.xim, atol=1.e-5)

    # Now do bin_slop = 0.1
    gg2 = treecorr.GGCorrelation(max_sep=max_sep, nbins=nbins, bin_type='TwoD', bin_slop=0.1,
                                 max_top=4)
    t0 = time.time()
    gg2.process(cat)
    t1 = time.time()
    print('t for bs=0.1 = ',t1-t0)
    print('max abs diff npairs = ',np.max(np.abs(gg2.npairs - gg0.npairs)))
    print('max rel diff npairs = ',np.max(np.abs(gg2.npairs - gg0.npairs)/np.abs(gg0.npairs)))
    print('max abs diff xip = ',np.max(np.abs(gg2.xip - gg0.xip)))
    print('max abs diff xim = ',np.max(np.abs(gg2.xim - gg0.xim)))
    np.testing.assert_allclose(gg2.npairs, gg0.npairs, rtol=3.e-3)
    np.testing.assert_allclose(gg2.xip, gg0.xip, atol=1.e-4)
    np.testing.assert_allclose(gg2.xim, gg0.xim, atol=1.e-4)

    # Repeat with NG and KG so we can test those routines too.
    ng0 = treecorr.NGCorrelation(max_sep=max_sep, nbins=nbins, bin_type='TwoD', brute=True)
    t0 = time.time()
    ng0.process(cat, cat)
    t1 = time.time()
    print('t for bs=0 = ',t1-t0)

    ng1 = treecorr.NGCorrelation(max_sep=max_sep, nbins=nbins, bin_type='TwoD', bin_slop=0)
    t0 = time.time()
    ng1.process(cat, cat)
    t1 = time.time()
    print('t for bs=1.e-10 = ',t1-t0)
    print('max abs diff ng.xi = ',np.max(np.abs(ng1.xi - ng0.xi)))
    np.testing.assert_array_equal(ng1.npairs, ng0.npairs)
    np.testing.assert_allclose(ng1.xi, ng0.xi, atol=2.e-4)

    ng2 = treecorr.NGCorrelation(max_sep=max_sep, nbins=nbins, bin_type='TwoD', bin_slop=0.1)
    t0 = time.time()
    ng2.process(cat, cat)
    t1 = time.time()
    print('t for bs=0.1 = ',t1-t0)
    print('max abs diff npairs = ',np.max(np.abs(ng2.npairs - ng0.npairs)))
    print('max rel diff npairs = ',np.max(np.abs(ng2.npairs - ng0.npairs)/np.abs(ng0.npairs)))
    print('max abs diff ng.xi = ',np.max(np.abs(ng2.xi - ng0.xi)))
    np.testing.assert_allclose(ng2.npairs, ng0.npairs, rtol=1.e-2)
    np.testing.assert_allclose(ng2.xi, ng0.xi, atol=5.e-4)

    kg1 = treecorr.KGCorrelation(max_sep=max_sep, nbins=nbins, bin_type='TwoD', bin_slop=0)
    t0 = time.time()
    kg1.process(cat, cat)
    t1 = time.time()
    print('t for bs=1.e-10 = ',t1-t0)
    print('ng0.xi = ',ng0.xi)
    print('ng1.xi = ',ng1.xi)
    print('kg1.xi = ',kg1.xi)
    print('max abs diff kg.xi = ',np.max(np.abs(kg1.xi - 10.*ng0.xi)))
    np.testing.assert_array_equal(kg1.npairs, ng0.npairs)
    np.testing.assert_allclose(kg1.xi, 10.*ng0.xi, atol=2.e-3)
    np.testing.assert_array_equal(kg1.npairs, ng1.npairs)
    np.testing.assert_allclose(kg1.xi, 10.*ng1.xi, atol=1.e-10)

    kg2 = treecorr.KGCorrelation(max_sep=max_sep, nbins=nbins, bin_type='TwoD', bin_slop=0.1)
    t0 = time.time()
    kg2.process(cat, cat)
    t1 = time.time()
    print('t for bs=0.1 = ',t1-t0)
    print('max abs diff npairs = ',np.max(np.abs(kg2.npairs - ng0.npairs)))
    print('max rel diff npairs = ',np.max(np.abs(kg2.npairs - ng0.npairs)/np.abs(ng0.npairs)))
    print('max abs diff kg.xi = ',np.max(np.abs(kg2.xi - 10.*ng0.xi)))
    np.testing.assert_allclose(kg2.npairs, ng0.npairs, rtol=1.e-2)
    np.testing.assert_allclose(kg2.xi, 10.*ng0.xi, atol=5.e-3)
    np.testing.assert_array_equal(kg2.npairs, ng2.npairs)
    np.testing.assert_allclose(kg2.xi, 10.*ng2.xi, atol=1.e-10)


def test_twod_equiv():
    # This test is based on a script from Daniel Charles that wasn't working correctly.
    # Mostly it checks that the TwoD binning gets the same answer as 1-d when the profile
    # really is radial.  It also checks that the weights are relatively uniform, which they
    # weren't originally due to a bug in the binslop calculations for TwoD.
    try:
        from scipy.fft import fftn, ifftn
    except ImportError:
        print('Skipping test_twod, since uses scipy, and scipy is not installed.')
        return

    #Define grid
    N = 1000
    pixel_size = 5e-5
    edges = np.linspace(-N/2*pixel_size,N/2*pixel_size,N+1)
    pixel_nominal_value = 0.5*(edges[1:] + edges[:-1])
    xy_grid = np.meshgrid(pixel_nominal_value, pixel_nominal_value)

    #grid in fourier space
    n_xy = len(pixel_nominal_value)
    k_xy = 2*np.pi*np.fft.fftfreq(n_xy)/pixel_size
    identity = np.ones_like(k_xy)
    KX2 = np.einsum('i,j', k_xy**2, identity)
    KY2 = np.einsum('i,j', identity, k_xy**2)
    grid_k = np.sqrt(KX2 + KY2)

    #Input correlation function
    def corr(x,y):
        c = np.exp(-1/2*(x/50e-4)**2)*np.exp(-1/2*(y/50e-4)**2)
        return(c)
    c = corr(xy_grid[0],xy_grid[1])

    #Power spectrum
    p = fftn(c)

    # flip phases of fft to account for origin in center
    tmp = np.ones(p.shape[0])
    tmp[1::2]=-1
    p *= tmp
    p *= tmp[:,np.newaxis]
    #print(p)

    #SAMPLE
    np.random.seed(12345)
    sample_k = np.random.normal(0,1,size=(N,N))
    sample_with_pk = sample_k*np.sqrt(p)
    rs = np.fft.ifftn(sample_with_pk, norm='ortho')
    real_space_map = np.real(rs)

    #Pick random positions to measure
    npoints = 50000
    x_indices = np.random.randint(0,N,size=npoints)
    y_indices = np.random.randint(0,N,size=npoints)
    scalar_field_values = real_space_map[x_indices,y_indices]
    x = pixel_nominal_value[x_indices]
    y = pixel_nominal_value[y_indices]

    #treecorr measurement
    cat = treecorr.Catalog(x=x, y=y, k=scalar_field_values)
    rr = treecorr.KKCorrelation(bin_type='TwoD', bin_size=5e-5, nbins=201)
    rr.process(cat)

    rr2 = treecorr.KKCorrelation(bin_type='Linear', bin_size=5e-5, nbins=100, min_sep=2.5e-5)
    rr2.process(cat)

    print(np.shape(rr.xi))
    print('rr.xi = ',rr.xi[100][100:])
    print('rr2.xi = ',rr2.xi)
    print('rr2.w = ',rr2.weight)
    print('rr.w = ',rr.weight[100][100:])

    # The TwoD version should match the Linear one along any of the spines.
    np.testing.assert_allclose(rr.xi[100,101:], rr2.xi, atol=0.05)
    np.testing.assert_allclose(rr.xi[100,99::-1], rr2.xi, atol=0.05)
    np.testing.assert_allclose(rr.xi[101:,100], rr2.xi, atol=0.05)
    np.testing.assert_allclose(rr.xi[99::-1,100], rr2.xi, atol=0.05)

    # The weights don't match of course.
    # But for TwoD, all the weights should be fairly similar, since they cover almost the same
    # amount of phase space in how 2 points can be arranged.
    # The exception is the very center, where there are no pairs (since in this case bin_size
    # and the original pixel_size are equal).
    # Note: this didn't used to be true before fixing how bin_slop is handled for TwoD binning.
    assert rr.weight[100,100] == 0
    nz_weight = rr.weight[rr.weight > 0]
    np.testing.assert_allclose(nz_weight, np.mean(nz_weight), rtol=0.2)

    # Finally, if someone tries to run TwoD on a catalog with spherical or 3d coordinates, it
    # should raise an error.  (This was Daniel's original problem, which was super confusing
    # to diagnose, since TreeCorr just blithely gave wrong answers for this before.)
    cat_sph = treecorr.Catalog(ra=x, dec=y, k=scalar_field_values, ra_units='rad', dec_units='rad')
    with assert_raises(ValueError):
        rr.process(cat_sph)

if __name__ == '__main__':
    test_twod()
    test_twod_singlebin()
    test_twod_equiv()
