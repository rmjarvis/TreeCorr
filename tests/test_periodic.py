# Copyright (c) 2003-2015 by Mike Jarvis
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

from __future__ import print_function
import numpy as np
import time
import os
import treecorr

from test_helper import assert_raises


def test_direct_count():
    # This is essentially the same as test_nn.py:test_direct_count, but using periodic distances.
    # And the points are uniform in the box, so plenty of pairs crossing the edges.

    ngal = 100
    Lx = 50.
    Ly = 80.
    rng = np.random.RandomState(8675309)
    x1 = (rng.random_sample(ngal)-0.5) * Lx
    y1 = (rng.random_sample(ngal)-0.5) * Ly
    cat1 = treecorr.Catalog(x=x1, y=y1)
    x2 = (rng.random_sample(ngal)-0.5) * Lx
    y2 = (rng.random_sample(ngal)-0.5) * Ly
    cat2 = treecorr.Catalog(x=x2, y=y2)

    min_sep = 1.
    max_sep = 50.
    nbins = 50
    dd = treecorr.NNCorrelation(min_sep=min_sep, max_sep=max_sep, nbins=nbins, bin_slop=0,
                                xperiod=Lx, yperiod=Ly)
    dd.process(cat1, cat2, metric='Periodic')
    print('dd.npairs = ',dd.npairs)

    log_min_sep = np.log(min_sep)
    log_max_sep = np.log(max_sep)
    true_npairs = np.zeros(nbins)
    bin_size = (log_max_sep - log_min_sep) / nbins
    for i in range(ngal):
        for j in range(ngal):
            dx = (x1[i]-x2[j]+Lx/2) % Lx - Lx/2
            dy = (y1[i]-y2[j]+Ly/2) % Ly - Ly/2
            rsq = dx**2 + dy**2
            logr = 0.5 * np.log(rsq)
            k = int(np.floor( (logr-log_min_sep) / bin_size ))
            if k < 0: continue
            if k >= nbins: continue
            true_npairs[k] += 1

    print('true_npairs = ',true_npairs)
    print('diff = ',dd.npairs - true_npairs)
    np.testing.assert_array_equal(dd.npairs, true_npairs)

    # Check that running via the corr2 script works correctly.
    file_name1 = os.path.join('data','nn_periodic_data1.dat')
    with open(file_name1, 'w') as fid:
        for i in range(ngal):
            fid.write(('%.20f %.20f\n')%(x1[i],y1[i]))
    file_name2 = os.path.join('data','nn_periodic_data2.dat')
    with open(file_name2, 'w') as fid:
        for i in range(ngal):
            fid.write(('%.20f %.20f\n')%(x2[i],y2[i]))
    nrand = ngal
    rx1 = (rng.random_sample(nrand)-0.5) * Lx
    ry1 = (rng.random_sample(nrand)-0.5) * Ly
    rx2 = (rng.random_sample(nrand)-0.5) * Lx
    ry2 = (rng.random_sample(nrand)-0.5) * Ly
    rcat1 = treecorr.Catalog(x=rx1, y=ry1)
    rcat2 = treecorr.Catalog(x=rx2, y=ry2)
    rand_file_name1 = os.path.join('data','nn_periodic_rand1.dat')
    with open(rand_file_name1, 'w') as fid:
        for i in range(nrand):
            fid.write(('%.20f %.20f\n')%(rx1[i],ry1[i]))
    rand_file_name2 = os.path.join('data','nn_periodic_rand2.dat')
    with open(rand_file_name2, 'w') as fid:
        for i in range(nrand):
            fid.write(('%.20f %.20f\n')%(rx2[i],ry2[i]))
    rr = treecorr.NNCorrelation(min_sep=min_sep, max_sep=max_sep, nbins=nbins, bin_slop=0,
                                verbose=0, xperiod=Lx, yperiod=Ly)
    rr.process(rcat1,rcat2, metric='Periodic')
    xi, varxi = dd.calculateXi(rr)
    print('xi = ',xi)

    # Do this via the corr2 function.
    config = treecorr.config.read_config('configs/nn_periodic.yaml')
    logger = treecorr.config.setup_logger(2)
    treecorr.corr2(config, logger)
    corr2_output = np.genfromtxt(os.path.join('output','nn_periodic.out'), names=True,
                                    skip_header=1)
    np.testing.assert_allclose(corr2_output['R_nom'], dd.rnom, rtol=1.e-3)
    np.testing.assert_allclose(corr2_output['DD'], dd.npairs, rtol=1.e-3)
    np.testing.assert_allclose(corr2_output['npairs'], dd.npairs, rtol=1.e-3)
    np.testing.assert_allclose(corr2_output['RR'], rr.npairs, rtol=1.e-3)
    np.testing.assert_allclose(corr2_output['xi'], xi, rtol=1.e-3)


def test_direct_3d():
    # This is the same as the above test, but using the 3d correlations

    ngal = 100
    Lx = 50.
    Ly = 80.
    Lz = 100.
    rng = np.random.RandomState(8675309)
    x1 = (rng.random_sample(ngal)-0.5) * Lx
    y1 = (rng.random_sample(ngal)-0.5) * Ly
    z1 = (rng.random_sample(ngal)-0.5) * Lz
    cat1 = treecorr.Catalog(x=x1, y=y1, z=z1)
    x2 = (rng.random_sample(ngal)-0.5) * Lx
    y2 = (rng.random_sample(ngal)-0.5) * Ly
    z2 = (rng.random_sample(ngal)-0.5) * Lz
    cat2 = treecorr.Catalog(x=x2, y=y2, z=z2)

    min_sep = 1.
    max_sep = 50.
    nbins = 50
    dd = treecorr.NNCorrelation(min_sep=min_sep, max_sep=max_sep, nbins=nbins, bin_slop=0,
                                xperiod=Lx, yperiod=Ly, zperiod=Lz)
    dd.process(cat1, cat2, metric='Periodic')
    print('dd.npairs = ',dd.npairs)

    log_min_sep = np.log(min_sep)
    log_max_sep = np.log(max_sep)
    true_npairs = np.zeros(nbins)
    bin_size = (log_max_sep - log_min_sep) / nbins
    for i in range(ngal):
        for j in range(ngal):
            dx = (x1[i]-x2[j]+Lx/2) % Lx - Lx/2
            dy = (y1[i]-y2[j]+Ly/2) % Ly - Ly/2
            dz = (z1[i]-z2[j]+Lz/2) % Lz - Lz/2
            rsq = dx**2 + dy**2 + dz**2
            logr = 0.5 * np.log(rsq)
            k = int(np.floor( (logr-log_min_sep) / bin_size ))
            if k < 0: continue
            if k >= nbins: continue
            true_npairs[k] += 1

    print('true_npairs = ',true_npairs)
    print('diff = ',dd.npairs - true_npairs)
    np.testing.assert_array_equal(dd.npairs, true_npairs)


def test_periodic_ps():
    # Make a kappa field with a known power spectrum on a periodic grid.
    # This is heavily based on the GalSim PowerSpectrumRealizer class.

    ngrid = 256
    L = 100   # grid size
    r0 = 10  # scale size of Gaussian in real space
    k0 = 1./ (2.*np.pi * r0)  # scale in Fourier space

    # Set up grid in k space
    kx_1d = np.fft.fftfreq(ngrid, L/ngrid)
    ky_1d = np.fft.fftfreq(ngrid, L/ngrid)
    kx, ky = np.meshgrid(kx_1d, ky_1d)

    # And real space
    x_1d = np.linspace(0, L, ngrid)
    y_1d = np.linspace(0, L, ngrid)
    x, y = np.meshgrid(x_1d, y_1d)

    # Use a Gaussian for the power spectrum, so the correlation function is also Gaussian.
    kz = kx + 1j * ky
    ksq = kz*np.conj(kz)
    pk = np.exp(-ksq/(2*k0**2))

    # Multiply by a Gaussian random field
    rng = np.random.RandomState(8675309)
    r1 = rng.normal(0,1,ngrid**2).reshape(pk.shape)
    r2 = rng.normal(0,1,ngrid**2).reshape(pk.shape)
    pk *= (r1 + 1j*r2) / np.sqrt(2.)
    pk[0,0] = 0.

    # pk is now the FT for the kappa field. 
    # The gamma field is just rotated by exp(2ipsi)
    ksq[0,0] = 1.  # Not zero, so no division by 0.
    exp2ipsi = kz*kz / ksq
    pg = pk * exp2ipsi

    # Take the inverse FFT to get the real-space fields
    kappa = ngrid * np.fft.ifft2(pk)
    gamma = ngrid * np.fft.ifft2(pg)

    # Now measure the correlation function with TreeCorr using periodic boundary conditions.
    cat = treecorr.Catalog(x=x.ravel(), y=y.ravel(), k=kappa.real.ravel(),
                           g1=gamma.real.ravel(), g2=gamma.imag.ravel())
    kk = treecorr.KKCorrelation(min_sep=3, max_sep=10, nbins=20, period=ngrid)
    kk.process(cat, metric='Periodic')
    print('kk.xi = ',kk.xi)

    # I admit that I didn't determine the amplitude of this from first principles.
    # I think it has to scale as k0**2 (or 1/r0**2), but the factors of 2 and pi confuse me.
    # The 1/pi here is empirical.
    true_xi = np.exp(-kk.meanr**2/(2*r0**2)) * k0**2/np.pi
    print('true_xi = ',true_xi)
    print('ratio = ',kk.xi/true_xi)
    np.testing.assert_allclose(kk.xi, true_xi, rtol=0.15)

    gg = treecorr.GGCorrelation(min_sep=3, max_sep=10, nbins=20, period=ngrid)
    gg.process(cat, metric='Periodic')
    print('gg.xip = ',gg.xip)
    print('gg.xim = ',gg.xim)

    # I thought these were supposed to be equal, but xip is larger by a factor of sqrt(2).
    # I'm sure this somehow makes sense because of the 2 complex components of the shear,
    # but I didn't try to figure it out.  I'm just going with it and assuming this is right.
    true_xip = true_xi * np.sqrt(2)
    print('true_xip = ',true_xip)
    print('ratio = ',gg.xip/true_xip)
    np.testing.assert_allclose(gg.xip, true_xip, rtol=0.15)


def test_halotools():
    try:
        import halotools
        from halotools.mock_observables import npairs_3d
    except ImportError:
        print('Skipping test_halotools, since halotools not installed.')

    # Compare the Periodic metric with the same calculation in halotools
    # This first bit is directly from the documentation for halotools.npairs_3d
    # https://halotools.readthedocs.io/en/latest/api/halotools.mock_observables.npairs_3d.html
    
    Npts1, Npts2, Lbox = 100000, 100000, 250.
    period = [Lbox, Lbox, Lbox]
    rbins = np.logspace(-1, 1.5, 15)
 
    rng = np.random.RandomState(8675309)
    x1 = rng.uniform(0, Lbox, Npts1)
    y1 = rng.uniform(0, Lbox, Npts1)
    z1 = rng.uniform(0, Lbox, Npts1)
    x2 = rng.uniform(0, Lbox, Npts2)
    y2 = rng.uniform(0, Lbox, Npts2)
    z2 = rng.uniform(0, Lbox, Npts2)

    sample1 = np.vstack([x1, y1, z1]).T
    sample2 = np.vstack([x2, y2, z2]).T

    t0 = time.time()
    result = npairs_3d(sample1, sample2, rbins, period = period)
    t1 = time.time()

    print('rbins = ',rbins)

    # Try to do the same thing with TreeCorr
    cat1 = treecorr.Catalog(x=x1, y=y1, z=z1)
    cat2 = treecorr.Catalog(x=x2, y=y2, z=z2)
    corr = treecorr.NNCorrelation(min_sep=rbins[0], max_sep=rbins[-1], nbins=len(rbins)-1,
                                  period=Lbox, bin_slop=0)
    t2 = time.time()
    corr.process(cat1, cat2, metric='Periodic')
    t3 = time.time()
    print('corr.rnom = ',corr.rnom)
    print('corr.left_edges = ',corr.left_edges)
    print('corr.right_edges = ',corr.right_edges)
    print('result = ',result)
    print('t = ',t1-t0)
    print('corr.npairs = ',corr.npairs.astype(int))
    print('t = ',t3-t2)
    print('cum(npairs) = ',np.cumsum(corr.npairs.astype(int)))

    np.testing.assert_allclose(corr.left_edges, rbins[:-1])
    np.testing.assert_allclose(corr.right_edges, rbins[1:])

    # Halotools counts pairs *closer* than r, not in the range between each r value.
    # So we compare it to np.cumsum(corr.npairs).  This only works if result[0] == 0.
    assert result[0] == 0
    np.testing.assert_array_equal(np.cumsum(corr.npairs).astype(int), result[1:])

    # TreeCorr is only a little faster than halotools for this, which makes sense, since
    # this is the exact calculation (bin_slop=0).  But with just a little slop, we get
    # almost the same result bute faster.  This matters even more for larger N.
    corr = treecorr.NNCorrelation(min_sep=rbins[0], max_sep=rbins[-1], nbins=len(rbins)-1,
                                  period=Lbox, bin_slop=0.2)
    t4 = time.time()
    corr.process(cat1, cat2, metric='Periodic')
    t5 = time.time()
    print('bs=0.2: cum(npairs) = ',np.cumsum(corr.npairs.astype(int)))
    print('t = ',t5-t4)
    np.testing.assert_allclose(np.cumsum(corr.npairs).astype(int), result[1:], rtol=0.01)


if __name__ == '__main__':
    test_direct_count()
    test_direct_3d()
    test_periodic_ps()
    test_halotools()
