# Copyright (c) 2003-2019 by Mike Jarvis
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

from test_helper import assert_raises, timer

@timer
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
            dx = min(abs(x1[i]-x2[j]), Lx - abs(x1[i]-x2[j]))
            dy = min(abs(y1[i]-y2[j]), Ly - abs(y1[i]-y2[j]))
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
    np.testing.assert_allclose(corr2_output['r_nom'], dd.rnom, rtol=1.e-3)
    np.testing.assert_allclose(corr2_output['DD'], dd.npairs, rtol=1.e-3)
    np.testing.assert_allclose(corr2_output['npairs'], dd.npairs, rtol=1.e-3)
    np.testing.assert_allclose(corr2_output['RR'], rr.npairs, rtol=1.e-3)
    np.testing.assert_allclose(corr2_output['xi'], xi, rtol=1.e-3)

    # If don't give a period, then an error.
    rr = treecorr.NNCorrelation(min_sep=min_sep, max_sep=max_sep, nbins=nbins)
    with assert_raises(ValueError):
        rr.process(rcat1,rcat2, metric='Periodic')

    # Or if only give one kind of period
    rr = treecorr.NNCorrelation(min_sep=min_sep, max_sep=max_sep, nbins=nbins, xperiod=3)
    with assert_raises(ValueError):
        rr.process(rcat1,rcat2, metric='Periodic')
    rr = treecorr.NNCorrelation(min_sep=min_sep, max_sep=max_sep, nbins=nbins, yperiod=3)
    with assert_raises(ValueError):
        rr.process(rcat1,rcat2, metric='Periodic')

    # If give period, but then don't use Periodic metric, that's also an error.
    rr = treecorr.NNCorrelation(min_sep=min_sep, max_sep=max_sep, nbins=nbins, period=3)
    with assert_raises(ValueError):
        rr.process(rcat1,rcat2)

@timer
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
            dx = min(abs(x1[i]-x2[j]), Lx - abs(x1[i]-x2[j]))
            dy = min(abs(y1[i]-y2[j]), Ly - abs(y1[i]-y2[j]))
            dz = min(abs(z1[i]-z2[j]), Lz - abs(z1[i]-z2[j]))
            rsq = dx**2 + dy**2 + dz**2
            logr = 0.5 * np.log(rsq)
            k = int(np.floor( (logr-log_min_sep) / bin_size ))
            if k < 0: continue
            if k >= nbins: continue
            true_npairs[k] += 1

    print('true_npairs = ',true_npairs)
    print('diff = ',dd.npairs - true_npairs)
    np.testing.assert_array_equal(dd.npairs, true_npairs)


@timer
def test_periodic_ps():
    # Make a kappa field with a known power spectrum on a periodic grid.
    # This is heavily based on the GalSim PowerSpectrumRealizer class.

    ngrid = 256
    L = 100.   # grid size
    r0 = 10.  # scale size of Gaussian in real space
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


@timer
def test_halotools():
    try:
        import halotools  # noqa: F401
        from astropy.utils.exceptions import AstropyWarning

        # Note: halotools as of version 0.6 use astropy.extern.six, which is deprecated.
        # Ignore the warning that is emitted about this.  And in later astropy versions, it
        # now raises a ModuleNotFoundError.  So put it inside this try block.
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', category=AstropyWarning)
            from halotools.mock_observables import npairs_3d
    except ImportError:
        print('Skipping test_halotools, since either halotools or astropy is not installed.')
        return


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
    result = npairs_3d(sample1, sample2, rbins, period=period)
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


def wrap(x1, x2, xp, x3):
    if x2 > x1:
        if (x1+xp - x2 < x2 - x1):
            if abs(x2-xp-x3) > abs(x1+xp-x3):
                return x1+xp, x2
            else:
                return x1, x2-xp
        else: return x1, x2
    else:
        if (x2+xp - x1 < x1 - x2):
            if abs(x2+xp-x3) > abs(x1-xp-x3):
                return x1-xp, x2
            else:
                return x1, x2+xp
        else: return x1, x2

@timer
def test_3pt():
    # Test a direct calculation of the 3pt function with the Periodic metric.

    from test_nnn import is_ccw

    ngal = 50
    Lx = 250.
    Ly = 180.
    rng = np.random.RandomState(8675309)
    x = (rng.random_sample(ngal)-0.5) * Lx
    y = (rng.random_sample(ngal)-0.5) * Ly
    cat = treecorr.Catalog(x=x, y=y)

    min_sep = 1.
    max_sep = 40.  # This only really makes sense if max_sep < L/4 for all L.
    nbins = 50
    min_u = 0.13
    max_u = 0.89
    nubins = 10
    min_v = 0.13
    max_v = 0.59
    nvbins = 10

    ddd = treecorr.NNNCorrelation(min_sep=min_sep, max_sep=max_sep, nbins=nbins,
                                  min_u=min_u, max_u=max_u, nubins=nubins,
                                  min_v=min_v, max_v=max_v, nvbins=nvbins,
                                  bin_slop=0, xperiod=Lx, yperiod=Ly, brute=True)
    ddd.process(cat, metric='Periodic', num_threads=1)
    #print('ddd.ntri = ',ddd.ntri)

    log_min_sep = np.log(min_sep)
    log_max_sep = np.log(max_sep)
    true_ntri = np.zeros( (nbins, nubins, 2*nvbins) )
    bin_size = (log_max_sep - log_min_sep) / nbins
    ubin_size = (max_u-min_u) / nubins
    vbin_size = (max_v-min_v) / nvbins
    for i in range(ngal):
        for j in range(i+1,ngal):
            for k in range(j+1,ngal):
                xi = x[i]
                xj = x[j]
                xk = x[k]
                yi = y[i]
                yj = y[j]
                yk = y[k]
                #print(i,j,k,xi,yi,xj,yj,xk,yk)
                xi,xj = wrap(xi, xj, Lx, xk)
                #print('  ',xi,xj,xk)
                xi,xk = wrap(xi, xk, Lx, xj)
                #print('  ',xi,xj,xk)
                xj,xk = wrap(xj, xk, Lx, xi)
                #print('  ',xi,xj,xk)
                yi,yj = wrap(yi, yj, Ly, yk)
                #print('  ',yi,yj,yk)
                yi,yk = wrap(yi, yk, Ly, yj)
                #print('  ',yi,yj,yk)
                yj,yk = wrap(yj, yk, Ly, yi)
                #print('  ',yi,yj,yk)
                #print('->',xi,yi,xj,yj,xk,yk)
                dij = np.sqrt((xi-xj)**2 + (yi-yj)**2)
                dik = np.sqrt((xi-xk)**2 + (yi-yk)**2)
                djk = np.sqrt((xj-xk)**2 + (yj-yk)**2)
                if dij == 0.: continue
                if dik == 0.: continue
                if djk == 0.: continue
                ccw = True
                if dij < dik:
                    if dik < djk:
                        d3 = dij; d2 = dik; d1 = djk
                        ccw = is_ccw(xi,yi,xj,yj,xk,yk)
                    elif dij < djk:
                        d3 = dij; d2 = djk; d1 = dik
                        ccw = is_ccw(xj,yj,xi,yi,xk,yk)
                    else:
                        d3 = djk; d2 = dij; d1 = dik
                        ccw = is_ccw(xj,yj,xk,yk,xi,yi)
                else:
                    if dij < djk:
                        d3 = dik; d2 = dij; d1 = djk
                        ccw = is_ccw(xi,yi,xk,yk,xj,yj)
                    elif dik < djk:
                        d3 = dik; d2 = djk; d1 = dij
                        ccw = is_ccw(xk,yk,xi,yi,xj,yj)
                    else:
                        d3 = djk; d2 = dik; d1 = dij
                        ccw = is_ccw(xk,yk,xj,yj,xi,yi)

                #print('d1,d2,d3 = ',d1,d2,d3)
                r = d2
                u = d3/d2
                v = (d1-d2)/d3
                if r < min_sep or r >= max_sep: continue
                if u < min_u or u >= max_u: continue
                if v < min_v or v >= max_v: continue
                if not ccw:
                    v = -v
                #print('r,u,v = ',r,u,v)
                kr = int(np.floor( (np.log(r)-log_min_sep) / bin_size ))
                ku = int(np.floor( (u-min_u) / ubin_size ))
                if v > 0:
                    kv = int(np.floor( (v-min_v) / vbin_size )) + nvbins
                else:
                    kv = int(np.floor( (v-(-max_v)) / vbin_size ))
                #print('kr,ku,kv = ',kr,ku,kv)
                assert 0 <= kr < nbins
                assert 0 <= ku < nubins
                assert 0 <= kv < 2*nvbins
                true_ntri[kr,ku,kv] += 1
                #print('good.', true_ntri[kr,ku,kv])

    np.testing.assert_array_equal(ddd.ntri, true_ntri)

    # If don't give a period, then an error.
    ddd = treecorr.NNNCorrelation(min_sep=min_sep, max_sep=max_sep, nbins=nbins,
                                  min_u=min_u, max_u=max_u, nubins=nubins,
                                  min_v=min_v, max_v=max_v, nvbins=nvbins)
    with assert_raises(ValueError):
        ddd.process(cat, metric='Periodic')

    # Or if only give one kind of period
    ddd = treecorr.NNNCorrelation(min_sep=min_sep, max_sep=max_sep, nbins=nbins,
                                  min_u=min_u, max_u=max_u, nubins=nubins,
                                  min_v=min_v, max_v=max_v, nvbins=nvbins,
                                  xperiod=3)
    with assert_raises(ValueError):
        ddd.process(cat, metric='Periodic')
    ddd = treecorr.NNNCorrelation(min_sep=min_sep, max_sep=max_sep, nbins=nbins,
                                  min_u=min_u, max_u=max_u, nubins=nubins,
                                  min_v=min_v, max_v=max_v, nvbins=nvbins,
                                  yperiod=3)
    with assert_raises(ValueError):
        ddd.process(cat, metric='Periodic')

    # If give period, but then don't use Periodic metric, that's also an error.
    ddd = treecorr.NNNCorrelation(min_sep=min_sep, max_sep=max_sep, nbins=nbins,
                                  min_u=min_u, max_u=max_u, nubins=nubins,
                                  min_v=min_v, max_v=max_v, nvbins=nvbins,
                                  period=3)
    with assert_raises(ValueError):
        ddd.process(cat)



if __name__ == '__main__':
    test_direct_count()
    test_direct_3d()
    test_periodic_ps()
    test_halotools()
    test_3pt()
