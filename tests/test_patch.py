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
import os
import coord
import time
import treecorr

from test_helper import assert_raises, do_pickle, profile

def test_cat_patches():
    # Test the different ways to set patches in the catalog.

    # Use the same input as test_radec()
    ngal = 100000
    s = 10.
    rng = np.random.RandomState(8675309)
    x = rng.normal(0,s, (ngal,) )
    y = rng.normal(0,s, (ngal,) ) + 100  # Put everything at large y, so smallish angle on sky
    z = rng.normal(0,s, (ngal,) )
    ra, dec = coord.CelestialCoord.xyz_to_radec(x,y,z)

    # cat0 is the base catalog without patches
    cat0 = treecorr.Catalog(ra=ra, dec=dec, ra_units='rad', dec_units='rad')
    assert len(cat0.getPatches()) == 1
    assert cat0.getPatches()[0].ntot == ngal

    # 1. Make the patches automatically using kmeans
    #    Note: If npatch is a power of two, then the patch determination is completely
    #          deterministic, which is helpful for this test.
    cat1 = treecorr.Catalog(ra=ra, dec=dec, ra_units='rad', dec_units='rad', npatch=128)
    p2 = cat0.getNField().run_kmeans(128)
    np.testing.assert_array_equal(cat1.patch, p2)
    assert len(cat1.getPatches()) == 128
    assert np.sum([p.ntot for p in cat1.getPatches()]) == ngal

    # 2. Optionally can use alt algorithm
    cat2 = treecorr.Catalog(ra=ra, dec=dec, ra_units='rad', dec_units='rad', npatch=128,
                            kmeans_alt=True)
    p3 = cat0.getNField().run_kmeans(128, alt=True)
    np.testing.assert_array_equal(cat2.patch, p3)
    assert len(cat2.getPatches()) == 128

    # 3. Optionally can set different init method
    cat3 = treecorr.Catalog(ra=ra, dec=dec, ra_units='rad', dec_units='rad', npatch=128,
                            kmeans_init='kmeans++')
    # Can't test this equalling a repeat run from cat0, because kmpp has a random aspect to it.
    # But at least check that it isn't equal to the other two versions.
    assert not np.array_equal(cat3.patch, p2)
    assert not np.array_equal(cat3.patch, p3)
    cat3b = treecorr.Catalog(ra=ra, dec=dec, ra_units='rad', dec_units='rad', npatch=128,
                             kmeans_init='random')
    assert not np.array_equal(cat3b.patch, p2)
    assert not np.array_equal(cat3b.patch, p3)
    assert not np.array_equal(cat3b.patch, cat3.patch)

    # 4. Pass in patch array explicitly
    cat4 = treecorr.Catalog(ra=ra, dec=dec, ra_units='rad', dec_units='rad', patch=p2)
    np.testing.assert_array_equal(cat4.patch, p2)

    # 5. Read patch from a column in ASCII file
    file_name5 = os.path.join('output','test_cat_patches.dat')
    cat4.write(file_name5)
    cat5 = treecorr.Catalog(file_name5, ra_col=1, dec_col=2, ra_units='rad', dec_units='rad',
                            patch_col=3)
    np.testing.assert_array_equal(cat5.patch, p2)

    # 6. Read patch from a column in FITS file
    try:
        import fitsio
    except ImportError:
        print('Skip fitsio tests of patch_col')
    else:
        file_name6 = os.path.join('output','test_cat_patches.fits')
        cat4.write(file_name6)
        cat6 = treecorr.Catalog(file_name6, ra_col='ra', dec_col='dec',
                                ra_units='rad', dec_units='rad', patch_col='patch')
        np.testing.assert_array_equal(cat6.patch, p2)
        cat6b = treecorr.Catalog(file_name6, ra_col='ra', dec_col='dec',
                                 ra_units='rad', dec_units='rad', patch_col='patch', patch_hdu=1)
        np.testing.assert_array_equal(cat6b.patch, p2)
        assert len(cat6.getPatches()) == 128
        assert len(cat6b.getPatches()) == 128

    # Check serialization with patch
    do_pickle(cat2)

    # Check some invalid parameters
    with assert_raises(ValueError):
        treecorr.Catalog(ra=ra, dec=dec, ra_units='rad', dec_units='rad', npatch=128, patch=p2)
    with assert_raises(ValueError):
        treecorr.Catalog(ra=ra, dec=dec, ra_units='rad', dec_units='rad', patch=p2[:1000])
    with assert_raises(ValueError):
        treecorr.Catalog(ra=ra, dec=dec, ra_units='rad', dec_units='rad', npatch=0)
    with assert_raises(ValueError):
        treecorr.Catalog(ra=ra, dec=dec, ra_units='rad', dec_units='rad', npatch=128,
                         kmeans_init='invalid')
    with assert_raises(ValueError):
        treecorr.Catalog(ra=ra, dec=dec, ra_units='rad', dec_units='rad', npatch=128,
                         kmeans_alt='maybe')
    with assert_raises(ValueError):
        treecorr.Catalog(file_name5, ra_col=1, dec_col=2, ra_units='rad', dec_units='rad',
                         patch_col='invalid')
    with assert_raises(ValueError):
        treecorr.Catalog(file_name5, ra_col=1, dec_col=2, ra_units='rad', dec_units='rad',
                         patch_col=4)
    with assert_raises(IOError):
        treecorr.Catalog(file_name6, ra_col='ra', dec_col='dec', ra_units='rad', dec_units='rad',
                         patch_col='patch', patch_hdu=2)
    with assert_raises(ValueError):
        treecorr.Catalog(file_name6, ra_col='ra', dec_col='dec', ra_units='rad', dec_units='rad',
                         patch_col='patches')

def generate_shear_field(nside):
   # Generate a random shear field with a well-defined power spectrum.
   # It generates shears on a grid nside x nside, and returns, x, y, g1, g2
   kvals = np.fft.fftfreq(nside) * 2*np.pi
   kx,ky = np.meshgrid(kvals,kvals)
   k = kx + 1j*ky
   ksq = kx**2 + ky**2

   # Use a power spectrum with lots of large scale power.
   # The rms shape ends up around 0.2 and min/max are around +-1.
   # Having a lot more large-scale than small-scale power means that sample variance is
   # very important, so the shot noise estimate of the variance is particularly bad.
   Pk = 1.e4 * ksq / (1. + 300.*ksq)**2

   # Make complex gaussian field in k-space.
   f1 = np.random.normal(size=Pk.shape)
   f2 = np.random.normal(size=Pk.shape)
   f = (f1 + 1j*f2) * np.sqrt(0.5)

   # Make f Hermitian, to correspond to E-mode-only field.
   # Hermitian means f(-k) = conj(f(k)).
   # Note: this is approximate.  It doesn't get all the k=0 and k=nside/2 correct.
   # But this is good enough for xi- to be not close to zero.
   ikxp = slice(1,(nside+1)//2)   # kx > 0
   ikxn = slice(-1,nside//2,-1)   # kx < 0
   ikyp = slice(1,(nside+1)//2)   # ky > 0
   ikyn = slice(-1,nside//2,-1)   # ky < 0
   f[ikyp,ikxn] = np.conj(f[ikyn,ikxp])
   f[ikyn,ikxn] = np.conj(f[ikyp,ikxp])

   # Multiply by the power spectrum to get a realization of a field with this P(k)
   f *= Pk

   # Inverse fft gives the real-space field.
   kappa = nside * np.fft.ifft2(f)

   # Multiply by exp(2iphi) to get gamma field, rather than kappa.
   ksq[0,0] = 1.  # Avoid division by zero
   exp2iphi = k**2 / ksq
   f *= exp2iphi
   gamma = nside * np.fft.ifft2(f)

   # Generate x,y values for the real-space field
   x,y = np.meshgrid(np.linspace(0.,1000.,nside), np.linspace(0.,1000.,nside))

   x = x.ravel()
   y = y.ravel()
   gamma = gamma.ravel()
   kappa = np.real(kappa.ravel())

   return x, y, np.real(gamma), np.imag(gamma), kappa


def test_gg_jk():
    # Test the variance estimate for GG correlation with jackknife error estimate.

    if __name__ == '__main__':
        # 1000 x 1000, so 10^6 points.  With jackknifing, that gives 10^4 per region.
        nside = 1000
        npatch = 64
        tol_factor = 1
    else:
        # Use ~1/10 of the objects when running unit tests
        nside = 300
        npatch = 64
        tol_factor = 4

    # The full simulation needs to run a lot of times to get a good estimate of the variance,
    # but this takes a long time.  So we store the results in the repo.
    # To redo the simulation, just delete the file data/test_gg_jk.fits
    file_name = 'data/test_gg_jk_{}.npz'.format(nside)
    print(file_name)
    if not os.path.isfile(file_name):
        nruns = 1000
        all_ggs = []
        for run in range(nruns):
            x, y, g1, g2, _ = generate_shear_field(nside)
            print(run,': ',np.mean(g1),np.std(g1),np.min(g1),np.max(g1))
            cat = treecorr.Catalog(x=x, y=y, g1=g1, g2=g2)
            gg = treecorr.GGCorrelation(bin_size=0.3, min_sep=10., max_sep=50.)
            gg.process(cat)
            all_ggs.append(gg)

        mean_xip = np.mean([gg.xip for gg in all_ggs], axis=0)
        var_xip = np.var([gg.xip for gg in all_ggs], axis=0)
        mean_xim = np.mean([gg.xim for gg in all_ggs], axis=0)
        var_xim = np.var([gg.xim for gg in all_ggs], axis=0)
        mean_varxip = np.mean([gg.varxip for gg in all_ggs], axis=0)
        mean_varxim = np.mean([gg.varxim for gg in all_ggs], axis=0)

        np.savez(file_name,
                 mean_xip=mean_xip, mean_xim=mean_xim,
                 var_xip=var_xip, var_xim=var_xim,
                 mean_varxip=mean_varxip, mean_varxim=mean_varxim)

    data = np.load(file_name)
    mean_xip = data['mean_xip']
    mean_xim = data['mean_xim']
    var_xip = data['var_xip']
    var_xim = data['var_xim']
    mean_varxip = data['mean_varxip']
    mean_varxim = data['mean_varxim']

    print('mean_xip = ',mean_xip)
    print('mean_xim = ',mean_xim)
    print('mean_varxip = ',mean_varxip)
    print('mean_varxim = ',mean_varxim)
    print('var_xip = ',var_xip)
    print('ratio = ',var_xip / mean_varxip)
    print('var_xim = ',var_xim)
    print('ratio = ',var_xim / mean_varxim)

    np.random.seed(1234)
    # First run with the normal variance estimate, which is too small.
    x, y, g1, g2, _ = generate_shear_field(nside)

    cat = treecorr.Catalog(x=x, y=y, g1=g1, g2=g2)
    gg1 = treecorr.GGCorrelation(bin_size=0.3, min_sep=10., max_sep=50.)
    t0 = time.time()
    gg1.process(cat)
    t1 = time.time()
    print('Time for non-patch processing = ',t1-t0)

    print('npairs = ',gg1.npairs)
    print('xip = ',gg1.xip)
    print('xim = ',gg1.xim)
    print('varxip = ',gg1.varxip)
    print('varxim = ',gg1.varxim)
    print('pullsq for xip = ',(gg1.xip-mean_xip)**2/var_xip)
    print('pullsq for xim = ',(gg1.xim-mean_xim)**2/var_xim)
    print('max pull for xip = ',np.sqrt(np.max((gg1.xip-mean_xip)**2/var_xip)))
    print('max pull for xim = ',np.sqrt(np.max((gg1.xim-mean_xim)**2/var_xim)))
    np.testing.assert_array_less((gg1.xip - mean_xip)**2/var_xip, 25) # within 5 sigma
    np.testing.assert_array_less((gg1.xim - mean_xim)**2/var_xim, 25)
    np.testing.assert_allclose(gg1.varxip, mean_varxip, rtol=0.03 * tol_factor)
    np.testing.assert_allclose(gg1.varxim, mean_varxim, rtol=0.03 * tol_factor)

    # The naive error estimates only includes shape noise, so it is an underestimate of
    # the full variance, which includes sample variance.
    np.testing.assert_array_less(mean_varxip, var_xip)
    np.testing.assert_array_less(mean_varxim, var_xim)
    np.testing.assert_array_less(gg1.varxip, var_xip)
    np.testing.assert_array_less(gg1.varxim, var_xim)

    # Now run with patches, but still with shot variance.  Should be basically the same answer.
    cat = treecorr.Catalog(x=x, y=y, g1=g1, g2=g2, npatch=npatch)
    gg2 = treecorr.GGCorrelation(bin_size=0.3, min_sep=10., max_sep=50., var_method='shot')
    t0 = time.time()
    gg2.process(cat)
    t1 = time.time()
    print('Time for shot processing = ',t1-t0)
    print('npairs = ',gg2.npairs)
    print('xip = ',gg2.xip)
    print('xim = ',gg2.xim)
    print('varxip = ',gg2.varxip)
    print('varxim = ',gg2.varxim)
    np.testing.assert_allclose(gg2.npairs, gg1.npairs, rtol=1.e-2*tol_factor)
    np.testing.assert_allclose(gg2.xip, gg1.xip, rtol=1.e-2*tol_factor)
    np.testing.assert_allclose(gg2.xim, gg1.xim, rtol=3.e-2*tol_factor)
    np.testing.assert_allclose(gg2.varxip, gg1.varxip, rtol=1.e-2*tol_factor)
    np.testing.assert_allclose(gg2.varxim, gg1.varxim, rtol=1.e-2*tol_factor)

    # Can get this as a (diagonal) covariance matrix using estimate_cov
    np.testing.assert_allclose(gg2.estimate_cov('xip','weight','shot'), np.diag(gg2.varxip))
    np.testing.assert_allclose(gg2.estimate_cov('xim','weight','shot'), np.diag(gg2.varxim))
    np.testing.assert_allclose(gg1.estimate_cov('xip','weight','shot'), np.diag(gg1.varxip))
    np.testing.assert_allclose(gg1.estimate_cov('xim','weight','shot'), np.diag(gg1.varxim))

    # Now run with jackknife variance estimate.  Should be much better.
    gg3 = treecorr.GGCorrelation(bin_size=0.3, min_sep=10., max_sep=50., var_method='jackknife')
    t0 = time.time()
    gg3.process(cat)
    t1 = time.time()
    print('Time for jackknife processing = ',t1-t0)
    print('xip = ',gg3.xip)
    print('xim = ',gg3.xim)
    print('varxip = ',gg3.varxip)
    print('ratio = ',gg3.varxip / var_xip)
    print('varxim = ',gg3.varxim)
    print('ratio = ',gg3.varxim / var_xim)
    np.testing.assert_allclose(gg3.npairs, gg2.npairs)
    np.testing.assert_allclose(gg3.xip, gg2.xip)
    np.testing.assert_allclose(gg3.xim, gg2.xim)
    # Not perfect, but within about 30%.
    np.testing.assert_allclose(gg3.varxip, var_xip, rtol=0.3*tol_factor)
    np.testing.assert_allclose(gg3.varxim, var_xim, rtol=0.3*tol_factor)

    # Can get the covariance matrix using estimate_cov, which is also stored as covxi? attributes
    t0 = time.time()
    np.testing.assert_allclose(gg3.estimate_cov('xip','weight','jackknife'), gg3.covxip)
    np.testing.assert_allclose(gg3.estimate_cov('xim','weight','jackknife'), gg3.covxim)
    t1 = time.time()
    print('Time to calculate jackknife covariance = ',t1-t0)

    # Can also get the shot covariance matrix using estimate_cov
    np.testing.assert_allclose(gg3.estimate_cov('xip','weight','shot'), np.diag(gg2.varxip))
    np.testing.assert_allclose(gg3.estimate_cov('xim','weight','shot'), np.diag(gg2.varxim))

    # And can even get the jackknife covariance from a run that used var_method='shot'
    np.testing.assert_allclose(gg2.estimate_cov('xip','weight','jackknife'), gg3.covxip)
    np.testing.assert_allclose(gg2.estimate_cov('xim','weight','jackknife'), gg3.covxim)

    # Use estimate_multi_cov to get combined xip, xim covariance
    t0 = time.time()
    cov_xipm = treecorr.estimate_multi_cov([gg2,gg2], ['xip','xim'], ['weight','weight'],
                                           'jackknife')
    t1 = time.time()
    print('Time for jackknife cross-covariance = ',t1-t0)
    n1 = len(gg2.xip)
    np.testing.assert_allclose(cov_xipm[:n1,:n1], gg3.covxip)
    np.testing.assert_allclose(cov_xipm[n1:,n1:], gg3.covxim)
    print('cross covariance = ',cov_xipm[:n1,n1:],np.sum(cov_xipm[n1:,n1:]**2))
    # Make cross correlation matrix
    c = cov_xipm[:n1,n1:]
    c /= np.sqrt(gg3.varxip)[:,np.newaxis]
    c /= np.sqrt(gg3.varxim)[np.newaxis,:]
    print('cross correlation = ',c)
    assert np.sum(c**2) > 1.e-2  # Should be significantly non-zero
    assert np.all(np.abs(c) < 1.)  # And all are between -1 and -1.

    # Check sample covariance estimate
    t0 = time.time()
    cov_xip = gg3.estimate_cov('xip','weight','sample')
    cov_xim = gg3.estimate_cov('xim','weight','sample')
    t1 = time.time()
    print('Time to calculate sample covariance = ',t1-t0)
    print('varxip = ',cov_xip.diagonal())
    print('ratio = ',cov_xip.diagonal() / var_xip)
    print('varxim = ',cov_xim.diagonal())
    print('ratio = ',cov_xim.diagonal() / var_xim)
    # It's not too bad ast small scales, but at larger scales the variance in the number of pairs
    # among the different samples gets bigger (since some are near the edge, and others not).
    # So this is only good to a little worse than a factor of 2.
    np.testing.assert_allclose(cov_xip.diagonal(), var_xip, rtol=0.6*tol_factor)
    np.testing.assert_allclose(cov_xim.diagonal(), var_xim, rtol=0.6*tol_factor)

    # Check some invalid actions
    with assert_raises(ValueError):
        gg2.estimate_cov('xip','weight','invalid')
    with assert_raises(ValueError):
        gg3.estimate_cov('xip','weight','invalid')
    with assert_raises(AttributeError):
        gg2.estimate_cov('invalid','weight','shot')
    with assert_raises(AttributeError):
        gg2.estimate_cov('xip','invalid','shot')
    with assert_raises(AttributeError):
        gg2.estimate_cov('invalid','weight','jackknife')
    with assert_raises(AttributeError):
        gg2.estimate_cov('xip','invalid','jackknife')
    with assert_raises(ValueError):
        gg1.estimate_cov('xip','weight','jackknife')
    with assert_raises(ValueError):
        treecorr.estimate_multi_cov([gg2, gg1],['xip','xim'], ['weight','weight'], 'jackknife')
    with assert_raises(ValueError):
        treecorr.estimate_multi_cov([gg1, gg2],['xip','xim'], ['weight','weight'], 'jackknife')
    with assert_raises(ValueError):
        treecorr.estimate_multi_cov([gg1, gg2],['xip','xim'], ['weight','weight'], 'sample')
    with assert_raises(ValueError):
        treecorr.estimate_multi_cov([gg2, gg1],['xip','xim'], ['weight','weight'], 'sample')


def test_ng_jk():
    # Test the variance estimate for NG correlation with jackknife error estimate.

    if __name__ == '__main__':
        # 1000 x 1000, so 10^6 points.  With jackknifing, that gives 10^4 per region.
        nside = 1000
        nlens = 50000
        npatch = 64
        tol_factor = 1
    else:
        # If much smaller, then there can be no lenses in some patches, so only 1/4 the galaxies
        # and use half the number of patches
        nside = 500
        nlens = 30000
        npatch = 32
        tol_factor = 3

    file_name = 'data/test_ng_jk_{}.npz'.format(nside)
    print(file_name)
    if not os.path.isfile(file_name):
        nruns = 1000
        all_ngs = []
        for run in range(nruns):
            x, y, g1, g2, k = generate_shear_field(nside)
            thresh = np.partition(k.flatten(), -nlens)[-nlens]
            w = np.zeros_like(k)
            w[k>=thresh] = 1.
            print(run,': ',np.mean(g1),np.std(g1),np.min(g1),np.max(g1),thresh)
            cat1 = treecorr.Catalog(x=x, y=y, w=w)
            cat2 = treecorr.Catalog(x=x, y=y, g1=g1, g2=g2)
            ng = treecorr.NGCorrelation(bin_size=0.3, min_sep=10., max_sep=50.)
            ng.process(cat1, cat2)
            all_ngs.append(ng)

        mean_xi = np.mean([ng.xi for ng in all_ngs], axis=0)
        var_xi = np.var([ng.xi for ng in all_ngs], axis=0)
        mean_varxi = np.mean([ng.varxi for ng in all_ngs], axis=0)

        np.savez(file_name,
                 mean_xi=mean_xi, var_xi=var_xi, mean_varxi=mean_varxi)

    data = np.load(file_name)
    mean_xi = data['mean_xi']
    var_xi = data['var_xi']
    mean_varxi = data['mean_varxi']

    print('mean_xi = ',mean_xi)
    print('mean_varxi = ',mean_varxi)
    print('var_xi = ',var_xi)
    print('ratio = ',var_xi / mean_varxi)

    np.random.seed(1234)
    # First run with the normal variance estimate, which is too small.
    x, y, g1, g2, k = generate_shear_field(nside)
    thresh = np.partition(k.flatten(), -nlens)[-nlens]
    w = np.zeros_like(k)
    w[k>=thresh] = 1.
    cat1 = treecorr.Catalog(x=x, y=y, w=w)
    cat2 = treecorr.Catalog(x=x, y=y, g1=g1, g2=g2)
    ng1 = treecorr.NGCorrelation(bin_size=0.3, min_sep=10., max_sep=50.)
    t0 = time.time()
    ng1.process(cat1, cat2)
    t1 = time.time()
    print('Time for non-patch processing = ',t1-t0)

    print('npairs = ',ng1.npairs)
    print('xi = ',ng1.xi)
    print('varxi = ',ng1.varxi)
    print('pullsq for xi = ',(ng1.xi-mean_xi)**2/var_xi)
    print('max pull for xi = ',np.sqrt(np.max((ng1.xi-mean_xi)**2/var_xi)))
    np.testing.assert_array_less((ng1.xi - mean_xi)**2/var_xi, 25) # within 5 sigma
    np.testing.assert_allclose(ng1.varxi, mean_varxi, rtol=0.03 * tol_factor)

    # The naive error estimates only includes shape noise, so it is an underestimate of
    # the full variance, which includes sample variance.
    np.testing.assert_array_less(mean_varxi, var_xi)
    np.testing.assert_array_less(ng1.varxi, var_xi)

    # Now run with patches, but still with shot variance.  Should be basically the same answer.
    cat2p = treecorr.Catalog(x=x, y=y, g1=g1, g2=g2, npatch=npatch)
    cat1p = treecorr.Catalog(x=x, y=y, w=w, patch=cat2p.patch)
    print('tot w = ',np.sum(w))
    print('Patch\tNlens')
    for i in range(npatch):
        print('%d\t%d'%(i,np.sum(w[cat2p.patch==i])))
    ng2 = treecorr.NGCorrelation(bin_size=0.3, min_sep=10., max_sep=50., var_method='shot')
    t0 = time.time()
    ng2.process(cat1p, cat2p)
    t1 = time.time()
    print('Time for shot processing = ',t1-t0)
    print('npairs = ',ng2.npairs)
    print('xi = ',ng2.xi)
    print('varxi = ',ng2.varxi)
    np.testing.assert_allclose(ng2.npairs, ng1.npairs, rtol=1.e-2*tol_factor)
    np.testing.assert_allclose(ng2.xi, ng1.xi, rtol=3.e-2*tol_factor)
    np.testing.assert_allclose(ng2.varxi, ng1.varxi, rtol=1.e-2*tol_factor)

    # Can get this as a (diagonal) covariance matrix using estimate_cov
    np.testing.assert_allclose(ng2.estimate_cov('xi','weight','shot'), np.diag(ng2.varxi))
    np.testing.assert_allclose(ng1.estimate_cov('xi','weight','shot'), np.diag(ng1.varxi))

    # Now run with jackknife variance estimate.  Should be much better.
    ng3 = treecorr.NGCorrelation(bin_size=0.3, min_sep=10., max_sep=50., var_method='jackknife')
    t0 = time.time()
    ng3.process(cat1p, cat2p)
    t1 = time.time()
    print('Time for jackknife processing = ',t1-t0)
    print('xi = ',ng3.xi)
    print('varxi = ',ng3.varxi)
    print('ratio = ',ng3.varxi / var_xi)
    np.testing.assert_allclose(ng3.npairs, ng2.npairs)
    np.testing.assert_allclose(ng3.xi, ng2.xi)
    np.testing.assert_allclose(ng3.varxi, var_xi, rtol=0.4*tol_factor)

    # Check using estimate_cov
    t0 = time.time()
    np.testing.assert_allclose(ng3.estimate_cov('xi','weight','jackknife'), ng3.covxi)
    t1 = time.time()
    print('Time to calculate jackknife covariance = ',t1-t0)
    np.testing.assert_allclose(ng3.estimate_cov('xi','weight','shot'), np.diag(ng2.varxi))
    np.testing.assert_allclose(ng2.estimate_cov('xi','weight','jackknife'), ng3.covxi)

    # Check only using patches for one of the two catalogs.
    # Not as good as using patches for both, but not much worse.
    ng4 = treecorr.NGCorrelation(bin_size=0.3, min_sep=10., max_sep=50., var_method='jackknife')
    t0 = time.time()
    ng4.process(cat1p, cat2)
    t1 = time.time()
    print('Time for only patches for cat1 processing = ',t1-t0)
    print('npairs = ',ng4.npairs)
    print('xi = ',ng4.xi)
    print('varxi = ',ng4.varxi)
    np.testing.assert_allclose(ng4.npairs, ng1.npairs, rtol=1.e-2*tol_factor)
    np.testing.assert_allclose(ng4.xi, ng1.xi, rtol=3.e-2*tol_factor)
    np.testing.assert_allclose(ng4.varxi, var_xi, rtol=0.5*tol_factor)

    ng5 = treecorr.NGCorrelation(bin_size=0.3, min_sep=10., max_sep=50., var_method='jackknife')
    t0 = time.time()
    ng5.process(cat1, cat2p)
    t1 = time.time()
    print('Time for only patches for cat2 processing = ',t1-t0)
    print('npairs = ',ng5.npairs)
    print('xi = ',ng5.xi)
    print('varxi = ',ng5.varxi)
    np.testing.assert_allclose(ng5.npairs, ng1.npairs, rtol=1.e-2*tol_factor)
    np.testing.assert_allclose(ng5.xi, ng1.xi, rtol=3.e-2*tol_factor)
    np.testing.assert_allclose(ng5.varxi, var_xi, rtol=0.6*tol_factor)

    # Check sample covariance estimate
    t0 = time.time()
    cov_xi = ng3.estimate_cov('xi','weight','sample')
    t1 = time.time()
    print('Time to calculate sample covariance = ',t1-t0)
    print('varxi = ',cov_xi.diagonal())
    print('ratio = ',cov_xi.diagonal() / var_xi)
    np.testing.assert_allclose(cov_xi.diagonal(), var_xi, rtol=0.7*tol_factor)

    cov_xi = ng4.estimate_cov('xi','weight','sample')
    print('varxi = ',cov_xi.diagonal())
    np.testing.assert_allclose(cov_xi.diagonal(), var_xi, rtol=0.7*tol_factor)

    cov_xi = ng5.estimate_cov('xi','weight','sample')
    print('varxi = ',cov_xi.diagonal())
    np.testing.assert_allclose(cov_xi.diagonal(), var_xi, rtol=0.7*tol_factor)

    # Check some invalid actions
    with assert_raises(ValueError):
        ng2.estimate_cov('xi','weight','invalid')
    with assert_raises(ValueError):
        ng3.estimate_cov('xi','weight','invalid')
    with assert_raises(AttributeError):
        ng2.estimate_cov('invalid','weight','shot')
    with assert_raises(AttributeError):
        ng2.estimate_cov('xi','invalid','shot')
    with assert_raises(AttributeError):
        ng2.estimate_cov('invalid','weight','jackknife')
    with assert_raises(AttributeError):
        ng2.estimate_cov('xi','invalid','jackknife')
    with assert_raises(ValueError):
        ng1.estimate_cov('xi','weight','jackknife')

    cat1a = treecorr.Catalog(x=x[:100], y=y[:100], npatch=10)
    cat2a = treecorr.Catalog(x=x[:100], y=y[:100], g1=g1[:100], g2=g2[:100], npatch=10)
    cat1b = treecorr.Catalog(x=x[:100], y=y[:100], npatch=2)
    cat2b = treecorr.Catalog(x=x[:100], y=y[:100], g1=g1[:100], g2=g2[:100], npatch=2)
    ng6 = treecorr.NGCorrelation(bin_size=0.3, min_sep=10., max_sep=50., var_method='jackknife')
    ng7 = treecorr.NGCorrelation(bin_size=0.3, min_sep=10., max_sep=50., var_method='jackknife')
    with assert_raises(RuntimeError):
        ng6.process(cat1a,cat2b)
    with assert_raises(RuntimeError):
        ng6.estimate_cov('xi','weight','sample')
    with assert_raises(RuntimeError):
        ng7.process(cat1b,cat2a)
    with assert_raises(RuntimeError):
        ng7.estimate_cov('xi','weight','sample')

def test_kappa_jk():
    # Test NK, KK, and KG with jackknife.
    # There's not really anything new to test here.  So just checking the interface works.

    if __name__ == '__main__':
        nside = 1000
        nlens = 50000
        npatch = 64
        tol_factor = 1
    else:
        nside = 500
        nlens = 30000
        npatch = 32
        tol_factor = 3

    file_name = 'data/test_kappa_jk_{}.npz'.format(nside)
    print(file_name)
    if not os.path.isfile(file_name):
        nruns = 1000
        all_nks = []
        all_kks = []
        all_kgs = []
        for run in range(nruns):
            x, y, g1, g2, k = generate_shear_field(nside)
            thresh = np.partition(k.flatten(), -nlens)[-nlens]
            w = np.zeros_like(k)
            w[k>=thresh] = 1.
            print(run,': ',np.mean(k),np.std(k),np.min(k),np.max(k),thresh)
            cat1 = treecorr.Catalog(x=x, y=y, k=k, w=w)
            cat2 = treecorr.Catalog(x=x, y=y, k=k, g1=g1, g2=g2)
            nk = treecorr.NKCorrelation(bin_size=0.3, min_sep=10., max_sep=30.)
            kk = treecorr.KKCorrelation(bin_size=0.3, min_sep=6., max_sep=30.)
            kg = treecorr.KGCorrelation(bin_size=0.3, min_sep=10., max_sep=50.)
            nk.process(cat1, cat2)
            kk.process(cat2)
            kg.process(cat2, cat2)
            all_nks.append(nk)
            all_kks.append(kk)
            all_kgs.append(kg)

        mean_nk_xi = np.mean([nk.xi for nk in all_nks], axis=0)
        var_nk_xi = np.var([nk.xi for nk in all_nks], axis=0)
        mean_kk_xi = np.mean([kk.xi for kk in all_kks], axis=0)
        var_kk_xi = np.var([kk.xi for kk in all_kks], axis=0)
        mean_kg_xi = np.mean([kg.xi for kg in all_kgs], axis=0)
        var_kg_xi = np.var([kg.xi for kg in all_kgs], axis=0)

        np.savez(file_name,
                 mean_nk_xi=mean_nk_xi, var_nk_xi=var_nk_xi,
                 mean_kk_xi=mean_kk_xi, var_kk_xi=var_kk_xi,
                 mean_kg_xi=mean_kg_xi, var_kg_xi=var_kg_xi)

    data = np.load(file_name)
    mean_nk_xi = data['mean_nk_xi']
    var_nk_xi = data['var_nk_xi']
    mean_kk_xi = data['mean_kk_xi']
    var_kk_xi = data['var_kk_xi']
    mean_kg_xi = data['mean_kg_xi']
    var_kg_xi = data['var_kg_xi']

    print('mean_nk_xi = ',mean_nk_xi)
    print('var_nk_xi = ',var_nk_xi)
    print('mean_kk_xi = ',mean_kk_xi)
    print('var_kk_xi = ',var_kk_xi)
    print('mean_kg_xi = ',mean_kg_xi)
    print('var_kg_xi = ',var_kg_xi)

    np.random.seed(1234)
    x, y, g1, g2, k = generate_shear_field(nside)
    thresh = np.partition(k.flatten(), -nlens)[-nlens]
    w = np.zeros_like(k)
    w[k>=thresh] = 1.
    cat1 = treecorr.Catalog(x=x, y=y, k=k, w=w)
    cat2 = treecorr.Catalog(x=x, y=y, g1=g1, g2=g2, k=k)
    cat2p = treecorr.Catalog(x=x, y=y, g1=g1, g2=g2, k=k, npatch=npatch)
    cat1p = treecorr.Catalog(x=x, y=y, k=k, w=w, patch=cat2p.patch)

    # NK
    # This one is a bit touchy.  It only works well for a small range of scales.
    # At smaller scales, there just aren't enough sources "behind" the lenses.
    # And at larger scales, the power drops off too quickly (more quickly than shear),
    # since convergence is a more local effect.  So for this choice of ngal, nlens,
    # and power spectrum, this is where the covariance estimate works out reasonably well.
    nk = treecorr.NKCorrelation(bin_size=0.3, min_sep=10, max_sep=30., var_method='jackknife')
    t0 = time.time()
    nk.process(cat1p, cat2p)
    t1 = time.time()
    print('Time for NK jackknife processing = ',t1-t0)
    print('xi = ',nk.xi)
    print('varxi = ',nk.varxi)
    print('ratio = ',nk.varxi / var_nk_xi)
    np.testing.assert_allclose(nk.npairs, nk.npairs)
    np.testing.assert_allclose(nk.xi, nk.xi)
    np.testing.assert_allclose(nk.varxi, var_nk_xi, rtol=0.5*tol_factor)

    # Check sample covariance estimate
    cov_xi = nk.estimate_cov('xi','weight','sample')
    print('NK sample variance:')
    print('varxi = ',cov_xi.diagonal())
    print('ratio = ',cov_xi.diagonal() / var_nk_xi)
    np.testing.assert_allclose(cov_xi.diagonal(), var_nk_xi, rtol=0.6*tol_factor)

    # KK
    # Smaller scales to capture the more local kappa correlations.
    kk = treecorr.KKCorrelation(bin_size=0.3, min_sep=6, max_sep=30., var_method='jackknife')
    t0 = time.time()
    kk.process(cat2p)
    t1 = time.time()
    print('Time for KK jackknife processing = ',t1-t0)
    print('xi = ',kk.xi)
    print('varxi = ',kk.varxi)
    print('ratio = ',kk.varxi / var_kk_xi)
    np.testing.assert_allclose(kk.npairs, kk.npairs)
    np.testing.assert_allclose(kk.xi, kk.xi)
    np.testing.assert_allclose(kk.varxi, var_kk_xi, rtol=0.4*tol_factor)

    # Check sample covariance estimate
    cov_xi = kk.estimate_cov('xi','weight','sample')
    print('KK sample variance:')
    print('varxi = ',cov_xi.diagonal())
    print('ratio = ',cov_xi.diagonal() / var_kk_xi)
    np.testing.assert_allclose(cov_xi.diagonal(), var_kk_xi, rtol=0.4*tol_factor)

    # KG
    # Same scales as we used for NG, which works fine with kappa as the "lens" too.
    kg = treecorr.KGCorrelation(bin_size=0.3, min_sep=10, max_sep=50., var_method='jackknife')
    t0 = time.time()
    kg.process(cat2p, cat2p)
    t1 = time.time()
    print('Time for KG jackknife processing = ',t1-t0)
    print('xi = ',kg.xi)
    print('varxi = ',kg.varxi)
    print('ratio = ',kg.varxi / var_kg_xi)
    np.testing.assert_allclose(kg.npairs, kg.npairs)
    np.testing.assert_allclose(kg.xi, kg.xi)
    np.testing.assert_allclose(kg.varxi, var_kg_xi, rtol=0.3*tol_factor)

    # Check sample covariance estimate
    cov_xi = kg.estimate_cov('xi','weight','sample')
    print('KG sample variance:')
    print('varxi = ',cov_xi.diagonal())
    print('ratio = ',cov_xi.diagonal() / var_kg_xi)
    np.testing.assert_allclose(cov_xi.diagonal(), var_kg_xi, rtol=0.5*tol_factor)


if __name__ == '__main__':
    test_cat_patches()
    test_gg_jk()
    test_ng_jk()
    test_kappa_jk()
