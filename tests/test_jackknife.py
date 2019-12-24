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

from test_helper import assert_raises

def test_gg_jk():
    # Test the variance estimate for GG correlation with jackknife error estimate.

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

        # Multiply by exp(2iphi) to get gamma field, rather than kappa.
        ksq[0,0] = 1.  # Avoid division by zero
        exp2iphi = k**2 / ksq
        f *= exp2iphi

        # Inverse fft gives the real-space field.
        gamma = nside * np.fft.ifft2(f)

        # Generate x,y values for the real-space field
        x,y = np.meshgrid(np.linspace(0.,1000.,nside), np.linspace(0.,1000.,nside))

        x = x.ravel()
        y = y.ravel()
        gamma = gamma.ravel()
        return x, y, np.real(gamma), np.imag(gamma)

    if __name__ == '__main__':
        # 1000 x 1000, so 10^6 points.  With jackknifing, that gives 10^4 per region.
        nside = 1000
        tol_factor = 1
    else:
        # Use 1/10 of the objects when running unit tests
        nside = 300
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
            x, y, g1, g2 = generate_shear_field(nside)
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
    x, y, g1, g2 = generate_shear_field(nside)

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
    cat = treecorr.Catalog(x=x, y=y, g1=g1, g2=g2, npatch=64)
    gg2 = treecorr.GGCorrelation(bin_size=0.3, min_sep=10., max_sep=50., var_method='shot')
    t0 = time.time()
    gg2.process(cat)
    t1 = time.time()
    print('time for shot processing = ',t1-t0)
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
    print('time for jackknife processing = ',t1-t0)
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
    np.testing.assert_allclose(gg3.estimate_cov('xip','weight','jackknife'), gg3.covxip)
    np.testing.assert_allclose(gg3.estimate_cov('xim','weight','jackknife'), gg3.covxim)

    # Can also get the shot covariance matrix using estimate_cov
    np.testing.assert_allclose(gg3.estimate_cov('xip','weight','shot'), np.diag(gg2.varxip))
    np.testing.assert_allclose(gg3.estimate_cov('xim','weight','shot'), np.diag(gg2.varxim))

    # And can even get the jackknife covariance from a run that used var_method='shot'
    np.testing.assert_allclose(gg2.estimate_cov('xip','weight','jackknife'), gg3.covxip)
    np.testing.assert_allclose(gg2.estimate_cov('xim','weight','jackknife'), gg3.covxim)

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


if __name__ == '__main__':
    test_gg_jk()
