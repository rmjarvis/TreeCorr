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
import fitsio
import treecorr

from test_helper import assert_raises, do_pickle, timer, get_from_wiki, CaptureLog, clear_save

def generate_shear_field(nside, rng=None, sqr=False):
    # For these, we don't want a Gaussian power spectrum, since there won't be any
    # significant 3pt power.  Take the nominal power spectrum squared to give it significant
    # non-Gaussianity.
    # Otherwise, this is basically the same as the version in test_patch.py.

    if rng is None:
        rng = np.random.RandomState()

    kvals = np.fft.fftfreq(nside) * 2*np.pi
    kx,ky = np.meshgrid(kvals,kvals)
    k = kx + 1j*ky
    ksq = kx**2 + ky**2

    # For 3pt, we don't put as much power at large scales, since we aren't actually going
    # to measure large triangles anyway.  This puts most of the power where we actually
    # measure the correlations.
    Pk = 1.e4 * ksq**3 / (1. + 300.*ksq)**2

    # Make complex gaussian field in k-space.
    f1 = rng.normal(size=Pk.shape)
    f2 = rng.normal(size=Pk.shape)
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

    if sqr:
        # Square to make it non-Gaussian and have a non-zero mean.
        kappa *= kappa

        # Get the corresponding f for gamma.
        f = np.fft.fft2(kappa)

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


@timer
def test_kkk_jk():
    # Test jackknife and other covariance estimates for kkk correlations.
    # Note: This test takes a while!
    # The main version I think is a pretty decent test of the code correctness.
    # It shows that bootstrap in particular easily gets to within 50% of the right variance.
    # Sometimes within 20%, but because of the randomness there, it varies a bit.
    # Jackknife isn't much worse.  Just a little below 50%.  But still pretty good.
    # Sample and Marked are not great for this test.  I think they will work ok when the
    # triangles of interest are mostly within single patches, but that's not the case we
    # have here, and it would take a lot more points to get to that regime.  So the
    # accuracy tests for those two are pretty loose.

    if __name__ == '__main__':
        # This setup takes about 700 sec to run.
        nside = 100
        nsource = 10000
        npatch = 64
        tol_factor = 1
    elif False:
        # This setup takes about 130 sec to run.
        nside = 100
        nsource = 2000
        npatch = 16
        tol_factor = 3
    elif False:
        # This setup takes about 55 sec to run.
        nside = 100
        nsource = 1000
        npatch = 16
        tol_factor = 20
    else:
        # This setup takes about 13 sec to run.
        # So we use this one for regular unit test runs.
        # It's pretty terrible in terms of testing the accuracy, but it works for code coverage.
        # But whenever actually working on this part of the code, definitely need to switch
        # to one of the above setups.  Preferably run the name==main version to get a good
        # test of the code correctness.
        nside = 100
        nsource = 500
        npatch = 8
        tol_factor = 30

    file_name = 'data/test_kkk_jk_{}.npz'.format(nsource)
    print(file_name)
    if not os.path.isfile(file_name):
        nruns = 1000
        all_kkks = []
        for run in range(nruns):
            x, y, _, _, k = generate_shear_field(nside, sqr=True)
            print(run,': ',np.mean(k),np.std(k))
            indx = np.random.choice(range(len(x)),nsource,replace=False)
            cat = treecorr.Catalog(x=x[indx], y=y[indx], k=k[indx])
            kkk = treecorr.KKKCorrelation(nbins=3, min_sep=30., max_sep=100.,
                                           min_u=0.9, max_u=1.0, nubins=1,
                                           min_v=0.0, max_v=0.1, nvbins=1)
            kkk.process(cat)
            print(kkk.ntri.ravel().tolist())
            print(kkk.zeta.ravel().tolist())
            all_kkks.append(kkk)
        mean_kkk = np.mean([kkk.zeta.ravel() for kkk in all_kkks], axis=0)
        var_kkk = np.var([kkk.zeta.ravel() for kkk in all_kkks], axis=0)

        np.savez(file_name, all_kkk=np.array([kkk.zeta.ravel() for kkk in all_kkks]),
                 mean_kkk=mean_kkk, var_kkk=var_kkk)

    data = np.load(file_name)
    mean_kkk = data['mean_kkk']
    var_kkk = data['var_kkk']
    print('mean = ',mean_kkk)
    print('var = ',var_kkk)

    rng = np.random.RandomState(12345)
    x, y, _, _, k = generate_shear_field(nside, rng, sqr=True)
    indx = rng.choice(range(len(x)),nsource,replace=False)
    cat = treecorr.Catalog(x=x[indx], y=y[indx], k=k[indx])
    kkk = treecorr.KKKCorrelation(nbins=3, min_sep=30., max_sep=100.,
                                  min_u=0.9, max_u=1.0, nubins=1,
                                  min_v=0.0, max_v=0.1, nvbins=1)
    kkk.process(cat)
    print(kkk.ntri.ravel())
    print(kkk.zeta.ravel())
    print(kkk.varzeta.ravel())

    kkkp = kkk.copy()
    catp = treecorr.Catalog(x=x[indx], y=y[indx], k=k[indx], npatch=npatch)

    # Do the same thing with patches.
    kkkp.process(catp)
    print('with patches:')
    print(kkkp.ntri.ravel())
    print(kkkp.zeta.ravel())
    print(kkkp.varzeta.ravel())

    np.testing.assert_allclose(kkkp.ntri, kkk.ntri, rtol=0.05 * tol_factor)
    np.testing.assert_allclose(kkkp.zeta, kkk.zeta, rtol=0.1 * tol_factor, atol=1e-3 * tol_factor)
    np.testing.assert_allclose(kkkp.varzeta, kkk.varzeta, rtol=0.05 * tol_factor)

    print('jackknife:')
    cov = kkkp.estimate_cov('jackknife')
    print(np.diagonal(cov))
    print('max log(ratio) = ',np.max(np.abs(np.log(np.diagonal(cov))-np.log(var_kkk))))
    np.testing.assert_allclose(np.diagonal(cov), var_kkk, rtol=0.5 * tol_factor)
    np.testing.assert_allclose(np.log(np.diagonal(cov)), np.log(var_kkk), atol=0.6*tol_factor)

    # Both sample and marked are pretty bad for this case.  I think because we have
    # a lot of triangles that cross regions, and these methods don't handle that as well
    # as jackknife or bootstrap.  But it's possible there is a better version of the triple
    # selection that would work better, and I just haven't found it.
    # (I tried a few other plausible choices, but they were even worse.)
    print('sample:')
    cov = kkkp.estimate_cov('sample')
    print(np.diagonal(cov))
    print('max log(ratio) = ',np.max(np.abs(np.log(np.diagonal(cov))-np.log(var_kkk))))
    np.testing.assert_allclose(np.diagonal(cov), var_kkk, rtol=0.8 * tol_factor)
    np.testing.assert_allclose(np.log(np.diagonal(cov)), np.log(var_kkk), atol=1.5*tol_factor)

    print('marked:')
    cov = kkkp.estimate_cov('marked_bootstrap')
    print(np.diagonal(cov))
    print('max log(ratio) = ',np.max(np.abs(np.log(np.diagonal(cov))-np.log(var_kkk))))
    np.testing.assert_allclose(np.diagonal(cov), var_kkk, rtol=0.8 * tol_factor)
    np.testing.assert_allclose(np.log(np.diagonal(cov)), np.log(var_kkk), atol=1.5*tol_factor)

    # As for 2pt, bootstrap seems to be pretty reliably the best estimator out of these.
    # However, because it's random, it can occasionally come out even slightly worse than jackknife.
    # So the test tolerance is the same as jackknife, even though the typical performance is
    # quite a bit better usually.
    print('bootstrap:')
    cov = kkkp.estimate_cov('bootstrap')
    print(np.diagonal(cov))
    print('max log(ratio) = ',np.max(np.abs(np.log(np.diagonal(cov))-np.log(var_kkk))))
    np.testing.assert_allclose(np.diagonal(cov), var_kkk, rtol=0.5 * tol_factor)
    np.testing.assert_allclose(np.log(np.diagonal(cov)), np.log(var_kkk), atol=0.6*tol_factor)

    # Now as a cross correlation with all 3 using the same patch catalog.
    print('with 3 patched catalogs:')
    kkkp.process(catp, catp, catp)
    print(kkkp.zeta.ravel())
    np.testing.assert_allclose(kkkp.zeta, kkk.zeta, rtol=0.1 * tol_factor, atol=1e-3 * tol_factor)

    print('jackknife:')
    cov = kkkp.estimate_cov('jackknife')
    print(np.diagonal(cov))
    print('max log(ratio) = ',np.max(np.abs(np.log(np.diagonal(cov))-np.log(var_kkk))))
    np.testing.assert_allclose(np.diagonal(cov), var_kkk, rtol=0.5 * tol_factor)
    np.testing.assert_allclose(np.log(np.diagonal(cov)), np.log(var_kkk), atol=0.6*tol_factor)

    # Note: sample works worse here because of the asymmetry it has between the "first" catalog
    # and the others.
    print('sample:')
    cov = kkkp.estimate_cov('sample')
    print(np.diagonal(cov))
    print('max log(ratio) = ',np.max(np.abs(np.log(np.diagonal(cov))-np.log(var_kkk))))
    np.testing.assert_allclose(np.diagonal(cov), var_kkk, rtol=0.9 * tol_factor)
    np.testing.assert_allclose(np.log(np.diagonal(cov)), np.log(var_kkk), atol=1.9*tol_factor)

    print('marked:')
    cov = kkkp.estimate_cov('marked_bootstrap')
    print(np.diagonal(cov))
    print('max log(ratio) = ',np.max(np.abs(np.log(np.diagonal(cov))-np.log(var_kkk))))
    np.testing.assert_allclose(np.diagonal(cov), var_kkk, rtol=0.9 * tol_factor)
    np.testing.assert_allclose(np.log(np.diagonal(cov)), np.log(var_kkk), atol=1.9*tol_factor)

    print('bootstrap:')
    cov = kkkp.estimate_cov('bootstrap')
    print(np.diagonal(cov))
    print('max log(ratio) = ',np.max(np.abs(np.log(np.diagonal(cov))-np.log(var_kkk))))
    np.testing.assert_allclose(np.diagonal(cov), var_kkk, rtol=0.5 * tol_factor)
    np.testing.assert_allclose(np.log(np.diagonal(cov)), np.log(var_kkk), atol=0.6*tol_factor)

    # Repeat this test with different combinations of patch with non-patch catalogs:
    # All the methods work best when the patches are used for all 3 catalogs.  But there
    # are probably cases where this kind of cross correlation with only some catalogs having
    # patches could be desired.  So this mostly just checks that the code runs properly.

    # Patch on 1 only:
    print('with patches on 1 only:')
    kkkp.process(catp, cat)
    print(kkkp.zeta.ravel())
    np.testing.assert_allclose(kkkp.zeta, kkk.zeta, rtol=0.1 * tol_factor, atol=1e-3 * tol_factor)

    print('jackknife:')
    cov = kkkp.estimate_cov('jackknife')
    print(np.diagonal(cov))
    print('max log(ratio) = ',np.max(np.abs(np.log(np.diagonal(cov))-np.log(var_kkk))))
    np.testing.assert_allclose(np.diagonal(cov), var_kkk, rtol=0.9 * tol_factor)
    np.testing.assert_allclose(np.log(np.diagonal(cov)), np.log(var_kkk), atol=2.0*tol_factor)

    print('sample:')
    cov = kkkp.estimate_cov('sample')
    print(np.diagonal(cov))
    print('max log(ratio) = ',np.max(np.abs(np.log(np.diagonal(cov))-np.log(var_kkk))))
    np.testing.assert_allclose(np.diagonal(cov), var_kkk, rtol=0.9 * tol_factor)
    np.testing.assert_allclose(np.log(np.diagonal(cov)), np.log(var_kkk), atol=2.0*tol_factor)

    print('marked:')
    cov = kkkp.estimate_cov('marked_bootstrap')
    print(np.diagonal(cov))
    print('max log(ratio) = ',np.max(np.abs(np.log(np.diagonal(cov))-np.log(var_kkk))))
    np.testing.assert_allclose(np.diagonal(cov), var_kkk, rtol=1.0 * tol_factor)
    np.testing.assert_allclose(np.log(np.diagonal(cov)), np.log(var_kkk), atol=2.1*tol_factor)

    print('bootstrap:')
    cov = kkkp.estimate_cov('bootstrap')
    print(np.diagonal(cov))
    print('max log(ratio) = ',np.max(np.abs(np.log(np.diagonal(cov))-np.log(var_kkk))))
    np.testing.assert_allclose(np.diagonal(cov), var_kkk, rtol=0.9 * tol_factor)
    np.testing.assert_allclose(np.log(np.diagonal(cov)), np.log(var_kkk), atol=2.1*tol_factor)

    # Patch on 2 only:
    print('with patches on 2 only:')
    kkkp.process(cat, catp, cat)
    print(kkkp.zeta.ravel())
    np.testing.assert_allclose(kkkp.zeta, kkk.zeta, rtol=0.1 * tol_factor, atol=1e-3 * tol_factor)

    print('jackknife:')
    cov = kkkp.estimate_cov('jackknife')
    print(np.diagonal(cov))
    print('max log(ratio) = ',np.max(np.abs(np.log(np.diagonal(cov))-np.log(var_kkk))))
    np.testing.assert_allclose(np.diagonal(cov), var_kkk, rtol=0.9 * tol_factor)
    np.testing.assert_allclose(np.log(np.diagonal(cov)), np.log(var_kkk), atol=2.0*tol_factor)

    print('sample:')
    cov = kkkp.estimate_cov('sample')
    print(np.diagonal(cov))
    print('max log(ratio) = ',np.max(np.abs(np.log(np.diagonal(cov))-np.log(var_kkk))))
    np.testing.assert_allclose(np.diagonal(cov), var_kkk, rtol=0.9 * tol_factor)
    np.testing.assert_allclose(np.log(np.diagonal(cov)), np.log(var_kkk), atol=2.0*tol_factor)

    print('marked:')
    cov = kkkp.estimate_cov('marked_bootstrap')
    print(np.diagonal(cov))
    print('max log(ratio) = ',np.max(np.abs(np.log(np.diagonal(cov))-np.log(var_kkk))))
    np.testing.assert_allclose(np.diagonal(cov), var_kkk, rtol=1.0 * tol_factor)
    np.testing.assert_allclose(np.log(np.diagonal(cov)), np.log(var_kkk), atol=2.1*tol_factor)

    print('bootstrap:')
    cov = kkkp.estimate_cov('bootstrap')
    print(np.diagonal(cov))
    print('max log(ratio) = ',np.max(np.abs(np.log(np.diagonal(cov))-np.log(var_kkk))))
    np.testing.assert_allclose(np.diagonal(cov), var_kkk, rtol=0.9 * tol_factor)
    np.testing.assert_allclose(np.log(np.diagonal(cov)), np.log(var_kkk), atol=2.1*tol_factor)

    # Patch on 3 only:
    print('with patches on 3 only:')
    kkkp.process(cat, cat, catp)
    print(kkkp.zeta.ravel())
    np.testing.assert_allclose(kkkp.zeta, kkk.zeta, rtol=0.1 * tol_factor, atol=1e-3 * tol_factor)

    print('jackknife:')
    cov = kkkp.estimate_cov('jackknife')
    print(np.diagonal(cov))
    print('max log(ratio) = ',np.max(np.abs(np.log(np.diagonal(cov))-np.log(var_kkk))))
    np.testing.assert_allclose(np.diagonal(cov), var_kkk, rtol=0.9 * tol_factor)
    np.testing.assert_allclose(np.log(np.diagonal(cov)), np.log(var_kkk), atol=2.0*tol_factor)

    print('sample:')
    cov = kkkp.estimate_cov('sample')
    print(np.diagonal(cov))
    print('max log(ratio) = ',np.max(np.abs(np.log(np.diagonal(cov))-np.log(var_kkk))))
    np.testing.assert_allclose(np.diagonal(cov), var_kkk, rtol=0.9 * tol_factor)
    np.testing.assert_allclose(np.log(np.diagonal(cov)), np.log(var_kkk), atol=2.0*tol_factor)

    print('marked:')
    cov = kkkp.estimate_cov('marked_bootstrap')
    print(np.diagonal(cov))
    print('max log(ratio) = ',np.max(np.abs(np.log(np.diagonal(cov))-np.log(var_kkk))))
    np.testing.assert_allclose(np.diagonal(cov), var_kkk, rtol=1.0 * tol_factor)
    np.testing.assert_allclose(np.log(np.diagonal(cov)), np.log(var_kkk), atol=2.1*tol_factor)

    print('bootstrap:')
    cov = kkkp.estimate_cov('bootstrap')
    print(np.diagonal(cov))
    print('max log(ratio) = ',np.max(np.abs(np.log(np.diagonal(cov))-np.log(var_kkk))))
    np.testing.assert_allclose(np.diagonal(cov), var_kkk, rtol=0.9 * tol_factor)
    np.testing.assert_allclose(np.log(np.diagonal(cov)), np.log(var_kkk), atol=2.1*tol_factor)

    # Patch on 1,2
    print('with patches on 1,2:')
    kkkp.process(catp, catp, cat)
    print(kkkp.zeta.ravel())
    np.testing.assert_allclose(kkkp.zeta, kkk.zeta, rtol=0.1 * tol_factor, atol=1e-3 * tol_factor)

    print('jackknife:')
    cov = kkkp.estimate_cov('jackknife')
    print(np.diagonal(cov))
    print('max log(ratio) = ',np.max(np.abs(np.log(np.diagonal(cov))-np.log(var_kkk))))
    np.testing.assert_allclose(np.diagonal(cov), var_kkk, rtol=0.6 * tol_factor)
    np.testing.assert_allclose(np.log(np.diagonal(cov)), np.log(var_kkk), atol=0.9*tol_factor)

    print('sample:')
    cov = kkkp.estimate_cov('sample')
    print(np.diagonal(cov))
    print('max log(ratio) = ',np.max(np.abs(np.log(np.diagonal(cov))-np.log(var_kkk))))
    np.testing.assert_allclose(np.diagonal(cov), var_kkk, rtol=0.9 * tol_factor)
    np.testing.assert_allclose(np.log(np.diagonal(cov)), np.log(var_kkk), atol=2.0*tol_factor)

    print('marked:')
    cov = kkkp.estimate_cov('marked_bootstrap')
    print(np.diagonal(cov))
    print('max log(ratio) = ',np.max(np.abs(np.log(np.diagonal(cov))-np.log(var_kkk))))
    np.testing.assert_allclose(np.diagonal(cov), var_kkk, rtol=0.9 * tol_factor)
    np.testing.assert_allclose(np.log(np.diagonal(cov)), np.log(var_kkk), atol=2.0*tol_factor)

    print('bootstrap:')
    cov = kkkp.estimate_cov('bootstrap')
    print(np.diagonal(cov))
    print('max log(ratio) = ',np.max(np.abs(np.log(np.diagonal(cov))-np.log(var_kkk))))
    np.testing.assert_allclose(np.diagonal(cov), var_kkk, rtol=0.6 * tol_factor)
    np.testing.assert_allclose(np.log(np.diagonal(cov)), np.log(var_kkk), atol=0.9*tol_factor)

    # Patch on 2,3
    print('with patches on 2,3:')
    kkkp.process(cat, catp)
    print(kkkp.zeta.ravel())
    np.testing.assert_allclose(kkkp.zeta, kkk.zeta, rtol=0.1 * tol_factor, atol=1e-3 * tol_factor)

    print('jackknife:')
    cov = kkkp.estimate_cov('jackknife')
    print(np.diagonal(cov))
    print('max log(ratio) = ',np.max(np.abs(np.log(np.diagonal(cov))-np.log(var_kkk))))
    np.testing.assert_allclose(np.diagonal(cov), var_kkk, rtol=0.6 * tol_factor)
    np.testing.assert_allclose(np.log(np.diagonal(cov)), np.log(var_kkk), atol=0.9*tol_factor)

    print('sample:')
    cov = kkkp.estimate_cov('sample')
    print(np.diagonal(cov))
    print('max log(ratio) = ',np.max(np.abs(np.log(np.diagonal(cov))-np.log(var_kkk))))
    np.testing.assert_allclose(np.diagonal(cov), var_kkk, rtol=0.9 * tol_factor)
    np.testing.assert_allclose(np.log(np.diagonal(cov)), np.log(var_kkk), atol=2.0*tol_factor)

    print('marked:')
    cov = kkkp.estimate_cov('marked_bootstrap')
    print(np.diagonal(cov))
    print('max log(ratio) = ',np.max(np.abs(np.log(np.diagonal(cov))-np.log(var_kkk))))
    np.testing.assert_allclose(np.diagonal(cov), var_kkk, rtol=0.9 * tol_factor)
    np.testing.assert_allclose(np.log(np.diagonal(cov)), np.log(var_kkk), atol=2.0*tol_factor)

    print('bootstrap:')
    cov = kkkp.estimate_cov('bootstrap')
    print(np.diagonal(cov))
    print('max log(ratio) = ',np.max(np.abs(np.log(np.diagonal(cov))-np.log(var_kkk))))
    np.testing.assert_allclose(np.diagonal(cov), var_kkk, rtol=0.6 * tol_factor)
    np.testing.assert_allclose(np.log(np.diagonal(cov)), np.log(var_kkk), atol=0.9*tol_factor)

    # Patch on 1,3
    print('with patches on 1,3:')
    kkkp.process(catp, cat, catp)
    print(kkkp.zeta.ravel())
    np.testing.assert_allclose(kkkp.zeta, kkk.zeta, rtol=0.1 * tol_factor, atol=1e-3 * tol_factor)

    print('jackknife:')
    cov = kkkp.estimate_cov('jackknife')
    print(np.diagonal(cov))
    print('max log(ratio) = ',np.max(np.abs(np.log(np.diagonal(cov))-np.log(var_kkk))))
    np.testing.assert_allclose(np.diagonal(cov), var_kkk, rtol=0.6 * tol_factor)
    np.testing.assert_allclose(np.log(np.diagonal(cov)), np.log(var_kkk), atol=0.9*tol_factor)

    print('sample:')
    cov = kkkp.estimate_cov('sample')
    print(np.diagonal(cov))
    print('max log(ratio) = ',np.max(np.abs(np.log(np.diagonal(cov))-np.log(var_kkk))))
    np.testing.assert_allclose(np.diagonal(cov), var_kkk, rtol=0.9 * tol_factor)
    np.testing.assert_allclose(np.log(np.diagonal(cov)), np.log(var_kkk), atol=2.0*tol_factor)

    print('marked:')
    cov = kkkp.estimate_cov('marked_bootstrap')
    print(np.diagonal(cov))
    print('max log(ratio) = ',np.max(np.abs(np.log(np.diagonal(cov))-np.log(var_kkk))))
    np.testing.assert_allclose(np.diagonal(cov), var_kkk, rtol=0.9 * tol_factor)
    np.testing.assert_allclose(np.log(np.diagonal(cov)), np.log(var_kkk), atol=2.0*tol_factor)

    print('bootstrap:')
    cov = kkkp.estimate_cov('bootstrap')
    print(np.diagonal(cov))
    print('max log(ratio) = ',np.max(np.abs(np.log(np.diagonal(cov))-np.log(var_kkk))))
    np.testing.assert_allclose(np.diagonal(cov), var_kkk, rtol=0.6 * tol_factor)
    np.testing.assert_allclose(np.log(np.diagonal(cov)), np.log(var_kkk), atol=0.9*tol_factor)

    # Finally a set (with all patches) using the KKKCrossCorrelation class.
    kkkc = treecorr.KKKCrossCorrelation(nbins=3, min_sep=30., max_sep=100.,
                                        min_u=0.9, max_u=1.0, nubins=1,
                                        min_v=0.0, max_v=0.1, nvbins=1)
    print('CrossCorrelation:')
    kkkc.process(catp, catp, catp)
    for k1 in kkkc._all:
        print(k1.ntri.ravel())
        print(k1.zeta.ravel())
        print(k1.varzeta.ravel())

        np.testing.assert_allclose(k1.ntri, kkk.ntri, rtol=0.05 * tol_factor)
        np.testing.assert_allclose(k1.zeta, kkk.zeta, rtol=0.1 * tol_factor, atol=1e-3 * tol_factor)
        np.testing.assert_allclose(k1.varzeta, kkk.varzeta, rtol=0.05 * tol_factor)

    print('jackknife:')
    cov = kkkc.estimate_cov('jackknife')
    print(np.diagonal(cov))
    for i in range(6):
        v = np.diagonal(cov)[i*6:(i+1)*6]
        print('max log(ratio) = ',np.max(np.abs(np.log(v)-np.log(var_kkk))))
        np.testing.assert_allclose(v, var_kkk, rtol=0.5 * tol_factor)
        np.testing.assert_allclose(np.log(v), np.log(var_kkk), atol=0.6*tol_factor)

    print('sample:')
    cov = kkkc.estimate_cov('sample')
    print(np.diagonal(cov))
    for i in range(6):
        v = np.diagonal(cov)[i*6:(i+1)*6]
        print('max log(ratio) = ',np.max(np.abs(np.log(v)-np.log(var_kkk))))
        np.testing.assert_allclose(v, var_kkk, rtol=0.9 * tol_factor)
        np.testing.assert_allclose(np.log(v), np.log(var_kkk), atol=2.0*tol_factor)

    print('marked:')
    cov = kkkc.estimate_cov('marked_bootstrap')
    print(np.diagonal(cov))
    for i in range(6):
        v = np.diagonal(cov)[i*6:(i+1)*6]
        print('max log(ratio) = ',np.max(np.abs(np.log(v)-np.log(var_kkk))))
        np.testing.assert_allclose(v, var_kkk, rtol=0.9 * tol_factor)
        np.testing.assert_allclose(np.log(v), np.log(var_kkk), atol=2.0*tol_factor)

    print('bootstrap:')
    cov = kkkc.estimate_cov('bootstrap')
    print(np.diagonal(cov))
    for i in range(6):
        v = np.diagonal(cov)[i*6:(i+1)*6]
        print('max log(ratio) = ',np.max(np.abs(np.log(v)-np.log(var_kkk))))
        np.testing.assert_allclose(v, var_kkk, rtol=0.5 * tol_factor)
        np.testing.assert_allclose(np.log(v), np.log(var_kkk), atol=0.6*tol_factor)

    # All catalogs need to have the same number of patches
    catq = treecorr.Catalog(x=x[indx], y=y[indx], k=k[indx], npatch=2*npatch)
    with assert_raises(RuntimeError):
        kkkp.process(catp, catq)
    with assert_raises(RuntimeError):
        kkkp.process(catp, catq, catq)
    with assert_raises(RuntimeError):
        kkkp.process(catq, catp, catq)
    with assert_raises(RuntimeError):
        kkkp.process(catq, catq, catp)

@timer
def test_brute_jk():
    # With bin_slop = 0, the jackknife calculation from patches should match a
    # brute force calcaulation where we literally remove one patch at a time to make
    # the vectors.
    if __name__ == '__main__':
        nside = 100
        nsource = 500
        npatch = 16
        rand_factor = 5
    else:
        nside = 100
        nsource = 300
        npatch = 8
        rand_factor = 5

    rng = np.random.RandomState(8675309)
    x, y, g1, g2, k = generate_shear_field(nside)
    # randomize positions slightly, since with grid, can get v=0 exactly, which is ambiguous
    # as to +- sign for v.  So complicates verification of equal results.
    x += rng.normal(0,0.01,len(x))
    y += rng.normal(0,0.01,len(y))

    indx = rng.choice(range(len(x)),nsource,replace=False)
    source_cat_nopatch = treecorr.Catalog(x=x[indx], y=y[indx],
                                          g1=g1[indx], g2=g2[indx], k=k[indx])
    source_cat = treecorr.Catalog(x=x[indx], y=y[indx],
                                  g1=g1[indx], g2=g2[indx], k=k[indx],
                                  npatch=npatch)
    print('source_cat patches = ',np.unique(source_cat.patch))
    print('len = ',source_cat.nobj, source_cat.ntot)
    assert source_cat.nobj == nsource

    # Start with KKK, since relatively simple.
    kkk1 = treecorr.KKKCorrelation(nbins=3, min_sep=30., max_sep=100., brute=True,
                                   min_u=0.8, max_u=1.0, nubins=1,
                                   min_v=0., max_v=0.2, nvbins=1)
    kkk1.process(source_cat_nopatch)

    kkk = treecorr.KKKCorrelation(nbins=3, min_sep=30., max_sep=100., brute=True,
                                  min_u=0.8, max_u=1.0, nubins=1,
                                  min_v=0., max_v=0.2, nvbins=1,
                                  var_method='jackknife')
    kkk.process(source_cat)
    np.testing.assert_allclose(kkk.zeta, kkk1.zeta)

    kkk_zeta_list = []
    for i in range(npatch):
        source_cat1 = treecorr.Catalog(x=source_cat.x[source_cat.patch != i],
                                       y=source_cat.y[source_cat.patch != i],
                                       k=source_cat.k[source_cat.patch != i],
                                       g1=source_cat.g1[source_cat.patch != i],
                                       g2=source_cat.g2[source_cat.patch != i])
        kkk1 = treecorr.KKKCorrelation(nbins=3, min_sep=30., max_sep=100., brute=True,
                                       min_u=0.8, max_u=1.0, nubins=1,
                                       min_v=0., max_v=0.2, nvbins=1)
        kkk1.process(source_cat1)
        print('zeta = ',kkk1.zeta.ravel())
        kkk_zeta_list.append(kkk1.zeta.ravel())

    kkk_zeta_list = np.array(kkk_zeta_list)
    cov = np.cov(kkk_zeta_list.T, bias=True) * (len(kkk_zeta_list)-1)
    varzeta = np.diagonal(np.cov(kkk_zeta_list.T, bias=True)) * (len(kkk_zeta_list)-1)
    print('KKK: treecorr jackknife varzeta = ',kkk.varzeta.ravel())
    print('KKK: direct jackknife varzeta = ',varzeta)
    np.testing.assert_allclose(kkk.varzeta.ravel(), varzeta)

    # Now GGG
    ggg1 = treecorr.GGGCorrelation(nbins=3, min_sep=30., max_sep=100., brute=True,
                                   min_u=0.8, max_u=1.0, nubins=1,
                                   min_v=0., max_v=0.2, nvbins=1)
    ggg1.process(source_cat_nopatch)

    ggg = treecorr.GGGCorrelation(nbins=3, min_sep=30., max_sep=100., brute=True,
                                  min_u=0.8, max_u=1.0, nubins=1,
                                  min_v=0., max_v=0.2, nvbins=1,
                                  var_method='jackknife')
    ggg.process(source_cat)
    np.testing.assert_allclose(ggg.gam0, ggg1.gam0)
    np.testing.assert_allclose(ggg.gam1, ggg1.gam1)
    np.testing.assert_allclose(ggg.gam2, ggg1.gam2)
    np.testing.assert_allclose(ggg.gam3, ggg1.gam3)

    ggg_gam0_list = []
    ggg_gam1_list = []
    ggg_gam2_list = []
    ggg_gam3_list = []
    ggg_map3_list = []
    for i in range(npatch):
        source_cat1 = treecorr.Catalog(x=source_cat.x[source_cat.patch != i],
                                       y=source_cat.y[source_cat.patch != i],
                                       k=source_cat.k[source_cat.patch != i],
                                       g1=source_cat.g1[source_cat.patch != i],
                                       g2=source_cat.g2[source_cat.patch != i])
        ggg1 = treecorr.GGGCorrelation(nbins=3, min_sep=30., max_sep=100., brute=True,
                                       min_u=0.8, max_u=1.0, nubins=1,
                                       min_v=0., max_v=0.2, nvbins=1)
        ggg1.process(source_cat1)
        ggg_gam0_list.append(ggg1.gam0.ravel())
        ggg_gam1_list.append(ggg1.gam1.ravel())
        ggg_gam2_list.append(ggg1.gam2.ravel())
        ggg_gam3_list.append(ggg1.gam3.ravel())
        ggg_map3_list.append(ggg1.calculateMap3()[0])

    ggg_gam0_list = np.array(ggg_gam0_list)
    vargam0 = np.diagonal(np.cov(ggg_gam0_list.T, bias=True)) * (len(ggg_gam0_list)-1)
    print('GG: treecorr jackknife vargam0 = ',ggg.vargam0.ravel())
    print('GG: direct jackknife vargam0 = ',vargam0)
    np.testing.assert_allclose(ggg.vargam0.ravel(), vargam0)
    ggg_gam1_list = np.array(ggg_gam1_list)
    vargam1 = np.diagonal(np.cov(ggg_gam1_list.T, bias=True)) * (len(ggg_gam1_list)-1)
    print('GG: treecorr jackknife vargam1 = ',ggg.vargam1.ravel())
    print('GG: direct jackknife vargam1 = ',vargam1)
    np.testing.assert_allclose(ggg.vargam1.ravel(), vargam1)
    ggg_gam2_list = np.array(ggg_gam2_list)
    vargam2 = np.diagonal(np.cov(ggg_gam2_list.T, bias=True)) * (len(ggg_gam2_list)-1)
    print('GG: treecorr jackknife vargam2 = ',ggg.vargam2.ravel())
    print('GG: direct jackknife vargam2 = ',vargam2)
    np.testing.assert_allclose(ggg.vargam2.ravel(), vargam2)
    ggg_gam3_list = np.array(ggg_gam3_list)
    vargam3 = np.diagonal(np.cov(ggg_gam3_list.T, bias=True)) * (len(ggg_gam3_list)-1)
    print('GG: treecorr jackknife vargam3 = ',ggg.vargam3.ravel())
    print('GG: direct jackknife vargam3 = ',vargam3)
    np.testing.assert_allclose(ggg.vargam3.ravel(), vargam3)

    ggg_map3_list = np.array(ggg_map3_list)
    varmap3 = np.diagonal(np.cov(ggg_map3_list.T, bias=True)) * (len(ggg_map3_list)-1)
    covmap3 = treecorr.estimate_multi_cov([ggg], 'jackknife',
                                          lambda corrs: corrs[0].calculateMap3()[0])
    print('GG: treecorr jackknife varmap3 = ',np.diagonal(covmap3))
    print('GG: direct jackknife varmap3 = ',varmap3)
    np.testing.assert_allclose(np.diagonal(covmap3), varmap3)

    return
    # Finally, test NN, which is complicated, since several different combinations of randoms.
    # 1. (DD-RR)/RR
    # 2. (DD-2DR+RR)/RR
    # 3. (DD-2RD+RR)/RR
    # 4. (DD-DR-RD+RR)/RR

    rand_source_cat = treecorr.Catalog(x=rng.uniform(0,1000,nsource*rand_factor),
                                       y=rng.uniform(0,1000,nsource*rand_factor),
                                       patch_centers=source_cat.patch_centers)
    print('rand_source_cat patches = ',np.unique(rand_source_cat.patch))
    print('len = ',rand_source_cat.nobj, rand_source_cat.ntot)

    dd = treecorr.NNCorrelation(bin_size=0.3, min_sep=10., max_sep=30., bin_slop=0,
                                var_method='jackknife')
    dd.process(lens_cat, source_cat)
    rr = treecorr.NNCorrelation(bin_size=0.3, min_sep=10., max_sep=30., bin_slop=0,
                                var_method='jackknife')
    rr.process(rand_lens_cat, rand_source_cat)
    rd = treecorr.NNCorrelation(bin_size=0.3, min_sep=10., max_sep=30., bin_slop=0,
                                var_method='jackknife')
    rd.process(rand_lens_cat, source_cat)
    dr = treecorr.NNCorrelation(bin_size=0.3, min_sep=10., max_sep=30., bin_slop=0,
                                var_method='jackknife')
    dr.process(lens_cat, rand_source_cat)

    # Now do this using brute force calculation.
    xi1_list = []
    xi2_list = []
    xi3_list = []
    xi4_list = []
    for i in range(npatch):
        lens_cat1 = treecorr.Catalog(x=lens_cat.x[lens_cat.patch != i],
                                     y=lens_cat.y[lens_cat.patch != i])
        source_cat1 = treecorr.Catalog(x=source_cat.x[source_cat.patch != i],
                                       y=source_cat.y[source_cat.patch != i])
        rand_lens_cat1 = treecorr.Catalog(x=rand_lens_cat.x[rand_lens_cat.patch != i],
                                          y=rand_lens_cat.y[rand_lens_cat.patch != i])
        rand_source_cat1 = treecorr.Catalog(x=rand_source_cat.x[rand_source_cat.patch != i],
                                            y=rand_source_cat.y[rand_source_cat.patch != i])
        dd1 = treecorr.NNCorrelation(bin_size=0.3, min_sep=10., max_sep=30., bin_slop=0)
        dd1.process(lens_cat1, source_cat1)
        rr1 = treecorr.NNCorrelation(bin_size=0.3, min_sep=10., max_sep=30., bin_slop=0)
        rr1.process(rand_lens_cat1, rand_source_cat1)
        rd1 = treecorr.NNCorrelation(bin_size=0.3, min_sep=10., max_sep=30., bin_slop=0)
        rd1.process(rand_lens_cat1, source_cat1)
        dr1 = treecorr.NNCorrelation(bin_size=0.3, min_sep=10., max_sep=30., bin_slop=0)
        dr1.process(lens_cat1, rand_source_cat1)
        xi1_list.append(dd1.calculateXi(rr1)[0])
        xi2_list.append(dd1.calculateXi(rr1,dr=dr1)[0])
        xi3_list.append(dd1.calculateXi(rr1,rd=rd1)[0])
        xi4_list.append(dd1.calculateXi(rr1,dr=dr1,rd=rd1)[0])

    print('(DD-RR)/RR')
    xi1_list = np.array(xi1_list)
    xi1, varxi1 = dd.calculateXi(rr)
    varxi = np.diagonal(np.cov(xi1_list.T, bias=True)) * (len(xi1_list)-1)
    print('treecorr jackknife varxi = ',varxi1)
    print('direct jackknife varxi = ',varxi)
    np.testing.assert_allclose(dd.varxi, varxi)

    print('(DD-2DR+RR)/RR')
    xi2_list = np.array(xi2_list)
    xi2, varxi2 = dd.calculateXi(rr, dr=dr)
    varxi = np.diagonal(np.cov(xi2_list.T, bias=True)) * (len(xi2_list)-1)
    print('treecorr jackknife varxi = ',varxi2)
    print('direct jackknife varxi = ',varxi)
    np.testing.assert_allclose(dd.varxi, varxi)

    print('(DD-2RD+RR)/RR')
    xi3_list = np.array(xi3_list)
    xi3, varxi3 = dd.calculateXi(rr, rd=rd)
    varxi = np.diagonal(np.cov(xi3_list.T, bias=True)) * (len(xi3_list)-1)
    print('treecorr jackknife varxi = ',varxi3)
    print('direct jackknife varxi = ',varxi)
    np.testing.assert_allclose(dd.varxi, varxi)

    print('(DD-DR-RD+RR)/RR')
    xi4_list = np.array(xi4_list)
    xi4, varxi4 = dd.calculateXi(rr, rd=rd, dr=dr)
    varxi = np.diagonal(np.cov(xi4_list.T, bias=True)) * (len(xi4_list)-1)
    print('treecorr jackknife varxi = ',varxi4)
    print('direct jackknife varxi = ',varxi)
    np.testing.assert_allclose(dd.varxi, varxi)

@timer
def test_finalize_false():

    nside = 100
    nsource = 80
    npatch = 16

    # Make three independent data sets
    rng = np.random.RandomState(8675309)
    x_1, y_1, g1_1, g2_1, k_1 = generate_shear_field(nside)
    indx = rng.choice(range(len(x_1)),nsource,replace=False)
    x_1 = x_1[indx]
    y_1 = y_1[indx]
    g1_1 = g1_1[indx]
    g2_1 = g2_1[indx]
    k_1 = k_1[indx]
    x_1 += rng.normal(0,0.01,nsource)
    y_1 += rng.normal(0,0.01,nsource)
    x_2, y_2, g1_2, g2_2, k_2 = generate_shear_field(nside)
    indx = rng.choice(range(len(x_2)),nsource,replace=False)
    x_2 = x_2[indx]
    y_2 = y_2[indx]
    g1_2 = g1_2[indx]
    g2_2 = g2_2[indx]
    k_2 = k_2[indx]
    x_2 += rng.normal(0,0.01,nsource)
    y_2 += rng.normal(0,0.01,nsource)
    x_3, y_3, g1_3, g2_3, k_3 = generate_shear_field(nside)
    indx = rng.choice(range(len(x_3)),nsource,replace=False)
    x_3 = x_3[indx]
    y_3 = y_3[indx]
    g1_3 = g1_3[indx]
    g2_3 = g2_3[indx]
    k_3 = k_3[indx]
    x_3 += rng.normal(0,0.01,nsource)
    y_3 += rng.normal(0,0.01,nsource)

    # Make a single catalog with all three together
    cat = treecorr.Catalog(x=np.concatenate([x_1, x_2, x_3]),
                           y=np.concatenate([y_1, y_2, y_3]),
                           g1=np.concatenate([g1_1, g1_2, g1_3]),
                           g2=np.concatenate([g2_1, g2_2, g2_3]),
                           k=np.concatenate([k_1, k_2, k_3]),
                           npatch=npatch)

    # Now the three separately, using the same patch centers
    cat1 = treecorr.Catalog(x=x_1, y=y_1, g1=g1_1, g2=g2_1, k=k_1, patch_centers=cat.patch_centers)
    cat2 = treecorr.Catalog(x=x_2, y=y_2, g1=g1_2, g2=g2_2, k=k_2, patch_centers=cat.patch_centers)
    cat3 = treecorr.Catalog(x=x_3, y=y_3, g1=g1_3, g2=g2_3, k=k_3, patch_centers=cat.patch_centers)

    np.testing.assert_array_equal(cat1.patch, cat.patch[0:nsource])
    np.testing.assert_array_equal(cat2.patch, cat.patch[nsource:2*nsource])
    np.testing.assert_array_equal(cat3.patch, cat.patch[2*nsource:3*nsource])

    # KKK auto
    kkk1 = treecorr.KKKCorrelation(nbins=3, min_sep=30., max_sep=100., brute=True,
                                   min_u=0.8, max_u=1.0, nubins=1,
                                   min_v=0., max_v=0.2, nvbins=1)
    kkk1.process(cat)

    kkk2 = treecorr.KKKCorrelation(nbins=3, min_sep=30., max_sep=100., brute=True,
                                   min_u=0.8, max_u=1.0, nubins=1,
                                   min_v=0., max_v=0.2, nvbins=1)
    kkk2.process(cat1, initialize=True, finalize=False)
    kkk2.process(cat2, initialize=False, finalize=False)
    kkk2.process(cat3, initialize=False, finalize=False)
    kkk2.process(cat1, cat2, initialize=False, finalize=False)
    kkk2.process(cat1, cat3, initialize=False, finalize=False)
    kkk2.process(cat2, cat1, initialize=False, finalize=False)
    kkk2.process(cat2, cat3, initialize=False, finalize=False)
    kkk2.process(cat3, cat1, initialize=False, finalize=False)
    kkk2.process(cat3, cat2, initialize=False, finalize=False)
    kkk2.process(cat1, cat2, cat3, initialize=False, finalize=True)

    np.testing.assert_allclose(kkk1.ntri, kkk2.ntri)
    np.testing.assert_allclose(kkk1.weight, kkk2.weight)
    np.testing.assert_allclose(kkk1.meand1, kkk2.meand1)
    np.testing.assert_allclose(kkk1.meand2, kkk2.meand2)
    np.testing.assert_allclose(kkk1.meand3, kkk2.meand3)
    np.testing.assert_allclose(kkk1.zeta, kkk2.zeta)

    # KKK cross12
    cat23 = treecorr.Catalog(x=np.concatenate([x_2, x_3]),
                             y=np.concatenate([y_2, y_3]),
                             g1=np.concatenate([g1_2, g1_3]),
                             g2=np.concatenate([g2_2, g2_3]),
                             k=np.concatenate([k_2, k_3]),
                             patch_centers=cat.patch_centers)
    np.testing.assert_array_equal(cat23.patch, cat.patch[nsource:3*nsource])

    kkk1.process(cat1, cat23)
    kkk2.process(cat1, cat2, initialize=True, finalize=False)
    kkk2.process(cat1, cat3, initialize=False, finalize=False)
    kkk2.process(cat1, cat2, cat3, initialize=False, finalize=True)

    np.testing.assert_allclose(kkk1.ntri, kkk2.ntri)
    np.testing.assert_allclose(kkk1.weight, kkk2.weight)
    np.testing.assert_allclose(kkk1.meand1, kkk2.meand1)
    np.testing.assert_allclose(kkk1.meand2, kkk2.meand2)
    np.testing.assert_allclose(kkk1.meand3, kkk2.meand3)
    np.testing.assert_allclose(kkk1.zeta, kkk2.zeta)

    # KKKCross cross12
    kkkc1 = treecorr.KKKCrossCorrelation(nbins=3, min_sep=30., max_sep=100., brute=True,
                                         min_u=0.8, max_u=1.0, nubins=1,
                                         min_v=0., max_v=0.2, nvbins=1)
    kkkc1.process(cat1, cat23)

    kkkc2 = treecorr.KKKCrossCorrelation(nbins=3, min_sep=30., max_sep=100., brute=True,
                                         min_u=0.8, max_u=1.0, nubins=1,
                                         min_v=0., max_v=0.2, nvbins=1)
    kkkc2.process(cat1, cat2, initialize=True, finalize=False)
    kkkc2.process(cat1, cat3, initialize=False, finalize=False)
    kkkc2.process(cat1, cat2, cat3, initialize=False, finalize=True)

    for perm in ['k1k2k3', 'k1k3k2', 'k2k1k3', 'k2k3k1', 'k3k1k2', 'k3k2k1']:
        kkk1 = getattr(kkkc1, perm)
        kkk2 = getattr(kkkc2, perm)
        np.testing.assert_allclose(kkk1.ntri, kkk2.ntri)
        np.testing.assert_allclose(kkk1.weight, kkk2.weight)
        np.testing.assert_allclose(kkk1.meand1, kkk2.meand1)
        np.testing.assert_allclose(kkk1.meand2, kkk2.meand2)
        np.testing.assert_allclose(kkk1.meand3, kkk2.meand3)
        np.testing.assert_allclose(kkk1.zeta, kkk2.zeta)

    # KKK cross
    kkk1.process(cat, cat2, cat3)
    kkk2.process(cat1, cat2, cat3, initialize=True, finalize=False)
    kkk2.process(cat2, cat2, cat3, initialize=False, finalize=False)
    kkk2.process(cat3, cat2, cat3, initialize=False, finalize=True)

    np.testing.assert_allclose(kkk1.ntri, kkk2.ntri)
    np.testing.assert_allclose(kkk1.weight, kkk2.weight)
    np.testing.assert_allclose(kkk1.meand1, kkk2.meand1)
    np.testing.assert_allclose(kkk1.meand2, kkk2.meand2)
    np.testing.assert_allclose(kkk1.meand3, kkk2.meand3)
    np.testing.assert_allclose(kkk1.zeta, kkk2.zeta)

    # KKKCross cross
    kkkc1.process(cat, cat2, cat3)
    kkkc2.process(cat1, cat2, cat3, initialize=True, finalize=False)
    kkkc2.process(cat2, cat2, cat3, initialize=False, finalize=False)
    kkkc2.process(cat3, cat2, cat3, initialize=False, finalize=True)

    for perm in ['k1k2k3', 'k1k3k2', 'k2k1k3', 'k2k3k1', 'k3k1k2', 'k3k2k1']:
        kkk1 = getattr(kkkc1, perm)
        kkk2 = getattr(kkkc2, perm)
        np.testing.assert_allclose(kkk1.ntri, kkk2.ntri)
        np.testing.assert_allclose(kkk1.weight, kkk2.weight)
        np.testing.assert_allclose(kkk1.meand1, kkk2.meand1)
        np.testing.assert_allclose(kkk1.meand2, kkk2.meand2)
        np.testing.assert_allclose(kkk1.meand3, kkk2.meand3)
        np.testing.assert_allclose(kkk1.zeta, kkk2.zeta)

    # GGG auto
    ggg1 = treecorr.GGGCorrelation(nbins=3, min_sep=30., max_sep=100., brute=True,
                                   min_u=0.8, max_u=1.0, nubins=1,
                                   min_v=0., max_v=0.2, nvbins=1)
    ggg1.process(cat)

    ggg2 = treecorr.GGGCorrelation(nbins=3, min_sep=30., max_sep=100., brute=True,
                                   min_u=0.8, max_u=1.0, nubins=1,
                                   min_v=0., max_v=0.2, nvbins=1)
    ggg2.process(cat1, initialize=True, finalize=False)
    ggg2.process(cat2, initialize=False, finalize=False)
    ggg2.process(cat3, initialize=False, finalize=False)
    ggg2.process(cat1, cat2, initialize=False, finalize=False)
    ggg2.process(cat1, cat3, initialize=False, finalize=False)
    ggg2.process(cat2, cat1, initialize=False, finalize=False)
    ggg2.process(cat2, cat3, initialize=False, finalize=False)
    ggg2.process(cat3, cat1, initialize=False, finalize=False)
    ggg2.process(cat3, cat2, initialize=False, finalize=False)
    ggg2.process(cat1, cat2, cat3, initialize=False, finalize=True)

    np.testing.assert_allclose(ggg1.ntri, ggg2.ntri)
    np.testing.assert_allclose(ggg1.weight, ggg2.weight)
    np.testing.assert_allclose(ggg1.meand1, ggg2.meand1)
    np.testing.assert_allclose(ggg1.meand2, ggg2.meand2)
    np.testing.assert_allclose(ggg1.meand3, ggg2.meand3)
    np.testing.assert_allclose(ggg1.gam0, ggg2.gam0)
    np.testing.assert_allclose(ggg1.gam1, ggg2.gam1)
    np.testing.assert_allclose(ggg1.gam2, ggg2.gam2)
    np.testing.assert_allclose(ggg1.gam3, ggg2.gam3)

    # GGG cross12
    cat23 = treecorr.Catalog(x=np.concatenate([x_2, x_3]),
                             y=np.concatenate([y_2, y_3]),
                             g1=np.concatenate([g1_2, g1_3]),
                             g2=np.concatenate([g2_2, g2_3]),
                             k=np.concatenate([k_2, k_3]),
                             patch_centers=cat.patch_centers)
    np.testing.assert_array_equal(cat23.patch, cat.patch[nsource:3*nsource])

    ggg1.process(cat1, cat23)
    ggg2.process(cat1, cat2, initialize=True, finalize=False)
    ggg2.process(cat1, cat3, initialize=False, finalize=False)
    ggg2.process(cat1, cat2, cat3, initialize=False, finalize=True)

    np.testing.assert_allclose(ggg1.ntri, ggg2.ntri)
    np.testing.assert_allclose(ggg1.weight, ggg2.weight)
    np.testing.assert_allclose(ggg1.meand1, ggg2.meand1)
    np.testing.assert_allclose(ggg1.meand2, ggg2.meand2)
    np.testing.assert_allclose(ggg1.meand3, ggg2.meand3)
    np.testing.assert_allclose(ggg1.gam0, ggg2.gam0)
    np.testing.assert_allclose(ggg1.gam1, ggg2.gam1)
    np.testing.assert_allclose(ggg1.gam2, ggg2.gam2)
    np.testing.assert_allclose(ggg1.gam3, ggg2.gam3)

    # GGGCross cross12
    gggc1 = treecorr.GGGCrossCorrelation(nbins=3, min_sep=30., max_sep=100., brute=True,
                                         min_u=0.8, max_u=1.0, nubins=1,
                                         min_v=0., max_v=0.2, nvbins=1)
    gggc1.process(cat1, cat23)

    gggc2 = treecorr.GGGCrossCorrelation(nbins=3, min_sep=30., max_sep=100., brute=True,
                                         min_u=0.8, max_u=1.0, nubins=1,
                                         min_v=0., max_v=0.2, nvbins=1)
    gggc2.process(cat1, cat2, initialize=True, finalize=False)
    gggc2.process(cat1, cat3, initialize=False, finalize=False)
    gggc2.process(cat1, cat2, cat3, initialize=False, finalize=True)

    for perm in ['g1g2g3', 'g1g3g2', 'g2g1g3', 'g2g3g1', 'g3g1g2', 'g3g2g1']:
        ggg1 = getattr(gggc1, perm)
        ggg2 = getattr(gggc2, perm)
        np.testing.assert_allclose(ggg1.ntri, ggg2.ntri)
        np.testing.assert_allclose(ggg1.weight, ggg2.weight)
        np.testing.assert_allclose(ggg1.meand1, ggg2.meand1)
        np.testing.assert_allclose(ggg1.meand2, ggg2.meand2)
        np.testing.assert_allclose(ggg1.meand3, ggg2.meand3)
        np.testing.assert_allclose(ggg1.gam0, ggg2.gam0)
        np.testing.assert_allclose(ggg1.gam1, ggg2.gam1)
        np.testing.assert_allclose(ggg1.gam2, ggg2.gam2)
        np.testing.assert_allclose(ggg1.gam3, ggg2.gam3)

    # GGG cross
    ggg1.process(cat, cat2, cat3)
    ggg2.process(cat1, cat2, cat3, initialize=True, finalize=False)
    ggg2.process(cat2, cat2, cat3, initialize=False, finalize=False)
    ggg2.process(cat3, cat2, cat3, initialize=False, finalize=True)

    np.testing.assert_allclose(ggg1.ntri, ggg2.ntri)
    np.testing.assert_allclose(ggg1.weight, ggg2.weight)
    np.testing.assert_allclose(ggg1.meand1, ggg2.meand1)
    np.testing.assert_allclose(ggg1.meand2, ggg2.meand2)
    np.testing.assert_allclose(ggg1.meand3, ggg2.meand3)
    np.testing.assert_allclose(ggg1.gam0, ggg2.gam0)
    np.testing.assert_allclose(ggg1.gam1, ggg2.gam1)
    np.testing.assert_allclose(ggg1.gam2, ggg2.gam2)
    np.testing.assert_allclose(ggg1.gam3, ggg2.gam3)

    # GGGCross cross
    gggc1.process(cat, cat2, cat3)
    gggc2.process(cat1, cat2, cat3, initialize=True, finalize=False)
    gggc2.process(cat2, cat2, cat3, initialize=False, finalize=False)
    gggc2.process(cat3, cat2, cat3, initialize=False, finalize=True)

    for perm in ['g1g2g3', 'g1g3g2', 'g2g1g3', 'g2g3g1', 'g3g1g2', 'g3g2g1']:
        ggg1 = getattr(gggc1, perm)
        ggg2 = getattr(gggc2, perm)
        np.testing.assert_allclose(ggg1.ntri, ggg2.ntri)
        np.testing.assert_allclose(ggg1.weight, ggg2.weight)
        np.testing.assert_allclose(ggg1.meand1, ggg2.meand1)
        np.testing.assert_allclose(ggg1.meand2, ggg2.meand2)
        np.testing.assert_allclose(ggg1.meand3, ggg2.meand3)
        np.testing.assert_allclose(ggg1.gam0, ggg2.gam0)
        np.testing.assert_allclose(ggg1.gam1, ggg2.gam1)
        np.testing.assert_allclose(ggg1.gam2, ggg2.gam2)
        np.testing.assert_allclose(ggg1.gam3, ggg2.gam3)

@timer
def test_lowmem():
    # Test using patches to keep the memory usage lower.

    nside = 100
    if __name__ == '__main__':
        nsource = 10000
        npatch = 4
        himem = 7.e5
        lomem = 7.e4
    else:
        nsource = 1000
        npatch = 4
        himem = 1.e5
        lomem = 6.e4

    rng = np.random.RandomState(8675309)
    rng = np.random.RandomState(8675309)
    x, y, g1, g2, k = generate_shear_field(nside)
    indx = rng.choice(range(len(x)),nsource,replace=False)
    x = x[indx]
    y = y[indx]
    g1 = g1[indx]
    g2 = g2[indx]
    k = k[indx]
    x += rng.normal(0,0.01,nsource)
    y += rng.normal(0,0.01,nsource)

    file_name = os.path.join('output','test_lowmem_3pt.fits')
    orig_cat = treecorr.Catalog(x=x, y=y, g1=g1, g2=g2, k=k, npatch=npatch)
    patch_centers = orig_cat.patch_centers
    orig_cat.write(file_name)
    del orig_cat

    try:
        import guppy
        hp = guppy.hpy()
        hp.setrelheap()
    except Exception:
        hp = None

    full_cat = treecorr.Catalog(file_name,
                                x_col='x', y_col='y', g1_col='g1', g2_col='g2', k_col='k',
                                patch_centers=patch_centers)

    kkk = treecorr.KKKCorrelation(nbins=1, min_sep=280., max_sep=300.,
                                  min_u=0.95, max_u=1.0, nubins=1,
                                  min_v=0., max_v=0.05, nvbins=1)

    t0 = time.time()
    s0 = hp.heap().size if hp else 0
    kkk.process(full_cat)
    t1 = time.time()
    s1 = hp.heap().size if hp else 2*himem
    print('regular: ',s1, t1-t0, s1-s0)
    assert s1-s0 > himem  # This version uses a lot of memory.

    ntri1 = kkk.ntri
    zeta1 = kkk.zeta
    full_cat.unload()
    kkk.clear()

    # Remake with save_patch_dir.
    clear_save('test_lowmem_3pt_%03d.fits', npatch)
    save_cat = treecorr.Catalog(file_name,
                                x_col='x', y_col='y', g1_col='g1', g2_col='g2', k_col='k',
                                patch_centers=patch_centers, save_patch_dir='output')

    t0 = time.time()
    s0 = hp.heap().size if hp else 0
    kkk.process(save_cat, low_mem=True, finalize=False)
    t1 = time.time()
    s1 = hp.heap().size if hp else 0
    print('lomem 1: ',s1, t1-t0, s1-s0)
    assert s1-s0 < lomem  # This version uses a lot less memory
    ntri2 = kkk.ntri
    zeta2 = kkk.zeta
    print('ntri1 = ',ntri1)
    print('zeta1 = ',zeta1)
    np.testing.assert_array_equal(ntri2, ntri1)
    np.testing.assert_array_equal(zeta2, zeta1)

    # Check running as a cross-correlation
    save_cat.unload()
    t0 = time.time()
    s0 = hp.heap().size if hp else 0
    kkk.process(save_cat, save_cat, low_mem=True)
    t1 = time.time()
    s1 = hp.heap().size if hp else 0
    print('lomem 2: ',s1, t1-t0, s1-s0)
    assert s1-s0 < lomem
    ntri3 = kkk.ntri
    zeta3 = kkk.zeta
    np.testing.assert_array_equal(ntri3, ntri1)
    np.testing.assert_array_equal(zeta3, zeta1)

    # Check running as a cross-correlation
    save_cat.unload()
    t0 = time.time()
    s0 = hp.heap().size if hp else 0
    kkk.process(save_cat, save_cat, save_cat, low_mem=True)
    t1 = time.time()
    s1 = hp.heap().size if hp else 0
    print('lomem 3: ',s1, t1-t0, s1-s0)
    assert s1-s0 < lomem
    ntri4 = kkk.ntri
    zeta4 = kkk.zeta
    np.testing.assert_array_equal(ntri4, ntri1)
    np.testing.assert_array_equal(zeta4, zeta1)


if __name__ == '__main__':
    test_brute_jk()
    test_finalize_false()
    test_lowmem
