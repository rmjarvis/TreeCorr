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
import treecorr
import os
import coord
import time
import sys

from test_helper import get_from_wiki, assert_raises, timer

@timer
def test_nn_direct_rperp():
    # This is the same as test_nn.py:test_direct_3d, but using the perpendicular distance metric

    ngal = 100
    s = 10.
    rng = np.random.RandomState(8675309)
    x1 = rng.normal(312, s, (ngal,) )
    y1 = rng.normal(728, s, (ngal,) )
    z1 = rng.normal(-932, s, (ngal,) )
    r1 = np.sqrt( x1*x1 + y1*y1 + z1*z1 )
    dec1 = np.arcsin(z1/r1)
    ra1 = np.arctan2(y1,x1)
    cat1 = treecorr.Catalog(ra=ra1, dec=dec1, r=r1, ra_units='rad', dec_units='rad')

    x2 = rng.normal(312, s, (ngal,) )
    y2 = rng.normal(728, s, (ngal,) )
    z2 = rng.normal(-932, s, (ngal,) )
    r2 = np.sqrt( x2*x2 + y2*y2 + z2*z2 )
    dec2 = np.arcsin(z2/r2)
    ra2 = np.arctan2(y2,x2)
    cat2 = treecorr.Catalog(ra=ra2, dec=dec2, r=r2, ra_units='rad', dec_units='rad')

    min_sep = 1.
    max_sep = 50.
    nbins = 50
    dd = treecorr.NNCorrelation(min_sep=min_sep, max_sep=max_sep, nbins=nbins, brute=True)
    dd.process(cat1, cat2, metric='FisherRperp')
    print('dd.npairs = ',dd.npairs)

    log_min_sep = np.log(min_sep)
    log_max_sep = np.log(max_sep)
    true_npairs = np.zeros(nbins)
    bin_size = (log_max_sep - log_min_sep) / nbins
    for i in range(ngal):
        for j in range(ngal):
            rsq = (x1[i]-x2[j])**2 + (y1[i]-y2[j])**2 + (z1[i]-z2[j])**2
            Lsq = ((x1[i]+x2[j])**2 + (y1[i]+y2[j])**2 + (z1[i]+z2[j])**2) / 4.
            rpar = abs(r1[i]**2-r2[j]**2) / (2.*np.sqrt(Lsq))
            rsq -= rpar**2
            logr = 0.5 * np.log(rsq)
            k = int(np.floor( (logr-log_min_sep) / bin_size ))
            if k < 0: continue
            if k >= nbins: continue
            true_npairs[k] += 1

    print('true_npairs = ',true_npairs)
    print('diff = ',dd.npairs - true_npairs)
    np.testing.assert_array_equal(dd.npairs, true_npairs)

    # Can also specify coords directly as x,y,z
    cat1 = treecorr.Catalog(x=x1, y=y1, z=z1)
    cat2 = treecorr.Catalog(x=x2, y=y2, z=z2)
    dd.process(cat1, cat2, metric='FisherRperp')
    np.testing.assert_array_equal(dd.npairs, true_npairs)

    # Rperp is (by default) an alias for FisherRperp
    dd.process(cat1, cat2, metric='Rperp')
    np.testing.assert_array_equal(dd.npairs, true_npairs)


@timer
def test_nn_direct_oldrperp():
    # This is the same as the above test, but using the old perpendicular distance metric

    ngal = 100
    s = 10.
    rng = np.random.RandomState(8675309)
    x1 = rng.normal(312, s, (ngal,) )
    y1 = rng.normal(728, s, (ngal,) )
    z1 = rng.normal(-932, s, (ngal,) )
    r1 = np.sqrt( x1*x1 + y1*y1 + z1*z1 )
    dec1 = np.arcsin(z1/r1)
    ra1 = np.arctan2(y1,x1)
    cat1 = treecorr.Catalog(ra=ra1, dec=dec1, r=r1, ra_units='rad', dec_units='rad')

    x2 = rng.normal(312, s, (ngal,) )
    y2 = rng.normal(728, s, (ngal,) )
    z2 = rng.normal(-932, s, (ngal,) )
    r2 = np.sqrt( x2*x2 + y2*y2 + z2*z2 )
    dec2 = np.arcsin(z2/r2)
    ra2 = np.arctan2(y2,x2)
    cat2 = treecorr.Catalog(ra=ra2, dec=dec2, r=r2, ra_units='rad', dec_units='rad')

    min_sep = 1.
    max_sep = 50.
    nbins = 50
    dd = treecorr.NNCorrelation(min_sep=min_sep, max_sep=max_sep, nbins=nbins, brute=True)
    dd.process(cat1, cat2, metric='OldRperp')
    print('dd.npairs = ',dd.npairs)

    log_min_sep = np.log(min_sep)
    log_max_sep = np.log(max_sep)
    true_npairs = np.zeros(nbins)
    bin_size = (log_max_sep - log_min_sep) / nbins
    for i in range(ngal):
        for j in range(ngal):
            rsq = (x1[i]-x2[j])**2 + (y1[i]-y2[j])**2 + (z1[i]-z2[j])**2
            rsq -= (r1[i] - r2[j])**2
            logr = 0.5 * np.log(rsq)
            k = int(np.floor( (logr-log_min_sep) / bin_size ))
            if k < 0: continue
            if k >= nbins: continue
            true_npairs[k] += 1

    print('true_npairs = ',true_npairs)
    print('diff = ',dd.npairs - true_npairs)
    np.testing.assert_array_equal(dd.npairs, true_npairs)

    # Can also specify coords directly as x,y,z
    cat1 = treecorr.Catalog(x=x1, y=y1, z=z1)
    cat2 = treecorr.Catalog(x=x2, y=y2, z=z2)
    dd.process(cat1, cat2, metric='OldRperp')
    np.testing.assert_array_equal(dd.npairs, true_npairs)

    # If we set Rperp_alias = 'OldRperp', we can use Rperp.
    # Use mock for this
    if sys.version_info < (3,): return  # mock only available on python 3
    from unittest import mock
    with mock.patch('treecorr.util.Rperp_alias', 'OldRperp'):
        dd.process(cat1, cat2, metric='Rperp')
    np.testing.assert_array_equal(dd.npairs, true_npairs)


@timer
def test_nn_direct_rlens():
    # This is the same as the above test, but using the Rlens distance metric

    ngal = 100
    s = 10.
    rng = np.random.RandomState(8675309)
    x1 = rng.normal(312, s, (ngal,) )
    y1 = rng.normal(728, s, (ngal,) )
    z1 = rng.normal(-932, s, (ngal,) )
    r1 = np.sqrt( x1*x1 + y1*y1 + z1*z1 )
    dec1 = np.arcsin(z1/r1)
    ra1 = np.arctan2(y1,x1)
    cat1 = treecorr.Catalog(ra=ra1, dec=dec1, r=r1, ra_units='rad', dec_units='rad')

    x2 = rng.normal(312, s, (ngal,) )
    y2 = rng.normal(728, s, (ngal,) )
    z2 = rng.normal(-932, s, (ngal,) )
    r2 = np.sqrt( x2*x2 + y2*y2 + z2*z2 )
    dec2 = np.arcsin(z2/r2)
    ra2 = np.arctan2(y2,x2)
    cat2 = treecorr.Catalog(ra=ra2, dec=dec2, r=r2, ra_units='rad', dec_units='rad')

    min_sep = 1.
    max_sep = 50.
    nbins = 50
    dd = treecorr.NNCorrelation(min_sep=min_sep, max_sep=max_sep, nbins=nbins, brute=True)
    dd.process(cat1, cat2, metric='Rlens')
    print('dd.npairs = ',dd.npairs)

    log_min_sep = np.log(min_sep)
    log_max_sep = np.log(max_sep)
    true_npairs = np.zeros(nbins)
    bin_size = (log_max_sep - log_min_sep) / nbins
    for i in range(ngal):
        for j in range(ngal):
            # L = |r1| sin(theta)
            #   = |r1 x r2| / |r2|
            xcross = y1[i] * z2[j] - z1[i] * y2[j]
            ycross = z1[i] * x2[j] - x1[i] * z2[j]
            zcross = x1[i] * y2[j] - y1[i] * x2[j]
            Rlens = np.sqrt(xcross**2 + ycross**2 + zcross**2) / r2[j]
            logr = np.log(Rlens)
            k = int(np.floor( (logr-log_min_sep) / bin_size ))
            if k < 0: continue
            if k >= nbins: continue
            true_npairs[k] += 1

    print('true_npairs = ',true_npairs)
    print('diff = ',dd.npairs - true_npairs)
    np.testing.assert_array_equal(dd.npairs, true_npairs)

    # The distance is only dependent on r for cat1, so if you don't know r for cat2, that's ok.
    cat2 = treecorr.Catalog(ra=ra2, dec=dec2, ra_units='rad', dec_units='rad')
    dd.process(cat1, cat2, metric='Rlens')
    print('no r2: dd.npairs = ',dd.npairs)
    print('diff = ',dd.npairs - true_npairs)
    np.testing.assert_array_equal(dd.npairs, true_npairs)

    # Can also specify coords directly as x,y,z
    cat1 = treecorr.Catalog(x=x1, y=y1, z=z1)
    cat2 = treecorr.Catalog(x=x2, y=y2, z=z2)
    dd.process(cat1, cat2, metric='Rlens')
    print('xyz: dd.npairs = ',dd.npairs)
    print('diff = ',dd.npairs - true_npairs)
    np.testing.assert_array_equal(dd.npairs, true_npairs)


@timer
def test_nk_patch_rlens():
    # This is based on a bug report by Marina Ricci that Rlens with patches failed when
    # one catalog used ra, dec, r, and the other used just ra, dec.

    ngal = 200
    s = 10.
    rng = np.random.RandomState(8675309)
    x1 = rng.normal(312, s, (ngal,) )
    y1 = rng.normal(728, s, (ngal,) )
    z1 = rng.normal(-932, s, (ngal,) )
    r1 = np.sqrt( x1*x1 + y1*y1 + z1*z1 )
    dec1 = np.arcsin(z1/r1)
    ra1 = np.arctan2(y1,x1)
    cat1 = treecorr.Catalog(ra=ra1, dec=dec1, r=r1, ra_units='rad', dec_units='rad')

    x2 = rng.normal(312, s, (ngal,) )
    y2 = rng.normal(728, s, (ngal,) )
    z2 = rng.normal(-932, s, (ngal,) )
    r2 = np.sqrt( x2*x2 + y2*y2 + z2*z2 )
    k2 = rng.uniform(0.98, 1.03, (ngal,) )
    dec2 = np.arcsin(z2/r2)
    ra2 = np.arctan2(y2,x2)
    cat2 = treecorr.Catalog(ra=ra2, dec=dec2, k=k2, ra_units='rad', dec_units='rad')

    min_sep = 1.
    max_sep = 50.
    nbins = 50
    nk = treecorr.NKCorrelation(min_sep=min_sep, max_sep=max_sep, nbins=nbins, brute=True)
    nk.process(cat1, cat2, metric='Rlens')
    print('nk.npairs = ',nk.npairs)

    log_min_sep = np.log(min_sep)
    log_max_sep = np.log(max_sep)
    true_npairs = np.zeros(nbins)
    true_kappa = np.zeros(nbins)
    bin_size = (log_max_sep - log_min_sep) / nbins
    for i in range(ngal):
        for j in range(ngal):
            # L = |r1| sin(theta)
            #   = |r1 x r2| / |r2|
            xcross = y1[i] * z2[j] - z1[i] * y2[j]
            ycross = z1[i] * x2[j] - x1[i] * z2[j]
            zcross = x1[i] * y2[j] - y1[i] * x2[j]
            Rlens = np.sqrt(xcross**2 + ycross**2 + zcross**2) / r2[j]
            logr = np.log(Rlens)
            k = int(np.floor( (logr-log_min_sep) / bin_size ))
            if k < 0: continue
            if k >= nbins: continue
            true_npairs[k] += 1
            true_kappa[k] += k2[j]
    true_kappa /= true_npairs

    print('true_npairs = ',true_npairs)
    print('diff = ',nk.npairs - true_npairs)
    np.testing.assert_array_equal(nk.npairs, true_npairs)
    print('nk.xi = ',nk.xi)
    print('true_kappa = ',true_kappa)
    print('diff = ',nk.xi - true_kappa)
    np.testing.assert_allclose(nk.xi, true_kappa)

    # This version failed on v4.2.3.
    cat1p = treecorr.Catalog(ra=ra1, dec=dec1, r=r1, ra_units='rad', dec_units='rad', npatch=4)
    cat2p = treecorr.Catalog(ra=ra2, dec=dec2, k=k2, ra_units='rad', dec_units='rad',
                             patch_centers=cat1p.patch_centers)
    nk2 = treecorr.NKCorrelation(min_sep=min_sep, max_sep=max_sep, nbins=nbins, bin_slop=0)
    nk2.process(cat1p, cat2p, metric='Rlens')
    print('nk2.npairs = ',nk2.npairs)
    print('nk2.xi = ',nk2.xi)
    np.testing.assert_array_equal(nk2.npairs, true_npairs)
    np.testing.assert_allclose(nk2.xi, true_kappa)


@timer
def test_rperp_minmax():
    """This test is based on a bug report from Erika Wagoner where the lowest bins were
    getting spuriously high w(rp) values.  It stemmed from a subtlety about how large
    rp can be compared to minsep.  The maximum rp is more than just rp + s1 + s2.
    So this test checks that when the min and max are expanded a bit, the number of pairs
    doesn't change much in the bins that used to be the min and max.
    """
    # Just use Erika's files for data and rand.
    config = {
        'ra_col' : 1,
        'dec_col' : 2,
        'ra_units' : 'deg',
        'dec_units' : 'deg',
        'r_col' : 3,
        'min_sep' : 20,
        'bin_size' : 0.036652,
        'nbins' : 50,
        'verbose' : 1
    }

    # Speed up for nosetests runs
    if __name__ != "__main__":
        config['nbins'] = 5
        config['bin_size'] = 0.1
        config['last_row'] = 30000  # Full catalog has 100,000 objects

    get_from_wiki('nn_perp_data.dat')
    dcat = treecorr.Catalog('data/nn_perp_data.dat', config)

    dd1 = treecorr.NNCorrelation(config)
    dd1.process(dcat, metric='OldRperp')

    lower_min_sep = config['min_sep'] * np.exp(-2.*config['bin_size'])
    more_nbins = config['nbins'] + 4
    dd2 = treecorr.NNCorrelation(config, min_sep=lower_min_sep, nbins=more_nbins)
    dd2.process(dcat, metric='OldRperp')

    print('dd1 npairs = ',dd1.npairs)
    print('dd2 npairs = ',dd2.npairs[2:-2])
    # First a basic sanity check.  The values not near the edge should be identical.
    np.testing.assert_equal(dd1.npairs[2:-2], dd2.npairs[4:-4])
    # The edge bins may differ slightly from the binning approximations (bin_slop and such),
    # but the differences should be very small.  (When Erika reported the problem, the differences
    # were a few percent, which ended up making a bit difference in the correlation function.)
    np.testing.assert_allclose(dd1.npairs, dd2.npairs[2:-2], rtol=1.e-6)

    if __name__ == '__main__':
        # If we're running from the command line, go ahead and finish the calculation
        # This catalog has 10^6 objects, which takes quite a while.  I should really investigate
        # how to speed up the Rperp distance calculation.  Probably by having a faster over-
        # and under-estimate first, and then only do the full calculation when it seems like we
        # will actually need it.
        # Anyway, until then, let's not take forever by using last_row=200000
        get_from_wiki('nn_perp_rand.dat')
        rcat = treecorr.Catalog('data/nn_perp_rand.dat', config, last_row=200000)

        rr1 = treecorr.NNCorrelation(config)
        rr1.process(rcat, metric='OldRperp')
        rr2 = treecorr.NNCorrelation(config, min_sep=lower_min_sep, nbins=more_nbins)
        rr2.process(rcat, metric='OldRperp')
        print('rr1 npairs = ',rr1.npairs)
        print('rr2 npairs = ',rr2.npairs[2:-2])
        np.testing.assert_allclose(rr1.npairs, rr2.npairs[2:-2], rtol=1.e-6)

        dr1 = treecorr.NNCorrelation(config)
        dr1.process(dcat, rcat, metric='OldRperp')
        dr2 = treecorr.NNCorrelation(config, min_sep=lower_min_sep, nbins=more_nbins)
        dr2.process(dcat, rcat, metric='OldRperp')
        print('dr1 npairs = ',dr1.npairs)
        print('dr2 npairs = ',dr2.npairs[2:-2])
        np.testing.assert_allclose(dr1.npairs, dr2.npairs[2:-2], rtol=1.e-6)

        xi1, varxi1 = dd1.calculateXi(rr1, dr1)
        xi2, varxi2 = dd2.calculateXi(rr2, dr2)
        print('xi1 = ',xi1)
        print('xi2 = ',xi2[2:-2])
        np.testing.assert_allclose(xi1, xi2[2:-2], rtol=1.e-6)

    # Also check the new Rperp metric
    dd1 = treecorr.NNCorrelation(config)
    dd1.process(dcat, metric='FisherRperp')

    lower_min_sep = config['min_sep'] * np.exp(-2.*config['bin_size'])
    more_nbins = config['nbins'] + 4
    dd2 = treecorr.NNCorrelation(config, min_sep=lower_min_sep, nbins=more_nbins)
    dd2.process(dcat, metric='FisherRperp')

    print('dd1 npairs = ',dd1.npairs)
    print('dd2 npairs = ',dd2.npairs[2:-2])
    # First a basic sanity check.  The values not near the edge should be identical.
    np.testing.assert_equal(dd1.npairs[2:-2], dd2.npairs[4:-4])
    # The edge bins may differ slightly from the binning approximations (bin_slop and such),
    # but the differences should be very small.  (When Erika reported the problem, the differences
    # were a few percent, which ended up making a bit difference in the correlation function.)
    np.testing.assert_allclose(dd1.npairs, dd2.npairs[2:-2], rtol=1.e-6)


@timer
def test_ng_rlens():
    # Same test_ng.py:test_ng, except use R_lens for separation.
    # Use gamma_t(r) = gamma0 exp(-R^2/2R0^2) around a bunch of foreground lenses.

    nlens = 100
    nsource = 200000
    gamma0 = 0.05
    R0 = 10.
    L = 50. * R0
    rng = np.random.RandomState(8675309)
    xl = (rng.random_sample(nlens)-0.5) * L  # -250 < x < 250
    zl = (rng.random_sample(nlens)-0.5) * L  # -250 < y < 250
    yl = rng.random_sample(nlens) * 4*L + 10*L  # 5000 < z < 7000
    rl = np.sqrt(xl**2 + yl**2 + zl**2)
    xs = (rng.random_sample(nsource)-0.5) * L
    zs = (rng.random_sample(nsource)-0.5) * L
    ys = rng.random_sample(nsource) * 8*L + 160*L  # 80000 < z < 84000
    rs = np.sqrt(xs**2 + ys**2 + zs**2)
    g1 = np.zeros( (nsource,) )
    g2 = np.zeros( (nsource,) )
    bin_size = 0.1
    # min_sep is set so the first bin doesn't have 0 pairs.
    min_sep = 1.3*R0
    # max_sep can't be too large, since the measured value starts to have shape noise for larger
    # values of separation.  We're not adding any shape noise directly, but the shear from other
    # lenses is effectively a shape noise, and that comes to dominate the measurement above ~4R0.
    max_sep = 3.5*R0
    nbins = int(np.ceil(np.log(max_sep/min_sep)/bin_size))
    bs = np.log(max_sep/min_sep)/nbins
    true_gt = np.zeros( (nbins,) )
    true_npairs = np.zeros((nbins,), dtype=int)
    print('Making shear vectors')
    for x,y,z,r in zip(xl,yl,zl,rl):
        # Rlens = |r1 x r2| / |r2|
        xcross = ys * z - zs * y
        ycross = zs * x - xs * z
        zcross = xs * y - ys * x
        Rlens = np.sqrt(xcross**2 + ycross**2 + zcross**2) / rs

        gammat = gamma0 * np.exp(-0.5*Rlens**2/R0**2)
        # For the rotation, approximate that the x,z coords are approx the perpendicular plane.
        # So just normalize back to the unit sphere and do the 2d projection calculation.
        # It's not exactly right, but it should be good enough for this unit test.
        dx = xs/rs-x/r
        dz = zs/rs-z/r
        drsq = dx**2 + dz**2
        g1 += -gammat * (dx**2-dz**2)/drsq
        g2 += -gammat * (2.*dx*dz)/drsq
        index = np.floor( np.log(Rlens/min_sep) / bs).astype(int)
        mask = (index >= 0) & (index < nbins)
        np.add.at(true_gt, index[mask], gammat[mask])
        np.add.at(true_npairs, index[mask], 1)
    true_gt /= true_npairs

    # Start with brute force.  With only 100 lenses, this still runs very fast.
    lens_cat = treecorr.Catalog(x=xl, y=yl, z=zl)
    source_cat = treecorr.Catalog(x=xs, y=ys, z=zs, g1=g1, g2=g2)
    ng0 = treecorr.NGCorrelation(bin_size=bin_size, min_sep=min_sep, max_sep=max_sep, verbose=1,
                                 metric='Rlens', brute=True)
    ng0.process(lens_cat, source_cat)

    Rlens = ng0.meanr
    theory_gt = gamma0 * np.exp(-0.5*Rlens**2/R0**2)

    print('Results with brute force:')
    print('ng.npairs = ',ng0.npairs)
    print('true_npairs = ',true_npairs)
    print('ng.xi = ',ng0.xi)
    print('true_gammat = ',true_gt)
    print('ratio = ',ng0.xi / true_gt)
    print('diff = ',ng0.xi - true_gt)
    print('max diff = ',max(abs(ng0.xi - true_gt)))
    print('ng.xi_im = ',ng0.xi_im)
    np.testing.assert_allclose(ng0.xi, true_gt, rtol=1.e-3)
    np.testing.assert_allclose(ng0.xi_im, 0, atol=1.e-6)

    print('ng.xi = ',ng0.xi)
    print('theory_gammat = ',theory_gt)
    print('ratio = ',ng0.xi / theory_gt)
    print('diff = ',ng0.xi - theory_gt)
    print('max diff = ',max(abs(ng0.xi - theory_gt)))
    np.testing.assert_allclose(ng0.xi, theory_gt, rtol=0.1)

    # Now use a more normal value for bin_slop.
    ng1 = treecorr.NGCorrelation(bin_size=bin_size, min_sep=min_sep, max_sep=max_sep, verbose=1,
                                 metric='Rlens', bin_slop=0.5)
    ng1.process(lens_cat, source_cat)
    Rlens = ng1.meanr
    theory_gt = gamma0 * np.exp(-0.5*Rlens**2/R0**2)

    print('Results with bin_slop = 0.5')
    print('ng.npairs = ',ng1.npairs)
    print('ng.xi = ',ng1.xi)
    print('theory_gammat = ',theory_gt)
    print('ratio = ',ng1.xi / theory_gt)
    print('diff = ',ng1.xi - theory_gt)
    print('max diff = ',max(abs(ng1.xi - theory_gt)))
    print('ng.xi_im = ',ng1.xi_im)
    np.testing.assert_allclose(ng1.xi, true_gt, rtol=0.02)
    np.testing.assert_allclose(ng1.xi_im, 0, atol=1.e-4)

    # Check that we get the same result using the corr2 function:
    lens_cat.write(os.path.join('data','ng_rlens_lens.dat'))
    source_cat.write(os.path.join('data','ng_rlens_source.dat'))
    config = treecorr.read_config('configs/ng_rlens.yaml')
    config['verbose'] = 0
    treecorr.corr2(config)
    corr2_output = np.genfromtxt(os.path.join('output','ng_rlens.out'), names=True,
                                 skip_header=1)
    print('ng.xi = ',ng1.xi)
    print('from corr2 output = ',corr2_output['gamT'])
    print('ratio = ',corr2_output['gamT']/ng1.xi)
    print('diff = ',corr2_output['gamT']-ng1.xi)
    np.testing.assert_allclose(corr2_output['gamT'], ng1.xi, rtol=1.e-3)
    np.testing.assert_allclose(corr2_output['gamX'], ng1.xi_im, rtol=1.e-3)

    # Repeat with the sources being given as RA/Dec only.
    ral, decl = coord.CelestialCoord.xyz_to_radec(xl,yl,zl)
    ras, decs = coord.CelestialCoord.xyz_to_radec(xs,ys,zs)
    lens_cat = treecorr.Catalog(ra=ral, dec=decl, ra_units='radians', dec_units='radians', r=rl)
    source_cat = treecorr.Catalog(ra=ras, dec=decs, ra_units='radians', dec_units='radians',
                                  g1=g1, g2=g2)

    # Again, start with brute force.
    # This version should be identical to the 3D version.  When bin_slop = 0, it won't be
    # exactly identical, since the tree construction will have different decisions along the
    # way (since everything is at the same radius here), but the results are consistent.
    ng0s = treecorr.NGCorrelation(bin_size=bin_size, min_sep=min_sep, max_sep=max_sep, verbose=1,
                                  metric='Rlens', brute=True)
    ng0s.process(lens_cat, source_cat)

    Rlens = ng0s.meanr
    theory_gt = gamma0 * np.exp(-0.5*Rlens**2/R0**2)

    print('Results when sources have no radius information, first brute force')
    print('ng.npairs = ',ng0s.npairs)
    print('true_npairs = ',true_npairs)
    print('ng.xi = ',ng0s.xi)
    print('true_gammat = ',true_gt)
    print('ratio = ',ng0s.xi / true_gt)
    print('diff = ',ng0s.xi - true_gt)
    print('max diff = ',max(abs(ng0s.xi - true_gt)))
    print('ng.xi_im = ',ng0s.xi_im)
    np.testing.assert_allclose(ng0s.xi, true_gt, rtol=1.e-4)
    np.testing.assert_allclose(ng0s.xi_im, 0, atol=1.e-5)

    print('ng.xi = ',ng0s.xi)
    print('theory_gammat = ',theory_gt)
    print('ratio = ',ng0s.xi / theory_gt)
    print('diff = ',ng0s.xi - theory_gt)
    print('max diff = ',max(abs(ng0s.xi - theory_gt)))
    np.testing.assert_allclose(ng0s.xi, theory_gt, rtol=0.05)

    np.testing.assert_allclose(ng0s.xi, ng0.xi, rtol=1.e-6)
    np.testing.assert_allclose(ng0s.xi_im, 0, atol=1.e-6)
    np.testing.assert_allclose(ng0s.npairs, ng0.npairs, atol=1.e-6)

    # Now use a more normal value for bin_slop.
    ng1s = treecorr.NGCorrelation(bin_size=bin_size, min_sep=min_sep, max_sep=max_sep, verbose=1,
                                  metric='Rlens', bin_slop=0.3)
    ng1s.process(lens_cat, source_cat)
    Rlens = ng1s.meanr
    theory_gt = gamma0 * np.exp(-0.5*Rlens**2/R0**2)

    print('Results with bin_slop = 0.3')
    print('ng.npairs = ',ng1s.npairs)
    print('ng.xi = ',ng1s.xi)
    print('theory_gammat = ',theory_gt)
    print('ratio = ',ng1s.xi / theory_gt)
    print('diff = ',ng1s.xi - theory_gt)
    print('max diff = ',max(abs(ng1s.xi - theory_gt)))
    print('ng.xi_im = ',ng1s.xi_im)
    np.testing.assert_allclose(ng1s.xi, theory_gt, rtol=0.1)
    np.testing.assert_allclose(ng1s.xi_im, 0, atol=1.e-5)


@timer
def test_ng_rlens_bkg():
    # Same as above, except limit the sources to be in the background of the lens.

    nlens = 100
    nsource = 100000
    gamma0 = 0.05
    R0 = 10.
    L = 50. * R0
    rng = np.random.RandomState(8675309)
    xl = (rng.random_sample(nlens)-0.5) * L  # -250 < x < 250
    zl = (rng.random_sample(nlens)-0.5) * L  # -250 < y < 250
    yl = rng.random_sample(nlens) * 4*L + 10*L  # 5000 < z < 7000
    rl = np.sqrt(xl**2 + yl**2 + zl**2)
    xs = (rng.random_sample(nsource)-0.5) * L
    zs = (rng.random_sample(nsource)-0.5) * L
    ys = rng.random_sample(nsource) * 12*L + 8*L  # 4000 < z < 10000
    rs = np.sqrt(xs**2 + ys**2 + zs**2)
    print('xl = ',np.min(xl),np.max(xl))
    print('yl = ',np.min(yl),np.max(yl))
    print('zl = ',np.min(zl),np.max(zl))
    print('xs = ',np.min(xs),np.max(xs))
    print('ys = ',np.min(ys),np.max(ys))
    print('zs = ',np.min(zs),np.max(zs))
    g1 = np.zeros( (nsource,) )
    g2 = np.zeros( (nsource,) )
    bin_size = 0.1
    # min_sep is set so the first bin doesn't have 0 pairs.
    min_sep = 1.3*R0
    # max_sep can't be too large, since the measured value starts to have shape noise for larger
    # values of separation.  We're not adding any shape noise directly, but the shear from other
    # lenses is effectively a shape noise, and that comes to dominate the measurement above ~4R0.
    max_sep = 2.5*R0
    nbins = int(np.ceil(np.log(max_sep/min_sep)/bin_size))
    bs = np.log(max_sep/min_sep)/nbins
    print('Making shear vectors')
    for x,y,z,r in zip(xl,yl,zl,rl):
        # This time, only give the true shear to the background galaxies.
        bkg = (rs > r)

        # Rlens = |r1 x r2| / |r2|
        xcross = ys[bkg] * z - zs[bkg] * y
        ycross = zs[bkg] * x - xs[bkg] * z
        zcross = xs[bkg] * y - ys[bkg] * x
        Rlens = np.sqrt(xcross**2 + ycross**2 + zcross**2) / (rs[bkg])

        gammat = gamma0 * np.exp(-0.5*Rlens**2/R0**2)
        # For the rotation, approximate that the x,z coords are approx the perpendicular plane.
        # So just normalize back to the unit sphere and do the 2d projection calculation.
        # It's not exactly right, but it should be good enough for this unit test.
        dx = (xs/rs)[bkg]-x/r
        dz = (zs/rs)[bkg]-z/r
        drsq = dx**2 + dz**2

        g1[bkg] += -gammat * (dx**2-dz**2)/drsq
        g2[bkg] += -gammat * (2.*dx*dz)/drsq

    # Slight subtlety in this test vs the previous one.  We need to build up the full g1,g2
    # arrays first before calculating the true_gt value, since we need to include the background
    # galaxies for each lens regardless of whether they had signal or not.
    true_gt = np.zeros( (nbins,) )
    true_npairs = np.zeros((nbins,), dtype=int)
    # Along the way, do the same test for Arc metric.
    min_sep_arc = 10   # arcmin
    max_sep_arc = 200
    min_sep_arc_rad = min_sep_arc * coord.arcmin / coord.radians
    nbins_arc = int(np.ceil(np.log(max_sep_arc/min_sep_arc)/bin_size))
    bs_arc = np.log(max_sep_arc/min_sep_arc)/nbins_arc
    true_gt_arc = np.zeros( (nbins_arc,) )
    true_npairs_arc = np.zeros((nbins_arc,), dtype=int)
    for x,y,z,r in zip(xl,yl,zl,rl):
        # Rlens = |r1 x r2| / |r2|
        xcross = ys * z - zs * y
        ycross = zs * x - xs * z
        zcross = xs * y - ys * x
        Rlens = np.sqrt(xcross**2 + ycross**2 + zcross**2) / rs
        dx = xs/rs-x/r
        dz = zs/rs-z/r
        drsq = dx**2 + dz**2
        gt = -g1 * (dx**2-dz**2)/drsq - g2 * (2.*dx*dz)/drsq
        bkg = (rs > r)
        index = np.floor( np.log(Rlens/min_sep) / bs).astype(int)
        mask = (index >= 0) & (index < nbins) & bkg
        np.add.at(true_gt, index[mask], gt[mask])
        np.add.at(true_npairs, index[mask], 1)

        # Arc bins by theta, which is arcsin(Rlens / r)
        theta = np.arcsin(Rlens / r)
        index = np.floor( np.log(theta / min_sep_arc_rad) / bs_arc).astype(int)
        mask = (index >= 0) & (index < nbins_arc) & bkg
        np.add.at(true_gt_arc, index[mask], gt[mask])
        np.add.at(true_npairs_arc, index[mask], 1)

    true_gt /= true_npairs
    true_gt_arc /= true_npairs_arc

    # Start with brute force.  With only 100 lenses, this still runs very fast.
    lens_cat = treecorr.Catalog(x=xl, y=yl, z=zl)
    source_cat = treecorr.Catalog(x=xs, y=ys, z=zs, g1=g1, g2=g2)
    ng0 = treecorr.NGCorrelation(bin_size=bin_size, min_sep=min_sep, max_sep=max_sep, verbose=1,
                                 metric='Rlens', brute=True, min_rpar=0)
    ng0.process(lens_cat, source_cat)

    Rlens = ng0.meanr
    theory_gt = gamma0 * np.exp(-0.5*Rlens**2/R0**2)

    print('Results with brute=True:')
    print('ng.npairs = ',ng0.npairs)
    print('true_npairs = ',true_npairs)
    print('ng.xi = ',ng0.xi)
    print('true_gammat = ',true_gt)
    print('ratio = ',ng0.xi / true_gt)
    print('diff = ',ng0.xi - true_gt)
    print('max diff = ',max(abs(ng0.xi - true_gt)))
    np.testing.assert_allclose(ng0.xi, true_gt, rtol=1.e-3)

    print('ng.xi = ',ng0.xi)
    print('theory_gammat = ',theory_gt)
    print('ratio = ',ng0.xi / theory_gt)
    print('diff = ',ng0.xi - theory_gt)
    print('max diff = ',max(abs(ng0.xi - theory_gt)))
    print('ng.xi_im = ',ng0.xi_im)
    np.testing.assert_allclose(ng0.xi, theory_gt, rtol=0.5)
    np.testing.assert_allclose(ng0.xi, theory_gt, atol=1.e-3)
    np.testing.assert_allclose(ng0.xi_im, 0, atol=1.e-3)

    # Without min_rpar, this should fail.
    lens_cat = treecorr.Catalog(x=xl, y=yl, z=zl)
    source_cat = treecorr.Catalog(x=xs, y=ys, z=zs, g1=g1, g2=g2)
    ng0 = treecorr.NGCorrelation(bin_size=bin_size, min_sep=min_sep, max_sep=max_sep, verbose=1,
                                 metric='Rlens', brute=True)
    ng0.process(lens_cat, source_cat)
    Rlens = ng0.meanr

    print('Results without min_rpar')
    print('ng.xi = ',ng0.xi)
    print('true_gammat = ',true_gt)
    print('max diff = ',max(abs(ng0.xi - true_gt)))
    assert max(abs(ng0.xi - true_gt)) > 5.e-3

    # Now use a more normal value for bin_slop.
    ng1 = treecorr.NGCorrelation(bin_size=bin_size, min_sep=min_sep, max_sep=max_sep, verbose=1,
                                 metric='Rlens', bin_slop=0.5, min_rpar=0)
    ng1.process(lens_cat, source_cat)
    Rlens = ng1.meanr
    theory_gt = gamma0 * np.exp(-0.5*Rlens**2/R0**2)

    print('Results with bin_slop = 0.5')
    print('ng.npairs = ',ng1.npairs)
    print('ng.xi = ',ng1.xi)
    print('theory_gammat = ',theory_gt)
    print('ratio = ',ng1.xi / theory_gt)
    print('diff = ',ng1.xi - theory_gt)
    print('max diff = ',max(abs(ng1.xi - theory_gt)))
    print('ng.xi_im = ',ng1.xi_im)
    np.testing.assert_allclose(ng1.xi, theory_gt, rtol=0.5)
    np.testing.assert_allclose(ng1.xi, theory_gt, atol=1.e-3)
    np.testing.assert_allclose(ng1.xi_im, 0, atol=1.e-3)

    # Check that we get the same result using the corr2 function:
    lens_cat.write(os.path.join('data','ng_rlens_bkg_lens.dat'))
    source_cat.write(os.path.join('data','ng_rlens_bkg_source.dat'))
    config = treecorr.read_config('configs/ng_rlens_bkg.yaml')
    config['verbose'] = 0
    treecorr.corr2(config)
    corr2_output = np.genfromtxt(os.path.join('output','ng_rlens_bkg.out'), names=True,
                                 skip_header=1)
    print('ng.xi = ',ng1.xi)
    print('from corr2 output = ',corr2_output['gamT'])
    print('ratio = ',corr2_output['gamT']/ng1.xi)
    print('diff = ',corr2_output['gamT']-ng1.xi)
    np.testing.assert_allclose(corr2_output['gamT'], ng1.xi, rtol=1.e-3)
    np.testing.assert_allclose(corr2_output['gamX'], ng1.xi_im, atol=1.e-3)

    # Repeat with Arc metric
    ng2 = treecorr.NGCorrelation(bin_size=bin_size, min_sep=min_sep_arc, max_sep=max_sep_arc,
                                 metric='Arc', brute=True, min_rpar=0, sep_units='arcmin')
    ng2.process(lens_cat, source_cat)

    print('Results with brute=True:')
    print('ng.npairs = ',ng2.npairs)
    print('true_npairs = ',true_npairs_arc)
    print('ng.xi = ',ng2.xi)
    print('true_gammat = ',true_gt_arc)
    print('ratio = ',ng2.xi / true_gt_arc)
    print('diff = ',ng2.xi - true_gt_arc)
    print('max diff = ',max(abs(ng2.xi - true_gt_arc)))
    np.testing.assert_array_equal(ng2.npairs, true_npairs_arc)
    np.testing.assert_allclose(ng2.xi, true_gt_arc, rtol=5.e-3)
    print('ng.xi_im = ',ng2.xi_im)
    print('max im = ',max(abs(ng2.xi_im)))
    np.testing.assert_allclose(ng2.xi_im, 0, atol=5.e-4)

    # Without min_rpar, this should fail.
    lens_cat = treecorr.Catalog(x=xl, y=yl, z=zl)
    source_cat = treecorr.Catalog(x=xs, y=ys, z=zs, g1=g1, g2=g2)
    ng2 = treecorr.NGCorrelation(bin_size=bin_size, min_sep=min_sep_arc, max_sep=max_sep_arc,
                                 metric='Arc', brute=True, sep_units='arcmin')
    ng2.process(lens_cat, source_cat)
    Rlens = ng2.meanr

    print('Results without min_rpar')
    print('ng.xi = ',ng2.xi)
    print('true_gammat = ',true_gt_arc)
    print('max diff = ',max(abs(ng2.xi - true_gt_arc)))
    assert max(abs(ng2.xi - true_gt_arc)) > 2.e-3

    # Now use a more normal value for bin_slop.
    ng3 = treecorr.NGCorrelation(bin_size=bin_size, min_sep=min_sep_arc, max_sep=max_sep_arc,
                                 metric='Arc', bin_slop=0.5, min_rpar=0, sep_units='arcmin')
    ng3.process(lens_cat, source_cat)

    print('Results with bin_slop = 0.5')
    print('ng.npairs = ',ng3.npairs)
    print('ng.xi = ',ng3.xi)
    print('ratio = ',ng3.xi / true_gt_arc)
    print('ng.xi_im = ',ng3.xi_im)
    np.testing.assert_allclose(ng3.xi, true_gt_arc, rtol=1.e-2, atol=5.e-5)
    np.testing.assert_allclose(ng3.xi_im, 0, atol=1.e-3)


@timer
def test_ng_rperp():
    # This test stems from a bug report by Eske Pedersen, which had been giving seg faults
    # with version 4.0.2.  So the main test here is that it doesn't give a seg fault.

    file_name = os.path.join('data','oldrperptest.dat')
    cat = treecorr.Catalog(file_name, ra_col=1, dec_col=2, r_col=3, w_col=4, g1_col=5, g2_col=6,
                           ra_units='deg',dec_units='deg')

    dmax = np.max(cat.r)-np.min(cat.r)
    # Note: many of the redshifts are equal to other redshifts, so setting max_rpar=0 would
    # be unstable to numerical differences on different machines, as 0 isn't exact, so it would
    # come out to +- 1.e-13 or so.  Use max_rpar=1.e-8 as meaning <= 0.
    ng = treecorr.NGCorrelation(nbins=10, min_sep=0.5, max_sep=60,
                                min_rpar=-dmax, max_rpar=1.e-8, bin_slop=0.01)
    ng.process(cat, cat, metric='OldRperp')

    print('OldRperp:')
    print('ng.npairs = ',repr(ng.npairs))
    print('ng.xi = ',repr(ng.xi))

    true_npairs = [  2193.,   4940.,  10792.,  21846.,  39847.,  53867.,  80555.,
                    105466., 126601.,  80360.]
    true_xi = [-0.00630174, -0.00049443,  0.00213589, -0.00138997, -0.00106433,
                0.00246789,  0.00806851,  0.00505585,  0.00719003, -0.00491967]

    np.testing.assert_allclose(ng.npairs, true_npairs, rtol=1.e-3)
    # Note: the atol=1.e-4 is only required for a few machines, including Travis and nersc.
    # Other machines match more closely.  Might be worth investigating at some point why this
    # has some platform dependence.
    np.testing.assert_allclose(ng.xi, true_xi, rtol=1.e-3, atol=1.e-4)

    # Rperp doesn't get exactly the same values, but it's similar.
    ng.process(cat, cat, metric='FisherRperp')

    print('FisherRperp:')
    print('ng.npairs = ',repr(ng.npairs))
    print('ng.xi = ',repr(ng.xi))

    true_npairs = [  2191.,   4941.,  10820.,  21857.,  39881.,  53873.,  80599.,
                   105355., 126529.,  79872.]
    true_xi = [-0.0066858, -0.0000599,  0.00175524, -0.00135803, -0.00126972,
                0.0024636,  0.00787932,  0.00486865, 0.00722583, -0.00532089]

    np.testing.assert_allclose(ng.npairs, true_npairs, rtol=1.e-3)
    np.testing.assert_allclose(ng.xi, true_xi, rtol=1.e-3)


@timer
def test_gg_rlens():
    # Similar to test_ng_rlens, but we give the lenses a shape and do a GG correlation.
    # Use gamma_t(r) = gamma0 exp(-R^2/2R0^2) around a bunch of foreground lenses.

    nlens = 100
    nsource = 200000
    gamma0 = 0.05
    R0 = 10.
    L = 50. * R0
    rng = np.random.RandomState(8675309)

    # Lenses are randomly located with random shapes.
    xl = (rng.random_sample(nlens)-0.5) * L  # -250 < x < 250
    zl = (rng.random_sample(nlens)-0.5) * L  # -250 < y < 250
    yl = rng.random_sample(nlens) * 4*L + 10*L  # 5000 < z < 7000
    rl = np.sqrt(xl**2 + yl**2 + zl**2)
    g1l = rng.normal(0., 0.1, (nlens,))
    g2l = rng.normal(0., 0.1, (nlens,))
    gl = g1l + 1j * g2l
    gl /= np.abs(gl)
    print('Made lenses')

    # For the signal, we'll do a pure quadrupole halo lens signal.  cf. test_haloellip()
    xs = (rng.random_sample(nsource)-0.5) * L
    zs = (rng.random_sample(nsource)-0.5) * L
    ys = rng.random_sample(nsource) * 8*L + 160*L  # 80000 < z < 84000
    rs = np.sqrt(xs**2 + ys**2 + zs**2)
    g1 = np.zeros( (nsource,) )
    g2 = np.zeros( (nsource,) )
    bin_size = 0.1
    # min_sep is set so the first bin doesn't have 0 pairs.
    min_sep = 1.3*R0
    # max_sep can't be too large, since the measured value starts to have shape noise for larger
    # values of separation.  We're not adding any shape noise directly, but the shear from other
    # lenses is effectively a shape noise, and that comes to dominate the measurement above ~4R0.
    max_sep = 4.*R0
    nbins = int(np.ceil(np.log(max_sep/min_sep)/bin_size))
    bs = np.log(max_sep/min_sep)/nbins  # Real bin size to use.
    true_gQ = np.zeros( (nbins,) )
    true_gCr = np.zeros( (nbins,) )
    true_gCi = np.zeros( (nbins,) )
    true_npairs = np.zeros((nbins,), dtype=int)
    print('Making shear vectors')
    for x,y,z,r,g in zip(xl,yl,zl,rl,gl):
        # Rlens = |r1 x r2| / |r2|
        xcross = ys * z - zs * y
        ycross = zs * x - xs * z
        zcross = xs * y - ys * x
        Rlens = np.sqrt(xcross**2 + ycross**2 + zcross**2) / rs

        gammaQ = gamma0 * np.exp(-0.5*Rlens**2/R0**2)

        # For the alpha angle, approximate that the x,z coords are approx the perpendicular plane.
        # So just normalize back to the unit sphere and do the 2d projection calculation.
        # It's not exactly right, but it should be good enough for this unit test.
        dx = xs/rs-x/r
        dz = zs/rs-z/r
        expialpha = dx + 1j*dz
        expialpha /= np.abs(expialpha)

        # In frame where halo is along x axis,
        #   g_source = gammaQ exp(4itheta)
        # In real frame, theta = alpha - phi, and we need to rotate the shear an extra exp(2iphi)
        #   g_source = gammaQ exp(4ialpha) exp(-2iphi)
        gQ = gammaQ * expialpha**4 * np.conj(g)
        g1 += gQ.real
        g2 += gQ.imag

        index = np.floor( np.log(Rlens/min_sep) / bs).astype(int)
        mask = (index >= 0) & (index < nbins)
        np.add.at(true_gQ, index[mask], gammaQ[mask])
        np.add.at(true_npairs, index[mask], 1)

        # We aren't intentionally making a constant term, but there will be some C signal due to
        # the finite number of pairs being rendered.  So let's figure out how much there is.
        gC = gQ * np.conj(g)
        np.add.at(true_gCr, index[mask], gC[mask].real)
        np.add.at(true_gCi, index[mask], -gC[mask].imag)

    true_gQ /= true_npairs
    true_gCr /= true_npairs
    true_gCi /= true_npairs
    print('true_gQ = ',true_gQ)
    print('true_gCr = ',true_gCr)
    print('true_gCi = ',true_gCi)

    # Start with brute force.
    # With only 100 lenses, this still runs very fast.
    lens_cat = treecorr.Catalog(x=xl, y=yl, z=zl, g1=gl.real, g2=gl.imag)
    source_cat = treecorr.Catalog(x=xs, y=ys, z=zs, g1=g1, g2=g2)
    gg0 = treecorr.GGCorrelation(bin_size=bin_size, min_sep=min_sep, max_sep=max_sep, verbose=1,
                                 metric='Rlens', brute=True)
    t0 = time.time()
    gg0.process(lens_cat, source_cat)
    t1 = time.time()

    Rlens = gg0.meanr
    theory_gQ = gamma0 * np.exp(-0.5*Rlens**2/R0**2)

    print('Results with brute force:')
    print('time = ',t1-t0)
    print('gg.npairs = ',gg0.npairs)
    print('true_npairs = ',true_npairs)
    np.testing.assert_array_equal(gg0.npairs, true_npairs)
    print('gg.xim = ',gg0.xim)
    print('true_gQ = ',true_gQ)
    print('ratio = ',gg0.xim / true_gQ)
    print('diff = ',gg0.xim - true_gQ)
    print('max diff = ',max(abs(gg0.xim - true_gQ)))
    assert max(abs(gg0.xim - true_gQ)) < 2.e-6
    print('gg.xim_im = ',gg0.xim_im)
    assert max(abs(gg0.xim_im)) < 3.e-6
    print('gg.xip = ',gg0.xip)
    print('true_gCr = ',true_gCr)
    print('diff = ',gg0.xip - true_gCr)
    print('max diff = ',max(abs(gg0.xip - true_gCr)))
    assert max(abs(gg0.xip - true_gCr)) < 2.e-6
    print('gg.xip_im = ',gg0.xip_im)
    print('true_gCi = ',true_gCi)
    print('diff = ',gg0.xip_im - true_gCi)
    print('max diff = ',max(abs(gg0.xip_im - true_gCi)))
    assert max(abs(gg0.xip_im - true_gCi)) < 2.e-6

    print('gg.xim = ',gg0.xim)
    print('theory_gammat = ',theory_gQ)
    print('ratio = ',gg0.xim / theory_gQ)
    print('diff = ',gg0.xim - theory_gQ)
    print('max diff = ',max(abs(gg0.xim - theory_gQ)))
    assert max(abs(gg0.xim - theory_gQ)) < 4.e-5

    # With bin_slop = 0, it should get the same npairs, but the shapes will be slightly off, since
    # the directions won't be exactly right.
    gg1 = treecorr.GGCorrelation(bin_size=bin_size, min_sep=min_sep, max_sep=max_sep, verbose=1,
                                 metric='Rlens', bin_slop=0)
    t0 = time.time()
    gg1.process(lens_cat, source_cat)
    t1 = time.time()

    Rlens = gg1.meanr
    theory_gQ = gamma0 * np.exp(-0.5*Rlens**2/R0**2)

    print('Results with bin_slop = 0:')
    print('time = ',t1-t0)
    print('gg.npairs = ',gg1.npairs)
    print('true_npairs = ',true_npairs)
    np.testing.assert_array_equal(gg1.npairs, true_npairs)
    print('gg.xim = ',gg1.xim)
    print('true_gQ = ',true_gQ)
    print('ratio = ',gg1.xim / true_gQ)
    print('diff = ',gg1.xim - true_gQ)
    print('max diff = ',max(abs(gg1.xim - true_gQ)))
    assert max(abs(gg1.xim - true_gQ)) < 2.e-5
    print('gg.xim_im = ',gg1.xim_im)
    assert max(abs(gg1.xim_im)) < 2.e-5
    print('gg.xip = ',gg1.xip)
    print('true_gCr = ',true_gCr)
    print('diff = ',gg1.xip - true_gCr)
    print('max diff = ',max(abs(gg1.xip - true_gCr)))
    assert max(abs(gg1.xip - true_gCr)) < 2.e-5
    print('gg.xip_im = ',gg1.xip_im)
    print('true_gCi = ',true_gCi)
    print('diff = ',gg1.xip_im - true_gCi)
    print('max diff = ',max(abs(gg1.xip_im - true_gCi)))
    assert max(abs(gg1.xip_im - true_gCi)) < 2.e-5

    print('gg.xim = ',gg1.xim)
    print('theory_gammat = ',theory_gQ)
    print('ratio = ',gg1.xim / theory_gQ)
    print('diff = ',gg1.xim - theory_gQ)
    print('max diff = ',max(abs(gg1.xim - theory_gQ)))
    assert max(abs(gg1.xim - theory_gQ)) < 4.e-5

    # Can also do brute force just on one cat or the other.
    # In this case, just cat2 is equivalent to full brute force, since nlens << nsource.
    gg1a = treecorr.GGCorrelation(bin_size=bin_size, min_sep=min_sep, max_sep=max_sep, verbose=1,
                                  metric='Rlens', bin_slop=0, brute=1)
    gg1a.process(lens_cat, source_cat)
    assert lens_cat.field.brute is True
    assert source_cat.field.brute is False
    gg1b = treecorr.GGCorrelation(bin_size=bin_size, min_sep=min_sep, max_sep=max_sep, verbose=1,
                                  metric='Rlens', bin_slop=0, brute=2)
    gg1b.process(lens_cat, source_cat)
    assert lens_cat.field.brute is False
    assert source_cat.field.brute is True

    assert max(abs(gg0.xim - true_gQ)) < 2.e-6
    assert max(abs(gg1.xim - true_gQ)) < 2.e-5
    assert max(abs(gg1a.xim - true_gQ)) < 2.e-5
    assert max(abs(gg1b.xim - true_gQ)) < 2.e-6
    assert max(abs(gg0.xip - true_gCr)) < 2.e-6
    assert max(abs(gg1.xip - true_gCr)) < 2.e-5
    assert max(abs(gg1a.xip - true_gCr)) < 2.e-5
    assert max(abs(gg1b.xip - true_gCr)) < 2.e-6

    # Now use a more normal value for bin_slop.
    gg2 = treecorr.GGCorrelation(bin_size=bin_size, min_sep=min_sep, max_sep=max_sep, verbose=1,
                                 metric='Rlens', bin_slop=0.3)
    t0 = time.time()
    gg2.process(lens_cat, source_cat)
    t1 = time.time()
    Rlens = gg2.meanr
    theory_gQ = gamma0 * np.exp(-0.5*Rlens**2/R0**2)

    print('Results with bin_slop = 0.3')
    print('time = ',t1-t0)
    print('gg.npairs = ',gg2.npairs)
    print('gg.xim = ',gg2.xim)
    print('theory_gammat = ',theory_gQ)
    print('ratio = ',gg2.xim / theory_gQ)
    print('diff = ',gg2.xim - theory_gQ)
    print('max diff = ',max(abs(gg2.xim - theory_gQ)))
    assert max(abs(gg2.xim - theory_gQ)) < 4.e-5
    print('gg.xim_im = ',gg2.xim_im)
    assert max(abs(gg2.xim_im)) < 7.e-6

    # Check that we get the same result using the corr2 function
    lens_cat.write(os.path.join('data','gg_rlens_lens.dat'))
    source_cat.write(os.path.join('data','gg_rlens_source.dat'))
    config = treecorr.read_config('configs/gg_rlens.yaml')
    config['verbose'] = 0
    treecorr.corr2(config)
    corr2_output = np.genfromtxt(os.path.join('output','gg_rlens.out'),names=True, skip_header=1)
    print('gg.xim = ',gg2.xim)
    print('from corr2 output = ',corr2_output['xim'])
    print('ratio = ',corr2_output['xim']/gg2.xim)
    print('diff = ',corr2_output['xim']-gg2.xim)
    np.testing.assert_allclose(corr2_output['xim'], gg2.xim, rtol=1.e-3)
    np.testing.assert_allclose(corr2_output['xim_im'], gg2.xim_im, rtol=1.e-3)
    np.testing.assert_allclose(corr2_output['xip'], gg2.xip, rtol=1.e-3)
    np.testing.assert_allclose(corr2_output['xip_im'], gg2.xip_im, rtol=1.e-3)

    # Repeat with the sources being given as RA/Dec only.
    ral, decl = coord.CelestialCoord.xyz_to_radec(xl,yl,zl)
    ras, decs = coord.CelestialCoord.xyz_to_radec(xs,ys,zs)
    lens_cat = treecorr.Catalog(ra=ral, dec=decl, ra_units='radians', dec_units='radians', r=rl,
                                g1=gl.real, g2=gl.imag)
    source_cat = treecorr.Catalog(ra=ras, dec=decs, ra_units='radians', dec_units='radians',
                                  g1=g1, g2=g2)

    gg0s = treecorr.GGCorrelation(bin_size=bin_size, min_sep=min_sep, max_sep=max_sep, verbose=1,
                                  metric='Rlens', brute=True)
    gg0s.process(lens_cat, source_cat)

    Rlens = gg0s.meanr
    theory_gQ = gamma0 * np.exp(-0.5*Rlens**2/R0**2)

    print('Results with brute=True')
    print('gg.npairs = ',gg0s.npairs)
    print('true_npairs = ',true_npairs)
    np.testing.assert_array_equal(gg0s.npairs, true_npairs)
    print('gg.xim = ',gg0s.xim)
    print('true_gQ = ',true_gQ)
    print('ratio = ',gg0s.xim / true_gQ)
    print('diff = ',gg0s.xim - true_gQ)
    print('max diff = ',max(abs(gg0s.xim - true_gQ)))
    assert max(abs(gg0s.xim - true_gQ)) < 2.e-6
    print('gg.xim_im = ',gg0s.xim_im)
    assert max(abs(gg0s.xim_im)) < 3.e-6
    print('gg.xip = ',gg0s.xip)
    print('true_gCr = ',true_gCr)
    print('diff = ',gg0s.xip - true_gCr)
    print('max diff = ',max(abs(gg0s.xip - true_gCr)))
    assert max(abs(gg0s.xip - true_gCr)) < 2.e-6
    print('gg.xip_im = ',gg0s.xip_im)
    print('true_gCi = ',true_gCi)
    print('diff = ',gg0s.xip_im - true_gCi)
    print('max diff = ',max(abs(gg0s.xip_im - true_gCi)))
    assert max(abs(gg0s.xip_im - true_gCi)) < 2.e-6

    print('gg.xim = ',gg0s.xim)
    print('theory_gammat = ',theory_gQ)
    print('ratio = ',gg0s.xim / theory_gQ)
    print('diff = ',gg0s.xim - theory_gQ)
    print('max diff = ',max(abs(gg0s.xim - theory_gQ)))
    assert max(abs(gg0s.xim - theory_gQ)) < 4.e-5

    # This should be identical to the 3d version, since going all the way to leaves.
    # (The next test will be different, since tree creation is different.)
    assert max(abs(gg0s.xim - gg0.xim)) < 1.e-7
    assert max(abs(gg0s.xip - gg0.xip)) < 1.e-7
    assert max(abs(gg0s.xim_im - gg0.xim_im)) < 1.e-7
    assert max(abs(gg0s.xip_im - gg0.xip_im)) < 1.e-7
    assert max(abs(gg0s.npairs - gg0.npairs)) < 1.e-7

    # Now use a more normal value for bin_slop.
    ggs2 = treecorr.GGCorrelation(bin_size=bin_size, min_sep=min_sep, max_sep=max_sep, verbose=1,
                                  metric='Rlens', bin_slop=0.3)
    ggs2.process(lens_cat, source_cat)
    Rlens = ggs2.meanr
    theory_gQ = gamma0 * np.exp(-0.5*Rlens**2/R0**2)

    print('Results with bin_slop = 0.3')
    print('gg.npairs = ',ggs2.npairs)
    print('gg.xim = ',ggs2.xim)
    print('theory_gammat = ',theory_gQ)
    print('ratio = ',ggs2.xim / theory_gQ)
    print('diff = ',ggs2.xim - theory_gQ)
    print('max diff = ',max(abs(ggs2.xim - theory_gQ)))
    # Not quite as accurate as above, since the cells that get used tend to be larger, so more
    # slop happens in the binning.
    assert max(abs(ggs2.xim - theory_gQ)) < 5.e-5
    print('gg.xim_im = ',ggs2.xim_im)
    assert max(abs(ggs2.xim_im)) < 7.e-6


@timer
def test_gg_rperp():
    # Same as above, but using Rperp.

    nlens = 100
    nsource = 100000
    gamma0 = 0.05
    R0 = 5.
    L = 100. * R0
    rng = np.random.RandomState(8675309)

    # Lenses are randomly located with random shapes.
    xl = (rng.random_sample(nlens)-0.5) * L  # -250 < x < 250
    zl = (rng.random_sample(nlens)-0.5) * L  # -250 < y < 250
    yl = rng.random_sample(nlens) * 4*L + 10*L  # 5000 < z < 7000
    rl = np.sqrt(xl**2 + yl**2 + zl**2)
    g1l = rng.normal(0., 0.1, (nlens,))
    g2l = rng.normal(0., 0.1, (nlens,))
    gl = g1l + 1j * g2l
    gl /= np.abs(gl)
    print('Made lenses')

    # For the signal, we'll do a pure quadrupole halo lens signal.  cf. test_haloellip()
    xs = (rng.random_sample(nsource)-0.5) * L
    zs = (rng.random_sample(nsource)-0.5) * L
    ys = rng.random_sample(nsource) * 8*L + 160*L  # 80000 < z < 84000
    rs = np.sqrt(xs**2 + ys**2 + zs**2)
    g1 = np.zeros( (nsource,) )
    g2 = np.zeros( (nsource,) )
    bin_size = 0.1
    # min_sep is set so the first bin doesn't have 0 pairs.
    # Both this and max_sep need to be larger than what we used for Rlens.
    min_sep = 4.5*R0
    # max_sep can't be too large, since the measured value starts to have shape noise for larger
    # values of separation.  We're not adding any shape noise directly, but the shear from other
    # lenses is effectively a shape noise, and that comes to dominate the measurement above ~12R0.
    max_sep = 12.*R0
    # Because the Rperp values are a lot larger than the Rlens values, use a larger scale radius
    # in the gaussian signal.
    R1 = 4. * R0
    nbins = int(np.ceil(np.log(max_sep/min_sep)/bin_size))
    bs = np.log(max_sep/min_sep)/nbins  # Real bin size to use.
    true_gQ = np.zeros( (nbins,) )
    true_gCr = np.zeros( (nbins,) )
    true_gCi = np.zeros( (nbins,) )
    true_npairs = np.zeros((nbins,), dtype=int)
    print('Making shear vectors')
    for x,y,z,r,g in zip(xl,yl,zl,rl,gl):
        dsq = (x-xs)**2 + (y-ys)**2 + (z-zs)**2
        Lsq = ((x+xs)**2 + (y+ys)**2 + (z+zs)**2) / 4.
        Rpar = abs(rs**2 - r**2) / (2 * np.sqrt(Lsq))
        Rperpsq = dsq - Rpar**2
        Rperp = np.sqrt(Rperpsq)
        gammaQ = gamma0 * np.exp(-0.5*Rperp**2/R1**2)

        dx = xs/rs-x/r
        dz = zs/rs-z/r
        expialpha = dx + 1j*dz
        expialpha /= np.abs(expialpha)

        gQ = gammaQ * expialpha**4 * np.conj(g)
        g1 += gQ.real
        g2 += gQ.imag

        index = np.floor( np.log(Rperp/min_sep) / bs).astype(int)
        mask = (index >= 0) & (index < nbins)
        np.add.at(true_gQ, index[mask], gammaQ[mask])
        np.add.at(true_npairs, index[mask], 1)

        gC = gQ * np.conj(g)
        np.add.at(true_gCr, index[mask], gC[mask].real)
        np.add.at(true_gCi, index[mask], -gC[mask].imag)

    true_gQ /= true_npairs
    true_gCr /= true_npairs
    true_gCi /= true_npairs
    print('true_gQ = ',true_gQ)
    print('true_gCr = ',true_gCr)
    print('true_gCi = ',true_gCi)

    # Start with brute force.
    # With only 100 lenses, this still runs very fast.
    lens_cat = treecorr.Catalog(x=xl, y=yl, z=zl, g1=gl.real, g2=gl.imag)
    source_cat = treecorr.Catalog(x=xs, y=ys, z=zs, g1=g1, g2=g2)
    gg = treecorr.GGCorrelation(bin_size=bin_size, min_sep=min_sep, max_sep=max_sep, verbose=1,
                                metric='FisherRperp', brute=True)
    gg.process(lens_cat, source_cat)

    Rperp = gg.meanr
    theory_gQ = gamma0 * np.exp(-0.5*Rperp**2/R1**2)

    print('Results with brute=True')
    print('gg.npairs = ',gg.npairs)
    print('true_npairs = ',true_npairs)
    np.testing.assert_array_equal(gg.npairs, true_npairs)
    print('gg.xim = ',gg.xim)
    print('true_gQ = ',true_gQ)
    print('ratio = ',gg.xim / true_gQ)
    print('diff = ',gg.xim - true_gQ)
    print('max diff = ',max(abs(gg.xim - true_gQ)))
    assert max(abs(gg.xim - true_gQ)) < 1.e-5
    print('gg.xim_im = ',gg.xim_im)
    assert max(abs(gg.xim_im)) < 1.e-5
    print('gg.xip = ',gg.xip)
    print('true_gCr = ',true_gCr)
    print('diff = ',gg.xip - true_gCr)
    print('max diff = ',max(abs(gg.xip - true_gCr)))
    assert max(abs(gg.xip - true_gCr)) < 1.e-5
    print('gg.xip_im = ',gg.xip_im)
    print('true_gCi = ',true_gCi)
    print('diff = ',gg.xip_im - true_gCi)
    print('max diff = ',max(abs(gg.xip_im - true_gCi)))
    assert max(abs(gg.xip_im - true_gCi)) < 1.e-5

    print('gg.xim = ',gg.xim)
    print('theory_gammat = ',theory_gQ)
    print('ratio = ',gg.xim / theory_gQ)
    print('diff = ',gg.xim - theory_gQ)
    print('max diff = ',max(abs(gg.xim - theory_gQ)))
    assert max(abs(gg.xim - theory_gQ)) < 4.e-5

    # With bin_slop = 0, it should get the same npairs, but the shapes will be slightly off,
    # since the directions won't be exactly right.
    gg = treecorr.GGCorrelation(bin_size=bin_size, min_sep=min_sep, max_sep=max_sep, verbose=1,
                                metric='FisherRperp', bin_slop=0)
    gg.process(lens_cat, source_cat)

    Rperp = gg.meanr
    theory_gQ = gamma0 * np.exp(-0.5*Rperp**2/R1**2)

    print('Results with bin_slop = 0:')
    print('gg.npairs = ',gg.npairs)
    print('true_npairs = ',true_npairs)
    np.testing.assert_array_equal(gg.npairs, true_npairs)
    print('gg.xim = ',gg.xim)
    print('true_gQ = ',true_gQ)
    print('ratio = ',gg.xim / true_gQ)
    print('diff = ',gg.xim - true_gQ)
    print('max diff = ',max(abs(gg.xim - true_gQ)))
    assert max(abs(gg.xim - true_gQ)) < 1.e-5
    print('gg.xim_im = ',gg.xim_im)
    assert max(abs(gg.xim_im)) < 1.e-5
    print('gg.xip = ',gg.xip)
    print('true_gCr = ',true_gCr)
    print('diff = ',gg.xip - true_gCr)
    print('max diff = ',max(abs(gg.xip - true_gCr)))
    assert max(abs(gg.xip - true_gCr)) < 1.e-5
    print('gg.xip_im = ',gg.xip_im)
    print('true_gCi = ',true_gCi)
    print('diff = ',gg.xip_im - true_gCi)
    print('max diff = ',max(abs(gg.xip_im - true_gCi)))
    assert max(abs(gg.xip_im - true_gCi)) < 1.e-5

    print('gg.xim = ',gg.xim)
    print('theory_gammat = ',theory_gQ)
    print('ratio = ',gg.xim / theory_gQ)
    print('diff = ',gg.xim - theory_gQ)
    print('max diff = ',max(abs(gg.xim - theory_gQ)))
    assert max(abs(gg.xim - theory_gQ)) < 4.e-5

    # Now use a more normal value for bin_slop.
    gg = treecorr.GGCorrelation(bin_size=bin_size, min_sep=min_sep, max_sep=max_sep, verbose=1,
                                metric='FisherRperp', bin_slop=0.3)
    gg.process(lens_cat, source_cat)
    Rperp = gg.meanr
    theory_gQ = gamma0 * np.exp(-0.5*Rperp**2/R1**2)

    print('Results with bin_slop = 0.3')
    print('gg.npairs = ',gg.npairs)
    print('gg.xim = ',gg.xim)
    print('theory_gammat = ',theory_gQ)
    print('ratio = ',gg.xim / theory_gQ)
    print('diff = ',gg.xim - theory_gQ)
    print('max diff = ',max(abs(gg.xim - theory_gQ)))
    assert max(abs(gg.xim - theory_gQ)) < 4.e-5
    print('gg.xim_im = ',gg.xim_im)
    assert max(abs(gg.xim_im)) < 1.e-5

    # Check that we get the same result using the corr2 function
    lens_cat.write(os.path.join('data','gg_rperp_lens.dat'))
    source_cat.write(os.path.join('data','gg_rperp_source.dat'))
    config = treecorr.read_config('configs/gg_rperp.yaml')
    config['verbose'] = 0
    treecorr.corr2(config)
    corr2_output = np.genfromtxt(os.path.join('output','gg_rperp.out'),names=True,
                                 skip_header=1)
    print('gg.xim = ',gg.xim)
    print('from corr2 output = ',corr2_output['xim'])
    print('ratio = ',corr2_output['xim']/gg.xim)
    print('diff = ',corr2_output['xim']-gg.xim)
    np.testing.assert_allclose(corr2_output['xim'], gg.xim, rtol=1.e-3)
    np.testing.assert_allclose(corr2_output['xim_im'], gg.xim_im, rtol=1.e-3)
    np.testing.assert_allclose(corr2_output['xip'], gg.xip, rtol=1.e-3)
    np.testing.assert_allclose(corr2_output['xip_im'], gg.xip_im, rtol=1.e-3)


@timer
def test_gg_rperp_local():
    # Same as above, but using min_rpar, max_rpar to get local (intrinsic alignment) correlations.

    nlens = 1
    nsource = 100000
    gamma0 = 0.05
    R0 = 5.
    L = 100. * R0
    rng = np.random.RandomState(8675309)

    # Lenses are randomly located with random shapes.
    xl = (rng.random_sample(nlens)-0.5) * L  # -250 < x < 250
    zl = (rng.random_sample(nlens)-0.5) * L  # -250 < y < 250
    yl = rng.random_sample(nlens) * 8*L + 10*L  # 5000 < z < 9000
    rl = np.sqrt(xl**2 + yl**2 + zl**2)
    g1l = rng.normal(0., 0.1, (nlens,))
    g2l = rng.normal(0., 0.1, (nlens,))
    gl = g1l + 1j * g2l
    gl /= np.abs(gl)
    print('Made lenses')

    # For the signal, we'll do a pure quadrupole halo lens signal.  cf. test_haloellip()
    # We also only apply it to sources within L of the lens.
    xs = (rng.random_sample(nsource)-0.5) * L
    zs = (rng.random_sample(nsource)-0.5) * L
    ys = rng.random_sample(nsource) * 8*L + 10*L  # 5000 < z < 9000
    rs = np.sqrt(xs**2 + ys**2 + zs**2)
    g1 = np.zeros( (nsource,) )
    g2 = np.zeros( (nsource,) )
    bin_size = 0.1
    # The min/max sep range can be larger here than above, since we're not diluted by the signal
    # from other background galaxies around different lenses.
    min_sep = 4*R0
    max_sep = 50.*R0
    # Because the Rperp values are a lot larger than the Rlens values, use a larger scale radius
    # in the gaussian signal.
    R1 = 4. * R0
    nbins = int(np.ceil(np.log(max_sep/min_sep)/bin_size))
    bs = np.log(max_sep/min_sep)/nbins  # Real bin size to use.

    print('Making shear vectors')
    for x,y,z,r,g in zip(xl,yl,zl,rl,gl):
        # This time, only apply the shape to the nearby galaxies.
        near = np.abs(rs-r) < 50

        dsq = (x-xs[near])**2 + (y-ys[near])**2 + (z-zs[near])**2
        Lsq = ((x+xs[near])**2 + (y+ys[near])**2 + (z+zs[near])**2) / 4.
        Rpar = abs(rs[near]**2 - r**2) / (2 * np.sqrt(Lsq))
        Rperp = np.sqrt(dsq - Rpar**2)
        gammaQ = gamma0 * np.exp(-0.5*Rperp**2/R1**2)

        dx = (xs/rs)[near]-x/r
        dz = (zs/rs)[near]-z/r
        expialpha = dx + 1j*dz
        expialpha /= np.abs(expialpha)

        gQ = gammaQ * expialpha**4 * np.conj(g)
        g1[near] += gQ.real
        g2[near] += gQ.imag

    # Like in test_rlens_bkg, we need to calculate the full g1,g2 arrays first, and then
    # go back and calculate the true_g values, since we need to include the contamination signal
    # from galaxies that are nearby multiple halos.
    print('Calculating true shears')
    true_gQ = np.zeros( (nbins,) )
    true_gCr = np.zeros( (nbins,) )
    true_gCi = np.zeros( (nbins,) )
    true_npairs = np.zeros((nbins,), dtype=int)
    for x,y,z,r,g in zip(xl,yl,zl,rl,gl):
        near = np.abs(rs-r) < 50

        dsq = (x-xs[near])**2 + (y-ys[near])**2 + (z-zs[near])**2
        Lsq = ((x+xs[near])**2 + (y+ys[near])**2 + (z+zs[near])**2) / 4.
        Rpar = abs(rs[near]**2 - r**2) / (2 * np.sqrt(Lsq))
        Rperp = np.sqrt(dsq - Rpar**2)

        dx = (xs/rs)[near]-x/r
        dz = (zs/rs)[near]-z/r
        expmialpha = dx - 1j*dz
        expmialpha /= np.abs(expmialpha)
        gs = (g1 + 1j * g2)[near]
        gQ = gs * expmialpha**4 * g

        index = np.floor( np.log(Rperp/min_sep) / bs).astype(int)
        mask = (index >= 0) & (index < nbins)
        np.add.at(true_gQ, index[mask], gQ[mask].real)
        np.add.at(true_npairs, index[mask], 1)

        gC = gs * np.conj(g)
        np.add.at(true_gCr, index[mask], gC[mask].real)
        np.add.at(true_gCi, index[mask], -gC[mask].imag)

    true_gQ /= true_npairs
    true_gCr /= true_npairs
    true_gCi /= true_npairs
    print('true_gQ = ',true_gQ)
    print('true_gCr = ',true_gCr)
    print('true_gCi = ',true_gCi)

    # Start with brute force.
    # With only 100 lenses, this still runs very fast.
    lens_cat = treecorr.Catalog(x=xl, y=yl, z=zl, g1=gl.real, g2=gl.imag)
    source_cat = treecorr.Catalog(x=xs, y=ys, z=zs, g1=g1, g2=g2)
    gg = treecorr.GGCorrelation(bin_size=bin_size, min_sep=min_sep, max_sep=max_sep, verbose=1,
                                metric='FisherRperp', brute=True, min_rpar=-50, max_rpar=50)
    gg.process(lens_cat, source_cat)

    Rperp = gg.meanr
    theory_gQ = gamma0 * np.exp(-0.5*Rperp**2/R1**2)

    print('Results with brute force:')
    print('gg.npairs = ',gg.npairs)
    print('true_npairs = ',true_npairs)
    np.testing.assert_array_equal(gg.npairs, true_npairs)
    print('gg.xim = ',gg.xim)
    print('true_gQ = ',true_gQ)
    print('ratio = ',gg.xim / true_gQ)
    print('diff = ',gg.xim - true_gQ)
    print('max diff = ',max(abs(gg.xim - true_gQ)))
    assert max(abs(gg.xim - true_gQ)) < 3.e-6
    print('gg.xim_im = ',gg.xim_im)
    print('max = ',max(abs(gg.xim_im)))
    assert max(abs(gg.xim_im)) < 1.e-4
    print('gg.xip = ',gg.xip)
    print('true_gCr = ',true_gCr)
    print('diff = ',gg.xip - true_gCr)
    print('max diff = ',max(abs(gg.xip - true_gCr)))
    assert max(abs(gg.xip - true_gCr)) < 3.e-6
    print('gg.xip_im = ',gg.xip_im)
    print('true_gCi = ',true_gCi)
    print('diff = ',gg.xip_im - true_gCi)
    print('max diff = ',max(abs(gg.xip_im - true_gCi)))
    assert max(abs(gg.xip_im - true_gCi)) < 3.e-6

    print('gg.xim = ',gg.xim)
    print('theory_gammat = ',theory_gQ)
    print('ratio = ',gg.xim / theory_gQ)
    print('diff = ',gg.xim - theory_gQ)
    print('max diff = ',max(abs(gg.xim - theory_gQ)))
    assert max(abs(gg.xim - theory_gQ)) < 4.e-5

    # Now bin_slop=0
    gg = treecorr.GGCorrelation(bin_size=bin_size, min_sep=min_sep, max_sep=max_sep, verbose=1,
                                metric='FisherRperp', bin_slop=0, min_rpar=-50, max_rpar=50)
    gg.process(lens_cat, source_cat)

    Rperp = gg.meanr
    theory_gQ = gamma0 * np.exp(-0.5*Rperp**2/R1**2)

    print('Results with bin_slop = 0:')
    print('gg.npairs = ',gg.npairs)
    print('true_npairs = ',true_npairs)
    np.testing.assert_array_equal(gg.npairs, true_npairs)
    print('gg.xim = ',gg.xim)
    print('true_gQ = ',true_gQ)
    print('ratio = ',gg.xim / true_gQ)
    print('diff = ',gg.xim - true_gQ)
    print('max diff = ',max(abs(gg.xim - true_gQ)))
    assert max(abs(gg.xim - true_gQ)) < 1.e-5
    print('gg.xim_im = ',gg.xim_im)
    print('max = ',max(abs(gg.xim_im)))
    assert max(abs(gg.xim_im)) < 1.e-4
    print('gg.xip = ',gg.xip)
    print('true_gCr = ',true_gCr)
    print('diff = ',gg.xip - true_gCr)
    print('max diff = ',max(abs(gg.xip - true_gCr)))
    assert max(abs(gg.xip - true_gCr)) < 1.e-5
    print('gg.xip_im = ',gg.xip_im)
    print('true_gCi = ',true_gCi)
    print('diff = ',gg.xip_im - true_gCi)
    print('max diff = ',max(abs(gg.xip_im - true_gCi)))
    assert max(abs(gg.xip_im - true_gCi)) < 1.e-5

    print('gg.xim = ',gg.xim)
    print('theory_gammat = ',theory_gQ)
    print('ratio = ',gg.xim / theory_gQ)
    print('diff = ',gg.xim - theory_gQ)
    print('max diff = ',max(abs(gg.xim - theory_gQ)))
    assert max(abs(gg.xim - theory_gQ)) < 4.e-5

    # Now use a more normal value for bin_slop.
    # Need a little smaller bin_slop here to help limit the number of galaxies without any
    # signal from contributing to the sum.
    gg = treecorr.GGCorrelation(bin_size=bin_size, min_sep=min_sep, max_sep=max_sep, verbose=1,
                                metric='FisherRperp', bin_slop=0.1, min_rpar=-50, max_rpar=50)
    gg.process(lens_cat, source_cat)
    Rperp = gg.meanr
    theory_gQ = gamma0 * np.exp(-0.5*Rperp**2/R1**2)

    print('Results with bin_slop = 0.1')
    print('gg.npairs = ',gg.npairs)
    print('gg.xim = ',gg.xim)
    print('theory_gammat = ',theory_gQ)
    print('ratio = ',gg.xim / theory_gQ)
    print('diff = ',gg.xim - theory_gQ)
    print('max diff = ',max(abs(gg.xim - theory_gQ)))
    assert max(abs(gg.xim - theory_gQ)) < 1.e-4
    print('gg.xim_im = ',gg.xim_im)
    assert max(abs(gg.xim_im)) < 1.e-4

    # Check that we get the same result using the corr2 function
    lens_cat.write(os.path.join('data','gg_rperp_local_lens.dat'))
    source_cat.write(os.path.join('data','gg_rperp_local_source.dat'))
    config = treecorr.read_config('configs/gg_rperp_local.yaml')
    config['verbose'] = 0
    treecorr.corr2(config)
    corr2_output = np.genfromtxt(os.path.join('output','gg_rperp_local.out'),names=True,
                                    skip_header=1)
    print('gg.xim = ',gg.xim)
    print('from corr2 output = ',corr2_output['xim'])
    print('ratio = ',corr2_output['xim']/gg.xim)
    print('diff = ',corr2_output['xim']-gg.xim)
    np.testing.assert_allclose(corr2_output['xim'], gg.xim, rtol=1.e-3)
    np.testing.assert_allclose(corr2_output['xim_im'], gg.xim_im, rtol=1.e-3)
    np.testing.assert_allclose(corr2_output['xip'], gg.xip, rtol=1.e-3)
    np.testing.assert_allclose(corr2_output['xip_im'], gg.xip_im, rtol=1.e-3)

    # Finally, with a local measurement, Rperp isn't too different from Arc using the
    # mean distance to normalize the angles.
    d = np.mean(rl)
    gg2 = treecorr.GGCorrelation(bin_size=bin_size, min_sep=min_sep/d, max_sep=max_sep/d, verbose=1,
                                 metric='Arc', bin_slop=0.1, min_rpar=-50, max_rpar=50)
    gg2.process(lens_cat, source_cat)
    Rperp = gg2.meanr * d
    theory_gQ = gamma0 * np.exp(-0.5*Rperp**2/R1**2)

    print('Results with bin_slop = 0.1, metric=Arc')
    print('gg2.npairs = ',gg2.npairs)
    print('gg2.xim = ',gg2.xim)
    print('theory_gammat = ',theory_gQ)
    print('ratio = ',gg2.xim / theory_gQ)
    print('diff = ',gg2.xim - theory_gQ)
    print('max diff = ',max(abs(gg2.xim - theory_gQ)))
    assert max(abs(gg2.xim - theory_gQ)) < 1.e-4
    print('gg2.xim_im = ',gg2.xim_im)
    assert max(abs(gg2.xim_im)) < 1.e-4

    # Euclidean is pretty different, but this is mostly a sanity check that it does something
    # reasonable and doesn't die.
    gg3 = treecorr.GGCorrelation(bin_size=bin_size, min_sep=min_sep, max_sep=max_sep, verbose=1,
                                 metric='Euclidean', bin_slop=0.1, min_rpar=-50, max_rpar=50)
    gg3.process(lens_cat, source_cat)
    Rperp = gg3.meanr
    theory_gQ = gamma0 * np.exp(-0.5*Rperp**2/R1**2)

    print('Results with bin_slop = 0.1, metric=Arc')
    print('gg3.npairs = ',gg3.npairs)
    print('gg3.xim = ',gg3.xim)
    print('theory_gammat = ',theory_gQ)
    print('ratio = ',gg3.xim / theory_gQ)
    print('diff = ',gg3.xim - theory_gQ)
    print('max diff = ',max(abs(gg3.xim - theory_gQ)))
    assert max(abs(gg3.xim - theory_gQ)) < 0.1  # Not a good match.  As expected.
    print('gg3.xim_im = ',gg3.xim_im)
    assert max(abs(gg3.xim_im)) < 1.e-4

    # Invalid to have min > max
    with assert_raises(ValueError):
        treecorr.GGCorrelation(bin_size=bin_size, min_sep=min_sep, max_sep=max_sep, verbose=1,
                               metric='FisherRperp', bin_slop=0.1, min_rpar=50, max_rpar=-50)

    # Invalid to use min/max rpar for flat or spherical coords
    lens_cat_flat = treecorr.Catalog(x=xl, y=yl, g1=gl.real, g2=gl.imag)
    source_cat_flat = treecorr.Catalog(x=xs, y=ys, g1=g1, g2=g2)
    ral, decl = coord.CelestialCoord.xyz_to_radec(xl,yl,zl)
    ras, decs = coord.CelestialCoord.xyz_to_radec(xs,ys,zs)
    lens_cat_spher = treecorr.Catalog(ra=ral, dec=decl, g1=gl.real, g2=gl.imag,
                                      ra_units='rad', dec_units='rad')
    source_cat_spher = treecorr.Catalog(ra=ras, dec=decs, g1=g1, g2=g2,
                                        ra_units='rad', dec_units='rad')
    gg4 = treecorr.GGCorrelation(bin_size=bin_size, min_sep=min_sep, max_sep=max_sep, verbose=1,
                                 bin_slop=0.1, min_rpar=-50)
    with assert_raises(ValueError):
        gg4.process(lens_cat_flat, source_cat_flat)
    with assert_raises(ValueError):
        gg4.process(lens_cat_spher, source_cat_spher)

    gg5 = treecorr.GGCorrelation(bin_size=bin_size, min_sep=min_sep, max_sep=max_sep, verbose=1,
                                 bin_slop=0.1, max_rpar=50)
    with assert_raises(ValueError):
        gg5.process(lens_cat_flat, source_cat_flat)
    with assert_raises(ValueError):
        gg5.process(lens_cat_spher, source_cat_spher)

@timer
def test_gg_oldrperp():
    # Same as above, but using OldRperp.

    nlens = 100
    nsource = 100000
    gamma0 = 0.05
    R0 = 10.
    L = 50. * R0
    rng = np.random.RandomState(8675309)

    # Lenses are randomly located with random shapes.
    xl = (rng.random_sample(nlens)-0.5) * L  # -250 < x < 250
    zl = (rng.random_sample(nlens)-0.5) * L  # -250 < y < 250
    yl = rng.random_sample(nlens) * 4*L + 10*L  # 5000 < z < 7000
    rl = np.sqrt(xl**2 + yl**2 + zl**2)
    g1l = rng.normal(0., 0.1, (nlens,))
    g2l = rng.normal(0., 0.1, (nlens,))
    gl = g1l + 1j * g2l
    gl /= np.abs(gl)
    print('Made lenses')

    # For the signal, we'll do a pure quadrupole halo lens signal.  cf. test_haloellip()
    xs = (rng.random_sample(nsource)-0.5) * L
    zs = (rng.random_sample(nsource)-0.5) * L
    ys = rng.random_sample(nsource) * 8*L + 160*L  # 80000 < z < 84000
    rs = np.sqrt(xs**2 + ys**2 + zs**2)
    g1 = np.zeros( (nsource,) )
    g2 = np.zeros( (nsource,) )
    bin_size = 0.1
    # min_sep is set so the first bin doesn't have 0 pairs.
    # Both this and max_sep need to be larger than what we used for Rlens.
    min_sep = 5.*R0
    # max_sep can't be too large, since the measured value starts to have shape noise for larger
    # values of separation.  We're not adding any shape noise directly, but the shear from other
    # lenses is effectively a shape noise, and that comes to dominate the measurement above ~4R0.
    max_sep = 14.*R0
    # Because the Rperp values are a lot larger than the Rlens values, use a larger scale radius
    # in the gaussian signal.
    R1 = 4. * R0
    nbins = int(np.ceil(np.log(max_sep/min_sep)/bin_size))
    bs = np.log(max_sep/min_sep)/nbins  # Real bin size to use.
    true_gQ = np.zeros( (nbins,) )
    true_gCr = np.zeros( (nbins,) )
    true_gCi = np.zeros( (nbins,) )
    true_npairs = np.zeros((nbins,), dtype=int)
    print('Making shear vectors')
    for x,y,z,r,g in zip(xl,yl,zl,rl,gl):
        dsq = (x-xs)**2 + (y-ys)**2 + (z-zs)**2
        rparsq = (r-rs)**2
        Rperp = np.sqrt(dsq - rparsq)
        gammaQ = gamma0 * np.exp(-0.5*Rperp**2/R1**2)

        dx = xs/rs-x/r
        dz = zs/rs-z/r
        expialpha = dx + 1j*dz
        expialpha /= np.abs(expialpha)

        gQ = gammaQ * expialpha**4 * np.conj(g)
        g1 += gQ.real
        g2 += gQ.imag

        index = np.floor( np.log(Rperp/min_sep) / bs).astype(int)
        mask = (index >= 0) & (index < nbins)
        np.add.at(true_gQ, index[mask], gammaQ[mask])
        np.add.at(true_npairs, index[mask], 1)

        gC = gQ * np.conj(g)
        np.add.at(true_gCr, index[mask], gC[mask].real)
        np.add.at(true_gCi, index[mask], -gC[mask].imag)

    true_gQ /= true_npairs
    true_gCr /= true_npairs
    true_gCi /= true_npairs
    print('true_gQ = ',true_gQ)
    print('true_gCr = ',true_gCr)
    print('true_gCi = ',true_gCi)

    # Start with brute force.
    # With only 100 lenses, this still runs very fast.
    lens_cat = treecorr.Catalog(x=xl, y=yl, z=zl, g1=gl.real, g2=gl.imag)
    source_cat = treecorr.Catalog(x=xs, y=ys, z=zs, g1=g1, g2=g2)
    gg = treecorr.GGCorrelation(bin_size=bin_size, min_sep=min_sep, max_sep=max_sep, verbose=1,
                                metric='OldRperp', brute=True)
    gg.process(lens_cat, source_cat)

    Rperp = gg.meanr
    theory_gQ = gamma0 * np.exp(-0.5*Rperp**2/R1**2)

    print('Results with brute force:')
    print('gg.npairs = ',gg.npairs)
    print('true_npairs = ',true_npairs)
    np.testing.assert_array_equal(gg.npairs, true_npairs)
    print('gg.xim = ',gg.xim)
    print('true_gQ = ',true_gQ)
    print('ratio = ',gg.xim / true_gQ)
    print('diff = ',gg.xim - true_gQ)
    print('max diff = ',max(abs(gg.xim - true_gQ)))
    assert max(abs(gg.xim - true_gQ)) < 1.e-5
    print('gg.xim_im = ',gg.xim_im)
    assert max(abs(gg.xim_im)) < 1.e-5
    print('gg.xip = ',gg.xip)
    print('true_gCr = ',true_gCr)
    print('diff = ',gg.xip - true_gCr)
    print('max diff = ',max(abs(gg.xip - true_gCr)))
    assert max(abs(gg.xip - true_gCr)) < 1.e-5
    print('gg.xip_im = ',gg.xip_im)
    print('true_gCi = ',true_gCi)
    print('diff = ',gg.xip_im - true_gCi)
    print('max diff = ',max(abs(gg.xip_im - true_gCi)))
    assert max(abs(gg.xip_im - true_gCi)) < 1.e-5

    print('gg.xim = ',gg.xim)
    print('theory_gammat = ',theory_gQ)
    print('ratio = ',gg.xim / theory_gQ)
    print('diff = ',gg.xim - theory_gQ)
    print('max diff = ',max(abs(gg.xim - theory_gQ)))
    assert max(abs(gg.xim - theory_gQ)) < 4.e-5

    # With bin_slop = 0, it should get the same npairs, but the shapes will be slightly off,
    # since the directions won't be exactly right.
    gg = treecorr.GGCorrelation(bin_size=bin_size, min_sep=min_sep, max_sep=max_sep, verbose=1,
                                metric='OldRperp', bin_slop=0)
    gg.process(lens_cat, source_cat)

    Rperp = gg.meanr
    theory_gQ = gamma0 * np.exp(-0.5*Rperp**2/R1**2)

    print('Results with bin_slop = 0:')
    print('gg.npairs = ',gg.npairs)
    print('true_npairs = ',true_npairs)
    np.testing.assert_array_equal(gg.npairs, true_npairs)
    print('gg.xim = ',gg.xim)
    print('true_gQ = ',true_gQ)
    print('ratio = ',gg.xim / true_gQ)
    print('diff = ',gg.xim - true_gQ)
    print('max diff = ',max(abs(gg.xim - true_gQ)))
    assert max(abs(gg.xim - true_gQ)) < 1.e-5
    print('gg.xim_im = ',gg.xim_im)
    assert max(abs(gg.xim_im)) < 1.e-5
    print('gg.xip = ',gg.xip)
    print('true_gCr = ',true_gCr)
    print('diff = ',gg.xip - true_gCr)
    print('max diff = ',max(abs(gg.xip - true_gCr)))
    assert max(abs(gg.xip - true_gCr)) < 1.e-5
    print('gg.xip_im = ',gg.xip_im)
    print('true_gCi = ',true_gCi)
    print('diff = ',gg.xip_im - true_gCi)
    print('max diff = ',max(abs(gg.xip_im - true_gCi)))
    assert max(abs(gg.xip_im - true_gCi)) < 1.e-5

    print('gg.xim = ',gg.xim)
    print('theory_gammat = ',theory_gQ)
    print('ratio = ',gg.xim / theory_gQ)
    print('diff = ',gg.xim - theory_gQ)
    print('max diff = ',max(abs(gg.xim - theory_gQ)))
    assert max(abs(gg.xim - theory_gQ)) < 4.e-5

    # Now use a more normal value for bin_slop.
    gg = treecorr.GGCorrelation(bin_size=bin_size, min_sep=min_sep, max_sep=max_sep, verbose=1,
                                metric='OldRperp', bin_slop=0.3)
    gg.process(lens_cat, source_cat)
    Rperp = gg.meanr
    theory_gQ = gamma0 * np.exp(-0.5*Rperp**2/R1**2)

    print('Results with bin_slop = 0.3')
    print('gg.npairs = ',gg.npairs)
    print('gg.xim = ',gg.xim)
    print('theory_gammat = ',theory_gQ)
    print('ratio = ',gg.xim / theory_gQ)
    print('diff = ',gg.xim - theory_gQ)
    print('max diff = ',max(abs(gg.xim - theory_gQ)))
    assert max(abs(gg.xim - theory_gQ)) < 4.e-5
    print('gg.xim_im = ',gg.xim_im)
    assert max(abs(gg.xim_im)) < 1.e-5

    # Check that we get the same result using the corr2 function:
    lens_cat.write(os.path.join('data','gg_oldrperp_lens.dat'))
    source_cat.write(os.path.join('data','gg_oldrperp_source.dat'))
    config = treecorr.read_config('configs/gg_oldrperp.yaml')
    config['verbose'] = 0
    treecorr.corr2(config)
    corr2_output = np.genfromtxt(os.path.join('output','gg_oldrperp.out'),names=True,
                                    skip_header=1)
    print('gg.xim = ',gg.xim)
    print('from corr2 output = ',corr2_output['xim'])
    print('ratio = ',corr2_output['xim']/gg.xim)
    print('diff = ',corr2_output['xim']-gg.xim)
    np.testing.assert_allclose(corr2_output['xim'], gg.xim, rtol=1.e-3)
    np.testing.assert_allclose(corr2_output['xim_im'], gg.xim_im, rtol=1.e-3)
    np.testing.assert_allclose(corr2_output['xip'], gg.xip, rtol=1.e-3)
    np.testing.assert_allclose(corr2_output['xip_im'], gg.xip_im, rtol=1.e-3)


@timer
def test_gg_oldrperp_local():
    # Same as above, but using min_rpar, max_rpar to get local (intrinsic alignment) correlations.

    nlens = 1
    nsource = 500000
    gamma0 = 0.05
    R0 = 10.
    L = 50. * R0
    rng = np.random.RandomState(8675309)

    # Lenses are randomly located with random shapes.
    xl = (rng.random_sample(nlens)-0.5) * L  # -250 < x < 250
    zl = (rng.random_sample(nlens)-0.5) * L  # -250 < y < 250
    yl = rng.random_sample(nlens) * 8*L + 10*L  # 5000 < z < 9000
    rl = np.sqrt(xl**2 + yl**2 + zl**2)
    g1l = rng.normal(0., 0.1, (nlens,))
    g2l = rng.normal(0., 0.1, (nlens,))
    gl = g1l + 1j * g2l
    gl /= np.abs(gl)
    print('Made lenses')

    # For the signal, we'll do a pure quadrupole halo lens signal.  cf. test_haloellip()
    # We also only apply it to sources within L of the lens.
    xs = (rng.random_sample(nsource)-0.5) * L
    zs = (rng.random_sample(nsource)-0.5) * L
    ys = rng.random_sample(nsource) * 8*L + 10*L  # 5000 < z < 9000
    rs = np.sqrt(xs**2 + ys**2 + zs**2)
    g1 = np.zeros( (nsource,) )
    g2 = np.zeros( (nsource,) )
    bin_size = 0.1
    # The min/max sep range can be larger here than above, since we're not diluted by the signal
    # from other background galaxies around different lenses.
    min_sep = R0
    max_sep = 30.*R0
    # Because the Rperp values are a lot larger than the Rlens values, use a larger scale radius
    # in the gaussian signal.
    R1 = 4. * R0
    nbins = int(np.ceil(np.log(max_sep/min_sep)/bin_size))
    bs = np.log(max_sep/min_sep)/nbins  # Real bin size to use.

    print('Making shear vectors')
    for x,y,z,r,g in zip(xl,yl,zl,rl,gl):
        # This time, only apply the shape to the nearby galaxies.
        near = np.abs(rs-r) < 50

        dsq = (x-xs[near])**2 + (y-ys[near])**2 + (z-zs[near])**2
        rparsq = (r-rs[near])**2
        Rperp = np.sqrt(dsq - rparsq)
        gammaQ = gamma0 * np.exp(-0.5*Rperp**2/R1**2)

        dx = (xs/rs)[near]-x/r
        dz = (zs/rs)[near]-z/r
        expialpha = dx + 1j*dz
        expialpha /= np.abs(expialpha)

        gQ = gammaQ * expialpha**4 * np.conj(g)
        g1[near] += gQ.real
        g2[near] += gQ.imag

    # Like in test_rlens_bkg, we need to calculate the full g1,g2 arrays first, and then
    # go back and calculate the true_g values, since we need to include the contamination signal
    # from galaxies that are nearby multiple halos.
    print('Calculating true shears')
    true_gQ = np.zeros( (nbins,) )
    true_gCr = np.zeros( (nbins,) )
    true_gCi = np.zeros( (nbins,) )
    true_npairs = np.zeros((nbins,), dtype=int)
    for x,y,z,r,g in zip(xl,yl,zl,rl,gl):
        near = np.abs(rs-r) < 50

        dsq = (x-xs[near])**2 + (y-ys[near])**2 + (z-zs[near])**2
        rparsq = (r-rs[near])**2
        Rperp = np.sqrt(dsq - rparsq)

        dx = (xs/rs)[near]-x/r
        dz = (zs/rs)[near]-z/r
        expmialpha = dx - 1j*dz
        expmialpha /= np.abs(expmialpha)
        gs = (g1 + 1j * g2)[near]
        gQ = gs * expmialpha**4 * g

        index = np.floor( np.log(Rperp/min_sep) / bs).astype(int)
        mask = (index >= 0) & (index < nbins)
        np.add.at(true_gQ, index[mask], gQ[mask].real)
        np.add.at(true_npairs, index[mask], 1)

        gC = gs * np.conj(g)
        np.add.at(true_gCr, index[mask], gC[mask].real)
        np.add.at(true_gCi, index[mask], -gC[mask].imag)

    true_gQ /= true_npairs
    true_gCr /= true_npairs
    true_gCi /= true_npairs
    print('true_gQ = ',true_gQ)
    print('true_gCr = ',true_gCr)
    print('true_gCi = ',true_gCi)

    # Start with brute force.
    # With only 100 lenses, this still runs very fast.
    lens_cat = treecorr.Catalog(x=xl, y=yl, z=zl, g1=gl.real, g2=gl.imag)
    source_cat = treecorr.Catalog(x=xs, y=ys, z=zs, g1=g1, g2=g2)
    gg = treecorr.GGCorrelation(bin_size=bin_size, min_sep=min_sep, max_sep=max_sep, verbose=1,
                                metric='OldRperp', brute=True, min_rpar=-50, max_rpar=50)
    gg.process(lens_cat, source_cat)

    Rperp = gg.meanr
    theory_gQ = gamma0 * np.exp(-0.5*Rperp**2/R1**2)

    print('Results with brute force:')
    print('gg.npairs = ',gg.npairs)
    print('true_npairs = ',true_npairs)
    np.testing.assert_array_equal(gg.npairs, true_npairs)
    print('gg.xim = ',gg.xim)
    print('true_gQ = ',true_gQ)
    print('ratio = ',gg.xim / true_gQ)
    print('diff = ',gg.xim - true_gQ)
    print('max diff = ',max(abs(gg.xim - true_gQ)))
    assert max(abs(gg.xim - true_gQ)) < 3.e-6
    print('gg.xim_im = ',gg.xim_im)
    print('max = ',max(abs(gg.xim_im)))
    assert max(abs(gg.xim_im)) < 1.e-4
    print('gg.xip = ',gg.xip)
    print('true_gCr = ',true_gCr)
    print('diff = ',gg.xip - true_gCr)
    print('max diff = ',max(abs(gg.xip - true_gCr)))
    assert max(abs(gg.xip - true_gCr)) < 3.e-6
    print('gg.xip_im = ',gg.xip_im)
    print('true_gCi = ',true_gCi)
    print('diff = ',gg.xip_im - true_gCi)
    print('max diff = ',max(abs(gg.xip_im - true_gCi)))
    assert max(abs(gg.xip_im - true_gCi)) < 3.e-6

    print('gg.xim = ',gg.xim)
    print('theory_gammat = ',theory_gQ)
    print('ratio = ',gg.xim / theory_gQ)
    print('diff = ',gg.xim - theory_gQ)
    print('max diff = ',max(abs(gg.xim - theory_gQ)))
    assert max(abs(gg.xim - theory_gQ)) < 4.e-5

    # Now bin_slop=0
    gg = treecorr.GGCorrelation(bin_size=bin_size, min_sep=min_sep, max_sep=max_sep, verbose=1,
                                metric='OldRperp', bin_slop=0, min_rpar=-50, max_rpar=50)
    gg.process(lens_cat, source_cat)

    Rperp = gg.meanr
    theory_gQ = gamma0 * np.exp(-0.5*Rperp**2/R1**2)

    print('Results with bin_slop = 0:')
    print('gg.npairs = ',gg.npairs)
    print('true_npairs = ',true_npairs)
    np.testing.assert_array_equal(gg.npairs, true_npairs)
    print('gg.xim = ',gg.xim)
    print('true_gQ = ',true_gQ)
    print('ratio = ',gg.xim / true_gQ)
    print('diff = ',gg.xim - true_gQ)
    print('max diff = ',max(abs(gg.xim - true_gQ)))
    assert max(abs(gg.xim - true_gQ)) < 1.e-5
    print('gg.xim_im = ',gg.xim_im)
    print('max = ',max(abs(gg.xim_im)))
    assert max(abs(gg.xim_im)) < 1.e-4
    print('gg.xip = ',gg.xip)
    print('true_gCr = ',true_gCr)
    print('diff = ',gg.xip - true_gCr)
    print('max diff = ',max(abs(gg.xip - true_gCr)))
    assert max(abs(gg.xip - true_gCr)) < 1.e-5
    print('gg.xip_im = ',gg.xip_im)
    print('true_gCi = ',true_gCi)
    print('diff = ',gg.xip_im - true_gCi)
    print('max diff = ',max(abs(gg.xip_im - true_gCi)))
    assert max(abs(gg.xip_im - true_gCi)) < 1.e-5

    print('gg.xim = ',gg.xim)
    print('theory_gammat = ',theory_gQ)
    print('ratio = ',gg.xim / theory_gQ)
    print('diff = ',gg.xim - theory_gQ)
    print('max diff = ',max(abs(gg.xim - theory_gQ)))
    assert max(abs(gg.xim - theory_gQ)) < 4.e-5

    # Now use a more normal value for bin_slop.
    # Need a little smaller bin_slop here to help limit the number of galaxies without any
    # signal from contributing to the sum.
    gg = treecorr.GGCorrelation(bin_size=bin_size, min_sep=min_sep, max_sep=max_sep, verbose=1,
                                metric='OldRperp', bin_slop=0.1, min_rpar=-50, max_rpar=50)
    gg.process(lens_cat, source_cat)
    Rperp = gg.meanr
    theory_gQ = gamma0 * np.exp(-0.5*Rperp**2/R1**2)

    print('Results with bin_slop = 0.1')
    print('gg.npairs = ',gg.npairs)
    print('gg.xim = ',gg.xim)
    print('theory_gammat = ',theory_gQ)
    print('ratio = ',gg.xim / theory_gQ)
    print('diff = ',gg.xim - theory_gQ)
    print('max diff = ',max(abs(gg.xim - theory_gQ)))
    assert max(abs(gg.xim - theory_gQ)) < 1.e-4
    print('gg.xim_im = ',gg.xim_im)
    assert max(abs(gg.xim_im)) < 1.e-4

    # Check that we get the same result using the corr2 function:
    lens_cat.write(os.path.join('data','gg_oldrperp_local_lens.dat'))
    source_cat.write(os.path.join('data','gg_oldrperp_local_source.dat'))
    config = treecorr.config.read_config('configs/gg_oldrperp_local.yaml')
    logger = treecorr.config.setup_logger(0)
    treecorr.corr2(config, logger)
    corr2_output = np.genfromtxt(os.path.join('output','gg_oldrperp_local.out'),names=True,
                                    skip_header=1)
    print('gg.xim = ',gg.xim)
    print('from corr2 output = ',corr2_output['xim'])
    print('ratio = ',corr2_output['xim']/gg.xim)
    print('diff = ',corr2_output['xim']-gg.xim)
    np.testing.assert_allclose(corr2_output['xim'], gg.xim, rtol=1.e-3)
    np.testing.assert_allclose(corr2_output['xim_im'], gg.xim_im, rtol=1.e-3)
    np.testing.assert_allclose(corr2_output['xip'], gg.xip, rtol=1.e-3)
    np.testing.assert_allclose(corr2_output['xip_im'], gg.xip_im, rtol=1.e-3)


@timer
def test_symmetric():
    """Test that RPerp is symmetric in rpar values around 0.

    This test is from Simon Samuroff, who found a bug in verion 4.0.5 where the Rperp
    results for negative rpar are not correct.  In particular, cat1 - cat2 with positive rpar
    should match exactly cat2 - cat1 with negative rpar.
    """
    nbins = 10

    # generate some data. There doesn't need to be a real signal - random numbers will do.
    dat = {}
    ndat = 20000
    rng = np.random.RandomState(9000)
    dat['ra'] = rng.rand(ndat) * 5
    dat['dec'] = rng.rand(ndat) * 5
    dat['g1'] = rng.normal(scale=0.05,size=ndat)
    dat['g2'] = rng.normal(scale=0.05,size=ndat)
    dat['r'] = rng.rand(ndat) * 5000
    print('r = ',dat['r'])

    # Set up the catalogues
    cat = treecorr.Catalog(w=None, ra_units='deg', dec_units='deg', **dat)

    pilo = 20
    pihi = 30

    # Test NN counts with + and - rpar values
    nn1 = treecorr.NNCorrelation(nbins=nbins, min_sep=0.1, max_sep=100,
                                 min_rpar=pilo, max_rpar=pihi)
    nn1.process(cat, cat, metric='Rperp')

    nn2 = treecorr.NNCorrelation(nbins=nbins, min_sep=0.1, max_sep=100,
                                 min_rpar=-pihi, max_rpar=-pilo)
    nn2.process(cat, cat, metric='Rperp')

    print('+rpar weight = ',nn1.weight)
    print('-rpar weight = ',nn2.weight)
    np.testing.assert_allclose(nn1.weight, nn2.weight)

    # Check GG xi+, xi-
    gg1 = treecorr.GGCorrelation(nbins=nbins, min_sep=0.1, max_sep=100,
                                 min_rpar=pilo, max_rpar=pihi)
    gg1.process(cat, cat, metric='Rperp')

    gg2 = treecorr.GGCorrelation(nbins=nbins, min_sep=0.1, max_sep=100,
                                 min_rpar=-pihi, max_rpar=-pilo)
    gg2.process(cat, cat, metric='Rperp')

    print('+rpar xip = ',gg1.xip)
    print('-rpar xip = ',gg2.xip)
    np.testing.assert_allclose(gg1.xip, gg2.xip)
    np.testing.assert_allclose(gg1.xim, gg2.xim)


if __name__ == '__main__':
    test_nn_direct_rperp()
    test_nn_direct_oldrperp()
    test_nn_direct_rlens()
    test_rperp_minmax()
    test_ng_rlens()
    test_ng_rlens_bkg()
    test_ng_rperp()
    test_gg_rlens()
    test_gg_rperp()
    test_gg_rperp_local()
    test_gg_oldrperp()
    test_gg_oldrperp_local()
    test_symmetric()
