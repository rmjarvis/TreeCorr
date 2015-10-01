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
import numpy
import treecorr
import os

from test_helper import get_from_wiki

def test_binnedcorr2():
    import math
    # Test some basic properties of the base class

    # Check the different ways to set up the binning:
    # Omit bin_size
    nn = treecorr.NNCorrelation(min_sep=5, max_sep=20, nbins=20)
    print(nn.min_sep,nn.max_sep,nn.bin_size,nn.nbins)
    assert nn.min_sep == 5.
    assert nn.max_sep == 20.
    assert nn.nbins == 20
    numpy.testing.assert_almost_equal(nn.bin_size * nn.nbins, math.log(nn.max_sep/nn.min_sep))
    print('logr = ',nn.logr)
    numpy.testing.assert_almost_equal(nn.logr[0], math.log(nn.min_sep) + 0.5*nn.bin_size)
    numpy.testing.assert_almost_equal(nn.logr[-1], math.log(nn.max_sep) - 0.5*nn.bin_size)
    assert len(nn.logr) == nn.nbins

    # Omit min_sep
    nn = treecorr.NNCorrelation(max_sep=20, nbins=20, bin_size=0.1)
    print(nn.min_sep,nn.max_sep,nn.bin_size,nn.nbins)
    assert nn.bin_size == 0.1
    assert nn.max_sep == 20.
    assert nn.nbins == 20
    numpy.testing.assert_almost_equal(nn.bin_size * nn.nbins, math.log(nn.max_sep/nn.min_sep))
    numpy.testing.assert_almost_equal(nn.logr[0], math.log(nn.min_sep) + 0.5*nn.bin_size)
    numpy.testing.assert_almost_equal(nn.logr[-1], math.log(nn.max_sep) - 0.5*nn.bin_size)
    assert len(nn.logr) == nn.nbins

    # Omit max_sep
    nn = treecorr.NNCorrelation(min_sep=5, nbins=20, bin_size=0.1)
    print(nn.min_sep,nn.max_sep,nn.bin_size,nn.nbins)
    assert nn.bin_size == 0.1
    assert nn.min_sep == 5.
    assert nn.nbins == 20
    numpy.testing.assert_almost_equal(nn.bin_size * nn.nbins, math.log(nn.max_sep/nn.min_sep))
    numpy.testing.assert_almost_equal(nn.logr[0], math.log(nn.min_sep) + 0.5*nn.bin_size)
    numpy.testing.assert_almost_equal(nn.logr[-1], math.log(nn.max_sep) - 0.5*nn.bin_size)
    assert len(nn.logr) == nn.nbins

    # Omit nbins
    nn = treecorr.NNCorrelation(min_sep=5, max_sep=20, bin_size=0.1)
    print(nn.min_sep,nn.max_sep,nn.bin_size,nn.nbins)
    assert nn.bin_size == 0.1
    assert nn.min_sep == 5.
    assert nn.max_sep >= 20.
    assert nn.max_sep <= 20.*math.exp(nn.bin_size)
    numpy.testing.assert_almost_equal(nn.bin_size * nn.nbins, math.log(nn.max_sep/nn.min_sep))
    numpy.testing.assert_almost_equal(nn.logr[0], math.log(nn.min_sep) + 0.5*nn.bin_size)
    numpy.testing.assert_almost_equal(nn.logr[-1], math.log(nn.max_sep) - 0.5*nn.bin_size)
    assert len(nn.logr) == nn.nbins

    # Check the use of sep_units
    # radians
    nn = treecorr.NNCorrelation(min_sep=5, max_sep=20, nbins=20, sep_units='radians')
    print(nn.min_sep,nn.max_sep,nn.bin_size,nn.nbins)
    assert nn.min_sep == 5.
    assert nn.max_sep == 20.
    assert nn.nbins == 20
    numpy.testing.assert_almost_equal(nn.bin_size * nn.nbins, math.log(nn.max_sep/nn.min_sep))
    numpy.testing.assert_almost_equal(nn.logr[0], math.log(5) + 0.5*nn.bin_size)
    numpy.testing.assert_almost_equal(nn.logr[-1], math.log(20) - 0.5*nn.bin_size)
    assert len(nn.logr) == nn.nbins

    # arcsec
    nn = treecorr.NNCorrelation(min_sep=5, max_sep=20, nbins=20, sep_units='arcsec')
    print(nn.min_sep,nn.max_sep,nn.bin_size,nn.nbins)
    numpy.testing.assert_almost_equal(nn.min_sep, 5. * math.pi/180/3600)
    numpy.testing.assert_almost_equal(nn.max_sep, 20. * math.pi/180/3600)
    assert nn.nbins == 20
    numpy.testing.assert_almost_equal(nn.bin_size * nn.nbins, math.log(nn.max_sep/nn.min_sep))
    # Note that logr is in the separation units, not radians.
    numpy.testing.assert_almost_equal(nn.logr[0], math.log(5) + 0.5*nn.bin_size)
    numpy.testing.assert_almost_equal(nn.logr[-1], math.log(20) - 0.5*nn.bin_size)
    assert len(nn.logr) == nn.nbins

    # arcmin
    nn = treecorr.NNCorrelation(min_sep=5, max_sep=20, nbins=20, sep_units='arcmin')
    print(nn.min_sep,nn.max_sep,nn.bin_size,nn.nbins)
    numpy.testing.assert_almost_equal(nn.min_sep, 5. * math.pi/180/60)
    numpy.testing.assert_almost_equal(nn.max_sep, 20. * math.pi/180/60)
    assert nn.nbins == 20
    numpy.testing.assert_almost_equal(nn.bin_size * nn.nbins, math.log(nn.max_sep/nn.min_sep))
    numpy.testing.assert_almost_equal(nn.logr[0], math.log(5) + 0.5*nn.bin_size)
    numpy.testing.assert_almost_equal(nn.logr[-1], math.log(20) - 0.5*nn.bin_size)
    assert len(nn.logr) == nn.nbins

    # degrees
    nn = treecorr.NNCorrelation(min_sep=5, max_sep=20, nbins=20, sep_units='degrees')
    print(nn.min_sep,nn.max_sep,nn.bin_size,nn.nbins)
    numpy.testing.assert_almost_equal(nn.min_sep, 5. * math.pi/180)
    numpy.testing.assert_almost_equal(nn.max_sep, 20. * math.pi/180)
    assert nn.nbins == 20
    numpy.testing.assert_almost_equal(nn.bin_size * nn.nbins, math.log(nn.max_sep/nn.min_sep))
    numpy.testing.assert_almost_equal(nn.logr[0], math.log(5) + 0.5*nn.bin_size)
    numpy.testing.assert_almost_equal(nn.logr[-1], math.log(20) - 0.5*nn.bin_size)
    assert len(nn.logr) == nn.nbins

    # hours
    nn = treecorr.NNCorrelation(min_sep=5, max_sep=20, nbins=20, sep_units='hours')
    print(nn.min_sep,nn.max_sep,nn.bin_size,nn.nbins)
    numpy.testing.assert_almost_equal(nn.min_sep, 5. * math.pi/12)
    numpy.testing.assert_almost_equal(nn.max_sep, 20. * math.pi/12)
    assert nn.nbins == 20
    numpy.testing.assert_almost_equal(nn.bin_size * nn.nbins, math.log(nn.max_sep/nn.min_sep))
    numpy.testing.assert_almost_equal(nn.logr[0], math.log(5) + 0.5*nn.bin_size)
    numpy.testing.assert_almost_equal(nn.logr[-1], math.log(20) - 0.5*nn.bin_size)
    assert len(nn.logr) == nn.nbins

    # Check bin_slop
    nn = treecorr.NNCorrelation(min_sep=5, max_sep=20, bin_size=0.1)
    print(nn.bin_size,nn.bin_slop,nn.b)
    assert nn.bin_size == 0.1
    assert nn.bin_slop == 1.0
    numpy.testing.assert_almost_equal(nn.b, 0.1)

    nn = treecorr.NNCorrelation(min_sep=5, max_sep=20, bin_size=0.1, bin_slop=1.0)
    print(nn.bin_size,nn.bin_slop,nn.b)
    assert nn.bin_size == 0.1
    assert nn.bin_slop == 1.0
    numpy.testing.assert_almost_equal(nn.b, 0.1)

    nn = treecorr.NNCorrelation(min_sep=5, max_sep=20, bin_size=0.1, bin_slop=0.2)
    print(nn.bin_size,nn.bin_slop,nn.b)
    assert nn.bin_size == 0.1
    assert nn.bin_slop == 0.2
    numpy.testing.assert_almost_equal(nn.b, 0.02)

    nn = treecorr.NNCorrelation(min_sep=5, max_sep=20, bin_size=0.1, bin_slop=0.0)
    print(nn.bin_size,nn.bin_slop,nn.b)
    assert nn.bin_size == 0.1
    assert nn.bin_slop == 0.0
    assert nn.b == 0.0

    nn = treecorr.NNCorrelation(min_sep=5, max_sep=20, bin_size=0.1, bin_slop=2.0)
    print(nn.bin_size,nn.bin_slop,nn.b)
    assert nn.bin_size == 0.1
    assert nn.bin_slop == 2.0
    numpy.testing.assert_almost_equal(nn.b, 0.2)

    nn = treecorr.NNCorrelation(min_sep=5, max_sep=20, bin_size=0.4, bin_slop=1.0)
    print(nn.bin_size,nn.bin_slop,nn.b)
    assert nn.bin_size == 0.4
    assert nn.bin_slop == 1.0
    numpy.testing.assert_almost_equal(nn.b, 0.4)

    nn = treecorr.NNCorrelation(min_sep=5, max_sep=20, bin_size=0.4)
    print(nn.bin_size,nn.bin_slop,nn.b)
    assert nn.bin_size == 0.4
    numpy.testing.assert_almost_equal(nn.b, 0.1)
    numpy.testing.assert_almost_equal(nn.bin_slop, 0.25)

    nn = treecorr.NNCorrelation(min_sep=5, max_sep=20, bin_size=0.4, bin_slop=0.1)
    print(nn.bin_size,nn.bin_slop,nn.b)
    assert nn.bin_size == 0.4
    assert nn.bin_slop == 0.1
    numpy.testing.assert_almost_equal(nn.b, 0.04)

    nn = treecorr.NNCorrelation(min_sep=5, max_sep=20, bin_size=0.05, bin_slop=1.0)
    print(nn.bin_size,nn.bin_slop,nn.b)
    assert nn.bin_size == 0.05
    assert nn.bin_slop == 1.0
    numpy.testing.assert_almost_equal(nn.b, 0.05)

    nn = treecorr.NNCorrelation(min_sep=5, max_sep=20, bin_size=0.05)
    print(nn.bin_size,nn.bin_slop,nn.b)
    assert nn.bin_size == 0.05
    assert nn.bin_slop == 1.0
    numpy.testing.assert_almost_equal(nn.b, 0.05)

    nn = treecorr.NNCorrelation(min_sep=5, max_sep=20, bin_size=0.05, bin_slop=3)
    print(nn.bin_size,nn.bin_slop,nn.b)
    assert nn.bin_size == 0.05
    assert nn.bin_slop == 3.0
    numpy.testing.assert_almost_equal(nn.b, 0.15)



def test_direct_count():
    # If the catalogs are small enough, we can do a direct count of the number of pairs
    # to see if comes out right.  This should exactly match the treecorr code if bin_slop=0.

    ngal = 100
    s = 10.
    numpy.random.seed(8675309)
    x1 = numpy.random.normal(0,s, (ngal,) )
    y1 = numpy.random.normal(0,s, (ngal,) )
    cat1 = treecorr.Catalog(x=x1, y=y1)
    x2 = numpy.random.normal(0,s, (ngal,) )
    y2 = numpy.random.normal(0,s, (ngal,) )
    cat2 = treecorr.Catalog(x=x2, y=y2)

    min_sep = 1.
    max_sep = 50.
    nbins = 50
    dd = treecorr.NNCorrelation(min_sep=min_sep, max_sep=max_sep, nbins=nbins, bin_slop=0.)
    dd.process(cat1, cat2)
    print('dd.npairs = ',dd.npairs)

    log_min_sep = numpy.log(min_sep)
    log_max_sep = numpy.log(max_sep)
    true_npairs = numpy.zeros(nbins)
    bin_size = (log_max_sep - log_min_sep) / nbins
    for i in range(ngal):
        for j in range(ngal):
            rsq = (x1[i]-x2[j])**2 + (y1[i]-y2[j])**2
            logr = 0.5 * numpy.log(rsq)
            k = int(numpy.floor( (logr-log_min_sep) / bin_size ))
            if k < 0: continue
            if k >= nbins: continue
            true_npairs[k] += 1

    print('true_npairs = ',true_npairs)
    print('diff = ',dd.npairs - true_npairs)
    numpy.testing.assert_array_equal(dd.npairs, true_npairs)

def test_direct_3d():
    # This is the same as the above test, but using the 3d correlations

    ngal = 100
    s = 10.
    numpy.random.seed(8675309)
    x1 = numpy.random.normal(312, s, (ngal,) )
    y1 = numpy.random.normal(728, s, (ngal,) )
    z1 = numpy.random.normal(-932, s, (ngal,) )
    r1 = numpy.sqrt( x1*x1 + y1*y1 + z1*z1 )
    dec1 = numpy.arcsin(z1/r1)
    ra1 = numpy.arctan2(y1,x1)
    cat1 = treecorr.Catalog(ra=ra1, dec=dec1, r=r1, ra_units='rad', dec_units='rad')

    x2 = numpy.random.normal(312, s, (ngal,) )
    y2 = numpy.random.normal(728, s, (ngal,) )
    z2 = numpy.random.normal(-932, s, (ngal,) )
    r2 = numpy.sqrt( x2*x2 + y2*y2 + z2*z2 )
    dec2 = numpy.arcsin(z2/r2)
    ra2 = numpy.arctan2(y2,x2)
    cat2 = treecorr.Catalog(ra=ra2, dec=dec2, r=r2, ra_units='rad', dec_units='rad')

    min_sep = 1.
    max_sep = 50.
    nbins = 50
    dd = treecorr.NNCorrelation(min_sep=min_sep, max_sep=max_sep, nbins=nbins, bin_slop=0.)
    dd.process(cat1, cat2)
    print('dd.npairs = ',dd.npairs)

    log_min_sep = numpy.log(min_sep)
    log_max_sep = numpy.log(max_sep)
    true_npairs = numpy.zeros(nbins)
    bin_size = (log_max_sep - log_min_sep) / nbins
    for i in range(ngal):
        for j in range(ngal):
            rsq = (x1[i]-x2[j])**2 + (y1[i]-y2[j])**2 + (z1[i]-z2[j])**2
            logr = 0.5 * numpy.log(rsq)
            k = int(numpy.floor( (logr-log_min_sep) / bin_size ))
            if k < 0: continue
            if k >= nbins: continue
            true_npairs[k] += 1

    print('true_npairs = ',true_npairs)
    print('diff = ',dd.npairs - true_npairs)
    numpy.testing.assert_array_equal(dd.npairs, true_npairs)

    # Can also specify coords directly as x,y,z
    cat1 = treecorr.Catalog(x=x1, y=y1, z=z1)
    cat2 = treecorr.Catalog(x=x2, y=y2, z=z2)
    dd.process(cat1, cat2)
    numpy.testing.assert_array_equal(dd.npairs, true_npairs)


def test_direct_perp():
    # This is the same as the above test, but using the perpendicular distance metric

    get_from_wiki('nn_perp_data.dat')
    get_from_wiki('nn_perp_rand.dat')

    ngal = 100
    s = 10.
    numpy.random.seed(8675309)
    x1 = numpy.random.normal(312, s, (ngal,) )
    y1 = numpy.random.normal(728, s, (ngal,) )
    z1 = numpy.random.normal(-932, s, (ngal,) )
    r1 = numpy.sqrt( x1*x1 + y1*y1 + z1*z1 )
    dec1 = numpy.arcsin(z1/r1)
    ra1 = numpy.arctan2(y1,x1)
    cat1 = treecorr.Catalog(ra=ra1, dec=dec1, r=r1, ra_units='rad', dec_units='rad')

    x2 = numpy.random.normal(312, s, (ngal,) )
    y2 = numpy.random.normal(728, s, (ngal,) )
    z2 = numpy.random.normal(-932, s, (ngal,) )
    r2 = numpy.sqrt( x2*x2 + y2*y2 + z2*z2 )
    dec2 = numpy.arcsin(z2/r2)
    ra2 = numpy.arctan2(y2,x2)
    cat2 = treecorr.Catalog(ra=ra2, dec=dec2, r=r2, ra_units='rad', dec_units='rad')

    min_sep = 1.
    max_sep = 50.
    nbins = 50
    dd = treecorr.NNCorrelation(min_sep=min_sep, max_sep=max_sep, nbins=nbins, bin_slop=0.)
    dd.process(cat1, cat2, metric='Rperp')
    print('dd.npairs = ',dd.npairs)

    log_min_sep = numpy.log(min_sep)
    log_max_sep = numpy.log(max_sep)
    true_npairs = numpy.zeros(nbins)
    bin_size = (log_max_sep - log_min_sep) / nbins
    for i in range(ngal):
        for j in range(ngal):
            rsq = (x1[i]-x2[j])**2 + (y1[i]-y2[j])**2 + (z1[i]-z2[j])**2
            rsq -= (r1[i] - r2[j])**2
            logr = 0.5 * numpy.log(rsq)
            k = int(numpy.floor( (logr-log_min_sep) / bin_size ))
            if k < 0: continue
            if k >= nbins: continue
            true_npairs[k] += 1

    print('true_npairs = ',true_npairs)
    print('diff = ',dd.npairs - true_npairs)
    numpy.testing.assert_array_equal(dd.npairs, true_npairs)

    # Can also specify coords directly as x,y,z
    cat1 = treecorr.Catalog(x=x1, y=y1, z=z1)
    cat2 = treecorr.Catalog(x=x2, y=y2, z=z2)
    dd.process(cat1, cat2, metric='Rperp')
    numpy.testing.assert_array_equal(dd.npairs, true_npairs)

def test_direct_partial():
    # There are two ways to specify using only parts of a catalog:
    # 1. The parameters first_row and last_row would usually be used for files, but they are a 
    #    general way to use only a (contiguous) subset of the rows
    # 2. You can also set weights to 0 for the rows you don't want to use.

    # First test first_row, last_row
    ngal = 200
    s = 10.
    numpy.random.seed(8675309)
    x1 = numpy.random.normal(0,s, (ngal,) )
    y1 = numpy.random.normal(0,s, (ngal,) )
    cat1a = treecorr.Catalog(x=x1, y=y1, first_row=28, last_row=144)
    x2 = numpy.random.normal(0,s, (ngal,) )
    y2 = numpy.random.normal(0,s, (ngal,) )
    cat2a = treecorr.Catalog(x=x2, y=y2, first_row=48, last_row=129)

    min_sep = 1.
    max_sep = 50.
    nbins = 50
    dda = treecorr.NNCorrelation(min_sep=min_sep, max_sep=max_sep, nbins=nbins, bin_slop=0.)
    dda.process(cat1a, cat2a)
    print('dda.npairs = ',dda.npairs)

    log_min_sep = numpy.log(min_sep)
    log_max_sep = numpy.log(max_sep)
    true_npairs = numpy.zeros(nbins)
    bin_size = (log_max_sep - log_min_sep) / nbins
    for i in range(27,144):
        for j in range(47,129):
            rsq = (x1[i]-x2[j])**2 + (y1[i]-y2[j])**2
            logr = 0.5 * numpy.log(rsq)
            k = int(numpy.floor( (logr-log_min_sep) / bin_size ))
            if k < 0: continue
            if k >= nbins: continue
            true_npairs[k] += 1

    print('true_npairs = ',true_npairs)
    print('diff = ',dda.npairs - true_npairs)
    numpy.testing.assert_array_equal(dda.npairs, true_npairs)

    # Now check that we get the same thing with all the points, but with w=0 for the ones 
    # we don't want.
    w1 = numpy.zeros(ngal)
    w1[27:144] = 1.
    w2 = numpy.zeros(ngal)
    w2[47:129] = 1.
    cat1b = treecorr.Catalog(x=x1, y=y1, w=w1)
    cat2b = treecorr.Catalog(x=x2, y=y2, w=w2)
    ddb = treecorr.NNCorrelation(min_sep=min_sep, max_sep=max_sep, nbins=nbins, bin_slop=0.)
    ddb.process(cat1b, cat2b)
    print('ddb.npairs = ',ddb.npairs)
    print('diff = ',ddb.npairs - true_npairs)
    numpy.testing.assert_array_equal(ddb.npairs, true_npairs)


def test_nn():
    # Use a simple probability distribution for the galaxies:
    #
    # n(r) = (2pi s^2)^-1 exp(-r^2/2s^2)
    #
    # The Fourier transform is: n~(k) = exp(-s^2 k^2/2)
    # P(k) = <|n~(k)|^2> = exp(-s^2 k^2)
    # xi(r) = (1/2pi) int( dk k P(k) J0(kr) ) 
    #       = 1/(4 pi s^2) exp(-r^2/4s^2)
    #
    # However, we need to correct for the uniform density background, so the real result
    # is this minus 1/L^2 divided by 1/L^2.  So:
    #
    # xi(r) = 1/4pi (L/s)^2 exp(-r^2/4s^2) - 1

    s = 10.
    if __name__ == "__main__":
        ngal = 1000000
        nrand = 5 * ngal
        L = 50. * s  # Not infinity, so this introduces some error.  Our integrals were to infinity.
        req_factor = 1
    else:
        ngal = 100000
        nrand = 2 * ngal
        L = 20. * s
        req_factor = 3
    numpy.random.seed(8675309)
    x = numpy.random.normal(0,s, (ngal,) )
    y = numpy.random.normal(0,s, (ngal,) )

    cat = treecorr.Catalog(x=x, y=y, x_units='arcmin', y_units='arcmin')
    dd = treecorr.NNCorrelation(bin_size=0.1, min_sep=1., max_sep=25., sep_units='arcmin',
                                verbose=2)
    dd.process(cat)
    print('dd.npairs = ',dd.npairs)

    # log(<R>) != <logR>, but it should be close:
    print('meanlogr - log(meanr) = ',dd.meanlogr - numpy.log(dd.meanr))
    numpy.testing.assert_almost_equal(dd.meanlogr, numpy.log(dd.meanr), decimal=3)

    rx = (numpy.random.random_sample(nrand)-0.5) * L
    ry = (numpy.random.random_sample(nrand)-0.5) * L
    rand = treecorr.Catalog(x=rx,y=ry, x_units='arcmin', y_units='arcmin')
    rr = treecorr.NNCorrelation(bin_size=0.1, min_sep=1., max_sep=25., sep_units='arcmin',
                                verbose=2)
    rr.process(rand)
    print('rr.npairs = ',rr.npairs)

    dr = treecorr.NNCorrelation(bin_size=0.1, min_sep=1., max_sep=25., sep_units='arcmin',
                                verbose=2)
    dr.process(cat,rand)
    print('dr.npairs = ',dr.npairs)

    r = dd.meanr
    true_xi = 0.25/numpy.pi * (L/s)**2 * numpy.exp(-0.25*r**2/s**2) - 1.

    xi, varxi = dd.calculateXi(rr,dr)
    print('xi = ',xi)
    print('true_xi = ',true_xi)
    print('ratio = ',xi / true_xi)
    print('diff = ',xi - true_xi)
    print('max rel diff = ',max(abs((xi - true_xi)/true_xi)))
    # This isn't super accurate.  But the agreement improves as L increase, so I think it is 
    # merely a matter of the finite field and the integrals going to infinity.  (Sort of, since
    # we still have L in there.)
    assert max(abs(xi - true_xi)/true_xi)/req_factor < 0.1
    numpy.testing.assert_almost_equal(numpy.log(numpy.abs(xi))/req_factor, 
                                      numpy.log(numpy.abs(true_xi))/req_factor, decimal=1)

    simple_xi, simple_varxi = dd.calculateXi(rr)
    print('simple xi = ',simple_xi)
    print('max rel diff = ',max(abs((simple_xi - true_xi)/true_xi)))
    # The simple calculation (i.e. dd/rr-1, rather than (dd-2dr+rr)/rr as above) is only 
    # slightly less accurate in this case.  Probably because the mask is simple (a box), so
    # the difference is relatively minor.  The error is slightly higher in this case, but testing
    # that it is everywhere < 0.1 is still appropriate.
    assert max(abs(simple_xi - true_xi)/true_xi)/req_factor < 0.1

    # Check that we get the same result using the corr2 executable:
    if __name__ == '__main__':
        cat.write(os.path.join('data','nn_data.dat'))
        rand.write(os.path.join('data','nn_rand.dat'))
        import subprocess
        p = subprocess.Popen( ["corr2","nn.params"] )
        p.communicate()
        corr2_output = numpy.genfromtxt(os.path.join('output','nn.out'),names=True)
        print('xi = ',xi)
        print('from corr2 output = ',corr2_output['xi'])
        print('ratio = ',corr2_output['xi']/xi)
        print('diff = ',corr2_output['xi']-xi)
        numpy.testing.assert_almost_equal(corr2_output['xi']/xi, 1., decimal=3)

    # Check the fits write option
    out_file_name1 = os.path.join('output','nn_out1.fits')
    dd.write(out_file_name1)
    try:
        import fitsio
        data = fitsio.read(out_file_name1)
        numpy.testing.assert_almost_equal(data['R_nom'], numpy.exp(dd.logr))
        numpy.testing.assert_almost_equal(data['meanR'], dd.meanr)
        numpy.testing.assert_almost_equal(data['meanlogR'], dd.meanlogr)
        numpy.testing.assert_almost_equal(data['npairs'], dd.npairs)
    except ImportError:
        print('Unable to import fitsio.  Skipping fits tests.')

    out_file_name2 = os.path.join('output','nn_out2.fits')
    dd.write(out_file_name2, rr)
    try:
        import fitsio
        data = fitsio.read(out_file_name2)
        numpy.testing.assert_almost_equal(data['R_nom'], numpy.exp(dd.logr))
        numpy.testing.assert_almost_equal(data['meanR'], dd.meanr)
        numpy.testing.assert_almost_equal(data['meanlogR'], dd.meanlogr)
        numpy.testing.assert_almost_equal(data['xi'], simple_xi)
        numpy.testing.assert_almost_equal(data['sigma_xi'], numpy.sqrt(simple_varxi))
        numpy.testing.assert_almost_equal(data['DD'], dd.npairs)
        numpy.testing.assert_almost_equal(data['RR'], rr.npairs * (dd.tot / rr.tot))
    except ImportError:
        print('Unable to import fitsio.  Skipping fits tests.')

    out_file_name3 = os.path.join('output','nn_out3.fits')
    dd.write(out_file_name3, rr, dr)
    try:
        import fitsio
        data = fitsio.read(out_file_name3)
        numpy.testing.assert_almost_equal(data['R_nom'], numpy.exp(dd.logr))
        numpy.testing.assert_almost_equal(data['meanR'], dd.meanr)
        numpy.testing.assert_almost_equal(data['meanlogR'], dd.meanlogr)
        numpy.testing.assert_almost_equal(data['xi'], xi)
        numpy.testing.assert_almost_equal(data['sigma_xi'], numpy.sqrt(varxi))
        numpy.testing.assert_almost_equal(data['DD'], dd.npairs)
        numpy.testing.assert_almost_equal(data['RR'], rr.npairs * (dd.tot / rr.tot))
        numpy.testing.assert_almost_equal(data['DR'], dr.npairs * (dd.tot / dr.tot))
    except ImportError:
        print('Unable to import fitsio.  Skipping fits tests.')

    # Check the read function
    dd2 = treecorr.NNCorrelation(bin_size=0.1, min_sep=1., max_sep=25., sep_units='arcmin')
    dd2.read(out_file_name1)
    numpy.testing.assert_almost_equal(dd2.logr, dd.logr)
    numpy.testing.assert_almost_equal(dd2.meanr, dd.meanr)
    numpy.testing.assert_almost_equal(dd2.meanlogr, dd.meanlogr)
    numpy.testing.assert_almost_equal(dd2.npairs, dd.npairs)

    dd2.read(out_file_name3)
    numpy.testing.assert_almost_equal(dd2.logr, dd.logr)
    numpy.testing.assert_almost_equal(dd2.meanr, dd.meanr)
    numpy.testing.assert_almost_equal(dd2.meanlogr, dd.meanlogr)
    numpy.testing.assert_almost_equal(dd2.npairs, dd.npairs)


def test_3d():
    # For this one, build a Gaussian cloud around some random point in 3D space and do the 
    # correlation function in 3D.
    #
    # Use n(r) = (2pi s^2)^-3/2 exp(-r^2/2s^2)
    #
    # The 3D Fourier transform is: n~(k) = exp(-s^2 k^2/2)
    # P(k) = <|n~(k)|^2> = exp(-s^2 k^2)
    # xi(r) = 1/2pi^2 int( dk k^2 P(k) j0(kr) )
    #       = 1/(8 pi^3/2) 1/s^3 exp(-r^2/4s^2)
    #
    # And as before, we need to correct for the randoms, so the final xi(r) is
    #
    # xi(r) = 1/(8 pi^3/2) (L/s)^3 exp(-r^2/4s^2) - 1

    xcen = 823  # Mpc maybe?
    ycen = 342
    zcen = -672
    s = 10.
    if __name__ == "__main__":
        ngal = 100000
        nrand = 5 * ngal
        L = 50. * s  # Not infinity, so this introduces some error.  Our integrals were to infinity.
        req_factor = 1
    else:
        ngal = 20000
        nrand = 2 * ngal
        L = 20. * s
        req_factor = 3
    numpy.random.seed(8675309)
    x = numpy.random.normal(xcen, s, (ngal,) )
    y = numpy.random.normal(ycen, s, (ngal,) )
    z = numpy.random.normal(zcen, s, (ngal,) )

    r = numpy.sqrt(x*x+y*y+z*z)
    dec = numpy.arcsin(z/r) / treecorr.degrees
    ra = numpy.arctan2(y,x) / treecorr.degrees

    cat = treecorr.Catalog(ra=ra, dec=dec, r=r, ra_units='deg', dec_units='deg')
    dd = treecorr.NNCorrelation(bin_size=0.1, min_sep=1., max_sep=25., verbose=2)
    dd.process(cat)
    print('dd.npairs = ',dd.npairs)

    rx = (numpy.random.random_sample(nrand)-0.5) * L + xcen
    ry = (numpy.random.random_sample(nrand)-0.5) * L + ycen
    rz = (numpy.random.random_sample(nrand)-0.5) * L + zcen
    rr = numpy.sqrt(rx*rx+ry*ry+rz*rz)
    rdec = numpy.arcsin(rz/rr) / treecorr.degrees
    rra = numpy.arctan2(ry,rx) / treecorr.degrees
    rand = treecorr.Catalog(ra=rra, dec=rdec, r=rr, ra_units='deg', dec_units='deg')
    rr = treecorr.NNCorrelation(bin_size=0.1, min_sep=1., max_sep=25., verbose=2)
    rr.process(rand)
    print('rr.npairs = ',rr.npairs)

    dr = treecorr.NNCorrelation(bin_size=0.1, min_sep=1., max_sep=25., verbose=2)
    dr.process(cat,rand)
    print('dr.npairs = ',dr.npairs)

    r = dd.meanr
    true_xi = 1./(8.*numpy.pi**1.5) * (L/s)**3 * numpy.exp(-0.25*r**2/s**2) - 1.

    xi, varxi = dd.calculateXi(rr,dr)
    print('xi = ',xi)
    print('true_xi = ',true_xi)
    print('ratio = ',xi / true_xi)
    print('diff = ',xi - true_xi)
    print('max rel diff = ',max(abs((xi - true_xi)/true_xi)))
    assert max(abs(xi - true_xi)/true_xi)/req_factor < 0.1
    numpy.testing.assert_almost_equal(numpy.log(numpy.abs(xi))/req_factor, 
                                      numpy.log(numpy.abs(true_xi))/req_factor, decimal=1)

    simple_xi, varxi = dd.calculateXi(rr)
    print('simple xi = ',simple_xi)
    print('max rel diff = ',max(abs((simple_xi - true_xi)/true_xi)))
    assert max(abs(simple_xi - true_xi)/true_xi)/req_factor < 0.1
    numpy.testing.assert_almost_equal(numpy.log(numpy.abs(simple_xi))/req_factor, 
                                      numpy.log(numpy.abs(true_xi))/req_factor, decimal=1)

    # Check that we get the same result using the corr2 executable:
    if __name__ == '__main__':
        cat.write(os.path.join('data','nn_3d_data.dat'))
        rand.write(os.path.join('data','nn_3d_rand.dat'))
        import subprocess
        p = subprocess.Popen( ["corr2","nn_3d.params"] )
        p.communicate()
        corr2_output = numpy.genfromtxt(os.path.join('output','nn_3d.out'),names=True)
        print('xi = ',xi)
        print('from corr2 output = ',corr2_output['xi'])
        print('ratio = ',corr2_output['xi']/xi)
        print('diff = ',corr2_output['xi']-xi)
        numpy.testing.assert_almost_equal(corr2_output['xi']/xi, 1., decimal=3)

    # And repeat with Catalogs that use x,y,z
    cat = treecorr.Catalog(x=x, y=y, z=z)
    rand = treecorr.Catalog(x=rx, y=ry, z=rz)
    dd.process(cat)
    rr.process(rand)
    dr.process(cat,rand)
    xi, varxi = dd.calculateXi(rr,dr)
    assert max(abs(xi - true_xi)/true_xi)/req_factor < 0.1
    numpy.testing.assert_almost_equal(numpy.log(numpy.abs(xi))/req_factor, 
                                      numpy.log(numpy.abs(true_xi))/req_factor, decimal=1)


def test_list():
    # Test that we can use a list of files for either data or rand or both.

    nobj = 5000
    numpy.random.seed(8675309)

    ncats = 3
    data_cats = []
    rand_cats = []

    s = 10.
    L = 50. * s
    numpy.random.seed(8675309)

    x = numpy.random.normal(0,s, (nobj,ncats) )
    y = numpy.random.normal(0,s, (nobj,ncats) )
    data_cats = [ treecorr.Catalog(x=x[:,k],y=y[:,k]) for k in range(ncats) ]
    rx = (numpy.random.random_sample((nobj,ncats))-0.5) * L
    ry = (numpy.random.random_sample((nobj,ncats))-0.5) * L
    rand_cats = [ treecorr.Catalog(x=rx[:,k],y=ry[:,k]) for k in range(ncats) ]

    dd = treecorr.NNCorrelation(bin_size=0.1, min_sep=1., max_sep=25., verbose=2)
    dd.process(data_cats)
    print('dd.npairs = ',dd.npairs)

    rr = treecorr.NNCorrelation(bin_size=0.1, min_sep=1., max_sep=25., verbose=2)
    rr.process(rand_cats)
    print('rr.npairs = ',rr.npairs)

    xi, varxi = dd.calculateXi(rr)
    print('xi = ',xi)

    # Now do the same thing with one big catalog for each.
    ddx = treecorr.NNCorrelation(bin_size=0.1, min_sep=1., max_sep=25., verbose=2)
    rrx = treecorr.NNCorrelation(bin_size=0.1, min_sep=1., max_sep=25., verbose=2)
    data_catx = treecorr.Catalog(x=x.reshape( (nobj*ncats,) ), y=y.reshape( (nobj*ncats,) ))
    rand_catx = treecorr.Catalog(x=rx.reshape( (nobj*ncats,) ), y=ry.reshape( (nobj*ncats,) ))
    ddx.process(data_catx)
    rrx.process(rand_catx)
    xix, varxix = ddx.calculateXi(rrx)

    print('ddx.npairs = ',ddx.npairs)
    print('rrx.npairs = ',rrx.npairs)
    print('xix = ',xix)
    print('ratio = ',xi/xix)
    print('diff = ',xi-xix)
    numpy.testing.assert_almost_equal(xix/xi, 1., decimal=2)

    # Check that we get the same result using the corr2 executable:
    file_list = []
    rand_file_list = []
    for k in range(ncats):
        file_name = os.path.join('data','nn_list_data%d.dat'%k)
        with open(file_name, 'w') as fid:
            for i in range(nobj):
                fid.write(('%.8f %.8f\n')%(x[i,k],y[i,k]))
        file_list.append(file_name)

        rand_file_name = os.path.join('data','nn_list_rand%d.dat'%k)
        with open(rand_file_name, 'w') as fid:
            for i in range(nobj):
                fid.write(('%.8f %.8f\n')%(rx[i,k],ry[i,k]))
        rand_file_list.append(rand_file_name)

    list_name = os.path.join('data','nn_list_data_files.txt')
    with open(list_name, 'w') as fid:
        for file_name in file_list:
            fid.write('%s\n'%file_name)
    rand_list_name = os.path.join('data','nn_list_rand_files.txt')
    with open(rand_list_name, 'w') as fid:
        for file_name in rand_file_list:
            fid.write('%s\n'%file_name)

    file_namex = os.path.join('data','nn_list_datax.dat')
    with open(file_namex, 'w') as fid:
        for k in range(ncats):
            for i in range(nobj):
                fid.write(('%.8f %.8f\n')%(x[i,k],y[i,k]))

    rand_file_namex = os.path.join('data','nn_list_randx.dat')
    with open(rand_file_namex, 'w') as fid:
        for k in range(ncats):
            for i in range(nobj):
                fid.write(('%.8f %.8f\n')%(rx[i,k],ry[i,k]))

    import subprocess
    p = subprocess.Popen( ["corr2","nn_list1.params"] )
    p.communicate()
    corr2_output = numpy.genfromtxt(os.path.join('output','nn_list1.out'),names=True)
    print('xi = ',xi)
    print('from corr2 output = ',corr2_output['xi'])
    print('ratio = ',corr2_output['xi']/xi)
    print('diff = ',corr2_output['xi']-xi)
    numpy.testing.assert_almost_equal(corr2_output['xi']/xi, 1., decimal=3)

    import subprocess
    p = subprocess.Popen( ["corr2","nn_list2.params"] )
    p.communicate()
    corr2_output = numpy.genfromtxt(os.path.join('output','nn_list2.out'),names=True)
    print('xi = ',xi)
    print('from corr2 output = ',corr2_output['xi'])
    print('ratio = ',corr2_output['xi']/xi)
    print('diff = ',corr2_output['xi']-xi)
    numpy.testing.assert_almost_equal(corr2_output['xi']/xi, 1., decimal=2)

    import subprocess
    p = subprocess.Popen( ["corr2","nn_list3.params"] )
    p.communicate()
    corr2_output = numpy.genfromtxt(os.path.join('output','nn_list3.out'),names=True)
    print('xi = ',xi)
    print('from corr2 output = ',corr2_output['xi'])
    print('ratio = ',corr2_output['xi']/xi)
    print('diff = ',corr2_output['xi']-xi)
    numpy.testing.assert_almost_equal(corr2_output['xi']/xi, 1., decimal=2)

def test_perp_minmax():
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
        'min_sep' : 40,
        'bin_size' : 0.036652,
        'nbins' : 50,
        'verbose' : 2
    }

    # Speed up for nosetests runs
    if __name__ != "__main__":
        config['nbins'] = 5
        config['min_sep'] = 20
        config['bin_size'] = 0.1

    dcat = treecorr.Catalog('data/nn_perp_data.dat', config)

    dd1 = treecorr.NNCorrelation(config)
    dd1.process(dcat, metric='Rperp')

    lower_min_sep = config['min_sep'] * numpy.exp(-2.*config['bin_size'])
    more_nbins = config['nbins'] + 4
    dd2 = treecorr.NNCorrelation(config, min_sep=lower_min_sep, nbins=more_nbins)
    dd2.process(dcat, metric='Rperp')

    print('dd1 npairs = ',dd1.npairs)
    print('dd2 npairs = ',dd2.npairs[2:-2])
    # First a basic sanity check.  The values not near the edge should be identical.
    numpy.testing.assert_equal(dd1.npairs[2:-2], dd2.npairs[4:-4])
    # The edge bins may differ slightly from the binning approximations (bin_slop and such),
    # but the differences should be very small.  (When Erika reported the problem, the differences
    # were a few percent, which ended up making a bit difference in the correlation function.)
    numpy.testing.assert_almost_equal( dd1.npairs / dd2.npairs[2:-2], 1., decimal=4)

    if __name__ == '__main__':
        # If we're running from the command line, go ahead and finish the calculation
        # This catalog has 10^6 objects, which takes quite a while.  I should really investigate
        # how to speed up the Rperp distance calculation.  Probably by having a faster over-
        # and under-estimate first, and then only do the full calculation when it seems like we
        # will actually need it.  
        # Anyway, until then, let's not take forever by using last_row=200000
        rcat = treecorr.Catalog('data/nn_perp_rand.dat', config, last_row=200000)

        rr1 = treecorr.NNCorrelation(config)
        rr1.process(rcat, metric='Rperp')
        rr2 = treecorr.NNCorrelation(config, min_sep=lower_min_sep, nbins=more_nbins)
        rr2.process(rcat, metric='Rperp')
        print('rr1 npairs = ',rr1.npairs)
        print('rr2 npairs = ',rr2.npairs[2:-2])
        numpy.testing.assert_almost_equal( rr1.npairs / rr2.npairs[2:-2], 1., decimal=4)

        dr1 = treecorr.NNCorrelation(config)
        dr1.process(dcat, rcat, metric='Rperp')
        dr2 = treecorr.NNCorrelation(config, min_sep=lower_min_sep, nbins=more_nbins)
        dr2.process(dcat, rcat, metric='Rperp')
        print('dr1 npairs = ',dr1.npairs)
        print('dr2 npairs = ',dr2.npairs[2:-2])
        numpy.testing.assert_almost_equal( dr1.npairs / dr2.npairs[2:-2], 1., decimal=4)

        xi1, varxi1 = dd1.calculateXi(rr1, dr1)
        xi2, varxi2 = dd2.calculateXi(rr2, dr2)
        print('xi1 = ',xi1)
        print('xi2 = ',xi2[2:-2])
        numpy.testing.assert_almost_equal( xi1 / xi2[2:-2], 1., decimal=2)

        # Check that we get the same result with the corr2 executable.
        import subprocess
        p = subprocess.Popen( ["corr2","nn_rperp.params"] )
        p.communicate()
        corr2_output = numpy.genfromtxt(os.path.join('output','nn_rperp.out'),names=True)
        print('xi = ',xi1)
        print('from corr2 output = ',corr2_output['xi'])
        print('ratio = ',corr2_output['xi']/xi1)
        print('diff = ',corr2_output['xi']-xi1)
        numpy.testing.assert_almost_equal(corr2_output['xi']/xi1, 1., decimal=3)



if __name__ == '__main__':
    test_binnedcorr2()
    test_direct_count()
    test_direct_3d()
    test_direct_perp()
    test_direct_partial()
    test_nn()
    test_3d()
    test_list()
    test_perp_minmax()
