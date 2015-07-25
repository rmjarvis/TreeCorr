# Copyright (c) 2003-2014 by Mike Jarvis
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


import numpy
import treecorr
import os

def test_binnedcorr3():
    import math
    # Test some basic properties of the base class

    def check_arrays(nnn):
        numpy.testing.assert_almost_equal(nnn.bin_size * nnn.nbins, math.log(nnn.max_sep/nnn.min_sep))
        numpy.testing.assert_almost_equal(nnn.ubin_size * nnn.nubins, nnn.max_u-nnn.min_u)
        numpy.testing.assert_almost_equal(nnn.vbin_size * nnn.nvbins, nnn.max_v-nnn.min_v)
        print 'logr = ',nnn.logr
        numpy.testing.assert_almost_equal(nnn.logr[0], math.log(nnn.min_sep) + 0.5*nnn.bin_size)
        numpy.testing.assert_almost_equal(nnn.logr[-1], math.log(nnn.max_sep) - 0.5*nnn.bin_size)
        assert len(nnn.logr) == nnn.nbins
        print 'u = ',nnn.u
        numpy.testing.assert_almost_equal(nnn.u[0], nnn.min_u + 0.5*nnn.ubin_size)
        numpy.testing.assert_almost_equal(nnn.u[-1], nnn.max_u - 0.5*nnn.ubin_size)
        print 'v = ',nnn.v
        numpy.testing.assert_almost_equal(nnn.v[0], nnn.min_v + 0.5*nnn.vbin_size)
        numpy.testing.assert_almost_equal(nnn.v[-1], nnn.max_v - 0.5*nnn.vbin_size)

    def check_defaultuv(nnn):
        assert nnn.min_u == 0.
        assert nnn.max_u == 1.
        assert nnn.nubins == numpy.ceil(1./nnn.bin_size)
        assert nnn.min_v == -1.
        assert nnn.max_v == 1.
        assert nnn.nvbins == 2.*numpy.ceil(1./nnn.bin_size)

    # Check the different ways to set up the binning:
    # Omit bin_size
    nnn = treecorr.NNNCorrelation(min_sep=5, max_sep=20, nbins=20)
    print nnn.min_sep,nnn.max_sep,nnn.bin_size,nnn.nbins
    print nnn.min_u,nnn.max_u,nnn.ubin_size,nnn.nubins
    print nnn.min_v,nnn.max_v,nnn.vbin_size,nnn.nvbins
    assert nnn.min_sep == 5.
    assert nnn.max_sep == 20.
    assert nnn.nbins == 20
    check_defaultuv(nnn)
    check_arrays(nnn)
    # Specify min, max, n for u,v too.
    nnn = treecorr.NNNCorrelation(min_sep=5, max_sep=20, nbins=20,
                                  min_u=0.2, max_u=0.9, nubins=12,
                                  min_v=-0.2, max_v=0.2, nvbins=4)
    print nnn.min_sep,nnn.max_sep,nnn.bin_size,nnn.nbins
    print nnn.min_u,nnn.max_u,nnn.ubin_size,nnn.nubins
    print nnn.min_v,nnn.max_v,nnn.vbin_size,nnn.nvbins
    assert nnn.min_sep == 5.
    assert nnn.max_sep == 20.
    assert nnn.nbins == 20
    assert nnn.min_u == 0.2
    assert nnn.max_u == 0.9
    assert nnn.nubins == 12
    assert nnn.min_v == -0.2
    assert nnn.max_v == 0.2
    assert nnn.nvbins == 4
    check_arrays(nnn)

    # Omit min_sep
    nnn = treecorr.NNNCorrelation(max_sep=20, nbins=20, bin_size=0.1)
    print nnn.min_sep,nnn.max_sep,nnn.bin_size,nnn.nbins
    print nnn.min_u,nnn.max_u,nnn.ubin_size,nnn.nubins
    print nnn.min_v,nnn.max_v,nnn.vbin_size,nnn.nvbins
    assert nnn.bin_size == 0.1
    assert nnn.max_sep == 20.
    assert nnn.nbins == 20
    check_defaultuv(nnn)
    check_arrays(nnn)
    # Specify max, n, bs for u,v too.
    nnn = treecorr.NNNCorrelation(max_sep=20, nbins=20, bin_size=0.1,
                                  max_u=0.9, nubins=3, ubin_size=0.05,
                                  max_v=0.2, nvbins=4, vbin_size=0.05)
    print nnn.min_sep,nnn.max_sep,nnn.bin_size,nnn.nbins
    print nnn.min_u,nnn.max_u,nnn.ubin_size,nnn.nubins
    print nnn.min_v,nnn.max_v,nnn.vbin_size,nnn.nvbins
    assert nnn.bin_size == 0.1
    assert nnn.max_sep == 20.
    assert nnn.nbins == 20
    assert nnn.ubin_size == 0.05
    assert nnn.max_u == 0.9
    assert nnn.nubins == 3
    assert nnn.vbin_size == 0.05
    assert nnn.max_v == 0.2
    assert nnn.nvbins == 4
    check_arrays(nnn)

    # Omit max_sep
    nnn = treecorr.NNNCorrelation(min_sep=5, nbins=20, bin_size=0.1)
    print nnn.min_sep,nnn.max_sep,nnn.bin_size,nnn.nbins
    print nnn.min_u,nnn.max_u,nnn.ubin_size,nnn.nubins
    print nnn.min_v,nnn.max_v,nnn.vbin_size,nnn.nvbins
    assert nnn.bin_size == 0.1
    assert nnn.min_sep == 5.
    assert nnn.nbins == 20
    check_defaultuv(nnn)
    check_arrays(nnn)
    # Specify min, n, bs for u,v too.
    nnn = treecorr.NNNCorrelation(min_sep=5, nbins=20, bin_size=0.1,
                                  min_u=0.7, nubins=4, ubin_size=0.05,
                                  min_v=-0.2, nvbins=4, vbin_size=0.05)
    print nnn.min_sep,nnn.max_sep,nnn.bin_size,nnn.nbins
    print nnn.min_u,nnn.max_u,nnn.ubin_size,nnn.nubins
    print nnn.min_v,nnn.max_v,nnn.vbin_size,nnn.nvbins
    assert nnn.min_sep == 5.
    assert nnn.bin_size == 0.1
    assert nnn.nbins == 20
    assert nnn.min_u == 0.7
    assert nnn.ubin_size == 0.05
    assert nnn.nubins == 4
    assert nnn.min_v == -0.2
    assert nnn.vbin_size == 0.05
    assert nnn.nvbins == 4
    check_arrays(nnn)

    # Omit nbins
    nnn = treecorr.NNNCorrelation(min_sep=5, max_sep=20, bin_size=0.1)
    print nnn.min_sep,nnn.max_sep,nnn.bin_size,nnn.nbins
    print nnn.min_u,nnn.max_u,nnn.ubin_size,nnn.nubins
    print nnn.min_v,nnn.max_v,nnn.vbin_size,nnn.nvbins
    assert nnn.bin_size == 0.1
    assert nnn.min_sep == 5.
    assert nnn.max_sep >= 20.  # Expanded a bit.
    assert nnn.max_sep < 20. * numpy.exp(nnn.bin_size)
    check_defaultuv(nnn)
    check_arrays(nnn)
    # Specify min, max, bs for u,v too.
    nnn = treecorr.NNNCorrelation(min_sep=5, max_sep=20, bin_size=0.1,
                                  min_u=0.2, max_u=0.9, ubin_size=0.03,
                                  min_v=-0.2, max_v=0.2, vbin_size=0.07)
    print nnn.min_sep,nnn.max_sep,nnn.bin_size,nnn.nbins
    print nnn.min_u,nnn.max_u,nnn.ubin_size,nnn.nubins
    print nnn.min_v,nnn.max_v,nnn.vbin_size,nnn.nvbins
    assert nnn.min_sep == 5.
    assert nnn.max_sep >= 20.
    assert nnn.max_sep < 20. * numpy.exp(nnn.bin_size)
    assert nnn.bin_size == 0.1
    assert nnn.min_u <= 0.2
    assert nnn.min_u >= 0.2 - nnn.ubin_size
    assert nnn.max_u == 0.9
    assert nnn.ubin_size == 0.03
    assert nnn.min_v <= -0.2
    assert nnn.min_v >= -0.2 - nnn.vbin_size
    assert nnn.max_v >= 0.2
    assert nnn.min_v <= 0.2 + nnn.vbin_size
    assert nnn.vbin_size == 0.07
    check_arrays(nnn)

    # Check the use of sep_units
    # radians
    nnn = treecorr.NNNCorrelation(min_sep=5, max_sep=20, nbins=20, sep_units='radians')
    print nnn.min_sep,nnn.max_sep,nnn.bin_size,nnn.nbins
    print nnn.min_u,nnn.max_u,nnn.ubin_size,nnn.nubins
    print nnn.min_v,nnn.max_v,nnn.vbin_size,nnn.nvbins
    assert nnn.min_sep == 5.
    assert nnn.max_sep == 20.
    assert nnn.nbins == 20
    check_defaultuv(nnn)
    check_arrays(nnn)

    # arcsec
    nnn = treecorr.NNNCorrelation(min_sep=5, max_sep=20, nbins=20, sep_units='arcsec')
    print nnn.min_sep,nnn.max_sep,nnn.bin_size,nnn.nbins
    print nnn.min_u,nnn.max_u,nnn.ubin_size,nnn.nubins
    print nnn.min_v,nnn.max_v,nnn.vbin_size,nnn.nvbins
    numpy.testing.assert_almost_equal(nnn.min_sep, 5. * math.pi/180/3600)
    numpy.testing.assert_almost_equal(nnn.max_sep, 20. * math.pi/180/3600)
    assert nnn.nbins == 20
    numpy.testing.assert_almost_equal(nnn.bin_size * nnn.nbins, math.log(nnn.max_sep/nnn.min_sep))
    # Note that logr is in the separation units, not radians.
    numpy.testing.assert_almost_equal(nnn.logr[0], math.log(5) + 0.5*nnn.bin_size)
    numpy.testing.assert_almost_equal(nnn.logr[-1], math.log(20) - 0.5*nnn.bin_size)
    assert len(nnn.logr) == nnn.nbins
    check_defaultuv(nnn)

    # arcmin
    nnn = treecorr.NNNCorrelation(min_sep=5, max_sep=20, nbins=20, sep_units='arcmin')
    print nnn.min_sep,nnn.max_sep,nnn.bin_size,nnn.nbins
    print nnn.min_u,nnn.max_u,nnn.ubin_size,nnn.nubins
    print nnn.min_v,nnn.max_v,nnn.vbin_size,nnn.nvbins
    numpy.testing.assert_almost_equal(nnn.min_sep, 5. * math.pi/180/60)
    numpy.testing.assert_almost_equal(nnn.max_sep, 20. * math.pi/180/60)
    assert nnn.nbins == 20
    numpy.testing.assert_almost_equal(nnn.bin_size * nnn.nbins, math.log(nnn.max_sep/nnn.min_sep))
    numpy.testing.assert_almost_equal(nnn.logr[0], math.log(5) + 0.5*nnn.bin_size)
    numpy.testing.assert_almost_equal(nnn.logr[-1], math.log(20) - 0.5*nnn.bin_size)
    assert len(nnn.logr) == nnn.nbins
    check_defaultuv(nnn)

    # degrees
    nnn = treecorr.NNNCorrelation(min_sep=5, max_sep=20, nbins=20, sep_units='degrees')
    print nnn.min_sep,nnn.max_sep,nnn.bin_size,nnn.nbins
    print nnn.min_u,nnn.max_u,nnn.ubin_size,nnn.nubins
    print nnn.min_v,nnn.max_v,nnn.vbin_size,nnn.nvbins
    numpy.testing.assert_almost_equal(nnn.min_sep, 5. * math.pi/180)
    numpy.testing.assert_almost_equal(nnn.max_sep, 20. * math.pi/180)
    assert nnn.nbins == 20
    numpy.testing.assert_almost_equal(nnn.bin_size * nnn.nbins, math.log(nnn.max_sep/nnn.min_sep))
    numpy.testing.assert_almost_equal(nnn.logr[0], math.log(5) + 0.5*nnn.bin_size)
    numpy.testing.assert_almost_equal(nnn.logr[-1], math.log(20) - 0.5*nnn.bin_size)
    assert len(nnn.logr) == nnn.nbins
    check_defaultuv(nnn)

    # hours
    nnn = treecorr.NNNCorrelation(min_sep=5, max_sep=20, nbins=20, sep_units='hours')
    print nnn.min_sep,nnn.max_sep,nnn.bin_size,nnn.nbins
    print nnn.min_u,nnn.max_u,nnn.ubin_size,nnn.nubins
    print nnn.min_v,nnn.max_v,nnn.vbin_size,nnn.nvbins
    numpy.testing.assert_almost_equal(nnn.min_sep, 5. * math.pi/12)
    numpy.testing.assert_almost_equal(nnn.max_sep, 20. * math.pi/12)
    assert nnn.nbins == 20
    numpy.testing.assert_almost_equal(nnn.bin_size * nnn.nbins, math.log(nnn.max_sep/nnn.min_sep))
    numpy.testing.assert_almost_equal(nnn.logr[0], math.log(5) + 0.5*nnn.bin_size)
    numpy.testing.assert_almost_equal(nnn.logr[-1], math.log(20) - 0.5*nnn.bin_size)
    assert len(nnn.logr) == nnn.nbins
    check_defaultuv(nnn)

    # Check bin_slop
    # Start with default behavior
    nnn = treecorr.NNNCorrelation(min_sep=5, max_sep=20, bin_size=0.1,
                                  min_u=0.2, max_u=0.9, ubin_size=0.03,
                                  min_v=-0.2, max_v=0.2, vbin_size=0.07)
    print nnn.bin_size,nnn.bin_slop,nnn.b
    print nnn.ubin_size,nnn.bu
    print nnn.vbin_size,nnn.bv
    assert nnn.bin_slop == 1.0
    assert nnn.bin_size == 0.1
    assert nnn.ubin_size == 0.03
    assert nnn.vbin_size == 0.07
    numpy.testing.assert_almost_equal(nnn.b, 0.1)
    numpy.testing.assert_almost_equal(nnn.bu, 0.03)
    numpy.testing.assert_almost_equal(nnn.bv, 0.07)

    # Explicitly set bin_slop=1.0 does the same thing.
    nnn = treecorr.NNNCorrelation(min_sep=5, max_sep=20, bin_size=0.1, bin_slop=1.0,
                                  min_u=0.2, max_u=0.9, ubin_size=0.03,
                                  min_v=-0.2, max_v=0.2, vbin_size=0.07)
    print nnn.bin_size,nnn.bin_slop,nnn.b
    print nnn.ubin_size,nnn.bu
    print nnn.vbin_size,nnn.bv
    assert nnn.bin_slop == 1.0
    assert nnn.bin_size == 0.1
    assert nnn.ubin_size == 0.03
    assert nnn.vbin_size == 0.07
    numpy.testing.assert_almost_equal(nnn.b, 0.1)
    numpy.testing.assert_almost_equal(nnn.bu, 0.03)
    numpy.testing.assert_almost_equal(nnn.bv, 0.07)

    # Use a smaller bin_slop
    nnn = treecorr.NNNCorrelation(min_sep=5, max_sep=20, bin_size=0.1, bin_slop=0.2,
                                  min_u=0.2, max_u=0.9, ubin_size=0.03,
                                  min_v=-0.2, max_v=0.2, vbin_size=0.07)
    print nnn.bin_size,nnn.bin_slop,nnn.b
    print nnn.ubin_size,nnn.bu
    print nnn.vbin_size,nnn.bv
    assert nnn.bin_slop == 0.2
    assert nnn.bin_size == 0.1
    assert nnn.ubin_size == 0.03
    assert nnn.vbin_size == 0.07
    numpy.testing.assert_almost_equal(nnn.b, 0.02)
    numpy.testing.assert_almost_equal(nnn.bu, 0.006)
    numpy.testing.assert_almost_equal(nnn.bv, 0.014)

    # Use bin_slop == 0
    nnn = treecorr.NNNCorrelation(min_sep=5, max_sep=20, bin_size=0.1, bin_slop=0.0,
                                  min_u=0.2, max_u=0.9, ubin_size=0.03,
                                  min_v=-0.2, max_v=0.2, vbin_size=0.07)
    print nnn.bin_size,nnn.bin_slop,nnn.b
    print nnn.ubin_size,nnn.bu
    print nnn.vbin_size,nnn.bv
    assert nnn.bin_slop == 0.0
    assert nnn.bin_size == 0.1
    assert nnn.ubin_size == 0.03
    assert nnn.vbin_size == 0.07
    numpy.testing.assert_almost_equal(nnn.b, 0.0)
    numpy.testing.assert_almost_equal(nnn.bu, 0.0)
    numpy.testing.assert_almost_equal(nnn.bv, 0.0)

    # Bigger bin_slop
    nnn = treecorr.NNNCorrelation(min_sep=5, max_sep=20, bin_size=0.1, bin_slop=2.0,
                                  min_u=0.2, max_u=0.9, ubin_size=0.03,
                                  min_v=-0.2, max_v=0.2, vbin_size=0.07)
    print nnn.bin_size,nnn.bin_slop,nnn.b
    print nnn.ubin_size,nnn.bu
    print nnn.vbin_size,nnn.bv
    assert nnn.bin_slop == 2.0
    assert nnn.bin_size == 0.1
    assert nnn.ubin_size == 0.03
    assert nnn.vbin_size == 0.07
    numpy.testing.assert_almost_equal(nnn.b, 0.2)
    numpy.testing.assert_almost_equal(nnn.bu, 0.06)
    numpy.testing.assert_almost_equal(nnn.bv, 0.14)

    # With bin_size > 0.1, explicit bin_slop=1.0 is accepted.
    nnn = treecorr.NNNCorrelation(min_sep=5, max_sep=20, bin_size=0.4, bin_slop=1.0,
                                  min_u=0.2, max_u=0.9, ubin_size=0.03,
                                  min_v=-0.2, max_v=0.2, vbin_size=0.07)
    print nnn.bin_size,nnn.bin_slop,nnn.b
    print nnn.ubin_size,nnn.bu
    print nnn.vbin_size,nnn.bv
    assert nnn.bin_slop == 1.0
    assert nnn.bin_size == 0.4
    assert nnn.ubin_size == 0.03
    assert nnn.vbin_size == 0.07
    numpy.testing.assert_almost_equal(nnn.b, 0.4)
    numpy.testing.assert_almost_equal(nnn.bu, 0.03)
    numpy.testing.assert_almost_equal(nnn.bv, 0.07)

    # But implicit bin_slop is reduced so that b = 0.1
    nnn = treecorr.NNNCorrelation(min_sep=5, max_sep=20, bin_size=0.4,
                                  min_u=0.2, max_u=0.9, ubin_size=0.03,
                                  min_v=-0.2, max_v=0.2, vbin_size=0.07)
    print nnn.bin_size,nnn.bin_slop,nnn.b
    print nnn.ubin_size,nnn.bu
    print nnn.vbin_size,nnn.bv
    assert nnn.bin_size == 0.4
    assert nnn.ubin_size == 0.03
    assert nnn.vbin_size == 0.07
    numpy.testing.assert_almost_equal(nnn.b, 0.1)
    numpy.testing.assert_almost_equal(nnn.bu, 0.03)
    numpy.testing.assert_almost_equal(nnn.bv, 0.07)
    numpy.testing.assert_almost_equal(nnn.bin_slop, 0.25)

    # Separately for each of the three parameters
    nnn = treecorr.NNNCorrelation(min_sep=5, max_sep=20, bin_size=0.05,
                                  min_u=0.2, max_u=0.9, ubin_size=0.3,
                                  min_v=-0.2, max_v=0.2, vbin_size=0.17)
    print nnn.bin_size,nnn.bin_slop,nnn.b
    print nnn.ubin_size,nnn.bu
    print nnn.vbin_size,nnn.bv
    assert nnn.bin_size == 0.05
    assert nnn.ubin_size == 0.3
    assert nnn.vbin_size == 0.17
    numpy.testing.assert_almost_equal(nnn.b, 0.05)
    numpy.testing.assert_almost_equal(nnn.bu, 0.1)
    numpy.testing.assert_almost_equal(nnn.bv, 0.1)
    numpy.testing.assert_almost_equal(nnn.bin_slop, 1.0) # The stored bin_slop is just for lnr


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
    dd = treecorr.NNNCorrelation(min_sep=min_sep, max_sep=max_sep, nbins=nbins, bin_slop=0.)
    dd.process(cat1, cat2)
    print 'dd.npairs = ',dd.npairs

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

    print 'true_npairs = ',true_npairs
    print 'diff = ',dd.npairs - true_npairs
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
    dd = treecorr.NNNCorrelation(min_sep=min_sep, max_sep=max_sep, nbins=nbins, bin_slop=0.)
    dd.process(cat1, cat2)
    print 'dd.npairs = ',dd.npairs

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

    print 'true_npairs = ',true_npairs
    print 'diff = ',dd.npairs - true_npairs
    numpy.testing.assert_array_equal(dd.npairs, true_npairs)

def test_nnn():
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

    ngal = 1000000
    s = 10.
    L = 50. * s  # Not infinity, so this introduces some error.  Our integrals were to infinity.
    numpy.random.seed(8675309)
    x = numpy.random.normal(0,s, (ngal,) )
    y = numpy.random.normal(0,s, (ngal,) )

    cat = treecorr.Catalog(x=x, y=y, x_units='arcmin', y_units='arcmin')
    dd = treecorr.NNNCorrelation(bin_size=0.1, min_sep=1., max_sep=25., sep_units='arcmin',
                                verbose=2)
    dd.process(cat)
    print 'dd.npairs = ',dd.npairs

    nrand = 5 * ngal
    rx = (numpy.random.random_sample(nrand)-0.5) * L
    ry = (numpy.random.random_sample(nrand)-0.5) * L
    rand = treecorr.Catalog(x=rx,y=ry, x_units='arcmin', y_units='arcmin')
    rr = treecorr.NNNCorrelation(bin_size=0.1, min_sep=1., max_sep=25., sep_units='arcmin',
                                verbose=2)
    rr.process(rand)
    print 'rr.npairs = ',rr.npairs

    dr = treecorr.NNNCorrelation(bin_size=0.1, min_sep=1., max_sep=25., sep_units='arcmin',
                                verbose=2)
    dr.process(cat,rand)
    print 'dr.npairs = ',dr.npairs

    r = numpy.exp(dd.meanlogr)
    true_xi = 0.25/numpy.pi * (L/s)**2 * numpy.exp(-0.25*r**2/s**2) - 1.

    xi, varxi = dd.calculateXi(rr,dr)
    print 'xi = ',xi
    print 'true_xi = ',true_xi
    print 'ratio = ',xi / true_xi
    print 'diff = ',xi - true_xi
    print 'max rel diff = ',max(abs((xi - true_xi)/true_xi))
    # This isn't super accurate.  But the agreement improves as L increase, so I think it is 
    # merely a matter of the finite field and the integrals going to infinity.  (Sort of, since
    # we still have L in there.)
    assert max(abs(xi - true_xi)/true_xi) < 0.1

    simple_xi, simple_varxi = dd.calculateXi(rr)
    print 'simple xi = ',simple_xi
    print 'max rel diff = ',max(abs((simple_xi - true_xi)/true_xi))
    # The simple calculation (i.e. dd/rr-1, rather than (dd-2dr+rr)/rr as above) is only 
    # slightly less accurate in this case.  Probably because the mask is simple (a box), so
    # the difference is relatively minor.  The error is slightly higher in this case, but testing
    # that it is everywhere < 0.1 is still appropriate.
    assert max(abs(simple_xi - true_xi)/true_xi) < 0.1

    # Check that we get the same result using the corr2 executable:
    if __name__ == '__main__':
        cat.write(os.path.join('data','nnn_data.dat'))
        rand.write(os.path.join('data','nnn_rand.dat'))
        import subprocess
        p = subprocess.Popen( ["corr2","nnn.params"] )
        p.communicate()
        corr2_output = numpy.loadtxt(os.path.join('output','nnn.out'))
        print 'xi = ',xi
        print 'from corr2 output = ',corr2_output[:,2]
        print 'ratio = ',corr2_output[:,2]/xi
        print 'diff = ',corr2_output[:,2]-xi
        numpy.testing.assert_almost_equal(corr2_output[:,2]/xi, 1., decimal=3)

    # Check the fits write option
    out_file_name1 = os.path.join('output','nnn_out1.fits')
    dd.write(out_file_name1)
    import fitsio
    data = fitsio.read(out_file_name1)
    numpy.testing.assert_almost_equal(data['R_nom'], numpy.exp(dd.logr))
    numpy.testing.assert_almost_equal(data['<R>'], numpy.exp(dd.meanlogr))
    numpy.testing.assert_almost_equal(data['npairs'], dd.npairs)

    out_file_name2 = os.path.join('output','nnn_out2.fits')
    dd.write(out_file_name2, rr)
    data = fitsio.read(out_file_name2)
    numpy.testing.assert_almost_equal(data['R_nom'], numpy.exp(dd.logr))
    numpy.testing.assert_almost_equal(data['<R>'], numpy.exp(dd.meanlogr))
    numpy.testing.assert_almost_equal(data['xi'], simple_xi)
    numpy.testing.assert_almost_equal(data['sigma_xi'], numpy.sqrt(simple_varxi))
    numpy.testing.assert_almost_equal(data['DD'], dd.npairs)
    numpy.testing.assert_almost_equal(data['RR'], rr.npairs * (dd.tot / rr.tot))

    out_file_name3 = os.path.join('output','nnn_out3.fits')
    dd.write(out_file_name3, rr, dr)
    data = fitsio.read(out_file_name3)
    numpy.testing.assert_almost_equal(data['R_nom'], numpy.exp(dd.logr))
    numpy.testing.assert_almost_equal(data['<R>'], numpy.exp(dd.meanlogr))
    numpy.testing.assert_almost_equal(data['xi'], xi)
    numpy.testing.assert_almost_equal(data['sigma_xi'], numpy.sqrt(varxi))
    numpy.testing.assert_almost_equal(data['DD'], dd.npairs)
    numpy.testing.assert_almost_equal(data['RR'], rr.npairs * (dd.tot / rr.tot))
    numpy.testing.assert_almost_equal(data['DR'], dr.npairs * (dd.tot / dr.tot))
    numpy.testing.assert_almost_equal(data['RD'], dr.npairs * (dd.tot / dr.tot))

    # Check the read function
    dd2 = treecorr.NNNCorrelation(bin_size=0.1, min_sep=1., max_sep=25., sep_units='arcmin')
    dd2.read(out_file_name1)
    numpy.testing.assert_almost_equal(dd2.logr, dd.logr)
    numpy.testing.assert_almost_equal(dd2.meanlogr, dd.meanlogr)
    numpy.testing.assert_almost_equal(dd2.npairs, dd.npairs)

    dd2.read(out_file_name3)
    numpy.testing.assert_almost_equal(dd2.logr, dd.logr)
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

    ngal = 100000
    xcen = 823  # Mpc maybe?
    ycen = 342
    zcen = -672
    s = 10.
    L = 50. * s  # Not infinity, so this introduces some error.  Our integrals were to infinity.
    numpy.random.seed(8675309)
    x = numpy.random.normal(xcen, s, (ngal,) )
    y = numpy.random.normal(ycen, s, (ngal,) )
    z = numpy.random.normal(zcen, s, (ngal,) )

    r = numpy.sqrt(x*x+y*y+z*z)
    dec = numpy.arcsin(z/r) / treecorr.degrees
    ra = numpy.arctan2(y,x) / treecorr.degrees

    cat = treecorr.Catalog(ra=ra, dec=dec, r=r, ra_units='deg', dec_units='deg')
    dd = treecorr.NNNCorrelation(bin_size=0.1, min_sep=1., max_sep=25., verbose=2)
    dd.process(cat)
    print 'dd.npairs = ',dd.npairs

    nrand = 5 * ngal
    rx = (numpy.random.random_sample(nrand)-0.5) * L + xcen
    ry = (numpy.random.random_sample(nrand)-0.5) * L + ycen
    rz = (numpy.random.random_sample(nrand)-0.5) * L + zcen
    rr = numpy.sqrt(rx*rx+ry*ry+rz*rz)
    rdec = numpy.arcsin(rz/rr) / treecorr.degrees
    rra = numpy.arctan2(ry,rx) / treecorr.degrees
    rand = treecorr.Catalog(ra=rra, dec=rdec, r=rr, ra_units='deg', dec_units='deg')
    rr = treecorr.NNNCorrelation(bin_size=0.1, min_sep=1., max_sep=25., verbose=2)
    rr.process(rand)
    print 'rr.npairs = ',rr.npairs

    dr = treecorr.NNNCorrelation(bin_size=0.1, min_sep=1., max_sep=25., verbose=2)
    dr.process(cat,rand)
    print 'dr.npairs = ',dr.npairs

    r = numpy.exp(dd.meanlogr)
    true_xi = 1./(8.*numpy.pi**1.5) * (L/s)**3 * numpy.exp(-0.25*r**2/s**2) - 1.

    xi, varxi = dd.calculateXi(rr,dr)
    print 'xi = ',xi
    print 'true_xi = ',true_xi
    print 'ratio = ',xi / true_xi
    print 'diff = ',xi - true_xi
    print 'max rel diff = ',max(abs((xi - true_xi)/true_xi))
    assert max(abs(xi - true_xi)/true_xi) < 0.1

    simple_xi, varxi = dd.calculateXi(rr)
    print 'simple xi = ',simple_xi
    print 'max rel diff = ',max(abs((simple_xi - true_xi)/true_xi))
    assert max(abs(simple_xi - true_xi)/true_xi) < 0.1

    # Check that we get the same result using the corr2 executable:
    if __name__ == '__main__':
        cat.write(os.path.join('data','nnn_3d_data.dat'))
        rand.write(os.path.join('data','nnn_3d_rand.dat'))
        import subprocess
        p = subprocess.Popen( ["corr2","nnn_3d.params"] )
        p.communicate()
        corr2_output = numpy.loadtxt(os.path.join('output','nnn_3d.out'))
        print 'xi = ',xi
        print 'from corr2 output = ',corr2_output[:,2]
        print 'ratio = ',corr2_output[:,2]/xi
        print 'diff = ',corr2_output[:,2]-xi
        numpy.testing.assert_almost_equal(corr2_output[:,2]/xi, 1., decimal=3)


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

    dd = treecorr.NNNCorrelation(bin_size=0.1, min_sep=1., max_sep=25., verbose=2)
    dd.process(data_cats)
    print 'dd.npairs = ',dd.npairs

    rr = treecorr.NNNCorrelation(bin_size=0.1, min_sep=1., max_sep=25., verbose=2)
    rr.process(rand_cats)
    print 'rr.npairs = ',rr.npairs

    xi, varxi = dd.calculateXi(rr)
    print 'xi = ',xi

    # Now do the same thing with one big catalog for each.
    ddx = treecorr.NNNCorrelation(bin_size=0.1, min_sep=1., max_sep=25., verbose=2)
    rrx = treecorr.NNNCorrelation(bin_size=0.1, min_sep=1., max_sep=25., verbose=2)
    data_catx = treecorr.Catalog(x=x.reshape( (nobj*ncats,) ), y=y.reshape( (nobj*ncats,) ))
    rand_catx = treecorr.Catalog(x=rx.reshape( (nobj*ncats,) ), y=ry.reshape( (nobj*ncats,) ))
    ddx.process(data_catx)
    rrx.process(rand_catx)
    xix, varxix = ddx.calculateXi(rrx)

    print 'ddx.npairs = ',ddx.npairs
    print 'rrx.npairs = ',rrx.npairs
    print 'xix = ',xix
    print 'ratio = ',xi/xix
    print 'diff = ',xi-xix
    numpy.testing.assert_almost_equal(xix/xi, 1., decimal=2)

    # Check that we get the same result using the corr2 executable:
    file_list = []
    rand_file_list = []
    for k in range(ncats):
        file_name = os.path.join('data','nnn_list_data%d.dat'%k)
        with open(file_name, 'w') as fid:
            for i in range(nobj):
                fid.write(('%.8f %.8f\n')%(x[i,k],y[i,k]))
        file_list.append(file_name)

        rand_file_name = os.path.join('data','nnn_list_rand%d.dat'%k)
        with open(rand_file_name, 'w') as fid:
            for i in range(nobj):
                fid.write(('%.8f %.8f\n')%(rx[i,k],ry[i,k]))
        rand_file_list.append(rand_file_name)

    list_name = os.path.join('data','nnn_list_data_files.txt')
    with open(list_name, 'w') as fid:
        for file_name in file_list:
            fid.write('%s\n'%file_name)
    rand_list_name = os.path.join('data','nnn_list_rand_files.txt')
    with open(rand_list_name, 'w') as fid:
        for file_name in rand_file_list:
            fid.write('%s\n'%file_name)

    file_namex = os.path.join('data','nnn_list_datax.dat')
    with open(file_namex, 'w') as fid:
        for k in range(ncats):
            for i in range(nobj):
                fid.write(('%.8f %.8f\n')%(x[i,k],y[i,k]))

    rand_file_namex = os.path.join('data','nnn_list_randx.dat')
    with open(rand_file_namex, 'w') as fid:
        for k in range(ncats):
            for i in range(nobj):
                fid.write(('%.8f %.8f\n')%(rx[i,k],ry[i,k]))

    import subprocess
    p = subprocess.Popen( ["corr2","nnn_list1.params"] )
    p.communicate()
    corr2_output = numpy.loadtxt(os.path.join('output','nnn_list1.out'))
    print 'xi = ',xi
    print 'from corr2 output = ',corr2_output[:,2]
    print 'ratio = ',corr2_output[:,2]/xi
    print 'diff = ',corr2_output[:,2]-xi
    numpy.testing.assert_almost_equal(corr2_output[:,2]/xi, 1., decimal=3)

    import subprocess
    p = subprocess.Popen( ["corr2","nnn_list2.params"] )
    p.communicate()
    corr2_output = numpy.loadtxt(os.path.join('output','nnn_list2.out'))
    print 'xi = ',xi
    print 'from corr2 output = ',corr2_output[:,2]
    print 'ratio = ',corr2_output[:,2]/xi
    print 'diff = ',corr2_output[:,2]-xi
    numpy.testing.assert_almost_equal(corr2_output[:,2]/xi, 1., decimal=2)

    import subprocess
    p = subprocess.Popen( ["corr2","nnn_list3.params"] )
    p.communicate()
    corr2_output = numpy.loadtxt(os.path.join('output','nnn_list3.out'))
    print 'xi = ',xi
    print 'from corr2 output = ',corr2_output[:,2]
    print 'ratio = ',corr2_output[:,2]/xi
    print 'diff = ',corr2_output[:,2]-xi
    numpy.testing.assert_almost_equal(corr2_output[:,2]/xi, 1., decimal=2)


if __name__ == '__main__':
    test_binnedcorr3()
    #test_direct_count()
    #test_direct_3d()
    #test_nnn()
    #test_3d()
    #test_list()
