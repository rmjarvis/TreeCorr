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
        #print 'logr = ',nnn.logr1d
        numpy.testing.assert_equal(nnn.logr1d.shape, (nnn.nbins,) )
        numpy.testing.assert_almost_equal(nnn.logr1d[0], math.log(nnn.min_sep) + 0.5*nnn.bin_size)
        numpy.testing.assert_almost_equal(nnn.logr1d[-1], math.log(nnn.max_sep) - 0.5*nnn.bin_size)
        numpy.testing.assert_equal(nnn.logr.shape, (nnn.nbins, nnn.nubins, nnn.nvbins) )
        numpy.testing.assert_almost_equal(nnn.logr[:,0,0], nnn.logr1d)
        numpy.testing.assert_almost_equal(nnn.logr[:,-1,-1], nnn.logr1d)
        assert len(nnn.logr) == nnn.nbins
        #print 'u = ',nnn.u1d
        numpy.testing.assert_equal(nnn.u1d.shape, (nnn.nubins,) )
        numpy.testing.assert_almost_equal(nnn.u1d[0], nnn.min_u + 0.5*nnn.ubin_size)
        numpy.testing.assert_almost_equal(nnn.u1d[-1], nnn.max_u - 0.5*nnn.ubin_size)
        numpy.testing.assert_equal(nnn.u.shape, (nnn.nbins, nnn.nubins, nnn.nvbins) )
        numpy.testing.assert_almost_equal(nnn.u[0,:,0], nnn.u1d)
        numpy.testing.assert_almost_equal(nnn.u[-1,:,-1], nnn.u1d)
        #print 'v = ',nnn.v1d
        numpy.testing.assert_equal(nnn.v1d.shape, (nnn.nvbins,) )
        numpy.testing.assert_almost_equal(nnn.v1d[0], nnn.min_v + 0.5*nnn.vbin_size)
        numpy.testing.assert_almost_equal(nnn.v1d[-1], nnn.max_v - 0.5*nnn.vbin_size)
        numpy.testing.assert_equal(nnn.v.shape, (nnn.nbins, nnn.nubins, nnn.nvbins) )
        numpy.testing.assert_almost_equal(nnn.v[0,0,:], nnn.v1d)
        numpy.testing.assert_almost_equal(nnn.v[-1,-1,:], nnn.v1d)

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
    #print nnn.min_sep,nnn.max_sep,nnn.bin_size,nnn.nbins
    #print nnn.min_u,nnn.max_u,nnn.ubin_size,nnn.nubins
    #print nnn.min_v,nnn.max_v,nnn.vbin_size,nnn.nvbins
    assert nnn.min_sep == 5.
    assert nnn.max_sep == 20.
    assert nnn.nbins == 20
    check_defaultuv(nnn)
    check_arrays(nnn)
    # Specify min, max, n for u,v too.
    nnn = treecorr.NNNCorrelation(min_sep=5, max_sep=20, nbins=20,
                                  min_u=0.2, max_u=0.9, nubins=12,
                                  min_v=-0.2, max_v=0.2, nvbins=4)
    #print nnn.min_sep,nnn.max_sep,nnn.bin_size,nnn.nbins
    #print nnn.min_u,nnn.max_u,nnn.ubin_size,nnn.nubins
    #print nnn.min_v,nnn.max_v,nnn.vbin_size,nnn.nvbins
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
    #print nnn.min_sep,nnn.max_sep,nnn.bin_size,nnn.nbins
    #print nnn.min_u,nnn.max_u,nnn.ubin_size,nnn.nubins
    #print nnn.min_v,nnn.max_v,nnn.vbin_size,nnn.nvbins
    assert nnn.bin_size == 0.1
    assert nnn.max_sep == 20.
    assert nnn.nbins == 20
    check_defaultuv(nnn)
    check_arrays(nnn)
    # Specify max, n, bs for u,v too.
    nnn = treecorr.NNNCorrelation(max_sep=20, nbins=20, bin_size=0.1,
                                  max_u=0.9, nubins=3, ubin_size=0.05,
                                  max_v=0.2, nvbins=4, vbin_size=0.05)
    #print nnn.min_sep,nnn.max_sep,nnn.bin_size,nnn.nbins
    #print nnn.min_u,nnn.max_u,nnn.ubin_size,nnn.nubins
    #print nnn.min_v,nnn.max_v,nnn.vbin_size,nnn.nvbins
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
    #print nnn.min_sep,nnn.max_sep,nnn.bin_size,nnn.nbins
    #print nnn.min_u,nnn.max_u,nnn.ubin_size,nnn.nubins
    #print nnn.min_v,nnn.max_v,nnn.vbin_size,nnn.nvbins
    assert nnn.bin_size == 0.1
    assert nnn.min_sep == 5.
    assert nnn.nbins == 20
    check_defaultuv(nnn)
    check_arrays(nnn)
    # Specify min, n, bs for u,v too.
    nnn = treecorr.NNNCorrelation(min_sep=5, nbins=20, bin_size=0.1,
                                  min_u=0.7, nubins=4, ubin_size=0.05,
                                  min_v=-0.2, nvbins=4, vbin_size=0.05)
    #print nnn.min_sep,nnn.max_sep,nnn.bin_size,nnn.nbins
    #print nnn.min_u,nnn.max_u,nnn.ubin_size,nnn.nubins
    #print nnn.min_v,nnn.max_v,nnn.vbin_size,nnn.nvbins
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
    #print nnn.min_sep,nnn.max_sep,nnn.bin_size,nnn.nbins
    #print nnn.min_u,nnn.max_u,nnn.ubin_size,nnn.nubins
    #print nnn.min_v,nnn.max_v,nnn.vbin_size,nnn.nvbins
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
    #print nnn.min_sep,nnn.max_sep,nnn.bin_size,nnn.nbins
    #print nnn.min_u,nnn.max_u,nnn.ubin_size,nnn.nubins
    #print nnn.min_v,nnn.max_v,nnn.vbin_size,nnn.nvbins
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
    #print nnn.min_sep,nnn.max_sep,nnn.bin_size,nnn.nbins
    #print nnn.min_u,nnn.max_u,nnn.ubin_size,nnn.nubins
    #print nnn.min_v,nnn.max_v,nnn.vbin_size,nnn.nvbins
    assert nnn.min_sep == 5.
    assert nnn.max_sep == 20.
    assert nnn.nbins == 20
    check_defaultuv(nnn)
    check_arrays(nnn)

    # arcsec
    nnn = treecorr.NNNCorrelation(min_sep=5, max_sep=20, nbins=20, sep_units='arcsec')
    #print nnn.min_sep,nnn.max_sep,nnn.bin_size,nnn.nbins
    #print nnn.min_u,nnn.max_u,nnn.ubin_size,nnn.nubins
    #print nnn.min_v,nnn.max_v,nnn.vbin_size,nnn.nvbins
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
    #print nnn.min_sep,nnn.max_sep,nnn.bin_size,nnn.nbins
    #print nnn.min_u,nnn.max_u,nnn.ubin_size,nnn.nubins
    #print nnn.min_v,nnn.max_v,nnn.vbin_size,nnn.nvbins
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
    #print nnn.min_sep,nnn.max_sep,nnn.bin_size,nnn.nbins
    #print nnn.min_u,nnn.max_u,nnn.ubin_size,nnn.nubins
    #print nnn.min_v,nnn.max_v,nnn.vbin_size,nnn.nvbins
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
    #print nnn.min_sep,nnn.max_sep,nnn.bin_size,nnn.nbins
    #print nnn.min_u,nnn.max_u,nnn.ubin_size,nnn.nubins
    #print nnn.min_v,nnn.max_v,nnn.vbin_size,nnn.nvbins
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
    #print nnn.bin_size,nnn.bin_slop,nnn.b
    #print nnn.ubin_size,nnn.bu
    #print nnn.vbin_size,nnn.bv
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
    #print nnn.bin_size,nnn.bin_slop,nnn.b
    #print nnn.ubin_size,nnn.bu
    #print nnn.vbin_size,nnn.bv
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
    #print nnn.bin_size,nnn.bin_slop,nnn.b
    #print nnn.ubin_size,nnn.bu
    #print nnn.vbin_size,nnn.bv
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
    #print nnn.bin_size,nnn.bin_slop,nnn.b
    #print nnn.ubin_size,nnn.bu
    #print nnn.vbin_size,nnn.bv
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
    #print nnn.bin_size,nnn.bin_slop,nnn.b
    #print nnn.ubin_size,nnn.bu
    #print nnn.vbin_size,nnn.bv
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
    #print nnn.bin_size,nnn.bin_slop,nnn.b
    #print nnn.ubin_size,nnn.bu
    #print nnn.vbin_size,nnn.bv
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
    #print nnn.bin_size,nnn.bin_slop,nnn.b
    #print nnn.ubin_size,nnn.bu
    #print nnn.vbin_size,nnn.bv
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
    #print nnn.bin_size,nnn.bin_slop,nnn.b
    #print nnn.ubin_size,nnn.bu
    #print nnn.vbin_size,nnn.bv
    assert nnn.bin_size == 0.05
    assert nnn.ubin_size == 0.3
    assert nnn.vbin_size == 0.17
    numpy.testing.assert_almost_equal(nnn.b, 0.05)
    numpy.testing.assert_almost_equal(nnn.bu, 0.1)
    numpy.testing.assert_almost_equal(nnn.bv, 0.1)
    numpy.testing.assert_almost_equal(nnn.bin_slop, 1.0) # The stored bin_slop is just for lnr


def is_ccw(x1,y1, x2,y2, x3,y3):
    # Calculate the cross product of 1->2 with 1->3
    x2 -= x1
    x3 -= x1
    y2 -= y1
    y3 -= y1
    return x2*y3-x3*y2 > 0.

    
def test_direct_count_auto():
    # If the catalogs are small enough, we can do a direct count of the number of triangles
    # to see if comes out right.  This should exactly match the treecorr code if bin_slop=0.

    ngal = 100
    s = 10.
    numpy.random.seed(8675309)
    x = numpy.random.normal(0,s, (ngal,) )
    y = numpy.random.normal(0,s, (ngal,) )
    cat = treecorr.Catalog(x=x, y=y)

    min_sep = 1.
    max_sep = 50.
    nbins = 50
    min_u = 0.13
    max_u = 0.89
    nubins = 10
    min_v = -0.83
    max_v = 0.59
    nvbins = 20

    ddd = treecorr.NNNCorrelation(min_sep=min_sep, max_sep=max_sep, nbins=nbins, 
                                  min_u=min_u, max_u=max_u, nubins=nubins,
                                  min_v=min_v, max_v=max_v, nvbins=nvbins,
                                  bin_slop=0., verbose=3)
    ddd.process(cat)
    #print 'ddd.ntri = ',ddd.ntri

    log_min_sep = numpy.log(min_sep)
    log_max_sep = numpy.log(max_sep)
    true_ntri = numpy.zeros( (nbins, nubins, nvbins) )
    bin_size = (log_max_sep - log_min_sep) / nbins
    ubin_size = (max_u-min_u) / nubins
    vbin_size = (max_v-min_v) / nvbins
    for i in range(ngal):
        for j in range(i+1,ngal):
            for k in range(j+1,ngal):
                dij = numpy.sqrt((x[i]-x[j])**2 + (y[i]-y[j])**2)
                dik = numpy.sqrt((x[i]-x[k])**2 + (y[i]-y[k])**2)
                djk = numpy.sqrt((x[j]-x[k])**2 + (y[j]-y[k])**2)
                if dij == 0.: continue
                if dik == 0.: continue
                if djk == 0.: continue
                ccw = True
                if dij < dik:
                    if dik < djk:
                        d3 = dij; d2 = dik; d1 = djk;
                        ccw = is_ccw(x[i],y[i],x[j],y[j],x[k],y[k])
                    elif dij < djk:
                        d3 = dij; d2 = djk; d1 = dik;
                        ccw = is_ccw(x[j],y[j],x[i],y[i],x[k],y[k])
                    else:
                        d3 = djk; d2 = dij; d1 = dik;
                        ccw = is_ccw(x[j],y[j],x[k],y[k],x[i],y[i])
                else:
                    if dij < djk:
                        d3 = dik; d2 = dij; d1 = djk;
                        ccw = is_ccw(x[i],y[i],x[k],y[k],x[j],y[j])
                    elif dik < djk:
                        d3 = dik; d2 = djk; d1 = dij;
                        ccw = is_ccw(x[k],y[k],x[i],y[i],x[j],y[j])
                    else:
                        d3 = djk; d2 = dik; d1 = dij;
                        ccw = is_ccw(x[k],y[k],x[j],y[j],x[i],y[i])

                r = d2
                u = d3/d2
                v = (d1-d2)/d3
                if not ccw: 
                    v = -v
                kr = int(numpy.floor( (numpy.log(r)-log_min_sep) / bin_size ))
                ku = int(numpy.floor( (u-min_u) / ubin_size ))
                kv = int(numpy.floor( (v-min_v) / vbin_size ))
                if kr < 0: continue
                if kr >= nbins: continue
                if ku < 0: continue
                if ku >= nubins: continue
                if kv < 0: continue
                if kv >= nvbins: continue
                true_ntri[kr,ku,kv] += 1

    #print 'true_ntri => ',true_ntri
    #print 'diff = ',ddd.ntri - true_ntri
    numpy.testing.assert_array_equal(ddd.ntri, true_ntri)

    # Repeat with binslop not precisely 0, since the code flow is different for bin_slop == 0.
    ddd = treecorr.NNNCorrelation(min_sep=min_sep, max_sep=max_sep, nbins=nbins, 
                                  min_u=min_u, max_u=max_u, nubins=nubins,
                                  min_v=min_v, max_v=max_v, nvbins=nvbins,
                                  bin_slop=1.e-16, verbose=3)
    ddd.process(cat)
    #print 'ddd.ntri = ',ddd.ntri
    #print 'true_ntri => ',true_ntri
    #print 'diff = ',ddd.ntri - true_ntri
    numpy.testing.assert_array_equal(ddd.ntri, true_ntri)

    # And again with no top-level recursion
    ddd = treecorr.NNNCorrelation(min_sep=min_sep, max_sep=max_sep, nbins=nbins, 
                                  min_u=min_u, max_u=max_u, nubins=nubins,
                                  min_v=min_v, max_v=max_v, nvbins=nvbins,
                                  bin_slop=1.e-16, verbose=3, max_top=0)
    ddd.process(cat)
    #print 'ddd.ntri = ',ddd.ntri
    #print 'true_ntri => ',true_ntri
    #print 'diff = ',ddd.ntri - true_ntri
    numpy.testing.assert_array_equal(ddd.ntri, true_ntri)


def test_direct_count_cross():
    # If the catalogs are small enough, we can do a direct count of the number of triangles
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
    x3 = numpy.random.normal(0,s, (ngal,) )
    y3 = numpy.random.normal(0,s, (ngal,) )
    cat3 = treecorr.Catalog(x=x3, y=y3)

    min_sep = 1.
    max_sep = 50.
    nbins = 50
    min_u = 0.13
    max_u = 0.89
    nubins = 10
    min_v = -0.83
    max_v = 0.59
    nvbins = 20

    ddd = treecorr.NNNCorrelation(min_sep=min_sep, max_sep=max_sep, nbins=nbins, 
                                  min_u=min_u, max_u=max_u, nubins=nubins,
                                  min_v=min_v, max_v=max_v, nvbins=nvbins,
                                  bin_slop=0., verbose=3)
    ddd.process(cat1, cat2, cat3)
    #print 'ddd.ntri = ',ddd.ntri

    log_min_sep = numpy.log(min_sep)
    log_max_sep = numpy.log(max_sep)
    true_ntri = numpy.zeros( (nbins, nubins, nvbins) )
    bin_size = (log_max_sep - log_min_sep) / nbins
    ubin_size = (max_u-min_u) / nubins
    vbin_size = (max_v-min_v) / nvbins
    for i in range(ngal):
        for j in range(ngal):
            for k in range(ngal):
                d3 = numpy.sqrt((x1[i]-x2[j])**2 + (y1[i]-y2[j])**2)
                d2 = numpy.sqrt((x1[i]-x3[k])**2 + (y1[i]-y3[k])**2)
                d1 = numpy.sqrt((x2[j]-x3[k])**2 + (y2[j]-y3[k])**2)
                if d3 == 0.: continue
                if d2 == 0.: continue
                if d1 == 0.: continue
                if d1 < d2 or d2 < d3: continue;
                ccw = is_ccw(x1[i],y1[i],x2[j],y2[j],x3[k],y3[k])
                r = d2
                u = d3/d2
                v = (d1-d2)/d3
                if not ccw: 
                    v = -v
                kr = int(numpy.floor( (numpy.log(r)-log_min_sep) / bin_size ))
                ku = int(numpy.floor( (u-min_u) / ubin_size ))
                kv = int(numpy.floor( (v-min_v) / vbin_size ))
                if kr < 0: continue
                if kr >= nbins: continue
                if ku < 0: continue
                if ku >= nubins: continue
                if kv < 0: continue
                if kv >= nvbins: continue
                true_ntri[kr,ku,kv] += 1

    #print 'true_ntri = ',true_ntri
    #print 'diff = ',ddd.ntri - true_ntri
    numpy.testing.assert_array_equal(ddd.ntri, true_ntri)

    # Repeat with binslop not precisely 0, since the code flow is different for bin_slop == 0.
    ddd = treecorr.NNNCorrelation(min_sep=min_sep, max_sep=max_sep, nbins=nbins, 
                                  min_u=min_u, max_u=max_u, nubins=nubins,
                                  min_v=min_v, max_v=max_v, nvbins=nvbins,
                                  bin_slop=1.e-16, verbose=3)
    ddd.process(cat1, cat2, cat3)
    #print 'binslop > 0: ddd.ntri = ',ddd.ntri
    #print 'diff = ',ddd.ntri - true_ntri
    numpy.testing.assert_array_equal(ddd.ntri, true_ntri)

    # And again with no top-level recursion
    ddd = treecorr.NNNCorrelation(min_sep=min_sep, max_sep=max_sep, nbins=nbins, 
                                  min_u=min_u, max_u=max_u, nubins=nubins,
                                  min_v=min_v, max_v=max_v, nvbins=nvbins,
                                  bin_slop=1.e-16, verbose=3, max_top=0)
    ddd.process(cat1, cat2, cat3)
    #print 'max_top = 0: ddd.ntri = ',ddd.ntri
    #print 'true_ntri = ',true_ntri
    #print 'diff = ',ddd.ntri - true_ntri
    numpy.testing.assert_array_equal(ddd.ntri, true_ntri)


def is_ccw_3d(x1,y1,z1, x2,y2,z2, x3,y3,z3):
    # Calculate the cross product of 1->2 with 1->3
    x2 -= x1
    x3 -= x1
    y2 -= y1
    y3 -= y1
    z2 -= z1
    z3 -= z1

    # The cross product:
    x = y2*z3-y3*z2
    y = z2*x3-z3*x2
    z = x2*y3-x3*y2

    # ccw if the cross product is in the opposite direction of (x1,y1,z1) from (0,0,0)
    return x*x1 + y*y1 + z*z1 < 0.

def test_direct_3d_auto():
    # This is the same as the above test, but using the 3d correlations

    ngal = 100
    s = 10.
    numpy.random.seed(8675309)
    x = numpy.random.normal(312, s, (ngal,) )
    y = numpy.random.normal(728, s, (ngal,) )
    z = numpy.random.normal(-932, s, (ngal,) )
    r = numpy.sqrt( x*x + y*y + z*z )
    dec = numpy.arcsin(z/r)
    ra = numpy.arctan2(y,x)
    cat = treecorr.Catalog(ra=ra, dec=dec, r=r, ra_units='rad', dec_units='rad')

    min_sep = 1.
    max_sep = 50.
    nbins = 50
    min_u = 0.13
    max_u = 0.89
    nubins = 10
    min_v = -0.83
    max_v = 0.59
    nvbins = 20
    ddd = treecorr.NNNCorrelation(min_sep=min_sep, max_sep=max_sep, nbins=nbins, 
                                  min_u=min_u, max_u=max_u, nubins=nubins,
                                  min_v=min_v, max_v=max_v, nvbins=nvbins,
                                  bin_slop=0., verbose=3)
    ddd.process(cat)
    #print 'ddd.ntri = ',ddd.ntri

    log_min_sep = numpy.log(min_sep)
    log_max_sep = numpy.log(max_sep)
    true_ntri = numpy.zeros( (nbins, nubins, nvbins) )
    bin_size = (log_max_sep - log_min_sep) / nbins
    ubin_size = (max_u-min_u) / nubins
    vbin_size = (max_v-min_v) / nvbins
    for i in range(ngal):
        for j in range(i+1,ngal):
            for k in range(j+1,ngal):
                dij = numpy.sqrt((x[i]-x[j])**2 + (y[i]-y[j])**2 + (z[i]-z[j])**2)
                dik = numpy.sqrt((x[i]-x[k])**2 + (y[i]-y[k])**2 + (z[i]-z[k])**2)
                djk = numpy.sqrt((x[j]-x[k])**2 + (y[j]-y[k])**2 + (z[j]-z[k])**2)
                if dij == 0.: continue
                if dik == 0.: continue
                if djk == 0.: continue
                ccw = True

                if dij < dik:
                    if dik < djk:
                        d3 = dij; d2 = dik; d1 = djk;
                        ccw = is_ccw_3d(x[i],y[i],z[i],x[j],y[j],z[j],x[k],y[k],z[k])
                    elif dij < djk:
                        d3 = dij; d2 = djk; d1 = dik;
                        ccw = is_ccw_3d(x[j],y[j],z[j],x[i],y[i],z[i],x[k],y[k],z[k])
                    else:
                        d3 = djk; d2 = dij; d1 = dik;
                        ccw = is_ccw_3d(x[j],y[j],z[j],x[k],y[k],z[k],x[i],y[i],z[i])
                else:
                    if dij < djk:
                        d3 = dik; d2 = dij; d1 = djk;
                        ccw = is_ccw_3d(x[i],y[i],z[i],x[k],y[k],z[k],x[j],y[j],z[j])
                    elif dik < djk:
                        d3 = dik; d2 = djk; d1 = dij;
                        ccw = is_ccw_3d(x[k],y[k],z[k],x[i],y[i],z[i],x[j],y[j],z[j])
                    else:
                        d3 = djk; d2 = dik; d1 = dij;
                        ccw = is_ccw_3d(x[k],y[k],z[k],x[j],y[j],z[j],x[i],y[i],z[i])

                r = d2
                u = d3/d2
                v = (d1-d2)/d3
                if not ccw: 
                    v = -v
                kr = int(numpy.floor( (numpy.log(r)-log_min_sep) / bin_size ))
                ku = int(numpy.floor( (u-min_u) / ubin_size ))
                kv = int(numpy.floor( (v-min_v) / vbin_size ))
                if kr < 0: continue
                if kr >= nbins: continue
                if ku < 0: continue
                if ku >= nubins: continue
                if kv < 0: continue
                if kv >= nvbins: continue
                true_ntri[kr,ku,kv] += 1

    #print 'true_ntri => ',true_ntri
    #print 'diff = ',ddd.ntri - true_ntri
    numpy.testing.assert_array_equal(ddd.ntri, true_ntri)

    # Repeat with binslop not precisely 0, since the code flow is different for bin_slop == 0.
    ddd = treecorr.NNNCorrelation(min_sep=min_sep, max_sep=max_sep, nbins=nbins, 
                                  min_u=min_u, max_u=max_u, nubins=nubins,
                                  min_v=min_v, max_v=max_v, nvbins=nvbins,
                                  bin_slop=1.e-16, verbose=3)
    ddd.process(cat)
    #print 'ddd.ntri = ',ddd.ntri
    #print 'diff = ',ddd.ntri - true_ntri
    numpy.testing.assert_array_equal(ddd.ntri, true_ntri)

    # And again with no top-level recursion
    ddd = treecorr.NNNCorrelation(min_sep=min_sep, max_sep=max_sep, nbins=nbins, 
                                  min_u=min_u, max_u=max_u, nubins=nubins,
                                  min_v=min_v, max_v=max_v, nvbins=nvbins,
                                  bin_slop=1.e-16, verbose=3, max_top=0)
    ddd.process(cat)
    #print 'ddd.ntri = ',ddd.ntri
    #print 'true_ntri => ',true_ntri
    #print 'diff = ',ddd.ntri - true_ntri
    numpy.testing.assert_array_equal(ddd.ntri, true_ntri)


def test_direct_3d_cross():
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

    x3 = numpy.random.normal(312, s, (ngal,) )
    y3 = numpy.random.normal(728, s, (ngal,) )
    z3 = numpy.random.normal(-932, s, (ngal,) )
    r3 = numpy.sqrt( x3*x3 + y3*y3 + z3*z3 )
    dec3 = numpy.arcsin(z3/r3)
    ra3 = numpy.arctan2(y3,x3)
    cat3 = treecorr.Catalog(ra=ra3, dec=dec3, r=r3, ra_units='rad', dec_units='rad')

    min_sep = 1.
    max_sep = 50.
    nbins = 50
    min_u = 0.13
    max_u = 0.89
    nubins = 10
    min_v = -0.83
    max_v = 0.59
    nvbins = 20
    ddd = treecorr.NNNCorrelation(min_sep=min_sep, max_sep=max_sep, nbins=nbins, 
                                  min_u=min_u, max_u=max_u, nubins=nubins,
                                  min_v=min_v, max_v=max_v, nvbins=nvbins,
                                  bin_slop=0., verbose=3)
    ddd.process(cat1, cat2, cat3)
    #print 'ddd.ntri = ',ddd.ntri

    log_min_sep = numpy.log(min_sep)
    log_max_sep = numpy.log(max_sep)
    true_ntri = numpy.zeros( (nbins, nubins, nvbins) )
    bin_size = (log_max_sep - log_min_sep) / nbins
    ubin_size = (max_u-min_u) / nubins
    vbin_size = (max_v-min_v) / nvbins
    for i in range(ngal):
        for j in range(ngal):
            for k in range(ngal):
                d1sq = (x2[j]-x3[k])**2 + (y2[j]-y3[k])**2 + (z2[j]-z3[k])**2
                d2sq = (x1[i]-x3[k])**2 + (y1[i]-y3[k])**2 + (z1[i]-z3[k])**2
                d3sq = (x1[i]-x2[j])**2 + (y1[i]-y2[j])**2 + (z1[i]-z2[j])**2
                d1 = numpy.sqrt(d1sq)
                d2 = numpy.sqrt(d2sq)
                d3 = numpy.sqrt(d3sq)
                if d3 == 0.: continue
                if d2 == 0.: continue
                if d1 == 0.: continue
                if d1 < d2 or d2 < d3: continue;
                ccw = is_ccw_3d(x1[i],y1[i],z1[i],x2[j],y2[j],z2[j],x3[k],y3[k],z3[k])
                r = d2
                u = d3/d2
                v = (d1-d2)/d3
                if not ccw: 
                    v = -v
                kr = int(numpy.floor( (numpy.log(r)-log_min_sep) / bin_size ))
                ku = int(numpy.floor( (u-min_u) / ubin_size ))
                kv = int(numpy.floor( (v-min_v) / vbin_size ))
                if kr < 0: continue
                if kr >= nbins: continue
                if ku < 0: continue
                if ku >= nubins: continue
                if kv < 0: continue
                if kv >= nvbins: continue
                true_ntri[kr,ku,kv] += 1

    #print 'true_ntri = ',true_ntri
    #print 'diff = ',ddd.ntri - true_ntri
    numpy.testing.assert_array_equal(ddd.ntri, true_ntri)

    # Repeat with binslop not precisely 0, since the code flow is different for bin_slop == 0.
    ddd = treecorr.NNNCorrelation(min_sep=min_sep, max_sep=max_sep, nbins=nbins, 
                                  min_u=min_u, max_u=max_u, nubins=nubins,
                                  min_v=min_v, max_v=max_v, nvbins=nvbins,
                                  bin_slop=1.e-16, verbose=3)
    ddd.process(cat1, cat2, cat3)
    #print 'binslop > 0: ddd.ntri = ',ddd.ntri
    #print 'diff = ',ddd.ntri - true_ntri
    numpy.testing.assert_array_equal(ddd.ntri, true_ntri)

    # And again with no top-level recursion
    ddd = treecorr.NNNCorrelation(min_sep=min_sep, max_sep=max_sep, nbins=nbins, 
                                  min_u=min_u, max_u=max_u, nubins=nubins,
                                  min_v=min_v, max_v=max_v, nvbins=nvbins,
                                  bin_slop=1.e-16, verbose=3, max_top=0)
    ddd.process(cat1, cat2, cat3)
    #print 'max_top = 0: ddd.ntri = ',ddd.ntri
    #print 'true_ntri = ',true_ntri
    #print 'diff = ',ddd.ntri - true_ntri
    numpy.testing.assert_array_equal(ddd.ntri, true_ntri)


def test_nnn():
    # Use a simple probability distribution for the galaxies:
    #
    # n(r) = (2pi s^2)^-1 exp(-r^2/2s^2)
    #
    # The Fourier transform is: n~(k) = exp(-s^2 k^2/2)
    # B(k1,k2) = <n~(k1) n~(k2) n~(-k1-k2)>
    #          = exp(-s^2 (|k1|^2 + |k2|^2 - k1.k2))
    #          = exp(-s^2 (|k1|^2 + |k2|^2 + |k3|^2)/2)
    #
    # zeta(r1,r2) = (1/2pi)^4 int(d^2k1 int(d^2k2 exp(ik1.x1) exp(ik2.x2) B(k1,k2) ))
    #             = exp(-(x1^2 + y1^2 + x2^2 + y2^2 - x1x2 - y1y2)/3s^2) / 12 pi^2 s^4
    #             = exp(-(d1^2 + d2^2 + d3^2)/6s^2) / 12 pi^2 s^4
    #
    # This is also derivable as:
    # zeta(r1,r2) = int(dx int(dy n(x,y) n(x+x1,y+y1) n(x+x2,y+y2)))
    # which is also analytically integrable and gives the same answer.
    #
    # However, we need to correct for the uniform density background, so the real result
    # is this minus 1/L^4 divided by 1/L^4.  So:
    #
    # zeta(r1,r2) = 1/(12 pi^2) (L/s)^4 exp(-(d1^2+d2^2+d3^2)/6s^2) - 1

    # Doing the full correlation function takes a long time.  Here, we just test a small range
    # of separations and a moderate range for u, v, which gives us a variety of triangle lengths.
    ngal = 20000
    s = 10.
    L = 50. * s  # Not infinity, so this introduces some error.  Our integrals were to infinity.
    numpy.random.seed(8675309)
    x = numpy.random.normal(0,s, (ngal,) )
    y = numpy.random.normal(0,s, (ngal,) )
    min_sep = 11.
    max_sep = 13.
    nbins = 2
    min_u = 0.6
    max_u = 0.9
    nubins = 3
    min_v = -0.7
    max_v = 0.7
    nvbins = 10

    cat = treecorr.Catalog(x=x, y=y, x_units='arcmin', y_units='arcmin')
    ddd = treecorr.NNNCorrelation(min_sep=min_sep, max_sep=max_sep, nbins=nbins,
                                  min_u=min_u, max_u=max_u, min_v=min_v, max_v=max_v,
                                  nubins=nubins, nvbins=nvbins,
                                  sep_units='arcmin', verbose=3)
    ddd.process(cat)
    #print 'ddd.ntri = ',ddd.ntri

    nrand = 2 * ngal
    rx = (numpy.random.random_sample(nrand)-0.5) * L
    ry = (numpy.random.random_sample(nrand)-0.5) * L
    rand = treecorr.Catalog(x=rx,y=ry, x_units='arcmin', y_units='arcmin')
    rrr = treecorr.NNNCorrelation(min_sep=min_sep, max_sep=max_sep, nbins=nbins,
                                  min_u=min_u, max_u=max_u, min_v=min_v, max_v=max_v,
                                  nubins=nubins, nvbins=nvbins,
                                  sep_units='arcmin', verbose=3)
    rrr.process(rand)
    #print 'rrr.ntri = ',rrr.ntri

    r = numpy.exp(ddd.meanlogr)
    u = ddd.meanu
    v = ddd.meanv
    d2 = r
    d3 = u * r
    d1 = numpy.abs(v) * d3 + d2
    #print 'rnom = ',numpy.exp(ddd.logr)
    #print 'unom = ',ddd.u
    #print 'vnom = ',ddd.v
    #print 'r = ',r
    #print 'u = ',u
    #print 'v = ',v
    #print 'd2 = ',d2
    #print 'd3 = ',d3
    #print 'd1 = ',d1
    true_zeta = (1./(12.*numpy.pi**2)) * (L/s)**4 * numpy.exp(-(d1**2+d2**2+d3**2)/(6.*s**2)) - 1.

    zeta, varzeta = ddd.calculateZeta(rrr)
    print 'zeta = ',zeta
    print 'true_zeta = ',true_zeta
    print 'ratio = ',zeta / true_zeta
    print 'diff = ',zeta - true_zeta
    print 'max rel diff = ',numpy.max(numpy.abs((zeta - true_zeta)/true_zeta))
    # The simple calculation (i.e. ddd/rrr-1, rather than (ddd-3ddr+3drr-rrr)/rrr as above) is only 
    # slightly less accurate in this case.  Probably because the mask is simple (a box), so
    # the difference is relatively minor.  The error is slightly higher in this case, but testing
    # that it is everywhere < 0.1 is still appropriate.
    assert numpy.max(numpy.abs(zeta - true_zeta)/true_zeta) < 0.1

    # Check that we get the same result using the corr2 executable:
    #if __name__ == '__main__':
    if False:
        cat.write(os.path.join('data','nnn_data.dat'))
        rand.write(os.path.join('data','nnn_rand.dat'))
        import subprocess
        p = subprocess.Popen( ["corr2","nnn.params"] )
        p.communicate()
        corr2_output = numpy.loadtxt(os.path.join('output','nnn.out'))
        print 'zeta = ',simple_zeta
        print 'from corr2 output = ',corr2_output[:,2]
        print 'ratio = ',corr2_output[:,2]/zeta
        print 'diff = ',corr2_output[:,2]-zeta
        numpy.testing.assert_almost_equal(corr2_output[:,2]/zeta, 1., decimal=3)

    # Check the fits write option
    out_file_name1 = os.path.join('output','nnn_out1.fits')
    ddd.write(out_file_name1)
    import fitsio
    data = fitsio.read(out_file_name1)
    numpy.testing.assert_almost_equal(data['R_nom'], numpy.exp(ddd.logr).flatten())
    numpy.testing.assert_almost_equal(data['<R>'], numpy.exp(ddd.meanlogr).flatten())
    numpy.testing.assert_almost_equal(data['ntri'], ddd.ntri.flatten())

    out_file_name2 = os.path.join('output','nnn_out2.fits')
    ddd.write(out_file_name2, rrr)
    data = fitsio.read(out_file_name2)
    numpy.testing.assert_almost_equal(data['R_nom'], numpy.exp(ddd.logr).flatten())
    numpy.testing.assert_almost_equal(data['<R>'], numpy.exp(ddd.meanlogr).flatten())
    numpy.testing.assert_almost_equal(data['zeta'], zeta.flatten())
    numpy.testing.assert_almost_equal(data['sigma_zeta'], numpy.sqrt(varzeta).flatten())
    numpy.testing.assert_almost_equal(data['DDD'], ddd.ntri.flatten())
    numpy.testing.assert_almost_equal(data['RRR'], rrr.ntri.flatten() * (ddd.tot / rrr.tot))

    # Check the read function
    # Note: These don't need the flatten. The read function should reshape them to the right shape.
    ddd2 = treecorr.NNNCorrelation(min_sep=min_sep, max_sep=max_sep, nbins=nbins,
                                   min_u=min_u, max_u=max_u, min_v=min_v, max_v=max_v,
                                   nubins=nubins, nvbins=nvbins,
                                   sep_units='arcmin', verbose=3)
    ddd2.read(out_file_name1)
    numpy.testing.assert_almost_equal(ddd2.logr, ddd.logr)
    numpy.testing.assert_almost_equal(ddd2.meanlogr, ddd.meanlogr)
    numpy.testing.assert_almost_equal(ddd2.ntri, ddd.ntri)

    ddd2.read(out_file_name2)
    numpy.testing.assert_almost_equal(ddd2.logr, ddd.logr)
    numpy.testing.assert_almost_equal(ddd2.meanlogr, ddd.meanlogr)
    numpy.testing.assert_almost_equal(ddd2.ntri, ddd.ntri)

    # Test compensated zeta
    # Note: I don't think this is actually right. The error is more like 0.075, rather than 
    #       0.05 for simple. I think the problem is that DDR is not nearly zero like DRR and RRR.
    #       It has the 2pt correlation still. So I'm not sure if this is really the right thing
    #       to do for the compensated zeta.  Still, it does pass the test, since the values are
    #       so large that the DDR result is much less than DDD.
    if __name__ == '__main__':
        ddr = treecorr.NNNCorrelation(min_sep=min_sep, max_sep=max_sep, nbins=nbins,
                                      min_u=min_u, max_u=max_u, min_v=min_v, max_v=max_v,
                                      nubins=nubins, nvbins=nvbins,
                                      sep_units='arcmin', verbose=3)
        ddr.process(cat,cat,rand)
        #print 'ddr.ntri = ',ddr.ntri

        drr = treecorr.NNNCorrelation(min_sep=min_sep, max_sep=max_sep, nbins=nbins,
                                      min_u=min_u, max_u=max_u, min_v=min_v, max_v=max_v,
                                      nubins=nubins, nvbins=nvbins,
                                      sep_units='arcmin', verbose=3)
        drr.process(cat,rand,rand)
        #print 'drr.ntri = ',drr.ntri

        zeta, varzeta = ddd.calculateZeta(rrr,drr,ddr)
        print 'compensated zeta = ',zeta
        print 'true_zeta = ',true_zeta
        print 'ratio = ',zeta / true_zeta
        print 'diff = ',zeta - true_zeta
        print 'max rel diff = ',numpy.max(numpy.abs((zeta - true_zeta)/true_zeta))
        assert numpy.max(numpy.abs(zeta - true_zeta)/true_zeta) < 0.1

        out_file_name3 = os.path.join('output','nnn_out3.fits')
        ddd.write(out_file_name3, rrr, drr, ddr)
        data = fitsio.read(out_file_name3)
        numpy.testing.assert_almost_equal(data['R_nom'], numpy.exp(ddd.logr).flatten())
        numpy.testing.assert_almost_equal(data['<R>'], numpy.exp(ddd.meanlogr).flatten())
        numpy.testing.assert_almost_equal(data['zeta'], zeta.flatten())
        numpy.testing.assert_almost_equal(data['sigma_zeta'], numpy.sqrt(varzeta).flatten())
        numpy.testing.assert_almost_equal(data['DDD'], ddd.ntri.flatten())
        numpy.testing.assert_almost_equal(data['RRR'], rrr.ntri.flatten() * (ddd.tot / rrr.tot))
        numpy.testing.assert_almost_equal(data['DDR'], ddr.ntri.flatten() * (ddd.tot / ddr.tot))
        numpy.testing.assert_almost_equal(data['DRR'], drr.ntri.flatten() * (ddd.tot / drr.tot))

        ddd2.read(out_file_name3)
        numpy.testing.assert_almost_equal(ddd2.logr, ddd.logr)
        numpy.testing.assert_almost_equal(ddd2.meanlogr, ddd.meanlogr)
        numpy.testing.assert_almost_equal(ddd2.ntri, ddd.ntri)



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

    ngal = 5000
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
    dd = treecorr.NNNCorrelation(bin_size=0.1, min_sep=1., max_sep=25., verbose=3)
    dd.process(cat)
    print 'dd.ntri = ',dd.ntri

    nrand = 5 * ngal
    rx = (numpy.random.random_sample(nrand)-0.5) * L + xcen
    ry = (numpy.random.random_sample(nrand)-0.5) * L + ycen
    rz = (numpy.random.random_sample(nrand)-0.5) * L + zcen
    rr = numpy.sqrt(rx*rx+ry*ry+rz*rz)
    rdec = numpy.arcsin(rz/rr) / treecorr.degrees
    rra = numpy.arctan2(ry,rx) / treecorr.degrees
    rand = treecorr.Catalog(ra=rra, dec=rdec, r=rr, ra_units='deg', dec_units='deg')
    rr = treecorr.NNNCorrelation(bin_size=0.1, min_sep=1., max_sep=25., verbose=3)
    rr.process(rand)
    print 'rr.ntri = ',rr.ntri

    dr = treecorr.NNNCorrelation(bin_size=0.1, min_sep=1., max_sep=25., verbose=3)
    dr.process(cat,rand)
    print 'dr.ntri = ',dr.ntri

    r = numpy.exp(dd.meanlogr)
    true_xi = 1./(8.*numpy.pi**1.5) * (L/s)**3 * numpy.exp(-0.25*r**2/s**2) - 1.

    xi, varxi = dd.calculateXi(rr,dr)
    print 'xi = ',xi
    print 'true_xi = ',true_xi
    print 'ratio = ',xi / true_xi
    print 'diff = ',xi - true_xi
    print 'max rel diff = ',numpy.max(numpy.abs((xi - true_xi)/true_xi))
    assert numpy.max(numpy.abs(xi - true_xi)/true_xi) < 0.1

    simple_xi, varxi = dd.calculateXi(rr)
    print 'simple xi = ',simple_xi
    print 'max rel diff = ',numpy.max(numpy.abs((simple_xi - true_xi)/true_xi))
    assert numpy.max(numpy.abs(simple_xi - true_xi)/true_xi) < 0.1

    # Check that we get the same result using the corr2 executable:
    #if __name__ == '__main__':
    if False:
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

    dd = treecorr.NNNCorrelation(bin_size=0.1, min_sep=1., max_sep=25., verbose=3)
    dd.process(data_cats)
    print 'dd.ntri = ',dd.ntri

    rr = treecorr.NNNCorrelation(bin_size=0.1, min_sep=1., max_sep=25., verbose=3)
    rr.process(rand_cats)
    print 'rr.ntri = ',rr.ntri

    xi, varxi = dd.calculateXi(rr)
    print 'xi = ',xi

    # Now do the same thing with one big catalog for each.
    ddx = treecorr.NNNCorrelation(bin_size=0.1, min_sep=1., max_sep=25., verbose=3)
    rrx = treecorr.NNNCorrelation(bin_size=0.1, min_sep=1., max_sep=25., verbose=3)
    data_catx = treecorr.Catalog(x=x.reshape( (nobj*ncats,) ), y=y.reshape( (nobj*ncats,) ))
    rand_catx = treecorr.Catalog(x=rx.reshape( (nobj*ncats,) ), y=ry.reshape( (nobj*ncats,) ))
    ddx.process(data_catx)
    rrx.process(rand_catx)
    xix, varxix = ddx.calculateXi(rrx)

    print 'ddx.ntri = ',ddx.ntri
    print 'rrx.ntri = ',rrx.ntri
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
    #test_binnedcorr3()
    #test_direct_count_auto()
    #test_direct_count_cross()
    #test_direct_3d_auto()
    #test_direct_3d_cross()
    test_nnn()
    #test_3d()
    #test_list()
