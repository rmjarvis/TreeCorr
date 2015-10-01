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

def test_binnedcorr3():
    import math
    # Test some basic properties of the base class

    def check_arrays(nnn):
        numpy.testing.assert_almost_equal(nnn.bin_size * nnn.nbins, math.log(nnn.max_sep/nnn.min_sep))
        numpy.testing.assert_almost_equal(nnn.ubin_size * nnn.nubins, nnn.max_u-nnn.min_u)
        numpy.testing.assert_almost_equal(nnn.vbin_size * nnn.nvbins, nnn.max_v-nnn.min_v)
        #print('logr = ',nnn.logr1d)
        numpy.testing.assert_equal(nnn.logr1d.shape, (nnn.nbins,) )
        numpy.testing.assert_almost_equal(nnn.logr1d[0], math.log(nnn.min_sep) + 0.5*nnn.bin_size)
        numpy.testing.assert_almost_equal(nnn.logr1d[-1], math.log(nnn.max_sep) - 0.5*nnn.bin_size)
        numpy.testing.assert_equal(nnn.logr.shape, (nnn.nbins, nnn.nubins, nnn.nvbins) )
        numpy.testing.assert_almost_equal(nnn.logr[:,0,0], nnn.logr1d)
        numpy.testing.assert_almost_equal(nnn.logr[:,-1,-1], nnn.logr1d)
        assert len(nnn.logr) == nnn.nbins
        #print('u = ',nnn.u1d)
        numpy.testing.assert_equal(nnn.u1d.shape, (nnn.nubins,) )
        numpy.testing.assert_almost_equal(nnn.u1d[0], nnn.min_u + 0.5*nnn.ubin_size)
        numpy.testing.assert_almost_equal(nnn.u1d[-1], nnn.max_u - 0.5*nnn.ubin_size)
        numpy.testing.assert_equal(nnn.u.shape, (nnn.nbins, nnn.nubins, nnn.nvbins) )
        numpy.testing.assert_almost_equal(nnn.u[0,:,0], nnn.u1d)
        numpy.testing.assert_almost_equal(nnn.u[-1,:,-1], nnn.u1d)
        #print('v = ',nnn.v1d)
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
    #print(nnn.min_sep,nnn.max_sep,nnn.bin_size,nnn.nbins)
    #print(nnn.min_u,nnn.max_u,nnn.ubin_size,nnn.nubins)
    #print(nnn.min_v,nnn.max_v,nnn.vbin_size,nnn.nvbins)
    assert nnn.min_sep == 5.
    assert nnn.max_sep == 20.
    assert nnn.nbins == 20
    check_defaultuv(nnn)
    check_arrays(nnn)
    # Specify min, max, n for u,v too.
    nnn = treecorr.NNNCorrelation(min_sep=5, max_sep=20, nbins=20,
                                  min_u=0.2, max_u=0.9, nubins=12,
                                  min_v=-0.2, max_v=0.2, nvbins=4)
    #print(nnn.min_sep,nnn.max_sep,nnn.bin_size,nnn.nbins)
    #print(nnn.min_u,nnn.max_u,nnn.ubin_size,nnn.nubins)
    #print(nnn.min_v,nnn.max_v,nnn.vbin_size,nnn.nvbins)
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
    #print(nnn.min_sep,nnn.max_sep,nnn.bin_size,nnn.nbins)
    #print(nnn.min_u,nnn.max_u,nnn.ubin_size,nnn.nubins)
    #print(nnn.min_v,nnn.max_v,nnn.vbin_size,nnn.nvbins)
    assert nnn.bin_size == 0.1
    assert nnn.max_sep == 20.
    assert nnn.nbins == 20
    check_defaultuv(nnn)
    check_arrays(nnn)
    # Specify max, n, bs for u,v too.
    nnn = treecorr.NNNCorrelation(max_sep=20, nbins=20, bin_size=0.1,
                                  max_u=0.9, nubins=3, ubin_size=0.05,
                                  max_v=0.2, nvbins=4, vbin_size=0.05)
    #print(nnn.min_sep,nnn.max_sep,nnn.bin_size,nnn.nbins)
    #print(nnn.min_u,nnn.max_u,nnn.ubin_size,nnn.nubins)
    #print(nnn.min_v,nnn.max_v,nnn.vbin_size,nnn.nvbins)
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
    #print(nnn.min_sep,nnn.max_sep,nnn.bin_size,nnn.nbins)
    #print(nnn.min_u,nnn.max_u,nnn.ubin_size,nnn.nubins)
    #print(nnn.min_v,nnn.max_v,nnn.vbin_size,nnn.nvbins)
    assert nnn.bin_size == 0.1
    assert nnn.min_sep == 5.
    assert nnn.nbins == 20
    check_defaultuv(nnn)
    check_arrays(nnn)
    # Specify min, n, bs for u,v too.
    nnn = treecorr.NNNCorrelation(min_sep=5, nbins=20, bin_size=0.1,
                                  min_u=0.7, nubins=4, ubin_size=0.05,
                                  min_v=-0.2, nvbins=4, vbin_size=0.05)
    #print(nnn.min_sep,nnn.max_sep,nnn.bin_size,nnn.nbins)
    #print(nnn.min_u,nnn.max_u,nnn.ubin_size,nnn.nubins)
    #print(nnn.min_v,nnn.max_v,nnn.vbin_size,nnn.nvbins)
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
    #print(nnn.min_sep,nnn.max_sep,nnn.bin_size,nnn.nbins)
    #print(nnn.min_u,nnn.max_u,nnn.ubin_size,nnn.nubins)
    #print(nnn.min_v,nnn.max_v,nnn.vbin_size,nnn.nvbins)
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
    #print(nnn.min_sep,nnn.max_sep,nnn.bin_size,nnn.nbins)
    #print(nnn.min_u,nnn.max_u,nnn.ubin_size,nnn.nubins)
    #print(nnn.min_v,nnn.max_v,nnn.vbin_size,nnn.nvbins)
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
    #print(nnn.min_sep,nnn.max_sep,nnn.bin_size,nnn.nbins)
    #print(nnn.min_u,nnn.max_u,nnn.ubin_size,nnn.nubins)
    #print(nnn.min_v,nnn.max_v,nnn.vbin_size,nnn.nvbins)
    assert nnn.min_sep == 5.
    assert nnn.max_sep == 20.
    assert nnn.nbins == 20
    check_defaultuv(nnn)
    check_arrays(nnn)

    # arcsec
    nnn = treecorr.NNNCorrelation(min_sep=5, max_sep=20, nbins=20, sep_units='arcsec')
    #print(nnn.min_sep,nnn.max_sep,nnn.bin_size,nnn.nbins)
    #print(nnn.min_u,nnn.max_u,nnn.ubin_size,nnn.nubins)
    #print(nnn.min_v,nnn.max_v,nnn.vbin_size,nnn.nvbins)
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
    #print(nnn.min_sep,nnn.max_sep,nnn.bin_size,nnn.nbins)
    #print(nnn.min_u,nnn.max_u,nnn.ubin_size,nnn.nubins)
    #print(nnn.min_v,nnn.max_v,nnn.vbin_size,nnn.nvbins)
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
    #print(nnn.min_sep,nnn.max_sep,nnn.bin_size,nnn.nbins)
    #print(nnn.min_u,nnn.max_u,nnn.ubin_size,nnn.nubins)
    #print(nnn.min_v,nnn.max_v,nnn.vbin_size,nnn.nvbins)
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
    #print(nnn.min_sep,nnn.max_sep,nnn.bin_size,nnn.nbins)
    #print(nnn.min_u,nnn.max_u,nnn.ubin_size,nnn.nubins)
    #print(nnn.min_v,nnn.max_v,nnn.vbin_size,nnn.nvbins)
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
    #print(nnn.bin_size,nnn.bin_slop,nnn.b)
    #print(nnn.ubin_size,nnn.bu)
    #print(nnn.vbin_size,nnn.bv)
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
    #print(nnn.bin_size,nnn.bin_slop,nnn.b)
    #print(nnn.ubin_size,nnn.bu)
    #print(nnn.vbin_size,nnn.bv)
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
    #print(nnn.bin_size,nnn.bin_slop,nnn.b)
    #print(nnn.ubin_size,nnn.bu)
    #print(nnn.vbin_size,nnn.bv)
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
    #print(nnn.bin_size,nnn.bin_slop,nnn.b)
    #print(nnn.ubin_size,nnn.bu)
    #print(nnn.vbin_size,nnn.bv)
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
    #print(nnn.bin_size,nnn.bin_slop,nnn.b)
    #print(nnn.ubin_size,nnn.bu)
    #print(nnn.vbin_size,nnn.bv)
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
    #print(nnn.bin_size,nnn.bin_slop,nnn.b)
    #print(nnn.ubin_size,nnn.bu)
    #print(nnn.vbin_size,nnn.bv)
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
    #print(nnn.bin_size,nnn.bin_slop,nnn.b)
    #print(nnn.ubin_size,nnn.bu)
    #print(nnn.vbin_size,nnn.bv)
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
    #print(nnn.bin_size,nnn.bin_slop,nnn.b)
    #print(nnn.ubin_size,nnn.bu)
    #print(nnn.vbin_size,nnn.bv)
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
    #print('ddd.ntri = ',ddd.ntri)

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

    #print('true_ntri => ',true_ntri)
    #print('diff = ',ddd.ntri - true_ntri)
    numpy.testing.assert_array_equal(ddd.ntri, true_ntri)

    # Repeat with binslop not precisely 0, since the code flow is different for bin_slop == 0.
    ddd = treecorr.NNNCorrelation(min_sep=min_sep, max_sep=max_sep, nbins=nbins, 
                                  min_u=min_u, max_u=max_u, nubins=nubins,
                                  min_v=min_v, max_v=max_v, nvbins=nvbins,
                                  bin_slop=1.e-16, verbose=3)
    ddd.process(cat)
    #print('ddd.ntri = ',ddd.ntri)
    #print('true_ntri => ',true_ntri)
    #print('diff = ',ddd.ntri - true_ntri)
    numpy.testing.assert_array_equal(ddd.ntri, true_ntri)

    # And again with no top-level recursion
    ddd = treecorr.NNNCorrelation(min_sep=min_sep, max_sep=max_sep, nbins=nbins, 
                                  min_u=min_u, max_u=max_u, nubins=nubins,
                                  min_v=min_v, max_v=max_v, nvbins=nvbins,
                                  bin_slop=1.e-16, verbose=3, max_top=0)
    ddd.process(cat)
    #print('ddd.ntri = ',ddd.ntri)
    #print('true_ntri => ',true_ntri)
    #print('diff = ',ddd.ntri - true_ntri)
    numpy.testing.assert_array_equal(ddd.ntri, true_ntri)

    # This should be equivalent to processing a cross correlation with each catalog being
    # the same thing.
    ddd.clear()
    ddd.process(cat,cat,cat)
    #print('ddd.ntri = ',ddd.ntri)
    #print('true_ntri => ',true_ntri)
    #print('diff = ',ddd.ntri - true_ntri)
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
    #print('ddd.ntri = ',ddd.ntri)

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

    #print('true_ntri = ',true_ntri)
    #print('diff = ',ddd.ntri - true_ntri)
    numpy.testing.assert_array_equal(ddd.ntri, true_ntri)

    # Repeat with binslop not precisely 0, since the code flow is different for bin_slop == 0.
    ddd = treecorr.NNNCorrelation(min_sep=min_sep, max_sep=max_sep, nbins=nbins, 
                                  min_u=min_u, max_u=max_u, nubins=nubins,
                                  min_v=min_v, max_v=max_v, nvbins=nvbins,
                                  bin_slop=1.e-16, verbose=3)
    ddd.process(cat1, cat2, cat3)
    #print('binslop > 0: ddd.ntri = ',ddd.ntri)
    #print('diff = ',ddd.ntri - true_ntri)
    numpy.testing.assert_array_equal(ddd.ntri, true_ntri)

    # And again with no top-level recursion
    ddd = treecorr.NNNCorrelation(min_sep=min_sep, max_sep=max_sep, nbins=nbins, 
                                  min_u=min_u, max_u=max_u, nubins=nubins,
                                  min_v=min_v, max_v=max_v, nvbins=nvbins,
                                  bin_slop=1.e-16, verbose=3, max_top=0)
    ddd.process(cat1, cat2, cat3)
    #print('max_top = 0: ddd.ntri = ',ddd.ntri)
    #print('true_ntri = ',true_ntri)
    #print('diff = ',ddd.ntri - true_ntri)
    numpy.testing.assert_array_equal(ddd.ntri, true_ntri)

def test_direct_partial():
    # Test the two ways to only use parts of a catalog:

    ngal = 200
    s = 10.
    numpy.random.seed(8675309)
    x1 = numpy.random.normal(0,s, (ngal,) )
    y1 = numpy.random.normal(0,s, (ngal,) )
    cat1a = treecorr.Catalog(x=x1, y=y1, first_row=28, last_row=144)
    x2 = numpy.random.normal(0,s, (ngal,) )
    y2 = numpy.random.normal(0,s, (ngal,) )
    cat2a = treecorr.Catalog(x=x2, y=y2, first_row=48, last_row=129)
    x3 = numpy.random.normal(0,s, (ngal,) )
    y3 = numpy.random.normal(0,s, (ngal,) )
    cat3a = treecorr.Catalog(x=x3, y=y3, first_row=82, last_row=167)

    min_sep = 1.
    max_sep = 50.
    nbins = 50
    min_u = 0.13
    max_u = 0.89
    nubins = 10
    min_v = -0.83
    max_v = 0.59
    nvbins = 20

    ddda = treecorr.NNNCorrelation(min_sep=min_sep, max_sep=max_sep, nbins=nbins, 
                                   min_u=min_u, max_u=max_u, nubins=nubins,
                                   min_v=min_v, max_v=max_v, nvbins=nvbins,
                                   bin_slop=0.)
    ddda.process(cat1a, cat2a, cat3a)
    #print('ddda.ntri = ',ddda.ntri)

    log_min_sep = numpy.log(min_sep)
    log_max_sep = numpy.log(max_sep)
    true_ntri = numpy.zeros( (nbins, nubins, nvbins) )
    bin_size = (log_max_sep - log_min_sep) / nbins
    ubin_size = (max_u-min_u) / nubins
    vbin_size = (max_v-min_v) / nvbins
    for i in range(27,144):
        for j in range(47,129):
            for k in range(81,167):
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

    #print('true_ntri = ',true_ntri)
    #print('diff = ',ddda.ntri - true_ntri)
    numpy.testing.assert_array_equal(ddda.ntri, true_ntri)

    # Now check that we get the same thing with all the points, but with w=0 for the ones 
    # we don't want.
    w1 = numpy.zeros(ngal)
    w1[27:144] = 1.
    w2 = numpy.zeros(ngal)
    w2[47:129] = 1.
    w3 = numpy.zeros(ngal)
    w3[81:167] = 1.
    cat1b = treecorr.Catalog(x=x1, y=y1, w=w1)
    cat2b = treecorr.Catalog(x=x2, y=y2, w=w2)
    cat3b = treecorr.Catalog(x=x3, y=y3, w=w3)
    dddb = treecorr.NNNCorrelation(min_sep=min_sep, max_sep=max_sep, nbins=nbins, 
                                   min_u=min_u, max_u=max_u, nubins=nubins,
                                   min_v=min_v, max_v=max_v, nvbins=nvbins,
                                   bin_slop=0.)
    dddb.process(cat1b, cat2b, cat3b)
    #print('dddb.ntri = ',dddb.ntri)
    #print('diff = ',dddb.ntri - true_ntri)
    numpy.testing.assert_array_equal(dddb.ntri, true_ntri)


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
    #print('ddd.ntri = ',ddd.ntri)

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

    #print('true_ntri => ',true_ntri)
    #print('diff = ',ddd.ntri - true_ntri)
    numpy.testing.assert_array_equal(ddd.ntri, true_ntri)

    # Repeat with binslop not precisely 0, since the code flow is different for bin_slop == 0.
    ddd = treecorr.NNNCorrelation(min_sep=min_sep, max_sep=max_sep, nbins=nbins, 
                                  min_u=min_u, max_u=max_u, nubins=nubins,
                                  min_v=min_v, max_v=max_v, nvbins=nvbins,
                                  bin_slop=1.e-16, verbose=3)
    ddd.process(cat)
    #print('ddd.ntri = ',ddd.ntri)
    #print('diff = ',ddd.ntri - true_ntri)
    numpy.testing.assert_array_equal(ddd.ntri, true_ntri)

    # And again with no top-level recursion
    ddd = treecorr.NNNCorrelation(min_sep=min_sep, max_sep=max_sep, nbins=nbins, 
                                  min_u=min_u, max_u=max_u, nubins=nubins,
                                  min_v=min_v, max_v=max_v, nvbins=nvbins,
                                  bin_slop=1.e-16, verbose=3, max_top=0)
    ddd.process(cat)
    #print('ddd.ntri = ',ddd.ntri)
    #print('true_ntri => ',true_ntri)
    #print('diff = ',ddd.ntri - true_ntri)
    numpy.testing.assert_array_equal(ddd.ntri, true_ntri)

    # And compare to the cross correlation
    ddd.clear()
    ddd.process(cat,cat,cat)
    #print('ddd.ntri = ',ddd.ntri)
    #print('true_ntri => ',true_ntri)
    #print('diff = ',ddd.ntri - true_ntri)
    numpy.testing.assert_array_equal(ddd.ntri, true_ntri)

    # Also compare to using x,y,z rather than ra,dec,r
    cat = treecorr.Catalog(x=x, y=y, z=z)
    ddd.process(cat)
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
    #print('ddd.ntri = ',ddd.ntri)

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

    #print('true_ntri = ',true_ntri)
    #print('diff = ',ddd.ntri - true_ntri)
    numpy.testing.assert_array_equal(ddd.ntri, true_ntri)

    # Repeat with binslop not precisely 0, since the code flow is different for bin_slop == 0.
    ddd = treecorr.NNNCorrelation(min_sep=min_sep, max_sep=max_sep, nbins=nbins, 
                                  min_u=min_u, max_u=max_u, nubins=nubins,
                                  min_v=min_v, max_v=max_v, nvbins=nvbins,
                                  bin_slop=1.e-16, verbose=3)
    ddd.process(cat1, cat2, cat3)
    #print('binslop > 0: ddd.ntri = ',ddd.ntri)
    #print('diff = ',ddd.ntri - true_ntri)
    numpy.testing.assert_array_equal(ddd.ntri, true_ntri)

    # And again with no top-level recursion
    ddd = treecorr.NNNCorrelation(min_sep=min_sep, max_sep=max_sep, nbins=nbins, 
                                  min_u=min_u, max_u=max_u, nubins=nubins,
                                  min_v=min_v, max_v=max_v, nvbins=nvbins,
                                  bin_slop=1.e-16, verbose=3, max_top=0)
    ddd.process(cat1, cat2, cat3)
    #print('max_top = 0: ddd.ntri = ',ddd.ntri)
    #print('true_ntri = ',true_ntri)
    #print('diff = ',ddd.ntri - true_ntri)
    numpy.testing.assert_array_equal(ddd.ntri, true_ntri)

    # Also compare to using x,y,z rather than ra,dec,r
    cat1 = treecorr.Catalog(x=x1, y=y1, z=z1)
    cat2 = treecorr.Catalog(x=x2, y=y2, z=z2)
    cat3 = treecorr.Catalog(x=x3, y=y3, z=z3)
    ddd.process(cat1, cat2, cat3)
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
    s = 10.
    if __name__ == "__main__":
        ngal = 20000
        nrand = 2 * ngal
        L = 50. * s  # Not infinity, so this introduces some error.  Our integrals were to infinity.
        req_factor = 1
    else:
        ngal = 5000
        nrand = ngal
        L = 10. * s
        req_factor = 5
    numpy.random.seed(8675309)
    x = numpy.random.normal(0,s, (ngal,) )
    y = numpy.random.normal(0,s, (ngal,) )
    min_sep = 11.
    max_sep = 13.
    nbins = 2
    min_u = 0.6
    max_u = 0.9
    nubins = 3
    min_v = 0.5
    max_v = 0.9
    nvbins = 5

    cat = treecorr.Catalog(x=x, y=y, x_units='arcmin', y_units='arcmin')
    ddd = treecorr.NNNCorrelation(min_sep=min_sep, max_sep=max_sep, nbins=nbins,
                                  min_u=min_u, max_u=max_u, min_v=min_v, max_v=max_v,
                                  nubins=nubins, nvbins=nvbins,
                                  sep_units='arcmin', verbose=3)
    ddd.process(cat)
    #print('ddd.ntri = ',ddd.ntri)

    # log(<d>) != <logd>, but it should be close:
    print('meanlogd1 - log(meand1) = ',ddd.meanlogd1 - numpy.log(ddd.meand1))
    print('meanlogd2 - log(meand2) = ',ddd.meanlogd2 - numpy.log(ddd.meand2))
    print('meanlogd3 - log(meand3) = ',ddd.meanlogd3 - numpy.log(ddd.meand3))
    print('meanlogd3 - meanlogd2 - log(meanu) = ',ddd.meanlogd3 - ddd.meanlogd2 - numpy.log(ddd.meanu))
    print('log(meand1-meand2) - meanlogd3 - log(meanv) = ',numpy.log(ddd.meand1-ddd.meand2) - ddd.meanlogd3 - numpy.log(ddd.meanv))
    numpy.testing.assert_almost_equal(ddd.meanlogd1, numpy.log(ddd.meand1), decimal=3)
    numpy.testing.assert_almost_equal(ddd.meanlogd2, numpy.log(ddd.meand2), decimal=3)
    numpy.testing.assert_almost_equal(ddd.meanlogd3, numpy.log(ddd.meand3), decimal=3)
    numpy.testing.assert_almost_equal(ddd.meanlogd3-ddd.meanlogd2, numpy.log(ddd.meanu), decimal=3)
    numpy.testing.assert_almost_equal(numpy.log(ddd.meand1-ddd.meand2)-ddd.meanlogd3, 
                                      numpy.log(ddd.meanv), decimal=3)

    rx = (numpy.random.random_sample(nrand)-0.5) * L
    ry = (numpy.random.random_sample(nrand)-0.5) * L
    rand = treecorr.Catalog(x=rx,y=ry, x_units='arcmin', y_units='arcmin')
    rrr = treecorr.NNNCorrelation(min_sep=min_sep, max_sep=max_sep, nbins=nbins,
                                  min_u=min_u, max_u=max_u, min_v=min_v, max_v=max_v,
                                  nubins=nubins, nvbins=nvbins,
                                  sep_units='arcmin', verbose=3)
    rrr.process(rand)
    #print('rrr.ntri = ',rrr.ntri)

    d1 = ddd.meand1
    d2 = ddd.meand2
    d3 = ddd.meand3
    #print('rnom = ',numpy.exp(ddd.logr))
    #print('unom = ',ddd.u)
    #print('vnom = ',ddd.v)
    #print('d1 = ',d1)
    #print('d2 = ',d2)
    #print('d3 = ',d3)
    true_zeta = (1./(12.*numpy.pi**2)) * (L/s)**4 * numpy.exp(-(d1**2+d2**2+d3**2)/(6.*s**2)) - 1.

    zeta, varzeta = ddd.calculateZeta(rrr)
    print('zeta = ',zeta)
    print('true_zeta = ',true_zeta)
    print('ratio = ',zeta / true_zeta)
    print('diff = ',zeta - true_zeta)
    print('max rel diff = ',numpy.max(numpy.abs((zeta - true_zeta)/true_zeta)))
    assert numpy.max(numpy.abs((zeta - true_zeta)/true_zeta))/req_factor < 0.1
    numpy.testing.assert_almost_equal(numpy.log(numpy.abs(zeta))/req_factor, 
                                      numpy.log(numpy.abs(true_zeta))/req_factor, decimal=1)

    # Check that we get the same result using the corr3 executable:
    if __name__ == '__main__':
        cat.write(os.path.join('data','nnn_data.dat'))
        rand.write(os.path.join('data','nnn_rand.dat'))
        import subprocess
        p = subprocess.Popen( ["corr3","nnn.params"] )
        p.communicate()
        corr3_output = numpy.genfromtxt(os.path.join('output','nnn.out'), names=True)
        print('zeta = ',zeta)
        print('from corr3 output = ',corr3_output['zeta'])
        print('ratio = ',corr3_output['zeta']/zeta.flatten())
        print('diff = ',corr3_output['zeta']-zeta.flatten())
        numpy.testing.assert_almost_equal(corr3_output['zeta']/zeta.flatten(), 1., decimal=3)

    # Check the fits write option
    out_file_name1 = os.path.join('output','nnn_out1.fits')
    ddd.write(out_file_name1)
    try:
        import fitsio
        data = fitsio.read(out_file_name1)
        numpy.testing.assert_almost_equal(data['R_nom'], numpy.exp(ddd.logr).flatten())
        numpy.testing.assert_almost_equal(data['u_nom'], ddd.u.flatten())
        numpy.testing.assert_almost_equal(data['v_nom'], ddd.v.flatten())
        numpy.testing.assert_almost_equal(data['meand1'], ddd.meand1.flatten())
        numpy.testing.assert_almost_equal(data['meanlogd1'], ddd.meanlogd1.flatten())
        numpy.testing.assert_almost_equal(data['meand2'], ddd.meand2.flatten())
        numpy.testing.assert_almost_equal(data['meanlogd2'], ddd.meanlogd2.flatten())
        numpy.testing.assert_almost_equal(data['meand3'], ddd.meand3.flatten())
        numpy.testing.assert_almost_equal(data['meanlogd3'], ddd.meanlogd3.flatten())
        numpy.testing.assert_almost_equal(data['meanu'], ddd.meanu.flatten())
        numpy.testing.assert_almost_equal(data['meanv'], ddd.meanv.flatten())
        numpy.testing.assert_almost_equal(data['ntri'], ddd.ntri.flatten())
    except ImportError:
        print('Unable to import fitsio.  Skipping fits tests.')

    out_file_name2 = os.path.join('output','nnn_out2.fits')
    ddd.write(out_file_name2, rrr)
    try:
        import fitsio
        data = fitsio.read(out_file_name2)
        numpy.testing.assert_almost_equal(data['R_nom'], numpy.exp(ddd.logr).flatten())
        numpy.testing.assert_almost_equal(data['u_nom'], ddd.u.flatten())
        numpy.testing.assert_almost_equal(data['v_nom'], ddd.v.flatten())
        numpy.testing.assert_almost_equal(data['meand1'], ddd.meand1.flatten())
        numpy.testing.assert_almost_equal(data['meanlogd1'], ddd.meanlogd1.flatten())
        numpy.testing.assert_almost_equal(data['meand2'], ddd.meand2.flatten())
        numpy.testing.assert_almost_equal(data['meanlogd2'], ddd.meanlogd2.flatten())
        numpy.testing.assert_almost_equal(data['meand3'], ddd.meand3.flatten())
        numpy.testing.assert_almost_equal(data['meanlogd3'], ddd.meanlogd3.flatten())
        numpy.testing.assert_almost_equal(data['meanu'], ddd.meanu.flatten())
        numpy.testing.assert_almost_equal(data['meanv'], ddd.meanv.flatten())
        numpy.testing.assert_almost_equal(data['zeta'], zeta.flatten())
        numpy.testing.assert_almost_equal(data['sigma_zeta'], numpy.sqrt(varzeta).flatten())
        numpy.testing.assert_almost_equal(data['DDD'], ddd.ntri.flatten())
        numpy.testing.assert_almost_equal(data['RRR'], rrr.ntri.flatten() * (ddd.tot / rrr.tot))
    except ImportError:
        print('Unable to import fitsio.  Skipping fits tests.')

    # Check the read function
    # Note: These don't need the flatten. The read function should reshape them to the right shape.
    ddd2 = treecorr.NNNCorrelation(min_sep=min_sep, max_sep=max_sep, nbins=nbins,
                                   min_u=min_u, max_u=max_u, min_v=min_v, max_v=max_v,
                                   nubins=nubins, nvbins=nvbins,
                                   sep_units='arcmin', verbose=3)
    ddd2.read(out_file_name1)
    numpy.testing.assert_almost_equal(ddd2.logr, ddd.logr)
    numpy.testing.assert_almost_equal(ddd2.u, ddd.u)
    numpy.testing.assert_almost_equal(ddd2.v, ddd.v)
    numpy.testing.assert_almost_equal(ddd2.meand1, ddd.meand1)
    numpy.testing.assert_almost_equal(ddd2.meanlogd1, ddd.meanlogd1)
    numpy.testing.assert_almost_equal(ddd2.meand2, ddd.meand2)
    numpy.testing.assert_almost_equal(ddd2.meanlogd2, ddd.meanlogd2)
    numpy.testing.assert_almost_equal(ddd2.meand3, ddd.meand3)
    numpy.testing.assert_almost_equal(ddd2.meanlogd3, ddd.meanlogd3)
    numpy.testing.assert_almost_equal(ddd2.meanu, ddd.meanu)
    numpy.testing.assert_almost_equal(ddd2.meanv, ddd.meanv)
    numpy.testing.assert_almost_equal(ddd2.ntri, ddd.ntri)

    ddd2.read(out_file_name2)
    numpy.testing.assert_almost_equal(ddd2.logr, ddd.logr)
    numpy.testing.assert_almost_equal(ddd2.u, ddd.u)
    numpy.testing.assert_almost_equal(ddd2.v, ddd.v)
    numpy.testing.assert_almost_equal(ddd2.meand1, ddd.meand1)
    numpy.testing.assert_almost_equal(ddd2.meanlogd1, ddd.meanlogd1)
    numpy.testing.assert_almost_equal(ddd2.meand2, ddd.meand2)
    numpy.testing.assert_almost_equal(ddd2.meanlogd2, ddd.meanlogd2)
    numpy.testing.assert_almost_equal(ddd2.meand3, ddd.meand3)
    numpy.testing.assert_almost_equal(ddd2.meanlogd3, ddd.meanlogd3)
    numpy.testing.assert_almost_equal(ddd2.meanu, ddd.meanu)
    numpy.testing.assert_almost_equal(ddd2.meanv, ddd.meanv)
    numpy.testing.assert_almost_equal(ddd2.ntri, ddd.ntri)

    # Test compensated zeta
    # This version computes the three-point function after subtracting off the appropriate
    # two-point functions xi(d1) + xi(d2) + xi(d3), where [cf. test_nn() in test_nn.py]
    # xi(r) = 1/4pi (L/s)^2 exp(-r^2/4s^2) - 1
    if __name__ == '__main__':
        ddr = ddd.copy()
        drd = ddd.copy()
        rdd = ddd.copy()
        drr = ddd.copy()
        rdr = ddd.copy()
        rrd = ddd.copy()

        ddr.process(cat,cat,rand)
        drd.process(cat,rand,cat)
        rdd.process(rand,cat,cat)
        drr.process(cat,rand,rand)
        rdr.process(rand,cat,rand)
        rrd.process(rand,rand,cat)

        zeta, varzeta = ddd.calculateZeta(rrr,drr,rdr,rrd,ddr,drd,rdd)
        print('compensated zeta = ',zeta)

        xi1 = (1./(4.*numpy.pi)) * (L/s)**2 * numpy.exp(-d1**2/(4.*s**2)) - 1.
        xi2 = (1./(4.*numpy.pi)) * (L/s)**2 * numpy.exp(-d2**2/(4.*s**2)) - 1.
        xi3 = (1./(4.*numpy.pi)) * (L/s)**2 * numpy.exp(-d3**2/(4.*s**2)) - 1.
        print('xi1 = ',xi1)
        print('xi2 = ',xi2)
        print('xi3 = ',xi3)
        print('true_zeta + xi1 + xi2 + xi3 = ',true_zeta)
        true_zeta -= xi1 + xi2 + xi3
        print('true_zeta => ',true_zeta)
        print('ratio = ',zeta / true_zeta)
        print('diff = ',zeta - true_zeta)
        print('max rel diff = ',numpy.max(numpy.abs((zeta - true_zeta)/true_zeta)))
        assert numpy.max(numpy.abs((zeta - true_zeta)/true_zeta))/req_factor < 0.1
        numpy.testing.assert_almost_equal(numpy.log(numpy.abs(zeta))/req_factor, 
                                          numpy.log(numpy.abs(true_zeta))/req_factor, decimal=1)

        out_file_name3 = os.path.join('output','nnn_out3.fits')
        ddd.write(out_file_name3, rrr,drr,rdr,rrd,ddr,drd,rdd)
        try:
            import fitsio
            data = fitsio.read(out_file_name3)
            numpy.testing.assert_almost_equal(data['R_nom'], numpy.exp(ddd.logr).flatten())
            numpy.testing.assert_almost_equal(data['u_nom'], ddd.u.flatten())
            numpy.testing.assert_almost_equal(data['v_nom'], ddd.v.flatten())
            numpy.testing.assert_almost_equal(data['meand1'], ddd.meand1.flatten())
            numpy.testing.assert_almost_equal(data['meanlogd1'], ddd.meanlogd1.flatten())
            numpy.testing.assert_almost_equal(data['meand2'], ddd.meand2.flatten())
            numpy.testing.assert_almost_equal(data['meanlogd2'], ddd.meanlogd2.flatten())
            numpy.testing.assert_almost_equal(data['meand3'], ddd.meand3.flatten())
            numpy.testing.assert_almost_equal(data['meanlogd3'], ddd.meanlogd3.flatten())
            numpy.testing.assert_almost_equal(data['meanu'], ddd.meanu.flatten())
            numpy.testing.assert_almost_equal(data['meanv'], ddd.meanv.flatten())
            numpy.testing.assert_almost_equal(data['zeta'], zeta.flatten())
            numpy.testing.assert_almost_equal(data['sigma_zeta'], numpy.sqrt(varzeta).flatten())
            numpy.testing.assert_almost_equal(data['DDD'], ddd.ntri.flatten())
            numpy.testing.assert_almost_equal(data['RRR'], rrr.ntri.flatten() * (ddd.tot / rrr.tot))
            numpy.testing.assert_almost_equal(data['DRR'], drr.ntri.flatten() * (ddd.tot / drr.tot))
            numpy.testing.assert_almost_equal(data['RDR'], rdr.ntri.flatten() * (ddd.tot / rdr.tot))
            numpy.testing.assert_almost_equal(data['RRD'], rrd.ntri.flatten() * (ddd.tot / rrd.tot))
            numpy.testing.assert_almost_equal(data['DDR'], ddr.ntri.flatten() * (ddd.tot / ddr.tot))
            numpy.testing.assert_almost_equal(data['DRD'], drd.ntri.flatten() * (ddd.tot / drd.tot))
            numpy.testing.assert_almost_equal(data['RDD'], rdd.ntri.flatten() * (ddd.tot / rdd.tot))
        except ImportError:
            print('Unable to import fitsio.  Skipping fits tests.')

        ddd2.read(out_file_name3)
        numpy.testing.assert_almost_equal(ddd2.logr, ddd.logr)
        numpy.testing.assert_almost_equal(ddd2.u, ddd.u)
        numpy.testing.assert_almost_equal(ddd2.v, ddd.v)
        numpy.testing.assert_almost_equal(ddd2.meand1, ddd.meand1)
        numpy.testing.assert_almost_equal(ddd2.meanlogd1, ddd.meanlogd1)
        numpy.testing.assert_almost_equal(ddd2.meand2, ddd.meand2)
        numpy.testing.assert_almost_equal(ddd2.meanlogd2, ddd.meanlogd2)
        numpy.testing.assert_almost_equal(ddd2.meand3, ddd.meand3)
        numpy.testing.assert_almost_equal(ddd2.meanlogd3, ddd.meanlogd3)
        numpy.testing.assert_almost_equal(ddd2.meanu, ddd.meanu)
        numpy.testing.assert_almost_equal(ddd2.meanv, ddd.meanv)
        numpy.testing.assert_almost_equal(ddd2.ntri, ddd.ntri)


def test_3d():
    # For this one, build a Gaussian cloud around some random point in 3D space and do the 
    # correlation function in 3D.
    #
    # The 3D Fourier transform is: n~(k) = exp(-s^2 k^2/2)
    # B(k1,k2) = <n~(k1) n~(k2) n~(-k1-k2)>
    #          = exp(-s^2 (|k1|^2 + |k2|^2 - k1.k2))
    #          = exp(-s^2 (|k1|^2 + |k2|^2 + |k3|^2)/2)
    # as before, except now k1,k2 are 3d vectors, not 2d.
    #
    # zeta(r1,r2) = (1/2pi)^4 int(d^2k1 int(d^2k2 exp(ik1.x1) exp(ik2.x2) B(k1,k2) ))
    #             = exp(-(x1^2 + y1^2 + x2^2 + y2^2 - x1x2 - y1y2)/3s^2) / 12 pi^2 s^4
    #             = exp(-(d1^2 + d2^2 + d3^2)/6s^2) / 24 sqrt(3) pi^3 s^6
    #
    # And again, this is also derivable as:
    # zeta(r1,r2) = int(dx int(dy int(dz n(x,y,z) n(x+x1,y+y1,z+z1) n(x+x2,y+y2,z+z2)))
    # which is also analytically integrable and gives the same answer.
    #
    # However, we need to correct for the uniform density background, so the real result
    # is this minus 1/L^6 divided by 1/L^6.  So:
    #
    # zeta(r1,r2) = 1/(24 sqrt(3) pi^3) (L/s)^4 exp(-(d1^2+d2^2+d3^2)/6s^2) - 1

    # Doing the full correlation function takes a long time.  Here, we just test a small range
    # of separations and a moderate range for u, v, which gives us a variety of triangle lengths.
    xcen = 823  # Mpc maybe?
    ycen = 342
    zcen = -672
    s = 10.
    if __name__ == "__main__":
        ngal = 5000
        nrand = 20 * ngal
        L = 50. * s
        req_factor = 1
    else:
        ngal = 1000
        nrand = 5 * ngal
        L = 20. * s
        req_factor = 5
    numpy.random.seed(8675309)
    x = numpy.random.normal(xcen, s, (ngal,) )
    y = numpy.random.normal(ycen, s, (ngal,) )
    z = numpy.random.normal(zcen, s, (ngal,) )

    r = numpy.sqrt(x*x+y*y+z*z)
    dec = numpy.arcsin(z/r) / treecorr.degrees
    ra = numpy.arctan2(y,x) / treecorr.degrees

    min_sep = 10.
    max_sep = 20.
    nbins = 8
    min_u = 0.9
    max_u = 1.0
    nubins = 1
    min_v = -0.05
    max_v = 0.05
    nvbins = 1

    cat = treecorr.Catalog(ra=ra, dec=dec, r=r, ra_units='deg', dec_units='deg')
    ddd = treecorr.NNNCorrelation(min_sep=min_sep, max_sep=max_sep, nbins=nbins,
                                  min_u=min_u, max_u=max_u, min_v=min_v, max_v=max_v,
                                  nubins=nubins, nvbins=nvbins, verbose=3)
    ddd.process(cat)
    print('ddd.ntri = ',ddd.ntri.flatten())

    rx = (numpy.random.random_sample(nrand)-0.5) * L + xcen
    ry = (numpy.random.random_sample(nrand)-0.5) * L + ycen
    rz = (numpy.random.random_sample(nrand)-0.5) * L + zcen
    rr = numpy.sqrt(rx*rx+ry*ry+rz*rz)
    rdec = numpy.arcsin(rz/rr) / treecorr.degrees
    rra = numpy.arctan2(ry,rx) / treecorr.degrees

    rand = treecorr.Catalog(ra=rra, dec=rdec, r=rr, ra_units='deg', dec_units='deg')
    rrr = treecorr.NNNCorrelation(min_sep=min_sep, max_sep=max_sep, nbins=nbins,
                                  min_u=min_u, max_u=max_u, min_v=min_v, max_v=max_v,
                                  nubins=nubins, nvbins=nvbins, verbose=3)
    rrr.process(rand)
    print('rrr.ntri = ',rrr.ntri.flatten())

    d1 = ddd.meand1
    d2 = ddd.meand2
    d3 = ddd.meand3
    print('rnom = ',numpy.exp(ddd.logr).flatten())
    print('unom = ',ddd.u.flatten())
    print('vnom = ',ddd.v.flatten())
    print('d1 = ',d1.flatten())
    print('d2 = ',d2.flatten())
    print('d3 = ',d3.flatten())
    true_zeta = ((1./(24.*numpy.sqrt(3)*numpy.pi**3)) * (L/s)**6 *
                 numpy.exp(-(d1**2+d2**2+d3**2)/(6.*s**2)) - 1.)

    zeta, varzeta = ddd.calculateZeta(rrr)
    print('zeta = ',zeta.flatten())
    print('true_zeta = ',true_zeta.flatten())
    print('ratio = ',(zeta / true_zeta).flatten())
    print('diff = ',(zeta - true_zeta).flatten())
    print('max rel diff = ',numpy.max(numpy.abs((zeta - true_zeta)/true_zeta)))
    assert numpy.max(numpy.abs((zeta - true_zeta)/true_zeta))/req_factor < 0.1
    numpy.testing.assert_almost_equal(numpy.log(numpy.abs(zeta))/req_factor, 
                                      numpy.log(numpy.abs(true_zeta))/req_factor, decimal=1)

    # Check that we get the same result using the corr3 executable:
    if __name__ == '__main__':
        cat.write(os.path.join('data','nnn_3d_data.dat'))
        rand.write(os.path.join('data','nnn_3d_rand.dat'))
        import subprocess
        p = subprocess.Popen( ["corr3","nnn_3d.params"] )
        p.communicate()
        corr3_output = numpy.genfromtxt(os.path.join('output','nnn_3d.out'), names=True)
        print('zeta = ',zeta.flatten())
        print('from corr3 output = ',corr3_output['zeta'])
        print('ratio = ',corr3_output['zeta']/zeta.flatten())
        print('diff = ',corr3_output['zeta']-zeta.flatten())
        numpy.testing.assert_almost_equal(corr3_output['zeta']/zeta.flatten(), 1., decimal=3)
    
        # Check that we get the same thing when using x,y,z rather than ra,dec,r
        cat = treecorr.Catalog(x=x, y=y, z=z)
        rand = treecorr.Catalog(x=rx, y=ry, z=rz)
        ddd.process(cat)
        rrr.process(rand)
        zeta, varzeta = ddd.calculateZeta(rrr)
        assert numpy.max(numpy.abs((zeta - true_zeta)/true_zeta))/req_factor < 0.1
        numpy.testing.assert_almost_equal(numpy.log(numpy.abs(zeta))/req_factor, 
                                          numpy.log(numpy.abs(true_zeta))/req_factor, decimal=1)


def test_list():
    # Test that we can use a list of files for either data or rand or both.
    ncats = 3
    data_cats = []
    rand_cats = []

    ngal = 1000
    s = 10.
    L = 50. * s
    numpy.random.seed(8675309)

    min_sep = 30.
    max_sep = 50.
    nbins = 5
    min_u = 0
    max_u = 0.3
    nubins = 3
    min_v = 0.5
    max_v = 0.9
    nvbins = 5

    x = numpy.random.normal(0,s, (ngal,ncats) )
    y = numpy.random.normal(0,s, (ngal,ncats) )
    data_cats = [ treecorr.Catalog(x=x[:,k], y=y[:,k]) for k in range(ncats) ]
    nrand = 2 * ngal
    rx = (numpy.random.random_sample((nrand,ncats))-0.5) * L
    ry = (numpy.random.random_sample((nrand,ncats))-0.5) * L
    rand_cats = [ treecorr.Catalog(x=rx[:,k], y=ry[:,k]) for k in range(ncats) ]

    ddd = treecorr.NNNCorrelation(min_sep=min_sep, max_sep=max_sep, nbins=nbins,
                                  min_u=min_u, max_u=max_u, min_v=min_v, max_v=max_v,
                                  nubins=nubins, nvbins=nvbins, verbose=2)
    ddd.process(data_cats)
    print('From multiple catalogs: ddd.ntri = ',ddd.ntri)

    # Now do the same thing with one big catalog
    dddx = treecorr.NNNCorrelation(min_sep=min_sep, max_sep=max_sep, nbins=nbins,
                                   min_u=min_u, max_u=max_u, min_v=min_v, max_v=max_v,
                                   nubins=nubins, nvbins=nvbins, verbose=2)
    data_catx = treecorr.Catalog(x=x.reshape( (ngal*ncats,) ), y=y.reshape( (ngal*ncats,) ))
    dddx.process(data_catx)
    print('From single catalog: dddx.ntri = ',dddx.ntri)
    # Only test to 1 digit, since there are now differences between the auto and cross related
    # to how they characterize triangles especially when d1 ~= d2 or d2 ~= d3.
    numpy.testing.assert_almost_equal(numpy.log(ddd.ntri), numpy.log(dddx.ntri), decimal=1)

    # If we're running from nosetests, stop here.  That's a sufficient regression test.
    # Otherwise continue to make sure that the full calculation works as expected including
    # automatically running the multiple files from the corr3 executable with either just
    # data, just rand or both as multiples.
    if __name__ == "__main__":
        rrr = treecorr.NNNCorrelation(min_sep=min_sep, max_sep=max_sep, nbins=nbins,
                                    min_u=min_u, max_u=max_u, min_v=min_v, max_v=max_v,
                                  nubins=nubins, nvbins=nvbins, verbose=2)
        rrr.process(rand_cats)
        print('rrr.ntri = ',rrr.ntri)

        rrrx = treecorr.NNNCorrelation(min_sep=min_sep, max_sep=max_sep, nbins=nbins,
                                    min_u=min_u, max_u=max_u, min_v=min_v, max_v=max_v,
                                   nubins=nubins, nvbins=nvbins, verbose=2)
        rand_catx = treecorr.Catalog(x=rx.reshape( (nrand*ncats,) ), y=ry.reshape( (nrand*ncats,) ))
        rrrx.process(rand_catx)
        print('rrrx.ntri = ',rrrx.ntri)
        numpy.testing.assert_almost_equal(numpy.log(ddd.ntri), numpy.log(dddx.ntri), decimal=1)

        zeta, varzeta = ddd.calculateZeta(rrr)
        zetax, varzetax = dddx.calculateZeta(rrrx)
        print('zeta = ',zeta)
        print('zetax = ',zetax)
        #print('ratio = ',zeta/zetax)
        #print('diff = ',zeta-zetax)
        numpy.testing.assert_almost_equal(zetax/zeta, 1., decimal=1)

        # Check that we get the same result using the corr3 executable:
        file_list = []
        rand_file_list = []
        for k in range(ncats):
            file_name = os.path.join('data','nnn_list_data%d.dat'%k)
            data_cats[k].write(file_name)
            file_list.append(file_name)

            rand_file_name = os.path.join('data','nnn_list_rand%d.dat'%k)
            rand_cats[k].write(rand_file_name)
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
        data_catx.write(file_namex)

        rand_file_namex = os.path.join('data','nnn_list_randx.dat')
        rand_catx.write(rand_file_namex)

        import subprocess
        p = subprocess.Popen( ["corr3","nnn_list1.params"] )
        p.communicate()
        corr3_output = numpy.genfromtxt(os.path.join('output','nnn_list1.out'), names=True)
        print('zeta = ',zeta)
        print('from corr3 output = ',corr3_output['zeta'])
        print('ratio = ',corr3_output['zeta']/zeta.flatten())
        print('diff = ',corr3_output['zeta']-zeta.flatten())
        numpy.testing.assert_almost_equal(corr3_output['zeta']/zeta.flatten(), 1., decimal=3)

        p = subprocess.Popen( ["corr3","nnn_list2.params"] )
        p.communicate()
        corr3_output = numpy.genfromtxt(os.path.join('output','nnn_list2.out'), names=True)
        print('zeta = ',zeta)
        print('from corr3 output = ',corr3_output['zeta'])
        print('ratio = ',corr3_output['zeta']/zeta.flatten())
        print('diff = ',corr3_output['zeta']-zeta.flatten())
        numpy.testing.assert_almost_equal(corr3_output['zeta']/zeta.flatten(), 1., decimal=1)

        p = subprocess.Popen( ["corr3","nnn_list3.params"] )
        p.communicate()
        corr3_output = numpy.genfromtxt(os.path.join('output','nnn_list3.out'), names=True)
        print('zeta = ',zeta)
        print('from corr3 output = ',corr3_output['zeta'])
        print('ratio = ',corr3_output['zeta']/zeta.flatten())
        print('diff = ',corr3_output['zeta']-zeta.flatten())
        numpy.testing.assert_almost_equal(corr3_output['zeta']/zeta.flatten(), 1., decimal=1)


if __name__ == '__main__':
    test_binnedcorr3()
    test_direct_count_auto()
    test_direct_count_cross()
    test_direct_3d_auto()
    test_direct_3d_cross()
    test_direct_partial()
    test_nnn()
    test_3d()
    test_list()
