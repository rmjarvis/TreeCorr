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
import treecorr
import os
import coord
import time
import shutil

from test_helper import get_from_wiki, get_script_name, do_pickle, CaptureLog, assert_raises

def test_log_binning():
    import math
    # Test some basic properties of the base class with respect to Log binning

    # Check the different ways to set up the binning:
    # Omit bin_size
    nn = treecorr.NNCorrelation(min_sep=5, max_sep=20, nbins=20)
    print(nn.min_sep,nn.max_sep,nn.bin_size,nn.nbins)
    assert nn.min_sep == 5.
    assert nn.max_sep == 20.
    assert nn.nbins == 20
    np.testing.assert_almost_equal(nn.bin_size * nn.nbins, math.log(nn.max_sep/nn.min_sep))
    print('logr = ',nn.logr)
    np.testing.assert_almost_equal(nn.logr[0], math.log(nn.min_sep) + 0.5*nn.bin_size)
    np.testing.assert_almost_equal(nn.logr[-1], math.log(nn.max_sep) - 0.5*nn.bin_size)
    assert len(nn.logr) == nn.nbins

    # Omit min_sep
    nn = treecorr.NNCorrelation(max_sep=20, nbins=20, bin_size=0.1)
    print(nn.min_sep,nn.max_sep,nn.bin_size,nn.nbins)
    assert nn.bin_size == 0.1
    assert nn.max_sep == 20.
    assert nn.nbins == 20
    np.testing.assert_almost_equal(nn.bin_size * nn.nbins, math.log(nn.max_sep/nn.min_sep))
    np.testing.assert_almost_equal(nn.logr[0], math.log(nn.min_sep) + 0.5*nn.bin_size)
    np.testing.assert_almost_equal(nn.logr[-1], math.log(nn.max_sep) - 0.5*nn.bin_size)
    assert len(nn.logr) == nn.nbins

    # Omit max_sep
    nn = treecorr.NNCorrelation(min_sep=5, nbins=20, bin_size=0.1)
    print(nn.min_sep,nn.max_sep,nn.bin_size,nn.nbins)
    assert nn.bin_size == 0.1
    assert nn.min_sep == 5.
    assert nn.nbins == 20
    np.testing.assert_almost_equal(nn.bin_size * nn.nbins, math.log(nn.max_sep/nn.min_sep))
    np.testing.assert_almost_equal(nn.logr[0], math.log(nn.min_sep) + 0.5*nn.bin_size)
    np.testing.assert_almost_equal(nn.logr[-1], math.log(nn.max_sep) - 0.5*nn.bin_size)
    assert len(nn.logr) == nn.nbins

    # Omit nbins
    nn = treecorr.NNCorrelation(min_sep=5, max_sep=20, bin_size=0.1)
    print(nn.min_sep,nn.max_sep,nn.bin_size,nn.nbins)
    assert nn.bin_size == 0.1
    assert nn.min_sep == 5.
    assert nn.max_sep >= 20.
    assert nn.max_sep <= 20.*math.exp(nn.bin_size)
    np.testing.assert_almost_equal(nn.bin_size * nn.nbins, math.log(nn.max_sep/nn.min_sep))
    np.testing.assert_almost_equal(nn.logr[0], math.log(nn.min_sep) + 0.5*nn.bin_size)
    np.testing.assert_almost_equal(nn.logr[-1], math.log(nn.max_sep) - 0.5*nn.bin_size)
    assert len(nn.logr) == nn.nbins

    # Check the use of sep_units
    # radians
    nn = treecorr.NNCorrelation(min_sep=5, max_sep=20, nbins=20, sep_units='radians')
    print(nn.min_sep,nn.max_sep,nn.bin_size,nn.nbins)
    np.testing.assert_almost_equal(nn.min_sep, 5.)
    np.testing.assert_almost_equal(nn.max_sep, 20.)
    np.testing.assert_almost_equal(nn._min_sep, 5.)
    np.testing.assert_almost_equal(nn._max_sep, 20.)
    assert nn.min_sep == 5.
    assert nn.max_sep == 20.
    assert nn.nbins == 20
    np.testing.assert_almost_equal(nn.bin_size * nn.nbins, math.log(nn.max_sep/nn.min_sep))
    np.testing.assert_almost_equal(nn.logr[0], math.log(5) + 0.5*nn.bin_size)
    np.testing.assert_almost_equal(nn.logr[-1], math.log(20) - 0.5*nn.bin_size)
    assert len(nn.logr) == nn.nbins

    # arcsec
    nn = treecorr.NNCorrelation(min_sep=5, max_sep=20, nbins=20, sep_units='arcsec')
    print(nn.min_sep,nn.max_sep,nn.bin_size,nn.nbins)
    np.testing.assert_almost_equal(nn.min_sep, 5.)
    np.testing.assert_almost_equal(nn.max_sep, 20.)
    np.testing.assert_almost_equal(nn._min_sep, 5. * math.pi/180/3600)
    np.testing.assert_almost_equal(nn._max_sep, 20. * math.pi/180/3600)
    assert nn.nbins == 20
    np.testing.assert_almost_equal(nn.bin_size * nn.nbins, math.log(nn.max_sep/nn.min_sep))
    # Note that logr is in the separation units, not radians.
    np.testing.assert_almost_equal(nn.logr[0], math.log(5) + 0.5*nn.bin_size)
    np.testing.assert_almost_equal(nn.logr[-1], math.log(20) - 0.5*nn.bin_size)
    assert len(nn.logr) == nn.nbins

    # arcmin
    nn = treecorr.NNCorrelation(min_sep=5, max_sep=20, nbins=20, sep_units='arcmin')
    print(nn.min_sep,nn.max_sep,nn.bin_size,nn.nbins)
    np.testing.assert_almost_equal(nn.min_sep, 5.)
    np.testing.assert_almost_equal(nn.max_sep, 20.)
    np.testing.assert_almost_equal(nn._min_sep, 5. * math.pi/180/60)
    np.testing.assert_almost_equal(nn._max_sep, 20. * math.pi/180/60)
    assert nn.nbins == 20
    np.testing.assert_almost_equal(nn.bin_size * nn.nbins, math.log(nn.max_sep/nn.min_sep))
    np.testing.assert_almost_equal(nn.logr[0], math.log(5) + 0.5*nn.bin_size)
    np.testing.assert_almost_equal(nn.logr[-1], math.log(20) - 0.5*nn.bin_size)
    assert len(nn.logr) == nn.nbins

    # degrees
    nn = treecorr.NNCorrelation(min_sep=5, max_sep=20, nbins=20, sep_units='degrees')
    print(nn.min_sep,nn.max_sep,nn.bin_size,nn.nbins)
    np.testing.assert_almost_equal(nn.min_sep, 5.)
    np.testing.assert_almost_equal(nn.max_sep, 20.)
    np.testing.assert_almost_equal(nn._min_sep, 5. * math.pi/180)
    np.testing.assert_almost_equal(nn._max_sep, 20. * math.pi/180)
    assert nn.nbins == 20
    np.testing.assert_almost_equal(nn.bin_size * nn.nbins, math.log(nn.max_sep/nn.min_sep))
    np.testing.assert_almost_equal(nn.logr[0], math.log(5) + 0.5*nn.bin_size)
    np.testing.assert_almost_equal(nn.logr[-1], math.log(20) - 0.5*nn.bin_size)
    assert len(nn.logr) == nn.nbins

    # hours
    nn = treecorr.NNCorrelation(min_sep=5, max_sep=20, nbins=20, sep_units='hours')
    print(nn.min_sep,nn.max_sep,nn.bin_size,nn.nbins)
    np.testing.assert_almost_equal(nn.min_sep, 5.)
    np.testing.assert_almost_equal(nn.max_sep, 20.)
    np.testing.assert_almost_equal(nn._min_sep, 5. * math.pi/12)
    np.testing.assert_almost_equal(nn._max_sep, 20. * math.pi/12)
    assert nn.nbins == 20
    np.testing.assert_almost_equal(nn.bin_size * nn.nbins, math.log(nn.max_sep/nn.min_sep))
    np.testing.assert_almost_equal(nn.logr[0], math.log(5) + 0.5*nn.bin_size)
    np.testing.assert_almost_equal(nn.logr[-1], math.log(20) - 0.5*nn.bin_size)
    assert len(nn.logr) == nn.nbins

    # Check bin_slop
    nn = treecorr.NNCorrelation(min_sep=5, max_sep=20, bin_size=0.1)
    print(nn.bin_size,nn.bin_slop,nn.b)
    assert nn.bin_size == 0.1
    assert nn.bin_slop == 1.0
    np.testing.assert_almost_equal(nn.b, 0.1)

    nn = treecorr.NNCorrelation(min_sep=5, max_sep=20, bin_size=0.1, bin_slop=1.0)
    print(nn.bin_size,nn.bin_slop,nn.b)
    assert nn.bin_size == 0.1
    assert nn.bin_slop == 1.0
    np.testing.assert_almost_equal(nn.b, 0.1)

    nn = treecorr.NNCorrelation(min_sep=5, max_sep=20, bin_size=0.1, bin_slop=0.2)
    print(nn.bin_size,nn.bin_slop,nn.b)
    assert nn.bin_size == 0.1
    assert nn.bin_slop == 0.2
    np.testing.assert_almost_equal(nn.b, 0.02)

    nn = treecorr.NNCorrelation(min_sep=5, max_sep=20, bin_size=0.1, bin_slop=0.0)
    print(nn.bin_size,nn.bin_slop,nn.b)
    assert nn.bin_size == 0.1
    assert nn.bin_slop == 0.0
    assert nn.b == 0.0

    nn = treecorr.NNCorrelation(min_sep=5, max_sep=20, bin_size=0.1, bin_slop=2.0, verbose=0)
    print(nn.bin_size,nn.bin_slop,nn.b)
    assert nn.bin_size == 0.1
    assert nn.bin_slop == 2.0
    np.testing.assert_almost_equal(nn.b, 0.2)

    nn = treecorr.NNCorrelation(min_sep=5, max_sep=20, bin_size=0.4, bin_slop=1.0, verbose=0)
    print(nn.bin_size,nn.bin_slop,nn.b)
    assert nn.bin_size == 0.4
    assert nn.bin_slop == 1.0
    np.testing.assert_almost_equal(nn.b, 0.4)

    nn = treecorr.NNCorrelation(min_sep=5, max_sep=20, bin_size=0.4)
    print(nn.bin_size,nn.bin_slop,nn.b)
    assert nn.bin_size == 0.4
    np.testing.assert_almost_equal(nn.b, 0.1)
    np.testing.assert_almost_equal(nn.bin_slop, 0.25)

    nn = treecorr.NNCorrelation(min_sep=5, max_sep=20, bin_size=0.4, bin_slop=0.1)
    print(nn.bin_size,nn.bin_slop,nn.b)
    assert nn.bin_size == 0.4
    assert nn.bin_slop == 0.1
    np.testing.assert_almost_equal(nn.b, 0.04)

    nn = treecorr.NNCorrelation(min_sep=5, max_sep=20, bin_size=0.05, bin_slop=1.0)
    print(nn.bin_size,nn.bin_slop,nn.b)
    assert nn.bin_size == 0.05
    assert nn.bin_slop == 1.0
    np.testing.assert_almost_equal(nn.b, 0.05)

    nn = treecorr.NNCorrelation(min_sep=5, max_sep=20, bin_size=0.05)
    print(nn.bin_size,nn.bin_slop,nn.b)
    assert nn.bin_size == 0.05
    assert nn.bin_slop == 1.0
    np.testing.assert_almost_equal(nn.b, 0.05)

    nn = treecorr.NNCorrelation(min_sep=5, max_sep=20, bin_size=0.05, bin_slop=3, verbose=0)
    print(nn.bin_size,nn.bin_slop,nn.b)
    assert nn.bin_size == 0.05
    assert nn.bin_slop == 3.0
    np.testing.assert_almost_equal(nn.b, 0.15)


def test_linear_binning():
    import math
    # Test some basic properties of the base class with respect to Linear binning

    # Check the different ways to set up the binning:
    # Omit bin_size
    nn = treecorr.NNCorrelation(min_sep=5, max_sep=20, nbins=20, bin_type='Linear')
    print(nn.min_sep,nn.max_sep,nn.bin_size,nn.nbins)
    assert nn.min_sep == 5.
    assert nn.max_sep == 20.
    assert nn.nbins == 20
    np.testing.assert_almost_equal(nn.bin_size * nn.nbins, nn.max_sep-nn.min_sep)
    np.testing.assert_almost_equal(nn.rnom[0], nn.min_sep + 0.5*nn.bin_size)
    np.testing.assert_almost_equal(nn.rnom[-1], nn.max_sep - 0.5*nn.bin_size)
    assert len(nn.rnom) == nn.nbins
    assert len(nn.logr) == nn.nbins

    # Omit min_sep
    nn = treecorr.NNCorrelation(max_sep=20, nbins=20, bin_size=0.1, bin_type='Linear')
    print(nn.min_sep,nn.max_sep,nn.bin_size,nn.nbins)
    assert nn.bin_size == 0.1
    assert nn.max_sep == 20.
    assert nn.nbins == 20
    np.testing.assert_almost_equal(nn.bin_size * nn.nbins, nn.max_sep-nn.min_sep)
    np.testing.assert_almost_equal(nn.rnom[0], nn.min_sep + 0.5*nn.bin_size)
    np.testing.assert_almost_equal(nn.rnom[-1],nn.max_sep - 0.5*nn.bin_size)
    assert len(nn.rnom) == nn.nbins
    assert len(nn.logr) == nn.nbins

    # Omit max_sep
    nn = treecorr.NNCorrelation(min_sep=5, nbins=20, bin_size=0.1, bin_type='Linear')
    print(nn.min_sep,nn.max_sep,nn.bin_size,nn.nbins)
    assert nn.bin_size == 0.1
    assert nn.min_sep == 5.
    assert nn.nbins == 20
    np.testing.assert_almost_equal(nn.bin_size * nn.nbins, nn.max_sep-nn.min_sep)
    np.testing.assert_almost_equal(nn.rnom[0], nn.min_sep + 0.5*nn.bin_size)
    np.testing.assert_almost_equal(nn.rnom[-1], nn.max_sep - 0.5*nn.bin_size)
    assert len(nn.rnom) == nn.nbins
    assert len(nn.logr) == nn.nbins

    # Omit nbins
    nn = treecorr.NNCorrelation(min_sep=5, max_sep=20, bin_size=0.1, bin_type='Linear')
    print(nn.min_sep,nn.max_sep,nn.bin_size,nn.nbins)
    assert nn.bin_size == 0.1
    assert nn.min_sep == 5.
    assert nn.max_sep >= 20.
    assert nn.max_sep <= 20.*math.exp(nn.bin_size)
    np.testing.assert_almost_equal(nn.bin_size * nn.nbins, nn.max_sep-nn.min_sep)
    np.testing.assert_almost_equal(nn.rnom[0], nn.min_sep + 0.5*nn.bin_size)
    np.testing.assert_almost_equal(nn.rnom[-1], nn.max_sep - 0.5*nn.bin_size)
    assert len(nn.rnom) == nn.nbins
    assert len(nn.logr) == nn.nbins

    # Check the use of sep_units
    # radians
    nn = treecorr.NNCorrelation(min_sep=5, max_sep=20, nbins=20, sep_units='radians', bin_type='Linear')
    print(nn.min_sep,nn.max_sep,nn.bin_size,nn.nbins)
    np.testing.assert_almost_equal(nn.min_sep, 5.)
    np.testing.assert_almost_equal(nn.max_sep, 20.)
    np.testing.assert_almost_equal(nn._min_sep, 5.)
    np.testing.assert_almost_equal(nn._max_sep, 20.)
    assert nn.min_sep == 5.
    assert nn.max_sep == 20.
    assert nn.nbins == 20
    np.testing.assert_almost_equal(nn.bin_size * nn.nbins, nn.max_sep-nn.min_sep)
    np.testing.assert_almost_equal(nn.rnom[0], 5 + 0.5*nn.bin_size)
    np.testing.assert_almost_equal(nn.rnom[-1], 20 - 0.5*nn.bin_size)
    assert len(nn.rnom) == nn.nbins
    assert len(nn.logr) == nn.nbins

    # arcsec
    nn = treecorr.NNCorrelation(min_sep=5, max_sep=20, nbins=20, sep_units='arcsec', bin_type='Linear')
    print(nn.min_sep,nn.max_sep,nn.bin_size,nn.nbins)
    np.testing.assert_almost_equal(nn.min_sep, 5.)
    np.testing.assert_almost_equal(nn.max_sep, 20.)
    np.testing.assert_almost_equal(nn._min_sep, 5. * math.pi/180/3600)
    np.testing.assert_almost_equal(nn._max_sep, 20. * math.pi/180/3600)
    assert nn.nbins == 20
    np.testing.assert_almost_equal(nn.bin_size * nn.nbins, nn.max_sep-nn.min_sep)
    # Note that rnom is in the separation units, not radians.
    np.testing.assert_almost_equal(nn.rnom[0], 5 + 0.5*nn.bin_size)
    np.testing.assert_almost_equal(nn.rnom[-1], 20 - 0.5*nn.bin_size)
    assert len(nn.rnom) == nn.nbins
    assert len(nn.logr) == nn.nbins

    # arcmin
    nn = treecorr.NNCorrelation(min_sep=5, max_sep=20, nbins=20, sep_units='arcmin', bin_type='Linear')
    print(nn.min_sep,nn.max_sep,nn.bin_size,nn.nbins)
    np.testing.assert_almost_equal(nn.min_sep, 5.)
    np.testing.assert_almost_equal(nn.max_sep, 20.)
    np.testing.assert_almost_equal(nn._min_sep, 5. * math.pi/180/60)
    np.testing.assert_almost_equal(nn._max_sep, 20. * math.pi/180/60)
    assert nn.nbins == 20
    np.testing.assert_almost_equal(nn.bin_size * nn.nbins, nn.max_sep-nn.min_sep)
    np.testing.assert_almost_equal(nn.rnom[0], 5 + 0.5*nn.bin_size)
    np.testing.assert_almost_equal(nn.rnom[-1], 20 - 0.5*nn.bin_size)
    assert len(nn.rnom) == nn.nbins
    assert len(nn.logr) == nn.nbins

    # degrees
    nn = treecorr.NNCorrelation(min_sep=5, max_sep=20, nbins=20, sep_units='degrees', bin_type='Linear')
    print(nn.min_sep,nn.max_sep,nn.bin_size,nn.nbins)
    np.testing.assert_almost_equal(nn.min_sep, 5.)
    np.testing.assert_almost_equal(nn.max_sep, 20.)
    np.testing.assert_almost_equal(nn._min_sep, 5. * math.pi/180)
    np.testing.assert_almost_equal(nn._max_sep, 20. * math.pi/180)
    assert nn.nbins == 20
    np.testing.assert_almost_equal(nn.bin_size * nn.nbins, nn.max_sep-nn.min_sep)
    np.testing.assert_almost_equal(nn.rnom[0], 5 + 0.5*nn.bin_size)
    np.testing.assert_almost_equal(nn.rnom[-1], 20 - 0.5*nn.bin_size)
    assert len(nn.rnom) == nn.nbins
    assert len(nn.logr) == nn.nbins

    # hours
    nn = treecorr.NNCorrelation(min_sep=5, max_sep=20, nbins=20, sep_units='hours', bin_type='Linear')
    print(nn.min_sep,nn.max_sep,nn.bin_size,nn.nbins)
    np.testing.assert_almost_equal(nn.min_sep, 5.)
    np.testing.assert_almost_equal(nn.max_sep, 20.)
    np.testing.assert_almost_equal(nn._min_sep, 5. * math.pi/12)
    np.testing.assert_almost_equal(nn._max_sep, 20. * math.pi/12)
    assert nn.nbins == 20
    np.testing.assert_almost_equal(nn.bin_size * nn.nbins, nn.max_sep-nn.min_sep)
    np.testing.assert_almost_equal(nn.rnom[0], 5 + 0.5*nn.bin_size)
    np.testing.assert_almost_equal(nn.rnom[-1], 20 - 0.5*nn.bin_size)
    assert len(nn.rnom) == nn.nbins
    assert len(nn.logr) == nn.nbins

    # Check bin_slop
    nn = treecorr.NNCorrelation(min_sep=0, max_sep=20, bin_size=1, bin_type='Linear')
    print(nn.bin_size,nn.bin_slop,nn.b)
    assert nn.bin_size == 1
    np.testing.assert_almost_equal(nn.bin_slop, 0.05)
    np.testing.assert_almost_equal(nn.b, 0.0025)

    with CaptureLog() as cl:
        nn = treecorr.NNCorrelation(min_sep=0, max_sep=20, bin_size=1, bin_slop=1.0,
                                    bin_type='Linear', logger=cl.logger)
    print(cl.output)
    assert "It is recommended to use bin_slop <= 0.05" in cl.output
    print(nn.bin_size,nn.bin_slop,nn.b)
    assert nn.bin_size == 1
    assert nn.bin_slop == 1.0
    np.testing.assert_almost_equal(nn.b, 0.05)

    with CaptureLog() as cl:
        nn = treecorr.NNCorrelation(min_sep=0, max_sep=20, bin_size=1, bin_slop=0.2,
                                    bin_type='Linear', logger=cl.logger)
    assert "It is recommended to use bin_slop <= 0.05" in cl.output
    print(nn.bin_size,nn.bin_slop,nn.b)
    assert nn.bin_size == 1
    assert nn.bin_slop == 0.2
    np.testing.assert_almost_equal(nn.b, 0.01)

    nn = treecorr.NNCorrelation(min_sep=0, max_sep=20, bin_size=1, bin_slop=0.0,
                                bin_type='Linear')
    print(nn.bin_size,nn.bin_slop,nn.b)
    assert nn.bin_size == 1
    assert nn.bin_slop == 0.0
    assert nn.b == 0.0

    nn = treecorr.NNCorrelation(min_sep=5, max_sep=20, bin_size=0.1, bin_slop=2.0, verbose=0,
                                bin_type='Linear')
    print(nn.bin_size,nn.bin_slop,nn.b)
    assert nn.bin_size == 0.1
    assert nn.bin_slop == 2.0
    np.testing.assert_almost_equal(nn.b, 0.01)

    nn = treecorr.NNCorrelation(min_sep=0, max_sep=20, bin_size=0.4, bin_type='Linear')
    print(nn.bin_size,nn.bin_slop,nn.b)
    assert nn.bin_size == 0.4
    np.testing.assert_almost_equal(nn.b, 0.001)
    np.testing.assert_almost_equal(nn.bin_slop, 0.05)

    nn = treecorr.NNCorrelation(min_sep=5, max_sep=20, bin_size=0.05, bin_type='Linear')
    print(nn.bin_size,nn.bin_slop,nn.b)
    assert nn.bin_size == 0.05
    np.testing.assert_almost_equal(nn.bin_slop, 1)
    np.testing.assert_almost_equal(nn.b, 0.0025)


def test_direct_count():
    # If the catalogs are small enough, we can do a direct count of the number of pairs
    # to see if comes out right.  This should exactly match the treecorr code if bin_slop=0.

    ngal = 100
    s = 10.
    rng = np.random.RandomState(8675309)
    x1 = rng.normal(0,s, (ngal,) )
    y1 = rng.normal(0,s, (ngal,) )
    cat1 = treecorr.Catalog(x=x1, y=y1)
    x2 = rng.normal(0,s, (ngal,) )
    y2 = rng.normal(0,s, (ngal,) )
    cat2 = treecorr.Catalog(x=x2, y=y2)

    min_sep = 1.
    max_sep = 50.
    nbins = 50
    dd = treecorr.NNCorrelation(min_sep=min_sep, max_sep=max_sep, nbins=nbins, brute=True)
    dd.process(cat1, cat2)
    print('dd.npairs = ',dd.npairs)

    log_min_sep = np.log(min_sep)
    log_max_sep = np.log(max_sep)
    true_npairs = np.zeros(nbins)
    bin_size = (log_max_sep - log_min_sep) / nbins
    for i in range(ngal):
        for j in range(ngal):
            rsq = (x1[i]-x2[j])**2 + (y1[i]-y2[j])**2
            logr = 0.5 * np.log(rsq)
            k = int(np.floor( (logr-log_min_sep) / bin_size ))
            if k < 0: continue
            if k >= nbins: continue
            true_npairs[k] += 1

    print('true_npairs = ',true_npairs)
    print('diff = ',dd.npairs - true_npairs)
    np.testing.assert_array_equal(dd.npairs, true_npairs)

    # Check that running via the corr2 script works correctly.
    file_name1 = os.path.join('data','nn_direct_data1.dat')
    with open(file_name1, 'w') as fid:
        for i in range(ngal):
            fid.write(('%.20f %.20f\n')%(x1[i],y1[i]))
    file_name2 = os.path.join('data','nn_direct_data2.dat')
    with open(file_name2, 'w') as fid:
        for i in range(ngal):
            fid.write(('%.20f %.20f\n')%(x2[i],y2[i]))
    L = 10*s
    nrand = ngal
    rx1 = (rng.random_sample(nrand)-0.5) * L
    ry1 = (rng.random_sample(nrand)-0.5) * L
    rx2 = (rng.random_sample(nrand)-0.5) * L
    ry2 = (rng.random_sample(nrand)-0.5) * L
    rcat1 = treecorr.Catalog(x=rx1, y=ry1)
    rcat2 = treecorr.Catalog(x=rx2, y=ry2)
    rand_file_name1 = os.path.join('data','nn_direct_rand1.dat')
    with open(rand_file_name1, 'w') as fid:
        for i in range(nrand):
            fid.write(('%.20f %.20f\n')%(rx1[i],ry1[i]))
    rand_file_name2 = os.path.join('data','nn_direct_rand2.dat')
    with open(rand_file_name2, 'w') as fid:
        for i in range(nrand):
            fid.write(('%.20f %.20f\n')%(rx2[i],ry2[i]))
    rr = treecorr.NNCorrelation(min_sep=min_sep, max_sep=max_sep, nbins=nbins, brute=True,
                                verbose=0)
    rr.process(rcat1,rcat2)
    xi, varxi = dd.calculateXi(rr)

    # First do this via the corr2 function.
    config = treecorr.config.read_config('configs/nn_direct.yaml')
    logger = treecorr.config.setup_logger(0)
    treecorr.corr2(config, logger)
    corr2_output = np.genfromtxt(os.path.join('output','nn_direct.out'), names=True,
                                    skip_header=1)
    print('corr2_output = ',corr2_output)
    print('corr2_output.dtype = ',corr2_output.dtype)
    print('rnom = ',dd.rnom)
    print('       ',corr2_output['R_nom'])
    np.testing.assert_allclose(corr2_output['R_nom'], dd.rnom, rtol=1.e-3)
    print('DD = ',dd.npairs)
    print('      ',corr2_output['DD'])
    np.testing.assert_allclose(corr2_output['DD'], dd.npairs, rtol=1.e-3)
    np.testing.assert_allclose(corr2_output['npairs'], dd.npairs, rtol=1.e-3)
    print('RR = ',rr.npairs)
    print('      ',corr2_output['RR'])
    np.testing.assert_allclose(corr2_output['RR'], rr.npairs, rtol=1.e-3)
    print('xi = ',xi)
    print('from corr2 output = ',corr2_output['xi'])
    print('diff = ',corr2_output['xi']-xi)
    diff_index = np.where(np.abs(corr2_output['xi']-xi) > 1.e-5)[0]
    print('different at ',diff_index)
    print('xi[diffs] = ',xi[diff_index])
    print('corr2.xi[diffs] = ',corr2_output['xi'][diff_index])
    print('diff[diffs] = ',xi[diff_index] - corr2_output['xi'][diff_index])
    np.testing.assert_allclose(corr2_output['xi'], xi, rtol=1.e-3)

    # Now calling out to the external corr2 executable.
    # Note: This is one of very few times we test the corr2 executable.
    # For most things, we just run the corr2 function so the test coverage is recorded.
    import subprocess
    corr2_exe = get_script_name('corr2')
    p = subprocess.Popen( [corr2_exe,"configs/nn_direct.yaml","verbose=0"] )
    p.communicate()
    corr2_output = np.genfromtxt(os.path.join('output','nn_direct.out'), names=True,
                                    skip_header=1)
    np.testing.assert_allclose(corr2_output['xi'], xi, rtol=1.e-3)

    # Repeat with binslop = 0, since the code flow is different from brute=True
    dd = treecorr.NNCorrelation(min_sep=min_sep, max_sep=max_sep, nbins=nbins, bin_slop=0)
    dd.process(cat1, cat2)
    np.testing.assert_array_equal(dd.npairs, true_npairs)

    # And again with no top-level recursion
    dd = treecorr.NNCorrelation(min_sep=min_sep, max_sep=max_sep, nbins=nbins, bin_slop=0,
                                max_top=0)
    dd.process(cat1, cat2)
    np.testing.assert_array_equal(dd.npairs, true_npairs)

    # Do this one with corr2 to test the automatic output_dots=True functionality.
    # It's not particularly annoying with max_top = 0.
    config = treecorr.config.read_config('configs/nn_direct.yaml')
    config['verbose'] = 2
    config['max_top'] = 0
    config['bin_slop'] = 0
    treecorr.corr2(config)
    data = np.genfromtxt(config['nn_file_name'], names=True, skip_header=1)
    np.testing.assert_array_equal(data['npairs'], true_npairs)

    # Check a few basic operations with a NNCorrelation object.
    do_pickle(dd)

    dd2 = dd.copy()
    dd2 += dd
    np.testing.assert_allclose(dd2.npairs, 2*dd.npairs)
    np.testing.assert_allclose(dd2.weight, 2*dd.weight)
    np.testing.assert_allclose(dd2.meanr, 2*dd.meanr)
    np.testing.assert_allclose(dd2.meanlogr, 2*dd.meanlogr)

    dd2.clear()
    dd2 += dd
    np.testing.assert_allclose(dd2.npairs, dd.npairs)
    np.testing.assert_allclose(dd2.weight, dd.weight)
    np.testing.assert_allclose(dd2.meanr, dd.meanr)
    np.testing.assert_allclose(dd2.meanlogr, dd.meanlogr)

    ascii_name = 'output/dd_ascii.txt'
    dd.write(ascii_name, precision=16, file_type='ascii')
    dd3 = treecorr.NNCorrelation(min_sep=min_sep, max_sep=max_sep, nbins=nbins)
    dd3.read(ascii_name, file_type='ascii')
    np.testing.assert_allclose(dd3.npairs, dd.npairs)
    np.testing.assert_allclose(dd3.weight, dd.weight)
    np.testing.assert_allclose(dd3.meanr, dd.meanr)
    np.testing.assert_allclose(dd3.meanlogr, dd.meanlogr)

    try:
        import fitsio
    except ImportError:
        print('Skipping FITS tests, since fitsio is not installed')
        return

    # For this one, also check that it automatically makes the directory if necessary.
    shutil.rmtree('output/tmp', ignore_errors=True)
    fits_name = 'output/tmp/dd_fits.fits'
    dd.write(fits_name)
    dd4 = treecorr.NNCorrelation(min_sep=min_sep, max_sep=max_sep, nbins=nbins)
    dd4.read(fits_name)
    np.testing.assert_allclose(dd4.npairs, dd.npairs)
    np.testing.assert_allclose(dd4.weight, dd.weight)
    np.testing.assert_allclose(dd4.meanr, dd.meanr)
    np.testing.assert_allclose(dd4.meanlogr, dd.meanlogr)


def test_direct_spherical():
    # Repeat in spherical coords

    ngal = 100
    s = 10.
    rng = np.random.RandomState(8675309)
    x1 = rng.normal(0,s, (ngal,) )
    y1 = rng.normal(0,s, (ngal,) ) + 200  # Put everything at large y, so small angle on sky
    z1 = rng.normal(0,s, (ngal,) )
    w1 = rng.random_sample(ngal)

    x2 = rng.normal(0,s, (ngal,) )
    y2 = rng.normal(0,s, (ngal,) ) + 200
    z2 = rng.normal(0,s, (ngal,) )
    w2 = rng.random_sample(ngal)

    ra1, dec1 = coord.CelestialCoord.xyz_to_radec(x1,y1,z1)
    ra2, dec2 = coord.CelestialCoord.xyz_to_radec(x2,y2,z2)

    cat1 = treecorr.Catalog(ra=ra1, dec=dec1, ra_units='rad', dec_units='rad', w=w1)
    cat2 = treecorr.Catalog(ra=ra2, dec=dec2, ra_units='rad', dec_units='rad', w=w2)

    min_sep = 1.
    max_sep = 10.
    nbins = 50
    bin_size = np.log(max_sep/min_sep) / nbins
    dd = treecorr.NNCorrelation(min_sep=min_sep, max_sep=max_sep, nbins=nbins,
                                sep_units='deg', brute=True)
    dd.process(cat1, cat2)

    r1 = np.sqrt(x1**2 + y1**2 + z1**2)
    r2 = np.sqrt(x2**2 + y2**2 + z2**2)
    x1 /= r1;  y1 /= r1;  z1 /= r1
    x2 /= r2;  y2 /= r2;  z2 /= r2

    north_pole = coord.CelestialCoord(0*coord.radians, 90*coord.degrees)

    log_min_sep = np.log(min_sep)
    log_max_sep = np.log(max_sep)
    true_npairs = np.zeros(nbins, dtype=int)
    true_weight = np.zeros(nbins, dtype=float)
    bin_size = (log_max_sep - log_min_sep) / nbins

    c1 = [coord.CelestialCoord(r*coord.radians, d*coord.radians) for (r,d) in zip(ra1, dec1)]
    c2 = [coord.CelestialCoord(r*coord.radians, d*coord.radians) for (r,d) in zip(ra2, dec2)]
    for i in range(ngal):
        for j in range(ngal):
            rsq = (x1[i]-x2[j])**2 + (y1[i]-y2[j])**2 + (z1[i]-z2[j])**2
            r = np.sqrt(rsq)
            r *= coord.radians / coord.degrees
            logr = np.log(r)

            index = np.floor(np.log(r/min_sep) / bin_size).astype(int)
            if index < 0 or index >= nbins:
                continue

            # Rotate shears to coordinates where line connecting is horizontal.
            # Original orientation is where north is up.
            theta1 = 90*coord.degrees - c1[i].angleBetween(north_pole, c2[j])
            theta2 = 90*coord.degrees - c2[j].angleBetween(c1[i], north_pole)
            exp2theta1 = np.cos(2*theta1) + 1j * np.sin(2*theta1)
            expm2theta2 = np.cos(2*theta2) - 1j * np.sin(2*theta2)

            ww = w1[i] * w2[j]

            true_npairs[index] += 1
            true_weight[index] += ww

    print('true_npairs = ',true_npairs)
    print('diff = ',dd.npairs - true_npairs)
    np.testing.assert_array_equal(dd.npairs, true_npairs)

    print('true_weight = ',true_weight)
    print('diff = ',dd.weight - true_weight)
    np.testing.assert_allclose(dd.weight, true_weight, rtol=1.e-5, atol=1.e-8)

    try:
        import fitsio
    except ImportError:
        print('Skipping FITS tests, since fitsio is not installed')
        return

    # Check that running via the corr2 script works correctly.
    config = treecorr.config.read_config('configs/nn_direct_spherical.yaml')
    cat1.write(config['file_name'])
    cat2.write(config['file_name2'])
    treecorr.corr2(config)
    data = fitsio.read(config['nn_file_name'])
    print(data.dtype)
    np.testing.assert_allclose(data['R_nom'], dd.rnom)
    np.testing.assert_allclose(data['npairs'], dd.npairs)
    np.testing.assert_allclose(data['DD'], dd.weight)

    # Repeat with binslop = 0
    # And don't do any top-level recursion so we actually test not going to the leaves.
    dd = treecorr.NNCorrelation(min_sep=min_sep, max_sep=max_sep, nbins=nbins,
                                sep_units='deg', bin_slop=0, max_top=0)
    dd.process(cat1, cat2)
    np.testing.assert_array_equal(dd.npairs, true_npairs)
    np.testing.assert_allclose(dd.weight, true_weight, rtol=1.e-5, atol=1.e-8)


def test_pairwise():
    # Test the pairwise option.

    ngal = 1000
    s = 10.
    rng = np.random.RandomState(8675309)
    x1 = rng.normal(0,s, (ngal,) )
    y1 = rng.normal(0,s, (ngal,) )
    w1 = rng.random_sample(ngal)

    x2 = rng.normal(0,s, (ngal,) )
    y2 = rng.normal(0,s, (ngal,) )
    w2 = rng.random_sample(ngal)

    w1 = np.ones_like(w1)
    w2 = np.ones_like(w2)

    cat1 = treecorr.Catalog(x=x1, y=y1, w=w1)
    cat2 = treecorr.Catalog(x=x2, y=y2, w=w2)

    min_sep = 5.
    max_sep = 50.
    nbins = 10
    bin_size = np.log(max_sep/min_sep) / nbins
    dd = treecorr.NNCorrelation(min_sep=min_sep, max_sep=max_sep, nbins=nbins)
    dd.process_pairwise(cat1, cat2)
    dd.finalize()

    true_npairs = np.zeros(nbins, dtype=int)
    true_weight = np.zeros(nbins, dtype=float)

    rsq = (x1-x2)**2 + (y1-y2)**2
    r = np.sqrt(rsq)
    logr = np.log(r)
    expmialpha = ((x1-x2) - 1j*(y1-y2)) / r

    ww = w1 * w2

    index = np.floor(np.log(r/min_sep) / bin_size).astype(int)
    mask = (index >= 0) & (index < nbins)
    np.add.at(true_npairs, index[mask], 1)
    np.add.at(true_weight, index[mask], ww[mask])

    np.testing.assert_array_equal(dd.npairs, true_npairs)
    np.testing.assert_allclose(dd.weight, true_weight, rtol=1.e-5, atol=1.e-8)

    # If cats have names, then the logger will mention them.
    # Also, test running with optional args.
    cat1.name = "first"
    cat2.name = "second"
    with CaptureLog() as cl:
        dd.logger = cl.logger
        dd.process_pairwise(cat1, cat2, metric='Euclidean', num_threads=2)
    assert "for cats first, second" in cl.output



def test_direct_3d():
    # This is the same as the above test, but using the 3d correlations

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
    dd.process(cat1, cat2)
    print('dd.npairs = ',dd.npairs)

    log_min_sep = np.log(min_sep)
    log_max_sep = np.log(max_sep)
    true_npairs = np.zeros(nbins)
    bin_size = (log_max_sep - log_min_sep) / nbins
    for i in range(ngal):
        for j in range(ngal):
            rsq = (x1[i]-x2[j])**2 + (y1[i]-y2[j])**2 + (z1[i]-z2[j])**2
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
    dd.process(cat1, cat2)
    np.testing.assert_array_equal(dd.npairs, true_npairs)


def test_direct_perp():
    # This is the same as the above test, but using the perpendicular distance metric

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


def test_direct_old_perp():
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

    # And for now, Rperp is the same as OldRperp
    dd.process(cat1, cat2, metric='Rperp')
    np.testing.assert_array_equal(dd.npairs, true_npairs)


def test_direct_lens():
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


def test_direct_arc():
    # This is the same as the above test, but using the Arc distance metric

    ngal = 100
    s = 10.
    rng = np.random.RandomState(8675309)
    x1 = rng.normal(312, s, (ngal,) )
    y1 = rng.normal(728, s, (ngal,) )
    z1 = rng.normal(-932, s, (ngal,) )
    r1 = np.sqrt( x1*x1 + y1*y1 + z1*z1 )
    dec1 = np.arcsin(z1/r1)
    ra1 = np.arctan2(y1,x1)
    cat1 = treecorr.Catalog(ra=ra1, dec=dec1, ra_units='rad', dec_units='rad')

    x2 = rng.normal(312, s, (ngal,) )
    y2 = rng.normal(728, s, (ngal,) )
    z2 = rng.normal(-932, s, (ngal,) )
    r2 = np.sqrt( x2*x2 + y2*y2 + z2*z2 )
    dec2 = np.arcsin(z2/r2)
    ra2 = np.arctan2(y2,x2)
    cat2 = treecorr.Catalog(ra=ra2, dec=dec2, ra_units='rad', dec_units='rad')

    min_sep = 1.
    max_sep = 50.
    nbins = 50
    dd = treecorr.NNCorrelation(min_sep=min_sep, max_sep=max_sep, nbins=nbins, brute=True,
                                sep_units='arcmin')
    dd.process(cat1, cat2, metric='Arc')
    print('dd.npairs = ',dd.npairs)

    log_min_sep = np.log(min_sep)
    log_max_sep = np.log(max_sep)
    true_npairs = np.zeros(nbins)
    bin_size = (log_max_sep - log_min_sep) / nbins
    for i in range(ngal):
        for j in range(ngal):
            c1 = coord.CelestialCoord(ra1[i] * coord.radians, dec1[i] * coord.radians)
            c2 = coord.CelestialCoord(ra2[j] * coord.radians, dec2[j] * coord.radians)
            theta = c1.distanceTo(c2)
            theta /= coord.arcmin
            logr = np.log(theta)
            k = int(np.floor( (logr-log_min_sep) / bin_size ))
            if k < 0: continue
            if k >= nbins: continue
            true_npairs[k] += 1

    print('true_npairs = ',true_npairs)
    print('diff = ',dd.npairs - true_npairs)
    np.testing.assert_array_equal(dd.npairs, true_npairs)

    # Repeat with cat2 using 3d positions
    cat2r = treecorr.Catalog(ra=ra2, dec=dec2, r=r2, ra_units='rad', dec_units='rad')
    dd2r = treecorr.NNCorrelation(min_sep=min_sep, max_sep=max_sep, nbins=nbins, brute=True,
                                  sep_units='arcmin')
    dd2r.process(cat1, cat2r, metric='Arc')
    print('dd2r.npairs = ',dd2r.npairs)
    print('true_npairs = ',true_npairs)
    print('diff = ',dd2r.npairs - true_npairs)
    np.testing.assert_array_equal(dd2r.npairs, true_npairs)

    # And cat1 with 3d positions
    cat1r = treecorr.Catalog(ra=ra1, dec=dec1, r=r1, ra_units='rad', dec_units='rad')
    dd1r = treecorr.NNCorrelation(min_sep=min_sep, max_sep=max_sep, nbins=nbins, brute=True,
                                  sep_units='arcmin')
    dd1r.process(cat1r, cat2, metric='Arc')
    print('dd1r.npairs = ',dd1r.npairs)
    print('true_npairs = ',true_npairs)
    print('diff = ',dd1r.npairs - true_npairs)
    np.testing.assert_array_equal(dd1r.npairs, true_npairs)

    # And now both with 3d positions
    ddr = treecorr.NNCorrelation(min_sep=min_sep, max_sep=max_sep, nbins=nbins, brute=True,
                                 sep_units='arcmin')
    ddr.process(cat1r, cat2r, metric='Arc')
    print('ddr.npairs = ',ddr.npairs)
    print('true_npairs = ',true_npairs)
    print('diff = ',ddr.npairs - true_npairs)
    np.testing.assert_array_equal(ddr.npairs, true_npairs)

    # Can't use flat with Arc
    cat2f = treecorr.Catalog(x=x2, y=y2, x_units='rad', y_units='rad')
    with assert_raises(ValueError):
        ddr.process(cat1, cat2f, metric='Arc')
    with assert_raises(ValueError):
        ddr.process(cat2f, cat2, metric='Arc')
    with assert_raises(ValueError):
        ddr.process(cat2f, cat2r, metric='Arc')


def test_direct_partial():
    # There are two ways to specify using only parts of a catalog:
    # 1. The parameters first_row and last_row would usually be used for files, but they are a
    #    general way to use only a (contiguous) subset of the rows
    # 2. You can also set weights to 0 for the rows you don't want to use.

    # First test first_row, last_row
    ngal = 200
    s = 10.
    rng = np.random.RandomState(8675309)
    x1 = rng.normal(0,s, (ngal,) )
    y1 = rng.normal(0,s, (ngal,) )
    cat1a = treecorr.Catalog(x=x1, y=y1, first_row=28, last_row=144)
    x2 = rng.normal(0,s, (ngal,) )
    y2 = rng.normal(0,s, (ngal,) )
    cat2a = treecorr.Catalog(x=x2, y=y2, first_row=48, last_row=129)

    min_sep = 1.
    max_sep = 50.
    nbins = 50
    dda = treecorr.NNCorrelation(min_sep=min_sep, max_sep=max_sep, nbins=nbins, brute=True)
    dda.process(cat1a, cat2a)
    print('dda.npairs = ',dda.npairs)

    log_min_sep = np.log(min_sep)
    log_max_sep = np.log(max_sep)
    true_npairs = np.zeros(nbins)
    bin_size = (log_max_sep - log_min_sep) / nbins
    for i in range(27,144):
        for j in range(47,129):
            rsq = (x1[i]-x2[j])**2 + (y1[i]-y2[j])**2
            logr = 0.5 * np.log(rsq)
            k = int(np.floor( (logr-log_min_sep) / bin_size ))
            if k < 0: continue
            if k >= nbins: continue
            true_npairs[k] += 1

    print('true_npairs = ',true_npairs)
    print('diff = ',dda.npairs - true_npairs)
    np.testing.assert_array_equal(dda.npairs, true_npairs)

    # Now check that we get the same thing with all the points, but with w=0 for the ones
    # we don't want.
    w1 = np.zeros(ngal)
    w1[27:144] = 1.
    w2 = np.zeros(ngal)
    w2[47:129] = 1.
    cat1b = treecorr.Catalog(x=x1, y=y1, w=w1)
    cat2b = treecorr.Catalog(x=x2, y=y2, w=w2)
    ddb = treecorr.NNCorrelation(min_sep=min_sep, max_sep=max_sep, nbins=nbins, brute=True)
    ddb.process(cat1b, cat2b)
    print('ddb.npairs = ',ddb.npairs)
    print('diff = ',ddb.npairs - true_npairs)
    np.testing.assert_array_equal(ddb.npairs, true_npairs)


def test_direct_linear():
    # This is the same as test_direct_3d, but using linear binning

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
    nbins = 49
    dd = treecorr.NNCorrelation(min_sep=min_sep, max_sep=max_sep, nbins=nbins, brute=True,
                                bin_type='Linear')
    dd.process(cat1, cat2)
    print('dd.npairs = ',dd.npairs)

    true_npairs = np.zeros(nbins)
    bin_size = (max_sep - min_sep) / nbins
    for i in range(ngal):
        for j in range(ngal):
            rsq = (x1[i]-x2[j])**2 + (y1[i]-y2[j])**2 + (z1[i]-z2[j])**2
            r = np.sqrt(rsq)
            k = int(np.floor( (r-min_sep) / bin_size ))
            if k < 0: continue
            if k >= nbins: continue
            true_npairs[k] += 1

    print('true_npairs = ',true_npairs)
    print('diff = ',dd.npairs - true_npairs)
    np.testing.assert_array_equal(dd.npairs, true_npairs)

    # Can also specify coords directly as x,y,z
    cat1 = treecorr.Catalog(x=x1, y=y1, z=z1)
    cat2 = treecorr.Catalog(x=x2, y=y2, z=z2)
    dd.process(cat1, cat2)
    np.testing.assert_array_equal(dd.npairs, true_npairs)

    # Check that running Linear binning via the corr2 script works correctly.
    file_name1 = os.path.join('data','nn_linear_data1.dat')
    with open(file_name1, 'w') as fid:
        for i in range(ngal):
            fid.write(('%.20f %.20f %.20f\n')%(x1[i],y1[i],z1[i]))
    file_name2 = os.path.join('data','nn_linear_data2.dat')
    with open(file_name2, 'w') as fid:
        for i in range(ngal):
            fid.write(('%.20f %.20f %.20f\n')%(x2[i],y2[i],z2[i]))
    L = 10*s
    nrand = ngal
    rx1 = (rng.random_sample(nrand)-0.5) * L
    ry1 = (rng.random_sample(nrand)-0.5) * L
    rz1 = (rng.random_sample(nrand)-0.5) * L
    rx2 = (rng.random_sample(nrand)-0.5) * L
    ry2 = (rng.random_sample(nrand)-0.5) * L
    rz2 = (rng.random_sample(nrand)-0.5) * L
    rcat1 = treecorr.Catalog(x=rx1, y=ry1, z=rz1)
    rcat2 = treecorr.Catalog(x=rx2, y=ry2, z=rz2)
    rand_file_name1 = os.path.join('data','nn_linear_rand1.dat')
    with open(rand_file_name1, 'w') as fid:
        for i in range(nrand):
            fid.write(('%.20f %.20f %.20f\n')%(rx1[i],ry1[i],rz1[i]))
    rand_file_name2 = os.path.join('data','nn_linear_rand2.dat')
    with open(rand_file_name2, 'w') as fid:
        for i in range(nrand):
            fid.write(('%.20f %.20f %.20f\n')%(rx2[i],ry2[i],rz2[i]))
    rr = treecorr.NNCorrelation(min_sep=min_sep, max_sep=max_sep, nbins=nbins, brute=True,
                                bin_type='Linear', verbose=0)
    rr.process(rcat1,rcat2)
    dr = treecorr.NNCorrelation(min_sep=min_sep, max_sep=max_sep, nbins=nbins, brute=True,
                                bin_type='Linear', verbose=0)
    dr.process(cat1,rcat2)
    rd = treecorr.NNCorrelation(min_sep=min_sep, max_sep=max_sep, nbins=nbins, brute=True,
                                bin_type='Linear', verbose=0)
    rd.process(rcat1,cat2)
    xi, varxi = dd.calculateXi(rr, dr, rd)

    config = treecorr.config.read_config('configs/nn_linear.yaml')
    logger = treecorr.config.setup_logger(0)
    treecorr.corr2(config, logger)
    corr2_output = np.genfromtxt(os.path.join('output','nn_linear.out'), names=True,
                                    skip_header=1)
    print('corr2_output = ',corr2_output)
    print('corr2_output.dtype = ',corr2_output.dtype)
    print('rnom = ',dd.rnom)
    print('       ',corr2_output['R_nom'])
    np.testing.assert_allclose(corr2_output['R_nom'], dd.rnom, rtol=1.e-3)
    print('DD = ',dd.npairs)
    print('     ',corr2_output['DD'])
    np.testing.assert_allclose(corr2_output['DD'], dd.npairs, rtol=1.e-3)
    np.testing.assert_allclose(corr2_output['npairs'], dd.npairs, rtol=1.e-3)
    print('RR = ',rr.npairs)
    print('     ',corr2_output['RR'])
    np.testing.assert_allclose(corr2_output['RR'], rr.npairs, rtol=1.e-3)
    print('xi = ',xi)
    print('from corr2 output = ',corr2_output['xi'])
    print('diff = ',corr2_output['xi']-xi)
    diff_index = np.where(np.abs(corr2_output['xi']-xi) > 1.e-5)[0]
    print('different at ',diff_index)
    print('xi[diffs] = ',xi[diff_index])
    print('corr2.xi[diffs] = ',corr2_output['xi'][diff_index])
    print('diff[diffs] = ',xi[diff_index] - corr2_output['xi'][diff_index])
    np.testing.assert_allclose(corr2_output['xi'], xi, rtol=1.e-3)

    # Repeat with binslop = 0
    dd = treecorr.NNCorrelation(min_sep=min_sep, max_sep=max_sep, nbins=nbins, bin_slop=0,
                                bin_type='Linear')
    dd.process(cat1, cat2)
    np.testing.assert_array_equal(dd.npairs, true_npairs)

    # And again with no top-level recursion
    dd = treecorr.NNCorrelation(min_sep=min_sep, max_sep=max_sep, nbins=nbins, bin_slop=0,
                                bin_type='Linear', max_top=0)
    dd.process(cat1, cat2)
    np.testing.assert_array_equal(dd.npairs, true_npairs)



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
        tol_factor = 1
    else:
        ngal = 100000
        nrand = 2 * ngal
        L = 20. * s
        tol_factor = 3
    rng = np.random.RandomState(8675309)
    x = rng.normal(0,s, (ngal,) )
    y = rng.normal(0,s, (ngal,) )

    cat = treecorr.Catalog(x=x, y=y, x_units='arcmin', y_units='arcmin')
    dd = treecorr.NNCorrelation(bin_size=0.1, min_sep=1., max_sep=25., sep_units='arcmin')
    dd.process(cat)
    print('dd.npairs = ',dd.npairs)

    # log(<R>) != <logR>, but it should be close:
    print('meanlogr - log(meanr) = ',dd.meanlogr - np.log(dd.meanr))
    np.testing.assert_allclose(dd.meanlogr, np.log(dd.meanr), atol=1.e-3)

    rx = (rng.random_sample(nrand)-0.5) * L
    ry = (rng.random_sample(nrand)-0.5) * L
    rand = treecorr.Catalog(x=rx,y=ry, x_units='arcmin', y_units='arcmin')
    rr = treecorr.NNCorrelation(bin_size=0.1, min_sep=1., max_sep=25., sep_units='arcmin')
    rr.process(rand)
    print('rr.npairs = ',rr.npairs)

    dr = treecorr.NNCorrelation(bin_size=0.1, min_sep=1., max_sep=25., sep_units='arcmin')
    dr.process(cat,rand)
    print('dr.npairs = ',dr.npairs)

    r = dd.meanr
    true_xi = 0.25/np.pi * (L/s)**2 * np.exp(-0.25*r**2/s**2) - 1.

    xi, varxi = dd.calculateXi(rr,dr)
    print('xi = ',xi)
    print('true_xi = ',true_xi)
    print('ratio = ',xi / true_xi)
    print('diff = ',xi - true_xi)
    print('max rel diff = ',max(abs((xi - true_xi)/true_xi)))
    # This isn't super accurate.  But the agreement improves as L increase, so I think it is
    # merely a matter of the finite field and the integrals going to infinity.  (Sort of, since
    # we still have L in there.)
    np.testing.assert_allclose(xi, true_xi, rtol=0.1*tol_factor)
    np.testing.assert_allclose(np.log(np.abs(xi)), np.log(np.abs(true_xi)),
                                  atol=0.1*tol_factor)

    simple_xi, simple_varxi = dd.calculateXi(rr)
    print('simple xi = ',simple_xi)
    print('max rel diff = ',max(abs((simple_xi - true_xi)/true_xi)))
    # The simple calculation (i.e. dd/rr-1, rather than (dd-2dr+rr)/rr as above) is only
    # slightly less accurate in this case.  Probably because the mask is simple (a box), so
    # the difference is relatively minor.  The error is slightly higher in this case, but testing
    # that it is everywhere < 0.1 is still appropriate.
    np.testing.assert_allclose(simple_xi, true_xi, rtol=0.1*tol_factor)

    # Check that we get the same result using the corr2 function:
    cat.write(os.path.join('data','nn_data.dat'))
    rand.write(os.path.join('data','nn_rand.dat'))
    config = treecorr.config.read_config('configs/nn.yaml')
    config['verbose'] = 0
    config['precision'] = 8
    treecorr.corr2(config)
    out_file_name = os.path.join('output','nn.out')
    corr2_output = np.genfromtxt(out_file_name, names=True, skip_header=1)
    print('xi = ',xi)
    print('from corr2 output = ',corr2_output['xi'])
    print('ratio = ',corr2_output['xi']/xi)
    print('diff = ',corr2_output['xi']-xi)
    np.testing.assert_allclose(corr2_output['xi'], xi, rtol=1.e-3)

    # Check the read function (not at very high accuracy for the ASCII I/O)
    dd2 = treecorr.NNCorrelation(bin_size=0.1, min_sep=1., max_sep=25., sep_units='arcmin')
    dd2.read(out_file_name)
    np.testing.assert_allclose(dd2.logr, dd.logr, rtol=1.e-3)
    np.testing.assert_allclose(dd2.meanr, dd.meanr, rtol=1.e-3)
    np.testing.assert_allclose(dd2.meanlogr, dd.meanlogr, rtol=1.e-3)
    np.testing.assert_allclose(dd2.npairs, dd.npairs, rtol=1.e-3)
    np.testing.assert_allclose(dd2.tot, dd.tot, rtol=1.e-3)
    assert dd2.coords == dd.coords
    assert dd2.metric == dd.metric
    assert dd2.sep_units == dd.sep_units
    assert dd2.bin_type == dd.bin_type

    try:
        import fitsio
    except ImportError:
        print('Skipping FITS tests, since fitsio is not installed')
        return

    # Check the fits write option
    out_file_name1 = os.path.join('output','nn_out1.fits')
    dd.write(out_file_name1)
    data = fitsio.read(out_file_name1)
    np.testing.assert_almost_equal(data['R_nom'], np.exp(dd.logr))
    np.testing.assert_almost_equal(data['meanR'], dd.meanr)
    np.testing.assert_almost_equal(data['meanlogR'], dd.meanlogr)
    np.testing.assert_almost_equal(data['npairs'], dd.npairs)
    header = fitsio.read_header(out_file_name1, 1)
    np.testing.assert_almost_equal(header['tot'], dd.tot)

    out_file_name2 = os.path.join('output','nn_out2.fits')
    dd.write(out_file_name2, rr)
    data = fitsio.read(out_file_name2)
    np.testing.assert_almost_equal(data['R_nom'], np.exp(dd.logr))
    np.testing.assert_almost_equal(data['meanR'], dd.meanr)
    np.testing.assert_almost_equal(data['meanlogR'], dd.meanlogr)
    np.testing.assert_almost_equal(data['xi'], simple_xi)
    np.testing.assert_almost_equal(data['sigma_xi'], np.sqrt(simple_varxi))
    np.testing.assert_almost_equal(data['DD'], dd.npairs)
    np.testing.assert_almost_equal(data['RR'], rr.npairs * (dd.tot / rr.tot))
    header = fitsio.read_header(out_file_name2, 1)
    np.testing.assert_almost_equal(header['tot'], dd.tot)

    out_file_name3 = os.path.join('output','nn_out3.fits')
    dd.write(out_file_name3, rr, dr)
    data = fitsio.read(out_file_name3)
    np.testing.assert_almost_equal(data['R_nom'], np.exp(dd.logr))
    np.testing.assert_almost_equal(data['meanR'], dd.meanr)
    np.testing.assert_almost_equal(data['meanlogR'], dd.meanlogr)
    np.testing.assert_almost_equal(data['xi'], xi)
    np.testing.assert_almost_equal(data['sigma_xi'], np.sqrt(varxi))
    np.testing.assert_almost_equal(data['DD'], dd.npairs)
    np.testing.assert_almost_equal(data['RR'], rr.npairs * (dd.tot / rr.tot))
    np.testing.assert_almost_equal(data['DR'], dr.npairs * (dd.tot / dr.tot))
    header = fitsio.read_header(out_file_name3, 1)
    np.testing.assert_almost_equal(header['tot'], dd.tot)

    # Check the read function
    dd2 = treecorr.NNCorrelation(bin_size=0.1, min_sep=1., max_sep=25., sep_units='arcmin')
    dd2.read(out_file_name1)
    np.testing.assert_almost_equal(dd2.logr, dd.logr)
    np.testing.assert_almost_equal(dd2.meanr, dd.meanr)
    np.testing.assert_almost_equal(dd2.meanlogr, dd.meanlogr)
    np.testing.assert_almost_equal(dd2.npairs, dd.npairs)
    np.testing.assert_almost_equal(dd2.tot, dd.tot)
    assert dd2.coords == dd.coords
    assert dd2.metric == dd.metric
    assert dd2.sep_units == dd.sep_units
    assert dd2.bin_type == dd.bin_type

    dd2.read(out_file_name3)
    np.testing.assert_almost_equal(dd2.logr, dd.logr)
    np.testing.assert_almost_equal(dd2.meanr, dd.meanr)
    np.testing.assert_almost_equal(dd2.meanlogr, dd.meanlogr)
    np.testing.assert_almost_equal(dd2.npairs, dd.npairs)
    np.testing.assert_almost_equal(dd2.tot, dd.tot)
    assert dd2.coords == dd.coords
    assert dd2.metric == dd.metric
    assert dd2.sep_units == dd.sep_units
    assert dd2.bin_type == dd.bin_type


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
        tol_factor = 1
    else:
        ngal = 20000
        nrand = 2 * ngal
        L = 20. * s
        tol_factor = 3
    rng = np.random.RandomState(8675309)
    x = rng.normal(xcen, s, (ngal,) )
    y = rng.normal(ycen, s, (ngal,) )
    z = rng.normal(zcen, s, (ngal,) )

    r = np.sqrt(x*x+y*y+z*z)
    dec = np.arcsin(z/r) * coord.radians / coord.degrees
    ra = np.arctan2(y,x) * coord.radians / coord.degrees

    cat = treecorr.Catalog(ra=ra, dec=dec, r=r, ra_units='deg', dec_units='deg')
    dd = treecorr.NNCorrelation(bin_size=0.1, min_sep=1., max_sep=25., verbose=1)
    dd.process(cat)
    print('dd.npairs = ',dd.npairs)

    rx = (rng.random_sample(nrand)-0.5) * L + xcen
    ry = (rng.random_sample(nrand)-0.5) * L + ycen
    rz = (rng.random_sample(nrand)-0.5) * L + zcen
    rr = np.sqrt(rx*rx+ry*ry+rz*rz)
    rdec = np.arcsin(rz/rr) * coord.radians / coord.degrees
    rra = np.arctan2(ry,rx) * coord.radians / coord.degrees
    rand = treecorr.Catalog(ra=rra, dec=rdec, r=rr, ra_units='deg', dec_units='deg')
    rr = treecorr.NNCorrelation(bin_size=0.1, min_sep=1., max_sep=25., verbose=1)
    rr.process(rand)
    print('rr.npairs = ',rr.npairs)

    dr = treecorr.NNCorrelation(bin_size=0.1, min_sep=1., max_sep=25., verbose=1)
    dr.process(cat,rand)
    print('dr.npairs = ',dr.npairs)

    r = dd.meanr
    true_xi = 1./(8.*np.pi**1.5) * (L/s)**3 * np.exp(-0.25*r**2/s**2) - 1.

    simple_xi, varxi = dd.calculateXi(rr)
    print('simple xi = ',simple_xi)
    print('true_xi = ',true_xi)
    print('max rel diff = ',max(abs((simple_xi - true_xi)/true_xi)))
    np.testing.assert_allclose(simple_xi, true_xi, rtol=0.1*tol_factor)
    np.testing.assert_allclose(np.log(np.abs(simple_xi)), np.log(np.abs(true_xi)),
                                  rtol=0.1*tol_factor)

    xi, varxi = dd.calculateXi(rr,dr)
    print('xi = ',xi)
    print('true_xi = ',true_xi)
    print('ratio = ',xi / true_xi)
    print('diff = ',xi - true_xi)
    print('max rel diff = ',max(abs((xi - true_xi)/true_xi)))
    np.testing.assert_allclose(xi, true_xi, rtol=0.1*tol_factor)
    np.testing.assert_allclose(np.log(np.abs(xi)), np.log(np.abs(true_xi)),
                                  rtol=0.1*tol_factor)

    try:
        import fitsio
    except ImportError:
        print('Skipping FITS tests, since fitsio is not installed')
        return

    # Check that we get the same result using the corr2 function:
    cat.write(os.path.join('data','nn_3d_data.dat'))
    rand.write(os.path.join('data','nn_3d_rand.dat'))
    config = treecorr.config.read_config('configs/nn_3d.yaml')
    config['verbose'] = 0
    treecorr.corr2(config)

    corr2_outfile = os.path.join('output','nn_3d.fits')
    corr2_output = fitsio.read(corr2_outfile)
    print('xi = ',xi)
    print('from corr2 output = ',corr2_output['xi'])
    print('ratio = ',corr2_output['xi']/xi)
    print('diff = ',corr2_output['xi']-xi)

    np.testing.assert_almost_equal(corr2_output['R_nom'], np.exp(dd.logr))
    np.testing.assert_almost_equal(corr2_output['meanR'], dd.meanr)
    np.testing.assert_almost_equal(corr2_output['meanlogR'], dd.meanlogr)
    np.testing.assert_almost_equal(corr2_output['xi'], xi)
    np.testing.assert_almost_equal(corr2_output['sigma_xi'], np.sqrt(varxi))
    np.testing.assert_almost_equal(corr2_output['DD'], dd.npairs)
    np.testing.assert_almost_equal(corr2_output['RR'], rr.npairs * (dd.tot / rr.tot))
    np.testing.assert_almost_equal(corr2_output['DR'], dr.npairs * (dd.tot / dr.tot))
    header = fitsio.read_header(corr2_outfile, 1)
    np.testing.assert_almost_equal(header['tot'], dd.tot)

    # And repeat with Catalogs that use x,y,z
    cat = treecorr.Catalog(x=x, y=y, z=z)
    rand = treecorr.Catalog(x=rx, y=ry, z=rz)
    dd.process(cat)
    rr.process(rand)
    dr.process(cat,rand)
    xi, varxi = dd.calculateXi(rr,dr)
    np.testing.assert_allclose(xi, true_xi, rtol=0.1*tol_factor)
    np.testing.assert_allclose(np.log(np.abs(xi)), np.log(np.abs(true_xi)),
                                  rtol=0.1*tol_factor)


def test_list():
    # Test that we can use a list of files for either data or rand or both.

    nobj = 5000
    rng = np.random.RandomState(8675309)

    ncats = 3
    data_cats = []
    rand_cats = []

    s = 10.
    L = 50. * s

    x = rng.normal(0,s, (nobj,ncats) )
    y = rng.normal(0,s, (nobj,ncats) )
    data_cats = [ treecorr.Catalog(x=x[:,k],y=y[:,k]) for k in range(ncats) ]
    rx = (rng.random_sample((nobj,ncats))-0.5) * L
    ry = (rng.random_sample((nobj,ncats))-0.5) * L
    rand_cats = [ treecorr.Catalog(x=rx[:,k],y=ry[:,k]) for k in range(ncats) ]
    print('made catalogs')

    dd = treecorr.NNCorrelation(bin_size=0.1, min_sep=1., max_sep=25., verbose=1)
    dd.process(data_cats)
    print('dd.npairs = ',dd.npairs)

    rr = treecorr.NNCorrelation(bin_size=0.1, min_sep=1., max_sep=25., verbose=1)
    rr.process(rand_cats)
    print('rr.npairs = ',rr.npairs)

    xi, varxi = dd.calculateXi(rr)
    print('xi = ',xi)

    # Now do the same thing with one big catalog for each.
    ddx = treecorr.NNCorrelation(bin_size=0.1, min_sep=1., max_sep=25., verbose=1)
    rrx = treecorr.NNCorrelation(bin_size=0.1, min_sep=1., max_sep=25., verbose=1)
    data_catx = treecorr.Catalog(x=x.reshape( (nobj*ncats,) ), y=y.reshape( (nobj*ncats,) ))
    rand_catx = treecorr.Catalog(x=rx.reshape( (nobj*ncats,) ), y=ry.reshape( (nobj*ncats,) ))
    print('made catalogs')
    ddx.process(data_catx)
    print('ddx.npairs = ',ddx.npairs)
    rrx.process(rand_catx)
    print('rrx.npairs = ',rrx.npairs)
    xix, varxix = ddx.calculateXi(rrx)

    print('ddx.npairs = ',ddx.npairs)
    print('rrx.npairs = ',rrx.npairs)
    print('xix = ',xix)
    print('ratio = ',xi/xix)
    print('diff = ',xi-xix)
    np.testing.assert_allclose(xix, xi, rtol=0.02)

    # Check that we get the same result using the corr2 function
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
    print('wrote dat files')

    list_name = os.path.join('data','nn_list_data_files.txt')
    with open(list_name, 'w') as fid:
        for file_name in file_list:
            fid.write('%s\n'%file_name)
    rand_list_name = os.path.join('data','nn_list_rand_files.txt')
    with open(rand_list_name, 'w') as fid:
        for file_name in rand_file_list:
            fid.write('%s\n'%file_name)
    print('wrote list files')

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
    print('wrote xdat files')

    # First do this via the corr2 function.
    config = treecorr.config.read_config('configs/nn_list1.yaml')
    print('config = ',config)
    logger = treecorr.config.setup_logger(0)
    treecorr.corr2(config, logger)
    print('finished corr2')
    corr2_output = np.genfromtxt(os.path.join('output','nn_list1.out'),names=True,skip_header=1)
    print('xi = ',xi)
    print('from corr2 output = ',corr2_output['xi'])
    print('ratio = ',corr2_output['xi']/xi)
    print('diff = ',corr2_output['xi']-xi)
    np.testing.assert_allclose(corr2_output['xi'], xi, rtol=1.e-3)

    # Now calling out to the external corr2 executable to test it with extra command-line params
    import subprocess
    corr2_exe = get_script_name('corr2')
    print('exe = ',corr2_exe)
    p = subprocess.Popen( [corr2_exe,"configs/nn_list1.yaml","verbose=0"] )
    p.communicate()
    print('finished corr2')
    corr2_output = np.genfromtxt(os.path.join('output','nn_list1.out'),names=True,skip_header=1)
    print('xi = ',xi)
    print('from corr2 output = ',corr2_output['xi'])
    print('ratio = ',corr2_output['xi']/xi)
    print('diff = ',corr2_output['xi']-xi)
    np.testing.assert_allclose(corr2_output['xi'], xi, rtol=1.e-3)

    config = treecorr.config.read_config('configs/nn_list2.json')
    treecorr.config.parse_variable(config, 'verbose=0')
    print('config = ',config)
    treecorr.corr2(config)
    print('finished corr2')
    corr2_output = np.genfromtxt(os.path.join('output','nn_list2.out'),names=True,skip_header=1)
    print('xi = ',xi)
    print('from corr2 output = ',corr2_output['xi'])
    print('ratio = ',corr2_output['xi']/xi)
    print('diff = ',corr2_output['xi']-xi)
    np.testing.assert_allclose(corr2_output['xi'], xi, rtol=2.e-2)

    config = treecorr.config.read_config('configs/nn_list3.params')
    treecorr.config.parse_variable(config, 'verbose=0')
    print('config = ',config)
    treecorr.corr2(config)
    print('finished corr2')
    corr2_output = np.genfromtxt(os.path.join('output','nn_list3.out'),names=True,skip_header=1)
    print('xi = ',xi)
    print('from corr2 output = ',corr2_output['xi'])
    print('ratio = ',corr2_output['xi']/xi)
    print('diff = ',corr2_output['xi']-xi)
    np.testing.assert_allclose(corr2_output['xi'], xi, rtol=2.e-2)

    config = treecorr.config.read_config('configs/nn_list4.config', file_type='yaml')
    treecorr.config.parse_variable(config, 'verbose=0')
    print('config = ',config)
    treecorr.corr2(config)
    print('finished corr2')
    corr2_output = np.genfromtxt(os.path.join('output','nn_list4.out'),names=True,skip_header=1)
    print('xi = ',xi)
    print('from corr2 output = ',corr2_output['xi'])
    print('ratio = ',corr2_output['xi']/xi)
    print('diff = ',corr2_output['xi']-xi)
    np.testing.assert_allclose(corr2_output['xi'], xi, rtol=1.e-2)

    # Repeat with exe to test -f flag
    p = subprocess.Popen( [corr2_exe, "-f", "yaml", "configs/nn_list4.config", "verbose=3"] )
    p.communicate()
    print('finished corr2_exe')
    corr2_output = np.genfromtxt(os.path.join('output','nn_list4.out'),names=True,skip_header=1)
    print('xi = ',xi)
    print('from corr2 output = ',corr2_output['xi'])
    print('ratio = ',corr2_output['xi']/xi)
    print('diff = ',corr2_output['xi']-xi)
    np.testing.assert_allclose(corr2_output['xi'], xi, rtol=1.e-2)

    config = treecorr.config.read_config('configs/nn_list5.config', file_type='json')
    treecorr.config.parse_variable(config, 'verbose=0')
    print('config = ',config)
    treecorr.corr2(config)
    print('finished corr2')
    corr2_output = np.genfromtxt(os.path.join('output','nn_list5.out'),names=True,skip_header=1)
    print('xi = ',xi)
    print('from corr2 output = ',corr2_output['xi'])
    print('ratio = ',corr2_output['xi']/xi)
    print('diff = ',corr2_output['xi']-xi)
    np.testing.assert_allclose(corr2_output['xi'], xi, rtol=1.e-2)

    # For this one, the output file is in the current directory, which used to give an error.
    config = treecorr.config.read_config('configs/nn_list6.config', file_type='params')
    treecorr.config.parse_variable(config, 'verbose=0')
    print('config = ',config)
    treecorr.corr2(config)
    print('finished corr2')
    output_file = 'nn_list6.out'
    corr2_output = np.genfromtxt(output_file,names=True,skip_header=1)
    print('xi = ',xi)
    print('from corr2 output = ',corr2_output['xi'])
    print('ratio = ',corr2_output['xi']/xi)
    print('diff = ',corr2_output['xi']-xi)
    np.testing.assert_allclose(corr2_output['xi'], xi, rtol=1.e-2)
    # Move it to the output directory now to keep the current directory clean.
    mv_output_file = os.path.join('output',output_file)
    print('move to ',mv_output_file)
    if os.path.exists(mv_output_file):
        os.remove(mv_output_file)
    os.rename(output_file, mv_output_file)
    print('done')

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
    dd1.process(dcat, metric='Rperp')

    lower_min_sep = config['min_sep'] * np.exp(-2.*config['bin_size'])
    more_nbins = config['nbins'] + 4
    dd2 = treecorr.NNCorrelation(config, min_sep=lower_min_sep, nbins=more_nbins)
    dd2.process(dcat, metric='Rperp')

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
        rr1.process(rcat, metric='Rperp')
        rr2 = treecorr.NNCorrelation(config, min_sep=lower_min_sep, nbins=more_nbins)
        rr2.process(rcat, metric='Rperp')
        print('rr1 npairs = ',rr1.npairs)
        print('rr2 npairs = ',rr2.npairs[2:-2])
        np.testing.assert_allclose(rr1.npairs, rr2.npairs[2:-2], rtol=1.e-6)

        dr1 = treecorr.NNCorrelation(config)
        dr1.process(dcat, rcat, metric='Rperp')
        dr2 = treecorr.NNCorrelation(config, min_sep=lower_min_sep, nbins=more_nbins)
        dr2.process(dcat, rcat, metric='Rperp')
        print('dr1 npairs = ',dr1.npairs)
        print('dr2 npairs = ',dr2.npairs[2:-2])
        np.testing.assert_allclose(dr1.npairs, dr2.npairs[2:-2], rtol=1.e-6)

        xi1, varxi1 = dd1.calculateXi(rr1, dr1)
        xi2, varxi2 = dd2.calculateXi(rr2, dr2)
        print('xi1 = ',xi1)
        print('xi2 = ',xi2[2:-2])
        np.testing.assert_allclose(xi1, xi2[2:-2], rtol=1.e-6)

    # Also check the new Rper metric
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

def test_split():
    # Time the various split_method options.

    ngal = 10000
    rng = np.random.RandomState(8675309)
    s = 10
    x = rng.normal(0,s, (ngal,) )
    y = rng.normal(0,s, (ngal,) )
    z = rng.normal(0,s, (ngal,) )
    cat = treecorr.Catalog(x=x, y=y, z=z)

    dd_mean = treecorr.NNCorrelation(bin_size=0.1, min_sep=5., max_sep=25., split_method='mean')
    t0 = time.time()
    dd_mean.process(cat)
    t1 = time.time()
    print('mean: time = ',t1-t0)
    print('npairs = ',dd_mean.npairs)

    dd_median = treecorr.NNCorrelation(bin_size=0.1, min_sep=5., max_sep=25., split_method='median')
    t0 = time.time()
    dd_median.process(cat)
    t1 = time.time()
    print('median: time = ',t1-t0)
    print('npairs = ',dd_median.npairs)

    dd_middle = treecorr.NNCorrelation(bin_size=0.1, min_sep=5., max_sep=25., split_method='middle')
    t0 = time.time()
    dd_middle.process(cat)
    t1 = time.time()
    print('middle: time = ',t1-t0)
    print('npairs = ',dd_middle.npairs)

    dd_random1 = treecorr.NNCorrelation(bin_size=0.1, min_sep=5., max_sep=25., split_method='random')
    t0 = time.time()
    dd_random1.process(cat)
    t1 = time.time()
    print('random1: time = ',t1-t0)
    print('npairs = ',dd_random1.npairs)

    # Random should be non-deterministic, so check a second version of it.
    # Need to clear the cache to get it to rebuild though.
    cat.nfields.clear()
    dd_random2 = treecorr.NNCorrelation(bin_size=0.1, min_sep=5., max_sep=25., split_method='random')
    t0 = time.time()
    dd_random2.process(cat)
    t1 = time.time()
    print('random2: time = ',t1-t0)
    print('npairs = ',dd_random2.npairs)

    # They should all be different, but not very.
    dd_list = [dd_mean, dd_median, dd_middle, dd_random1, dd_random2]
    for dd1 in dd_list:
        for dd2 in dd_list:
            if dd1 is dd2: continue
            assert not np.all(dd1.npairs == dd2.npairs)
            np.testing.assert_allclose(dd1.npairs, dd2.npairs,rtol=1.e-2)


def test_varxi():
    # Test that varxi is correct (or close) based on actual variance of many runs.

    L = 100
    rng = np.random.RandomState(8675309)

    if __name__ == '__main__':
        ngal = 50
        nrand = 200
        nruns = 50000
        tol_factor = 1
    else:
        ngal = 50
        nrand = 100
        nruns = 1000
        tol_factor = 5

    all_dds = []
    all_drs = []
    all_rrs = []
    for run in range(nruns):
        x1 = (rng.random_sample(ngal)-0.5) * L
        y1 = (rng.random_sample(ngal)-0.5) * L
        x2 = (rng.random_sample(nrand)-0.5) * L
        y2 = (rng.random_sample(nrand)-0.5) * L
        # Varied weights are hard, but at least check that non-unit weights work correctly.
        w = np.ones_like(x2) * 5
        wr = np.ones_like(x2) * 0.3

        data = treecorr.Catalog(x=x1, y=y1, w=w)
        rand = treecorr.Catalog(x=x2, y=y2, w=wr)
        dd = treecorr.NNCorrelation(bin_size=0.1, min_sep=6., max_sep=13.)
        dr = treecorr.NNCorrelation(bin_size=0.1, min_sep=6., max_sep=13.)
        rr = treecorr.NNCorrelation(bin_size=0.1, min_sep=6., max_sep=13.)
        dd.process(data)
        dr.process(data, rand)
        rr.process(rand)
        all_dds.append(dd)
        all_drs.append(dr)
        all_rrs.append(rr)

    print('Uncompensated:')

    all_xis = [dd.calculateXi(rr) for dd,rr in zip(all_dds, all_rrs)]
    mean_wt = np.mean([dd.weight for dd in all_dds], axis=0)
    mean_np = np.mean([dd.npairs for dd in all_dds], axis=0)
    mean_xi = np.mean([xi[0] for xi in all_xis], axis=0)
    var_xi = np.var([xi[0] for xi in all_xis], axis=0)
    mean_varxi = np.mean([xi[1] for xi in all_xis], axis=0)

    print('mean_xi = ',mean_xi)
    print('mean_wt = ',mean_wt)
    print('mean_np = ',mean_np)
    print('mean_varxi = ',mean_varxi)
    print('var_xi = ',var_xi)
    print('ratio = ',var_xi / mean_varxi)
    print('max relerr for xi = ',np.max(np.abs((var_xi - mean_varxi)/var_xi)))
    print('diff = ',var_xi - mean_varxi)
    np.testing.assert_allclose(mean_varxi, var_xi, rtol=0.1 * tol_factor)

    print('Compensated:')

    all_xis = [dd.calculateXi(rr, dr) for dd,dr,rr in zip(all_dds, all_drs, all_rrs)]
    mean_wt = np.mean([dd.weight for dd in all_dds], axis=0)
    mean_np = np.mean([dd.npairs for dd in all_dds], axis=0)
    mean_xi = np.mean([xi[0] for xi in all_xis], axis=0)
    var_xi = np.var([xi[0] for xi in all_xis], axis=0)
    mean_varxi = np.mean([xi[1] for xi in all_xis], axis=0)

    print('mean_xi = ',mean_xi)
    print('mean_wt = ',mean_wt)
    print('mean_np = ',mean_np)
    print('mean_varxi = ',mean_varxi)
    print('var_xi = ',var_xi)
    print('ratio = ',var_xi / mean_varxi)
    print('max relerr for xi = ',np.max(np.abs((var_xi - mean_varxi)/var_xi)))
    print('diff = ',var_xi - mean_varxi)
    np.testing.assert_allclose(mean_varxi, var_xi, rtol=0.05 * tol_factor)

    print('Compensated with both dr and rd:')

    all_xis = [dd.calculateXi(rr, dr, dr) for dd,dr,rr in zip(all_dds, all_drs, all_rrs)]
    mean_wt = np.mean([dd.weight for dd in all_dds], axis=0)
    mean_xi = np.mean([xi[0] for xi in all_xis], axis=0)
    var_xi = np.var([xi[0] for xi in all_xis], axis=0)
    mean_varxi = np.mean([xi[1] for xi in all_xis], axis=0)

    print('mean_xi = ',mean_xi)
    print('mean_wt = ',mean_wt)
    print('mean_varxi = ',mean_varxi)
    print('var_xi = ',var_xi)
    print('ratio = ',var_xi / mean_varxi)
    print('max relerr for xi = ',np.max(np.abs((var_xi - mean_varxi)/var_xi)))
    print('diff = ',var_xi - mean_varxi)
    np.testing.assert_allclose(mean_varxi, var_xi, rtol=0.05 * tol_factor)

    print('Compensated with just rd')

    all_xis = [dd.calculateXi(rr, rd=dr) for dd,dr,rr in zip(all_dds, all_drs, all_rrs)]
    mean_wt = np.mean([dd.weight for dd in all_dds], axis=0)
    mean_xi = np.mean([xi[0] for xi in all_xis], axis=0)
    var_xi = np.var([xi[0] for xi in all_xis], axis=0)
    mean_varxi = np.mean([xi[1] for xi in all_xis], axis=0)

    print('mean_xi = ',mean_xi)
    print('mean_wt = ',mean_wt)
    print('mean_varxi = ',mean_varxi)
    print('var_xi = ',var_xi)
    print('ratio = ',var_xi / mean_varxi)
    print('max relerr for xi = ',np.max(np.abs((var_xi - mean_varxi)/var_xi)))
    print('diff = ',var_xi - mean_varxi)
    np.testing.assert_allclose(mean_varxi, var_xi, rtol=0.05 * tol_factor)


def test_sph_linear():

    # Initially, there was an error using linear binning with sep_units.
    # This isn't always with Spherical coords, but that's the most typical case.
    # This unit test recapitulates the code in the report from Ismael Ferrero.
    config = {
        'nbins': 9,
        'min_sep'  : 0.5,
        'max_sep'  : 9.5,
        'sep_units':'degrees',
        'bin_type': 'Linear',
        'bin_slop' : 0.05,
        'metric' : 'Arc'
    }

    rng = np.random.RandomState(8675309)
    ngal = 100000
    x = rng.normal(10, 1, (ngal,) )
    y = rng.normal(30, 1, (ngal,) )
    z = rng.normal(20, 1, (ngal,) )
    r = np.sqrt(x*x+y*y+z*z)
    dec = np.arcsin(z/r) * coord.radians / coord.degrees
    ra = np.arctan2(y,x) * coord.radians / coord.degrees

    cat = treecorr.Catalog(ra=ra, dec=dec, ra_units='deg', dec_units='deg')

    dd = treecorr.NNCorrelation(config)
    dd.process(cat, num_threads=1)

    print('dd.rnom = ',dd.rnom)
    print('dd.meanr = ',dd.meanr)
    print('dd.npairs = ',dd.npairs)
    np.testing.assert_allclose(dd.rnom, range(1,10))
    np.testing.assert_allclose(dd.meanr, range(1,10), rtol=0.1)


if __name__ == '__main__':
    test_log_binning()
    test_linear_binning()
    test_direct_count()
    test_direct_3d()
    test_direct_perp()
    test_direct_lens()
    test_direct_arc()
    test_direct_partial()
    test_direct_linear()
    test_nn()
    test_3d()
    test_list()
    test_perp_minmax()
    test_split()
    test_sph_linear()
