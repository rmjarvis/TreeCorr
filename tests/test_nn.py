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

import numpy as np
import treecorr
import os
import coord
import time
import shutil

from test_helper import get_script_name, do_pickle, CaptureLog
from test_helper import assert_raises, timer, assert_warns

@timer
def test_log_binning():
    import math
    # Test some basic properties of the base class with respect to Log binning

    # Check the different ways to set up the binning:
    # Omit bin_size
    nn = treecorr.NNCorrelation(min_sep=5, max_sep=20, nbins=20, bin_type='Log')
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
    assert nn.bin_size <= 0.1
    assert nn.min_sep == 5.
    assert nn.max_sep == 20.
    np.testing.assert_almost_equal(nn.bin_size * nn.nbins, math.log(nn.max_sep/nn.min_sep))
    np.testing.assert_almost_equal(nn.logr[0], math.log(nn.min_sep) + 0.5*nn.bin_size)
    np.testing.assert_almost_equal(nn.logr[-1], math.log(nn.max_sep) - 0.5*nn.bin_size)
    assert len(nn.logr) == nn.nbins

    assert_raises(TypeError, treecorr.NNCorrelation)
    assert_raises(TypeError, treecorr.NNCorrelation, min_sep=5)
    assert_raises(TypeError, treecorr.NNCorrelation, max_sep=20)
    assert_raises(TypeError, treecorr.NNCorrelation, bin_size=0.1)
    assert_raises(TypeError, treecorr.NNCorrelation, nbins=20)
    assert_raises(TypeError, treecorr.NNCorrelation, min_sep=5, max_sep=20)
    assert_raises(TypeError, treecorr.NNCorrelation, min_sep=5, bin_size=0.1)
    assert_raises(TypeError, treecorr.NNCorrelation, min_sep=5, nbins=20)
    assert_raises(TypeError, treecorr.NNCorrelation, max_sep=5, bin_size=0.1)
    assert_raises(TypeError, treecorr.NNCorrelation, max_sep=5, nbins=20)
    assert_raises(TypeError, treecorr.NNCorrelation, bin_size=0.1, nbins=20)
    assert_raises(TypeError, treecorr.NNCorrelation, min_sep=5, max_sep=20, bin_size=0.1, nbins=20)
    assert_raises(ValueError, treecorr.NNCorrelation, min_sep=20, max_sep=5, bin_size=0.1)
    assert_raises(ValueError, treecorr.NNCorrelation, min_sep=20, max_sep=5, nbins=20)
    assert_raises(ValueError, treecorr.NNCorrelation, min_sep=5, max_sep=20, nbins=20,
                  bin_type='Invalid')

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
    nn = treecorr.NNCorrelation(min_sep=5, nbins=14, bin_size=0.1)
    print(nn.bin_size,nn.bin_slop,nn.b)
    assert nn.bin_size == 0.1
    assert nn.bin_slop == 1.0
    np.testing.assert_almost_equal(nn.b, 0.1)

    nn = treecorr.NNCorrelation(min_sep=5, nbins=14, bin_size=0.1, bin_slop=1.0)
    print(nn.bin_size,nn.bin_slop,nn.b)
    assert nn.bin_size == 0.1
    assert nn.bin_slop == 1.0
    np.testing.assert_almost_equal(nn.b, 0.1)

    nn = treecorr.NNCorrelation(min_sep=5, nbins=14, bin_size=0.1, bin_slop=0.2)
    print(nn.bin_size,nn.bin_slop,nn.b)
    assert nn.bin_size == 0.1
    assert nn.bin_slop == 0.2
    np.testing.assert_almost_equal(nn.b, 0.02)

    nn = treecorr.NNCorrelation(min_sep=5, nbins=14, bin_size=0.1, bin_slop=0.0)
    print(nn.bin_size,nn.bin_slop,nn.b)
    assert nn.bin_size == 0.1
    assert nn.bin_slop == 0.0
    assert nn.b == 0.0

    nn = treecorr.NNCorrelation(min_sep=5, nbins=14, bin_size=0.1, bin_slop=2.0, verbose=0)
    print(nn.bin_size,nn.bin_slop,nn.b)
    assert nn.bin_size == 0.1
    assert nn.bin_slop == 2.0
    np.testing.assert_almost_equal(nn.b, 0.2)

    nn = treecorr.NNCorrelation(min_sep=5, nbins=14, bin_size=0.4, bin_slop=1.0, verbose=0)
    print(nn.bin_size,nn.bin_slop,nn.b)
    assert nn.bin_size == 0.4
    assert nn.bin_slop == 1.0
    np.testing.assert_almost_equal(nn.b, 0.4)

    nn = treecorr.NNCorrelation(min_sep=5, nbins=14, bin_size=0.4)
    print(nn.bin_size,nn.bin_slop,nn.b)
    assert nn.bin_size == 0.4
    np.testing.assert_almost_equal(nn.b, 0.1)
    np.testing.assert_almost_equal(nn.bin_slop, 0.25)

    nn = treecorr.NNCorrelation(min_sep=5, nbins=14, bin_size=0.4, bin_slop=0.1)
    print(nn.bin_size,nn.bin_slop,nn.b)
    assert nn.bin_size == 0.4
    assert nn.bin_slop == 0.1
    np.testing.assert_almost_equal(nn.b, 0.04)

    nn = treecorr.NNCorrelation(min_sep=5, nbins=14, bin_size=0.05, bin_slop=1.0)
    print(nn.bin_size,nn.bin_slop,nn.b)
    assert nn.bin_size == 0.05
    assert nn.bin_slop == 1.0
    np.testing.assert_almost_equal(nn.b, 0.05)

    nn = treecorr.NNCorrelation(min_sep=5, nbins=14, bin_size=0.05)
    print(nn.bin_size,nn.bin_slop,nn.b)
    assert nn.bin_size == 0.05
    assert nn.bin_slop == 1.0
    np.testing.assert_almost_equal(nn.b, 0.05)

    nn = treecorr.NNCorrelation(min_sep=5, nbins=14, bin_size=0.05, bin_slop=3, verbose=0)
    print(nn.bin_size,nn.bin_slop,nn.b)
    assert nn.bin_size == 0.05
    assert nn.bin_slop == 3.0
    np.testing.assert_almost_equal(nn.b, 0.15)


@timer
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
    nn = treecorr.NNCorrelation(min_sep=5, max_sep=20, bin_size=0.13, bin_type='Linear')
    print(nn.min_sep,nn.max_sep,nn.bin_size,nn.nbins)
    assert nn.bin_size <= 0.13
    assert nn.min_sep == 5.
    assert nn.max_sep == 20.
    np.testing.assert_almost_equal(nn.bin_size * nn.nbins, nn.max_sep-nn.min_sep)
    np.testing.assert_almost_equal(nn.rnom[0], nn.min_sep + 0.5*nn.bin_size)
    np.testing.assert_almost_equal(nn.rnom[-1], nn.max_sep - 0.5*nn.bin_size)
    assert len(nn.rnom) == nn.nbins
    assert len(nn.logr) == nn.nbins

    # Check the use of sep_units
    # radians
    nn = treecorr.NNCorrelation(min_sep=5, max_sep=20, nbins=20, sep_units='radians',
                                bin_type='Linear')
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
    nn = treecorr.NNCorrelation(min_sep=5, max_sep=20, nbins=20, sep_units='arcsec',
                                bin_type='Linear')
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
    nn = treecorr.NNCorrelation(min_sep=5, max_sep=20, nbins=20, sep_units='arcmin',
                                bin_type='Linear')
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
    nn = treecorr.NNCorrelation(min_sep=5, max_sep=20, nbins=20, sep_units='degrees',
                                bin_type='Linear')
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
    nn = treecorr.NNCorrelation(min_sep=5, max_sep=20, nbins=20, sep_units='hours',
                                bin_type='Linear')
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


@timer
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
    xi, varxi = dd.calculateXi(rr=rr)

    # After calling calculateXi, you can access the result via attributes
    np.testing.assert_array_equal(xi, dd.xi)
    np.testing.assert_array_equal(varxi, dd.varxi)
    np.testing.assert_array_equal(varxi, dd.cov.diagonal())

    # rr is still allowed as a positional argument, but deprecated
    with assert_warns(FutureWarning):
        xi_2, varxi_2 = dd.calculateXi(rr)
    np.testing.assert_array_equal(xi_2, xi)
    np.testing.assert_array_equal(varxi_2, varxi)

    # First do this via the corr2 function.
    config = treecorr.config.read_config('configs/nn_direct.yaml')
    logger = treecorr.config.setup_logger(0)
    treecorr.corr2(config, logger)
    corr2_output = np.genfromtxt(os.path.join('output','nn_direct.out'), names=True,
                                 skip_header=1)
    print('corr2_output = ',corr2_output)
    print('corr2_output.dtype = ',corr2_output.dtype)
    print('rnom = ',dd.rnom)
    print('       ',corr2_output['r_nom'])
    np.testing.assert_allclose(corr2_output['r_nom'], dd.rnom, rtol=1.e-3)
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
    if os.name != 'nt':
        # The normal script doesn't execute on Windows.
        # If anyone knows how to change this for that platform, I'd welcome a PR.
        # Until then, skip it.
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

    # Invalid to omit file_name
    config['verbose'] = 0
    del config['file_name']
    with assert_raises(TypeError):
        treecorr.corr2(config)

    # Invalid to have rand_file_name2 but not file_name2
    config['file_name'] = 'data/nn_direct_data1.dat'
    del config['file_name2']
    with assert_raises(TypeError):
        treecorr.corr2(config)

    # Invalid when doing rands, to have file_name2, but not rand_file_name2
    config['file_name2'] = 'data/nn_direct_data2.dat'
    del config['rand_file_name2']
    with assert_raises(TypeError):
        treecorr.corr2(config)

    # OK to have neither rand_file_name nor rand_file_name2
    # Also, check the automatic setting of output_dots=True when verbose=2.
    # It's not too annoying if we also set max_top = 0.
    del config['rand_file_name']
    config['verbose'] = 2
    config['max_top'] = 0
    treecorr.corr2(config)
    data = np.genfromtxt(config['nn_file_name'], names=True, skip_header=1)
    np.testing.assert_array_equal(data['npairs'], true_npairs)
    assert 'xi' not in data.dtype.names

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

    with assert_raises(TypeError):
        dd2 += config
    dd4 = treecorr.NNCorrelation(min_sep=min_sep/2, max_sep=max_sep, nbins=nbins)
    with assert_raises(ValueError):
        dd2 += dd4
    dd5 = treecorr.NNCorrelation(min_sep=min_sep, max_sep=max_sep*2, nbins=nbins)
    with assert_raises(ValueError):
        dd2 += dd5
    dd6 = treecorr.NNCorrelation(min_sep=min_sep, max_sep=max_sep, nbins=nbins*2)
    with assert_raises(ValueError):
        dd2 += dd6

    # Cannot use some metrics with Flat coordinates
    dd7 = treecorr.NNCorrelation(min_sep=min_sep, max_sep=max_sep, nbins=nbins)
    with assert_raises(ValueError):
        dd7.process(cat1, cat2, metric='Rperp')
    with assert_raises(ValueError):
        dd7.process(cat1, cat2, metric='OldRperp')
    with assert_raises(ValueError):
        dd7.process(cat1, cat2, metric='FisherRperp')
    with assert_raises(ValueError):
        dd7.process(cat1, cat2, metric='Arc')
    with assert_raises(ValueError):
        dd7.process(cat1, metric='Arc')
    with assert_raises(ValueError):
        dd7.process(cat1, cat2, metric='Rlens')

    # For this one, also check that it automatically makes the directory if necessary.
    shutil.rmtree('output/tmp', ignore_errors=True)
    out_name = 'output/tmp/dd_out.dat'
    dd.write(out_name, precision=16)
    dd8 = treecorr.NNCorrelation(min_sep=min_sep, max_sep=max_sep, nbins=nbins)
    dd8.read(out_name)
    np.testing.assert_allclose(dd8.npairs, dd.npairs)
    np.testing.assert_allclose(dd8.weight, dd.weight)
    np.testing.assert_allclose(dd8.meanr, dd.meanr)
    np.testing.assert_allclose(dd8.meanlogr, dd.meanlogr)


@timer
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

    log_min_sep = np.log(min_sep)
    log_max_sep = np.log(max_sep)
    true_npairs = np.zeros(nbins, dtype=int)
    true_weight = np.zeros(nbins, dtype=float)
    bin_size = (log_max_sep - log_min_sep) / nbins

    for i in range(ngal):
        for j in range(ngal):
            rsq = (x1[i]-x2[j])**2 + (y1[i]-y2[j])**2 + (z1[i]-z2[j])**2
            r = np.sqrt(rsq)
            r *= coord.radians / coord.degrees

            index = np.floor(np.log(r/min_sep) / bin_size).astype(int)
            if index < 0 or index >= nbins:
                continue

            ww = w1[i] * w2[j]

            true_npairs[index] += 1
            true_weight[index] += ww

    print('true_npairs = ',true_npairs)
    print('diff = ',dd.npairs - true_npairs)
    np.testing.assert_array_equal(dd.npairs, true_npairs)

    print('true_weight = ',true_weight)
    print('diff = ',dd.weight - true_weight)
    np.testing.assert_allclose(dd.weight, true_weight, rtol=1.e-5, atol=1.e-8)

    # Check that running via the corr2 script works correctly.
    config = treecorr.config.read_config('configs/nn_direct_spherical.yaml')
    try:
        import fitsio
    except ImportError:
        pass
    else:
        cat1.write(config['file_name'])
        cat2.write(config['file_name2'])
        treecorr.corr2(config)
        data = fitsio.read(config['nn_file_name'])
        print(data.dtype)
        np.testing.assert_allclose(data['r_nom'], dd.rnom)
        np.testing.assert_allclose(data['npairs'], dd.npairs)
        np.testing.assert_allclose(data['DD'], dd.weight)

    # Repeat with binslop = 0
    # And don't do any top-level recursion so we actually test not going to the leaves.
    dd = treecorr.NNCorrelation(min_sep=min_sep, max_sep=max_sep, nbins=nbins,
                                sep_units='deg', bin_slop=0, max_top=0)
    dd.process(cat1, cat2)
    np.testing.assert_array_equal(dd.npairs, true_npairs)
    np.testing.assert_allclose(dd.weight, true_weight, rtol=1.e-5, atol=1.e-8)

    # Can't cross correlate different coordinate systems
    cat3 = treecorr.Catalog(x=ra1, y=dec1, x_units='rad', y_units='rad', w=w1)
    cat4 = treecorr.Catalog(ra=ra1, dec=dec1, r=ra1, ra_units='rad', dec_units='rad', w=w1)
    with assert_raises(ValueError):
        dd.process(cat3, cat2)
    with assert_raises(ValueError):
        dd.process(cat1, cat3)
    with assert_raises(ValueError):
        dd.process(cat4, cat2)
    with assert_raises(ValueError):
        dd.process(cat1, cat4)

    # Cannot use some metrics with spherical coordinates
    dd2 = treecorr.NNCorrelation(min_sep=min_sep, max_sep=max_sep, nbins=nbins)
    with assert_raises(ValueError):
        dd2.process(cat1, cat2, metric='Rperp')
    with assert_raises(ValueError):
        dd2.process(cat1, cat2, metric='OldRperp')
    with assert_raises(ValueError):
        dd2.process(cat1, cat2, metric='FisherRperp')
    with assert_raises(ValueError):
        dd2.process(cat1, cat2, metric='Rlens')


@timer
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
    with assert_warns(FutureWarning):
        dd.process_pairwise(cat1, cat2)
    dd.finalize()

    true_npairs = np.zeros(nbins, dtype=int)
    true_weight = np.zeros(nbins, dtype=float)

    rsq = (x1-x2)**2 + (y1-y2)**2
    r = np.sqrt(rsq)

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
        with assert_warns(FutureWarning):
            dd.process_pairwise(cat1, cat2, metric='Euclidean', num_threads=2)
    assert "for cats first, second" in cl.output

    # Can also run this via process if pairwise is set in constructor.
    dd2 = treecorr.NNCorrelation(min_sep=min_sep, max_sep=max_sep, nbins=nbins, pairwise=True)
    with assert_warns(FutureWarning):
        dd2.process(cat1, cat2)
    np.testing.assert_array_equal(dd2.npairs, true_npairs)
    np.testing.assert_allclose(dd2.weight, true_weight, rtol=1.e-5, atol=1.e-8)

    with assert_raises(ValueError):
        with assert_warns(FutureWarning):
            dd2.process(cat1, [cat2, cat2])
    with assert_raises(ValueError):
        with assert_warns(FutureWarning):
            dd2.process([cat1, cat1], cat2)
    cat3 = treecorr.Catalog(x=x2[:500], y=y2[:500], w=w2[:500])
    with assert_raises(ValueError):
        with assert_warns(FutureWarning):
            dd2.process(cat1, cat3)
    with assert_raises(ValueError):
        with assert_warns(FutureWarning):
            dd2.process(cat2, cat3)



@timer
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

    # Can't set sep_units with 3d
    dd2 = treecorr.NNCorrelation(min_sep=min_sep, max_sep=max_sep, nbins=nbins, sep_units='arcmin')
    with assert_raises(ValueError):
        dd2.process(cat1, cat2)

    # Cannot use Rlens metric with auto-correlation
    dd3 = treecorr.NNCorrelation(min_sep=min_sep, max_sep=max_sep, nbins=nbins)
    with assert_raises(ValueError):
        dd3.process(cat1, metric='Rlens')

@timer
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


@timer
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


@timer
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
    xi, varxi = dd.calculateXi(rr=rr, dr=dr, rd=rd)

    # After calling calculateXi, you can access the result via attributes
    np.testing.assert_array_equal(xi, dd.xi)
    np.testing.assert_array_equal(varxi, dd.varxi)
    np.testing.assert_array_equal(varxi, dd.cov.diagonal())

    with assert_raises(TypeError):
        dd.calculateXi(dr=dr)
    with assert_raises(TypeError):
        dd.calculateXi(rd=rd)
    with assert_raises(TypeError):
        dd.calculateXi(dr=dr, rd=rd)
    rr2 = treecorr.NNCorrelation(min_sep=min_sep, max_sep=max_sep, nbins=nbins, bin_type='Linear')
    with assert_raises(ValueError):
        dd.calculateXi(rr=rr2, dr=dr, rd=rd)
    with assert_raises(ValueError):
        dd.calculateXi(rr=rr, dr=rr2, rd=rd)
    with assert_raises(ValueError):
        dd.calculateXi(rr=rr, dr=dr, rd=rr2)
    with assert_raises(ValueError):
        dd.calculateXi(rr=rr, rd=rr2)
    with assert_raises(ValueError):
        dd.calculateXi(rr=rr, dr=rr2)

    config = treecorr.config.read_config('configs/nn_linear.yaml')
    logger = treecorr.config.setup_logger(0)
    treecorr.corr2(config, logger)
    corr2_output = np.genfromtxt(os.path.join('output','nn_linear.out'), names=True,
                                 skip_header=1)
    print('corr2_output = ',corr2_output)
    print('corr2_output.dtype = ',corr2_output.dtype)
    print('rnom = ',dd.rnom)
    print('       ',corr2_output['r_nom'])
    np.testing.assert_allclose(corr2_output['r_nom'], dd.rnom, rtol=1.e-3)
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



@timer
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

    # Using nbins=None rather than omitting nbins is equivalent.
    dd2 = treecorr.NNCorrelation(bin_size=0.1, min_sep=1., max_sep=25., nbins=None, sep_units='arcmin')
    dd2.process(cat, num_threads=1)
    dd.process(cat, num_threads=1)
    assert dd2 == dd

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

    xi, varxi = dd.calculateXi(rr=rr,dr=dr)
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

    simple_xi, simple_varxi = dd.calculateXi(rr=rr)
    print('simple xi = ',simple_xi)
    print('max rel diff = ',max(abs((simple_xi - true_xi)/true_xi)))
    # The simple calculation (i.e. dd/rr-1, rather than (dd-2dr+rr)/rr as above) is only
    # slightly less accurate in this case.  Probably because the mask is simple (a box), so
    # the difference is relatively minor.  The error is slightly higher in this case, but testing
    # that it is everywhere < 0.1 is still appropriate.
    np.testing.assert_allclose(simple_xi, true_xi, rtol=0.1*tol_factor)

    # This is one of the few tests we have where it doesn't by default max out the top layers.
    # So check that min_top actually does something.
    print('d top = ',cat.field.nTopLevelNodes)
    print('r top = ',rand.field.nTopLevelNodes)
    # For nosetests, this is 390, for main it is 402.  So just check a range.
    assert 380 < cat.field.nTopLevelNodes < 410
    assert rand.field.nTopLevelNodes == 1024
    dd2 = treecorr.NNCorrelation(bin_size=0.1, min_sep=1., max_sep=25., sep_units='arcmin',
                                 min_top=9)
    dd2.process(cat)
    print('d top = ',cat.field.nTopLevelNodes)
    assert 600 < cat.field.nTopLevelNodes < 610
    dd3 = treecorr.NNCorrelation(bin_size=0.1, min_sep=1., max_sep=25., sep_units='arcmin',
                                 min_top=10)
    dd3.process(cat)
    print('d top = ',cat.field.nTopLevelNodes)
    assert cat.field.nTopLevelNodes == 1024

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
    dd.calculateXi(rr=rr, dr=dr)  # reset this to the better calculation
    dd2 = treecorr.NNCorrelation(bin_size=0.1, min_sep=1., max_sep=25., sep_units='arcmin')
    dd2.read(out_file_name)
    np.testing.assert_allclose(dd2.logr, dd.logr, rtol=1.e-3)
    np.testing.assert_allclose(dd2.meanr, dd.meanr, rtol=1.e-3)
    np.testing.assert_allclose(dd2.meanlogr, dd.meanlogr, rtol=1.e-3)
    np.testing.assert_allclose(dd2.npairs, dd.npairs, rtol=1.e-3)
    np.testing.assert_allclose(dd2.tot, dd.tot, rtol=1.e-3)
    np.testing.assert_allclose(dd2.xi, dd.xi, rtol=1.e-3)
    np.testing.assert_allclose(dd2.varxi, dd.varxi, rtol=1.e-3)
    assert dd2.coords == dd.coords
    assert dd2.metric == dd.metric
    assert dd2.sep_units == dd.sep_units
    assert dd2.bin_type == dd.bin_type

    # Check the fits write option
    try:
        import fitsio
    except ImportError:
        pass
    else:
        out_file_name1 = os.path.join('output','nn_out1.fits')
        dd.write(out_file_name1)
        data = fitsio.read(out_file_name1)
        np.testing.assert_almost_equal(data['r_nom'], np.exp(dd.logr))
        np.testing.assert_almost_equal(data['meanr'], dd.meanr)
        np.testing.assert_almost_equal(data['meanlogr'], dd.meanlogr)
        np.testing.assert_almost_equal(data['npairs'], dd.npairs)
        header = fitsio.read_header(out_file_name1, 1)
        np.testing.assert_almost_equal(header['tot'], dd.tot)

        out_file_name2 = os.path.join('output','nn_out2.fits')
        dd.write(out_file_name2, rr=rr)
        data = fitsio.read(out_file_name2)
        np.testing.assert_almost_equal(data['r_nom'], np.exp(dd.logr))
        np.testing.assert_almost_equal(data['meanr'], dd.meanr)
        np.testing.assert_almost_equal(data['meanlogr'], dd.meanlogr)
        np.testing.assert_almost_equal(data['xi'], simple_xi)
        np.testing.assert_almost_equal(data['sigma_xi'], np.sqrt(simple_varxi))
        np.testing.assert_almost_equal(data['DD'], dd.npairs)
        np.testing.assert_almost_equal(data['RR'], rr.npairs * (dd.tot / rr.tot))
        header = fitsio.read_header(out_file_name2, 1)
        np.testing.assert_almost_equal(header['tot'], dd.tot)

        out_file_name3 = os.path.join('output','nn_out3.fits')
        dd.write(out_file_name3, rr=rr, dr=dr)
        data = fitsio.read(out_file_name3)
        np.testing.assert_almost_equal(data['r_nom'], np.exp(dd.logr))
        np.testing.assert_almost_equal(data['meanr'], dd.meanr)
        np.testing.assert_almost_equal(data['meanlogr'], dd.meanlogr)
        np.testing.assert_almost_equal(data['xi'], xi)
        np.testing.assert_almost_equal(data['sigma_xi'], np.sqrt(varxi))
        np.testing.assert_almost_equal(data['DD'], dd.npairs)
        np.testing.assert_almost_equal(data['RR'], rr.npairs * (dd.tot / rr.tot))
        np.testing.assert_almost_equal(data['DR'], dr.npairs * (dd.tot / dr.tot))
        header = fitsio.read_header(out_file_name3, 1)
        np.testing.assert_almost_equal(header['tot'], dd.tot)

        out_file_name4 = os.path.join('output','nn_out4.fits')
        dd.write(out_file_name4)  # Without rr, then xi and varxi are still written, but not RR, DR
        data = fitsio.read(out_file_name4)
        np.testing.assert_almost_equal(data['r_nom'], np.exp(dd.logr))
        np.testing.assert_almost_equal(data['meanr'], dd.meanr)
        np.testing.assert_almost_equal(data['meanlogr'], dd.meanlogr)
        np.testing.assert_almost_equal(data['xi'], xi)
        np.testing.assert_almost_equal(data['sigma_xi'], np.sqrt(varxi))
        np.testing.assert_almost_equal(data['DD'], dd.npairs)
        assert 'RR' not in data.dtype.names
        assert 'DR' not in data.dtype.names
        header = fitsio.read_header(out_file_name4, 1)
        np.testing.assert_almost_equal(header['tot'], dd.tot)

        out_file_name5 = os.path.join('output','nn_out5.fits')
        del dd.xi  # Equivalent to not having called calculateXi
        del dd.varxi
        dd.write(out_file_name5)
        data = fitsio.read(out_file_name5)
        np.testing.assert_almost_equal(data['r_nom'], np.exp(dd.logr))
        np.testing.assert_almost_equal(data['meanr'], dd.meanr)
        np.testing.assert_almost_equal(data['meanlogr'], dd.meanlogr)
        np.testing.assert_almost_equal(data['DD'], dd.npairs)
        assert 'xi' not in data.dtype.names
        assert 'varxi' not in data.dtype.names
        assert 'RR' not in data.dtype.names
        assert 'DR' not in data.dtype.names
        header = fitsio.read_header(out_file_name5, 1)
        np.testing.assert_almost_equal(header['tot'], dd.tot)

        # Check the read function
        dd.calculateXi(rr=rr, dr=dr)  # gets xi, varxi back in dd
        dd2 = treecorr.NNCorrelation(bin_size=0.1, min_sep=1., max_sep=25., sep_units='arcmin')
        dd2.read(out_file_name1)
        np.testing.assert_almost_equal(dd2.logr, dd.logr)
        np.testing.assert_almost_equal(dd2.meanr, dd.meanr)
        np.testing.assert_almost_equal(dd2.meanlogr, dd.meanlogr)
        np.testing.assert_almost_equal(dd2.npairs, dd.npairs)
        np.testing.assert_almost_equal(dd2.tot, dd.tot)
        np.testing.assert_almost_equal(dd2.xi, dd.xi)
        np.testing.assert_almost_equal(dd2.varxi, dd.varxi)
        assert dd2.coords == dd.coords
        assert dd2.metric == dd.metric
        assert dd2.sep_units == dd.sep_units
        assert dd2.bin_type == dd.bin_type

        dd3 = treecorr.NNCorrelation(bin_size=0.1, min_sep=1., max_sep=25., sep_units='arcmin')
        dd3.read(out_file_name3)
        np.testing.assert_almost_equal(dd3.logr, dd.logr)
        np.testing.assert_almost_equal(dd3.meanr, dd.meanr)
        np.testing.assert_almost_equal(dd3.meanlogr, dd.meanlogr)
        np.testing.assert_almost_equal(dd3.npairs, dd.npairs)
        np.testing.assert_almost_equal(dd3.tot, dd.tot)
        np.testing.assert_almost_equal(dd3.xi, dd.xi)
        np.testing.assert_almost_equal(dd3.varxi, dd.varxi)
        assert dd3.coords == dd.coords
        assert dd3.metric == dd.metric
        assert dd3.sep_units == dd.sep_units
        assert dd3.bin_type == dd.bin_type

        dd4 = treecorr.NNCorrelation(bin_size=0.1, min_sep=1., max_sep=25., sep_units='arcmin')
        dd4.read(out_file_name5)
        np.testing.assert_almost_equal(dd4.logr, dd.logr)
        np.testing.assert_almost_equal(dd4.meanr, dd.meanr)
        np.testing.assert_almost_equal(dd4.meanlogr, dd.meanlogr)
        np.testing.assert_almost_equal(dd4.npairs, dd.npairs)
        np.testing.assert_almost_equal(dd4.tot, dd.tot)
        assert not hasattr(dd4,'xi')
        assert not hasattr(dd4,'varxi')
        assert dd4.coords == dd.coords
        assert dd4.metric == dd.metric
        assert dd4.sep_units == dd.sep_units
        assert dd4.bin_type == dd.bin_type

    # Check the hdf5 write option
    try:
        import h5py  # noqa: F401
    except ImportError:
        pass
    else:
        out_file_name6 = os.path.join('output','nn_out5.hdf5')
        dd.write(out_file_name6)
        with h5py.File(out_file_name6, 'r') as hdf:
            data = hdf['/']
            np.testing.assert_almost_equal(data['r_nom'], np.exp(dd.logr))
            np.testing.assert_almost_equal(data['meanr'], dd.meanr)
            np.testing.assert_almost_equal(data['meanlogr'], dd.meanlogr)
            np.testing.assert_almost_equal(data['npairs'], dd.npairs)
            np.testing.assert_almost_equal(data['xi'], xi)
            np.testing.assert_almost_equal(data['sigma_xi'], np.sqrt(varxi))
            np.testing.assert_almost_equal(data['DD'], dd.npairs)
            attrs = data.attrs
            np.testing.assert_almost_equal(attrs['tot'], dd.tot)
            assert 'RR' not in data
            assert 'DR' not in data

        dd6 = treecorr.NNCorrelation(bin_size=0.1, min_sep=1., max_sep=25., sep_units='arcmin')
        dd6.read(out_file_name6)
        np.testing.assert_almost_equal(dd6.logr, dd.logr)
        np.testing.assert_almost_equal(dd6.meanr, dd.meanr)
        np.testing.assert_almost_equal(dd6.meanlogr, dd.meanlogr)
        np.testing.assert_almost_equal(dd6.npairs, dd.npairs)
        np.testing.assert_almost_equal(dd6.tot, dd.tot)
        np.testing.assert_almost_equal(dd6.xi, dd.xi)
        np.testing.assert_almost_equal(dd6.varxi, dd.varxi)
        assert dd6.coords == dd.coords
        assert dd6.metric == dd.metric
        assert dd6.sep_units == dd.sep_units
        assert dd6.bin_type == dd.bin_type

    # Cannot omit rr if giving either dr or rd
    with assert_raises(TypeError):
        dd.write(out_file_name, dr=dr)
    with assert_raises(TypeError):
        dd.write(out_file_name, rd=dr)


@timer
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

    simple_xi, varxi = dd.calculateXi(rr=rr)
    print('simple xi = ',simple_xi)
    print('true_xi = ',true_xi)
    print('max rel diff = ',max(abs((simple_xi - true_xi)/true_xi)))
    np.testing.assert_allclose(simple_xi, true_xi, rtol=0.1*tol_factor)
    np.testing.assert_allclose(np.log(np.abs(simple_xi)), np.log(np.abs(true_xi)),
                               rtol=0.1*tol_factor)

    xi, varxi = dd.calculateXi(rr=rr, dr=dr)
    print('xi = ',xi)
    print('true_xi = ',true_xi)
    print('ratio = ',xi / true_xi)
    print('diff = ',xi - true_xi)
    print('max rel diff = ',max(abs((xi - true_xi)/true_xi)))
    np.testing.assert_allclose(xi, true_xi, rtol=0.1*tol_factor)
    np.testing.assert_allclose(np.log(np.abs(xi)), np.log(np.abs(true_xi)),
                               rtol=0.1*tol_factor)

    # Check that we get the same result using the corr2 function:
    config = treecorr.config.read_config('configs/nn_3d.yaml')
    try:
        import fitsio
    except ImportError:
        pass
    else:
        cat.write(os.path.join('data','nn_3d_data.dat'))
        rand.write(os.path.join('data','nn_3d_rand.dat'))
        config['verbose'] = 0
        treecorr.corr2(config)
        corr2_outfile = os.path.join('output','nn_3d.fits')
        corr2_output = fitsio.read(corr2_outfile)
        print('xi = ',xi)
        print('from corr2 output = ',corr2_output['xi'])
        print('ratio = ',corr2_output['xi']/xi)
        print('diff = ',corr2_output['xi']-xi)

        np.testing.assert_almost_equal(corr2_output['r_nom'], np.exp(dd.logr))
        np.testing.assert_almost_equal(corr2_output['meanr'], dd.meanr)
        np.testing.assert_almost_equal(corr2_output['meanlogr'], dd.meanlogr)
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
    xi, varxi = dd.calculateXi(rr=rr, dr=dr)
    np.testing.assert_allclose(xi, true_xi, rtol=0.1*tol_factor)
    np.testing.assert_allclose(np.log(np.abs(xi)), np.log(np.abs(true_xi)),
                               rtol=0.1*tol_factor)


@timer
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

    dd = treecorr.NNCorrelation(bin_size=0.1, min_sep=1., max_sep=25., verbose=1)
    dd.process(data_cats)
    print('dd.npairs = ',dd.npairs)

    rr = treecorr.NNCorrelation(bin_size=0.1, min_sep=1., max_sep=25., verbose=1)
    rr.process(rand_cats)
    print('rr.npairs = ',rr.npairs)

    xi, varxi = dd.calculateXi(rr=rr)
    print('xi = ',xi)

    # Now do the same thing with one big catalog for each.
    ddx = treecorr.NNCorrelation(bin_size=0.1, min_sep=1., max_sep=25., verbose=1)
    rrx = treecorr.NNCorrelation(bin_size=0.1, min_sep=1., max_sep=25., verbose=1)
    data_catx = treecorr.Catalog(x=x.reshape( (nobj*ncats,) ), y=y.reshape( (nobj*ncats,) ))
    rand_catx = treecorr.Catalog(x=rx.reshape( (nobj*ncats,) ), y=ry.reshape( (nobj*ncats,) ))
    ddx.process(data_catx)
    rrx.process(rand_catx)
    xix, varxix = ddx.calculateXi(rr=rrx)

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

    # First do this via the corr2 function.
    config = treecorr.config.read_config('configs/nn_list1.yaml')
    logger = treecorr.config.setup_logger(0)
    treecorr.corr2(config, logger)
    corr2_output = np.genfromtxt(os.path.join('output','nn_list1.out'),names=True,skip_header=1)
    print('xi = ',xi)
    print('from corr2 output = ',corr2_output['xi'])
    print('ratio = ',corr2_output['xi']/xi)
    print('diff = ',corr2_output['xi']-xi)
    np.testing.assert_allclose(corr2_output['xi'], xi, rtol=1.e-3)

    # Now calling out to the external corr2 executable to test it with extra command-line params
    if os.name != 'nt':
        import subprocess
        corr2_exe = get_script_name('corr2')
        p = subprocess.Popen( [corr2_exe,"configs/nn_list1.yaml","verbose=0"] )
        p.communicate()
        corr2_output = np.genfromtxt(os.path.join('output','nn_list1.out'),names=True,skip_header=1)
        print('xi = ',xi)
        print('from corr2 output = ',corr2_output['xi'])
        print('ratio = ',corr2_output['xi']/xi)
        print('diff = ',corr2_output['xi']-xi)
        np.testing.assert_allclose(corr2_output['xi'], xi, rtol=1.e-3)

    config = treecorr.config.read_config('configs/nn_list2.json')
    treecorr.config.parse_variable(config, 'verbose=0')
    treecorr.corr2(config)
    corr2_output = np.genfromtxt(os.path.join('output','nn_list2.out'),names=True,skip_header=1)
    print('xi = ',xi)
    print('from corr2 output = ',corr2_output['xi'])
    print('ratio = ',corr2_output['xi']/xi)
    print('diff = ',corr2_output['xi']-xi)
    np.testing.assert_allclose(corr2_output['xi'], xi, rtol=2.e-2)

    config = treecorr.config.read_config('configs/nn_list3.params')
    treecorr.config.parse_variable(config, 'verbose=0')
    treecorr.corr2(config)
    corr2_output = np.genfromtxt(os.path.join('output','nn_list3.out'),names=True,skip_header=1)
    print('xi = ',xi)
    print('from corr2 output = ',corr2_output['xi'])
    print('ratio = ',corr2_output['xi']/xi)
    print('diff = ',corr2_output['xi']-xi)
    np.testing.assert_allclose(corr2_output['xi'], xi, rtol=2.e-2)

    config = treecorr.config.read_config('configs/nn_list4.config', file_type='yaml')
    treecorr.config.parse_variable(config, 'verbose=0')
    treecorr.corr2(config)
    corr2_output = np.genfromtxt(os.path.join('output','nn_list4.out'),names=True,skip_header=1)
    print('xi = ',xi)
    print('from corr2 output = ',corr2_output['xi'])
    print('ratio = ',corr2_output['xi']/xi)
    print('diff = ',corr2_output['xi']-xi)
    np.testing.assert_allclose(corr2_output['xi'], xi, rtol=1.e-2)

    # Repeat with exe to test -f flag
    if os.name != 'nt':
        p = subprocess.Popen([corr2_exe, "-f", "yaml", "configs/nn_list4.config", "verbose=3"],
                             stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        p.communicate()
        corr2_output = np.genfromtxt(os.path.join('output','nn_list4.out'),names=True,skip_header=1)
        print('xi = ',xi)
        print('from corr2 output = ',corr2_output['xi'])
        print('ratio = ',corr2_output['xi']/xi)
        print('diff = ',corr2_output['xi']-xi)
        np.testing.assert_allclose(corr2_output['xi'], xi, rtol=1.e-2)

    config = treecorr.config.read_config('configs/nn_list5.config', file_type='json')
    treecorr.config.parse_variable(config, 'verbose=0')
    treecorr.corr2(config)
    corr2_output = np.genfromtxt(os.path.join('output','nn_list5.out'),names=True,skip_header=1)
    print('xi = ',xi)
    print('from corr2 output = ',corr2_output['xi'])
    print('ratio = ',corr2_output['xi']/xi)
    print('diff = ',corr2_output['xi']-xi)
    np.testing.assert_allclose(corr2_output['xi'], xi, rtol=1.e-2)

    # For this one, the output file is in the current directory, which used to give an error.
    config = treecorr.config.read_config('configs/nn_list6.config', file_type='params')
    treecorr.config.parse_variable(config, 'verbose=0')
    treecorr.corr2(config)
    output_file = 'nn_list6.out'
    corr2_output = np.genfromtxt(output_file,names=True,skip_header=1)
    print('xi = ',xi)
    print('from corr2 output = ',corr2_output['xi'])
    print('ratio = ',corr2_output['xi']/xi)
    print('diff = ',corr2_output['xi']-xi)
    np.testing.assert_allclose(corr2_output['xi'], xi, rtol=1.e-2)
    # Move it to the output directory now to keep the current directory clean.
    mv_output_file = os.path.join('output',output_file)
    if os.path.exists(mv_output_file):
        os.remove(mv_output_file)
    os.rename(output_file, mv_output_file)

@timer
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

    dd_random1 = treecorr.NNCorrelation(bin_size=0.1, min_sep=5., max_sep=25.,
                                        split_method='random')
    t0 = time.time()
    dd_random1.process(cat)
    t1 = time.time()
    print('random1: time = ',t1-t0)
    print('npairs = ',dd_random1.npairs)

    # Random should be non-deterministic, so check a second version of it.
    # Need to clear the cache to get it to rebuild though.
    cat.nfields.clear()
    dd_random2 = treecorr.NNCorrelation(bin_size=0.1, min_sep=5., max_sep=25.,
                                        split_method='random')
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

    # If cat has an rng though, they should be identical.
    # (Also need single thread, else the random values come in a different order still.)
    cat.nfields.clear()
    cat._rng = np.random.RandomState(1234)
    dd_random1.process(cat, num_threads=1)
    cat.nfields.clear()
    cat._rng = np.random.RandomState(1234)
    dd_random2.process(cat, num_threads=1)
    np.testing.assert_array_equal(dd_random1.npairs, dd_random2.npairs)

    assert_raises(ValueError, treecorr.NNCorrelation, bin_size=0.1, min_sep=5., max_sep=25.,
                  split_method='invalid')


@timer
def test_varxi():
    # Test that varxi is correct (or close) based on actual variance of many runs.

    L = 100
    rng = np.random.RandomState(8675309)

    ngal = 50
    nrand = 200
    nruns = 50000

    file_name = 'data/test_varxi_nn.npz'
    print(file_name)
    if not os.path.isfile(file_name):
        all_dds = []
        all_drs = []
        all_rrs = []
        for run in range(nruns):
            print(f'{run}/{nruns}')
            x1 = (rng.random_sample(ngal)-0.5) * L
            y1 = (rng.random_sample(ngal)-0.5) * L
            x2 = (rng.random_sample(nrand)-0.5) * L
            y2 = (rng.random_sample(nrand)-0.5) * L
            # Varied weights are hard, but at least check that non-unit weights work correctly.
            w = np.ones_like(x1) * 5
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

        all_xis = [dd.calculateXi(rr=rr) for dd,rr in zip(all_dds, all_rrs)]
        var_xi_1 = np.var([xi[0] for xi in all_xis], axis=0)
        mean_varxi_1 = np.mean([xi[1] for xi in all_xis], axis=0)

        all_xis = [dd.calculateXi(rr=rr, dr=dr) for dd,dr,rr in zip(all_dds, all_drs, all_rrs)]
        var_xi_2 = np.var([xi[0] for xi in all_xis], axis=0)
        mean_varxi_2 = np.mean([xi[1] for xi in all_xis], axis=0)

        all_xis = [dd.calculateXi(rr=rr, dr=dr, rd=dr) for dd,dr,rr in zip(all_dds, all_drs, all_rrs)]
        var_xi_3 = np.var([xi[0] for xi in all_xis], axis=0)
        mean_varxi_3 = np.mean([xi[1] for xi in all_xis], axis=0)

        all_xis = [dd.calculateXi(rr=rr, rd=dr) for dd,dr,rr in zip(all_dds, all_drs, all_rrs)]
        var_xi_4 = np.var([xi[0] for xi in all_xis], axis=0)
        mean_varxi_4 = np.mean([xi[1] for xi in all_xis], axis=0)

        np.savez(file_name,
                 var_xi_1=var_xi_1, mean_varxi_1=mean_varxi_1,
                 var_xi_2=var_xi_2, mean_varxi_2=mean_varxi_2,
                 var_xi_3=var_xi_3, mean_varxi_3=mean_varxi_3,
                 var_xi_4=var_xi_4, mean_varxi_4=mean_varxi_4)

    data = np.load(file_name)

    print('nruns = ',nruns)
    print('Uncompensated:')
    var_xi_1 = data['var_xi_1']
    mean_varxi_1 = data['mean_varxi_1']
    print('mean_varxi = ',mean_varxi_1)
    print('var_xi = ',var_xi_1)
    print('ratio = ',var_xi_1 / mean_varxi_1)
    print('max relerr for xi = ',np.max(np.abs((var_xi_1 - mean_varxi_1)/var_xi_1)))
    print('diff = ',var_xi_1 - mean_varxi_1)
    np.testing.assert_allclose(mean_varxi_1, var_xi_1, rtol=0.1)

    print('Compensated:')
    var_xi_2 = data['var_xi_2']
    mean_varxi_2 = data['mean_varxi_2']
    print('mean_varxi = ',mean_varxi_2)
    print('var_xi = ',var_xi_2)
    print('ratio = ',var_xi_2 / mean_varxi_2)
    print('max relerr for xi = ',np.max(np.abs((var_xi_2 - mean_varxi_2)/var_xi_2)))
    print('diff = ',var_xi_2 - mean_varxi_2)
    np.testing.assert_allclose(mean_varxi_2, var_xi_2, rtol=0.05)

    print('Compensated with both dr and rd:')
    var_xi_3 = data['var_xi_3']
    mean_varxi_3 = data['mean_varxi_3']
    print('mean_varxi = ',mean_varxi_3)
    print('var_xi = ',var_xi_3)
    print('ratio = ',var_xi_3 / mean_varxi_3)
    print('max relerr for xi = ',np.max(np.abs((var_xi_3 - mean_varxi_3)/var_xi_3)))
    print('diff = ',var_xi_3 - mean_varxi_3)
    np.testing.assert_allclose(mean_varxi_3, var_xi_3, rtol=0.05)

    print('Compensated with just rd')
    var_xi_4 = data['var_xi_4']
    mean_varxi_4 = data['mean_varxi_4']
    print('mean_varxi = ',mean_varxi_4)
    print('var_xi = ',var_xi_4)
    print('ratio = ',var_xi_4 / mean_varxi_4)
    print('max relerr for xi = ',np.max(np.abs((var_xi_4 - mean_varxi_4)/var_xi_4)))
    print('diff = ',var_xi_4 - mean_varxi_4)
    np.testing.assert_allclose(mean_varxi_4, var_xi_4, rtol=0.05)

    # Now the actual test that's based on current code, not just from the saved file.
    # There is a bit more noise on a singe run, so the tolerance needs to be somewhat higher.
    x1 = (rng.random_sample(ngal)-0.5) * L
    y1 = (rng.random_sample(ngal)-0.5) * L
    x2 = (rng.random_sample(nrand)-0.5) * L
    y2 = (rng.random_sample(nrand)-0.5) * L
    # Varied weights are hard, but at least check that non-unit weights work correctly.
    w = np.ones_like(x1) * 5
    wr = np.ones_like(x2) * 0.3

    data = treecorr.Catalog(x=x1, y=y1, w=w)
    rand = treecorr.Catalog(x=x2, y=y2, w=wr)
    dd = treecorr.NNCorrelation(bin_size=0.1, min_sep=6., max_sep=13.)
    dr = treecorr.NNCorrelation(bin_size=0.1, min_sep=6., max_sep=13.)
    rr = treecorr.NNCorrelation(bin_size=0.1, min_sep=6., max_sep=13.)
    dd.process(data)
    dr.process(data, rand)
    rr.process(rand)

    print('single run')
    print('Uncompensated')
    xi, varxi = dd.calculateXi(rr=rr)
    print('ratio = ',varxi / var_xi_1)
    print('max relerr for xi = ',np.max(np.abs((varxi - var_xi_1)/var_xi_1)))
    np.testing.assert_allclose(varxi, var_xi_1, rtol=0.25)

    print('Compensated')
    xi, varxi = dd.calculateXi(rr=rr, dr=dr)
    print('ratio = ',varxi / var_xi_2)
    print('max relerr for xi = ',np.max(np.abs((varxi - var_xi_2)/var_xi_2)))
    np.testing.assert_allclose(varxi, var_xi_2, rtol=0.25)

    print('Compensated with both dr and rd:')
    xi, varxi = dd.calculateXi(rr=rr, dr=dr, rd=dr)
    print('ratio = ',varxi / var_xi_3)
    print('max relerr for xi = ',np.max(np.abs((varxi - var_xi_3)/var_xi_3)))
    np.testing.assert_allclose(varxi, var_xi_3, rtol=0.25)

    print('Compensated with just rd:')
    xi, varxi = dd.calculateXi(rr=rr, rd=dr)
    print('ratio = ',varxi / var_xi_4)
    print('max relerr for xi = ',np.max(np.abs((varxi - var_xi_4)/var_xi_4)))
    np.testing.assert_allclose(varxi, var_xi_4, rtol=0.25)



@timer
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


@timer
def test_linear_binslop():

    # Jack Elvin-Poole reported a problem with version 4.0.1 using linear binning
    # where the lowest bin was getting too many pairs by more than what should be allowed
    # by the non-zero binslop.  His test used DES redmagic data, but this reproduces the
    # error that he was seeing.

    rng = np.random.RandomState(8675309)
    ngal = 10000
    x = rng.normal(10, 0.1, (ngal,) )
    y = rng.normal(30, 0.1, (ngal,) )
    z = rng.normal(20, 0.1, (ngal,) )
    r = np.sqrt(x*x+y*y+z*z)
    dec = np.arcsin(z/r) * coord.radians / coord.degrees
    ra = np.arctan2(y,x) * coord.radians / coord.degrees

    cat = treecorr.Catalog(ra=ra, dec=dec, ra_units='deg', dec_units='deg')

    for bin_type in ['Log', 'Linear']:
        dd0 = treecorr.NNCorrelation(min_sep=0.5, max_sep=1, nbins=3, bin_slop=0.,
                                    bin_type=bin_type, sep_units='deg')
        dd0.process(cat, num_threads=1)
        print('dd0.rnom = ',dd0.rnom)
        print('dd0.meanr = ',dd0.meanr)
        print('dd0.npairs = ',dd0.npairs)

        for bin_slop in [1.e-3, 1.e-2, 1.e-1]:
            dd1 = treecorr.NNCorrelation(min_sep=0.5, max_sep=1, nbins=3, bin_slop=bin_slop,
                                        bin_type=bin_type, sep_units='deg')
            dd1.process(cat, num_threads=1)

            print(bin_type, 'bs = ',bin_slop)
            print('dd1.rnom = ',dd1.rnom)
            print('dd1.meanr = ',dd1.meanr)
            print('dd1.npairs = ',dd1.npairs)
            # The difference between the two should be less than bin_slop for all bins.
            print('rel diff = ',(dd1.npairs - dd0.npairs)/dd0.npairs)
            np.testing.assert_allclose(dd1.npairs, dd0.npairs, rtol=bin_slop)


if __name__ == '__main__':
    test_log_binning()
    test_linear_binning()
    test_direct_count()
    test_direct_spherical()
    test_pairwise()
    test_direct_3d()
    test_direct_arc()
    test_direct_partial()
    test_direct_linear()
    test_nn()
    test_3d()
    test_list()
    test_split()
    test_varxi()
    test_sph_linear()
    test_linear_binslop()
