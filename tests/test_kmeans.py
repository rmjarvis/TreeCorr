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
import time
import coord
import warnings
import treecorr

from test_helper import get_from_wiki, CaptureLog, assert_raises, do_pickle, profile

def test_dessv():
    try:
        import fitsio
    except ImportError:
        print('Skipping dessv test, since fitsio is not installed')
        return

    get_from_wiki('des_sv.fits')
    file_name = os.path.join('data','des_sv.fits')
    with profile():
        cat = treecorr.Catalog(file_name, ra_col='ra', dec_col='dec', ra_units='deg', dec_units='deg')

        npatch = 40
        field = cat.getNField()
        patches = field.run_kmeans(npatch)
    assert len(patches) == cat.ntot
    assert min(patches) == 0
    assert max(patches) == npatch-1

    # KMeans minimizes the total inertia.
    # Check this value and the rms inertia, which should also be quite small.
    xyz = np.array([cat.x, cat.y, cat.z]).T
    cen = np.array([xyz[patches==i].mean(axis=0) for i in range(npatch)])
    inertia = np.sum((xyz - cen[patches])**2)
    patch_inertia = np.array([np.sum((xyz[patches==i] - cen[i])**2) for i in range(npatch)])
    counts = np.array([np.sum(patches==i) for i in range(npatch)])

    print('mean inertia = ',np.mean(patch_inertia))
    print('rms inertia = ',np.std(patch_inertia))
    assert np.sum(inertia) < 215.  # This is specific to this particular field.
    assert np.std(inertia) < 0.2 * np.mean(inertia)  # rms is usually small relative to mean.

    # Should all have similar number of points.  Say within 30% of the average
    print('mean counts = ',np.mean(counts))
    print('min counts = ',np.min(counts))
    print('max counts = ',np.max(counts))
    ave_num = cat.ntot / npatch
    assert np.isclose(np.mean(counts), ave_num)
    assert np.min(counts) > ave_num * 0.7
    assert np.max(counts) < ave_num * 1.3


if __name__ == '__main__':
    test_dessv()
