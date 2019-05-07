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

    #treecorr.set_omp_threads(1);
    get_from_wiki('des_sv.fits')
    file_name = os.path.join('data','des_sv.fits')
    cat = treecorr.Catalog(file_name, ra_col='ra', dec_col='dec', ra_units='deg', dec_units='deg')

    npatch = 40
    field = cat.getNField()
    t0 = time.time()
    patches = field.run_kmeans(npatch)
    t1 = time.time()
    print('patches = ',np.unique(patches))
    assert len(patches) == cat.ntot
    assert min(patches) == 0
    assert max(patches) == npatch-1

    # KMeans minimizes the total inertia.
    # Check this value and the rms size, which should also be quite small.
    xyz = np.array([cat.x, cat.y, cat.z]).T
    cen = np.array([xyz[patches==i].mean(axis=0) for i in range(npatch)])
    inertia = np.sum((xyz - cen[patches])**2)
    sizes = np.array([np.mean((xyz[patches==i] - cen[i])**2) for i in range(npatch)])
    counts = np.array([np.sum(patches==i) for i in range(npatch)])

    print('With standard algorithm:')
    print('time = ',t1-t0)
    print('inertia = ',inertia)
    print('sizes = ',sizes)
    print('mean size= ',np.mean(sizes))
    print('rms size = ',np.std(sizes))
    assert inertia < 215.  # This is specific to this particular field.
    assert np.std(sizes) < 0.2 * np.mean(sizes)  # rms is usually small relative to mean.

    # Should all have similar number of points.  Say within 30% of the average
    print('mean counts = ',np.mean(counts))
    print('min counts = ',np.min(counts))
    print('max counts = ',np.max(counts))
    ave_num = cat.ntot / npatch
    assert np.isclose(np.mean(counts), ave_num)
    assert np.min(counts) > ave_num * 0.7
    assert np.max(counts) < ave_num * 1.3

    # Check the alternate algorithm.  rms size should be lower.
    t0 = time.time()
    patches = field.run_kmeans(npatch, alt=True)
    t1 = time.time()
    assert len(patches) == cat.ntot
    assert min(patches) == 0
    assert max(patches) == npatch-1

    cen = np.array([xyz[patches==i].mean(axis=0) for i in range(npatch)])
    inertia = np.sum((xyz - cen[patches])**2)
    sizes = np.array([np.mean((xyz[patches==i] - cen[i])**2) for i in range(npatch)])
    counts = np.array([np.sum(patches==i) for i in range(npatch)])

    print('With alternate algorithm:')
    print('time = ',t1-t0)
    print('inertia = ',inertia)
    print('sizes = ',sizes)
    print('mean size= ',np.mean(sizes))
    print('rms size = ',np.std(sizes))
    assert inertia < 215.  # Total shouldn't increase much.
    assert np.std(sizes) < 0.1 * np.mean(sizes)  # rms should be even smaller here.

    # This doesn't keep the counts as equal as the standard algorithm.
    print('mean counts = ',np.mean(counts))
    print('min counts = ',np.min(counts))
    print('max counts = ',np.max(counts))
    ave_num = cat.ntot / npatch
    assert np.isclose(np.mean(counts), ave_num)
    assert np.min(counts) > ave_num * 0.5
    assert np.max(counts) < ave_num * 1.5




if __name__ == '__main__':
    test_dessv()
