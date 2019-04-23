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

    # Should all have similar number of points.  Say within factor of 2 of the average
    ave_num = cat.ntot / npatch
    print('ave_num = ',ave_num)
    for i in range(npatch):
        print('count for i=%d = %d'%(i,np.sum(patches==i)))
        assert ave_num * 0.5 < np.sum(patches == i) < ave_num * 2.

if __name__ == '__main__':
    test_dessv()
