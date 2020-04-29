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
import time
import os
import unittest
import pytest
import treecorr

try:
    import fitsio
    from mpi4py import MPI
    no_mpi = False
except ImportError:
    print('Skipping mpi tests, since either mpi4py or fitsio is not installed')
    no_mpi = True

from test_helper import get_from_wiki, timer, NiceComm

# Might as well let these be globals.  They are the same for all tests below.
comm = NiceComm(MPI.COMM_WORLD)
rank = comm.Get_rank()
size = comm.Get_size()
file_name = os.path.join('data','Aardvark.fit')
patch_file = os.path.join('data','mpi_patches.fits')
if __name__ == '__main__':
    nrows = 0  # all rows
else:
    nrows = 10000  # quicker on pytest runs


@timer
@unittest.skipIf(no_mpi, 'Unable to import mpi4py or fitsio')
def setup():
    # Make sure we have Aardvark.fit
    if rank == 0:
        get_from_wiki('Aardvark.fit')

    # And all the tests will use these patches.  Make them once and save them.
    # For a real-life example, this might be made once and saved.
    # Or it might be made from a smaller version of the catalog:
    # either with the every_nth option, or maybe on a redmagic catalog or similar,
    # which would be smaller than the full source catalog, etc.
    if rank == 0 and not os.path.exists(patch_file):
        part_cat = treecorr.Catalog(file_name,
                                    ra_col='RA', dec_col='DEC', ra_units='deg', dec_units='deg',
                                    npatch=8)
        part_cat.write_patch_centers(patch_file)
        del part_cat

    comm.Barrier()


test_corr_params = [(treecorr.GGCorrelation, True, ['xip', 'xim', 'npairs']),
                    (treecorr.NGCorrelation, False, ['xi', 'xi_im', 'npairs']),
                    (treecorr.NKCorrelation, False, ['xi', 'npairs']),
                    (treecorr.NNCorrelation, True, ['npairs']),
                    (treecorr.KGCorrelation, False, ['xi', 'xi_im', 'npairs']),
                    (treecorr.KKCorrelation, True, ['xi', 'npairs'])]
@timer
@pytest.mark.parametrize('args', test_corr_params)
@unittest.skipIf(no_mpi, 'Unable to import mpi4py or fitsio')
def test_mpi_corr(args):
    Correlation, auto, attr = args
    if rank == 0:
        print('Start test_mpi_corr for ',Correlation.__name__,flush=True)

    # All processes make the full cat with these patches.
    # Note: this doesn't actually read anything from disk yet.
    full_cat = treecorr.Catalog(file_name,
                                ra_col='RA', dec_col='DEC', ra_units='deg', dec_units='deg',
                                g1_col='GAMMA1', g2_col='GAMMA2', k_col='KAPPA',
                                patch_centers=patch_file, last_row=nrows)

    # First run on one process
    t0 = time.time()
    if rank == 0:
        corr0 = Correlation(nbins=100, min_sep=1., max_sep=400., sep_units='arcmin')
        if auto:
            corr0.process(full_cat)
        else:
            corr0.process(full_cat, full_cat)

    t1 = time.time()
    comm.Barrier()
    print(rank,'Done with non-parallel computation',flush=True)

    # Now run in parallel.
    # Everyone needs to make their own Correlation object.
    corr1 = Correlation(nbins=100, min_sep=1., max_sep=400., sep_units='arcmin', verbose=1)

    # To use the multiple process, just pass comm to the process command.
    if auto:
        corr1.process(full_cat, comm=comm)
    else:
        corr1.process(full_cat, full_cat, comm=comm)
    t2 = time.time()
    comm.Barrier()
    print(rank,'Done with parallel computation',flush=True)

    # rank 0 has the completed result.
    if rank == 0:
        print('serial   %s = '%attr[0],getattr(corr0,attr[0]), t1-t0,flush=True)
        print('parallel %s = '%attr[0],getattr(corr1,attr[0]), t2-t1,flush=True)
        for a in attr:
            np.testing.assert_allclose(getattr(corr0,a), getattr(corr1,a))

if __name__ == '__main__':
    setup()
    for args in test_corr_params:
        test_mpi_corr(args)
