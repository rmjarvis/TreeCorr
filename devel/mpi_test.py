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

# Run using:
#   mpiexec -n 4 python mpi_test.py

# Note: This script is the basis for mpi_test.py in the tests directory.
#       This version is probably a bit closer to a real world application.

import numpy as np
import time
import os
import treecorr
from mpi4py import MPI

from test_helper import NiceComm

# NiceComm makes it so Barrier doesn't run 100% CPU.  This isn't so important if you're
# really running mpi on multiple machines, but it helps when running on a single machine.
comm = NiceComm(MPI.COMM_WORLD)

rank = comm.Get_rank()
nproc = comm.Get_size()

file_name = 'Aardvark.fit'
patch_file = 'mpi_patches.fits'

# First make the patches.  Do this on one process.
# For a real-life example, this might be made once and saved.
# Or it might be made from a smaller version of the catalog:
# either with the every_nth option, or maybe on a redmagic catalog or similar,
# which would be smaller than the full source catalog, etc.
if rank == 0 and not os.path.exists(patch_file):
    part_cat = treecorr.Catalog(file_name,
                                ra_col='RA', dec_col='DEC', ra_units='deg', dec_units='deg',
                                g1_col='GAMMA1', g2_col='GAMMA2',
                                npatch=32)
    part_cat.write_patch_centers(patch_file)
    del part_cat
    print('Done making patches',flush=True)

comm.Barrier()

# Now all processes make the full cat with these patches.
# Note: this doesn't actually read anything from disk yet.
full_cat = treecorr.Catalog(file_name,
                            ra_col='RA', dec_col='DEC', ra_units='deg', dec_units='deg',
                            g1_col='GAMMA1', g2_col='GAMMA2', k_col='KAPPA',
                            patch_centers=patch_file)
print(rank,'Made full_cat',flush=True)

# First run on one process.  A real world example would skip this part.
t0 = time.time()
if rank == 0:
    gg0 = treecorr.GGCorrelation(nbins=20, min_sep=1., max_sep=400., sep_units='arcmin')
    gg0.process(full_cat)
comm.Barrier()
t1 = time.time()
print(rank,'Done with non-parallel computation',t1-t0,flush=True)

# Now run in parallel.
# Everyone needs to make their own Correlation object.
gg1 = treecorr.GGCorrelation(nbins=20, min_sep=1., max_sep=400., sep_units='arcmin',
                             verbose=1)

# To use the multiple process, just pass comm to the process command.
gg1.process(full_cat, comm=comm)
comm.Barrier()
t2 = time.time()
print(rank,'Done with parallel computation',t2-t1,flush=True)

# rank 0 has the completed result.
if rank == 0:
    print('serial   xip = ',gg0.xip, t1-t0,flush=True)
    print('parallel xip = ',gg1.xip, t2-t1,flush=True)

    np.testing.assert_allclose(gg0.npairs, gg1.npairs)
    np.testing.assert_allclose(gg0.xip, gg1.xip)
    np.testing.assert_allclose(gg0.xim, gg1.xim)
