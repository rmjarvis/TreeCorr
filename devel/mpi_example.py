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
#
# Note for NERSC users: The conda (or pip) installed won't work correctly.
# You need to install mpi4py by hand using their cray compilers.  See:
# https://docs.nersc.gov/programming/high-level-environments/python/mpi4py/#mpi4py-in-your-custom-conda-environment
#
# See also the issue that I was having before resolving this:
# https://nersc.servicenowservices.com/nav_to.do?uri=%2Fincident.do%3Fsys_id%3Dcc9f38341be85c102548ea82f54bcbea%26sysparm_view%3Dess

import time
import os
import sys
import shutil
import socket
import fitsio
import treecorr
try:
    from urllib2 import urlopen
except ImportError:
    from urllib.request import urlopen
from mpi4py import MPI

# Some parameters you can play with here that will affect both serial (not really "serial", since
# it still uses OpenMP -- just running on 1 node) and parallel runs.
bin_size = 0.01
min_sep = 1         # arcmin
max_sep = 600
bin_slop = 1        # Can dial down to 0 to take longer
low_mem = False     # Set to True to use less memory during processing.

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
nproc = comm.Get_size()

if 1:
    # DES Y1: 80 GB
    file_name = "mcal-y1a1.fits"
    patch_file = 'y1_patches.fits'
    url = "http://desdr-server.ncsa.illinois.edu/despublic/y1a1_files/shear_catalogs/mcal-y1a1-combined-riz-unblind-v4-matched.fits"
    ra_col='ra'
    dec_col='dec'
    ra_units='deg'
    dec_units='deg'
    g1_col='e1'
    g2_col='e2'
    flag_col='flags_select'
else:
    # Aardvark: 15 MB
    file_name= "Aardvark.fit"
    patch_file = 'aardvark_patches.fits'
    url = "https://github.com/rmjarvis/TreeCorr/wiki/Aardvark.fit"
    ra_col='RA'
    dec_col='DEC'
    ra_units='deg'
    dec_units='deg'
    g1_col='GAMMA1'
    g2_col='GAMMA2'
    flag_col=None


def download_file():
    if not os.path.exists(file_name):
        u = urlopen(url)
        print('urlinfo: ')
        print(u.info())
        file_size = int(u.info().get("Content-Length"))
        print("file_size = %d MBytes"%(file_size/1024**2))
        with open('/proc/sys/net/core/rmem_default', 'r') as f:
            block_sz = int(f.read())
        print("block size = %d KBytes"%(block_sz/1024))
        with open(file_name, 'wb') as f:
            file_size_dl = 0
            dot_step = file_size / 400.
            next_dot = dot_step
            while True:
                buffer = u.read(block_sz)
                if not buffer: break
                file_size_dl += len(buffer)
                f.write(buffer)
                # Easy status bar
                if file_size_dl > next_dot:
                    sys.stdout.write('.')
                    sys.stdout.flush()
                    next_dot += dot_step

        print('Done downloading',file_name)
    else:
        print('Using catalog file %s'%file_name)

    # It's helpful to have a separate file for each process.  Otherwise they all end up
    # fighting over the read and the I/O becomes much slower.
    # It's also vv helpful to save a version with only the relevant columns, so fitsio
    # doesn't have to scan past all the useless extra information.

    fname_0 = file_name.replace('.fits','_0.fits')
    if not os.path.exists(fname_0):
        all_cols = [ra_col, dec_col, g1_col, g2_col, flag_col]
        all_cols = [c for c in all_cols if c is not None]
        with fitsio.FITS(file_name, 'r') as fits:
            data = fits[1][all_cols][:]
        fitsio.write(fname_0, data)
        print('wrote',fname_0)
    for p in range(nproc):
        fname_p = file_name.replace('.fits','_%d.fits'%p)
        if not os.path.exists(fname_p):
            shutil.copyfile(fname_0, fname_p)
            print('copied',fname_0,'to',fname_p)

def make_patches():
    # First make the patches.  Do this on one process.
    # For a real-life example, this might be made once and saved.
    # Or it might be made from a smaller version of the catalog:
    # either with the every_nth option, or maybe on a redmagic catalog or similar,
    # which would be smaller than the full source catalog, etc.
    if not os.path.exists(patch_file):
        print('Making patches')
        fname = file_name.replace('.fits','_0.fits')
        cat = treecorr.Catalog(fname,
                               ra_col=ra_col, dec_col=ra_col,
                               ra_units=ra_units, dec_units=dec_units,
                               g1_col=g1_col, g2_col=g1_col, flag_col=flag_col,
                               npatch=32, verbose=2)
        print('Done loading file: nobj = ',cat.nobj,cat.ntot)
        cat.get_patches()
        print('Made patches: ',cat.patch_centers)
        cat.write_patch_centers(patch_file)
        print('Wrote patch file ',patch_file)
    else:
        print('Using existing patch file')

def run_serial():
    from test_helper import profile
    t0 = time.time()
    fname = file_name.replace('.fits','_0.fits')
    log_file = 'serial.log'

    cat = treecorr.Catalog(fname,
                           ra_col=ra_col, dec_col=ra_col,
                           ra_units=ra_units, dec_units=dec_units,
                           g1_col=g1_col, g2_col=g1_col, flag_col=flag_col,
                           verbose=1, log_file=log_file,
                           patch_centers=patch_file)
    t1 = time.time()
    print('Made cat', t1-t0)

    gg = treecorr.GGCorrelation(bin_size=bin_size, min_sep=min_sep, max_sep=max_sep,
                                sep_units='arcmin', bin_slop=bin_slop,
                                verbose=1, log_file=log_file)

    # These next two steps don't need to be done separately.  They will automatically
    # happen when calling process.  But separating them out makes it easier to profile.
    with profile():
        cat.load()
    t2 = time.time()
    print('Loaded', t2-t1)

    with profile():
        cat.get_patches()
    t3 = time.time()
    print('Made patches', t3-t2)

    with profile():
        gg.process(cat, low_mem=low_mem)
    t4 = time.time()
    print('Processed', t4-t3)
    print('Done with non-parallel computation',t4-t0)
    print('xip = ',gg.xip, flush=True)

def run_parallel():

    t0 = time.time()
    print(rank,socket.gethostname(),flush=True)
    fname = file_name.replace('.fits','_%d.fits'%rank)[:]
    log_file = 'parallel_%d.log'%rank

    # All processes make the full cat with these patches.
    # Note: this doesn't actually read anything from disk yet.
    cat = treecorr.Catalog(fname,
                           ra_col=ra_col, dec_col=ra_col,
                           ra_units=ra_units, dec_units=dec_units,
                           g1_col=g1_col, g2_col=g1_col, flag_col=flag_col,
                           verbose=1, log_file=log_file,
                           patch_centers=patch_file)
    t1 = time.time()
    print('Made cat', t1-t0, flush=True)

    # Everyone needs to make their own Correlation object.
    gg = treecorr.GGCorrelation(bin_size=bin_size, min_sep=min_sep, max_sep=max_sep,
                                sep_units='arcmin', bin_slop=bin_slop,
                                verbose=1, log_file=log_file)

    cat.load()
    t2 = time.time()
    print(rank,'Loaded', t2-t1, flush=True)

    cat.get_patches()
    t3 = time.time()
    print(rank,'Made patches', t3-t2, flush=True)

    # To use the multiple process, just pass comm to the process command.
    gg.process(cat, comm=comm, low_mem=low_mem)
    t4 = time.time()
    print(rank,'Processed', t4-t3, flush=True)
    comm.Barrier()
    t5 = time.time()
    print(rank,'Barrier', t5-t4, flush=True)
    print(rank,'Done with parallel computation',t5-t0,flush=True)

    # rank 0 has the completed result.
    if rank == 0:
        print('xip = ',gg.xip, flush=True)

if __name__ == '__main__':
    if rank == 0:
        download_file()
        make_patches()
        run_serial()
    comm.Barrier()
    run_parallel()
