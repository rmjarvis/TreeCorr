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
import time
import os
import treecorr

def setup():
    from test_helper import get_from_wiki
    try:
        import fitsio
    except ImportError:
        print('Cannot do mpi tests since fitsio is not installed.')
        return

    file_name = os.path.join('data','Aardvark.fit')
    patch_file = os.path.join('data','mpi_patches.fits')

    # Make sure we have Aardvark.fit
    get_from_wiki('Aardvark.fit')

    # And all the tests will use these patches.  Make them once and save them.
    # For a real-life example, this might be made once and saved.
    # Or it might be made from a smaller version of the catalog:
    # either with the every_nth option, or maybe on a redmagic catalog or similar,
    # which would be smaller than the full source catalog, etc.
    if not os.path.exists(patch_file):
        part_cat = treecorr.Catalog(file_name,
                                    ra_col='RA', dec_col='DEC', ra_units='deg', dec_units='deg',
                                    npatch=8)
        part_cat.write_patch_centers(patch_file)
        del part_cat

def do_mpi_corr(comm, Correlation, auto, attr, output=True):
    rank = comm.Get_rank()
    size = comm.Get_size()
    file_name = os.path.join('data','Aardvark.fit')
    patch_file = os.path.join('data','mpi_patches.fits')
    if False:
        nrows = 0  # all rows
    else:
        nrows = 10000  # quicker, and still tests functionality.

    if rank == 0 and output:
        print('Start do_mpi_corr for ',Correlation.__name__,flush=True)
        print('size = ',size)

    # All processes make the full cat with these patches.
    # Note: this doesn't actually read anything from disk yet.
    cat = treecorr.Catalog(file_name,
                           ra_col='RA', dec_col='DEC', ra_units='deg', dec_units='deg',
                           g1_col='GAMMA1', g2_col='GAMMA2', k_col='KAPPA',
                           patch_centers=patch_file, last_row=nrows)
    if output:
        print(rank,'Made catalog',flush=True)

    # First run on one process
    t0 = time.time()
    if rank == 0:
        corr0 = Correlation(nbins=10, min_sep=1., max_sep=40., sep_units='arcmin')
        if auto:
            corr0.process(cat)
        else:
            corr0.process(cat, cat)

    t1 = time.time()
    comm.Barrier()
    if output:
        print(rank,'Done with non-parallel computation',flush=True)

    # Now run in parallel.
    # Everyone needs to make their own Correlation object.
    corr1 = Correlation(nbins=10, min_sep=1., max_sep=40., sep_units='arcmin', verbose=1)

    # To use the multiple process, just pass comm to the process command.
    if auto:
        corr1.process(cat, comm=comm)
    else:
        corr1.process(cat, cat, comm=comm)
    t2 = time.time()
    comm.Barrier()
    if output:
        print(rank,'Done with parallel computation',flush=True)

    # rank 0 has the completed result.
    if rank == 0 and output:
        print('serial   %s = '%attr[0],getattr(corr0,attr[0]), t1-t0,flush=True)
        print('parallel %s = '%attr[0],getattr(corr1,attr[0]), t2-t1,flush=True)
    if rank == 0:
        for a in attr:
            np.testing.assert_allclose(getattr(corr0,a), getattr(corr1,a))

def do_mpi_corr2(comm, Correlation, attr, output=True):
    # Repeat cross correlations where one of the two catalogs doesn't use patches.

    rank = comm.Get_rank()
    size = comm.Get_size()
    file_name = os.path.join('data','Aardvark.fit')
    patch_file = os.path.join('data','mpi_patches.fits')
    if False:
        nrows = 0  # all rows
    else:
        nrows = 10000  # quicker, and still tests functionality.

    if rank == 0 and output:
        print('Start do_mpi_corr2 for ',Correlation.__name__,flush=True)
        print('size = ',size)

    cat1 = treecorr.Catalog(file_name,
                            ra_col='RA', dec_col='DEC', ra_units='deg', dec_units='deg',
                            g1_col='GAMMA1', g2_col='GAMMA2', k_col='KAPPA',
                            patch_centers=patch_file, last_row=nrows)
    cat2 = treecorr.Catalog(file_name,
                            ra_col='RA', dec_col='DEC', ra_units='deg', dec_units='deg',
                            g1_col='GAMMA1', g2_col='GAMMA2', k_col='KAPPA',
                            last_row=nrows)

    # First run on one process
    t0 = time.time()
    if rank == 0:
        corr0 = Correlation(nbins=10, min_sep=1., max_sep=40., sep_units='arcmin')
        corr0.process(cat1, cat2)

    t1 = time.time()
    comm.Barrier()
    if output:
        print(rank,'Done with non-parallel computation',flush=True)

    # Now run in parallel.
    # Everyone needs to make their own Correlation object.
    corr1 = Correlation(nbins=10, min_sep=1., max_sep=40., sep_units='arcmin', verbose=1)

    # To use the multiple process, just pass comm to the process command.
    corr1.process(cat1, cat2, comm=comm)
    t2 = time.time()
    comm.Barrier()
    if output:
        print(rank,'Done with parallel computation',flush=True)

    # rank 0 has the completed result.
    if rank == 0 and output:
        print('serial   %s = '%attr[0],getattr(corr0,attr[0]), t1-t0,flush=True)
        print('parallel %s = '%attr[0],getattr(corr1,attr[0]), t2-t1,flush=True)
    if rank == 0:
        for a in attr:
            np.testing.assert_allclose(getattr(corr0,a), getattr(corr1,a))

    # Repeat in the other direction.
    t0 = time.time()
    if rank == 0:
        corr0.process(cat2, cat1)
    t1 = time.time()
    comm.Barrier()
    if output:
        print(rank,'Done with non-parallel computation',flush=True)

    corr1.process(cat2, cat1, comm=comm)
    t2 = time.time()
    comm.Barrier()
    if output:
        print(rank,'Done with parallel computation',flush=True)

    if rank == 0 and output:
        print('serial   %s = '%attr[0],getattr(corr0,attr[0]), t1-t0,flush=True)
        print('parallel %s = '%attr[0],getattr(corr1,attr[0]), t2-t1,flush=True)
    if rank == 0:
        for a in attr:
            np.testing.assert_allclose(getattr(corr0,a), getattr(corr1,a))

def do_mpi_gg(comm, output=True):
    do_mpi_corr(comm, treecorr.GGCorrelation, True, ['xip', 'xim', 'npairs'], output)

def do_mpi_ng(comm, output=True):
    do_mpi_corr(comm, treecorr.NGCorrelation, False, ['xi', 'xi_im', 'npairs'], output)
    do_mpi_corr2(comm, treecorr.NGCorrelation, ['xi', 'xi_im', 'npairs'], output)

def do_mpi_nk(comm, output=True):
    do_mpi_corr(comm, treecorr.NKCorrelation, False, ['xi', 'npairs'], output)
    do_mpi_corr2(comm, treecorr.NKCorrelation, ['xi', 'npairs'], output)

def do_mpi_nn(comm, output=True):
    do_mpi_corr(comm, treecorr.NNCorrelation, True, ['npairs'], output)

def do_mpi_kg(comm, output=True):
    do_mpi_corr(comm, treecorr.KGCorrelation, False, ['xi', 'xi_im', 'npairs'], output)
    do_mpi_corr2(comm, treecorr.KGCorrelation, ['xi', 'xi_im', 'npairs'], output)

def do_mpi_kk(comm, output=True):
    do_mpi_corr(comm, treecorr.KKCorrelation, True, ['xi', 'npairs'], output)


def do_mpi_cov(comm, method, output=True):
    # Test covariance estimation under MPI
    if output:
        print("Running test of", method, "with MPI size ",  comm.size)
    from test_patch import generate_shear_field
    nside = 200
    npatch = 16
    tol = 1.0e-8

    # Generate a random catalog to use. Because all the processes
    # have the same random seed they will get the same catalogs,
    # so everything should work.
    rng = np.random.RandomState(1234)
    x, y, g1, g2, _ = generate_shear_field(nside, rng)
    cat = treecorr.Catalog(x=x, y=y, g1=g1, g2=g2, npatch=npatch)
    xr, yr, _, _, _ = generate_shear_field(nside, rng)
    ran_cat = treecorr.Catalog(x=x, y=y, npatch=npatch)

    # Generate the three sets of correlations we will use
    rng = np.random.RandomState(31415)
    gg = treecorr.GGCorrelation(bin_size=0.3, min_sep=10., max_sep=50., rng=rng)
    ng = treecorr.NGCorrelation(bin_size=0.3, min_sep=10., max_sep=50., rng=rng)
    nn = treecorr.NNCorrelation(bin_size=0.3, min_sep=10., max_sep=50., rng=rng)
    rr = treecorr.NNCorrelation(bin_size=0.3, min_sep=10., max_sep=50., rng=rng)
    if output:
        print(comm.rank, "Running GG process")
    gg.process(cat, comm=comm)
    if output:
        print(comm.rank, "Running NG process")
    ng.process(cat, cat, comm=comm)
    if output:
        print(comm.rank, "Running NN process")
    nn.process(cat, comm=comm)
    if output:
        print(comm.rank, "Running RR process")
    rr.process(ran_cat, comm=comm)

    # Only the root process gets the complete version
    # when you call the above with comm
    if output:
        print(comm.rank, "Broadcasting")
    gg = comm.bcast(gg)
    ng = comm.bcast(ng)
    nn = comm.bcast(nn)
    rr = comm.bcast(rr)

    if output:
        print(comm.rank, "XI calc")
    ng.calculateXi()
    nn.calculateXi(rr=rr)

    corrs = [gg, ng, nn]
    gg_func = lambda corr: corr.calculateMapSq()[0]

    # Get the baseline single process covariance
    if comm.rank == 0:
        if output:
            print(comm.rank, "Single process covariance")
        cov1 = treecorr.estimate_multi_cov(corrs, method)
        A1, w1 = treecorr.build_multi_cov_design_matrix(corrs, method)
        cov1b = gg.estimate_cov(method, func=gg_func)
        A1b, w1b = gg.build_cov_design_matrix(method, func=gg_func)
    else:
        cov1 = A1 = w1 = cov1b = A1b = w1b = None
    cov1 = comm.bcast(cov1)
    A1 = comm.bcast(A1)
    w1 = comm.bcast(w1)
    cov1b = comm.bcast(cov1b)
    A1b = comm.bcast(A1b)
    w1b = comm.bcast(w1b)

    if output:
        print("\nCOV 1 \n", cov1[0:3,0:3], " for rank ", comm.rank, " of ", comm.size)

    rng = np.random.RandomState(31415)
    gg = treecorr.GGCorrelation(bin_size=0.3, min_sep=10., max_sep=50., rng=rng)
    ng = treecorr.NGCorrelation(bin_size=0.3, min_sep=10., max_sep=50., rng=rng)
    nn = treecorr.NNCorrelation(bin_size=0.3, min_sep=10., max_sep=50., rng=rng)
    rr = treecorr.NNCorrelation(bin_size=0.3, min_sep=10., max_sep=50., rng=rng)
    if output:
        print(comm.rank, "Running GG process 2")
    gg.process(cat, comm=comm)
    if output:
        print(comm.rank, "Running NG process 2")
    ng.process(cat, cat, comm=comm)
    if output:
        print(comm.rank, "Running NN process 2")
    nn.process(cat, comm=comm)
    if output:
        print(comm.rank, "Running RR process 2")
    rr.process(ran_cat, comm=comm)
    # Only the root process gets the complete version
    # when you call the above with comm
    if output:
        print(comm.rank, "Broadcasting 2")
    gg = comm.bcast(gg)
    ng = comm.bcast(ng)
    nn = comm.bcast(nn)
    rr = comm.bcast(rr)

    if output:
        print(comm.rank, "Calculate xi 2")
    ng.calculateXi()
    nn.calculateXi(rr=rr)
    corrs = [gg, ng, nn]

    # Compare to the MPI-estimated covariance
    if output:
        print(comm.rank, "MPI cov")
    cov2 = treecorr.estimate_multi_cov(corrs, method, comm=comm)
    if output:
        print("\nCOV 2\n", cov2[0:3,0:3], " for ", comm.rank, "\n")

    np.testing.assert_allclose(cov1, cov2, atol=tol)

    A2, w2 = treecorr.build_multi_cov_design_matrix(corrs, method, comm=comm)
    np.testing.assert_allclose(A1, A2, atol=tol)
    np.testing.assert_allclose(w1, w2, atol=tol)

    cov2b = gg.estimate_cov(method, func=gg_func, comm=comm)
    A2b, w2b = gg.build_cov_design_matrix(method, func=gg_func, comm=comm)
    np.testing.assert_allclose(A1b, A2b, atol=tol)
    np.testing.assert_allclose(w1b, w2b, atol=tol)


if __name__ == '__main__':
    from mpi4py import MPI
    from mpi_helper import NiceComm
    comm = NiceComm(MPI.COMM_WORLD)
    rank = comm.Get_rank()
    if rank == 0:
        setup()
    comm.Barrier()
    do_mpi_gg(comm)
    do_mpi_ng(comm)
    do_mpi_nk(comm)
    do_mpi_nn(comm)
    do_mpi_kg(comm)
    do_mpi_kk(comm)
    do_mpi_cov(comm, "jackknife")
    do_mpi_cov(comm, "bootstrap")
    do_mpi_cov(comm, "marked_bootstrap")
    do_mpi_cov(comm, "sample")
