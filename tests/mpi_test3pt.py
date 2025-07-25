# Copyright (c) 2003-2024 by Mike Jarvis
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

def do_mpi_corr(comm, Correlation, cross, attr, output=True, patch_method=None, low_mem=False):
    rank = comm.Get_rank()
    size = comm.Get_size()
    if rank == 0:
        setup()
    comm.Barrier()
    file_name = os.path.join('data','Aardvark.fit')
    patch_file = os.path.join('data','mpi_patches.fits')
    # Takes forever to do anywhere close to whole catalog.
    if __name__ == '__main__':
        nth = 2000
    else:
        nth = 5000

    if rank == 0 and output:
        print('Start do_mpi_corr for ',Correlation.__name__,flush=True)
        print('size = ',size)

    cat = treecorr.Catalog(file_name, every_nth=nth,
                           ra_col='RA', dec_col='DEC', ra_units='deg', dec_units='deg',
                           g1_col='GAMMA1', g2_col='GAMMA2', k_col='KAPPA',
                           patch_centers=patch_file)
    if rank == 0 and output:
        print('nobj = ',cat.nobj)
        print(rank,'Made catalog',flush=True)

    config = dict(nbins=3, min_sep=100., max_sep=200., sep_units='arcmin',
                  min_u=0.9, max_u=1.0, nubins=1,
                  min_v=0.0, max_v=0.1, nvbins=1, bin_slop=0, bin_type='LogRUV')

    # First run on one process
    t0 = time.time()
    if rank == 0:
        # Limit to fairly small nearly equilateral triangles.
        corr0 = Correlation(config)
        if cross == 0:
            corr0.process(cat, patch_method=patch_method, low_mem=low_mem)
        elif cross == 1:
            corr0.process(cat, cat, patch_method=patch_method, low_mem=low_mem)
        else:
            corr0.process(cat, cat, cat, patch_method=patch_method, low_mem=low_mem)

    t1 = time.time()
    comm.Barrier()
    if output:
        print(rank,'Done with non-parallel computation',flush=True)

    # Now run in parallel.
    # Everyone needs to make their own Correlation object.
    log_file='output/log%d.out'%rank
    if os.path.isfile(log_file):
        os.remove(log_file)
    corr1 = Correlation(config, verbose=2, log_file=log_file, output_dots=False)

    # To use the multiple process, just pass comm to the process command.
    if cross == 0:
        corr1.logger.info('cross=0')
        corr1.process(cat, comm=comm, patch_method=patch_method, low_mem=low_mem)
    elif cross == 1:
        corr1.logger.info('cross=1')
        corr1.process(cat, cat, comm=comm, patch_method=patch_method, low_mem=low_mem)
    else:
        corr1.logger.info('cross=2')
        corr1.process(cat, cat, cat, comm=comm, patch_method=patch_method, low_mem=low_mem)
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

def do_mpi_corr2(comm, Correlation, cross, attr, output=True, patch_method='global', low_mem=False):
    # Repeat cross correlations where one of the catalogs doesn't use patches.

    rank = comm.Get_rank()
    size = comm.Get_size()
    if rank == 0:
        setup()
    comm.Barrier()
    file_name = os.path.join('data','Aardvark.fit')
    patch_file = os.path.join('data','mpi_patches.fits')
    # Takes forever to do anywhere close to whole catalog.
    if __name__ == '__main__':
        nth = 2000
    else:
        nth = 5000

    if rank == 0 and output:
        print('Start do_mpi_corr2 for ',Correlation.__name__,flush=True)
        print('size = ',size)

    cat1 = treecorr.Catalog(file_name, every_nth=nth,
                            ra_col='RA', dec_col='DEC', ra_units='deg', dec_units='deg',
                            g1_col='GAMMA1', g2_col='GAMMA2', k_col='KAPPA',
                            patch_centers=patch_file)
    cat2 = treecorr.Catalog(file_name, every_nth=nth,
                            ra_col='RA', dec_col='DEC', ra_units='deg', dec_units='deg',
                            g1_col='GAMMA1', g2_col='GAMMA2', k_col='KAPPA')

    config = dict(nbins=3, min_sep=100., max_sep=200., sep_units='arcmin',
                  min_u=0.9, max_u=1.0, nubins=1,
                  min_v=0.0, max_v=0.1, nvbins=1, bin_slop=0, bin_type='LogRUV')

    # First run on one process
    t0 = time.time()
    if rank == 0:
        corr0 = Correlation(config)
        if cross == 0:
            corr0.process(cat1, cat2, patch_method=patch_method, low_mem=low_mem)
        elif cross == 1:
            corr0.process(cat2, cat1, patch_method=patch_method, low_mem=low_mem)
        elif cross == 2:
            corr0.process(cat1, cat2, cat2, patch_method=patch_method, low_mem=low_mem)
        elif cross == 3:
            corr0.process(cat2, cat1, cat2, patch_method=patch_method, low_mem=low_mem)
        elif cross == 4:
            corr0.process(cat2, cat2, cat1, patch_method=patch_method, low_mem=low_mem)
        elif cross == 5:
            corr0.process(cat1, cat2, ordered=False, patch_method=patch_method, low_mem=low_mem)
        else:
            corr0.process(cat1, cat2, cat2, ordered=False, patch_method=patch_method, low_mem=low_mem)

    t1 = time.time()
    comm.Barrier()
    if output:
        print(rank,'Done with non-parallel computation',flush=True)

    # Now run in parallel.
    # Everyone needs to make their own Correlation object.
    log_file='output/log%d.out'%rank
    if os.path.isfile(log_file):
        os.remove(log_file)
    corr1 = Correlation(config, verbose=2, log_file=log_file, output_dots=False)

    # To use the multiple process, just pass comm to the process command.
    if cross == 0:
        corr1.process(cat1, cat2, comm=comm, patch_method=patch_method, low_mem=low_mem)
    elif cross == 1:
        corr1.process(cat2, cat1, comm=comm, patch_method=patch_method, low_mem=low_mem)
    elif cross == 2:
        corr1.process(cat1, cat2, cat2, comm=comm, patch_method=patch_method, low_mem=low_mem)
    elif cross == 3:
        corr1.process(cat2, cat1, cat2, comm=comm, patch_method=patch_method, low_mem=low_mem)
    elif cross == 4:
        corr1.process(cat2, cat2, cat1, comm=comm, patch_method=patch_method, low_mem=low_mem)
    elif cross == 5:
        corr1.process(cat1, cat2, ordered=False, comm=comm, patch_method=patch_method, low_mem=low_mem)
    else:
        corr1.process(cat1, cat2, cat2, ordered=False, comm=comm, patch_method=patch_method, low_mem=low_mem)
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

def do_mpi_ggg(comm, output=True):
    do_mpi_corr(comm, treecorr.GGGCorrelation, 0, ['ntri', 'gam0', 'gam1', 'gam2', 'gam3'], output)
    do_mpi_corr(comm, treecorr.GGGCorrelation, 0, ['ntri', 'gam0', 'gam1', 'gam2', 'gam3'], output,
                'local')
    do_mpi_corr(comm, treecorr.GGGCorrelation, 0, ['ntri', 'gam0', 'gam1', 'gam2', 'gam3'], output,
                'global', True)
    do_mpi_corr(comm, treecorr.GGGCorrelation, 0, ['ntri', 'gam0', 'gam1', 'gam2', 'gam3'], output,
                'local', True)

def do_mpi_kkk(comm, output=True):
    do_mpi_corr(comm, treecorr.KKKCorrelation, 0, ['ntri', 'zeta'], output)
    do_mpi_corr(comm, treecorr.KKKCorrelation, 1, ['ntri', 'zeta'], output)
    do_mpi_corr(comm, treecorr.KKKCorrelation, 2, ['ntri', 'zeta'], output)
    do_mpi_corr(comm, treecorr.KKKCorrelation, 0, ['ntri', 'zeta'], output, 'local')
    do_mpi_corr(comm, treecorr.KKKCorrelation, 1, ['ntri', 'zeta'], output, 'local')
    do_mpi_corr(comm, treecorr.KKKCorrelation, 2, ['ntri', 'zeta'], output, 'local')
    do_mpi_corr(comm, treecorr.KKKCorrelation, 0, ['ntri', 'zeta'], output, 'global', True)
    do_mpi_corr(comm, treecorr.KKKCorrelation, 1, ['ntri', 'zeta'], output, 'global', True)
    do_mpi_corr(comm, treecorr.KKKCorrelation, 2, ['ntri', 'zeta'], output, 'global', True)
    do_mpi_corr(comm, treecorr.KKKCorrelation, 0, ['ntri', 'zeta'], output, 'local', True)
    do_mpi_corr(comm, treecorr.KKKCorrelation, 1, ['ntri', 'zeta'], output, 'local', True)
    do_mpi_corr(comm, treecorr.KKKCorrelation, 2, ['ntri', 'zeta'], output, 'local', True)

def do_mpi_kkk2(comm, output=True):
    do_mpi_corr2(comm, treecorr.KKKCorrelation, 0, ['ntri', 'zeta'], output)
    do_mpi_corr2(comm, treecorr.KKKCorrelation, 1, ['ntri', 'zeta'], output)
    do_mpi_corr2(comm, treecorr.KKKCorrelation, 2, ['ntri', 'zeta'], output)
    do_mpi_corr2(comm, treecorr.KKKCorrelation, 3, ['ntri', 'zeta'], output)
    do_mpi_corr2(comm, treecorr.KKKCorrelation, 4, ['ntri', 'zeta'], output)
    do_mpi_corr2(comm, treecorr.KKKCorrelation, 5, ['ntri', 'zeta'], output)
    do_mpi_corr2(comm, treecorr.KKKCorrelation, 6, ['ntri', 'zeta'], output)
    do_mpi_corr2(comm, treecorr.KKKCorrelation, 0, ['ntri', 'zeta'], output, 'local')
    do_mpi_corr2(comm, treecorr.KKKCorrelation, 1, ['ntri', 'zeta'], output, 'local')
    do_mpi_corr2(comm, treecorr.KKKCorrelation, 2, ['ntri', 'zeta'], output, 'local')
    do_mpi_corr2(comm, treecorr.KKKCorrelation, 3, ['ntri', 'zeta'], output, 'local')
    do_mpi_corr2(comm, treecorr.KKKCorrelation, 4, ['ntri', 'zeta'], output, 'local')
    do_mpi_corr2(comm, treecorr.KKKCorrelation, 5, ['ntri', 'zeta'], output, 'local')
    do_mpi_corr2(comm, treecorr.KKKCorrelation, 6, ['ntri', 'zeta'], output, 'local')
    do_mpi_corr2(comm, treecorr.KKKCorrelation, 0, ['ntri', 'zeta'], output, 'global', True)
    do_mpi_corr2(comm, treecorr.KKKCorrelation, 1, ['ntri', 'zeta'], output, 'global', True)
    do_mpi_corr2(comm, treecorr.KKKCorrelation, 2, ['ntri', 'zeta'], output, 'global', True)
    do_mpi_corr2(comm, treecorr.KKKCorrelation, 3, ['ntri', 'zeta'], output, 'global', True)
    do_mpi_corr2(comm, treecorr.KKKCorrelation, 4, ['ntri', 'zeta'], output, 'global', True)
    do_mpi_corr2(comm, treecorr.KKKCorrelation, 5, ['ntri', 'zeta'], output, 'global', True)
    do_mpi_corr2(comm, treecorr.KKKCorrelation, 6, ['ntri', 'zeta'], output, 'global', True)
    do_mpi_corr2(comm, treecorr.KKKCorrelation, 0, ['ntri', 'zeta'], output, 'local', True)
    do_mpi_corr2(comm, treecorr.KKKCorrelation, 1, ['ntri', 'zeta'], output, 'local', True)
    do_mpi_corr2(comm, treecorr.KKKCorrelation, 2, ['ntri', 'zeta'], output, 'local', True)
    do_mpi_corr2(comm, treecorr.KKKCorrelation, 3, ['ntri', 'zeta'], output, 'local', True)
    do_mpi_corr2(comm, treecorr.KKKCorrelation, 4, ['ntri', 'zeta'], output, 'local', True)
    do_mpi_corr2(comm, treecorr.KKKCorrelation, 5, ['ntri', 'zeta'], output, 'local', True)
    do_mpi_corr2(comm, treecorr.KKKCorrelation, 6, ['ntri', 'zeta'], output, 'local', True)

def do_mpi_nnk(comm, output=True):
    do_mpi_corr(comm, treecorr.NNKCorrelation, 1, ['ntri', 'zeta'], output)
    do_mpi_corr(comm, treecorr.NNKCorrelation, 1, ['ntri', 'zeta'], output, 'local')
    do_mpi_corr(comm, treecorr.NNKCorrelation, 1, ['ntri', 'zeta'], output, 'global', True)
    do_mpi_corr(comm, treecorr.NNKCorrelation, 1, ['ntri', 'zeta'], output, 'local', True)
    do_mpi_corr2(comm, treecorr.NNKCorrelation, 1, ['ntri', 'zeta'], output)
    do_mpi_corr(comm, treecorr.NKNCorrelation, 2, ['ntri', 'zeta'], output)
    do_mpi_corr(comm, treecorr.NKNCorrelation, 2, ['ntri', 'zeta'], output, 'local')
    do_mpi_corr(comm, treecorr.NKNCorrelation, 2, ['ntri', 'zeta'], output, 'global', True)
    do_mpi_corr(comm, treecorr.NKNCorrelation, 2, ['ntri', 'zeta'], output, 'local', True)
    do_mpi_corr2(comm, treecorr.NKNCorrelation, 2, ['ntri', 'zeta'], output)
    do_mpi_corr(comm, treecorr.KNNCorrelation, 1, ['ntri', 'zeta'], output)
    do_mpi_corr(comm, treecorr.KNNCorrelation, 1, ['ntri', 'zeta'], output, 'local')
    do_mpi_corr(comm, treecorr.KNNCorrelation, 1, ['ntri', 'zeta'], output, 'global', True)
    do_mpi_corr(comm, treecorr.KNNCorrelation, 1, ['ntri', 'zeta'], output, 'local', True)
    do_mpi_corr2(comm, treecorr.KNNCorrelation, 1, ['ntri', 'zeta'], output)

def do_mpi_nng(comm, output=True):
    do_mpi_corr(comm, treecorr.NNGCorrelation, 1, ['ntri', 'zeta'], output)
    do_mpi_corr(comm, treecorr.NNGCorrelation, 1, ['ntri', 'zeta'], output, 'local')
    do_mpi_corr(comm, treecorr.NNGCorrelation, 1, ['ntri', 'zeta'], output, 'global', True)
    do_mpi_corr(comm, treecorr.NNGCorrelation, 1, ['ntri', 'zeta'], output, 'local', True)
    do_mpi_corr2(comm, treecorr.NNGCorrelation, 1, ['ntri', 'zeta'], output)
    do_mpi_corr(comm, treecorr.NGNCorrelation, 2, ['ntri', 'zeta'], output)
    do_mpi_corr(comm, treecorr.NGNCorrelation, 2, ['ntri', 'zeta'], output, 'local')
    do_mpi_corr(comm, treecorr.NGNCorrelation, 2, ['ntri', 'zeta'], output, 'global', True)
    do_mpi_corr(comm, treecorr.NGNCorrelation, 2, ['ntri', 'zeta'], output, 'local', True)
    do_mpi_corr2(comm, treecorr.NGNCorrelation, 2, ['ntri', 'zeta'], output)
    do_mpi_corr(comm, treecorr.GNNCorrelation, 1, ['ntri', 'zeta'], output)
    do_mpi_corr(comm, treecorr.GNNCorrelation, 1, ['ntri', 'zeta'], output, 'local')
    do_mpi_corr(comm, treecorr.GNNCorrelation, 1, ['ntri', 'zeta'], output, 'global', True)
    do_mpi_corr(comm, treecorr.GNNCorrelation, 1, ['ntri', 'zeta'], output, 'local', True)
    do_mpi_corr2(comm, treecorr.GNNCorrelation, 1, ['ntri', 'zeta'], output)

def do_mpi_nkk(comm, output=True):
    do_mpi_corr(comm, treecorr.NKKCorrelation, 1, ['ntri', 'zeta'], output)
    do_mpi_corr(comm, treecorr.NKKCorrelation, 1, ['ntri', 'zeta'], output, 'local')
    do_mpi_corr(comm, treecorr.NKKCorrelation, 1, ['ntri', 'zeta'], output, 'global', True)
    do_mpi_corr(comm, treecorr.NKKCorrelation, 1, ['ntri', 'zeta'], output, 'local', True)
    do_mpi_corr2(comm, treecorr.NKKCorrelation, 1, ['ntri', 'zeta'], output)
    do_mpi_corr(comm, treecorr.KNKCorrelation, 2, ['ntri', 'zeta'], output)
    do_mpi_corr(comm, treecorr.KNKCorrelation, 2, ['ntri', 'zeta'], output, 'local')
    do_mpi_corr(comm, treecorr.KNKCorrelation, 2, ['ntri', 'zeta'], output, 'global', True)
    do_mpi_corr(comm, treecorr.KNKCorrelation, 2, ['ntri', 'zeta'], output, 'local', True)
    do_mpi_corr2(comm, treecorr.KNKCorrelation, 2, ['ntri', 'zeta'], output)
    do_mpi_corr(comm, treecorr.KKNCorrelation, 1, ['ntri', 'zeta'], output)
    do_mpi_corr(comm, treecorr.KKNCorrelation, 1, ['ntri', 'zeta'], output, 'local')
    do_mpi_corr(comm, treecorr.KKNCorrelation, 1, ['ntri', 'zeta'], output, 'global', True)
    do_mpi_corr(comm, treecorr.KKNCorrelation, 1, ['ntri', 'zeta'], output, 'local', True)
    do_mpi_corr2(comm, treecorr.KKNCorrelation, 1, ['ntri', 'zeta'], output)

def do_mpi_ngg(comm, output=True):
    do_mpi_corr(comm, treecorr.NGGCorrelation, 1, ['ntri', 'gam0', 'gam2'], output)
    do_mpi_corr(comm, treecorr.NGGCorrelation, 1, ['ntri', 'gam0', 'gam2'], output, 'local')
    do_mpi_corr(comm, treecorr.NGGCorrelation, 1, ['ntri', 'gam0', 'gam2'], output, 'global', True)
    do_mpi_corr(comm, treecorr.NGGCorrelation, 1, ['ntri', 'gam0', 'gam2'], output, 'local', True)
    do_mpi_corr2(comm, treecorr.NGGCorrelation, 1, ['ntri', 'gam0', 'gam2'], output)
    do_mpi_corr(comm, treecorr.GNGCorrelation, 2, ['ntri', 'gam0', 'gam1'], output)
    do_mpi_corr(comm, treecorr.GNGCorrelation, 2, ['ntri', 'gam0', 'gam1'], output, 'local')
    do_mpi_corr(comm, treecorr.GNGCorrelation, 2, ['ntri', 'gam0', 'gam1'], output, 'global', True)
    do_mpi_corr(comm, treecorr.GNGCorrelation, 2, ['ntri', 'gam0', 'gam1'], output, 'local', True)
    do_mpi_corr2(comm, treecorr.GNGCorrelation, 2, ['ntri', 'gam0', 'gam1'], output)
    do_mpi_corr(comm, treecorr.GGNCorrelation, 1, ['ntri', 'gam0', 'gam1'], output)
    do_mpi_corr(comm, treecorr.GGNCorrelation, 1, ['ntri', 'gam0', 'gam1'], output, 'local')
    do_mpi_corr(comm, treecorr.GGNCorrelation, 1, ['ntri', 'gam0', 'gam1'], output, 'global', True)
    do_mpi_corr(comm, treecorr.GGNCorrelation, 1, ['ntri', 'gam0', 'gam1'], output, 'local', True)
    do_mpi_corr2(comm, treecorr.GGNCorrelation, 1, ['ntri', 'gam0', 'gam1'], output)

def do_mpi_kkg(comm, output=True):
    do_mpi_corr(comm, treecorr.KKGCorrelation, 1, ['ntri', 'zeta'], output)
    do_mpi_corr(comm, treecorr.KKGCorrelation, 1, ['ntri', 'zeta'], output, 'local')
    do_mpi_corr(comm, treecorr.KKGCorrelation, 1, ['ntri', 'zeta'], output, 'global', True)
    do_mpi_corr(comm, treecorr.KKGCorrelation, 1, ['ntri', 'zeta'], output, 'local', True)
    do_mpi_corr2(comm, treecorr.KKGCorrelation, 1, ['ntri', 'zeta'], output)
    do_mpi_corr(comm, treecorr.KGKCorrelation, 2, ['ntri', 'zeta'], output)
    do_mpi_corr(comm, treecorr.KGKCorrelation, 2, ['ntri', 'zeta'], output, 'local')
    do_mpi_corr(comm, treecorr.KGKCorrelation, 2, ['ntri', 'zeta'], output, 'global', True)
    do_mpi_corr(comm, treecorr.KGKCorrelation, 2, ['ntri', 'zeta'], output, 'local', True)
    do_mpi_corr2(comm, treecorr.KGKCorrelation, 2, ['ntri', 'zeta'], output)
    do_mpi_corr(comm, treecorr.GKKCorrelation, 1, ['ntri', 'zeta'], output)
    do_mpi_corr(comm, treecorr.GKKCorrelation, 1, ['ntri', 'zeta'], output, 'local')
    do_mpi_corr(comm, treecorr.GKKCorrelation, 1, ['ntri', 'zeta'], output, 'global', True)
    do_mpi_corr(comm, treecorr.GKKCorrelation, 1, ['ntri', 'zeta'], output, 'local', True)
    do_mpi_corr2(comm, treecorr.GKKCorrelation, 1, ['ntri', 'zeta'], output)

def do_mpi_kgg(comm, output=True):
    do_mpi_corr(comm, treecorr.KGGCorrelation, 1, ['ntri', 'gam0', 'gam2'], output)
    do_mpi_corr(comm, treecorr.KGGCorrelation, 1, ['ntri', 'gam0', 'gam2'], output, 'local')
    do_mpi_corr(comm, treecorr.KGGCorrelation, 1, ['ntri', 'gam0', 'gam2'], output, 'global', True)
    do_mpi_corr(comm, treecorr.KGGCorrelation, 1, ['ntri', 'gam0', 'gam2'], output, 'local', True)
    do_mpi_corr2(comm, treecorr.KGGCorrelation, 1, ['ntri', 'gam0', 'gam2'], output)
    do_mpi_corr(comm, treecorr.GKGCorrelation, 2, ['ntri', 'gam0', 'gam1'], output)
    do_mpi_corr(comm, treecorr.GKGCorrelation, 2, ['ntri', 'gam0', 'gam1'], output, 'local')
    do_mpi_corr(comm, treecorr.GKGCorrelation, 2, ['ntri', 'gam0', 'gam1'], output, 'global', True)
    do_mpi_corr(comm, treecorr.GKGCorrelation, 2, ['ntri', 'gam0', 'gam1'], output, 'local', True)
    do_mpi_corr2(comm, treecorr.GKGCorrelation, 2, ['ntri', 'gam0', 'gam1'], output)
    do_mpi_corr(comm, treecorr.GGKCorrelation, 1, ['ntri', 'gam0', 'gam1'], output)
    do_mpi_corr(comm, treecorr.GGKCorrelation, 1, ['ntri', 'gam0', 'gam1'], output, 'local')
    do_mpi_corr(comm, treecorr.GGKCorrelation, 1, ['ntri', 'gam0', 'gam1'], output, 'global', True)
    do_mpi_corr(comm, treecorr.GGKCorrelation, 1, ['ntri', 'gam0', 'gam1'], output, 'local', True)
    do_mpi_corr2(comm, treecorr.GGKCorrelation, 1, ['ntri', 'gam0', 'gam1'], output)

if __name__ == '__main__':
    from mpi4py import MPI
    from mpi_helper import NiceComm
    comm = NiceComm(MPI.COMM_WORLD)
    rank = comm.Get_rank()
    if rank == 0:
        setup()
    comm.Barrier()
    do_mpi_ggg(comm)
    do_mpi_kkk(comm)
    do_mpi_kkk2(comm)
    do_mpi_nnk(comm)
    do_mpi_nng(comm)
    do_mpi_nkk(comm)
    do_mpi_ngg(comm)
    do_mpi_kkg(comm)
    do_mpi_kgg(comm)
