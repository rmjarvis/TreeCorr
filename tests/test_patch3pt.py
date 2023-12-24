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
import os
import coord
import time
import treecorr
try:
    import cPickle as pickle
except ImportError:
    import pickle

from test_helper import assert_raises, do_pickle, timer, get_from_wiki, CaptureLog, clear_save
from test_helper import profile

def generate_shear_field(npos, nhalo, rng=None):
    # We do something completely different here than we did for 2pt patch tests.
    # A straight Gaussian field with a given power spectrum has no significant 3pt power,
    # so it's not a great choice for simulating a field for 3pt tests.
    # Instead we place N SIS "halos" randomly in the grid.
    # Then we translate that to a shear field via FFT.

    if rng is None:
        rng = np.random.RandomState()

    # Generate x,y values for the real-space field
    x = rng.uniform(0,1000, size=npos)
    y = rng.uniform(0,1000, size=npos)

    nh = rng.poisson(nhalo)

    # Fill the kappa values with SIS halo profiles.
    xc = rng.uniform(0,1000, size=nh)
    yc = rng.uniform(0,1000, size=nh)
    scale = rng.uniform(20,50, size=nh)
    mass = rng.uniform(0.01, 0.05, size=nh)
    # Avoid making huge nhalo * nsource arrays.  Loop in blocks of 64 halos
    nblock = (nh-1) // 64 + 1
    kappa = np.zeros_like(x)
    gamma = np.zeros_like(x, dtype=complex)
    for iblock in range(nblock):
        i = iblock*64
        j = (iblock+1)*64
        dx = x[:,np.newaxis]-xc[np.newaxis,i:j]
        dy = y[:,np.newaxis]-yc[np.newaxis,i:j]
        dx[dx==0] = 1  # Avoid division by zero.
        dy[dy==0] = 1
        dx /= scale[i:j]
        dy /= scale[i:j]
        rsq = dx**2 + dy**2
        r = rsq**0.5
        k = mass[i:j] / r  # "Mass" here is really just a dimensionless normalization propto mass.
        kappa += np.sum(k, axis=1)

        # gamma_t = kappa for SIS.
        g = -k * (dx + 1j*dy)**2 / rsq
        gamma += np.sum(g, axis=1)

    return x, y, np.real(gamma), np.imag(gamma), kappa


@timer
def test_kkk_jk():
    # Test jackknife and other covariance estimates for kkk correlations.
    # Note: This test takes a while!
    # The main version I think is a pretty decent test of the code correctness.
    # It shows that bootstrap in particular easily gets to within 50% of the right variance.
    # Sometimes within 20%, but because of the randomness there, it varies a bit.
    # Jackknife isn't much worse.  Just a little below 50%.  But still pretty good.
    # Sample and Marked are not great for this test.  I think they will work ok when the
    # triangles of interest are mostly within single patches, but that's not the case we
    # have here, and it would take a lot more points to get to that regime.  So the
    # accuracy tests for those two are pretty loose.

    if __name__ == '__main__':
        # This setup takes about 740 sec to run.
        nhalo = 3000
        nsource = 5000
        npatch = 32
        tol_factor = 1
    elif False:
        # This setup takes about 180 sec to run.
        nhalo = 2000
        nsource = 2000
        npatch = 16
        tol_factor = 2
    elif False:
        # This setup takes about 51 sec to run.
        nhalo = 1000
        nsource = 1000
        npatch = 16
        tol_factor = 3
    else:
        # This setup takes about 20 sec to run.
        # So we use this one for regular unit test runs.
        # It's pretty terrible in terms of testing the accuracy, but it works for code coverage.
        # But whenever actually working on this part of the code, definitely need to switch
        # to one of the above setups.  Preferably run the name==main version to get a good
        # test of the code correctness.
        nhalo = 500
        nsource = 500
        npatch = 16
        tol_factor = 4

    file_name = 'data/test_kkk_jk_{}.npz'.format(nsource)
    print(file_name)
    if not os.path.isfile(file_name):
        nruns = 1000
        all_kkks = []
        rng1 = np.random.RandomState()
        for run in range(nruns):
            x, y, _, _, k = generate_shear_field(nsource, nhalo, rng1)
            print(run,': ',np.mean(k),np.std(k))
            cat = treecorr.Catalog(x=x, y=y, k=k)
            kkk = treecorr.KKKCorrelation(nbins=3, min_sep=30., max_sep=100.,
                                           min_u=0.9, max_u=1.0, nubins=1,
                                           min_v=0.0, max_v=0.1, nvbins=1)
            kkk.process(cat)
            print(kkk.ntri.ravel().tolist())
            print(kkk.zeta.ravel().tolist())
            all_kkks.append(kkk)
        mean_kkk = np.mean([kkk.zeta.ravel() for kkk in all_kkks], axis=0)
        var_kkk = np.var([kkk.zeta.ravel() for kkk in all_kkks], axis=0)

        np.savez(file_name, all_kkk=np.array([kkk.zeta.ravel() for kkk in all_kkks]),
                 mean_kkk=mean_kkk, var_kkk=var_kkk)

    data = np.load(file_name)
    mean_kkk = data['mean_kkk']
    var_kkk = data['var_kkk']
    print('mean = ',mean_kkk)
    print('var = ',var_kkk)

    rng = np.random.RandomState(12345)
    x, y, _, _, k = generate_shear_field(nsource, nhalo, rng)
    cat = treecorr.Catalog(x=x, y=y, k=k)
    kkk = treecorr.KKKCorrelation(nbins=3, min_sep=30., max_sep=100.,
                                  min_u=0.9, max_u=1.0, nubins=1,
                                  min_v=0.0, max_v=0.1, nvbins=1, rng=rng)
    kkk.process(cat)
    print(kkk.ntri.ravel())
    print(kkk.zeta.ravel())
    print(kkk.varzeta.ravel())

    kkkp = kkk.copy()
    catp = treecorr.Catalog(x=x, y=y, k=k, npatch=npatch)

    # Do the same thing with patches.
    kkkp.process(catp)
    print('with patches:')
    print(kkkp.ntri.ravel())
    print(kkkp.zeta.ravel())
    print(kkkp.varzeta.ravel())

    np.testing.assert_allclose(kkkp.ntri, kkk.ntri, rtol=0.05 * tol_factor)
    np.testing.assert_allclose(kkkp.zeta, kkk.zeta, rtol=0.1 * tol_factor, atol=1e-3 * tol_factor)
    np.testing.assert_allclose(kkkp.varzeta, kkk.varzeta, rtol=0.05 * tol_factor, atol=3.e-6)

    print('jackknife:')
    cov = kkkp.estimate_cov('jackknife')
    print(np.diagonal(cov))
    print('max log(ratio) = ',np.max(np.abs(np.log(np.diagonal(cov))-np.log(var_kkk))))
    np.testing.assert_allclose(np.diagonal(cov), var_kkk, rtol=0.6 * tol_factor)
    np.testing.assert_allclose(np.log(np.diagonal(cov)), np.log(var_kkk), atol=0.5*tol_factor)

    # Test design matrix
    A, w = kkkp.build_cov_design_matrix('jackknife')
    A -= np.mean(A, axis=0)
    C = (1-1/npatch) * A.conj().T.dot(A)
    np.testing.assert_allclose(C, cov)

    with assert_raises(ValueError):
        kkkp.build_cov_design_matrix('shot')
    with assert_raises(ValueError):
        kkkp.build_cov_design_matrix('invalid')

    print('sample:')
    cov = kkkp.estimate_cov('sample')
    print(np.diagonal(cov))
    print('max log(ratio) = ',np.max(np.abs(np.log(np.diagonal(cov))-np.log(var_kkk))))
    np.testing.assert_allclose(np.diagonal(cov), var_kkk, rtol=0.7 * tol_factor)
    np.testing.assert_allclose(np.log(np.diagonal(cov)), np.log(var_kkk), atol=0.7*tol_factor)

    # Test design matrix
    A, w = kkkp.build_cov_design_matrix('sample')
    w /= np.sum(w)
    A -= np.mean(A, axis=0)
    C = 1/(npatch-1) * (w*A.conj().T).dot(A)
    np.testing.assert_allclose(C, cov)

    print('marked:')
    rng_state = kkkp.rng.get_state()
    cov = kkkp.estimate_cov('marked_bootstrap')
    print(np.diagonal(cov))
    print('max log(ratio) = ',np.max(np.abs(np.log(np.diagonal(cov))-np.log(var_kkk))))
    np.testing.assert_allclose(np.diagonal(cov), var_kkk, rtol=0.7 * tol_factor)
    np.testing.assert_allclose(np.log(np.diagonal(cov)), np.log(var_kkk), atol=0.7*tol_factor)

    # Test design matrix
    kkkp.rng.set_state(rng_state)
    A, w = kkkp.build_cov_design_matrix('marked_bootstrap')
    nboot = A.shape[0]
    A -= np.mean(A, axis=0)
    C = 1/(nboot-1) * A.conj().T.dot(A)
    np.testing.assert_allclose(C, cov)

    print('bootstrap:')
    rng_state = kkkp.rng.get_state()
    cov = kkkp.estimate_cov('bootstrap')
    print(np.diagonal(cov))
    print('max log(ratio) = ',np.max(np.abs(np.log(np.diagonal(cov))-np.log(var_kkk))))
    np.testing.assert_allclose(np.diagonal(cov), var_kkk, rtol=0.5 * tol_factor)
    np.testing.assert_allclose(np.log(np.diagonal(cov)), np.log(var_kkk), atol=0.3*tol_factor)

    # Test design matrix
    kkkp.rng.set_state(rng_state)
    A, w = kkkp.build_cov_design_matrix('bootstrap')
    nboot = A.shape[0]
    A -= np.mean(A, axis=0)
    C = 1/(nboot-1) * A.conj().T.dot(A)
    np.testing.assert_allclose(C, cov)

    # Check that these still work after roundtripping through a file.
    cov1 = kkkp.estimate_cov('jackknife')
    file_name = os.path.join('output','test_write_results_kkk.pkl')
    with open(file_name, 'wb') as f:
        pickle.dump(kkkp, f)
    with open(file_name, 'rb') as f:
        kkk2 = pickle.load(f)
    cov2 = kkk2.estimate_cov('jackknife')
    np.testing.assert_allclose(cov2, cov1)

    # And again using the normal write command.
    try:
        import fitsio
    except ImportError:
        pass
    else:
        file_name = os.path.join('output','test_write_results_kkk.fits')
        kkkp.write(file_name, write_patch_results=True)
        kkk3 = treecorr.KKKCorrelation(nbins=3, min_sep=30., max_sep=100.,
                                       min_u=0.9, max_u=1.0, nubins=1,
                                       min_v=0.0, max_v=0.1, nvbins=1, rng=rng)
        kkk3.read(file_name)
        cov3 = kkk3.estimate_cov('jackknife')
        np.testing.assert_allclose(cov3, cov1)

    # Also with ascii, since that works differeny.
    file_name = os.path.join('output','test_write_results_kkk.dat')
    kkkp.write(file_name, write_patch_results=True)
    kkk4 = treecorr.KKKCorrelation(nbins=3, min_sep=30., max_sep=100.,
                                   min_u=0.9, max_u=1.0, nubins=1,
                                   min_v=0.0, max_v=0.1, nvbins=1, rng=rng)
    kkk4.read(file_name)
    cov4 = kkk4.estimate_cov('jackknife')
    np.testing.assert_allclose(cov4, cov1)

    # And also try to match the type if HDF
    try:
        import h5py
    except ImportError:
        print('Skipping saving HDF patches, since h5py not installed.')
        h5py = None

    if h5py is not None:
        # Finally with hdf
        file_name = os.path.join('output','test_write_results_kkk.hdf5')
        kkkp.write(file_name, write_patch_results=True)
        kkk5 = treecorr.KKKCorrelation(nbins=3, min_sep=30., max_sep=100.,
                                       min_u=0.9, max_u=1.0, nubins=1,
                                       min_v=0.0, max_v=0.1, nvbins=1, rng=rng)
        kkk5.read(file_name)
        cov5 = kkk5.estimate_cov('jackknife')
        np.testing.assert_allclose(cov5, cov1)

    # Now as a cross correlation with all 3 using the same patch catalog.
    print('with 3 patched catalogs:')
    kkkp.process(catp, catp, catp)
    print(kkkp.zeta.ravel())
    np.testing.assert_allclose(kkkp.zeta, kkk.zeta, rtol=0.1 * tol_factor, atol=1e-3 * tol_factor)

    print('jackknife:')
    cov = kkkp.estimate_cov('jackknife')
    print(np.diagonal(cov))
    print('max log(ratio) = ',np.max(np.abs(np.log(np.diagonal(cov))-np.log(var_kkk))))
    np.testing.assert_allclose(np.log(np.diagonal(cov)), np.log(var_kkk), atol=0.5*tol_factor)

    print('sample:')
    cov = kkkp.estimate_cov('sample')
    print(np.diagonal(cov))
    print('max log(ratio) = ',np.max(np.abs(np.log(np.diagonal(cov))-np.log(var_kkk))))
    np.testing.assert_allclose(np.log(np.diagonal(cov)), np.log(var_kkk), atol=0.8*tol_factor)

    print('marked:')
    cov = kkkp.estimate_cov('marked_bootstrap')
    print(np.diagonal(cov))
    print('max log(ratio) = ',np.max(np.abs(np.log(np.diagonal(cov))-np.log(var_kkk))))
    np.testing.assert_allclose(np.log(np.diagonal(cov)), np.log(var_kkk), atol=0.8*tol_factor)

    print('bootstrap:')
    cov = kkkp.estimate_cov('bootstrap')
    print(np.diagonal(cov))
    print('max log(ratio) = ',np.max(np.abs(np.log(np.diagonal(cov))-np.log(var_kkk))))
    np.testing.assert_allclose(np.log(np.diagonal(cov)), np.log(var_kkk), atol=0.3*tol_factor)

    # Repeat this test with different combinations of patch with non-patch catalogs:
    # All the methods work best when the patches are used for all 3 catalogs.  But there
    # are probably cases where this kind of cross correlation with only some catalogs having
    # patches could be desired.  So this mostly just checks that the code runs properly.

    # Patch on 1 only:
    print('with patches on 1 only:')
    kkkp.process(catp, cat)
    print(kkkp.zeta.ravel())
    np.testing.assert_allclose(kkkp.zeta, kkk.zeta, rtol=0.1 * tol_factor, atol=1e-3 * tol_factor)

    print('jackknife:')
    cov = kkkp.estimate_cov('jackknife')
    print(np.diagonal(cov))
    print('max log(ratio) = ',np.max(np.abs(np.log(np.diagonal(cov))-np.log(var_kkk))))
    np.testing.assert_allclose(np.log(np.diagonal(cov)), np.log(var_kkk), atol=0.8*tol_factor)

    print('sample:')
    cov = kkkp.estimate_cov('sample')
    print(np.diagonal(cov))
    print('max log(ratio) = ',np.max(np.abs(np.log(np.diagonal(cov))-np.log(var_kkk))))
    np.testing.assert_allclose(np.log(np.diagonal(cov)), np.log(var_kkk), atol=0.8*tol_factor)

    print('marked:')
    cov = kkkp.estimate_cov('marked_bootstrap')
    print(np.diagonal(cov))
    print('max log(ratio) = ',np.max(np.abs(np.log(np.diagonal(cov))-np.log(var_kkk))))
    np.testing.assert_allclose(np.log(np.diagonal(cov)), np.log(var_kkk), atol=0.8*tol_factor)

    print('bootstrap:')
    cov = kkkp.estimate_cov('bootstrap')
    print(np.diagonal(cov))
    print('max log(ratio) = ',np.max(np.abs(np.log(np.diagonal(cov))-np.log(var_kkk))))
    np.testing.assert_allclose(np.log(np.diagonal(cov)), np.log(var_kkk), atol=0.8*tol_factor)

    # Patch on 2 only:
    print('with patches on 2 only:')
    kkkp.process(cat, catp, cat)
    print(kkkp.zeta.ravel())
    np.testing.assert_allclose(kkkp.zeta, kkk.zeta, rtol=0.1 * tol_factor, atol=1e-3 * tol_factor)

    print('jackknife:')
    cov = kkkp.estimate_cov('jackknife')
    print(np.diagonal(cov))
    print('max log(ratio) = ',np.max(np.abs(np.log(np.diagonal(cov))-np.log(var_kkk))))
    np.testing.assert_allclose(np.diagonal(cov), var_kkk, rtol=0.9 * tol_factor)
    np.testing.assert_allclose(np.log(np.diagonal(cov)), np.log(var_kkk), atol=0.8*tol_factor)

    print('sample:')
    cov = kkkp.estimate_cov('sample')
    print(np.diagonal(cov))
    print('max log(ratio) = ',np.max(np.abs(np.log(np.diagonal(cov))-np.log(var_kkk))))
    np.testing.assert_allclose(np.log(np.diagonal(cov)), np.log(var_kkk), atol=0.8*tol_factor)

    print('marked:')
    cov = kkkp.estimate_cov('marked_bootstrap')
    print(np.diagonal(cov))
    print('max log(ratio) = ',np.max(np.abs(np.log(np.diagonal(cov))-np.log(var_kkk))))
    np.testing.assert_allclose(np.log(np.diagonal(cov)), np.log(var_kkk), atol=0.8*tol_factor)

    print('bootstrap:')
    cov = kkkp.estimate_cov('bootstrap')
    print(np.diagonal(cov))
    print('max log(ratio) = ',np.max(np.abs(np.log(np.diagonal(cov))-np.log(var_kkk))))
    np.testing.assert_allclose(np.log(np.diagonal(cov)), np.log(var_kkk), atol=0.8*tol_factor)

    # Patch on 3 only:
    print('with patches on 3 only:')
    kkkp.process(cat, cat, catp)
    print(kkkp.zeta.ravel())
    np.testing.assert_allclose(kkkp.zeta, kkk.zeta, rtol=0.1 * tol_factor, atol=1e-3 * tol_factor)

    print('jackknife:')
    cov = kkkp.estimate_cov('jackknife')
    print(np.diagonal(cov))
    print('max log(ratio) = ',np.max(np.abs(np.log(np.diagonal(cov))-np.log(var_kkk))))
    np.testing.assert_allclose(np.log(np.diagonal(cov)), np.log(var_kkk), atol=0.8*tol_factor)

    print('sample:')
    cov = kkkp.estimate_cov('sample')
    print(np.diagonal(cov))
    print('max log(ratio) = ',np.max(np.abs(np.log(np.diagonal(cov))-np.log(var_kkk))))
    np.testing.assert_allclose(np.log(np.diagonal(cov)), np.log(var_kkk), atol=0.8*tol_factor)

    print('marked:')
    cov = kkkp.estimate_cov('marked_bootstrap')
    print(np.diagonal(cov))
    print('max log(ratio) = ',np.max(np.abs(np.log(np.diagonal(cov))-np.log(var_kkk))))
    np.testing.assert_allclose(np.log(np.diagonal(cov)), np.log(var_kkk), atol=0.8*tol_factor)

    print('bootstrap:')
    cov = kkkp.estimate_cov('bootstrap')
    print(np.diagonal(cov))
    print('max log(ratio) = ',np.max(np.abs(np.log(np.diagonal(cov))-np.log(var_kkk))))
    np.testing.assert_allclose(np.log(np.diagonal(cov)), np.log(var_kkk), atol=0.8*tol_factor)

    # Patch on 1,2
    print('with patches on 1,2:')
    kkkp.process(catp, catp, cat)
    print(kkkp.zeta.ravel())
    np.testing.assert_allclose(kkkp.zeta, kkk.zeta, rtol=0.1 * tol_factor, atol=1e-3 * tol_factor)

    print('jackknife:')
    cov = kkkp.estimate_cov('jackknife')
    print(np.diagonal(cov))
    print('max log(ratio) = ',np.max(np.abs(np.log(np.diagonal(cov))-np.log(var_kkk))))
    np.testing.assert_allclose(np.log(np.diagonal(cov)), np.log(var_kkk), atol=0.3*tol_factor)

    print('sample:')
    cov = kkkp.estimate_cov('sample')
    print(np.diagonal(cov))
    print('max log(ratio) = ',np.max(np.abs(np.log(np.diagonal(cov))-np.log(var_kkk))))
    np.testing.assert_allclose(np.log(np.diagonal(cov)), np.log(var_kkk), atol=0.8*tol_factor)

    print('marked:')
    cov = kkkp.estimate_cov('marked_bootstrap')
    print(np.diagonal(cov))
    print('max log(ratio) = ',np.max(np.abs(np.log(np.diagonal(cov))-np.log(var_kkk))))
    np.testing.assert_allclose(np.log(np.diagonal(cov)), np.log(var_kkk), atol=0.8*tol_factor)

    print('bootstrap:')
    cov = kkkp.estimate_cov('bootstrap')
    print(np.diagonal(cov))
    print('max log(ratio) = ',np.max(np.abs(np.log(np.diagonal(cov))-np.log(var_kkk))))
    np.testing.assert_allclose(np.log(np.diagonal(cov)), np.log(var_kkk), atol=0.4*tol_factor)

    # Patch on 2,3
    print('with patches on 2,3:')
    kkkp.process(cat, catp)
    print(kkkp.zeta.ravel())
    np.testing.assert_allclose(kkkp.zeta, kkk.zeta, rtol=0.1 * tol_factor, atol=1e-3 * tol_factor)

    print('jackknife:')
    cov = kkkp.estimate_cov('jackknife')
    print(np.diagonal(cov))
    print('max log(ratio) = ',np.max(np.abs(np.log(np.diagonal(cov))-np.log(var_kkk))))
    np.testing.assert_allclose(np.log(np.diagonal(cov)), np.log(var_kkk), atol=0.3*tol_factor)

    print('sample:')
    cov = kkkp.estimate_cov('sample')
    print(np.diagonal(cov))
    print('max log(ratio) = ',np.max(np.abs(np.log(np.diagonal(cov))-np.log(var_kkk))))
    np.testing.assert_allclose(np.log(np.diagonal(cov)), np.log(var_kkk), atol=0.7*tol_factor)

    print('marked:')
    cov = kkkp.estimate_cov('marked_bootstrap')
    print(np.diagonal(cov))
    print('max log(ratio) = ',np.max(np.abs(np.log(np.diagonal(cov))-np.log(var_kkk))))
    np.testing.assert_allclose(np.log(np.diagonal(cov)), np.log(var_kkk), atol=0.8*tol_factor)

    print('bootstrap:')
    cov = kkkp.estimate_cov('bootstrap')
    print(np.diagonal(cov))
    print('max log(ratio) = ',np.max(np.abs(np.log(np.diagonal(cov))-np.log(var_kkk))))
    np.testing.assert_allclose(np.log(np.diagonal(cov)), np.log(var_kkk), atol=0.3*tol_factor)

    # Patch on 1,3
    print('with patches on 1,3:')
    kkkp.process(catp, cat, catp)
    print(kkkp.zeta.ravel())
    np.testing.assert_allclose(kkkp.zeta, kkk.zeta, rtol=0.1 * tol_factor, atol=1e-3 * tol_factor)

    print('jackknife:')
    cov = kkkp.estimate_cov('jackknife')
    print(np.diagonal(cov))
    print('max log(ratio) = ',np.max(np.abs(np.log(np.diagonal(cov))-np.log(var_kkk))))
    np.testing.assert_allclose(np.log(np.diagonal(cov)), np.log(var_kkk), atol=0.3*tol_factor)

    print('sample:')
    cov = kkkp.estimate_cov('sample')
    print(np.diagonal(cov))
    print('max log(ratio) = ',np.max(np.abs(np.log(np.diagonal(cov))-np.log(var_kkk))))
    np.testing.assert_allclose(np.log(np.diagonal(cov)), np.log(var_kkk), atol=0.8*tol_factor)

    print('marked:')
    cov = kkkp.estimate_cov('marked_bootstrap')
    print(np.diagonal(cov))
    print('max log(ratio) = ',np.max(np.abs(np.log(np.diagonal(cov))-np.log(var_kkk))))
    np.testing.assert_allclose(np.log(np.diagonal(cov)), np.log(var_kkk), atol=0.8*tol_factor)

    print('bootstrap:')
    cov = kkkp.estimate_cov('bootstrap')
    print(np.diagonal(cov))
    print('max log(ratio) = ',np.max(np.abs(np.log(np.diagonal(cov))-np.log(var_kkk))))
    np.testing.assert_allclose(np.log(np.diagonal(cov)), np.log(var_kkk), atol=0.3*tol_factor)

    # Finally a set (with all patches) using the KKKCrossCorrelation class.
    kkkc = treecorr.KKKCrossCorrelation(nbins=3, min_sep=30., max_sep=100.,
                                        min_u=0.9, max_u=1.0, nubins=1,
                                        min_v=0.0, max_v=0.1, nvbins=1, rng=rng)
    print('CrossCorrelation:')
    kkkc.process(catp, catp, catp)
    for k1 in kkkc._all:
        print(k1.ntri.ravel())
        print(k1.zeta.ravel())
        print(k1.varzeta.ravel())

        np.testing.assert_allclose(k1.ntri, kkk.ntri, rtol=0.05 * tol_factor)
        np.testing.assert_allclose(k1.zeta, kkk.zeta, rtol=0.1 * tol_factor, atol=1e-3 * tol_factor)
        np.testing.assert_allclose(k1.varzeta, kkk.varzeta, rtol=0.05 * tol_factor, atol=3.e-6)

    print('jackknife:')
    cov = kkkc.estimate_cov('jackknife')
    print(np.diagonal(cov))
    for i in range(6):
        v = np.diagonal(cov)[i*6:(i+1)*6]
        print('max log(ratio) = ',np.max(np.abs(np.log(v)-np.log(var_kkk))))
        np.testing.assert_allclose(np.log(v), np.log(var_kkk), atol=0.5*tol_factor)

    print('sample:')
    cov = kkkc.estimate_cov('sample')
    print(np.diagonal(cov))
    for i in range(6):
        v = np.diagonal(cov)[i*6:(i+1)*6]
        print('max log(ratio) = ',np.max(np.abs(np.log(v)-np.log(var_kkk))))
        np.testing.assert_allclose(np.log(v), np.log(var_kkk), atol=0.8*tol_factor)

    print('marked:')
    cov = kkkc.estimate_cov('marked_bootstrap')
    print(np.diagonal(cov))
    for i in range(6):
        v = np.diagonal(cov)[i*6:(i+1)*6]
        print('max log(ratio) = ',np.max(np.abs(np.log(v)-np.log(var_kkk))))
        np.testing.assert_allclose(np.log(v), np.log(var_kkk), atol=0.8*tol_factor)

    print('bootstrap:')
    cov = kkkc.estimate_cov('bootstrap')
    print(np.diagonal(cov))
    for i in range(6):
        v = np.diagonal(cov)[i*6:(i+1)*6]
        print('max log(ratio) = ',np.max(np.abs(np.log(v)-np.log(var_kkk))))
        np.testing.assert_allclose(np.log(v), np.log(var_kkk), atol=0.5*tol_factor)

    # All catalogs need to have the same number of patches
    catq = treecorr.Catalog(x=x, y=y, k=k, npatch=2*npatch)
    with assert_raises(RuntimeError):
        kkkp.process(catp, catq)
    with assert_raises(RuntimeError):
        kkkp.process(catp, catq, catq)
    with assert_raises(RuntimeError):
        kkkp.process(catq, catp, catq)
    with assert_raises(RuntimeError):
        kkkp.process(catq, catq, catp)

@timer
def test_ggg_jk():
    # Test jackknife and other covariance estimates for ggg correlations.

    if __name__ == '__main__':
        # This setup takes about 590 sec to run.
        nhalo = 5000
        nsource = 5000
        npatch = 32
        tol_factor = 1
    elif False:
        # This setup takes about 160 sec to run.
        nhalo = 2000
        nsource = 2000
        npatch = 16
        tol_factor = 2
    elif False:
        # This setup takes about 50 sec to run.
        nhalo = 1000
        nsource = 1000
        npatch = 16
        tol_factor = 3
    else:
        # This setup takes about 13 sec to run.
        nhalo = 500
        nsource = 500
        npatch = 8
        tol_factor = 3

    # I couldn't figure out a way to get reasonable S/N in the shear field.  I thought doing
    # discrete halos would give some significant 3pt shear pattern, at least for equilateral
    # triangles, but the signal here is still consistent with zero.  :(
    # The point is the variance, which is still calculated ok, but I would have rathered
    # have something with S/N > 0.

    # For these tests, I set up the binning to just accumulate all roughly equilateral triangles
    # in a small separation range.  The binning always uses two bins for each to get + and - v
    # bins.  So this function averages these two values to produce 1 value for each gamma.
    f = lambda g: np.array([np.mean(g.gam0), np.mean(g.gam1), np.mean(g.gam2), np.mean(g.gam3)])

    file_name = 'data/test_ggg_jk_{}.npz'.format(nsource)
    print(file_name)
    if not os.path.isfile(file_name):
        nruns = 1000
        all_gggs = []
        rng1 = np.random.RandomState()
        for run in range(nruns):
            x, y, g1, g2, _ = generate_shear_field(nsource, nhalo, rng1)
            # For some reason std(g2) is coming out about 1.5x larger than std(g1).
            # Probably a sign of some error in the generate function, but I don't see it.
            # For this purpose I think it doesn't really matter, but it's a bit odd.
            print(run,': ',np.mean(g1),np.std(g1),np.mean(g2),np.std(g2))
            cat = treecorr.Catalog(x=x, y=y, g1=g1, g2=g2)
            ggg = treecorr.GGGCorrelation(nbins=1, min_sep=20., max_sep=40.,
                                           min_u=0.6, max_u=1.0, nubins=1,
                                           min_v=0.0, max_v=0.6, nvbins=1)
            ggg.process(cat)
            print(ggg.ntri.ravel())
            print(f(ggg))
            all_gggs.append(ggg)
        all_ggg = np.array([f(ggg) for ggg in all_gggs])
        mean_ggg = np.mean(all_ggg, axis=0)
        var_ggg = np.var(all_ggg, axis=0)
        np.savez(file_name, mean_ggg=mean_ggg, var_ggg=var_ggg)

    data = np.load(file_name)
    mean_ggg = data['mean_ggg']
    var_ggg = data['var_ggg']
    print('mean = ',mean_ggg)
    print('var = ',var_ggg)

    rng = np.random.RandomState(12345)
    x, y, g1, g2, _ = generate_shear_field(nsource, nhalo, rng)
    cat = treecorr.Catalog(x=x, y=y, g1=g1, g2=g2)
    ggg = treecorr.GGGCorrelation(nbins=1, min_sep=20., max_sep=40.,
                                  min_u=0.6, max_u=1.0, nubins=1,
                                  min_v=0.0, max_v=0.6, nvbins=1, rng=rng)
    ggg.process(cat)
    print(ggg.ntri.ravel())
    print(ggg.gam0.ravel())
    print(ggg.gam1.ravel())
    print(ggg.gam2.ravel())
    print(ggg.gam3.ravel())

    gggp = ggg.copy()
    catp = treecorr.Catalog(x=x, y=y, g1=g1, g2=g2, npatch=npatch)

    # Do the same thing with patches.
    gggp.process(catp)
    print('with patches:')
    print(gggp.ntri.ravel())
    print(gggp.vargam0.ravel())
    print(gggp.vargam1.ravel())
    print(gggp.vargam2.ravel())
    print(gggp.vargam3.ravel())
    print(gggp.gam0.ravel())
    print(gggp.gam1.ravel())
    print(gggp.gam2.ravel())
    print(gggp.gam3.ravel())

    np.testing.assert_allclose(gggp.ntri, ggg.ntri, rtol=0.05 * tol_factor)
    np.testing.assert_allclose(gggp.gam0, ggg.gam0, rtol=0.3 * tol_factor, atol=0.3 * tol_factor)
    np.testing.assert_allclose(gggp.gam1, ggg.gam1, rtol=0.3 * tol_factor, atol=0.3 * tol_factor)
    np.testing.assert_allclose(gggp.gam2, ggg.gam2, rtol=0.3 * tol_factor, atol=0.3 * tol_factor)
    np.testing.assert_allclose(gggp.gam3, ggg.gam3, rtol=0.3 * tol_factor, atol=0.3 * tol_factor)
    np.testing.assert_allclose(gggp.vargam0, ggg.vargam0, rtol=0.1 * tol_factor)
    np.testing.assert_allclose(gggp.vargam1, ggg.vargam1, rtol=0.1 * tol_factor)
    np.testing.assert_allclose(gggp.vargam2, ggg.vargam2, rtol=0.1 * tol_factor)
    np.testing.assert_allclose(gggp.vargam3, ggg.vargam3, rtol=0.1 * tol_factor)

    print('jackknife:')
    cov = gggp.estimate_cov('jackknife', func=f)
    print(np.diagonal(cov).real)
    print('max log(ratio) = ',np.max(np.abs(np.log(np.diagonal(cov))-np.log(var_ggg))))
    np.testing.assert_allclose(np.log(np.diagonal(cov)), np.log(var_ggg), atol=0.4*tol_factor)

    # Test design matrix
    A, w = gggp.build_cov_design_matrix('jackknife', func=f)
    A -= np.mean(A, axis=0)
    C = (1-1/npatch) * A.conj().T.dot(A)
    np.testing.assert_allclose(C, cov)

    print('sample:')
    cov = gggp.estimate_cov('sample', func=f)
    print(np.diagonal(cov).real)
    print('max log(ratio) = ',np.max(np.abs(np.log(np.diagonal(cov))-np.log(var_ggg))))
    np.testing.assert_allclose(np.log(np.diagonal(cov)), np.log(var_ggg), atol=0.8*tol_factor)

    print('marked:')
    cov = gggp.estimate_cov('marked_bootstrap', func=f)
    print(np.diagonal(cov).real)
    print('max log(ratio) = ',np.max(np.abs(np.log(np.diagonal(cov))-np.log(var_ggg))))
    np.testing.assert_allclose(np.log(np.diagonal(cov)), np.log(var_ggg), atol=0.9*tol_factor)

    print('bootstrap:')
    cov = gggp.estimate_cov('bootstrap', func=f)
    print(np.diagonal(cov).real)
    print('max log(ratio) = ',np.max(np.abs(np.log(np.diagonal(cov))-np.log(var_ggg))))
    np.testing.assert_allclose(np.log(np.diagonal(cov)), np.log(var_ggg), atol=0.3*tol_factor)

    # Check that these still work after roundtripping through a file.
    cov1 = gggp.estimate_cov('jackknife', func=f)
    file_name = os.path.join('output','test_write_results_ggg.pkl')
    with open(file_name, 'wb') as fid:
        pickle.dump(gggp, fid)
    with open(file_name, 'rb') as fid:
        ggg2 = pickle.load(fid)
    cov2 = ggg2.estimate_cov('jackknife', func=f)
    print('cov1 = ',cov1)
    print('cov2 = ',cov2)
    np.testing.assert_allclose(cov2, cov1)

    # And again using the normal write command.
    try:
        import fitsio
    except ImportError:
        pass
    else:
        file_name = os.path.join('output','test_write_results_ggg.fits')
        gggp.write(file_name, write_patch_results=True)
        ggg3 = treecorr.GGGCorrelation(nbins=1, min_sep=20., max_sep=40.,
                                       min_u=0.6, max_u=1.0, nubins=1,
                                       min_v=0.0, max_v=0.6, nvbins=1)
        ggg3.read(file_name)
        cov3 = ggg3.estimate_cov('jackknife', func=f)
        print('cov3 = ',cov3)
        np.testing.assert_allclose(cov3, cov1)

    # Also with ascii, since that works differeny.
    file_name = os.path.join('output','test_write_results_ggg.dat')
    gggp.write(file_name, write_patch_results=True)
    ggg4 = treecorr.GGGCorrelation(nbins=1, min_sep=20., max_sep=40.,
                                   min_u=0.6, max_u=1.0, nubins=1,
                                   min_v=0.0, max_v=0.6, nvbins=1)
    ggg4.read(file_name)
    cov4 = ggg4.estimate_cov('jackknife', func=f)
    np.testing.assert_allclose(cov4, cov1)

    # And also try to match the type if HDF
    try:
        import h5py
    except ImportError:
        print('Skipping saving HDF patches, since h5py not installed.')
        h5py = None

    if h5py is not None:
        # Finally with hdf
        file_name = os.path.join('output','test_write_results_ggg.hdf5')
        gggp.write(file_name, write_patch_results=True)
        ggg5 = treecorr.GGGCorrelation(nbins=1, min_sep=20., max_sep=40.,
                                       min_u=0.6, max_u=1.0, nubins=1,
                                       min_v=0.0, max_v=0.6, nvbins=1)
        ggg5.read(file_name)
        cov5 = ggg5.estimate_cov('jackknife', func=f)
        np.testing.assert_allclose(cov5, cov1)

    # Now as a cross correlation with all 3 using the same patch catalog.
    print('with 3 patched catalogs:')
    gggp.process(catp, catp, catp)
    print(gggp.gam0.ravel())
    np.testing.assert_allclose(gggp.gam0, ggg.gam0, rtol=0.3 * tol_factor, atol=0.3 * tol_factor)
    np.testing.assert_allclose(gggp.gam1, ggg.gam1, rtol=0.3 * tol_factor, atol=0.3 * tol_factor)
    np.testing.assert_allclose(gggp.gam2, ggg.gam2, rtol=0.3 * tol_factor, atol=0.3 * tol_factor)
    np.testing.assert_allclose(gggp.gam3, ggg.gam3, rtol=0.3 * tol_factor, atol=0.3 * tol_factor)

    print('jackknife:')
    cov = gggp.estimate_cov('jackknife', func=f)
    print(np.diagonal(cov).real)
    print('max log(ratio) = ',np.max(np.abs(np.log(np.diagonal(cov))-np.log(var_ggg))))
    np.testing.assert_allclose(np.log(np.diagonal(cov)), np.log(var_ggg), atol=0.4*tol_factor)

    print('sample:')
    cov = gggp.estimate_cov('sample', func=f)
    print(np.diagonal(cov).real)
    print('max log(ratio) = ',np.max(np.abs(np.log(np.diagonal(cov))-np.log(var_ggg))))
    np.testing.assert_allclose(np.log(np.diagonal(cov)), np.log(var_ggg), atol=0.6*tol_factor)

    print('marked:')
    cov = gggp.estimate_cov('marked_bootstrap', func=f)
    print(np.diagonal(cov).real)
    print('max log(ratio) = ',np.max(np.abs(np.log(np.diagonal(cov))-np.log(var_ggg))))
    np.testing.assert_allclose(np.log(np.diagonal(cov)), np.log(var_ggg), atol=0.8*tol_factor)

    print('bootstrap:')
    cov = gggp.estimate_cov('bootstrap', func=f)
    print(np.diagonal(cov).real)
    print('max log(ratio) = ',np.max(np.abs(np.log(np.diagonal(cov))-np.log(var_ggg))))
    np.testing.assert_allclose(np.log(np.diagonal(cov)), np.log(var_ggg), atol=0.4*tol_factor)

    # The separate patch/non-patch combinations aren't that interesting, so skip them
    # for GGG unless running from main.
    if __name__ == '__main__':
        # Patch on 1 only:
        print('with patches on 1 only:')
        gggp.process(catp, cat)

        print('jackknife:')
        cov = gggp.estimate_cov('jackknife', func=f)
        print(np.diagonal(cov).real)
        print('max log(ratio) = ',np.max(np.abs(np.log(np.diagonal(cov))-np.log(var_ggg))))
        np.testing.assert_allclose(np.log(np.diagonal(cov)), np.log(var_ggg), atol=0.8*tol_factor)

        print('sample:')
        cov = gggp.estimate_cov('sample', func=f)
        print(np.diagonal(cov).real)
        print('max log(ratio) = ',np.max(np.abs(np.log(np.diagonal(cov))-np.log(var_ggg))))
        np.testing.assert_allclose(np.log(np.diagonal(cov)), np.log(var_ggg), atol=0.7*tol_factor)

        print('marked:')
        cov = gggp.estimate_cov('marked_bootstrap', func=f)
        print(np.diagonal(cov).real)
        print('max log(ratio) = ',np.max(np.abs(np.log(np.diagonal(cov))-np.log(var_ggg))))
        np.testing.assert_allclose(np.log(np.diagonal(cov)), np.log(var_ggg), atol=0.8*tol_factor)

        print('bootstrap:')
        cov = gggp.estimate_cov('bootstrap', func=f)
        print(np.diagonal(cov).real)
        print('max log(ratio) = ',np.max(np.abs(np.log(np.diagonal(cov))-np.log(var_ggg))))
        np.testing.assert_allclose(np.log(np.diagonal(cov)), np.log(var_ggg), atol=0.8*tol_factor)

        # Patch on 2 only:
        print('with patches on 2 only:')
        gggp.process(cat, catp, cat)

        print('jackknife:')
        cov = gggp.estimate_cov('jackknife', func=f)
        print(np.diagonal(cov).real)
        print('max log(ratio) = ',np.max(np.abs(np.log(np.diagonal(cov))-np.log(var_ggg))))
        np.testing.assert_allclose(np.log(np.diagonal(cov)), np.log(var_ggg), atol=0.8*tol_factor)

        print('sample:')
        cov = gggp.estimate_cov('sample', func=f)
        print(np.diagonal(cov).real)
        print('max log(ratio) = ',np.max(np.abs(np.log(np.diagonal(cov))-np.log(var_ggg))))
        np.testing.assert_allclose(np.log(np.diagonal(cov)), np.log(var_ggg), atol=0.7*tol_factor)

        print('marked:')
        cov = gggp.estimate_cov('marked_bootstrap', func=f)
        print(np.diagonal(cov).real)
        print('max log(ratio) = ',np.max(np.abs(np.log(np.diagonal(cov))-np.log(var_ggg))))
        np.testing.assert_allclose(np.log(np.diagonal(cov)), np.log(var_ggg), atol=0.8*tol_factor)

        print('bootstrap:')
        cov = gggp.estimate_cov('bootstrap', func=f)
        print(np.diagonal(cov).real)
        print('max log(ratio) = ',np.max(np.abs(np.log(np.diagonal(cov))-np.log(var_ggg))))
        np.testing.assert_allclose(np.log(np.diagonal(cov)), np.log(var_ggg), atol=0.8*tol_factor)

        # Patch on 3 only:
        print('with patches on 3 only:')
        gggp.process(cat, cat, catp)

        print('jackknife:')
        cov = gggp.estimate_cov('jackknife', func=f)
        print(np.diagonal(cov).real)
        print('max log(ratio) = ',np.max(np.abs(np.log(np.diagonal(cov))-np.log(var_ggg))))
        np.testing.assert_allclose(np.log(np.diagonal(cov)), np.log(var_ggg), atol=0.8*tol_factor)

        print('sample:')
        cov = gggp.estimate_cov('sample', func=f)
        print(np.diagonal(cov).real)
        print('max log(ratio) = ',np.max(np.abs(np.log(np.diagonal(cov))-np.log(var_ggg))))
        np.testing.assert_allclose(np.log(np.diagonal(cov)), np.log(var_ggg), atol=0.7*tol_factor)

        print('marked:')
        cov = gggp.estimate_cov('marked_bootstrap', func=f)
        print(np.diagonal(cov).real)
        print('max log(ratio) = ',np.max(np.abs(np.log(np.diagonal(cov))-np.log(var_ggg))))
        np.testing.assert_allclose(np.log(np.diagonal(cov)), np.log(var_ggg), atol=0.8*tol_factor)

        print('bootstrap:')
        cov = gggp.estimate_cov('bootstrap', func=f)
        print(np.diagonal(cov).real)
        print('max log(ratio) = ',np.max(np.abs(np.log(np.diagonal(cov))-np.log(var_ggg))))
        np.testing.assert_allclose(np.log(np.diagonal(cov)), np.log(var_ggg), atol=0.9*tol_factor)

        # Patch on 1,2
        print('with patches on 1,2:')
        gggp.process(catp, catp, cat)

        print('jackknife:')
        cov = gggp.estimate_cov('jackknife', func=f)
        print(np.diagonal(cov).real)
        print('max log(ratio) = ',np.max(np.abs(np.log(np.diagonal(cov))-np.log(var_ggg))))
        np.testing.assert_allclose(np.log(np.diagonal(cov)), np.log(var_ggg), atol=0.6*tol_factor)

        print('sample:')
        cov = gggp.estimate_cov('sample', func=f)
        print(np.diagonal(cov).real)
        print('max log(ratio) = ',np.max(np.abs(np.log(np.diagonal(cov))-np.log(var_ggg))))
        np.testing.assert_allclose(np.log(np.diagonal(cov)), np.log(var_ggg), atol=0.6*tol_factor)

        print('marked:')
        cov = gggp.estimate_cov('marked_bootstrap', func=f)
        print(np.diagonal(cov).real)
        print('max log(ratio) = ',np.max(np.abs(np.log(np.diagonal(cov))-np.log(var_ggg))))
        np.testing.assert_allclose(np.log(np.diagonal(cov)), np.log(var_ggg), atol=0.8*tol_factor)

        print('bootstrap:')
        cov = gggp.estimate_cov('bootstrap', func=f)
        print(np.diagonal(cov).real)
        print('max log(ratio) = ',np.max(np.abs(np.log(np.diagonal(cov))-np.log(var_ggg))))
        np.testing.assert_allclose(np.log(np.diagonal(cov)), np.log(var_ggg), atol=0.5*tol_factor)

        # Patch on 2,3
        print('with patches on 2,3:')
        gggp.process(cat, catp)

        print('jackknife:')
        cov = gggp.estimate_cov('jackknife', func=f)
        print(np.diagonal(cov).real)
        print('max log(ratio) = ',np.max(np.abs(np.log(np.diagonal(cov))-np.log(var_ggg))))
        np.testing.assert_allclose(np.log(np.diagonal(cov)), np.log(var_ggg), atol=0.6*tol_factor)

        print('sample:')
        cov = gggp.estimate_cov('sample', func=f)
        print(np.diagonal(cov).real)
        print('max log(ratio) = ',np.max(np.abs(np.log(np.diagonal(cov))-np.log(var_ggg))))
        np.testing.assert_allclose(np.log(np.diagonal(cov)), np.log(var_ggg), atol=0.8*tol_factor)

        print('marked:')
        cov = gggp.estimate_cov('marked_bootstrap', func=f)
        print(np.diagonal(cov).real)
        print('max log(ratio) = ',np.max(np.abs(np.log(np.diagonal(cov))-np.log(var_ggg))))
        np.testing.assert_allclose(np.log(np.diagonal(cov)), np.log(var_ggg), atol=1.0*tol_factor)

        print('bootstrap:')
        cov = gggp.estimate_cov('bootstrap', func=f)
        print(np.diagonal(cov).real)
        print('max log(ratio) = ',np.max(np.abs(np.log(np.diagonal(cov))-np.log(var_ggg))))
        np.testing.assert_allclose(np.log(np.diagonal(cov)), np.log(var_ggg), atol=0.3*tol_factor)

        # Patch on 1,3
        print('with patches on 1,3:')
        gggp.process(catp, cat, catp)

        print('jackknife:')
        cov = gggp.estimate_cov('jackknife', func=f)
        print(np.diagonal(cov).real)
        print('max log(ratio) = ',np.max(np.abs(np.log(np.diagonal(cov))-np.log(var_ggg))))
        np.testing.assert_allclose(np.log(np.diagonal(cov)), np.log(var_ggg), atol=0.6*tol_factor)

        print('sample:')
        cov = gggp.estimate_cov('sample', func=f)
        print(np.diagonal(cov).real)
        print('max log(ratio) = ',np.max(np.abs(np.log(np.diagonal(cov))-np.log(var_ggg))))
        np.testing.assert_allclose(np.log(np.diagonal(cov)), np.log(var_ggg), atol=0.6*tol_factor)

        print('marked:')
        cov = gggp.estimate_cov('marked_bootstrap', func=f)
        print(np.diagonal(cov).real)
        print('max log(ratio) = ',np.max(np.abs(np.log(np.diagonal(cov))-np.log(var_ggg))))
        np.testing.assert_allclose(np.log(np.diagonal(cov)), np.log(var_ggg), atol=0.7*tol_factor)

        print('bootstrap:')
        cov = gggp.estimate_cov('bootstrap', func=f)
        print(np.diagonal(cov).real)
        print('max log(ratio) = ',np.max(np.abs(np.log(np.diagonal(cov))-np.log(var_ggg))))
        np.testing.assert_allclose(np.log(np.diagonal(cov)), np.log(var_ggg), atol=0.5*tol_factor)

    # Finally a set (with all patches) using the GGGCrossCorrelation class.
    gggc = treecorr.GGGCrossCorrelation(nbins=1, min_sep=20., max_sep=40.,
                                        min_u=0.6, max_u=1.0, nubins=1,
                                        min_v=0.0, max_v=0.6, nvbins=1, rng=rng)
    print('CrossCorrelation:')
    gggc.process(catp, catp, catp)
    for g in gggc._all:
        print(g.ntri.ravel())
        print(g.gam0.ravel())
        print(g.vargam0.ravel())

        np.testing.assert_allclose(g.ntri, ggg.ntri, rtol=0.05 * tol_factor)
        np.testing.assert_allclose(g.gam0, ggg.gam0, rtol=0.3 * tol_factor, atol=0.3 * tol_factor)
        np.testing.assert_allclose(g.vargam0, ggg.vargam0, rtol=0.05 * tol_factor)
        np.testing.assert_allclose(g.gam1, ggg.gam1, rtol=0.3 * tol_factor, atol=0.3 * tol_factor)
        np.testing.assert_allclose(g.vargam1, ggg.vargam1, rtol=0.05 * tol_factor)
        np.testing.assert_allclose(g.gam2, ggg.gam2, rtol=0.3 * tol_factor, atol=0.3 * tol_factor)
        np.testing.assert_allclose(g.vargam2, ggg.vargam2, rtol=0.05 * tol_factor)
        np.testing.assert_allclose(g.gam3, ggg.gam3, rtol=0.3 * tol_factor, atol=0.3 * tol_factor)
        np.testing.assert_allclose(g.vargam3, ggg.vargam3, rtol=0.05 * tol_factor)

    fc = lambda gggc: np.concatenate([
            [np.mean(g.gam0), np.mean(g.gam1), np.mean(g.gam2), np.mean(g.gam3)]
            for g in gggc._all])

    print('jackknife:')
    cov = gggc.estimate_cov('jackknife', func=fc)
    print(np.diagonal(cov).real)
    for i in range(6):
        v = np.diagonal(cov)[i*4:(i+1)*4]
        print('max log(ratio) = ',np.max(np.abs(np.log(v)-np.log(var_ggg))))
        np.testing.assert_allclose(np.log(v), np.log(var_ggg), atol=0.4*tol_factor)

    print('sample:')
    cov = gggc.estimate_cov('sample', func=fc)
    print(np.diagonal(cov).real)
    for i in range(6):
        v = np.diagonal(cov)[i*4:(i+1)*4]
        print('max log(ratio) = ',np.max(np.abs(np.log(v)-np.log(var_ggg))))
        np.testing.assert_allclose(np.log(v), np.log(var_ggg), atol=0.6*tol_factor)

    print('marked:')
    cov = gggc.estimate_cov('marked_bootstrap', func=fc)
    print(np.diagonal(cov).real)
    for i in range(6):
        v = np.diagonal(cov)[i*4:(i+1)*4]
        print('max log(ratio) = ',np.max(np.abs(np.log(v)-np.log(var_ggg))))
        np.testing.assert_allclose(np.log(v), np.log(var_ggg), atol=0.8*tol_factor)

    print('bootstrap:')
    cov = gggc.estimate_cov('bootstrap', func=fc)
    print(np.diagonal(cov).real)
    for i in range(6):
        v = np.diagonal(cov)[i*4:(i+1)*4]
        print('max log(ratio) = ',np.max(np.abs(np.log(v)-np.log(var_ggg))))
        np.testing.assert_allclose(np.log(v), np.log(var_ggg), atol=0.3*tol_factor)

    # Without func, don't check the accuracy, but make sure it returns something the right shape.
    cov = gggc.estimate_cov('jackknife')
    assert cov.shape == (48, 48)

@timer
def test_nnn_jk():
    # Test jackknife and other covariance estimates for nnn correlations.

    if __name__ == '__main__':
        # This setup takes about 1200 sec to run.
        nhalo = 300
        nsource = 2000
        npatch = 16
        source_factor = 50
        rand_factor = 3
        tol_factor = 1
    elif False:
        # This setup takes about 250 sec to run.
        nhalo = 200
        nsource = 1000
        npatch = 16
        source_factor = 50
        rand_factor = 2
        tol_factor = 2
    else:
        # This setup takes about 44 sec to run.
        nhalo = 100
        nsource = 500
        npatch = 8
        source_factor = 30
        rand_factor = 1
        tol_factor = 3

    file_name = 'data/test_nnn_jk_{}.npz'.format(nsource)
    print(file_name)
    if not os.path.isfile(file_name):
        rng = np.random.RandomState()
        nruns = 1000
        all_nnns = []
        all_nnnc = []
        t0 = time.time()
        for run in range(nruns):
            t2 = time.time()
            x, y, _, _, k = generate_shear_field(nsource * source_factor, nhalo, rng)
            p = k**3
            p /= np.sum(p)
            ns = rng.poisson(nsource)
            select = rng.choice(range(len(x)), size=ns, replace=False, p=p)
            print(run,': ',np.mean(k),np.std(k),np.min(k),np.max(k))
            cat = treecorr.Catalog(x=x[select], y=y[select])
            ddd = treecorr.NNNCorrelation(nbins=3, min_sep=50., max_sep=100., bin_slop=0.2,
                                           min_u=0.8, max_u=1.0, nubins=1,
                                           min_v=0.0, max_v=0.2, nvbins=1)

            rx = rng.uniform(0,1000, rand_factor*nsource)
            ry = rng.uniform(0,1000, rand_factor*nsource)
            rand_cat = treecorr.Catalog(x=rx, y=ry)
            rrr = treecorr.NNNCorrelation(nbins=3, min_sep=50., max_sep=100., bin_slop=0.2,
                                          min_u=0.8, max_u=1.0, nubins=1,
                                          min_v=0.0, max_v=0.2, nvbins=1)
            rrr.process(rand_cat)
            rdd = ddd.copy()
            drr = ddd.copy()
            ddd.process(cat)
            rdd.process(rand_cat, cat)
            drr.process(cat, rand_cat)
            zeta_s, _ = ddd.calculateZeta(rrr=rrr)
            zeta_c, _ = ddd.calculateZeta(rrr=rrr, drr=drr, rdd=rdd)
            print('simple: ',zeta_s.ravel())
            print('compensated: ',zeta_c.ravel())
            all_nnns.append(zeta_s.ravel())
            all_nnnc.append(zeta_c.ravel())
            t3 = time.time()
            print('time: ',round(t3-t2),round((t3-t0)/60),round((t3-t0)*(nruns/(run+1)-1)/60))
        mean_nnns = np.mean(all_nnns, axis=0)
        var_nnns = np.var(all_nnns, axis=0)
        mean_nnnc = np.mean(all_nnnc, axis=0)
        var_nnnc = np.var(all_nnnc, axis=0)
        np.savez(file_name, mean_nnns=mean_nnns, var_nnns=var_nnns,
                 mean_nnnc=mean_nnnc, var_nnnc=var_nnnc)

    data = np.load(file_name)
    mean_nnns = data['mean_nnns']
    var_nnns = data['var_nnns']
    mean_nnnc = data['mean_nnnc']
    var_nnnc = data['var_nnnc']
    print('mean simple = ',mean_nnns)
    print('var simple = ',var_nnns)
    print('mean compensated = ',mean_nnnc)
    print('var compensated = ',var_nnnc)

    # Make a random catalog with 2x as many sources, randomly distributed .
    rng = np.random.RandomState(1234)
    rx = rng.uniform(0,1000, rand_factor*nsource)
    ry = rng.uniform(0,1000, rand_factor*nsource)
    rand_cat = treecorr.Catalog(x=rx, y=ry)
    rrr = treecorr.NNNCorrelation(nbins=3, min_sep=50., max_sep=100., bin_slop=0.2,
                                  min_u=0.8, max_u=1.0, nubins=1,
                                  min_v=0.0, max_v=0.2, nvbins=1)
    t0 = time.time()
    rrr.process(rand_cat)
    t1 = time.time()
    print('Time to process rand cat = ',t1-t0)
    print('RRR:',rrr.tot)
    print(rrr.ntri.ravel())

    # Make the data catalog
    x, y, _, _, k = generate_shear_field(nsource * source_factor, nhalo, rng=rng)
    print('mean k = ',np.mean(k))
    print('min,max = ',np.min(k),np.max(k))
    p = k**3
    p /= np.sum(p)
    select = rng.choice(range(len(x)), size=nsource, replace=False, p=p)
    cat = treecorr.Catalog(x=x[select], y=y[select])
    ddd = treecorr.NNNCorrelation(nbins=3, min_sep=50., max_sep=100., bin_slop=0.2,
                                  min_u=0.8, max_u=1.0, nubins=1,
                                  min_v=0.0, max_v=0.2, nvbins=1, rng=rng)
    rdd = ddd.copy()
    drr = ddd.copy()
    ddd.process(cat)
    rdd.process(rand_cat, cat)
    drr.process(cat, rand_cat)
    zeta_s1, var_zeta_s1 = ddd.calculateZeta(rrr=rrr)
    zeta_c1, var_zeta_c1 = ddd.calculateZeta(rrr=rrr, drr=drr, rdd=rdd)
    print('DDD:',ddd.tot)
    print(ddd.ntri.ravel())
    print('simple: ')
    print(zeta_s1.ravel())
    print(var_zeta_s1.ravel())
    print('DRR:',drr.tot)
    print(drr.ntri.ravel())
    print('RDD:',rdd.tot)
    print(rdd.ntri.ravel())
    print('compensated: ')
    print(zeta_c1.ravel())
    print(var_zeta_c1.ravel())

    # Make the patches with a large random catalog to make sure the patches are uniform area.
    big_rx = rng.uniform(0,1000, 100*nsource)
    big_ry = rng.uniform(0,1000, 100*nsource)
    big_catp = treecorr.Catalog(x=big_rx, y=big_ry, npatch=npatch, rng=rng)
    patch_centers = big_catp.patch_centers

    # Do the same thing with patches on D, but not yet on R.
    dddp = treecorr.NNNCorrelation(nbins=3, min_sep=50., max_sep=100., bin_slop=0.2,
                                   min_u=0.8, max_u=1.0, nubins=1,
                                   min_v=0.0, max_v=0.2, nvbins=1, rng=rng)
    rddp = dddp.copy()
    drrp = dddp.copy()
    catp = treecorr.Catalog(x=x[select], y=y[select], patch_centers=patch_centers)
    print('Patch\tNtot')
    for p in catp.patches:
        print(p.patch,'\t',p.ntot,'\t',patch_centers[p.patch])

    print('with patches on D:')
    dddp.process(catp)
    rddp.process(rand_cat, catp)
    drrp.process(catp, rand_cat)

    # Need to run calculateZeta to get patch-based covariance
    with assert_raises(RuntimeError):
        dddp.estimate_cov('jackknife')

    zeta_s2, var_zeta_s2 = dddp.calculateZeta(rrr=rrr)
    print('DDD:',dddp.tot)
    print(dddp.ntri.ravel())
    print('simple: ')
    print(zeta_s2.ravel())
    print(var_zeta_s2.ravel())
    np.testing.assert_allclose(zeta_s2, zeta_s1, rtol=0.05 * tol_factor)
    np.testing.assert_allclose(var_zeta_s2, var_zeta_s1, rtol=0.05 * tol_factor)

    # Check the _calculate_xi_from_pairs function.  Using all pairs, should get total xi.
    ddd1 = dddp.copy()
    ddd1._calculate_xi_from_pairs(dddp.results.keys())
    np.testing.assert_allclose(ddd1.zeta, dddp.zeta)

    # None of these are very good without the random using patches.
    # I think this is basically just that the approximations used for estimating the area_frac
    # to figure out the appropriate altered RRR counts isn't accurate enough when the total
    # counts are as low as this.  I think (hope) that it should be semi-ok when N is much larger,
    # but this is probably saying that for 3pt using patches for R is even more important than
    # for 2pt.
    # Ofc, it could also be that this is telling me I still have a bug somewhere that I haven't
    # managed to find...  :(
    print('jackknife:')
    cov = dddp.estimate_cov('jackknife')
    print(np.diagonal(cov))
    print('max log(ratio) = ',np.max(np.abs(np.log(np.diagonal(cov))-np.log(var_nnns))))
    np.testing.assert_allclose(np.log(np.diagonal(cov)), np.log(var_nnns), atol=2.3*tol_factor)

    print('sample:')
    cov = dddp.estimate_cov('sample')
    print(np.diagonal(cov))
    print('max log(ratio) = ',np.max(np.abs(np.log(np.diagonal(cov))-np.log(var_nnns))))
    np.testing.assert_allclose(np.log(np.diagonal(cov)), np.log(var_nnns), atol=1.2*tol_factor)

    print('marked:')
    cov = dddp.estimate_cov('marked_bootstrap')
    print(np.diagonal(cov))
    print('max log(ratio) = ',np.max(np.abs(np.log(np.diagonal(cov))-np.log(var_nnns))))
    np.testing.assert_allclose(np.log(np.diagonal(cov)), np.log(var_nnns), atol=1.3*tol_factor)

    print('bootstrap:')
    cov = dddp.estimate_cov('bootstrap')
    print(np.diagonal(cov))
    print('max log(ratio) = ',np.max(np.abs(np.log(np.diagonal(cov))-np.log(var_nnns))))
    np.testing.assert_allclose(np.log(np.diagonal(cov)), np.log(var_nnns), atol=2.2*tol_factor)

    zeta_c2, var_zeta_c2 = dddp.calculateZeta(rrr=rrr, drr=drrp, rdd=rddp)
    print('compensated: ')
    print('DRR:',drrp.tot)
    print(drrp.ntri.ravel())
    print('RDD:',rddp.tot)
    print(rddp.ntri.ravel())
    print(zeta_c2.ravel())
    print(var_zeta_c2.ravel())
    np.testing.assert_allclose(zeta_c2, zeta_c1, rtol=0.05 * tol_factor, atol=1.e-3 * tol_factor)
    np.testing.assert_allclose(var_zeta_c2, var_zeta_c1, rtol=0.05 * tol_factor)

    print('jackknife:')
    cov = dddp.estimate_cov('jackknife')
    print(np.diagonal(cov))
    print('max log(ratio) = ',np.max(np.abs(np.log(np.diagonal(cov))-np.log(var_nnnc))))
    np.testing.assert_allclose(np.log(np.diagonal(cov)), np.log(var_nnnc), atol=2.6*tol_factor)

    print('sample:')
    cov = dddp.estimate_cov('sample')
    print(np.diagonal(cov))
    print('max log(ratio) = ',np.max(np.abs(np.log(np.diagonal(cov))-np.log(var_nnnc))))
    np.testing.assert_allclose(np.log(np.diagonal(cov)), np.log(var_nnnc), atol=3.8*tol_factor)

    print('marked:')
    cov = dddp.estimate_cov('marked_bootstrap')
    print(np.diagonal(cov))
    print('max log(ratio) = ',np.max(np.abs(np.log(np.diagonal(cov))-np.log(var_nnnc))))
    np.testing.assert_allclose(np.log(np.diagonal(cov)), np.log(var_nnnc), atol=2.3*tol_factor)

    print('bootstrap:')
    cov = dddp.estimate_cov('bootstrap')
    print(np.diagonal(cov))
    print('max log(ratio) = ',np.max(np.abs(np.log(np.diagonal(cov))-np.log(var_nnnc))))
    np.testing.assert_allclose(np.log(np.diagonal(cov)), np.log(var_nnnc), atol=2.6*tol_factor)

    # Now with the random also using patches
    # These are a lot better than the above tests.  But still not nearly as good as we were able
    # to get in 2pt.  I'm pretty sure this is just due to the fact that we need to have much
    # smaller catalogs to make it feasible to run this in a reasonable amount of time.  I don't
    # think this is a sign of any bug in the code.
    print('with patched random catalog:')
    rand_catp = treecorr.Catalog(x=rx, y=ry, patch_centers=patch_centers)
    rrrp = rrr.copy()
    rrrp.process(rand_catp)
    drrp.process(catp, rand_catp)
    rddp.process(rand_catp, catp)
    print('simple: ')
    zeta_s2, var_zeta_s2 = dddp.calculateZeta(rrr=rrrp)
    print('DDD:',dddp.tot)
    print(dddp.ntri.ravel())
    print(zeta_s2.ravel())
    print(var_zeta_s2.ravel())
    np.testing.assert_allclose(zeta_s2, zeta_s1, rtol=0.05 * tol_factor)
    np.testing.assert_allclose(var_zeta_s2, var_zeta_s1, rtol=0.05 * tol_factor)

    ddd1 = dddp.copy()
    ddd1._calculate_xi_from_pairs(dddp.results.keys())
    np.testing.assert_allclose(ddd1.zeta, dddp.zeta)

    t0 = time.time()
    print('jackknife:')
    cov = dddp.estimate_cov('jackknife')
    print(np.diagonal(cov))
    print('max log(ratio) = ',np.max(np.abs(np.log(np.diagonal(cov))-np.log(var_nnns))))
    np.testing.assert_allclose(np.log(np.diagonal(cov)), np.log(var_nnns), atol=0.9*tol_factor)
    t1 = time.time()
    print('t = ',t1-t0)
    t0 = time.time()

    print('sample:')
    cov = dddp.estimate_cov('sample')
    print(np.diagonal(cov))
    print('max log(ratio) = ',np.max(np.abs(np.log(np.diagonal(cov))-np.log(var_nnns))))
    np.testing.assert_allclose(np.log(np.diagonal(cov)), np.log(var_nnns), atol=0.7*tol_factor)
    t1 = time.time()
    print('t = ',t1-t0)
    t0 = time.time()

    print('marked:')
    cov = dddp.estimate_cov('marked_bootstrap')
    print(np.diagonal(cov))
    print('max log(ratio) = ',np.max(np.abs(np.log(np.diagonal(cov))-np.log(var_nnns))))
    np.testing.assert_allclose(np.log(np.diagonal(cov)), np.log(var_nnns), atol=0.8*tol_factor)
    t1 = time.time()
    print('t = ',t1-t0)
    t0 = time.time()

    print('bootstrap:')
    cov = dddp.estimate_cov('bootstrap')
    print(np.diagonal(cov))
    print('max log(ratio) = ',np.max(np.abs(np.log(np.diagonal(cov))-np.log(var_nnns))))
    np.testing.assert_allclose(np.log(np.diagonal(cov)), np.log(var_nnns), atol=1.0*tol_factor)
    t1 = time.time()
    print('t = ',t1-t0)
    t0 = time.time()

    print('compensated: ')
    zeta_c2, var_zeta_c2 = dddp.calculateZeta(rrr=rrrp, drr=drrp, rdd=rddp)
    print('DRR:',drrp.tot)
    print(drrp.ntri.ravel())
    print('RDD:',rddp.tot)
    print(rddp.ntri.ravel())
    print(zeta_c2.ravel())
    print(var_zeta_c2.ravel())
    np.testing.assert_allclose(zeta_c2, zeta_c1, rtol=0.05 * tol_factor, atol=1.e-3 * tol_factor)
    np.testing.assert_allclose(var_zeta_c2, var_zeta_c1, rtol=0.05 * tol_factor)

    t0 = time.time()
    print('jackknife:')
    cov = dddp.estimate_cov('jackknife')
    print(np.diagonal(cov))
    print('max log(ratio) = ',np.max(np.abs(np.log(np.diagonal(cov))-np.log(var_nnnc))))
    np.testing.assert_allclose(np.log(np.diagonal(cov)), np.log(var_nnnc), atol=0.8*tol_factor)
    t1 = time.time()
    print('t = ',t1-t0)
    t0 = time.time()

    print('sample:')
    cov = dddp.estimate_cov('sample')
    print(np.diagonal(cov))
    print('max log(ratio) = ',np.max(np.abs(np.log(np.diagonal(cov))-np.log(var_nnnc))))
    np.testing.assert_allclose(np.log(np.diagonal(cov)), np.log(var_nnnc), atol=0.8*tol_factor)
    t1 = time.time()
    print('t = ',t1-t0)
    t0 = time.time()

    print('marked:')
    cov = dddp.estimate_cov('marked_bootstrap')
    print(np.diagonal(cov))
    print('max log(ratio) = ',np.max(np.abs(np.log(np.diagonal(cov))-np.log(var_nnnc))))
    np.testing.assert_allclose(np.log(np.diagonal(cov)), np.log(var_nnnc), atol=0.8*tol_factor)
    t1 = time.time()
    print('t = ',t1-t0)
    t0 = time.time()

    print('bootstrap:')
    cov = dddp.estimate_cov('bootstrap')
    print(np.diagonal(cov))
    print('max log(ratio) = ',np.max(np.abs(np.log(np.diagonal(cov))-np.log(var_nnnc))))
    np.testing.assert_allclose(np.log(np.diagonal(cov)), np.log(var_nnnc), atol=0.8*tol_factor)
    t1 = time.time()
    print('t = ',t1-t0)
    t0 = time.time()

    # Check that these still work after roundtripping through files.
    cov1 = dddp.estimate_cov('jackknife')
    try:
        import fitsio
    except ImportError:
        pass
    else:
        file_name = os.path.join('output','test_write_results_ddd.fits')
        rrr_file_name = os.path.join('output','test_write_results_rrr.fits')
        drr_file_name = os.path.join('output','test_write_results_drr.fits')
        rdd_file_name = os.path.join('output','test_write_results_rdd.fits')
        dddp.write(file_name, write_patch_results=True)
        rrrp.write(rrr_file_name, write_patch_results=True)
        drrp.write(drr_file_name, write_patch_results=True)
        rddp.write(rdd_file_name, write_patch_results=True)
        ddd3 = treecorr.NNNCorrelation(nbins=3, min_sep=50., max_sep=100., bin_slop=0.2,
                                       min_u=0.8, max_u=1.0, nubins=1,
                                       min_v=0.0, max_v=0.2, nvbins=1)
        rrr3 = ddd3.copy()
        drr3 = ddd3.copy()
        rdd3 = ddd3.copy()
        ddd3.read(file_name)
        rrr3.read(rrr_file_name)
        drr3.read(drr_file_name)
        rdd3.read(rdd_file_name)
        ddd3.calculateZeta(rrr=rrr3, drr=drr3, rdd=rdd3)
        cov3 = ddd3.estimate_cov('jackknife')
        np.testing.assert_allclose(cov3, cov1)

    # Also with ascii, since that works differeny.
    file_name = os.path.join('output','test_write_results_ddd.dat')
    rrr_file_name = os.path.join('output','test_write_results_rrr.dat')
    drr_file_name = os.path.join('output','test_write_results_drr.dat')
    rdd_file_name = os.path.join('output','test_write_results_rdd.dat')
    dddp.write(file_name, write_patch_results=True)
    rrrp.write(rrr_file_name, write_patch_results=True)
    drrp.write(drr_file_name, write_patch_results=True)
    rddp.write(rdd_file_name, write_patch_results=True)
    ddd4 = treecorr.NNNCorrelation(nbins=3, min_sep=50., max_sep=100., bin_slop=0.2,
                                   min_u=0.8, max_u=1.0, nubins=1,
                                   min_v=0.0, max_v=0.2, nvbins=1)
    rrr4 = ddd4.copy()
    drr4 = ddd4.copy()
    rdd4 = ddd4.copy()
    ddd4.read(file_name)
    rrr4.read(rrr_file_name)
    drr4.read(drr_file_name)
    rdd4.read(rdd_file_name)
    ddd4.calculateZeta(rrr=rrr4, drr=drr4, rdd=rdd4)
    cov4 = ddd4.estimate_cov('jackknife')
    np.testing.assert_allclose(cov4, cov1)

    # And also try to match the type if HDF
    try:
        import h5py
    except ImportError:
        print('Skipping saving HDF patches, since h5py not installed.')
        h5py = None

    if h5py is not None:
        # Finally with hdf
        file_name = os.path.join('output','test_write_results_ddd.hdf5')
        rrr_file_name = os.path.join('output','test_write_results_rrr.hdf5')
        drr_file_name = os.path.join('output','test_write_results_drr.hdf5')
        rdd_file_name = os.path.join('output','test_write_results_rdd.hdf5')
        dddp.write(file_name, write_patch_results=True)
        rrrp.write(rrr_file_name, write_patch_results=True)
        drrp.write(drr_file_name, write_patch_results=True)
        rddp.write(rdd_file_name, write_patch_results=True)
        ddd5 = treecorr.NNNCorrelation(nbins=3, min_sep=50., max_sep=100., bin_slop=0.2,
                                       min_u=0.8, max_u=1.0, nubins=1,
                                       min_v=0.0, max_v=0.2, nvbins=1)
        rrr5 = ddd5.copy()
        drr5 = ddd5.copy()
        rdd5 = ddd5.copy()
        ddd5.read(file_name)
        rrr5.read(rrr_file_name)
        drr5.read(drr_file_name)
        rdd5.read(rdd_file_name)
        ddd5.calculateZeta(rrr=rrr5, drr=drr5, rdd=rdd5)
        cov5 = ddd5.estimate_cov('jackknife')
        np.testing.assert_allclose(cov5, cov1)

    # I haven't implemented calculateZeta for the NNNCrossCorrelation class, because I'm not
    # actually sure what the right thing to do here is for calculating a single zeta vectors.
    # Do we do a different one for each of the 6 permutations?  Or one overall one?
    # So rather than just do something, I'll wait until someone has a coherent use case where
    # they want this and can explain exactly what the right thing to compute is.
    # So to just exercise the machinery with NNNCrossCorrelation, I'm using a func parameter
    # to compute something equivalent to the simple zeta calculation.

    dddc = treecorr.NNNCrossCorrelation(nbins=3, min_sep=50., max_sep=100., bin_slop=0.2,
                                        min_u=0.8, max_u=1.0, nubins=1,
                                        min_v=0.0, max_v=0.2, nvbins=1, rng=rng)
    rrrc = treecorr.NNNCrossCorrelation(nbins=3, min_sep=50., max_sep=100., bin_slop=0.2,
                                        min_u=0.8, max_u=1.0, nubins=1,
                                        min_v=0.0, max_v=0.2, nvbins=1)
    print('CrossCorrelation:')
    dddc.process(catp, catp, catp)
    rrrc.process(rand_catp, rand_catp, rand_catp)

    def cc_zeta(corrs):
        d, r = corrs
        d1 = d.n1n2n3.copy()
        d1._sum(d._all)
        r1 = r.n1n2n3.copy()
        r1._sum(r._all)
        zeta, _ = d1.calculateZeta(rrr=r1)
        return zeta.ravel()

    print('simple: ')
    zeta_s3 = cc_zeta([dddc, rrrc])
    print(zeta_s3)
    np.testing.assert_allclose(zeta_s3, zeta_s1.ravel(), rtol=0.05 * tol_factor)

    print('jackknife:')
    cov = treecorr.estimate_multi_cov([dddc,rrrc], 'jackknife', func=cc_zeta)
    print(np.diagonal(cov))
    print('max log(ratio) = ',np.max(np.abs(np.log(np.diagonal(cov))-np.log(var_nnns))))
    np.testing.assert_allclose(np.log(np.diagonal(cov)), np.log(var_nnns), atol=0.9*tol_factor)

    # Test design matrix
    A, w = treecorr.build_multi_cov_design_matrix([dddc,rrrc], method='jackknife', func=cc_zeta)
    A -= np.mean(A, axis=0)
    C = (1-1/npatch) * A.conj().T.dot(A)
    np.testing.assert_allclose(C, cov)

    print('sample:')
    cov = treecorr.estimate_multi_cov([dddc,rrrc], 'sample', func=cc_zeta)
    print(np.diagonal(cov))
    print('max log(ratio) = ',np.max(np.abs(np.log(np.diagonal(cov))-np.log(var_nnns))))
    np.testing.assert_allclose(np.log(np.diagonal(cov)), np.log(var_nnns), atol=1.2*tol_factor)

    print('marked:')
    cov = treecorr.estimate_multi_cov([dddc,rrrc], 'marked_bootstrap', func=cc_zeta)
    print(np.diagonal(cov))
    print('max log(ratio) = ',np.max(np.abs(np.log(np.diagonal(cov))-np.log(var_nnns))))
    np.testing.assert_allclose(np.log(np.diagonal(cov)), np.log(var_nnns), atol=1.5*tol_factor)

    print('bootstrap:')
    cov = treecorr.estimate_multi_cov([dddc,rrrc], 'bootstrap', func=cc_zeta)
    print(np.diagonal(cov))
    print('max log(ratio) = ',np.max(np.abs(np.log(np.diagonal(cov))-np.log(var_nnns))))
    np.testing.assert_allclose(np.log(np.diagonal(cov)), np.log(var_nnns), atol=0.6*tol_factor)

    # Repeat with a 1-2 cross-correlation
    print('CrossCorrelation 1-2:')
    dddc.process(catp, catp)
    rrrc.process(rand_catp, rand_catp)

    print('simple: ')
    zeta_s3 = cc_zeta([dddc, rrrc])
    print(zeta_s3)
    np.testing.assert_allclose(zeta_s3, zeta_s1.ravel(), rtol=0.05 * tol_factor)

    print('jackknife:')
    cov = treecorr.estimate_multi_cov([dddc,rrrc], 'jackknife', func=cc_zeta)
    print(np.diagonal(cov))
    print('max log(ratio) = ',np.max(np.abs(np.log(np.diagonal(cov))-np.log(var_nnns))))
    np.testing.assert_allclose(np.log(np.diagonal(cov)), np.log(var_nnns), atol=0.9*tol_factor)

    print('sample:')
    cov = treecorr.estimate_multi_cov([dddc,rrrc], 'sample', func=cc_zeta)
    print(np.diagonal(cov))
    print('max log(ratio) = ',np.max(np.abs(np.log(np.diagonal(cov))-np.log(var_nnns))))
    np.testing.assert_allclose(np.log(np.diagonal(cov)), np.log(var_nnns), atol=1.1*tol_factor)

    print('marked:')
    cov = treecorr.estimate_multi_cov([dddc,rrrc], 'marked_bootstrap', func=cc_zeta)
    print(np.diagonal(cov))
    print('max log(ratio) = ',np.max(np.abs(np.log(np.diagonal(cov))-np.log(var_nnns))))
    np.testing.assert_allclose(np.log(np.diagonal(cov)), np.log(var_nnns), atol=1.5*tol_factor)

    print('bootstrap:')
    cov = treecorr.estimate_multi_cov([dddc,rrrc], 'bootstrap', func=cc_zeta)
    print(np.diagonal(cov))
    print('max log(ratio) = ',np.max(np.abs(np.log(np.diagonal(cov))-np.log(var_nnns))))
    np.testing.assert_allclose(np.log(np.diagonal(cov)), np.log(var_nnns), atol=0.6*tol_factor)


@timer
def test_brute_jk():
    # With bin_slop = 0, the jackknife calculation from patches should match a
    # brute force calcaulation where we literally remove one patch at a time to make
    # the vectors.
    if __name__ == '__main__':
        nhalo = 100
        ngal = 500
        npatch = 16
        rand_factor = 5
    else:
        nhalo = 100
        ngal = 30
        npatch = 16
        rand_factor = 2

    rng = np.random.RandomState(8675309)
    x, y, g1, g2, k = generate_shear_field(ngal, nhalo, rng)

    rx = rng.uniform(0,1000, rand_factor*ngal)
    ry = rng.uniform(0,1000, rand_factor*ngal)
    rand_cat_nopatch = treecorr.Catalog(x=rx, y=ry)
    rand_cat = treecorr.Catalog(x=rx, y=ry, npatch=npatch, rng=rng)
    patch_centers = rand_cat.patch_centers

    cat_nopatch = treecorr.Catalog(x=x, y=y, g1=g1, g2=g2, k=k)
    cat = treecorr.Catalog(x=x, y=y, g1=g1, g2=g2, k=k, patch_centers=patch_centers)
    print('cat patches = ',np.unique(cat.patch))
    print('len = ',cat.nobj, cat.ntot)
    assert cat.nobj == ngal

    print('Patch\tNtot')
    for p in cat.patches:
        print(p.patch,'\t',p.ntot,'\t',patch_centers[p.patch])

    # Start with KKK, since relatively simple.
    kkk1 = treecorr.KKKCorrelation(nbins=3, min_sep=100., max_sep=300., brute=True,
                                   min_u=0., max_u=1.0, nubins=1,
                                   min_v=0., max_v=1.0, nvbins=1)
    kkk1.process(cat_nopatch)

    kkk = treecorr.KKKCorrelation(nbins=3, min_sep=100., max_sep=300., brute=True,
                                  min_u=0., max_u=1.0, nubins=1,
                                  min_v=0., max_v=1.0, nvbins=1,
                                  var_method='jackknife')
    kkk.process(cat)
    np.testing.assert_allclose(kkk.zeta, kkk1.zeta)

    kkk_zeta_list = []
    for i in range(npatch):
        cat1 = treecorr.Catalog(x=cat.x[cat.patch != i],
                                y=cat.y[cat.patch != i],
                                k=cat.k[cat.patch != i],
                                g1=cat.g1[cat.patch != i],
                                g2=cat.g2[cat.patch != i])
        kkk1 = treecorr.KKKCorrelation(nbins=3, min_sep=100., max_sep=300., brute=True,
                                       min_u=0., max_u=1.0, nubins=1,
                                       min_v=0., max_v=1.0, nvbins=1)
        kkk1.process(cat1)
        print('zeta = ',kkk1.zeta.ravel())
        kkk_zeta_list.append(kkk1.zeta.ravel())

    kkk_zeta_list = np.array(kkk_zeta_list)
    cov = np.cov(kkk_zeta_list.T, bias=True) * (len(kkk_zeta_list)-1)
    varzeta = np.diagonal(np.cov(kkk_zeta_list.T, bias=True)) * (len(kkk_zeta_list)-1)
    print('KKK: treecorr jackknife varzeta = ',kkk.varzeta.ravel())
    print('KKK: direct jackknife varzeta = ',varzeta)
    np.testing.assert_allclose(kkk.varzeta.ravel(), varzeta)

    # Now GGG
    ggg1 = treecorr.GGGCorrelation(nbins=3, min_sep=100., max_sep=300., brute=True,
                                   min_u=0., max_u=1.0, nubins=1,
                                   min_v=0., max_v=1.0, nvbins=1)
    ggg1.process(cat_nopatch)

    ggg = treecorr.GGGCorrelation(nbins=3, min_sep=100., max_sep=300., brute=True,
                                  min_u=0., max_u=1.0, nubins=1,
                                  min_v=0., max_v=1.0, nvbins=1,
                                  var_method='jackknife')
    ggg.process(cat)
    np.testing.assert_allclose(ggg.gam0, ggg1.gam0)
    np.testing.assert_allclose(ggg.gam1, ggg1.gam1)
    np.testing.assert_allclose(ggg.gam2, ggg1.gam2)
    np.testing.assert_allclose(ggg.gam3, ggg1.gam3)

    ggg_gam0_list = []
    ggg_gam1_list = []
    ggg_gam2_list = []
    ggg_gam3_list = []
    ggg_map3_list = []
    for i in range(npatch):
        cat1 = treecorr.Catalog(x=cat.x[cat.patch != i],
                                y=cat.y[cat.patch != i],
                                k=cat.k[cat.patch != i],
                                g1=cat.g1[cat.patch != i],
                                g2=cat.g2[cat.patch != i])
        ggg1 = treecorr.GGGCorrelation(nbins=3, min_sep=100., max_sep=300., brute=True,
                                       min_u=0., max_u=1.0, nubins=1,
                                       min_v=0., max_v=1.0, nvbins=1)
        ggg1.process(cat1)
        ggg_gam0_list.append(ggg1.gam0.ravel())
        ggg_gam1_list.append(ggg1.gam1.ravel())
        ggg_gam2_list.append(ggg1.gam2.ravel())
        ggg_gam3_list.append(ggg1.gam3.ravel())
        ggg_map3_list.append(ggg1.calculateMap3()[0])

    ggg_gam0_list = np.array(ggg_gam0_list)
    vargam0 = np.diagonal(np.cov(ggg_gam0_list.T, bias=True)) * (len(ggg_gam0_list)-1)
    print('GGG: treecorr jackknife vargam0 = ',ggg.vargam0.ravel())
    print('GGG: direct jackknife vargam0 = ',vargam0)
    np.testing.assert_allclose(ggg.vargam0.ravel(), vargam0)
    ggg_gam1_list = np.array(ggg_gam1_list)
    vargam1 = np.diagonal(np.cov(ggg_gam1_list.T, bias=True)) * (len(ggg_gam1_list)-1)
    print('GGG: treecorr jackknife vargam1 = ',ggg.vargam1.ravel())
    print('GGG: direct jackknife vargam1 = ',vargam1)
    np.testing.assert_allclose(ggg.vargam1.ravel(), vargam1)
    ggg_gam2_list = np.array(ggg_gam2_list)
    vargam2 = np.diagonal(np.cov(ggg_gam2_list.T, bias=True)) * (len(ggg_gam2_list)-1)
    print('GGG: treecorr jackknife vargam2 = ',ggg.vargam2.ravel())
    print('GGG: direct jackknife vargam2 = ',vargam2)
    np.testing.assert_allclose(ggg.vargam2.ravel(), vargam2)
    ggg_gam3_list = np.array(ggg_gam3_list)
    vargam3 = np.diagonal(np.cov(ggg_gam3_list.T, bias=True)) * (len(ggg_gam3_list)-1)
    print('GGG: treecorr jackknife vargam3 = ',ggg.vargam3.ravel())
    print('GGG: direct jackknife vargam3 = ',vargam3)
    np.testing.assert_allclose(ggg.vargam3.ravel(), vargam3)

    ggg_map3_list = np.array(ggg_map3_list)
    varmap3 = np.diagonal(np.cov(ggg_map3_list.T, bias=True)) * (len(ggg_map3_list)-1)

    # Use estimate_multi_cov
    covmap3 = treecorr.estimate_multi_cov([ggg], 'jackknife',
                                          func=lambda corrs: corrs[0].calculateMap3()[0])
    print('GGG: treecorr jackknife varmap3 = ',np.diagonal(covmap3))
    print('GGG: direct jackknife varmap3 = ',varmap3)
    np.testing.assert_allclose(np.diagonal(covmap3), varmap3)

    # Use estimate_cov
    covmap3b = ggg.estimate_cov('jackknife', func=lambda corr: corr.calculateMap3()[0])
    print('GGG: treecorr jackknife varmap3 = ',np.diagonal(covmap3b))
    print('GGG: direct jackknife varmap3 = ',varmap3)
    np.testing.assert_allclose(np.diagonal(covmap3b), varmap3)
    np.testing.assert_allclose(covmap3b, covmap3, rtol=1.e-10, atol=1.e-10)

    # Finally NNN, where we need to use randoms.  Both simple and compensated.
    ddd = treecorr.NNNCorrelation(nbins=3, min_sep=100., max_sep=300., bin_slop=0,
                                  min_u=0., max_u=1.0, nubins=1,
                                  min_v=0., max_v=1.0, nvbins=1,
                                  var_method='jackknife')
    drr = ddd.copy()
    rdd = ddd.copy()
    rrr = ddd.copy()
    ddd.process(cat)
    drr.process(cat, rand_cat)
    rdd.process(rand_cat, cat)
    rrr.process(rand_cat)

    zeta1_list = []
    zeta2_list = []
    for i in range(npatch):
        cat1 = treecorr.Catalog(x=cat.x[cat.patch != i],
                                y=cat.y[cat.patch != i],
                                k=cat.k[cat.patch != i],
                                g1=cat.g1[cat.patch != i],
                                g2=cat.g2[cat.patch != i])
        rand_cat1 = treecorr.Catalog(x=rand_cat.x[rand_cat.patch != i],
                                     y=rand_cat.y[rand_cat.patch != i])
        ddd1 = treecorr.NNNCorrelation(nbins=3, min_sep=100., max_sep=300., bin_slop=0,
                                       min_u=0., max_u=1.0, nubins=1,
                                       min_v=0., max_v=1.0, nvbins=1)
        drr1 = ddd1.copy()
        rdd1 = ddd1.copy()
        rrr1 = ddd1.copy()
        ddd1.process(cat1)
        drr1.process(cat1, rand_cat1)
        rdd1.process(rand_cat1, cat1)
        rrr1.process(rand_cat1)
        zeta1_list.append(ddd1.calculateZeta(rrr=rrr1)[0].ravel())
        zeta2_list.append(ddd1.calculateZeta(rrr=rrr1, drr=drr1, rdd=rdd1)[0].ravel())

    print('simple')
    zeta1_list = np.array(zeta1_list)
    zeta2, varzeta2 = ddd.calculateZeta(rrr=rrr)
    varzeta1 = np.diagonal(np.cov(zeta1_list.T, bias=True)) * (len(zeta1_list)-1)
    print('NNN: treecorr jackknife varzeta = ',ddd.varzeta.ravel())
    print('NNN: direct jackknife varzeta = ',varzeta1)
    np.testing.assert_allclose(ddd.varzeta.ravel(), varzeta1)

    print('compensated')
    print(zeta2_list)
    zeta2_list = np.array(zeta2_list)
    zeta2, varzeta2 = ddd.calculateZeta(rrr=rrr, drr=drr, rdd=rdd)
    varzeta2 = np.diagonal(np.cov(zeta2_list.T, bias=True)) * (len(zeta2_list)-1)
    print('NNN: treecorr jackknife varzeta = ',ddd.varzeta.ravel())
    print('NNN: direct jackknife varzeta = ',varzeta2)
    np.testing.assert_allclose(ddd.varzeta.ravel(), varzeta2)

    # Can't do patch calculation with different numbers of patches in rrr, drr, rdd.
    rand_cat3 = treecorr.Catalog(x=rx, y=ry, npatch=3)
    cat3 = treecorr.Catalog(x=x, y=y, patch_centers=rand_cat3.patch_centers)
    rrr3 = rrr.copy()
    drr3 = drr.copy()
    rdd3 = rdd.copy()
    rrr3.process(rand_cat3)
    drr3.process(cat3, rand_cat3)
    rdd3.process(rand_cat3, cat3)
    with assert_raises(RuntimeError):
        ddd.calculateZeta(rrr=rrr3)
    with assert_raises(RuntimeError):
        ddd.calculateZeta(rrr=rrr3, drr=drr, rdd=rdd)
    with assert_raises(RuntimeError):
        ddd.calculateZeta(rrr=rrr, drr=drr3, rdd=rdd3)
    with assert_raises(RuntimeError):
        ddd.calculateZeta(rrr=rrr, drr=drr, rdd=rdd3)
    with assert_raises(RuntimeError):
        ddd.calculateZeta(rrr=rrr, drr=drr3, rdd=rdd)


@timer
def test_finalize_false():

    if __name__ == '__main__':
        nsource = 80
        nhalo = 100
        npatch = 16
    else:
        nsource = 80
        nhalo = 50
        npatch = 4

    # Make three independent data sets
    rng = np.random.RandomState(8675309)
    x_1, y_1, g1_1, g2_1, k_1 = generate_shear_field(nsource, nhalo, rng)
    x_2, y_2, g1_2, g2_2, k_2 = generate_shear_field(nsource, nhalo, rng)
    x_3, y_3, g1_3, g2_3, k_3 = generate_shear_field(nsource, nhalo, rng)

    # Make a single catalog with all three together
    cat = treecorr.Catalog(x=np.concatenate([x_1, x_2, x_3]),
                           y=np.concatenate([y_1, y_2, y_3]),
                           g1=np.concatenate([g1_1, g1_2, g1_3]),
                           g2=np.concatenate([g2_1, g2_2, g2_3]),
                           k=np.concatenate([k_1, k_2, k_3]),
                           npatch=npatch)

    # Now the three separately, using the same patch centers
    cat1 = treecorr.Catalog(x=x_1, y=y_1, g1=g1_1, g2=g2_1, k=k_1, patch_centers=cat.patch_centers)
    cat2 = treecorr.Catalog(x=x_2, y=y_2, g1=g1_2, g2=g2_2, k=k_2, patch_centers=cat.patch_centers)
    cat3 = treecorr.Catalog(x=x_3, y=y_3, g1=g1_3, g2=g2_3, k=k_3, patch_centers=cat.patch_centers)

    np.testing.assert_array_equal(cat1.patch, cat.patch[0:nsource])
    np.testing.assert_array_equal(cat2.patch, cat.patch[nsource:2*nsource])
    np.testing.assert_array_equal(cat3.patch, cat.patch[2*nsource:3*nsource])

    # KKK auto
    kkk1 = treecorr.KKKCorrelation(nbins=3, min_sep=30., max_sep=100., brute=True,
                                   min_u=0.8, max_u=1.0, nubins=1,
                                   min_v=0., max_v=0.2, nvbins=1)
    kkk1.process(cat)

    kkk2 = treecorr.KKKCorrelation(nbins=3, min_sep=30., max_sep=100., brute=True,
                                   min_u=0.8, max_u=1.0, nubins=1,
                                   min_v=0., max_v=0.2, nvbins=1)
    kkk2.process(cat1, initialize=True, finalize=False)
    kkk2.process(cat2, initialize=False, finalize=False)
    kkk2.process(cat3, initialize=False, finalize=False)
    kkk2.process(cat1, cat2, initialize=False, finalize=False)
    kkk2.process(cat1, cat3, initialize=False, finalize=False)
    kkk2.process(cat2, cat1, initialize=False, finalize=False)
    kkk2.process(cat2, cat3, initialize=False, finalize=False)
    kkk2.process(cat3, cat1, initialize=False, finalize=False)
    kkk2.process(cat3, cat2, initialize=False, finalize=False)
    kkk2.process(cat1, cat2, cat3, initialize=False, finalize=True)

    np.testing.assert_allclose(kkk1.ntri, kkk2.ntri)
    np.testing.assert_allclose(kkk1.weight, kkk2.weight)
    np.testing.assert_allclose(kkk1.meand1, kkk2.meand1)
    np.testing.assert_allclose(kkk1.meand2, kkk2.meand2)
    np.testing.assert_allclose(kkk1.meand3, kkk2.meand3)
    np.testing.assert_allclose(kkk1.zeta, kkk2.zeta)

    # KKK cross12
    cat23 = treecorr.Catalog(x=np.concatenate([x_2, x_3]),
                             y=np.concatenate([y_2, y_3]),
                             g1=np.concatenate([g1_2, g1_3]),
                             g2=np.concatenate([g2_2, g2_3]),
                             k=np.concatenate([k_2, k_3]),
                             patch_centers=cat.patch_centers)
    np.testing.assert_array_equal(cat23.patch, cat.patch[nsource:3*nsource])

    kkk1.process(cat1, cat23)
    kkk2.process(cat1, cat2, initialize=True, finalize=False)
    kkk2.process(cat1, cat3, initialize=False, finalize=False)
    kkk2.process(cat1, cat2, cat3, initialize=False, finalize=True)

    np.testing.assert_allclose(kkk1.ntri, kkk2.ntri)
    np.testing.assert_allclose(kkk1.weight, kkk2.weight)
    np.testing.assert_allclose(kkk1.meand1, kkk2.meand1)
    np.testing.assert_allclose(kkk1.meand2, kkk2.meand2)
    np.testing.assert_allclose(kkk1.meand3, kkk2.meand3)
    np.testing.assert_allclose(kkk1.zeta, kkk2.zeta)

    # KKKCross cross12
    kkkc1 = treecorr.KKKCrossCorrelation(nbins=3, min_sep=30., max_sep=100., brute=True,
                                         min_u=0.8, max_u=1.0, nubins=1,
                                         min_v=0., max_v=0.2, nvbins=1)
    kkkc1.process(cat1, cat23)

    kkkc2 = treecorr.KKKCrossCorrelation(nbins=3, min_sep=30., max_sep=100., brute=True,
                                         min_u=0.8, max_u=1.0, nubins=1,
                                         min_v=0., max_v=0.2, nvbins=1)
    kkkc2.process(cat1, cat2, initialize=True, finalize=False)
    kkkc2.process(cat1, cat3, initialize=False, finalize=False)
    kkkc2.process(cat1, cat2, cat3, initialize=False, finalize=True)

    for perm in ['k1k2k3', 'k1k3k2', 'k2k1k3', 'k2k3k1', 'k3k1k2', 'k3k2k1']:
        kkk1 = getattr(kkkc1, perm)
        kkk2 = getattr(kkkc2, perm)
        np.testing.assert_allclose(kkk1.ntri, kkk2.ntri)
        np.testing.assert_allclose(kkk1.weight, kkk2.weight)
        np.testing.assert_allclose(kkk1.meand1, kkk2.meand1)
        np.testing.assert_allclose(kkk1.meand2, kkk2.meand2)
        np.testing.assert_allclose(kkk1.meand3, kkk2.meand3)
        np.testing.assert_allclose(kkk1.zeta, kkk2.zeta)

    # KKK cross
    kkk1.process(cat, cat2, cat3)
    kkk2.process(cat1, cat2, cat3, initialize=True, finalize=False)
    kkk2.process(cat2, cat2, cat3, initialize=False, finalize=False)
    kkk2.process(cat3, cat2, cat3, initialize=False, finalize=True)

    np.testing.assert_allclose(kkk1.ntri, kkk2.ntri)
    np.testing.assert_allclose(kkk1.weight, kkk2.weight)
    np.testing.assert_allclose(kkk1.meand1, kkk2.meand1)
    np.testing.assert_allclose(kkk1.meand2, kkk2.meand2)
    np.testing.assert_allclose(kkk1.meand3, kkk2.meand3)
    np.testing.assert_allclose(kkk1.zeta, kkk2.zeta)

    # KKKCross cross
    kkkc1.process(cat, cat2, cat3)
    kkkc2.process(cat1, cat2, cat3, initialize=True, finalize=False)
    kkkc2.process(cat2, cat2, cat3, initialize=False, finalize=False)
    kkkc2.process(cat3, cat2, cat3, initialize=False, finalize=True)

    for perm in ['k1k2k3', 'k1k3k2', 'k2k1k3', 'k2k3k1', 'k3k1k2', 'k3k2k1']:
        kkk1 = getattr(kkkc1, perm)
        kkk2 = getattr(kkkc2, perm)
        np.testing.assert_allclose(kkk1.ntri, kkk2.ntri)
        np.testing.assert_allclose(kkk1.weight, kkk2.weight)
        np.testing.assert_allclose(kkk1.meand1, kkk2.meand1)
        np.testing.assert_allclose(kkk1.meand2, kkk2.meand2)
        np.testing.assert_allclose(kkk1.meand3, kkk2.meand3)
        np.testing.assert_allclose(kkk1.zeta, kkk2.zeta)

    # GGG auto
    ggg1 = treecorr.GGGCorrelation(nbins=3, min_sep=30., max_sep=100., brute=True,
                                   min_u=0.8, max_u=1.0, nubins=1,
                                   min_v=0., max_v=0.2, nvbins=1)
    ggg1.process(cat)

    ggg2 = treecorr.GGGCorrelation(nbins=3, min_sep=30., max_sep=100., brute=True,
                                   min_u=0.8, max_u=1.0, nubins=1,
                                   min_v=0., max_v=0.2, nvbins=1)
    ggg2.process(cat1, initialize=True, finalize=False)
    ggg2.process(cat2, initialize=False, finalize=False)
    ggg2.process(cat3, initialize=False, finalize=False)
    ggg2.process(cat1, cat2, initialize=False, finalize=False)
    ggg2.process(cat1, cat3, initialize=False, finalize=False)
    ggg2.process(cat2, cat1, initialize=False, finalize=False)
    ggg2.process(cat2, cat3, initialize=False, finalize=False)
    ggg2.process(cat3, cat1, initialize=False, finalize=False)
    ggg2.process(cat3, cat2, initialize=False, finalize=False)
    ggg2.process(cat1, cat2, cat3, initialize=False, finalize=True)

    np.testing.assert_allclose(ggg1.ntri, ggg2.ntri)
    np.testing.assert_allclose(ggg1.weight, ggg2.weight)
    np.testing.assert_allclose(ggg1.meand1, ggg2.meand1)
    np.testing.assert_allclose(ggg1.meand2, ggg2.meand2)
    np.testing.assert_allclose(ggg1.meand3, ggg2.meand3)
    np.testing.assert_allclose(ggg1.gam0, ggg2.gam0)
    np.testing.assert_allclose(ggg1.gam1, ggg2.gam1)
    np.testing.assert_allclose(ggg1.gam2, ggg2.gam2)
    np.testing.assert_allclose(ggg1.gam3, ggg2.gam3)

    # GGG cross12
    ggg1.process(cat1, cat23)
    ggg2.process(cat1, cat2, initialize=True, finalize=False)
    ggg2.process(cat1, cat3, initialize=False, finalize=False)
    ggg2.process(cat1, cat2, cat3, initialize=False, finalize=True)

    np.testing.assert_allclose(ggg1.ntri, ggg2.ntri)
    np.testing.assert_allclose(ggg1.weight, ggg2.weight)
    np.testing.assert_allclose(ggg1.meand1, ggg2.meand1)
    np.testing.assert_allclose(ggg1.meand2, ggg2.meand2)
    np.testing.assert_allclose(ggg1.meand3, ggg2.meand3)
    np.testing.assert_allclose(ggg1.gam0, ggg2.gam0)
    np.testing.assert_allclose(ggg1.gam1, ggg2.gam1)
    np.testing.assert_allclose(ggg1.gam2, ggg2.gam2)
    np.testing.assert_allclose(ggg1.gam3, ggg2.gam3)

    # GGGCross cross12
    gggc1 = treecorr.GGGCrossCorrelation(nbins=3, min_sep=30., max_sep=100., brute=True,
                                         min_u=0.8, max_u=1.0, nubins=1,
                                         min_v=0., max_v=0.2, nvbins=1)
    gggc1.process(cat1, cat23)

    gggc2 = treecorr.GGGCrossCorrelation(nbins=3, min_sep=30., max_sep=100., brute=True,
                                         min_u=0.8, max_u=1.0, nubins=1,
                                         min_v=0., max_v=0.2, nvbins=1)
    gggc2.process(cat1, cat2, initialize=True, finalize=False)
    gggc2.process(cat1, cat3, initialize=False, finalize=False)
    gggc2.process(cat1, cat2, cat3, initialize=False, finalize=True)

    for perm in ['g1g2g3', 'g1g3g2', 'g2g1g3', 'g2g3g1', 'g3g1g2', 'g3g2g1']:
        ggg1 = getattr(gggc1, perm)
        ggg2 = getattr(gggc2, perm)
        np.testing.assert_allclose(ggg1.ntri, ggg2.ntri)
        np.testing.assert_allclose(ggg1.weight, ggg2.weight)
        np.testing.assert_allclose(ggg1.meand1, ggg2.meand1)
        np.testing.assert_allclose(ggg1.meand2, ggg2.meand2)
        np.testing.assert_allclose(ggg1.meand3, ggg2.meand3)
        np.testing.assert_allclose(ggg1.gam0, ggg2.gam0)
        np.testing.assert_allclose(ggg1.gam1, ggg2.gam1)
        np.testing.assert_allclose(ggg1.gam2, ggg2.gam2)
        np.testing.assert_allclose(ggg1.gam3, ggg2.gam3)

    # GGG cross
    ggg1.process(cat, cat2, cat3)
    ggg2.process(cat1, cat2, cat3, initialize=True, finalize=False)
    ggg2.process(cat2, cat2, cat3, initialize=False, finalize=False)
    ggg2.process(cat3, cat2, cat3, initialize=False, finalize=True)

    np.testing.assert_allclose(ggg1.ntri, ggg2.ntri)
    np.testing.assert_allclose(ggg1.weight, ggg2.weight)
    np.testing.assert_allclose(ggg1.meand1, ggg2.meand1)
    np.testing.assert_allclose(ggg1.meand2, ggg2.meand2)
    np.testing.assert_allclose(ggg1.meand3, ggg2.meand3)
    np.testing.assert_allclose(ggg1.gam0, ggg2.gam0)
    np.testing.assert_allclose(ggg1.gam1, ggg2.gam1)
    np.testing.assert_allclose(ggg1.gam2, ggg2.gam2)
    np.testing.assert_allclose(ggg1.gam3, ggg2.gam3)

    # GGGCross cross
    gggc1.process(cat, cat2, cat3)
    gggc2.process(cat1, cat2, cat3, initialize=True, finalize=False)
    gggc2.process(cat2, cat2, cat3, initialize=False, finalize=False)
    gggc2.process(cat3, cat2, cat3, initialize=False, finalize=True)

    for perm in ['g1g2g3', 'g1g3g2', 'g2g1g3', 'g2g3g1', 'g3g1g2', 'g3g2g1']:
        ggg1 = getattr(gggc1, perm)
        ggg2 = getattr(gggc2, perm)
        np.testing.assert_allclose(ggg1.ntri, ggg2.ntri)
        np.testing.assert_allclose(ggg1.weight, ggg2.weight)
        np.testing.assert_allclose(ggg1.meand1, ggg2.meand1)
        np.testing.assert_allclose(ggg1.meand2, ggg2.meand2)
        np.testing.assert_allclose(ggg1.meand3, ggg2.meand3)
        np.testing.assert_allclose(ggg1.gam0, ggg2.gam0)
        np.testing.assert_allclose(ggg1.gam1, ggg2.gam1)
        np.testing.assert_allclose(ggg1.gam2, ggg2.gam2)
        np.testing.assert_allclose(ggg1.gam3, ggg2.gam3)

    # NNN auto
    nnn1 = treecorr.NNNCorrelation(nbins=3, min_sep=10., max_sep=200., bin_slop=0,
                                   min_u=0.8, max_u=1.0, nubins=1,
                                   min_v=0., max_v=0.2, nvbins=1)
    nnn1.process(cat)

    nnn2 = treecorr.NNNCorrelation(nbins=3, min_sep=10., max_sep=200., bin_slop=0,
                                   min_u=0.8, max_u=1.0, nubins=1,
                                   min_v=0., max_v=0.2, nvbins=1)
    nnn2.process(cat1, initialize=True, finalize=False)
    nnn2.process(cat2, initialize=False, finalize=False)
    nnn2.process(cat3, initialize=False, finalize=False)
    nnn2.process(cat1, cat2, initialize=False, finalize=False)
    nnn2.process(cat1, cat3, initialize=False, finalize=False)
    nnn2.process(cat2, cat1, initialize=False, finalize=False)
    nnn2.process(cat2, cat3, initialize=False, finalize=False)
    nnn2.process(cat3, cat1, initialize=False, finalize=False)
    nnn2.process(cat3, cat2, initialize=False, finalize=False)
    nnn2.process(cat1, cat2, cat3, initialize=False, finalize=True)

    np.testing.assert_allclose(nnn1.ntri, nnn2.ntri)
    np.testing.assert_allclose(nnn1.weight, nnn2.weight)
    np.testing.assert_allclose(nnn1.meand1, nnn2.meand1)
    np.testing.assert_allclose(nnn1.meand2, nnn2.meand2)
    np.testing.assert_allclose(nnn1.meand3, nnn2.meand3)

    # NNN cross12
    nnn1.process(cat1, cat23)
    nnn2.process(cat1, cat2, initialize=True, finalize=False)
    nnn2.process(cat1, cat3, initialize=False, finalize=False)
    nnn2.process(cat1, cat2, cat3, initialize=False, finalize=True)

    np.testing.assert_allclose(nnn1.ntri, nnn2.ntri)
    np.testing.assert_allclose(nnn1.weight, nnn2.weight)
    np.testing.assert_allclose(nnn1.meand1, nnn2.meand1)
    np.testing.assert_allclose(nnn1.meand2, nnn2.meand2)
    np.testing.assert_allclose(nnn1.meand3, nnn2.meand3)

    # NNNCross cross12
    nnnc1 = treecorr.NNNCrossCorrelation(nbins=3, min_sep=10., max_sep=200., bin_slop=0,
                                         min_u=0.8, max_u=1.0, nubins=1,
                                         min_v=0., max_v=0.2, nvbins=1)
    nnnc1.process(cat1, cat23)

    nnnc2 = treecorr.NNNCrossCorrelation(nbins=3, min_sep=10., max_sep=200., bin_slop=0,
                                         min_u=0.8, max_u=1.0, nubins=1,
                                         min_v=0., max_v=0.2, nvbins=1)
    nnnc2.process(cat1, cat2, initialize=True, finalize=False)
    nnnc2.process(cat1, cat3, initialize=False, finalize=False)
    nnnc2.process(cat1, cat2, cat3, initialize=False, finalize=True)

    for perm in ['n1n2n3', 'n1n3n2', 'n2n1n3', 'n2n3n1', 'n3n1n2', 'n3n2n1']:
        nnn1 = getattr(nnnc1, perm)
        nnn2 = getattr(nnnc2, perm)
        np.testing.assert_allclose(nnn1.ntri, nnn2.ntri)
        np.testing.assert_allclose(nnn1.weight, nnn2.weight)
        np.testing.assert_allclose(nnn1.meand1, nnn2.meand1)
        np.testing.assert_allclose(nnn1.meand2, nnn2.meand2)
        np.testing.assert_allclose(nnn1.meand3, nnn2.meand3)

    # NNN cross
    nnn1.process(cat, cat2, cat3)
    nnn2.process(cat1, cat2, cat3, initialize=True, finalize=False)
    nnn2.process(cat2, cat2, cat3, initialize=False, finalize=False)
    nnn2.process(cat3, cat2, cat3, initialize=False, finalize=True)

    np.testing.assert_allclose(nnn1.ntri, nnn2.ntri)
    np.testing.assert_allclose(nnn1.weight, nnn2.weight)
    np.testing.assert_allclose(nnn1.meand1, nnn2.meand1)
    np.testing.assert_allclose(nnn1.meand2, nnn2.meand2)
    np.testing.assert_allclose(nnn1.meand3, nnn2.meand3)

    # NNNCross cross
    nnnc1.process(cat, cat2, cat3)
    nnnc2.process(cat1, cat2, cat3, initialize=True, finalize=False)
    nnnc2.process(cat2, cat2, cat3, initialize=False, finalize=False)
    nnnc2.process(cat3, cat2, cat3, initialize=False, finalize=True)

    for perm in ['n1n2n3', 'n1n3n2', 'n2n1n3', 'n2n3n1', 'n3n1n2', 'n3n2n1']:
        nnn1 = getattr(nnnc1, perm)
        nnn2 = getattr(nnnc2, perm)
        np.testing.assert_allclose(nnn1.ntri, nnn2.ntri)
        np.testing.assert_allclose(nnn1.weight, nnn2.weight)
        np.testing.assert_allclose(nnn1.meand1, nnn2.meand1)
        np.testing.assert_allclose(nnn1.meand2, nnn2.meand2)
        np.testing.assert_allclose(nnn1.meand3, nnn2.meand3)

@timer
def test_lowmem():
    # Test using patches to keep the memory usage lower.
    try:
        import fitsio
    except ImportError:
        print('Skip test_lowmem, since fitsio not installed.')
        return

    if __name__ == '__main__':
        nsource = 10000
        nhalo = 100
        npatch = 4
        himem = 7.e5
        lomem = 8.e4
    else:
        nsource = 1000
        nhalo = 100
        npatch = 4
        himem = 1.3e5
        lomem = 8.e4

    rng = np.random.RandomState(8675309)
    x, y, g1, g2, k = generate_shear_field(nsource, nhalo, rng)

    file_name = os.path.join('output','test_lowmem_3pt.fits')
    orig_cat = treecorr.Catalog(x=x, y=y, g1=g1, g2=g2, k=k, npatch=npatch)
    patch_centers = orig_cat.patch_centers
    orig_cat.write(file_name)
    del orig_cat

    try:
        import guppy
        hp = guppy.hpy()
        hp.setrelheap()
    except Exception:
        hp = None

    full_cat = treecorr.Catalog(file_name,
                                x_col='x', y_col='y', g1_col='g1', g2_col='g2', k_col='k',
                                patch_centers=patch_centers)

    kkk = treecorr.KKKCorrelation(nbins=1, min_sep=280., max_sep=300.,
                                  min_u=0.95, max_u=1.0, nubins=1,
                                  min_v=0., max_v=0.05, nvbins=1)

    t0 = time.time()
    s0 = hp.heap().size if hp else 0
    kkk.process(full_cat)
    t1 = time.time()
    s1 = hp.heap().size if hp else 2*himem
    print('regular: ',s1, t1-t0, s1-s0)
    assert s1-s0 > himem  # This version uses a lot of memory.

    ntri1 = kkk.ntri
    zeta1 = kkk.zeta
    full_cat.unload()
    kkk.clear()

    # Remake with save_patch_dir.
    clear_save('test_lowmem_3pt_%03d.fits', npatch)
    save_cat = treecorr.Catalog(file_name,
                                x_col='x', y_col='y', g1_col='g1', g2_col='g2', k_col='k',
                                patch_centers=patch_centers, save_patch_dir='output')

    t0 = time.time()
    s0 = hp.heap().size if hp else 0
    kkk.process(save_cat, low_mem=True, finalize=False)
    t1 = time.time()
    s1 = hp.heap().size if hp else 0
    print('lomem 1: ',s1, t1-t0, s1-s0)
    assert s1-s0 < lomem  # This version uses a lot less memory
    ntri2 = kkk.ntri
    zeta2 = kkk.zeta
    print('ntri1 = ',ntri1)
    print('zeta1 = ',zeta1)
    np.testing.assert_array_equal(ntri2, ntri1)
    np.testing.assert_array_equal(zeta2, zeta1)

    # Check running as a cross-correlation
    save_cat.unload()
    t0 = time.time()
    s0 = hp.heap().size if hp else 0
    kkk.process(save_cat, save_cat, low_mem=True)
    t1 = time.time()
    s1 = hp.heap().size if hp else 0
    print('lomem 2: ',s1, t1-t0, s1-s0)
    assert s1-s0 < lomem
    ntri3 = kkk.ntri
    zeta3 = kkk.zeta
    np.testing.assert_array_equal(ntri3, ntri1)
    np.testing.assert_array_equal(zeta3, zeta1)

    # Check running as a cross-correlation
    save_cat.unload()
    t0 = time.time()
    s0 = hp.heap().size if hp else 0
    kkk.process(save_cat, save_cat, save_cat, low_mem=True)
    t1 = time.time()
    s1 = hp.heap().size if hp else 0
    print('lomem 3: ',s1, t1-t0, s1-s0)
    assert s1-s0 < lomem
    ntri4 = kkk.ntri
    zeta4 = kkk.zeta
    np.testing.assert_array_equal(ntri4, ntri1)
    np.testing.assert_array_equal(zeta4, zeta1)


if __name__ == '__main__':
    test_kkk_jk()
    test_ggg_jk()
    test_nnn_jk()
    test_brute_jk()
    test_finalize_false()
    test_lowmem
