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
import treecorr
import time
import os
import sys
import coord
from unittest import mock

from test_helper import do_pickle, CaptureLog
from test_helper import assert_raises, timer, assert_warns

@timer
def test_direct():
    # If the catalogs are small enough, we can do a direct calculation to see if comes out right.
    # This should exactly match the treecorr result if brute force.

    ngal = 200
    s = 10.
    rng = np.random.RandomState(8675309)
    x1 = rng.normal(0,s, (ngal,) )
    y1 = rng.normal(0,s, (ngal,) )
    w1 = rng.random_sample(ngal)

    x2 = rng.normal(0,s, (ngal,) )
    y2 = rng.normal(0,s, (ngal,) )
    w2 = rng.random_sample(ngal)
    k2 = rng.normal(0,3, (ngal,) )

    cat1 = treecorr.Catalog(x=x1, y=y1, w=w1)
    cat2 = treecorr.Catalog(x=x2, y=y2, w=w2, k=k2)

    min_sep = 1.
    max_sep = 50.
    nbins = 50
    bin_size = np.log(max_sep/min_sep) / nbins
    nk = treecorr.NKCorrelation(min_sep=min_sep, max_sep=max_sep, nbins=nbins, brute=True)
    nk.process(cat1, cat2)

    nk2 = nk.copy()
    nk2.process(cat1, cat2, corr_only=True)
    np.testing.assert_allclose(nk2.weight, nk.weight)
    np.testing.assert_allclose(nk2.xi, nk.xi)
    np.testing.assert_allclose(nk2.npairs, nk.weight / (np.mean(w1) * np.mean(w2)))
    np.testing.assert_allclose(nk2.meanr, nk.rnom)
    np.testing.assert_allclose(nk2.meanlogr, nk.logr)

    true_npairs = np.zeros(nbins, dtype=int)
    true_weight = np.zeros(nbins, dtype=float)
    true_xi = np.zeros(nbins, dtype=float)
    for i in range(ngal):
        # It's hard to do all the pairs at once with numpy operations (although maybe possible).
        # But we can at least do all the pairs for each entry in cat1 at once with arrays.
        rsq = (x1[i]-x2)**2 + (y1[i]-y2)**2
        r = np.sqrt(rsq)

        ww = w1[i] * w2
        xi = ww * k2

        index = np.floor(np.log(r/min_sep) / bin_size).astype(int)
        mask = (index >= 0) & (index < nbins)
        np.add.at(true_npairs, index[mask], 1)
        np.add.at(true_weight, index[mask], ww[mask])
        np.add.at(true_xi, index[mask], xi[mask])

    true_xi /= true_weight

    print('true_npairs = ',true_npairs)
    print('diff = ',nk.npairs - true_npairs)
    np.testing.assert_array_equal(nk.npairs, true_npairs)

    print('true_weight = ',true_weight)
    print('diff = ',nk.weight - true_weight)
    np.testing.assert_allclose(nk.weight, true_weight, rtol=1.e-6, atol=1.e-8)

    print('true_xi = ',true_xi)
    print('nk.xi = ',nk.xi)
    np.testing.assert_allclose(nk.xi, true_xi, rtol=1.e-5, atol=2.e-8)

    # Check that running via the corr2 script works correctly.
    config = treecorr.config.read_config('configs/nk_direct.yaml')
    try:
        import fitsio
    except ImportError:
        pass
    else:
        cat1.write(config['file_name'])
        cat2.write(config['file_name2'])
        with CaptureLog() as cl:
            treecorr.corr2(config, cl.logger)
        assert "skipping k_col" in cl.output
        data = fitsio.read(config['nk_file_name'])
        np.testing.assert_allclose(data['r_nom'], nk.rnom)
        np.testing.assert_allclose(data['npairs'], nk.npairs)
        np.testing.assert_allclose(data['weight'], nk.weight)
        np.testing.assert_allclose(data['kappa'], nk.xi)

        # When not using corr2, it's invalid to specify invalid g1_col, g2_col
        with assert_raises(ValueError):
            cat = treecorr.Catalog(config['file_name'], config)

        # Invalid with only one file_name
        del config['file_name2']
        with assert_raises(TypeError):
            treecorr.corr2(config)
        config['file_name2'] = 'data/nk_direct_cat2.fits'
        # Invalid to request compoensated if no rand_file
        config['nk_statistic'] = 'compensated'
        with assert_raises(TypeError):
            treecorr.corr2(config)

    # Repeat with binslop = 0, since the code flow is different from brute=True
    # And don't do any top-level recursion so we actually test not going to the leaves.
    nk = treecorr.NKCorrelation(min_sep=min_sep, max_sep=max_sep, nbins=nbins, bin_slop=0,
                                max_top=0)
    nk.process(cat1, cat2)
    np.testing.assert_array_equal(nk.npairs, true_npairs)
    np.testing.assert_allclose(nk.weight, true_weight, rtol=1.e-6, atol=1.e-8)
    np.testing.assert_allclose(nk.xi, true_xi, rtol=1.e-5, atol=2.e-8)

    # Check a few basic operations with a NKCorrelation object.
    do_pickle(nk)

    nk2 = nk.copy()
    nk2 += nk
    np.testing.assert_allclose(nk2.npairs, 2*nk.npairs)
    np.testing.assert_allclose(nk2.weight, 2*nk.weight)
    np.testing.assert_allclose(nk2.meanr, 2*nk.meanr)
    np.testing.assert_allclose(nk2.meanlogr, 2*nk.meanlogr)
    np.testing.assert_allclose(nk2.xi, 2*nk.xi)

    nk2.clear()
    nk2 += nk
    np.testing.assert_allclose(nk2.npairs, nk.npairs)
    np.testing.assert_allclose(nk2.weight, nk.weight)
    np.testing.assert_allclose(nk2.meanr, nk.meanr)
    np.testing.assert_allclose(nk2.meanlogr, nk.meanlogr)
    np.testing.assert_allclose(nk2.xi, nk.xi)

    ascii_name = 'output/nk_ascii.txt'
    nk.write(ascii_name, precision=16)
    nk3 = treecorr.NKCorrelation(min_sep=min_sep, max_sep=max_sep, nbins=nbins, bin_type='Log')
    nk3.read(ascii_name)
    np.testing.assert_allclose(nk3.npairs, nk.npairs)
    np.testing.assert_allclose(nk3.weight, nk.weight)
    np.testing.assert_allclose(nk3.meanr, nk.meanr)
    np.testing.assert_allclose(nk3.meanlogr, nk.meanlogr)
    np.testing.assert_allclose(nk3.xi, nk.xi)

    # Check that the repr is minimal
    assert repr(nk3) == f'NKCorrelation(min_sep={min_sep}, max_sep={max_sep}, nbins={nbins})'

    # Simpler API using from_file:
    with CaptureLog() as cl:
        nk3b = treecorr.NKCorrelation.from_file(ascii_name, logger=cl.logger)
    assert ascii_name in cl.output
    np.testing.assert_allclose(nk3b.npairs, nk.npairs)
    np.testing.assert_allclose(nk3b.weight, nk.weight)
    np.testing.assert_allclose(nk3b.meanr, nk.meanr)
    np.testing.assert_allclose(nk3b.meanlogr, nk.meanlogr)
    np.testing.assert_allclose(nk3b.xi, nk.xi)

    # or using the Corr2 base class
    with CaptureLog() as cl:
        nk3c = treecorr.Corr2.from_file(ascii_name, logger=cl.logger)
    assert ascii_name in cl.output
    np.testing.assert_allclose(nk3c.npairs, nk.npairs)
    np.testing.assert_allclose(nk3c.weight, nk.weight)
    np.testing.assert_allclose(nk3c.meanr, nk.meanr)
    np.testing.assert_allclose(nk3c.meanlogr, nk.meanlogr)
    np.testing.assert_allclose(nk3c.xi, nk.xi)

    with assert_raises(TypeError):
        nk2 += config
    nk4 = treecorr.NKCorrelation(min_sep=min_sep/2, max_sep=max_sep, nbins=nbins)
    with assert_raises(ValueError):
        nk2 += nk4
    nk5 = treecorr.NKCorrelation(min_sep=min_sep, max_sep=max_sep*2, nbins=nbins)
    with assert_raises(ValueError):
        nk2 += nk5
    nk6 = treecorr.NKCorrelation(min_sep=min_sep, max_sep=max_sep, nbins=nbins*2)
    with assert_raises(ValueError):
        nk2 += nk6
    with assert_raises(TypeError):
        nk.process(cat1)

    try:
        import fitsio
    except ImportError:
        pass
    else:
        fits_name = 'output/nk_fits.fits'
        nk.write(fits_name)
        nk4 = treecorr.NKCorrelation(min_sep=min_sep, max_sep=max_sep, nbins=nbins)
        nk4.read(fits_name)
        np.testing.assert_allclose(nk4.npairs, nk.npairs)
        np.testing.assert_allclose(nk4.weight, nk.weight)
        np.testing.assert_allclose(nk4.meanr, nk.meanr)
        np.testing.assert_allclose(nk4.meanlogr, nk.meanlogr)
        np.testing.assert_allclose(nk4.xi, nk.xi)

        nk4b = treecorr.NKCorrelation.from_file(fits_name)
        np.testing.assert_allclose(nk4b.npairs, nk.npairs)
        np.testing.assert_allclose(nk4b.weight, nk.weight)
        np.testing.assert_allclose(nk4b.meanr, nk.meanr)
        np.testing.assert_allclose(nk4b.meanlogr, nk.meanlogr)
        np.testing.assert_allclose(nk4b.xi, nk.xi)

    with assert_raises(ValueError):
        nk.process(cat1, cat2, patch_method='nonlocal')


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
    k2 = rng.normal(0,3, (ngal,) )

    ra1, dec1 = coord.CelestialCoord.xyz_to_radec(x1,y1,z1)
    ra2, dec2 = coord.CelestialCoord.xyz_to_radec(x2,y2,z2)

    cat1 = treecorr.Catalog(ra=ra1, dec=dec1, ra_units='rad', dec_units='rad', w=w1)
    cat2 = treecorr.Catalog(ra=ra2, dec=dec2, ra_units='rad', dec_units='rad', w=w2, k=k2)

    min_sep = 1.
    max_sep = 10.
    nbins = 50
    bin_size = np.log(max_sep/min_sep) / nbins
    nk = treecorr.NKCorrelation(min_sep=min_sep, max_sep=max_sep, nbins=nbins,
                                sep_units='deg', brute=True)
    nk.process(cat1, cat2)

    r1 = np.sqrt(x1**2 + y1**2 + z1**2)
    r2 = np.sqrt(x2**2 + y2**2 + z2**2)
    x1 /= r1;  y1 /= r1;  z1 /= r1
    x2 /= r2;  y2 /= r2;  z2 /= r2

    true_npairs = np.zeros(nbins, dtype=int)
    true_weight = np.zeros(nbins, dtype=float)
    true_xi = np.zeros(nbins, dtype=float)

    for i in range(ngal):
        for j in range(ngal):
            rsq = (x1[i]-x2[j])**2 + (y1[i]-y2[j])**2 + (z1[i]-z2[j])**2
            r = np.sqrt(rsq)
            r *= coord.radians / coord.degrees

            index = np.floor(np.log(r/min_sep) / bin_size).astype(int)
            if index < 0 or index >= nbins:
                continue

            ww = w1[i] * w2[j]
            xi = ww * k2[j]

            true_npairs[index] += 1
            true_weight[index] += ww
            true_xi[index] += xi

    true_xi /= true_weight

    print('true_npairs = ',true_npairs)
    print('diff = ',nk.npairs - true_npairs)
    np.testing.assert_array_equal(nk.npairs, true_npairs)

    print('true_weight = ',true_weight)
    print('diff = ',nk.weight - true_weight)
    np.testing.assert_allclose(nk.weight, true_weight, rtol=1.e-5, atol=1.e-8)

    print('true_xi = ',true_xi)
    print('nk.xi = ',nk.xi)
    np.testing.assert_allclose(nk.xi, true_xi, rtol=1.e-4, atol=1.e-8)

    # Check that running via the corr2 script works correctly.
    config = treecorr.config.read_config('configs/nk_direct_spherical.yaml')
    try:
        import fitsio
    except ImportError:
        pass
    else:
        cat1.write(config['file_name'])
        cat2.write(config['file_name2'])
        treecorr.corr2(config)
        data = fitsio.read(config['nk_file_name'])
        np.testing.assert_allclose(data['r_nom'], nk.rnom)
        np.testing.assert_allclose(data['npairs'], nk.npairs)
        np.testing.assert_allclose(data['weight'], nk.weight)
        np.testing.assert_allclose(data['kappa'], nk.xi)

    # Repeat with binslop = 0, since the code flow is different from brute=True.
    # And don't do any top-level recursion so we actually test not going to the leaves.
    nk = treecorr.NKCorrelation(min_sep=min_sep, max_sep=max_sep, nbins=nbins,
                                sep_units='deg', bin_slop=0, max_top=0)
    nk.process(cat1, cat2)
    np.testing.assert_array_equal(nk.npairs, true_npairs)
    np.testing.assert_allclose(nk.weight, true_weight, rtol=1.e-6, atol=1.e-8)
    np.testing.assert_allclose(nk.xi, true_xi, rtol=1.e-6, atol=1.e-8)


@timer
def test_single():
    # Use kappa(r) = kappa0 exp(-r^2/2r0^2) (1-r^2/2r0^2) around a single lens

    nsource = 100000
    kappa0 = 0.05
    r0 = 10.
    L = 5. * r0
    rng = np.random.RandomState(8675309)
    x = (rng.random_sample(nsource)-0.5) * L
    y = (rng.random_sample(nsource)-0.5) * L
    r2 = (x**2 + y**2)
    k = kappa0 * np.exp(-0.5*r2/r0**2) * (1.-0.5*r2/r0**2)

    lens_cat = treecorr.Catalog(x=[0], y=[0], x_units='arcmin', y_units='arcmin')
    source_cat = treecorr.Catalog(x=x, y=y, k=k, x_units='arcmin', y_units='arcmin')
    nk = treecorr.NKCorrelation(bin_size=0.1, min_sep=1., max_sep=25., sep_units='arcmin',
                                verbose=1)
    nk.process(lens_cat, source_cat)

    r = nk.meanr
    true_k = kappa0 * np.exp(-0.5*r**2/r0**2) * (1.-0.5*r**2/r0**2)

    print('nk.xi = ',nk.xi)
    print('true_kappa = ',true_k)
    print('ratio = ',nk.xi / true_k)
    print('diff = ',nk.xi - true_k)
    print('max diff = ',max(abs(nk.xi - true_k)))
    # Note: there is a zero crossing, so need to include atol as well as rtol
    np.testing.assert_allclose(nk.xi, true_k, rtol=1.e-2, atol=1.e-4)

    # Check that we get the same result using the corr2 function
    lens_cat.write(os.path.join('data','nk_single_lens.dat'))
    source_cat.write(os.path.join('data','nk_single_source.dat'))
    config = treecorr.read_config('configs/nk_single.yaml')
    config['verbose'] = 0
    treecorr.corr2(config)
    corr2_output = np.genfromtxt(os.path.join('output','nk_single.out'), names=True,
                                 skip_header=1)
    print('nk.xi = ',nk.xi)
    print('from corr2 output = ',corr2_output['kappa'])
    print('ratio = ',corr2_output['kappa']/nk.xi)
    print('diff = ',corr2_output['kappa']-nk.xi)
    np.testing.assert_allclose(corr2_output['kappa'], nk.xi, rtol=1.e-3)

    # There is special handling for single-row catalogs when using np.genfromtxt rather
    # than pandas.  So mock it up to make sure we test it.
    treecorr.Catalog._emitted_pandas_warning = False  # Reset this, in case already triggered.
    with mock.patch.dict(sys.modules, {'pandas':None}):
        with CaptureLog() as cl:
            treecorr.corr2(config, logger=cl.logger)
        assert "Unable to import pandas" in cl.output
    corr2_output = np.genfromtxt(os.path.join('output','nk_single.out'), names=True,
                                 skip_header=1)
    np.testing.assert_allclose(corr2_output['kappa'], nk.xi, rtol=1.e-3)


@timer
def test_nk():
    # Use kappa(r) = kappa0 exp(-r^2/2r0^2) (1-r^2/2r0^2) around many lenses.

    nlens = 1000
    nsource = 100000
    kappa0 = 0.05
    r0 = 10.
    L = 100. * r0
    rng = np.random.RandomState(8675309)
    xl = (rng.random_sample(nlens)-0.5) * L
    yl = (rng.random_sample(nlens)-0.5) * L
    xs = (rng.random_sample(nsource)-0.5) * L
    ys = (rng.random_sample(nsource)-0.5) * L
    k = np.zeros( (nsource,) )
    for x,y in zip(xl,yl):
        dx = xs-x
        dy = ys-y
        r2 = dx**2 + dy**2
        k += kappa0 * np.exp(-0.5*r2/r0**2) * (1.-0.5*r2/r0**2)

    lens_cat = treecorr.Catalog(x=xl, y=yl, x_units='arcmin', y_units='arcmin')
    source_cat = treecorr.Catalog(x=xs, y=ys, k=k, x_units='arcmin', y_units='arcmin')
    nk = treecorr.NKCorrelation(bin_size=0.1, min_sep=1., max_sep=20., sep_units='arcmin',
                                verbose=1)
    t0 = time.time()
    nk.process(lens_cat, source_cat, num_threads=1)
    t1 = time.time()
    print('Time for nk process = ',t1-t0)

    # Using nbins=None rather than omitting nbins is equivalent.
    nk2 = treecorr.NKCorrelation(bin_size=0.1, min_sep=1., max_sep=20., nbins=None, sep_units='arcmin')
    nk2.process(lens_cat, source_cat, num_threads=1)
    assert nk2 == nk

    # log(<R>) != <logR>, but it should be close:
    print('meanlogr - log(meanr) = ',nk.meanlogr - np.log(nk.meanr))
    np.testing.assert_allclose(nk.meanlogr, np.log(nk.meanr), atol=1.e-3)

    t2 = time.time()
    nk2.process(lens_cat, source_cat, num_threads=1, corr_only=True)
    t3 = time.time()
    print('Time for corr-only nk process = ',t3-t2)
    np.testing.assert_allclose(nk2.xi, nk.xi)
    np.testing.assert_allclose(nk2.weight, nk.weight)
    np.testing.assert_allclose(nk2.npairs, nk.npairs)
    if __name__ == '__main__':
        assert t3-t2 < t1-t0

    r = nk.meanr
    true_k = kappa0 * np.exp(-0.5*r**2/r0**2) * (1.-0.5*r**2/r0**2)

    print('nk.xi = ',nk.xi)
    print('true_kappa = ',true_k)
    print('ratio = ',nk.xi / true_k)
    print('diff = ',nk.xi - true_k)
    print('max diff = ',max(abs(nk.xi - true_k)))
    np.testing.assert_allclose(nk.xi, true_k, rtol=0.1, atol=2.e-3)

    nrand = nlens * 10
    xr = (rng.random_sample(nrand)-0.5) * L
    yr = (rng.random_sample(nrand)-0.5) * L
    rand_cat = treecorr.Catalog(x=xr, y=yr, x_units='arcmin', y_units='arcmin')
    rk = treecorr.NKCorrelation(bin_size=0.1, min_sep=1., max_sep=20., sep_units='arcmin',
                                verbose=1)
    rk.process(rand_cat, source_cat)
    print('rk.xi = ',rk.xi)
    xi, varxi = nk.calculateXi(rk=rk)
    print('compensated xi = ',xi)
    print('true_kappa = ',true_k)
    print('ratio = ',xi / true_k)
    print('diff = ',xi - true_k)
    print('max diff = ',max(abs(xi - true_k)))
    # It turns out this doesn't come out much better.  I think the imprecision is mostly just due
    # to the smallish number of lenses, not to edge effects
    np.testing.assert_allclose(nk.xi, true_k, rtol=0.05, atol=1.e-3)

    # Check that we get the same result using the corr2 function
    try:
        import fitsio
    except ImportError:
        pass
    else:
        lens_cat.write(os.path.join('data','nk_lens.fits'))
        source_cat.write(os.path.join('data','nk_source.fits'))
        rand_cat.write(os.path.join('data','nk_rand.fits'))
        config = treecorr.read_config('configs/nk.yaml')
        config['verbose'] = 0
        config['precision'] = 8
        treecorr.corr2(config)
        corr2_output = np.genfromtxt(os.path.join('output','nk.out'), names=True, skip_header=1)
        print('nk.xi = ',nk.xi)
        print('xi = ',xi)
        print('from corr2 output = ',corr2_output['kappa'])
        print('ratio = ',corr2_output['kappa']/xi)
        print('diff = ',corr2_output['kappa']-xi)
        np.testing.assert_allclose(corr2_output['kappa'], xi)

        # In the corr2 context, you can turn off the compensated bit, even if there are randoms
        # (e.g. maybe you only want randoms for some nn calculation, but not nk.)
        config['nk_statistic'] = 'simple'
        treecorr.corr2(config)
        corr2_output = np.genfromtxt(os.path.join('output','nk.out'), names=True, skip_header=1)
        xi_simple, _ = nk.calculateXi()
        np.testing.assert_equal(xi_simple, nk.xi)
        np.testing.assert_allclose(corr2_output['kappa'], xi_simple)

        # Check the fits write option
        out_file_name1 = os.path.join('output','nk_out1.fits')
        nk.write(out_file_name1)
        data = fitsio.read(out_file_name1)
        np.testing.assert_almost_equal(data['r_nom'], np.exp(nk.logr))
        np.testing.assert_almost_equal(data['meanr'], nk.meanr)
        np.testing.assert_almost_equal(data['meanlogr'], nk.meanlogr)
        np.testing.assert_almost_equal(data['kappa'], nk.xi)
        np.testing.assert_almost_equal(data['sigma'], np.sqrt(nk.varxi))
        np.testing.assert_almost_equal(data['weight'], nk.weight)
        np.testing.assert_almost_equal(data['npairs'], nk.npairs)

        out_file_name2 = os.path.join('output','nk_out2.fits')
        nk.write(out_file_name2, rk=rk)
        data = fitsio.read(out_file_name2)
        np.testing.assert_almost_equal(data['r_nom'], np.exp(nk.logr))
        np.testing.assert_almost_equal(data['meanr'], nk.meanr)
        np.testing.assert_almost_equal(data['meanlogr'], nk.meanlogr)
        np.testing.assert_almost_equal(data['kappa'], xi)
        np.testing.assert_almost_equal(data['sigma'], np.sqrt(varxi))
        np.testing.assert_almost_equal(data['weight'], nk.weight)
        np.testing.assert_almost_equal(data['npairs'], nk.npairs)

        # Check the read function
        nk2 = treecorr.NKCorrelation.from_file(out_file_name2)
        np.testing.assert_almost_equal(nk2.logr, nk.logr)
        np.testing.assert_almost_equal(nk2.meanr, nk.meanr)
        np.testing.assert_almost_equal(nk2.meanlogr, nk.meanlogr)
        np.testing.assert_almost_equal(nk2.xi, nk.xi)
        np.testing.assert_almost_equal(nk2.varxi, nk.varxi)
        np.testing.assert_almost_equal(nk2.weight, nk.weight)
        np.testing.assert_almost_equal(nk2.npairs, nk.npairs)
        assert nk2.coords == nk.coords
        assert nk2.metric == nk.metric
        assert nk2.sep_units == nk.sep_units
        assert nk2.bin_type == nk.bin_type


@timer
def test_varxi():
    # Test that varxi is correct (or close) based on actual variance of many runs.

    kappa0 = 0.05
    r0 = 10.
    L = 10 * r0
    rng = np.random.RandomState(8675309)

    # Note: to get a good estimate of var(xi), you need a lot of runs.  The number of
    # runs matters much more than the number of galaxies for getting this to pass.
    # In addition, I found that the variance was significantly underestimated when there
    # were lots of lenses.  I guess because there were multiple lenses that paired with the
    # same sources in a given bin, which increased the variance of the mean <kappa>.
    # So there might be some adjustment that would help improve the estimate of varxi,
    # but at least this unit test shows that it's fairly accurate for *some* scenario.
    nsource = 1000
    nrand = 10
    nruns = 50000
    lens = treecorr.Catalog(x=[0], y=[0])

    file_name = 'data/test_varxi_nk.npz'
    print(file_name)
    if not os.path.isfile(file_name):
        all_nks = []
        all_rks = []
        for run in range(nruns):
            print(f'{run}/{nruns}')
            x2 = (rng.random_sample(nsource)-0.5) * L
            y2 = (rng.random_sample(nsource)-0.5) * L
            x3 = (rng.random_sample(nrand)-0.5) * L
            y3 = (rng.random_sample(nrand)-0.5) * L

            r2 = (x2**2 + y2**2)/r0**2
            k = kappa0 * np.exp(-r2/2.) * (1.-r2/2.)
            k += rng.normal(0, 0.1, size=nsource)
            # Varied weights are hard, but at least check that non-unit weights work correctly.
            w = np.ones_like(x2) * 5

            source = treecorr.Catalog(x=x2, y=y2, w=w, k=k)
            rand = treecorr.Catalog(x=x3, y=y3)
            nk = treecorr.NKCorrelation(bin_size=0.3, min_sep=6., max_sep=15., angle_slop=0.3)
            rk = treecorr.NKCorrelation(bin_size=0.3, min_sep=6., max_sep=15., angle_slop=0.3)
            nk.process(lens, source)
            rk.process(rand, source)
            all_nks.append(nk)
            all_rks.append(rk)

        all_xis = [nk.calculateXi() for nk in all_nks]
        var_xi_1 = np.var([xi[0] for xi in all_xis], axis=0)
        mean_varxi_1 = np.mean([xi[1] for xi in all_xis], axis=0)

        all_xis = [nk.calculateXi(rk=rk) for (nk,rk) in zip(all_nks, all_rks)]
        var_xi_2 = np.var([xi[0] for xi in all_xis], axis=0)
        mean_varxi_2 = np.mean([xi[1] for xi in all_xis], axis=0)

        np.savez(file_name,
                 var_xi_1=var_xi_1, mean_varxi_1=mean_varxi_1,
                 var_xi_2=var_xi_2, mean_varxi_2=mean_varxi_2)

    data = np.load(file_name)
    mean_varxi_1 = data['mean_varxi_1']
    var_xi_1 = data['var_xi_1']
    mean_varxi_2 = data['mean_varxi_2']
    var_xi_2 = data['var_xi_2']

    print('nruns = ',nruns)
    print('Uncompensated:')
    print('mean_varxi = ',mean_varxi_1)
    print('var_xi = ',var_xi_1)
    print('ratio = ',var_xi_1 / mean_varxi_1)
    print('max relerr for xi = ',np.max(np.abs((var_xi_1 - mean_varxi_1)/var_xi_1)))
    print('diff = ',var_xi_1 - mean_varxi_1)
    np.testing.assert_allclose(mean_varxi_1, var_xi_1, rtol=0.02)

    print('Compensated:')
    print('mean_varxi = ',mean_varxi_2)
    print('var_xi = ',var_xi_2)
    print('ratio = ',var_xi_2 / mean_varxi_2)
    print('max relerr for xi = ',np.max(np.abs((var_xi_2 - mean_varxi_2)/var_xi_2)))
    print('diff = ',var_xi_2 - mean_varxi_2)
    np.testing.assert_allclose(mean_varxi_2, var_xi_2, rtol=0.04)

    # Now the actual test that's based on current code, not just from the saved file.
    # There is a bit more noise on a singe run, so the tolerance needs to be somewhat higher.
    x2 = (rng.random_sample(nsource)-0.5) * L
    y2 = (rng.random_sample(nsource)-0.5) * L
    x3 = (rng.random_sample(nrand)-0.5) * L
    y3 = (rng.random_sample(nrand)-0.5) * L

    r2 = (x2**2 + y2**2)/r0**2
    k = kappa0 * np.exp(-r2/2.) * (1.-r2/2.)
    k += rng.normal(0, 0.1, size=nsource)
    w = np.ones_like(x2) * 5

    source = treecorr.Catalog(x=x2, y=y2, w=w, k=k)
    rand = treecorr.Catalog(x=x3, y=y3)
    nk = treecorr.NKCorrelation(bin_size=0.3, min_sep=6., max_sep=15., angle_slop=0.3)
    rk = treecorr.NKCorrelation(bin_size=0.3, min_sep=6., max_sep=15., angle_slop=0.3)
    nk.process(lens, source)
    rk.process(rand, source)

    print('single run:')
    print('Uncompensated')
    print('ratio = ',nk.varxi / var_xi_1)
    print('max relerr for xi = ',np.max(np.abs((nk.varxi - var_xi_1)/var_xi_1)))
    np.testing.assert_allclose(nk.varxi, var_xi_1, rtol=0.3)

    xi, varxi = nk.calculateXi(rk=rk)
    print('Compensated')
    print('ratio = ',varxi / var_xi_2)
    print('max relerr for xi = ',np.max(np.abs((varxi - var_xi_2)/var_xi_2)))
    np.testing.assert_allclose(varxi, var_xi_2, rtol=0.3)


if __name__ == '__main__':
    test_direct()
    test_direct_spherical()
    test_single()
    test_nk()
    test_varxi()
