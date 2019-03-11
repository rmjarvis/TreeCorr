# Copyright (c) 2003-2015 by Mike Jarvis
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
import treecorr
import os
import sys
import fitsio
import coord

from test_helper import get_script_name, do_pickle, CaptureLog, assert_raises

def test_direct():
    # If the catalogs are small enough, we can do a direct calculation to see if comes out right.
    # This should exactly match the treecorr result if bin_slop=0.

    ngal = 200
    s = 10.
    np.random.seed(8675309)
    x1 = np.random.normal(0,s, (ngal,) )
    y1 = np.random.normal(0,s, (ngal,) )
    w1 = np.random.random(ngal)

    x2 = np.random.normal(0,s, (ngal,) )
    y2 = np.random.normal(0,s, (ngal,) )
    w2 = np.random.random(ngal)
    k2 = np.random.normal(0,3, (ngal,) )

    cat1 = treecorr.Catalog(x=x1, y=y1, w=w1)
    cat2 = treecorr.Catalog(x=x2, y=y2, w=w2, k=k2)

    min_sep = 1.
    max_sep = 50.
    nbins = 50
    bin_size = np.log(max_sep/min_sep) / nbins
    nk = treecorr.NKCorrelation(min_sep=min_sep, max_sep=max_sep, nbins=nbins, bin_slop=0.)
    nk.process(cat1, cat2)

    true_npairs = np.zeros(nbins, dtype=int)
    true_weight = np.zeros(nbins, dtype=float)
    true_xi = np.zeros(nbins, dtype=float)
    for i in range(ngal):
        # It's hard to do all the pairs at once with numpy operations (although maybe possible).
        # But we can at least do all the pairs for each entry in cat1 at once with arrays.
        rsq = (x1[i]-x2)**2 + (y1[i]-y2)**2
        r = np.sqrt(rsq)
        logr = np.log(r)

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
    np.testing.assert_allclose(nk.weight, true_weight, rtol=1.e-5, atol=1.e-8)

    print('true_xi = ',true_xi)
    print('nk.xi = ',nk.xi)
    np.testing.assert_allclose(nk.xi, true_xi, rtol=1.e-4, atol=1.e-8)

    # Check that running via the corr2 script works correctly.
    config = treecorr.config.read_config('configs/nk_direct.yaml')
    cat1.write(config['file_name'])
    cat2.write(config['file_name2'])
    treecorr.corr2(config)
    data = fitsio.read(config['nk_file_name'])
    np.testing.assert_allclose(data['R_nom'], nk.rnom)
    np.testing.assert_allclose(data['npairs'], nk.npairs)
    np.testing.assert_allclose(data['weight'], nk.weight)
    np.testing.assert_allclose(data['kappa'], nk.xi, rtol=1.e-3)

    # Repeat with binslop not precisely 0, since the code flow is different for bin_slop == 0.
    # And don't do any top-level recursion so we actually test not going to the leaves.
    nk = treecorr.NKCorrelation(min_sep=min_sep, max_sep=max_sep, nbins=nbins, bin_slop=1.e-16,
                                max_top=0)
    nk.process(cat1, cat2)
    np.testing.assert_array_equal(nk.npairs, true_npairs)
    np.testing.assert_allclose(nk.weight, true_weight, rtol=1.e-5, atol=1.e-8)
    np.testing.assert_allclose(nk.xi, true_xi, rtol=1.e-4, atol=1.e-8)

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
    nk3 = treecorr.NKCorrelation(min_sep=min_sep, max_sep=max_sep, nbins=nbins)
    nk3.read(ascii_name)
    np.testing.assert_allclose(nk3.npairs, nk.npairs)
    np.testing.assert_allclose(nk3.weight, nk.weight)
    np.testing.assert_allclose(nk3.meanr, nk.meanr)
    np.testing.assert_allclose(nk3.meanlogr, nk.meanlogr)
    np.testing.assert_allclose(nk3.xi, nk.xi)

    fits_name = 'output/nk_fits.fits'
    nk.write(fits_name)
    nk4 = treecorr.NKCorrelation(min_sep=min_sep, max_sep=max_sep, nbins=nbins)
    nk4.read(fits_name)
    np.testing.assert_allclose(nk4.npairs, nk.npairs)
    np.testing.assert_allclose(nk4.weight, nk.weight)
    np.testing.assert_allclose(nk4.meanr, nk.meanr)
    np.testing.assert_allclose(nk4.meanlogr, nk.meanlogr)
    np.testing.assert_allclose(nk4.xi, nk.xi)


def test_direct_spherical():
    # Repeat in spherical coords

    ngal = 100
    s = 10.
    np.random.seed(8675309)
    x1 = np.random.normal(0,s, (ngal,) )
    y1 = np.random.normal(0,s, (ngal,) ) + 200  # Put everything at large y, so small angle on sky
    z1 = np.random.normal(0,s, (ngal,) )
    w1 = np.random.random(ngal)

    x2 = np.random.normal(0,s, (ngal,) )
    y2 = np.random.normal(0,s, (ngal,) ) + 200
    z2 = np.random.normal(0,s, (ngal,) )
    w2 = np.random.random(ngal)
    k2 = np.random.normal(0,3, (ngal,) )

    ra1, dec1 = coord.CelestialCoord.xyz_to_radec(x1,y1,z1)
    ra2, dec2 = coord.CelestialCoord.xyz_to_radec(x2,y2,z2)

    cat1 = treecorr.Catalog(ra=ra1, dec=dec1, ra_units='rad', dec_units='rad', w=w1)
    cat2 = treecorr.Catalog(ra=ra2, dec=dec2, ra_units='rad', dec_units='rad', w=w2, k=k2)

    min_sep = 1.
    max_sep = 10.
    nbins = 50
    bin_size = np.log(max_sep/min_sep) / nbins
    nk = treecorr.NKCorrelation(min_sep=min_sep, max_sep=max_sep, nbins=nbins,
                                sep_units='deg', bin_slop=0.)
    nk.process(cat1, cat2)

    r1 = np.sqrt(x1**2 + y1**2 + z1**2)
    r2 = np.sqrt(x2**2 + y2**2 + z2**2)
    x1 /= r1;  y1 /= r1;  z1 /= r1
    x2 /= r2;  y2 /= r2;  z2 /= r2

    north_pole = coord.CelestialCoord(0*coord.radians, 90*coord.degrees)

    true_npairs = np.zeros(nbins, dtype=int)
    true_weight = np.zeros(nbins, dtype=float)
    true_xi = np.zeros(nbins, dtype=float)

    c1 = [coord.CelestialCoord(r*coord.radians, d*coord.radians) for (r,d) in zip(ra1, dec1)]
    c2 = [coord.CelestialCoord(r*coord.radians, d*coord.radians) for (r,d) in zip(ra2, dec2)]
    for i in range(ngal):
        for j in range(ngal):
            rsq = (x1[i]-x2[j])**2 + (y1[i]-y2[j])**2 + (z1[i]-z2[j])**2
            r = np.sqrt(rsq)
            r *= coord.radians / coord.degrees
            logr = np.log(r)

            index = np.floor(np.log(r/min_sep) / bin_size).astype(int)
            if index < 0 or index >= nbins:
                continue

            # Rotate shears to coordinates where line connecting is horizontal.
            # Original orientation is where north is up.
            theta1 = 90*coord.degrees - c1[i].angleBetween(north_pole, c2[j])
            theta2 = 90*coord.degrees - c2[j].angleBetween(c1[i], north_pole)
            exp2theta1 = np.cos(2*theta1) + 1j * np.sin(2*theta1)
            expm2theta2 = np.cos(2*theta2) - 1j * np.sin(2*theta2)

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
    cat1.write(config['file_name'])
    cat2.write(config['file_name2'])
    treecorr.corr2(config)
    data = fitsio.read(config['nk_file_name'])
    np.testing.assert_allclose(data['R_nom'], nk.rnom)
    np.testing.assert_allclose(data['npairs'], nk.npairs)
    np.testing.assert_allclose(data['weight'], nk.weight)
    np.testing.assert_allclose(data['kappa'], nk.xi, rtol=1.e-3)

    # Repeat with binslop not precisely 0, since the code flow is different for bin_slop == 0.
    # And don't do any top-level recursion so we actually test not going to the leaves.
    nk = treecorr.NKCorrelation(min_sep=min_sep, max_sep=max_sep, nbins=nbins,
                                sep_units='deg', bin_slop=1.e-16, max_top=0)
    nk.process(cat1, cat2)
    np.testing.assert_array_equal(nk.npairs, true_npairs)
    np.testing.assert_allclose(nk.weight, true_weight, rtol=1.e-5, atol=1.e-8)
    np.testing.assert_allclose(nk.xi, true_xi, rtol=1.e-3, atol=1.e-6)


def test_pairwise():
    # Test the pairwise option.

    ngal = 1000
    s = 10.
    np.random.seed(8675309)
    x1 = np.random.normal(0,s, (ngal,) )
    y1 = np.random.normal(0,s, (ngal,) )
    w1 = np.random.random(ngal)

    x2 = np.random.normal(0,s, (ngal,) )
    y2 = np.random.normal(0,s, (ngal,) )
    w2 = np.random.random(ngal)
    k2 = np.random.normal(0,3, (ngal,) )

    w1 = np.ones_like(w1)
    w2 = np.ones_like(w2)

    cat1 = treecorr.Catalog(x=x1, y=y1, w=w1)
    cat2 = treecorr.Catalog(x=x2, y=y2, w=w2, k=k2)

    min_sep = 5.
    max_sep = 50.
    nbins = 10
    bin_size = np.log(max_sep/min_sep) / nbins
    nk = treecorr.NKCorrelation(min_sep=min_sep, max_sep=max_sep, nbins=nbins)
    nk.process_pairwise(cat1, cat2)
    nk.finalize(cat2.vark)

    true_npairs = np.zeros(nbins, dtype=int)
    true_weight = np.zeros(nbins, dtype=float)
    true_xi = np.zeros(nbins, dtype=float)

    rsq = (x1-x2)**2 + (y1-y2)**2
    r = np.sqrt(rsq)
    logr = np.log(r)

    ww = w1 * w2
    xi = ww * k2

    index = np.floor(np.log(r/min_sep) / bin_size).astype(int)
    mask = (index >= 0) & (index < nbins)
    np.add.at(true_npairs, index[mask], 1)
    np.add.at(true_weight, index[mask], ww[mask])
    np.add.at(true_xi, index[mask], xi[mask])

    true_xi /= true_weight

    np.testing.assert_array_equal(nk.npairs, true_npairs)
    np.testing.assert_allclose(nk.weight, true_weight, rtol=1.e-5, atol=1.e-8)
    np.testing.assert_allclose(nk.xi, true_xi, rtol=1.e-4, atol=1.e-8)

    # If cats have names, then the logger will mention them.
    # Also, test running with optional args.
    cat1.name = "first"
    cat2.name = "second"
    with CaptureLog() as cl:
        nk.logger = cl.logger
        nk.process_pairwise(cat1, cat2, metric='Euclidean', num_threads=2)
    assert "for cats first, second" in cl.output


def test_single():
    # Use kappa(r) = kappa0 exp(-r^2/2r0^2) (1-r^2/2r0^2) around a single lens

    nsource = 100000
    kappa0 = 0.05
    r0 = 10.
    L = 5. * r0
    np.random.seed(8675309)
    x = (np.random.random_sample(nsource)-0.5) * L
    y = (np.random.random_sample(nsource)-0.5) * L
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
    if sys.version_info < (3,): return  # mock only available on python 3
    from unittest import mock
    with mock.patch.dict(sys.modules, {'pandas':None}):
        with CaptureLog() as cl:
            treecorr.corr2(config, logger=cl.logger)
        assert "Unable to import pandas" in cl.output
    corr2_output = np.genfromtxt(os.path.join('output','nk_single.out'), names=True,
                                    skip_header=1)
    np.testing.assert_allclose(corr2_output['kappa'], nk.xi, rtol=1.e-3)


def test_nk():
    # Use kappa(r) = kappa0 exp(-r^2/2r0^2) (1-r^2/2r0^2) around many lenses.

    nlens = 1000
    nsource = 100000
    kappa0 = 0.05
    r0 = 10.
    L = 100. * r0
    np.random.seed(8675309)
    xl = (np.random.random_sample(nlens)-0.5) * L
    yl = (np.random.random_sample(nlens)-0.5) * L
    xs = (np.random.random_sample(nsource)-0.5) * L
    ys = (np.random.random_sample(nsource)-0.5) * L
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
    nk.process(lens_cat, source_cat)

    # log(<R>) != <logR>, but it should be close:
    print('meanlogr - log(meanr) = ',nk.meanlogr - np.log(nk.meanr))
    np.testing.assert_allclose(nk.meanlogr, np.log(nk.meanr), atol=1.e-3)

    r = nk.meanr
    true_k = kappa0 * np.exp(-0.5*r**2/r0**2) * (1.-0.5*r**2/r0**2)

    print('nk.xi = ',nk.xi)
    print('true_kappa = ',true_k)
    print('ratio = ',nk.xi / true_k)
    print('diff = ',nk.xi - true_k)
    print('max diff = ',max(abs(nk.xi - true_k)))
    np.testing.assert_allclose(nk.xi, true_k, rtol=0.1, atol=2.e-3)

    nrand = nlens * 13
    xr = (np.random.random_sample(nrand)-0.5) * L
    yr = (np.random.random_sample(nrand)-0.5) * L
    rand_cat = treecorr.Catalog(x=xr, y=yr, x_units='arcmin', y_units='arcmin')
    rk = treecorr.NKCorrelation(bin_size=0.1, min_sep=1., max_sep=20., sep_units='arcmin',
                                verbose=1)
    rk.process(rand_cat, source_cat)
    print('rk.xi = ',rk.xi)
    xi, varxi = nk.calculateXi(rk)
    print('compensated xi = ',xi)
    print('true_kappa = ',true_k)
    print('ratio = ',xi / true_k)
    print('diff = ',xi - true_k)
    print('max diff = ',max(abs(xi - true_k)))
    # It turns out this doesn't come out much better.  I think the imprecision is mostly just due
    # to the smallish number of lenses, not to edge effects
    np.testing.assert_allclose(nk.xi, true_k, rtol=0.05, atol=1.e-3)

    # Check that we get the same result using the corr2 function
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
    np.testing.assert_allclose(corr2_output['kappa'], xi, rtol=1.e-3)

    # In the corr2 context, you can turn off the compensated bit, even if there are randoms
    # (e.g. maybe you only want randoms for some nn calculation, but not nk.)
    config['nk_statistic'] = 'simple'
    treecorr.corr2(config)
    corr2_output = np.genfromtxt(os.path.join('output','nk.out'), names=True, skip_header=1)
    xi_simple, _ = nk.calculateXi()
    np.testing.assert_allclose(corr2_output['kappa'], xi_simple, rtol=1.e-3)

    # Check the fits write option
    out_file_name1 = os.path.join('output','nk_out1.fits')
    nk.write(out_file_name1)
    data = fitsio.read(out_file_name1)
    np.testing.assert_almost_equal(data['R_nom'], np.exp(nk.logr))
    np.testing.assert_almost_equal(data['meanR'], nk.meanr)
    np.testing.assert_almost_equal(data['meanlogR'], nk.meanlogr)
    np.testing.assert_almost_equal(data['kappa'], nk.xi)
    np.testing.assert_almost_equal(data['sigma'], np.sqrt(nk.varxi))
    np.testing.assert_almost_equal(data['weight'], nk.weight)
    np.testing.assert_almost_equal(data['npairs'], nk.npairs)

    out_file_name2 = os.path.join('output','nk_out2.fits')
    nk.write(out_file_name2, rk)
    data = fitsio.read(out_file_name2)
    np.testing.assert_almost_equal(data['R_nom'], np.exp(nk.logr))
    np.testing.assert_almost_equal(data['meanR'], nk.meanr)
    np.testing.assert_almost_equal(data['meanlogR'], nk.meanlogr)
    np.testing.assert_almost_equal(data['kappa'], xi)
    np.testing.assert_almost_equal(data['sigma'], np.sqrt(varxi))
    np.testing.assert_almost_equal(data['weight'], nk.weight)
    np.testing.assert_almost_equal(data['npairs'], nk.npairs)

    # Check the read function
    nk2 = treecorr.NKCorrelation(bin_size=0.1, min_sep=1., max_sep=20., sep_units='arcmin')
    nk2.read(out_file_name1)
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


def test_varxi():
    # Test that varxi is correct (or close) based on actual variance of many runs.

    kappa0 = 0.05
    r0 = 10.
    L = 10 * r0
    np.random.seed(8675309)

    # Note: to get a good estimate of var(xi), you need a lot of runs.  The number of
    # runs matters much more than the number of galaxies for getting this to pass.
    # In addition, I found that the variance was significantly underestimated when there
    # were lots of lenses.  I guess because there were multiple lenses that paired with the
    # same sources in a given bin, which increased the variance of the mean <g>.
    # So there might be some adjustment that would help improve the estimate of varxi,
    # but at least this unit test shows that it's fairly accurate for *some* scenario.
    if __name__ == '__main__':
        nlens = 1
        nsource = 1000
        nrand = 10
        nruns = 50000
        tol_factor = 1
    else:
        nlens = 1
        nsource = 100
        nrand = 2
        nruns = 5000
        tol_factor = 5

    lens = treecorr.Catalog(x=[0], y=[0])
    all_nks = []
    all_rks = []
    for run in range(nruns):
        x2 = (np.random.random_sample(nsource)-0.5) * L
        y2 = (np.random.random_sample(nsource)-0.5) * L
        x3 = (np.random.random_sample(nrand)-0.5) * L
        y3 = (np.random.random_sample(nrand)-0.5) * L

        r2 = (x2**2 + y2**2)/r0**2
        k = kappa0 * np.exp(-r2/2.) * (1.-r2/2.)
        k += np.random.normal(0, 0.1, size=nsource)

        source = treecorr.Catalog(x=x2, y=y2, k=k)
        rand = treecorr.Catalog(x=x3, y=y3)
        nk = treecorr.NKCorrelation(bin_size=0.1, min_sep=6., max_sep=15.)
        rk = treecorr.NKCorrelation(bin_size=0.1, min_sep=6., max_sep=15.)
        nk.process(lens, source)
        rk.process(rand, source)
        all_nks.append(nk)
        all_rks.append(rk)
        print('.', end='', flush=True)
    print()

    print('Uncompensated:')

    all_xis = [nk.calculateXi() for nk in all_nks]
    mean_wt = np.mean([nk.weight for nk in all_nks], axis=0)
    mean_xi = np.mean([xi[0] for xi in all_xis], axis=0)
    var_xi = np.var([xi[0] for xi in all_xis], axis=0)
    mean_varxi = np.mean([xi[1] for xi in all_xis], axis=0)

    print('mean_xi = ',mean_xi)
    print('mean_wt = ',mean_wt)
    print('mean_varxi = ',mean_varxi)
    print('var_xi = ',var_xi)
    print('ratio = ',var_xi / mean_varxi)
    print('max relerr for xi = ',np.max(np.abs((var_xi - mean_varxi)/var_xi)))
    print('diff = ',var_xi - mean_varxi)
    np.testing.assert_allclose(mean_varxi, var_xi, rtol=0.02 * tol_factor)

    print('Compensated:')

    all_xis = [nk.calculateXi(rk) for (nk,rk) in zip(all_nks, all_rks)]
    mean_wt = np.mean([nk.weight for nk in all_nks], axis=0)
    mean_xi = np.mean([xi[0] for xi in all_xis], axis=0)
    var_xi = np.var([xi[0] for xi in all_xis], axis=0)
    mean_varxi = np.mean([xi[1] for xi in all_xis], axis=0)

    print('mean_xi = ',mean_xi)
    print('mean_wt = ',mean_wt)
    print('mean_varxi = ',mean_varxi)
    print('var_xi = ',var_xi)
    print('ratio = ',var_xi / mean_varxi)
    print('max relerr for xi = ',np.max(np.abs((var_xi - mean_varxi)/var_xi)))
    print('diff = ',var_xi - mean_varxi)
    # Unlike for NG, the agreement is slightly worse for the compensated case.
    # Not sure if this is telling me something important, or just the way it turned out.
    np.testing.assert_allclose(mean_varxi, var_xi, rtol=0.03 * tol_factor)


if __name__ == '__main__':
    test_single()
    test_nk()
    test_varxi()
