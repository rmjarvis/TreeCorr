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
import treecorr
import os
import sys
import coord
import time
from unittest import mock

from test_helper import do_pickle, CaptureLog
from test_helper import assert_raises, timer, assert_warns

@timer
def test_direct():
    # If the catalogs are small enough, we can do a direct calculation to see if comes out right.
    # This should exactly match the treecorr result if brute=True.

    ngal = 200
    s = 10.
    rng = np.random.RandomState(8675309)
    x1 = rng.normal(0,s, (ngal,) )
    y1 = rng.normal(0,s, (ngal,) )
    w1 = rng.random_sample(ngal)

    x2 = rng.normal(0,s, (ngal,) )
    y2 = rng.normal(0,s, (ngal,) )
    w2 = rng.random_sample(ngal)
    g12 = rng.normal(0,0.2, (ngal,) )
    g22 = rng.normal(0,0.2, (ngal,) )

    cat1 = treecorr.Catalog(x=x1, y=y1, w=w1)
    cat2 = treecorr.Catalog(x=x2, y=y2, w=w2, g1=g12, g2=g22)

    min_sep = 1.
    max_sep = 50.
    nbins = 50
    bin_size = np.log(max_sep/min_sep) / nbins
    ng = treecorr.NGCorrelation(min_sep=min_sep, max_sep=max_sep, nbins=nbins, brute=True)
    ng.process(cat1, cat2)

    true_npairs = np.zeros(nbins, dtype=int)
    true_weight = np.zeros(nbins, dtype=float)
    true_xi = np.zeros(nbins, dtype=complex)
    for i in range(ngal):
        # It's hard to do all the pairs at once with numpy operations (although maybe possible).
        # But we can at least do all the pairs for each entry in cat1 at once with arrays.
        rsq = (x1[i]-x2)**2 + (y1[i]-y2)**2
        r = np.sqrt(rsq)
        expmialpha = ((x1[i]-x2) - 1j*(y1[i]-y2)) / r

        ww = w1[i] * w2
        xi = -ww * (g12 + 1j*g22) * expmialpha**2

        index = np.floor(np.log(r/min_sep) / bin_size).astype(int)
        mask = (index >= 0) & (index < nbins)
        np.add.at(true_npairs, index[mask], 1)
        np.add.at(true_weight, index[mask], ww[mask])
        np.add.at(true_xi, index[mask], xi[mask])

    true_xi /= true_weight

    print('true_npairs = ',true_npairs)
    print('diff = ',ng.npairs - true_npairs)
    np.testing.assert_array_equal(ng.npairs, true_npairs)

    print('true_weight = ',true_weight)
    print('diff = ',ng.weight - true_weight)
    np.testing.assert_allclose(ng.weight, true_weight, rtol=1.e-5, atol=1.e-8)

    print('true_xi = ',true_xi)
    print('ng.xi = ',ng.xi)
    print('ng.xi_im = ',ng.xi_im)
    np.testing.assert_allclose(ng.xi, true_xi.real, rtol=1.e-4, atol=1.e-8)
    np.testing.assert_allclose(ng.xi_im, true_xi.imag, rtol=1.e-4, atol=1.e-8)

    # Check that running via the corr2 script works correctly.
    config = treecorr.config.read_config('configs/ng_direct.yaml')
    try:
        import fitsio
    except ImportError:
        pass
    else:
        cat1.write(config['file_name'])
        cat2.write(config['file_name2'])
        treecorr.corr2(config)
        data = fitsio.read(config['ng_file_name'])
        np.testing.assert_allclose(data['r_nom'], ng.rnom)
        np.testing.assert_allclose(data['npairs'], ng.npairs)
        np.testing.assert_allclose(data['weight'], ng.weight)
        np.testing.assert_allclose(data['gamT'], ng.xi, rtol=1.e-3)
        np.testing.assert_allclose(data['gamX'], ng.xi_im, rtol=1.e-3)

        # Invalid with only one file_name
        del config['file_name2']
        with assert_raises(TypeError):
            treecorr.corr2(config)
        config['file_name2'] = 'data/ng_direct_cat2.fits'
        # Invalid to request compoensated if no rand_file
        config['ng_statistic'] = 'compensated'
        with assert_raises(TypeError):
            treecorr.corr2(config)

    # Repeat with binslop = 0
    # And don't do any top-level recursion so we actually test not going to the leaves.
    ng = treecorr.NGCorrelation(min_sep=min_sep, max_sep=max_sep, nbins=nbins, bin_slop=0,
                                max_top=0)
    ng.process(cat1, cat2)
    np.testing.assert_array_equal(ng.npairs, true_npairs)
    np.testing.assert_allclose(ng.weight, true_weight, rtol=1.e-5, atol=1.e-8)
    np.testing.assert_allclose(ng.xi, true_xi.real, rtol=1.e-3, atol=3.e-4)
    np.testing.assert_allclose(ng.xi_im, true_xi.imag, atol=3.e-4)

    # Check a few basic operations with a NGCorrelation object.
    do_pickle(ng)

    ng2 = ng.copy()
    ng2 += ng
    np.testing.assert_allclose(ng2.npairs, 2*ng.npairs)
    np.testing.assert_allclose(ng2.weight, 2*ng.weight)
    np.testing.assert_allclose(ng2.meanr, 2*ng.meanr)
    np.testing.assert_allclose(ng2.meanlogr, 2*ng.meanlogr)
    np.testing.assert_allclose(ng2.xi, 2*ng.xi)
    np.testing.assert_allclose(ng2.xi_im, 2*ng.xi_im)

    ng2.clear()
    ng2 += ng
    np.testing.assert_allclose(ng2.npairs, ng.npairs)
    np.testing.assert_allclose(ng2.weight, ng.weight)
    np.testing.assert_allclose(ng2.meanr, ng.meanr)
    np.testing.assert_allclose(ng2.meanlogr, ng.meanlogr)
    np.testing.assert_allclose(ng2.xi, ng.xi)
    np.testing.assert_allclose(ng2.xi_im, ng.xi_im)

    ascii_name = 'output/ng_ascii.txt'
    ng.write(ascii_name, precision=16)
    ng3 = treecorr.NGCorrelation(min_sep=min_sep, max_sep=max_sep, nbins=nbins)
    ng3.read(ascii_name)
    np.testing.assert_allclose(ng3.npairs, ng.npairs)
    np.testing.assert_allclose(ng3.weight, ng.weight)
    np.testing.assert_allclose(ng3.meanr, ng.meanr)
    np.testing.assert_allclose(ng3.meanlogr, ng.meanlogr)
    np.testing.assert_allclose(ng3.xi, ng.xi)
    np.testing.assert_allclose(ng3.xi_im, ng.xi_im)

    try:
        import fitsio
    except ImportError:
        pass
    else:
        fits_name = 'output/ng_fits.fits'
        ng.write(fits_name)
        ng4 = treecorr.NGCorrelation(min_sep=min_sep, max_sep=max_sep, nbins=nbins)
        ng4.read(fits_name)
        np.testing.assert_allclose(ng4.npairs, ng.npairs)
        np.testing.assert_allclose(ng4.weight, ng.weight)
        np.testing.assert_allclose(ng4.meanr, ng.meanr)
        np.testing.assert_allclose(ng4.meanlogr, ng.meanlogr)
        np.testing.assert_allclose(ng4.xi, ng.xi)
        np.testing.assert_allclose(ng4.xi_im, ng.xi_im)

    with assert_raises(TypeError):
        ng2 += config
    ng4 = treecorr.NGCorrelation(min_sep=min_sep/2, max_sep=max_sep, nbins=nbins)
    with assert_raises(ValueError):
        ng2 += ng4
    ng5 = treecorr.NGCorrelation(min_sep=min_sep, max_sep=max_sep*2, nbins=nbins)
    with assert_raises(ValueError):
        ng2 += ng5
    ng6 = treecorr.NGCorrelation(min_sep=min_sep, max_sep=max_sep, nbins=nbins*2)
    with assert_raises(ValueError):
        ng2 += ng6



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
    g12 = rng.normal(0,0.2, (ngal,) )
    g22 = rng.normal(0,0.2, (ngal,) )

    ra1, dec1 = coord.CelestialCoord.xyz_to_radec(x1,y1,z1)
    ra2, dec2 = coord.CelestialCoord.xyz_to_radec(x2,y2,z2)

    cat1 = treecorr.Catalog(ra=ra1, dec=dec1, ra_units='rad', dec_units='rad', w=w1)
    cat2 = treecorr.Catalog(ra=ra2, dec=dec2, ra_units='rad', dec_units='rad', w=w2, g1=g12, g2=g22)

    min_sep = 1.
    max_sep = 10.
    nbins = 50
    bin_size = np.log(max_sep/min_sep) / nbins
    ng = treecorr.NGCorrelation(min_sep=min_sep, max_sep=max_sep, nbins=nbins,
                                sep_units='deg', brute=True)
    ng.process(cat1, cat2)

    r1 = np.sqrt(x1**2 + y1**2 + z1**2)
    r2 = np.sqrt(x2**2 + y2**2 + z2**2)
    x1 /= r1;  y1 /= r1;  z1 /= r1
    x2 /= r2;  y2 /= r2;  z2 /= r2

    north_pole = coord.CelestialCoord(0*coord.radians, 90*coord.degrees)

    true_npairs = np.zeros(nbins, dtype=int)
    true_weight = np.zeros(nbins, dtype=float)
    true_xi = np.zeros(nbins, dtype=complex)

    c1 = [coord.CelestialCoord(r*coord.radians, d*coord.radians) for (r,d) in zip(ra1, dec1)]
    c2 = [coord.CelestialCoord(r*coord.radians, d*coord.radians) for (r,d) in zip(ra2, dec2)]
    for i in range(ngal):
        for j in range(ngal):
            rsq = (x1[i]-x2[j])**2 + (y1[i]-y2[j])**2 + (z1[i]-z2[j])**2
            r = np.sqrt(rsq)
            r *= coord.radians / coord.degrees

            index = np.floor(np.log(r/min_sep) / bin_size).astype(int)
            if index < 0 or index >= nbins:
                continue

            # Rotate shears to coordinates where line connecting is horizontal.
            # Original orientation is where north is up.
            theta2 = 90*coord.degrees - c2[j].angleBetween(c1[i], north_pole)
            expm2theta2 = np.cos(2*theta2) - 1j * np.sin(2*theta2)

            g2 = g12[j] + 1j * g22[j]
            g2 *= expm2theta2

            ww = w1[i] * w2[j]
            xi = -w1[i] * w2[j] * g2

            true_npairs[index] += 1
            true_weight[index] += ww
            true_xi[index] += xi

    true_xi /= true_weight

    print('true_npairs = ',true_npairs)
    print('diff = ',ng.npairs - true_npairs)
    np.testing.assert_array_equal(ng.npairs, true_npairs)

    print('true_weight = ',true_weight)
    print('diff = ',ng.weight - true_weight)
    np.testing.assert_allclose(ng.weight, true_weight, rtol=1.e-5, atol=1.e-8)

    print('true_xi = ',true_xi)
    print('ng.xi = ',ng.xi)
    print('ng.xi_im = ',ng.xi_im)
    np.testing.assert_allclose(ng.xi, true_xi.real, rtol=1.e-4, atol=1.e-8)
    np.testing.assert_allclose(ng.xi_im, true_xi.imag, rtol=1.e-4, atol=1.e-8)

    # Check that running via the corr2 script works correctly.
    config = treecorr.config.read_config('configs/ng_direct_spherical.yaml')
    try:
        import fitsio
    except ImportError:
        pass
    else:
        cat1.write(config['file_name'])
        cat2.write(config['file_name2'])
        treecorr.corr2(config)
        data = fitsio.read(config['ng_file_name'])
        np.testing.assert_allclose(data['r_nom'], ng.rnom)
        np.testing.assert_allclose(data['npairs'], ng.npairs)
        np.testing.assert_allclose(data['weight'], ng.weight)
        np.testing.assert_allclose(data['gamT'], ng.xi, rtol=1.e-3)
        np.testing.assert_allclose(data['gamX'], ng.xi_im, rtol=1.e-3)

    # Repeat with binslop = 0
    # And don't do any top-level recursion so we actually test not going to the leaves.
    ng = treecorr.NGCorrelation(min_sep=min_sep, max_sep=max_sep, nbins=nbins,
                                sep_units='deg', bin_slop=0, max_top=0)
    ng.process(cat1, cat2)
    np.testing.assert_array_equal(ng.npairs, true_npairs)
    np.testing.assert_allclose(ng.weight, true_weight, rtol=1.e-5, atol=1.e-8)
    np.testing.assert_allclose(ng.xi, true_xi.real, rtol=1.e-3, atol=2.e-4)
    np.testing.assert_allclose(ng.xi_im, true_xi.imag, atol=2.e-4)


@timer
def test_pairwise():
    # Test the pairwise option.

    ngal = 1000
    s = 10.
    rng = np.random.RandomState(8675309)
    x1 = rng.normal(0,s, (ngal,) )
    y1 = rng.normal(0,s, (ngal,) )
    w1 = rng.random_sample(ngal)

    x2 = rng.normal(0,s, (ngal,) )
    y2 = rng.normal(0,s, (ngal,) )
    w2 = rng.random_sample(ngal)
    g12 = rng.normal(0,0.2, (ngal,) )
    g22 = rng.normal(0,0.2, (ngal,) )

    w1 = np.ones_like(w1)
    w2 = np.ones_like(w2)

    cat1 = treecorr.Catalog(x=x1, y=y1, w=w1)
    cat2 = treecorr.Catalog(x=x2, y=y2, w=w2, g1=g12, g2=g22)

    min_sep = 5.
    max_sep = 50.
    nbins = 10
    bin_size = np.log(max_sep/min_sep) / nbins
    ng = treecorr.NGCorrelation(min_sep=min_sep, max_sep=max_sep, nbins=nbins)
    with assert_warns(FutureWarning):
        ng.process_pairwise(cat1, cat2)
    ng.finalize(cat2.varg)

    true_npairs = np.zeros(nbins, dtype=int)
    true_weight = np.zeros(nbins, dtype=float)
    true_xi = np.zeros(nbins, dtype=complex)

    rsq = (x1-x2)**2 + (y1-y2)**2
    r = np.sqrt(rsq)
    expmialpha = ((x1-x2) - 1j*(y1-y2)) / r

    ww = w1 * w2
    xi = -ww * (g12 + 1j*g22) * expmialpha**2

    index = np.floor(np.log(r/min_sep) / bin_size).astype(int)
    mask = (index >= 0) & (index < nbins)
    np.add.at(true_npairs, index[mask], 1)
    np.add.at(true_weight, index[mask], ww[mask])
    np.add.at(true_xi, index[mask], xi[mask])

    true_xi /= true_weight

    np.testing.assert_array_equal(ng.npairs, true_npairs)
    np.testing.assert_allclose(ng.weight, true_weight, rtol=1.e-5, atol=1.e-8)
    np.testing.assert_allclose(ng.xi, true_xi.real, rtol=1.e-4, atol=1.e-8)
    np.testing.assert_allclose(ng.xi_im, true_xi.imag, rtol=1.e-4, atol=1.e-8)

    # If cats have names, then the logger will mention them.
    # Also, test running with optional args.
    cat1.name = "first"
    cat2.name = "second"
    with CaptureLog() as cl:
        ng.logger = cl.logger
        with assert_warns(FutureWarning):
            ng.process_pairwise(cat1, cat2, metric='Euclidean', num_threads=2)
    assert "for cats first, second" in cl.output


@timer
def test_single():
    # Use gamma_t(r) = gamma0 exp(-r^2/2r0^2) around a single lens
    # i.e. gamma(r) = -gamma0 exp(-r^2/2r0^2) (x+iy)^2/r^2

    nsource = 300000
    gamma0 = 0.05
    r0 = 10.
    L = 5. * r0
    rng = np.random.RandomState(8675309)
    x = (rng.random_sample(nsource)-0.5) * L
    y = (rng.random_sample(nsource)-0.5) * L
    r2 = (x**2 + y**2)
    gammat = gamma0 * np.exp(-0.5*r2/r0**2)
    g1 = -gammat * (x**2-y**2)/r2
    g2 = -gammat * (2.*x*y)/r2

    lens_cat = treecorr.Catalog(x=[0], y=[0], x_units='arcmin', y_units='arcmin')
    source_cat = treecorr.Catalog(x=x, y=y, g1=g1, g2=g2, x_units='arcmin', y_units='arcmin')
    ng = treecorr.NGCorrelation(bin_size=0.1, min_sep=1., max_sep=20., sep_units='arcmin',
                                verbose=1)
    ng.process(lens_cat, source_cat)

    # log(<R>) != <logR>, but it should be close:
    print('meanlogr - log(meanr) = ',ng.meanlogr - np.log(ng.meanr))
    np.testing.assert_allclose(ng.meanlogr, np.log(ng.meanr), atol=1.e-3)

    r = ng.meanr
    true_gt = gamma0 * np.exp(-0.5*r**2/r0**2)

    print('ng.xi = ',ng.xi)
    print('ng.xi_im = ',ng.xi_im)
    print('true_gammat = ',true_gt)
    print('ratio = ',ng.xi / true_gt)
    print('diff = ',ng.xi - true_gt)
    print('max diff = ',max(abs(ng.xi - true_gt)))
    np.testing.assert_allclose(ng.xi, true_gt, rtol=3.e-2)
    np.testing.assert_allclose(ng.xi_im, 0, atol=1.e-4)

    # Check that we get the same result using the corr2 function:
    lens_cat.write(os.path.join('data','ng_single_lens.dat'))
    source_cat.write(os.path.join('data','ng_single_source.dat'))
    config = treecorr.read_config('configs/ng_single.yaml')
    config['verbose'] = 0
    treecorr.corr2(config)
    corr2_output = np.genfromtxt(os.path.join('output','ng_single.out'), names=True,
                                 skip_header=1)
    print('ng.xi = ',ng.xi)
    print('from corr2 output = ',corr2_output['gamT'])
    print('ratio = ',corr2_output['gamT']/ng.xi)
    print('diff = ',corr2_output['gamT']-ng.xi)
    print('xi_im from corr2 output = ',corr2_output['gamX'])
    np.testing.assert_allclose(corr2_output['gamT'], ng.xi, rtol=1.e-3)
    np.testing.assert_allclose(corr2_output['gamX'], 0, atol=1.e-4)

    # Check that adding results with different coords or metric emits a warning.
    lens_cat2 = treecorr.Catalog(x=[0], y=[0], z=[0])
    source_cat2 = treecorr.Catalog(x=x, y=y, z=x, g1=g1, g2=g2)
    with CaptureLog() as cl:
        ng2 = treecorr.NGCorrelation(bin_size=0.1, min_sep=1., max_sep=20., logger=cl.logger)
        ng2.process_cross(lens_cat2, source_cat2)
        ng2 += ng
    assert "Detected a change in catalog coordinate systems" in cl.output

    with CaptureLog() as cl:
        ng3 = treecorr.NGCorrelation(bin_size=0.1, min_sep=1., max_sep=20., logger=cl.logger)
        ng3.process_cross(lens_cat2, source_cat2, metric='Rperp')
        ng3 += ng2
    assert "Detected a change in metric" in cl.output

    # There is special handling for single-row catalogs when using np.genfromtxt rather
    # than pandas.  So mock it up to make sure we test it.
    treecorr.Catalog._emitted_pandas_warning = False  # Reset this, in case already triggered.
    with mock.patch.dict(sys.modules, {'pandas':None}):
        with CaptureLog() as cl:
            treecorr.corr2(config, logger=cl.logger)
        assert "Unable to import pandas" in cl.output
    corr2_output = np.genfromtxt(os.path.join('output','ng_single.out'), names=True,
                                 skip_header=1)
    np.testing.assert_allclose(corr2_output['gamT'], ng.xi, rtol=1.e-3)



@timer
def test_pairwise2():
    # Test the same profile, but with the pairwise calcualtion:
    nsource = 300000
    gamma0 = 0.05
    r0 = 10.
    L = 5. * r0
    rng = np.random.RandomState(8675309)
    x = (rng.random_sample(nsource)-0.5) * L
    y = (rng.random_sample(nsource)-0.5) * L
    r2 = (x**2 + y**2)
    gammat = gamma0 * np.exp(-0.5*r2/r0**2)
    g1 = -gammat * (x**2-y**2)/r2
    g2 = -gammat * (2.*x*y)/r2

    dx = (rng.random_sample(nsource)-0.5) * L
    dx = (rng.random_sample(nsource)-0.5) * L

    lens_cat = treecorr.Catalog(x=dx, y=dx, x_units='arcmin', y_units='arcmin')
    source_cat = treecorr.Catalog(x=x+dx, y=y+dx, g1=g1, g2=g2, x_units='arcmin', y_units='arcmin')
    ng = treecorr.NGCorrelation(bin_size=0.1, min_sep=1., max_sep=20., sep_units='arcmin',
                                verbose=1, pairwise=True)
    with assert_warns(FutureWarning):
        ng.process(lens_cat, source_cat)

    r = ng.meanr
    true_gt = gamma0 * np.exp(-0.5*r**2/r0**2)

    print('ng.xi = ',ng.xi)
    print('ng.xi_im = ',ng.xi_im)
    print('true_gammat = ',true_gt)
    print('ratio = ',ng.xi / true_gt)
    print('diff = ',ng.xi - true_gt)
    print('max diff = ',max(abs(ng.xi - true_gt)))
    # I don't really understand why this comes out slightly less accurate.
    # I would have thought it would be slightly more accurate because it doesn't use the
    # approximations intrinsic to the tree calculation.
    np.testing.assert_allclose(ng.xi, true_gt, rtol=3.e-2)
    np.testing.assert_allclose(ng.xi_im, 0, atol=1.e-4)

    # Check that we get the same result using the corr2 function:
    lens_cat.write(os.path.join('data','ng_pairwise_lens.dat'))
    source_cat.write(os.path.join('data','ng_pairwise_source.dat'))
    config = treecorr.read_config('configs/ng_pairwise.yaml')
    config['verbose'] = 0
    with assert_warns(FutureWarning):
        treecorr.corr2(config)
    corr2_output = np.genfromtxt(os.path.join('output','ng_pairwise.out'), names=True,
                                 skip_header=1)
    print('ng.xi = ',ng.xi)
    print('from corr2 output = ',corr2_output['gamT'])
    print('ratio = ',corr2_output['gamT']/ng.xi)
    print('diff = ',corr2_output['gamT']-ng.xi)
    np.testing.assert_allclose(corr2_output['gamT'], ng.xi, rtol=1.e-3)

    print('xi_im from corr2 output = ',corr2_output['gamX'])
    np.testing.assert_allclose(corr2_output['gamX'], 0, atol=1.e-4)


@timer
def test_spherical():
    # This is the same profile we used for test_single, but put into spherical coords.
    # We do the spherical trig by hand using the obvious formulae, rather than the clever
    # optimizations that are used by the TreeCorr code, thus serving as a useful test of
    # the latter.

    nsource = 300000
    gamma0 = 0.05
    r0 = 10. * coord.degrees / coord.radians
    L = 5. * r0
    rng = np.random.RandomState(8675309)
    x = (rng.random_sample(nsource)-0.5) * L
    y = (rng.random_sample(nsource)-0.5) * L
    r2 = (x**2 + y**2)
    gammat = gamma0 * np.exp(-0.5*r2/r0**2)
    g1 = -gammat * (x**2-y**2)/r2
    g2 = -gammat * (2.*x*y)/r2
    r = np.sqrt(r2)
    theta = np.arctan2(y,x)

    ng = treecorr.NGCorrelation(bin_size=0.1, min_sep=1., max_sep=20., sep_units='deg',
                                verbose=1)
    r1 = np.exp(ng.logr) * (coord.degrees / coord.radians)
    true_gt = gamma0 * np.exp(-0.5*r1**2/r0**2)

    # Test this around several central points
    if __name__ == '__main__':
        ra0_list = [ 0., 1., 1.3, 232., 0. ]
        dec0_list = [ 0., -0.3, 1.3, -1.4, np.pi/2.-1.e-6 ]
    else:
        ra0_list = [ 232 ]
        dec0_list = [ -1.4 ]
    for ra0, dec0 in zip(ra0_list, dec0_list):

        # Use spherical triangle with A = point, B = (ra0,dec0), C = N. pole
        # a = Pi/2-dec0
        # c = 2*asin(r/2)  (lambert projection)
        # B = Pi/2 - theta

        c = 2.*np.arcsin(r/2.)
        a = np.pi/2. - dec0
        B = np.pi/2. - theta
        B[x<0] *= -1.
        B[B<-np.pi] += 2.*np.pi
        B[B>np.pi] -= 2.*np.pi

        # Solve the rest of the triangle with spherical trig:
        cosb = np.cos(a)*np.cos(c) + np.sin(a)*np.sin(c)*np.cos(B)
        b = np.arccos(cosb)
        cosA = (np.cos(a) - np.cos(b)*np.cos(c)) / (np.sin(b)*np.sin(c))
        #A = np.arccos(cosA)
        A = np.zeros_like(cosA)
        A[abs(cosA)<1] = np.arccos(cosA[abs(cosA)<1])
        A[cosA<=-1] = np.pi
        cosC = (np.cos(c) - np.cos(a)*np.cos(b)) / (np.sin(a)*np.sin(b))
        #C = np.arccos(cosC)
        C = np.zeros_like(cosC)
        C[abs(cosC)<1] = np.arccos(cosC[abs(cosC)<1])
        C[cosC<=-1] = np.pi
        C[x<0] *= -1.

        ra = ra0 - C
        dec = np.pi/2. - b

        # Rotate shear relative to local west
        # gamma_sph = exp(2i beta) * gamma
        # where beta = pi - (A+B) is the angle between north and "up" in the tangent plane.
        beta = np.pi - (A+B)
        beta[x>0] *= -1.
        cos2beta = np.cos(2.*beta)
        sin2beta = np.sin(2.*beta)
        g1_sph = g1 * cos2beta - g2 * sin2beta
        g2_sph = g2 * cos2beta + g1 * sin2beta

        lens_cat = treecorr.Catalog(ra=[ra0], dec=[dec0], ra_units='rad', dec_units='rad')
        source_cat = treecorr.Catalog(ra=ra, dec=dec, g1=g1_sph, g2=g2_sph,
                                      ra_units='rad', dec_units='rad')
        ng.process(lens_cat, source_cat)

        print('ra0, dec0 = ',ra0,dec0)
        print('ng.xi = ',ng.xi)
        print('true_gammat = ',true_gt)
        print('ratio = ',ng.xi / true_gt)
        print('diff = ',ng.xi - true_gt)
        print('max diff = ',max(abs(ng.xi - true_gt)))
        # The 3rd and 4th centers are somewhat less accurate.  Not sure why.
        # The math seems to be right, since the last one that gets all the way to the pole
        # works, so I'm not sure what is going on.  It's just a few bins that get a bit less
        # accurate.  Possibly worth investigating further at some point...
        np.testing.assert_allclose(ng.xi, true_gt, rtol=0.1)

    # One more center that can be done very easily.  If the center is the north pole, then all
    # the tangential shears are pure (positive) g1.
    ra0 = 0
    dec0 = np.pi/2.
    ra = theta
    dec = np.pi/2. - 2.*np.arcsin(r/2.)

    lens_cat = treecorr.Catalog(ra=[ra0], dec=[dec0], ra_units='rad', dec_units='rad')
    source_cat = treecorr.Catalog(ra=ra, dec=dec, g1=gammat, g2=np.zeros_like(gammat),
                                  ra_units='rad', dec_units='rad')
    ng.process(lens_cat, source_cat)

    print('ng.xi = ',ng.xi)
    print('ng.xi_im = ',ng.xi_im)
    print('true_gammat = ',true_gt)
    print('ratio = ',ng.xi / true_gt)
    print('diff = ',ng.xi - true_gt)
    print('max diff = ',max(abs(ng.xi - true_gt)))
    np.testing.assert_allclose(ng.xi, true_gt, rtol=0.1)
    np.testing.assert_allclose(ng.xi_im, 0, atol=1.e-4)

    # Check that we get the same result using the corr2 function:
    lens_cat.write(os.path.join('data','ng_spherical_lens.dat'))
    source_cat.write(os.path.join('data','ng_spherical_source.dat'))
    config = treecorr.read_config('configs/ng_spherical.yaml')
    config['verbose'] = 0
    treecorr.corr2(config)
    corr2_output = np.genfromtxt(os.path.join('output','ng_spherical.out'), names=True,
                                 skip_header=1)
    print('ng.xi = ',ng.xi)
    print('from corr2 output = ',corr2_output['gamT'])
    print('ratio = ',corr2_output['gamT']/ng.xi)
    print('diff = ',corr2_output['gamT']-ng.xi)
    np.testing.assert_allclose(corr2_output['gamT'], ng.xi, rtol=1.e-3)

    print('xi_im from corr2 output = ',corr2_output['gamX'])
    np.testing.assert_allclose(corr2_output['gamX'], 0., atol=3.e-5)


@timer
def test_ng():
    # Use gamma_t(r) = gamma0 exp(-r^2/2r0^2) around a bunch of foreground lenses.
    # i.e. gamma(r) = -gamma0 exp(-r^2/2r0^2) (x+iy)^2/r^2

    nlens = 1000
    nsource = 100000
    gamma0 = 0.05
    r0 = 10.
    L = 50. * r0
    rng = np.random.RandomState(8675309)
    xl = (rng.random_sample(nlens)-0.5) * L
    yl = (rng.random_sample(nlens)-0.5) * L
    xs = (rng.random_sample(nsource)-0.5) * L
    ys = (rng.random_sample(nsource)-0.5) * L
    g1 = np.zeros( (nsource,) )
    g2 = np.zeros( (nsource,) )
    for x,y in zip(xl,yl):
        dx = xs-x
        dy = ys-y
        r2 = dx**2 + dy**2
        gammat = gamma0 * np.exp(-0.5*r2/r0**2)
        g1 += -gammat * (dx**2-dy**2)/r2
        g2 += -gammat * (2.*dx*dy)/r2

    lens_cat = treecorr.Catalog(x=xl, y=yl, x_units='arcmin', y_units='arcmin')
    source_cat = treecorr.Catalog(x=xs, y=ys, g1=g1, g2=g2, x_units='arcmin', y_units='arcmin')
    ng = treecorr.NGCorrelation(bin_size=0.1, min_sep=1., max_sep=20., sep_units='arcmin',
                                verbose=1)
    ng.process(lens_cat, source_cat)

    # Using nbins=None rather than omitting nbins is equivalent.
    ng2 = treecorr.NGCorrelation(bin_size=0.1, min_sep=1., max_sep=20., nbins=None, sep_units='arcmin')
    ng2.process(lens_cat, source_cat, num_threads=1)
    ng.process(lens_cat, source_cat, num_threads=1)
    assert ng2 == ng

    r = ng.meanr
    true_gt = gamma0 * np.exp(-0.5*r**2/r0**2)

    print('ng.xi = ',ng.xi)
    print('ng.xi_im = ',ng.xi_im)
    print('true_gammat = ',true_gt)
    print('ratio = ',ng.xi / true_gt)
    print('diff = ',ng.xi - true_gt)
    print('max diff = ',max(abs(ng.xi - true_gt)))
    np.testing.assert_allclose(ng.xi, true_gt, rtol=0.1)
    np.testing.assert_allclose(ng.xi_im, 0, atol=5.e-3)

    nrand = nlens * 3
    xr = (rng.random_sample(nrand)-0.5) * L
    yr = (rng.random_sample(nrand)-0.5) * L
    rand_cat = treecorr.Catalog(x=xr, y=yr, x_units='arcmin', y_units='arcmin')
    rg = treecorr.NGCorrelation(bin_size=0.1, min_sep=1., max_sep=20., sep_units='arcmin',
                                verbose=1)
    rg.process(rand_cat, source_cat)
    print('rg.xi = ',rg.xi)
    xi, xi_im, varxi = ng.calculateXi(rg=rg)
    print('compensated xi = ',xi)
    print('compensated xi_im = ',xi_im)
    print('true_gammat = ',true_gt)
    print('ratio = ',xi / true_gt)
    print('diff = ',xi - true_gt)
    print('max diff = ',max(abs(xi - true_gt)))
    # It turns out this doesn't come out much better.  I think the imprecision is mostly just due
    # to the smallish number of lenses, not to edge effects
    np.testing.assert_allclose(xi, true_gt, rtol=0.1)
    np.testing.assert_allclose(xi_im, 0, atol=5.e-3)

    # rg is still allowed as a positional argument, but deprecated
    with assert_warns(FutureWarning):
        xi_2, xi_im_2, varxi_2 = ng.calculateXi(rg)
    np.testing.assert_array_equal(xi_2, xi)
    np.testing.assert_array_equal(xi_im_2, xi_im)
    np.testing.assert_array_equal(varxi_2, varxi)

    # Check that we get the same result using the corr2 function:
    config = treecorr.read_config('configs/ng.yaml')
    try:
        import fitsio
    except ImportError:
        pass
    else:
        lens_cat.write(os.path.join('data','ng_lens.fits'))
        source_cat.write(os.path.join('data','ng_source.fits'))
        rand_cat.write(os.path.join('data','ng_rand.fits'))
        config['verbose'] = 0
        config['precision'] = 8
        treecorr.corr2(config)
        corr2_output = np.genfromtxt(os.path.join('output','ng.out'), names=True, skip_header=1)
        print('ng.xi = ',ng.xi)
        print('xi = ',xi)
        print('from corr2 output = ',corr2_output['gamT'])
        print('ratio = ',corr2_output['gamT']/xi)
        print('diff = ',corr2_output['gamT']-xi)
        np.testing.assert_allclose(corr2_output['gamT'], xi, rtol=1.e-3)
        print('xi_im from corr2 output = ',corr2_output['gamX'])
        np.testing.assert_allclose(corr2_output['gamX'], 0., atol=4.e-3)

        # In the corr2 context, you can turn off the compensated bit, even if there are randoms
        # (e.g. maybe you only want randoms for some nn calculation, but not ng.)
        config['ng_statistic'] = 'simple'
        treecorr.corr2(config)
        corr2_output = np.genfromtxt(os.path.join('output','ng.out'), names=True, skip_header=1)
        xi_simple, _, _ = ng.calculateXi()
        np.testing.assert_equal(xi_simple, ng.xi)
        np.testing.assert_allclose(corr2_output['gamT'], xi_simple, rtol=1.e-3)

    # Check the fits write option
    try:
        import fitsio
    except ImportError:
        pass
    else:
        out_file_name1 = os.path.join('output','ng_out1.fits')
        ng.write(out_file_name1)
        data = fitsio.read(out_file_name1)
        np.testing.assert_almost_equal(data['r_nom'], np.exp(ng.logr))
        np.testing.assert_almost_equal(data['meanr'], ng.meanr)
        np.testing.assert_almost_equal(data['meanlogr'], ng.meanlogr)
        np.testing.assert_almost_equal(data['gamT'], ng.xi)
        np.testing.assert_almost_equal(data['gamX'], ng.xi_im)
        np.testing.assert_almost_equal(data['sigma'], np.sqrt(ng.varxi))
        np.testing.assert_almost_equal(data['weight'], ng.weight)
        np.testing.assert_almost_equal(data['npairs'], ng.npairs)

        out_file_name2 = os.path.join('output','ng_out2.fits')
        ng.write(out_file_name2, rg=rg)
        data = fitsio.read(out_file_name2)
        np.testing.assert_almost_equal(data['r_nom'], np.exp(ng.logr))
        np.testing.assert_almost_equal(data['meanr'], ng.meanr)
        np.testing.assert_almost_equal(data['meanlogr'], ng.meanlogr)
        np.testing.assert_almost_equal(data['gamT'], xi)
        np.testing.assert_almost_equal(data['gamX'], xi_im)
        np.testing.assert_almost_equal(data['sigma'], np.sqrt(varxi))
        np.testing.assert_almost_equal(data['weight'], ng.weight)
        np.testing.assert_almost_equal(data['npairs'], ng.npairs)

        # Check the read function
        ng2 = treecorr.NGCorrelation(bin_size=0.1, min_sep=1., max_sep=20., sep_units='arcmin')
        ng2.read(out_file_name2)
        np.testing.assert_almost_equal(ng2.logr, ng.logr)
        np.testing.assert_almost_equal(ng2.meanr, ng.meanr)
        np.testing.assert_almost_equal(ng2.meanlogr, ng.meanlogr)
        np.testing.assert_almost_equal(ng2.xi, ng.xi)
        np.testing.assert_almost_equal(ng2.xi_im, ng.xi_im)
        np.testing.assert_almost_equal(ng2.varxi, ng.varxi)
        np.testing.assert_almost_equal(ng2.weight, ng.weight)
        np.testing.assert_almost_equal(ng2.npairs, ng.npairs)
        assert ng2.coords == ng.coords
        assert ng2.metric == ng.metric
        assert ng2.sep_units == ng.sep_units
        assert ng2.bin_type == ng.bin_type


@timer
def test_nmap():
    # Same scenario as above.
    # Use gamma_t(r) = gamma0 exp(-r^2/2r0^2) around a bunch of foreground lenses.
    # i.e. gamma(r) = -gamma0 exp(-r^2/2r0^2) (x+iy)^2/r^2

    # Crittenden NMap = int_0^inf gamma_t(r) T(r/R) rdr/R^2
    #                 = gamma0/4 r0^4 (r0^2 + 6R^2) / (r0^2 + 2R^2)^3

    nlens = 1000
    nsource = 10000
    gamma0 = 0.05
    r0 = 10.
    L = 50. * r0
    rng = np.random.RandomState(8675309)
    xl = (rng.random_sample(nlens)-0.5) * L
    yl = (rng.random_sample(nlens)-0.5) * L
    xs = (rng.random_sample(nsource)-0.5) * L
    ys = (rng.random_sample(nsource)-0.5) * L
    g1 = np.zeros( (nsource,) )
    g2 = np.zeros( (nsource,) )
    for x,y in zip(xl,yl):
        dx = xs-x
        dy = ys-y
        r2 = dx**2 + dy**2
        gammat = gamma0 * np.exp(-0.5*r2/r0**2)
        g1 += -gammat * (dx**2-dy**2)/r2
        g2 += -gammat * (2.*dx*dy)/r2

    lens_cat = treecorr.Catalog(x=xl, y=yl, x_units='arcmin', y_units='arcmin')
    source_cat = treecorr.Catalog(x=xs, y=ys, g1=g1, g2=g2, x_units='arcmin', y_units='arcmin')
    # Measure ng with a factor of 2 extra at high and low ends
    ng = treecorr.NGCorrelation(bin_size=0.1, min_sep=0.5, max_sep=40., sep_units='arcmin',
                                verbose=1)
    ng.process(lens_cat, source_cat)

    r = ng.meanr
    true_nmap = 0.25 * gamma0 * r0**4 * (r0**2 + 6*r**2) / (r0**2 + 2*r**2)**3
    nmap, nmx, varnmap = ng.calculateNMap()

    print('nmap = ',nmap)
    print('nmx = ',nmx)
    print('true_nmap = ',true_nmap)
    mask = (1 < r) & (r < 20)
    print('ratio = ',nmap[mask] / true_nmap[mask])
    print('max rel diff = ',max(abs((nmap[mask] - true_nmap[mask])/true_nmap[mask])))
    np.testing.assert_allclose(nmap[mask], true_nmap[mask], rtol=0.1)
    np.testing.assert_allclose(nmx[mask], 0, atol=5.e-3)

    nrand = nlens * 3
    xr = (rng.random_sample(nrand)-0.5) * L
    yr = (rng.random_sample(nrand)-0.5) * L
    rand_cat = treecorr.Catalog(x=xr, y=yr, x_units='arcmin', y_units='arcmin')
    rg = treecorr.NGCorrelation(bin_size=0.1, min_sep=0.5, max_sep=40., sep_units='arcmin',
                                verbose=1)
    rg.process(rand_cat, source_cat)
    nmap2, nmx2, varnmap2 = ng.calculateNMap(rg=rg, m2_uform='Crittenden')
    print('compensated nmap = ',nmap2)
    print('ratio = ',nmap2[mask] / true_nmap[mask])
    print('max rel diff = ',max(abs((nmap2[mask] - true_nmap[mask])/true_nmap[mask])))
    np.testing.assert_allclose(nmap2[mask], true_nmap[mask], rtol=0.1)
    np.testing.assert_allclose(nmx2[mask], 0, atol=5.e-3)

    # Check that we get the same result using the corr2 function:
    lens_cat.write(os.path.join('data','ng_nmap_lens.dat'))
    source_cat.write(os.path.join('data','ng_nmap_source.dat'))
    rand_cat.write(os.path.join('data','ng_nmap_rand.dat'))
    config = treecorr.read_config('configs/ng_nmap.yaml')
    config['verbose'] = 0
    treecorr.corr2(config)
    corr2_output = np.genfromtxt(os.path.join('output','ng_nmap.out'), names=True)
    np.testing.assert_allclose(corr2_output['NMap'], nmap2, rtol=1.e-3)
    np.testing.assert_allclose(corr2_output['NMx'], nmx2, atol=1.e-3)
    np.testing.assert_allclose(corr2_output['sig_nmap'], np.sqrt(varnmap2), rtol=1.e-3)

    # Check giving specific R values
    R = ng.meanr[mask]
    nmap2, nmx2, varnmap2 = ng.calculateNMap(R=R, rg=rg)
    print('compensated nmap = ',nmap2)
    print('ratio = ',nmap2 / true_nmap[mask])
    print('max rel diff = ',max(abs((nmap2 - true_nmap[mask])/true_nmap[mask])))
    np.testing.assert_allclose(nmap2, true_nmap[mask], rtol=0.1)
    np.testing.assert_allclose(nmx2, 0, atol=5.e-3)

    # Can also skip the randoms (even if listed in the file)
    config['ng_statistic'] = 'simple'
    config['nn_statistic'] = 'simple'  # For the later Norm tests
    treecorr.corr2(config)
    corr2_output = np.genfromtxt(os.path.join('output','ng_nmap.out'), names=True)
    np.testing.assert_allclose(corr2_output['NMap'], nmap, rtol=1.e-3)
    np.testing.assert_allclose(corr2_output['NMx'], nmx, atol=1.e-3)
    np.testing.assert_allclose(corr2_output['sig_nmap'], np.sqrt(varnmap), rtol=1.e-3)

    # And check the norm output file, which adds a few columns
    dd = treecorr.NNCorrelation(bin_size=0.1, min_sep=0.5, max_sep=40., sep_units='arcmin',
                                verbose=1)
    dd.process(lens_cat)
    rr = treecorr.NNCorrelation(bin_size=0.1, min_sep=0.5, max_sep=40., sep_units='arcmin',
                                verbose=1)
    rr.process(rand_cat)
    gg = treecorr.GGCorrelation(bin_size=0.1, min_sep=0.5, max_sep=40., sep_units='arcmin',
                                verbose=1)
    gg.process(source_cat)
    napsq, varnap = dd.calculateNapSq(rr=rr)
    mapsq, mapsq_im, mxsq, mxsq_im, varmap = gg.calculateMapSq(m2_uform='Crittenden')
    nmap_norm = nmap**2 / napsq / mapsq
    napsq_mapsq = napsq / mapsq
    corr2_output = np.genfromtxt(os.path.join('output','ng_norm.out'), names=True)
    np.testing.assert_allclose(corr2_output['NMap'], nmap, rtol=1.e-3)
    np.testing.assert_allclose(corr2_output['NMx'], nmx, atol=1.e-3)
    np.testing.assert_allclose(corr2_output['sig_nmap'], np.sqrt(varnmap), rtol=1.e-3)
    np.testing.assert_allclose(corr2_output['Napsq'], napsq, rtol=1.e-3)
    np.testing.assert_allclose(corr2_output['sig_napsq'], np.sqrt(varnap), rtol=1.e-3)
    np.testing.assert_allclose(corr2_output['Mapsq'], mapsq, rtol=1.e-3)
    np.testing.assert_allclose(corr2_output['sig_mapsq'], np.sqrt(varmap), rtol=1.e-3)
    np.testing.assert_allclose(corr2_output['NMap_norm'], nmap_norm, rtol=1.e-3)
    np.testing.assert_allclose(corr2_output['Nsq_Mapsq'], napsq_mapsq, rtol=1.e-3)

    # Also check writing to fits file.
    # For grins, also check the explicit file_type option (which is rarely necessary)
    try:
        import fitsio
    except ImportError:
        pass
    else:
        fits_name = os.path.join('output', 'ng_nmap.zzz')
        ng.writeNMap(fits_name, file_type='fits')
        data = fitsio.read(fits_name)
        np.testing.assert_allclose(data['NMap'], nmap, rtol=1.e-8)
        np.testing.assert_allclose(data['NMx'], nmx, atol=1.e-8)
        np.testing.assert_allclose(data['sig_nmap'], np.sqrt(varnmap), rtol=1.e-8)

        fits_name = os.path.join('output', 'ng_norm.zzz')
        ng.writeNorm(fits_name, gg=gg, dd=dd, rr=rr, file_type='fits')
        data = fitsio.read(fits_name)
        np.testing.assert_allclose(data['NMap'], nmap, rtol=1.e-6)
        np.testing.assert_allclose(data['NMx'], nmx, atol=1.e-6)
        np.testing.assert_allclose(data['sig_nmap'], np.sqrt(varnmap), rtol=1.e-6)
        np.testing.assert_allclose(data['Napsq'], napsq, rtol=1.e-6)
        np.testing.assert_allclose(data['sig_napsq'], np.sqrt(varnap), rtol=1.e-6)
        np.testing.assert_allclose(data['Mapsq'], mapsq, rtol=1.e-6)
        np.testing.assert_allclose(data['sig_mapsq'], np.sqrt(varmap), rtol=1.e-6)
        np.testing.assert_allclose(data['NMap_norm'], nmap_norm, rtol=1.e-6)
        np.testing.assert_allclose(data['Nsq_Mapsq'], napsq_mapsq, rtol=1.e-6)

        with assert_warns(FutureWarning):
            # This one in particular is worth checking, since some kw-onlt args don't have defaults,
            # so it didn't actually work with my original implementation of depr_pos_kwargs
            ng.writeNorm(fits_name, gg, dd, rr, file_type='fits')
        data2 = fitsio.read(fits_name)
        np.testing.assert_array_equal(data2, data)

        fits_name = os.path.join('output', 'ng_nmap2.fits')
        ng.writeNMap(fits_name, R=R, rg=rg)
        data = fitsio.read(fits_name)
        np.testing.assert_allclose(data['NMap'], nmap2, rtol=1.e-8)
        np.testing.assert_allclose(data['NMx'], nmx2, atol=1.e-8)
        np.testing.assert_allclose(data['sig_nmap'], np.sqrt(varnmap2), rtol=1.e-8)

        fits_name = os.path.join('output', 'ng_norm2.fits')
        ng.writeNorm(fits_name, R=R, gg=gg, dd=dd, rr=rr, rg=rg)
        data = fitsio.read(fits_name)
        napsq2, varnap2 = dd.calculateNapSq(R=R, rr=rr)
        mapsq2, mapsq2_im, mxsq2, mxsq2_im, varmap2 = gg.calculateMapSq(R=R, m2_uform='Crittenden')
        nmap_norm2 = nmap2**2 / napsq2 / mapsq2
        napsq_mapsq2 = napsq2 / mapsq2
        np.testing.assert_allclose(data['NMap'], nmap2, rtol=1.e-6)
        np.testing.assert_allclose(data['NMx'], nmx2, atol=1.e-6)
        np.testing.assert_allclose(data['sig_nmap'], np.sqrt(varnmap2), rtol=1.e-6)
        np.testing.assert_allclose(data['Napsq'], napsq2, rtol=2.e-3)
        np.testing.assert_allclose(data['sig_napsq'], np.sqrt(varnap2), rtol=1.e-6)
        np.testing.assert_allclose(data['Mapsq'], mapsq2, rtol=1.e-6)
        np.testing.assert_allclose(data['sig_mapsq'], np.sqrt(varmap2), rtol=1.e-6)
        np.testing.assert_allclose(data['NMap_norm'], nmap_norm2, rtol=1.e-6)
        np.testing.assert_allclose(data['Nsq_Mapsq'], napsq_mapsq2, rtol=1.e-6)

    # Finally, let's also check the Schneider definition.
    # It doesn't have a nice closed form solution (as far as I can figure out at least).
    # but it does look qualitatively similar to the Crittenden one.
    # Just its definition of R is different, so we need to compare a different subset to
    # get a decent match.  Also, the amplitude is different by a factor of 6/5.
    nmap_sch, nmx_sch, varnmap_sch = ng.calculateNMap(m2_uform='Schneider')
    print('Schneider nmap = ',nmap_sch[10:] * 5./6.)
    print('Crittenden nmap = ',nmap[:-10])
    print('ratio = ',nmap_sch[10:]*5./6. / nmap[:-10])
    np.testing.assert_allclose(nmap_sch[10:]*5./6., nmap[:-10], rtol=0.1)

    napsq_sch, varnap_sch = dd.calculateNapSq(rr=rr, m2_uform='Schneider')
    mapsq_sch, _, mxsq_sch, _, varmap_sch = gg.calculateMapSq(m2_uform='Schneider')
    print('Schneider napsq = ',napsq_sch[10:] * 5./6.)
    print('Crittenden napsq = ',napsq[:-10])
    print('ratio = ',napsq_sch[10:]*5./6. / napsq[:-10])
    print('diff = ',napsq_sch[10:]*5./6. - napsq[:-10])

    print('Schneider mapsq = ',mapsq_sch[10:] * 5./6.)
    print('Crittenden mapsq = ',mapsq[:-10])
    print('ratio = ',mapsq_sch[10:]*5./6. / mapsq[:-10])

    # These have zero crossings, where they have slightly different shapes, so the agreement
    # isn't as good as with nmap.
    np.testing.assert_allclose(napsq_sch[10:]*5./6., napsq[:-10], rtol=0.2, atol=5.e-3)
    np.testing.assert_allclose(mapsq_sch[10:]*5./6., mapsq[:-10], rtol=0.2, atol=5.e-5)

    nmap_norm_sch = nmap_sch**2 / napsq_sch / mapsq_sch
    napsq_mapsq_sch = napsq_sch / mapsq_sch

    config['m2_uform'] = 'Schneider'
    config['precision'] = 5
    treecorr.corr2(config)
    corr2_output = np.genfromtxt(os.path.join('output','ng_norm.out'), names=True)
    np.testing.assert_allclose(corr2_output['NMap'], nmap_sch, rtol=1.e-3)
    np.testing.assert_allclose(corr2_output['NMx'], nmx_sch, atol=1.e-3)
    np.testing.assert_allclose(corr2_output['sig_nmap'], np.sqrt(varnmap_sch), rtol=1.e-3)
    np.testing.assert_allclose(corr2_output['Napsq'], napsq_sch, rtol=1.e-3)
    np.testing.assert_allclose(corr2_output['sig_napsq'], np.sqrt(varnap_sch), rtol=1.e-3)
    np.testing.assert_allclose(corr2_output['Mapsq'], mapsq_sch, rtol=1.e-3)
    np.testing.assert_allclose(corr2_output['sig_mapsq'], np.sqrt(varmap_sch), rtol=1.e-3)
    np.testing.assert_allclose(corr2_output['NMap_norm'], nmap_norm_sch, rtol=1.e-3)
    np.testing.assert_allclose(corr2_output['Nsq_Mapsq'], napsq_mapsq_sch, rtol=1.e-3)

    with assert_raises(ValueError):
        ng.calculateNMap(m2_uform='Other')
    with assert_raises(ValueError):
        dd.calculateNapSq(rr=rr, m2_uform='Other')
    with assert_raises(ValueError):
        gg.calculateMapSq(m2_uform='Other')


@timer
def test_pieces():
    # Test that we can do the calculation in pieces and recombine the results

    try:
        import fitsio
    except ImportError:
        print('Skip test_pieces, since fitsio not installed.')
        return

    ncats = 3
    nlens = 1000
    nsource = 30000
    gamma0 = 0.05
    r0 = 10.
    L = 50. * r0
    rng = np.random.RandomState(8675309)
    xl = (rng.random_sample(nlens)-0.5) * L
    yl = (rng.random_sample(nlens)-0.5) * L
    xs = (rng.random_sample( (nsource,ncats) )-0.5) * L
    ys = (rng.random_sample( (nsource,ncats) )-0.5) * L
    g1 = np.zeros( (nsource,ncats) )
    g2 = np.zeros( (nsource,ncats) )
    w = rng.random_sample( (nsource,ncats) ) + 0.5
    for x,y in zip(xl,yl):
        dx = xs-x
        dy = ys-y
        r2 = dx**2 + dy**2
        gammat = gamma0 * np.exp(-0.5*r2/r0**2)
        g1 += -gammat * (dx**2-dy**2)/r2
        g2 += -gammat * (2.*dx*dy)/r2

    lens_cat = treecorr.Catalog(x=xl, y=yl, x_units='arcmin', y_units='arcmin')
    source_cats = [ treecorr.Catalog(x=xs[:,k], y=ys[:,k], g1=g1[:,k], g2=g2[:,k], w=w[:,k],
                                     x_units='arcmin', y_units='arcmin') for k in range(ncats) ]
    full_source_cat = treecorr.Catalog(x=xs.flatten(), y=ys.flatten(), w=w.flatten(),
                                       g1=g1.flatten(), g2=g2.flatten(),
                                       x_units='arcmin', y_units='arcmin')

    t0 = time.time()
    for k in range(ncats):
        # These could each be done on different machines in a real world application.
        ng = treecorr.NGCorrelation(bin_size=0.1, min_sep=1., max_sep=25., sep_units='arcmin',
                                    verbose=1)
        # These should use process_cross, not process, since we don't want to call finalize.
        ng.process_cross(lens_cat, source_cats[k])
        ng.write(os.path.join('output','ng_piece_%d.fits'%k))

    pieces_ng = treecorr.NGCorrelation(bin_size=0.1, min_sep=1., max_sep=25., sep_units='arcmin')
    for k in range(ncats):
        ng = pieces_ng.copy()
        ng.read(os.path.join('output','ng_piece_%d.fits'%k))
        pieces_ng += ng
    varg = treecorr.calculateVarG(source_cats)
    pieces_ng.finalize(varg)
    t1 = time.time()
    print('time for piece-wise processing (including I/O) = ',t1-t0)

    full_ng = treecorr.NGCorrelation(bin_size=0.1, min_sep=1., max_sep=25., sep_units='arcmin',
                                     verbose=1)
    full_ng.process(lens_cat, full_source_cat)
    t2 = time.time()
    print('time for full processing = ',t2-t1)

    print('max error in meanr = ',np.max(pieces_ng.meanr - full_ng.meanr),)
    print('    max meanr = ',np.max(full_ng.meanr))
    print('max error in meanlogr = ',np.max(pieces_ng.meanlogr - full_ng.meanlogr),)
    print('    max meanlogr = ',np.max(full_ng.meanlogr))
    print('max error in weight = ',np.max(pieces_ng.weight - full_ng.weight),)
    print('    max weight = ',np.max(full_ng.weight))
    print('max error in xi = ',np.max(pieces_ng.xi - full_ng.xi),)
    print('    max xi = ',np.max(full_ng.xi))
    print('max error in xi_im = ',np.max(pieces_ng.xi_im - full_ng.xi_im),)
    print('    max xi_im = ',np.max(full_ng.xi_im))
    print('max error in varxi = ',np.max(pieces_ng.varxi - full_ng.varxi),)
    print('    max varxi = ',np.max(full_ng.varxi))
    np.testing.assert_allclose(pieces_ng.meanr, full_ng.meanr, rtol=2.e-3)
    np.testing.assert_allclose(pieces_ng.meanlogr, full_ng.meanlogr, atol=2.e-3)
    np.testing.assert_allclose(pieces_ng.weight, full_ng.weight, rtol=3.e-2)
    np.testing.assert_allclose(pieces_ng.xi, full_ng.xi, rtol=0.1)
    np.testing.assert_allclose(pieces_ng.xi_im, full_ng.xi_im, atol=2.e-3)
    np.testing.assert_allclose(pieces_ng.varxi, full_ng.varxi, rtol=3.e-2)

    # A different way to do this can produce results that are essentially identical to the
    # full calculation.  We can use wpos = w, but set w = 0 for the items in the pieces catalogs
    # that we don't want to include.  This will force the tree to be built identically in each
    # case, but only use the subset of items in the calculation.  The sum of all these should
    # be identical to the full calculation aside from order of calculation differences.
    # However, we lose some to speed, since there are a lot more wasted calculations along the
    # way that have to be duplicated in each piece.
    w2 = [ np.empty(w.shape) for k in range(ncats) ]
    for k in range(ncats):
        w2[k][:,:] = 0.
        w2[k][:,k] = w[:,k]
    source_cats2 = [ treecorr.Catalog(x=xs.flatten(), y=ys.flatten(),
                                      g1=g1.flatten(), g2=g2.flatten(),
                                      wpos=w.flatten(), w=w2[k].flatten(),
                                      x_units='arcmin', y_units='arcmin') for k in range(ncats) ]

    t3 = time.time()
    ng2 = [ full_ng.copy() for k in range(ncats) ]
    for k in range(ncats):
        ng2[k].clear()
        ng2[k].process_cross(lens_cat, source_cats2[k])

    pieces_ng2 = full_ng.copy()
    pieces_ng2.clear()
    for k in range(ncats):
        pieces_ng2 += ng2[k]
    pieces_ng2.finalize(varg)
    t4 = time.time()
    print('time for zero-weight piece-wise processing = ',t4-t3)

    print('max error in meanr = ',np.max(pieces_ng2.meanr - full_ng.meanr),)
    print('    max meanr = ',np.max(full_ng.meanr))
    print('max error in meanlogr = ',np.max(pieces_ng2.meanlogr - full_ng.meanlogr),)
    print('    max meanlogr = ',np.max(full_ng.meanlogr))
    print('max error in weight = ',np.max(pieces_ng2.weight - full_ng.weight),)
    print('    max weight = ',np.max(full_ng.weight))
    print('max error in xi = ',np.max(pieces_ng2.xi - full_ng.xi),)
    print('    max xi = ',np.max(full_ng.xi))
    print('max error in xi_im = ',np.max(pieces_ng2.xi_im - full_ng.xi_im),)
    print('    max xi_im = ',np.max(full_ng.xi_im))
    print('max error in varxi = ',np.max(pieces_ng2.varxi - full_ng.varxi),)
    print('    max varxi = ',np.max(full_ng.varxi))
    np.testing.assert_allclose(pieces_ng2.meanr, full_ng.meanr, rtol=1.e-7)
    np.testing.assert_allclose(pieces_ng2.meanlogr, full_ng.meanlogr, rtol=1.e-7)
    np.testing.assert_allclose(pieces_ng2.weight, full_ng.weight, rtol=1.e-7)
    np.testing.assert_allclose(pieces_ng2.xi, full_ng.xi, rtol=1.e-7)
    np.testing.assert_allclose(pieces_ng2.xi_im, full_ng.xi_im, atol=1.e-10)
    np.testing.assert_allclose(pieces_ng2.varxi, full_ng.varxi, rtol=1.e-7)

    # Try this with corr2
    lens_cat.write(os.path.join('data','ng_wpos_lens.fits'))
    for i, sc in enumerate(source_cats2):
        sc.write(os.path.join('data','ng_wpos_source%d.fits'%i))
    config = treecorr.read_config('configs/ng_wpos.yaml')
    config['verbose'] = 0
    treecorr.corr2(config)
    data = fitsio.read(config['ng_file_name'])
    print('data.dtype = ',data.dtype)
    np.testing.assert_allclose(data['meanr'], full_ng.meanr, rtol=1.e-7)
    np.testing.assert_allclose(data['meanlogr'], full_ng.meanlogr, rtol=1.e-7)
    np.testing.assert_allclose(data['weight'], full_ng.weight, rtol=1.e-7)
    np.testing.assert_allclose(data['gamT'], full_ng.xi, rtol=1.e-7)
    np.testing.assert_allclose(data['gamX'], full_ng.xi_im, atol=1.e-10)
    np.testing.assert_allclose(data['sigma']**2, full_ng.varxi, rtol=1.e-7)


@timer
def test_haloellip():
    """This is similar to the Clampitt halo ellipticity measurement, but using counts for the
    background galaxies rather than shears.

    w_aligned = Sum_i (w_i * cos(2theta)) / Sum_i (w_i)
    w_cross = Sum_i (w_i * sin(2theta)) / Sum_i (w_i)

    where theta is measured w.r.t. the coordinate system where the halo ellitpicity
    is along the x-axis.  Converting this to complex notation, we obtain:

    w_a - i w_c = < exp(-2itheta) >
                = < exp(2iphi) exp(-2i(theta+phi)) >
                = < ehalo exp(-2i(theta+phi)) >

    where ehalo = exp(2iphi) is the unit-normalized shape of the halo in the normal world
    coordinate system.  Note that the combination theta+phi is the angle between the line joining
    the two points and the E-W coordinate, which means that

    w_a - i w_c = -gamma_t(n_bg, ehalo)

    so the reverse of the usual galaxy-galaxy lensing order.  The N is the background galaxies
    and G is the halo shapes (normalized to have |ehalo| = 1).
    """

    nhalo = 10
    nsource = 100000  # sources per halo
    ntot = nsource * nhalo
    L = 100000.  # The side length in which the halos are placed
    R = 10.      # The (rms) radius of the associated sources from the halos
                 # In this case, we want L >> R so that most sources are only associated
                 # with the one halo we used for assigning its shear value.

    # Lenses are randomly located with random shapes.
    rng = np.random.RandomState(8675309)
    halo_g1 = rng.normal(0., 0.3, (nhalo,))
    halo_g2 = rng.normal(0., 0.3, (nhalo,))
    halo_g = halo_g1 + 1j * halo_g2
    # The interpretation is simpler if they all have the same |g|, so just make them all 0.3.
    halo_g *= 0.3 / np.abs(halo_g)
    halo_absg = np.abs(halo_g)
    halo_x = (rng.random_sample(nhalo)-0.5) * L
    halo_y = (rng.random_sample(nhalo)-0.5) * L
    print('Made halos',len(halo_x))

    # For the sources, place nsource galaxies around each halo with the expected azimuthal pattern
    source_x = np.empty(ntot)
    source_y = np.empty(ntot)
    for i in range(nhalo):
        absg = halo_absg[i]
        # First position the sources in a Gaussian cloud around the halo center.
        dx = rng.normal(0., R, (nsource,))
        dy = rng.normal(0., R, (nsource,))
        r = np.sqrt(dx*dx + dy*dy)
        t = np.arctan2(dy,dx)
        # z = dx + idy = r exp(it)

        # Reposition the sources azimuthally so p(theta) ~ 1 + |g_halo| * cos(2 theta)
        # Currently t has p(t) = 1/2pi.
        # Let u be the new azimuthal angle with p(u) = (1/2pi) (1 + |g| cos(2u))
        # p(u) = |dt/du| p(t)
        # 1 + |g| cos(2u) = dt/du
        # t = int( (1 + |g| cos(2u)) du = u + 1/2 |g| sin(2u)

        # This doesn't have an analytic solution, but a few iterations of Newton-Raphson
        # should work well enough.
        u = t.copy()
        for k in range(4):
            u -= (u - t + 0.5 * absg * np.sin(2.*u)) / (1. + absg * np.cos(2.*u))

        z = r * np.exp(1j * u)

        # Now rotate the whole system by the phase of the halo ellipticity.
        exp2ialpha = halo_g[i] / absg
        expialpha = np.sqrt(exp2ialpha)
        z *= expialpha
        # Place the source galaxies at this dx,dy with this shape
        source_x[i*nsource: (i+1)*nsource] = halo_x[i] + z.real
        source_y[i*nsource: (i+1)*nsource] = halo_y[i] + z.imag
    print('Made sources',len(source_x))

    source_cat = treecorr.Catalog(x=source_x, y=source_y)
    # Big fat bin to increase S/N.  The way I set it up, the signal is the same in all
    # radial bins, so just combine them together for higher S/N.
    ng = treecorr.NGCorrelation(min_sep=5, max_sep=10, nbins=1)
    halo_mean_absg = np.mean(halo_absg)
    print('mean_absg = ',halo_mean_absg)

    # First the original version where we only use the phase of the halo ellipticities:
    halo_cat1 = treecorr.Catalog(x=halo_x, y=halo_y,
                                 g1=halo_g.real/halo_absg, g2=halo_g.imag/halo_absg)
    ng.process(source_cat, halo_cat1)
    print('ng.npairs = ',ng.npairs)
    print('ng.xi = ',ng.xi)
    # The expected signal is
    # E(ng) = - < int( p(t) cos(2t) ) >
    #       = - < int( (1 + e_halo cos(2t)) cos(2t) ) >
    #       = -0.5 <e_halo>
    print('expected signal = ',-0.5 * halo_mean_absg)
    np.testing.assert_allclose(ng.xi, -0.5 * halo_mean_absg, rtol=0.05)

    # Next weight the halos by their absg.
    halo_cat2 = treecorr.Catalog(x=halo_x, y=halo_y, w=halo_absg,
                                 g1=halo_g.real/halo_absg, g2=halo_g.imag/halo_absg)
    ng.process(source_cat, halo_cat2)
    print('ng.xi = ',ng.xi)
    # Now the net signal is
    # sum(w * p*cos(2t)) / sum(w)
    # = 0.5 * <absg^2> / <absg>
    halo_mean_gsq = np.mean(halo_absg**2)
    print('expected signal = ',0.5 * halo_mean_gsq / halo_mean_absg)
    np.testing.assert_allclose(ng.xi, -0.5 * halo_mean_gsq / halo_mean_absg, rtol=0.05)

    # Finally, use the unnormalized halo_g for the halo ellipticities
    halo_cat3 = treecorr.Catalog(x=halo_x, y=halo_y, g1=halo_g.real, g2=halo_g.imag)
    ng.process(source_cat, halo_cat3)
    print('ng.xi = ',ng.xi)
    # Now the net signal is
    # sum(absg * p*cos(2t)) / N
    # = 0.5 * <absg^2>
    print('expected signal = ',0.5 * halo_mean_gsq)
    np.testing.assert_allclose(ng.xi, -0.5 * halo_mean_gsq, rtol=0.05)

@timer
def test_varxi():
    # Test that varxi is correct (or close) based on actual variance of many runs.

    # Signal doesn't matter much.  Use the one from test_gg.
    gamma0 = 0.05
    r0 = 10.
    L = 10 * r0
    rng = np.random.RandomState(8675309)

    # Note: to get a good estimate of var(xi), you need a lot of runs.  The number of
    # runs matters much more than the number of galaxies for getting this to pass.
    # In addition, I found that the variance was significantly underestimated when there
    # were lots of lenses.  I guess because there were multiple lenses that paired with the
    # same sources in a given bin, which increased the variance of the mean <g>.
    # So there might be some adjustment that would help improve the estimate of varxi,
    # but at least this unit test shows that it's fairly accurate for *some* scenario.
    nsource = 1000
    nrand = 10
    nruns = 50000
    lens = treecorr.Catalog(x=[0], y=[0])

    file_name = 'data/test_varxi_ng.npz'
    print(file_name)
    if not os.path.isfile(file_name):
        all_ngs = []
        all_rgs = []
        for run in range(nruns):
            print(f'{run}/{nruns}')
            x2 = (rng.random_sample(nsource)-0.5) * L
            y2 = (rng.random_sample(nsource)-0.5) * L
            x3 = (rng.random_sample(nrand)-0.5) * L
            y3 = (rng.random_sample(nrand)-0.5) * L

            r2 = (x2**2 + y2**2)/r0**2
            g1 = -gamma0 * np.exp(-r2/2.) * (x2**2-y2**2)/r0**2
            g2 = -gamma0 * np.exp(-r2/2.) * (2.*x2*y2)/r0**2
            # This time, add some shape noise (different each run).
            g1 += rng.normal(0, 0.1, size=nsource)
            g2 += rng.normal(0, 0.1, size=nsource)
            # Varied weights are hard, but at least check that non-unit weights work correctly.
            w = np.ones_like(x2) * 5

            source = treecorr.Catalog(x=x2, y=y2, w=w, g1=g1, g2=g2)
            rand = treecorr.Catalog(x=x3, y=y3)
            ng = treecorr.NGCorrelation(bin_size=0.2, min_sep=10., max_sep=30.)
            rg = treecorr.NGCorrelation(bin_size=0.2, min_sep=10., max_sep=30.)
            ng.process(lens, source)
            rg.process(rand, source)
            all_ngs.append(ng)
            all_rgs.append(rg)

        all_xis = [ng.calculateXi() for ng in all_ngs]
        var_xi_1 = np.var([xi[0] for xi in all_xis], axis=0)
        mean_varxi_1 = np.mean([xi[2] for xi in all_xis], axis=0)

        all_xis = [ng.calculateXi(rg=rg) for (ng,rg) in zip(all_ngs, all_rgs)]
        var_xi_2 = np.var([xi[0] for xi in all_xis], axis=0)
        mean_varxi_2 = np.mean([xi[2] for xi in all_xis], axis=0)

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
    g1 = -gamma0 * np.exp(-r2/2.) * (x2**2-y2**2)/r0**2
    g2 = -gamma0 * np.exp(-r2/2.) * (2.*x2*y2)/r0**2
    # This time, add some shape noise (different each run).
    g1 += rng.normal(0, 0.1, size=nsource)
    g2 += rng.normal(0, 0.1, size=nsource)
    # Varied weights are hard, but at least check that non-unit weights work correctly.
    w = np.ones_like(x2) * 5

    source = treecorr.Catalog(x=x2, y=y2, w=w, g1=g1, g2=g2)
    rand = treecorr.Catalog(x=x3, y=y3)
    ng = treecorr.NGCorrelation(bin_size=0.2, min_sep=10., max_sep=30.)
    rg = treecorr.NGCorrelation(bin_size=0.2, min_sep=10., max_sep=30.)
    ng.process(lens, source)
    rg.process(rand, source)

    print('single run:')
    print('Uncompensated')
    print('ratio = ',ng.varxi / var_xi_1)
    print('max relerr for xi = ',np.max(np.abs((ng.varxi - var_xi_1)/var_xi_1)))
    np.testing.assert_allclose(ng.varxi, var_xi_1, rtol=0.6)

    xi, xi_im, varxi = ng.calculateXi(rg=rg)
    print('Compensated')
    print('ratio = ',varxi / var_xi_2)
    print('max relerr for xi = ',np.max(np.abs((varxi - var_xi_2)/var_xi_2)))
    np.testing.assert_allclose(varxi, var_xi_2, rtol=0.5)


@timer
def test_double():
    # Test in response to issue #134.  That was about GG correlations, but the same bug could
    # sho up here, so check that duplicating all the lenses gives the same answer.
    # Use same signal as in test_ng.

    nlens = 1000
    nsource = 100000
    gamma0 = 0.05
    r0 = 10.
    L = 50. * r0
    rng = np.random.RandomState(8675309)
    xl1 = (rng.random_sample(nlens)-0.5) * L
    yl1 = (rng.random_sample(nlens)-0.5) * L
    xs = (rng.random_sample(nsource)-0.5) * L
    ys = (rng.random_sample(nsource)-0.5) * L
    g1 = np.zeros( (nsource,) )
    g2 = np.zeros( (nsource,) )
    for x,y in zip(xl1,yl1):
        dx = xs-x
        dy = ys-y
        r2 = dx**2 + dy**2
        gammat = gamma0 * np.exp(-0.5*r2/r0**2)
        g1 += -gammat * (dx**2-dy**2)/r2
        g2 += -gammat * (2.*dx*dy)/r2

    # Double the lenses
    xl2 = np.concatenate([xl1,xl1])
    yl2 = np.concatenate([yl1,yl1])

    lens_cat1 = treecorr.Catalog(ra=xl1, dec=yl1, ra_units='arcmin', dec_units='arcmin')
    lens_cat2 = treecorr.Catalog(ra=xl2, dec=yl2, ra_units='arcmin', dec_units='arcmin')
    source_cat = treecorr.Catalog(ra=xs, dec=ys, g1=g1, g2=g2, ra_units='arcmin', dec_units='arcmin')
    ng1 = treecorr.NGCorrelation(bin_size=0.1, min_sep=1., max_sep=20., sep_units='arcmin',
                                 verbose=1)
    ng2 = treecorr.NGCorrelation(bin_size=0.1, min_sep=1., max_sep=20., sep_units='arcmin',
                                 verbose=1)

    print('Normal case (all lenses different)')
    ng1.process(lens_cat1, source_cat)
    print('ng.xi = ',ng1.xi)
    print('ng.xi_im = ',ng1.xi_im)

    print('Duplicated lenses')
    ng2.process(lens_cat2, source_cat)
    print('ng.xi = ',ng2.xi)
    print('ng.xi_im = ',ng2.xi_im)

    print('Difference')
    print('delta ng.xi = ',ng2.xi-ng1.xi)
    print('delta ng.xi_im = ',ng2.xi_im-ng1.xi_im)
    print('max diff = ',max(abs(ng2.xi - ng1.xi)))
    np.testing.assert_allclose(ng2.xi, ng1.xi, rtol=1.e-3)
    np.testing.assert_allclose(ng2.xi_im, ng1.xi_im, rtol=1.e-3, atol=1.e-6)


if __name__ == '__main__':
    test_direct()
    test_direct_spherical()
    test_pairwise()
    test_single()
    test_pairwise2()
    test_spherical()
    test_ng()
    test_nmap()
    test_pieces()
    test_haloellip()
    test_varxi()
    test_double()
