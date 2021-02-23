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
import treecorr
import os
import coord
import fitsio

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
    k1 = rng.normal(5,1, (ngal,) )

    x2 = rng.normal(0,s, (ngal,) )
    y2 = rng.normal(0,s, (ngal,) )
    w2 = rng.random_sample(ngal)
    g12 = rng.normal(0,0.2, (ngal,) )
    g22 = rng.normal(0,0.2, (ngal,) )

    cat1 = treecorr.Catalog(x=x1, y=y1, w=w1, k=k1)
    cat2 = treecorr.Catalog(x=x2, y=y2, w=w2, g1=g12, g2=g22)

    min_sep = 1.
    max_sep = 50.
    nbins = 50
    bin_size = np.log(max_sep/min_sep) / nbins
    kg = treecorr.KGCorrelation(min_sep=min_sep, max_sep=max_sep, nbins=nbins, brute=True)
    kg.process(cat1, cat2)

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
        xi = -ww * k1[i] * (g12 + 1j*g22) * expmialpha**2

        index = np.floor(np.log(r/min_sep) / bin_size).astype(int)
        mask = (index >= 0) & (index < nbins)
        np.add.at(true_npairs, index[mask], 1)
        np.add.at(true_weight, index[mask], ww[mask])
        np.add.at(true_xi, index[mask], xi[mask])

    true_xi /= true_weight

    print('true_npairs = ',true_npairs)
    print('diff = ',kg.npairs - true_npairs)
    np.testing.assert_array_equal(kg.npairs, true_npairs)

    print('true_weight = ',true_weight)
    print('diff = ',kg.weight - true_weight)
    np.testing.assert_allclose(kg.weight, true_weight, rtol=1.e-5, atol=1.e-8)

    print('true_xi = ',true_xi)
    print('kg.xi = ',kg.xi)
    print('kg.xi_im = ',kg.xi_im)
    np.testing.assert_allclose(kg.xi, true_xi.real, rtol=1.e-4, atol=1.e-8)
    np.testing.assert_allclose(kg.xi_im, true_xi.imag, rtol=1.e-4, atol=1.e-8)

    # Check that running via the corr2 script works correctly.
    config = treecorr.config.read_config('configs/kg_direct.yaml')
    cat1.write(config['file_name'])
    cat2.write(config['file_name2'])
    treecorr.corr2(config)
    data = fitsio.read(config['kg_file_name'])
    np.testing.assert_allclose(data['r_nom'], kg.rnom)
    np.testing.assert_allclose(data['npairs'], kg.npairs)
    np.testing.assert_allclose(data['weight'], kg.weight)
    np.testing.assert_allclose(data['kgamT'], kg.xi, rtol=1.e-3)
    np.testing.assert_allclose(data['kgamX'], kg.xi_im, rtol=1.e-3)

    # Invalid with only one file_name
    del config['file_name2']
    with assert_raises(TypeError):
        treecorr.corr2(config)

    # Repeat with binslop = 0, since code is different for bin_slop=0 and brute=True.
    # And don't do any top-level recursion so we actually test not going to the leaves.
    kg = treecorr.KGCorrelation(min_sep=min_sep, max_sep=max_sep, nbins=nbins, bin_slop=0,
                                max_top=0)
    kg.process(cat1, cat2)
    np.testing.assert_array_equal(kg.npairs, true_npairs)
    np.testing.assert_allclose(kg.weight, true_weight, rtol=1.e-5, atol=1.e-8)
    np.testing.assert_allclose(kg.xi, true_xi.real, rtol=1.e-3, atol=1.e-3)
    np.testing.assert_allclose(kg.xi_im, true_xi.imag, rtol=1.e-3, atol=1.e-3)

    # Check a few basic operations with a KGCorrelation object.
    do_pickle(kg)

    kg2 = kg.copy()
    kg2 += kg
    np.testing.assert_allclose(kg2.npairs, 2*kg.npairs)
    np.testing.assert_allclose(kg2.weight, 2*kg.weight)
    np.testing.assert_allclose(kg2.meanr, 2*kg.meanr)
    np.testing.assert_allclose(kg2.meanlogr, 2*kg.meanlogr)
    np.testing.assert_allclose(kg2.xi, 2*kg.xi)
    np.testing.assert_allclose(kg2.xi_im, 2*kg.xi_im)

    kg2.clear()
    kg2 += kg
    np.testing.assert_allclose(kg2.npairs, kg.npairs)
    np.testing.assert_allclose(kg2.weight, kg.weight)
    np.testing.assert_allclose(kg2.meanr, kg.meanr)
    np.testing.assert_allclose(kg2.meanlogr, kg.meanlogr)
    np.testing.assert_allclose(kg2.xi, kg.xi)
    np.testing.assert_allclose(kg2.xi_im, kg.xi_im)

    ascii_name = 'output/kg_ascii.txt'
    kg.write(ascii_name, precision=16)
    kg3 = treecorr.KGCorrelation(min_sep=min_sep, max_sep=max_sep, nbins=nbins)
    kg3.read(ascii_name)
    np.testing.assert_allclose(kg3.npairs, kg.npairs)
    np.testing.assert_allclose(kg3.weight, kg.weight)
    np.testing.assert_allclose(kg3.meanr, kg.meanr)
    np.testing.assert_allclose(kg3.meanlogr, kg.meanlogr)
    np.testing.assert_allclose(kg3.xi, kg.xi)
    np.testing.assert_allclose(kg3.xi_im, kg.xi_im)

    fits_name = 'output/kg_fits.fits'
    kg.write(fits_name)
    kg4 = treecorr.KGCorrelation(min_sep=min_sep, max_sep=max_sep, nbins=nbins)
    kg4.read(fits_name)
    np.testing.assert_allclose(kg4.npairs, kg.npairs)
    np.testing.assert_allclose(kg4.weight, kg.weight)
    np.testing.assert_allclose(kg4.meanr, kg.meanr)
    np.testing.assert_allclose(kg4.meanlogr, kg.meanlogr)
    np.testing.assert_allclose(kg4.xi, kg.xi)
    np.testing.assert_allclose(kg4.xi_im, kg.xi_im)

    with assert_raises(TypeError):
        kg2 += config
    kg4 = treecorr.KGCorrelation(min_sep=min_sep/2, max_sep=max_sep, nbins=nbins)
    with assert_raises(ValueError):
        kg2 += kg4
    kg5 = treecorr.KGCorrelation(min_sep=min_sep, max_sep=max_sep*2, nbins=nbins)
    with assert_raises(ValueError):
        kg2 += kg5
    kg6 = treecorr.KGCorrelation(min_sep=min_sep, max_sep=max_sep, nbins=nbins*2)
    with assert_raises(ValueError):
        kg2 += kg6

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
    k1 = rng.normal(5,1, (ngal,) )

    x2 = rng.normal(0,s, (ngal,) )
    y2 = rng.normal(0,s, (ngal,) ) + 200
    z2 = rng.normal(0,s, (ngal,) )
    w2 = rng.random_sample(ngal)
    g12 = rng.normal(0,0.2, (ngal,) )
    g22 = rng.normal(0,0.2, (ngal,) )

    ra1, dec1 = coord.CelestialCoord.xyz_to_radec(x1,y1,z1)
    ra2, dec2 = coord.CelestialCoord.xyz_to_radec(x2,y2,z2)

    cat1 = treecorr.Catalog(ra=ra1, dec=dec1, ra_units='rad', dec_units='rad', w=w1, k=k1)
    cat2 = treecorr.Catalog(ra=ra2, dec=dec2, ra_units='rad', dec_units='rad', w=w2, g1=g12, g2=g22)

    min_sep = 1.
    max_sep = 10.
    nbins = 50
    bin_size = np.log(max_sep/min_sep) / nbins
    kg = treecorr.KGCorrelation(min_sep=min_sep, max_sep=max_sep, nbins=nbins,
                                sep_units='deg', brute=True)
    kg.process(cat1, cat2)

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
            xi = -ww * k1[i] * g2

            true_npairs[index] += 1
            true_weight[index] += ww
            true_xi[index] += xi

    true_xi /= true_weight

    print('true_npairs = ',true_npairs)
    print('diff = ',kg.npairs - true_npairs)
    np.testing.assert_array_equal(kg.npairs, true_npairs)

    print('true_weight = ',true_weight)
    print('diff = ',kg.weight - true_weight)
    np.testing.assert_allclose(kg.weight, true_weight, rtol=1.e-5, atol=1.e-8)

    print('true_xi = ',true_xi)
    print('kg.xi = ',kg.xi)
    np.testing.assert_allclose(kg.xi, true_xi.real, rtol=1.e-4, atol=1.e-8)
    np.testing.assert_allclose(kg.xi_im, true_xi.imag, rtol=1.e-4, atol=1.e-8)

    # Check that running via the corr2 script works correctly.
    config = treecorr.config.read_config('configs/kg_direct_spherical.yaml')
    cat1.write(config['file_name'])
    cat2.write(config['file_name2'])
    treecorr.corr2(config)
    data = fitsio.read(config['kg_file_name'])
    np.testing.assert_allclose(data['r_nom'], kg.rnom)
    np.testing.assert_allclose(data['npairs'], kg.npairs)
    np.testing.assert_allclose(data['weight'], kg.weight)
    np.testing.assert_allclose(data['kgamT'], kg.xi, rtol=1.e-3)
    np.testing.assert_allclose(data['kgamX'], kg.xi_im, rtol=1.e-3)

    # Repeat with binslop = 0
    # And don't do any top-level recursion so we actually test not going to the leaves.
    kg = treecorr.KGCorrelation(min_sep=min_sep, max_sep=max_sep, nbins=nbins,
                                sep_units='deg', bin_slop=0, max_top=0)
    kg.process(cat1, cat2)
    np.testing.assert_array_equal(kg.npairs, true_npairs)
    np.testing.assert_allclose(kg.weight, true_weight, rtol=1.e-5, atol=1.e-8)
    np.testing.assert_allclose(kg.xi, true_xi.real, rtol=1.e-3, atol=1.e-3)
    np.testing.assert_allclose(kg.xi_im, true_xi.imag, rtol=1.e-3, atol=1.e-3)


@timer
def test_pairwise():
    # Test the pairwise option.

    ngal = 1000
    s = 10.
    rng = np.random.RandomState(8675309)
    x1 = rng.normal(0,s, (ngal,) )
    y1 = rng.normal(0,s, (ngal,) )
    w1 = rng.random_sample(ngal)
    k1 = rng.normal(5,1, (ngal,) )

    x2 = rng.normal(0,s, (ngal,) )
    y2 = rng.normal(0,s, (ngal,) )
    w2 = rng.random_sample(ngal)
    g12 = rng.normal(0,0.2, (ngal,) )
    g22 = rng.normal(0,0.2, (ngal,) )

    w1 = np.ones_like(w1)
    w2 = np.ones_like(w2)

    cat1 = treecorr.Catalog(x=x1, y=y1, w=w1, k=k1)
    cat2 = treecorr.Catalog(x=x2, y=y2, w=w2, g1=g12, g2=g22)

    min_sep = 5.
    max_sep = 50.
    nbins = 10
    bin_size = np.log(max_sep/min_sep) / nbins
    kg = treecorr.KGCorrelation(min_sep=min_sep, max_sep=max_sep, nbins=nbins)
    with assert_warns(FutureWarning):
        kg.process_pairwise(cat1, cat2)
    kg.finalize(cat1.vark, cat2.varg)

    true_npairs = np.zeros(nbins, dtype=int)
    true_weight = np.zeros(nbins, dtype=float)
    true_xi = np.zeros(nbins, dtype=complex)

    rsq = (x1-x2)**2 + (y1-y2)**2
    r = np.sqrt(rsq)
    expmialpha = ((x1-x2) - 1j*(y1-y2)) / r

    ww = w1 * w2
    xi = -ww * k1 * (g12 + 1j*g22) * expmialpha**2

    index = np.floor(np.log(r/min_sep) / bin_size).astype(int)
    mask = (index >= 0) & (index < nbins)
    np.add.at(true_npairs, index[mask], 1)
    np.add.at(true_weight, index[mask], ww[mask])
    np.add.at(true_xi, index[mask], xi[mask])

    true_xi /= true_weight

    np.testing.assert_array_equal(kg.npairs, true_npairs)
    np.testing.assert_allclose(kg.weight, true_weight, rtol=1.e-5, atol=1.e-8)
    np.testing.assert_allclose(kg.xi, true_xi.real, rtol=1.e-4, atol=1.e-8)
    np.testing.assert_allclose(kg.xi_im, true_xi.imag, rtol=1.e-4, atol=1.e-8)

    # If cats have names, then the logger will mention them.
    # Also, test running with optional args.
    cat1.name = "first"
    cat2.name = "second"
    with CaptureLog() as cl:
        kg.logger = cl.logger
        with assert_warns(FutureWarning):
            kg.process_pairwise(cat1, cat2, metric='Euclidean', num_threads=2)
    assert "for cats first, second" in cl.output


@timer
def test_single():
    # Use gamma_t(r) = gamma0 exp(-r^2/2r0^2) around a single lens
    # i.e. gamma(r) = -gamma0 exp(-r^2/2r0^2) (x+iy)^2/r^2

    nsource = 100000
    gamma0 = 0.05
    kappa = 0.23
    r0 = 10.
    L = 5. * r0
    rng = np.random.RandomState(8675309)
    x = (rng.random_sample(nsource)-0.5) * L
    y = (rng.random_sample(nsource)-0.5) * L
    r2 = (x**2 + y**2)
    gammat = gamma0 * np.exp(-0.5*r2/r0**2)
    g1 = -gammat * (x**2-y**2)/r2
    g2 = -gammat * (2.*x*y)/r2

    lens_cat = treecorr.Catalog(x=[0], y=[0], k=[kappa],  x_units='arcmin', y_units='arcmin')
    source_cat = treecorr.Catalog(x=x, y=y, g1=g1, g2=g2, x_units='arcmin', y_units='arcmin')
    kg = treecorr.KGCorrelation(bin_size=0.1, min_sep=1., max_sep=20., sep_units='arcmin',
                                verbose=1)
    kg.process(lens_cat, source_cat)

    # log(<R>) != <logR>, but it should be close:
    print('meanlogr - log(meanr) = ',kg.meanlogr - np.log(kg.meanr))
    np.testing.assert_allclose(kg.meanlogr, np.log(kg.meanr), atol=1.e-3)

    r = kg.meanr
    true_kgt = kappa * gamma0 * np.exp(-0.5*r**2/r0**2)

    print('kg.xi = ',kg.xi)
    print('kg.xi_im = ',kg.xi_im)
    print('true_gammat = ',true_kgt)
    print('ratio = ',kg.xi / true_kgt)
    print('diff = ',kg.xi - true_kgt)
    print('max diff = ',max(abs(kg.xi - true_kgt)))
    np.testing.assert_allclose(kg.xi, true_kgt, rtol=1.e-2)
    np.testing.assert_allclose(kg.xi_im, 0, atol=1.e-4)

    # Check that we get the same result using the corr2 function:
    lens_cat.write(os.path.join('data','kg_single_lens.dat'))
    source_cat.write(os.path.join('data','kg_single_source.dat'))
    config = treecorr.read_config('configs/kg_single.yaml')
    config['verbose'] = 0
    treecorr.corr2(config)
    corr2_output = np.genfromtxt(os.path.join('output','kg_single.out'), names=True,
                                 skip_header=1)
    print('kg.xi = ',kg.xi)
    print('from corr2 output = ',corr2_output['kgamT'])
    print('ratio = ',corr2_output['kgamT']/kg.xi)
    print('diff = ',corr2_output['kgamT']-kg.xi)
    np.testing.assert_allclose(corr2_output['kgamT'], kg.xi, rtol=1.e-3)

    print('xi_im from corr2 output = ',corr2_output['kgamX'])
    np.testing.assert_allclose(corr2_output['kgamX'], 0., atol=1.e-4)


@timer
def test_pairwise2():
    # Test the same profile, but with the pairwise calcualtion:
    nsource = 100000
    gamma0 = 0.05
    kappa = 0.23
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
    k = kappa * np.ones(nsource)

    lens_cat = treecorr.Catalog(x=dx, y=dx, k=k, x_units='arcmin', y_units='arcmin')
    source_cat = treecorr.Catalog(x=x+dx, y=y+dx, g1=g1, g2=g2, x_units='arcmin', y_units='arcmin')
    kg = treecorr.KGCorrelation(bin_size=0.1, min_sep=1., max_sep=20., sep_units='arcmin',
                                verbose=1, pairwise=True)
    with assert_warns(FutureWarning):
        kg.process(lens_cat, source_cat)

    r = kg.meanr
    true_kgt = kappa * gamma0 * np.exp(-0.5*r**2/r0**2)

    print('kg.xi = ',kg.xi)
    print('kg.xi_im = ',kg.xi_im)
    print('true_gammat = ',true_kgt)
    print('ratio = ',kg.xi / true_kgt)
    print('diff = ',kg.xi - true_kgt)
    print('max diff = ',max(abs(kg.xi - true_kgt)))
    np.testing.assert_allclose(kg.xi, true_kgt, rtol=1.e-2)
    np.testing.assert_allclose(kg.xi_im, 0, atol=1.e-4)

    # Check that we get the same result using the corr2 function
    lens_cat.write(os.path.join('data','kg_pairwise_lens.dat'))
    source_cat.write(os.path.join('data','kg_pairwise_source.dat'))
    config = treecorr.read_config('configs/kg_pairwise.yaml')
    config['verbose'] = 0
    with assert_warns(FutureWarning):
        treecorr.corr2(config)
    corr2_output = np.genfromtxt(os.path.join('output','kg_pairwise.out'), names=True,
                                 skip_header=1)
    print('kg.xi = ',kg.xi)
    print('from corr2 output = ',corr2_output['kgamT'])
    print('ratio = ',corr2_output['kgamT']/kg.xi)
    print('diff = ',corr2_output['kgamT']-kg.xi)
    np.testing.assert_allclose(corr2_output['kgamT'], kg.xi, rtol=1.e-3)

    print('xi_im from corr2 output = ',corr2_output['kgamX'])
    np.testing.assert_allclose(corr2_output['kgamX'], 0., atol=1.e-4)


@timer
def test_kg():
    # Use gamma_t(r) = gamma0 exp(-r^2/2r0^2) around a bunch of foreground lenses.
    # i.e. gamma(r) = -gamma0 exp(-r^2/2r0^2) (x+iy)^2/r^2
    # For each lens, we divide this by a random kappa value assigned to that lens, so
    # the final kg output shoudl be just gamma_t.

    nlens = 1000
    nsource = 30000
    r0 = 10.
    L = 50. * r0

    gamma0 = 0.05
    rng = np.random.RandomState(8675309)
    xl = (rng.random_sample(nlens)-0.5) * L
    yl = (rng.random_sample(nlens)-0.5) * L
    kl = rng.normal(0.23, 0.05, (nlens,) )
    xs = (rng.random_sample(nsource)-0.5) * L
    ys = (rng.random_sample(nsource)-0.5) * L
    g1 = np.zeros( (nsource,) )
    g2 = np.zeros( (nsource,) )
    for x,y,k in zip(xl,yl,kl):
        dx = xs-x
        dy = ys-y
        r2 = dx**2 + dy**2
        gammat = gamma0 * np.exp(-0.5*r2/r0**2) / k
        g1 += -gammat * (dx**2-dy**2)/r2
        g2 += -gammat * (2.*dx*dy)/r2

    lens_cat = treecorr.Catalog(x=xl, y=yl, k=kl, x_units='arcmin', y_units='arcmin')
    source_cat = treecorr.Catalog(x=xs, y=ys, g1=g1, g2=g2, x_units='arcmin', y_units='arcmin')
    kg = treecorr.KGCorrelation(bin_size=0.1, min_sep=1., max_sep=20., sep_units='arcmin',
                                verbose=1)
    kg.process(lens_cat, source_cat)

    r = kg.meanr
    true_gt = gamma0 * np.exp(-0.5*r**2/r0**2)

    print('kg.xi = ',kg.xi)
    print('kg.xi_im = ',kg.xi_im)
    print('true_gammat = ',true_gt)
    print('ratio = ',kg.xi / true_gt)
    print('diff = ',kg.xi - true_gt)
    print('max diff = ',max(abs(kg.xi - true_gt)))
    np.testing.assert_allclose(kg.xi, true_gt, rtol=0.1)
    np.testing.assert_allclose(kg.xi_im, 0., atol=1.e-2)

    # Check that we get the same result using the corr2 function:
    lens_cat.write(os.path.join('data','kg_lens.dat'))
    source_cat.write(os.path.join('data','kg_source.dat'))
    config = treecorr.read_config('configs/kg.yaml')
    config['verbose'] = 0
    config['precision'] = 8
    treecorr.corr2(config)
    corr2_output = np.genfromtxt(os.path.join('output','kg.out'), names=True, skip_header=1)
    print('kg.xi = ',kg.xi)
    print('from corr2 output = ',corr2_output['kgamT'])
    print('ratio = ',corr2_output['kgamT']/kg.xi)
    print('diff = ',corr2_output['kgamT']-kg.xi)
    np.testing.assert_allclose(corr2_output['kgamT'], kg.xi, rtol=1.e-3)

    print('xi_im from corr2 output = ',corr2_output['kgamX'])
    np.testing.assert_allclose(corr2_output['kgamX'], 0., atol=1.e-2)

    # Check the fits write option
    out_file_name1 = os.path.join('output','kg_out1.fits')
    kg.write(out_file_name1)
    data = fitsio.read(out_file_name1)
    np.testing.assert_almost_equal(data['r_nom'], np.exp(kg.logr))
    np.testing.assert_almost_equal(data['meanr'], kg.meanr)
    np.testing.assert_almost_equal(data['meanlogr'], kg.meanlogr)
    np.testing.assert_almost_equal(data['kgamT'], kg.xi)
    np.testing.assert_almost_equal(data['kgamX'], kg.xi_im)
    np.testing.assert_almost_equal(data['sigma'], np.sqrt(kg.varxi))
    np.testing.assert_almost_equal(data['weight'], kg.weight)
    np.testing.assert_almost_equal(data['npairs'], kg.npairs)

    # Check the read function
    kg2 = treecorr.KGCorrelation(bin_size=0.1, min_sep=1., max_sep=20., sep_units='arcmin')
    kg2.read(out_file_name1)
    np.testing.assert_almost_equal(kg2.logr, kg.logr)
    np.testing.assert_almost_equal(kg2.meanr, kg.meanr)
    np.testing.assert_almost_equal(kg2.meanlogr, kg.meanlogr)
    np.testing.assert_almost_equal(kg2.xi, kg.xi)
    np.testing.assert_almost_equal(kg2.xi_im, kg.xi_im)
    np.testing.assert_almost_equal(kg2.varxi, kg.varxi)
    np.testing.assert_almost_equal(kg2.weight, kg.weight)
    np.testing.assert_almost_equal(kg2.npairs, kg.npairs)
    assert kg2.coords == kg.coords
    assert kg2.metric == kg.metric
    assert kg2.sep_units == kg.sep_units
    assert kg2.bin_type == kg.bin_type


@timer
def test_varxi():
    # Test that varxi is correct (or close) based on actual variance of many runs.

    # Signal doesn't matter much.  Use the one from test_gg.
    gamma0 = 0.05
    kappa0 = 0.03
    r0 = 10.
    L = 50.*r0
    rng = np.random.RandomState(8675309)

    # Note: to get a good estimate of var(xi), you need a lot of runs.  The number of
    # runs matters much more than the number of galaxies for getting this to pass.
    if __name__ == '__main__':
        ngal = 1000
        nruns = 50000
        tol_factor = 1
    else:
        ngal = 100
        nruns = 5000
        tol_factor = 5

    all_kgs = []
    for run in range(nruns):
        # In addition to the shape noise below, there is shot noise from the random x,y positions.
        x = (rng.random_sample(ngal)-0.5) * L
        y = (rng.random_sample(ngal)-0.5) * L
        # Varied weights are hard, but at least check that non-unit weights work correctly.
        w = np.ones_like(x) * 5
        r2 = (x**2 + y**2)/r0**2
        g1 = -gamma0 * np.exp(-r2/2.) * (x**2-y**2)/r0**2
        g2 = -gamma0 * np.exp(-r2/2.) * (2.*x*y)/r0**2
        k = kappa0 * np.exp(-r2/2.)
        # This time, add some shape noise (different each run).
        g1 += rng.normal(0, 0.3, size=ngal)
        g2 += rng.normal(0, 0.3, size=ngal)
        k += rng.normal(0, 0.1, size=ngal)

        cat = treecorr.Catalog(x=x, y=y, w=w, g1=g1, g2=g2, k=k)
        kg = treecorr.KGCorrelation(bin_size=0.1, min_sep=10., max_sep=100.)
        kg.process(cat, cat)
        all_kgs.append(kg)

    mean_xi = np.mean([kg.xi for kg in all_kgs], axis=0)
    var_xi = np.var([kg.xi for kg in all_kgs], axis=0)
    mean_varxi = np.mean([kg.varxi for kg in all_kgs], axis=0)

    print('mean_xi = ',mean_xi)
    print('mean_varxi = ',mean_varxi)
    print('var_xi = ',var_xi)
    print('ratio = ',var_xi / mean_varxi)
    print('max relerr for xi = ',np.max(np.abs((var_xi - mean_varxi)/var_xi)))
    np.testing.assert_allclose(mean_varxi, var_xi, rtol=0.02 * tol_factor)

@timer
def test_negw():
    # Test having some weights be < 0.
    # Daniel Gruen had a use case where some weights were negative, and it used to die
    # with an assert error in the C++ layer.

    ngal = 200
    s = 10.
    rng = np.random.RandomState(8675309)
    x1 = rng.normal(0,s, (ngal,) )
    y1 = rng.normal(0,s, (ngal,) )
    w1 = rng.uniform(-0.2, 1.0, (ngal,))
    k1 = rng.normal(5,1, (ngal,) )

    x2 = rng.normal(0,s, (ngal,) )
    y2 = rng.normal(0,s, (ngal,) )
    w2 = rng.uniform(-3.0, 4.0, (ngal,))
    g12 = rng.normal(0,0.2, (ngal,) )
    g22 = rng.normal(0,0.2, (ngal,) )

    cat1 = treecorr.Catalog(x=x1, y=y1, w=w1, k=k1)
    cat2 = treecorr.Catalog(x=x2, y=y2, w=w2, g1=g12, g2=g22)

    min_sep = 1.
    max_sep = 50.
    nbins = 50
    bin_size = np.log(max_sep/min_sep) / nbins
    kg = treecorr.KGCorrelation(min_sep=min_sep, max_sep=max_sep, nbins=nbins, brute=True)
    kg.process(cat1, cat2)

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
        xi = -ww * k1[i] * (g12 + 1j*g22) * expmialpha**2

        index = np.floor(np.log(r/min_sep) / bin_size).astype(int)
        mask = (index >= 0) & (index < nbins)
        np.add.at(true_npairs, index[mask], 1)
        np.add.at(true_weight, index[mask], ww[mask])
        np.add.at(true_xi, index[mask], xi[mask])

    true_xi /= true_weight

    print('true_npairs = ',true_npairs)
    print('diff = ',kg.npairs - true_npairs)
    np.testing.assert_array_equal(kg.npairs, true_npairs)

    print('true_weight = ',true_weight)
    print('diff = ',kg.weight - true_weight)
    np.testing.assert_allclose(kg.weight, true_weight, rtol=1.e-5, atol=1.e-8)

    print('true_xi = ',true_xi)
    print('kg.xi = ',kg.xi)
    print('kg.xi_im = ',kg.xi_im)
    np.testing.assert_allclose(kg.xi, true_xi.real, rtol=1.e-4, atol=1.e-8)
    np.testing.assert_allclose(kg.xi_im, true_xi.imag, rtol=1.e-4, atol=1.e-8)


if __name__ == '__main__':
    test_direct()
    test_direct_spherical()
    test_pairwise()
    test_single()
    test_pairwise2()
    test_kg()
    test_varxi()
    test_negw()
