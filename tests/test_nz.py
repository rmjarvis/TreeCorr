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
    z12 = rng.normal(0,0.2, (ngal,) )
    z22 = rng.normal(0,0.2, (ngal,) )

    cat1 = treecorr.Catalog(x=x1, y=y1, w=w1)
    cat2 = treecorr.Catalog(x=x2, y=y2, w=w2, z1=z12, z2=z22)

    min_sep = 1.
    max_sep = 50.
    nbins = 50
    bin_size = np.log(max_sep/min_sep) / nbins
    nz = treecorr.NZCorrelation(min_sep=min_sep, max_sep=max_sep, nbins=nbins, brute=True)
    nz.process(cat1, cat2)

    true_npairs = np.zeros(nbins, dtype=int)
    true_weight = np.zeros(nbins, dtype=float)
    true_xi = np.zeros(nbins, dtype=complex)
    for i in range(ngal):
        # It's hard to do all the pairs at once with numpy operations (although maybe possible).
        # But we can at least do all the pairs for each entry in cat1 at once with arrays.
        rsq = (x1[i]-x2)**2 + (y1[i]-y2)**2
        r = np.sqrt(rsq)

        ww = w1[i] * w2
        xi = ww * (z12 + 1j*z22)

        index = np.floor(np.log(r/min_sep) / bin_size).astype(int)
        mask = (index >= 0) & (index < nbins)
        np.add.at(true_npairs, index[mask], 1)
        np.add.at(true_weight, index[mask], ww[mask])
        np.add.at(true_xi, index[mask], xi[mask])

    true_xi /= true_weight

    print('true_npairs = ',true_npairs)
    print('diff = ',nz.npairs - true_npairs)
    np.testing.assert_array_equal(nz.npairs, true_npairs)

    print('true_weight = ',true_weight)
    print('diff = ',nz.weight - true_weight)
    np.testing.assert_allclose(nz.weight, true_weight, rtol=1.e-6, atol=1.e-8)

    print('true_xi = ',true_xi)
    print('nz.xi = ',nz.xi)
    print('nz.xi_im = ',nz.xi_im)
    np.testing.assert_allclose(nz.xi, true_xi.real, rtol=1.e-6, atol=1.e-8)
    np.testing.assert_allclose(nz.xi_im, true_xi.imag, rtol=1.e-6, atol=1.e-8)

    # Check that running via the corr2 script works correctly.
    config = treecorr.config.read_config('configs/nz_direct.yaml')
    try:
        import fitsio
    except ImportError:
        pass
    else:
        cat1.write(config['file_name'])
        cat2.write(config['file_name2'])
        with CaptureLog() as cl:
            treecorr.corr2(config, logger=cl.logger)
        assert "skipping z1_col" in cl.output
        data = fitsio.read(config['nz_file_name'])
        np.testing.assert_allclose(data['r_nom'], nz.rnom)
        np.testing.assert_allclose(data['npairs'], nz.npairs)
        np.testing.assert_allclose(data['weight'], nz.weight)
        np.testing.assert_allclose(data['z_real'], nz.xi)
        np.testing.assert_allclose(data['z_imag'], nz.xi_im)

        # When not using corr2, it's invalid to specify invalid z1_col, z2_col
        with assert_raises(ValueError):
            cat = treecorr.Catalog(config['file_name'], config)

        # Invalid with only one file_name
        del config['file_name2']
        with assert_raises(TypeError):
            treecorr.corr2(config)
        config['file_name2'] = 'data/nz_direct_cat2.fits'
        # Invalid to request compoensated if no rand_file
        config['nz_statistic'] = 'compensated'
        with assert_raises(TypeError):
            treecorr.corr2(config)

    # Repeat with binslop = 0
    # And don't do any top-level recursion so we actually test not going to the leaves.
    nz = treecorr.NZCorrelation(min_sep=min_sep, max_sep=max_sep, nbins=nbins, bin_slop=0,
                                max_top=0)
    nz.process(cat1, cat2)
    np.testing.assert_array_equal(nz.npairs, true_npairs)
    np.testing.assert_allclose(nz.weight, true_weight, rtol=1.e-6, atol=1.e-8)
    np.testing.assert_allclose(nz.xi, true_xi.real, atol=1.e-4)
    np.testing.assert_allclose(nz.xi_im, true_xi.imag, atol=2.e-4)

    # With angle_slop = 0, it goes back to being basically exact (to single precision).
    nz = treecorr.NZCorrelation(min_sep=min_sep, max_sep=max_sep, nbins=nbins, bin_slop=0,
                                angle_slop=0, max_top=0)
    nz.process(cat1, cat2)
    np.testing.assert_array_equal(nz.npairs, true_npairs)
    np.testing.assert_allclose(nz.weight, true_weight, rtol=1.e-6, atol=1.e-8)
    np.testing.assert_allclose(nz.xi, true_xi.real, rtol=1.e-6, atol=1.e-8)
    np.testing.assert_allclose(nz.xi_im, true_xi.imag, rtol=1.e-6, atol=1.e-8)

    # Check a few basic operations with a NZCorrelation object.
    do_pickle(nz)

    nz2 = nz.copy()
    nz2 += nz
    np.testing.assert_allclose(nz2.npairs, 2*nz.npairs)
    np.testing.assert_allclose(nz2.weight, 2*nz.weight)
    np.testing.assert_allclose(nz2.meanr, 2*nz.meanr)
    np.testing.assert_allclose(nz2.meanlogr, 2*nz.meanlogr)
    np.testing.assert_allclose(nz2.xi, 2*nz.xi)
    np.testing.assert_allclose(nz2.xi_im, 2*nz.xi_im)

    nz2.clear()
    nz2 += nz
    np.testing.assert_allclose(nz2.npairs, nz.npairs)
    np.testing.assert_allclose(nz2.weight, nz.weight)
    np.testing.assert_allclose(nz2.meanr, nz.meanr)
    np.testing.assert_allclose(nz2.meanlogr, nz.meanlogr)
    np.testing.assert_allclose(nz2.xi, nz.xi)
    np.testing.assert_allclose(nz2.xi_im, nz.xi_im)

    ascii_name = 'output/nz_ascii.txt'
    nz.write(ascii_name, precision=16)
    nz3 = treecorr.NZCorrelation(min_sep=min_sep, max_sep=max_sep, nbins=nbins, bin_type='Log')
    nz3.read(ascii_name)
    np.testing.assert_allclose(nz3.npairs, nz.npairs)
    np.testing.assert_allclose(nz3.weight, nz.weight)
    np.testing.assert_allclose(nz3.meanr, nz.meanr)
    np.testing.assert_allclose(nz3.meanlogr, nz.meanlogr)
    np.testing.assert_allclose(nz3.xi, nz.xi)
    np.testing.assert_allclose(nz3.xi_im, nz.xi_im)

    # Check that the repr is minimal
    assert repr(nz3) == f'NZCorrelation(min_sep={min_sep}, max_sep={max_sep}, nbins={nbins})'

    # Simpler API using from_file:
    with CaptureLog() as cl:
        nz3b = treecorr.NZCorrelation.from_file(ascii_name, logger=cl.logger)
    assert ascii_name in cl.output
    np.testing.assert_allclose(nz3b.npairs, nz.npairs)
    np.testing.assert_allclose(nz3b.weight, nz.weight)
    np.testing.assert_allclose(nz3b.meanr, nz.meanr)
    np.testing.assert_allclose(nz3b.meanlogr, nz.meanlogr)
    np.testing.assert_allclose(nz3b.xi, nz.xi)
    np.testing.assert_allclose(nz3b.xi_im, nz.xi_im)

    # or using the Corr2 base class
    with CaptureLog() as cl:
        nz3c = treecorr.Corr2.from_file(ascii_name, logger=cl.logger)
    assert ascii_name in cl.output
    np.testing.assert_allclose(nz3c.npairs, nz.npairs)
    np.testing.assert_allclose(nz3c.weight, nz.weight)
    np.testing.assert_allclose(nz3c.meanr, nz.meanr)
    np.testing.assert_allclose(nz3c.meanlogr, nz.meanlogr)
    np.testing.assert_allclose(nz3c.xi, nz.xi)
    np.testing.assert_allclose(nz3c.xi_im, nz.xi_im)

    try:
        import fitsio
    except ImportError:
        pass
    else:
        fits_name = 'output/nz_fits.fits'
        nz.write(fits_name)
        nz4 = treecorr.NZCorrelation(min_sep=min_sep, max_sep=max_sep, nbins=nbins)
        nz4.read(fits_name)
        np.testing.assert_allclose(nz4.npairs, nz.npairs)
        np.testing.assert_allclose(nz4.weight, nz.weight)
        np.testing.assert_allclose(nz4.meanr, nz.meanr)
        np.testing.assert_allclose(nz4.meanlogr, nz.meanlogr)
        np.testing.assert_allclose(nz4.xi, nz.xi)
        np.testing.assert_allclose(nz4.xi_im, nz.xi_im)

        nz4b = treecorr.NZCorrelation.from_file(fits_name)
        np.testing.assert_allclose(nz4b.npairs, nz.npairs)
        np.testing.assert_allclose(nz4b.weight, nz.weight)
        np.testing.assert_allclose(nz4b.meanr, nz.meanr)
        np.testing.assert_allclose(nz4b.meanlogr, nz.meanlogr)
        np.testing.assert_allclose(nz4b.xi, nz.xi)
        np.testing.assert_allclose(nz4b.xi_im, nz.xi_im)

    with assert_raises(TypeError):
        nz2 += config
    nz4 = treecorr.NZCorrelation(min_sep=min_sep/2, max_sep=max_sep, nbins=nbins)
    with assert_raises(ValueError):
        nz2 += nz4
    nz5 = treecorr.NZCorrelation(min_sep=min_sep, max_sep=max_sep*2, nbins=nbins)
    with assert_raises(ValueError):
        nz2 += nz5
    nz6 = treecorr.NZCorrelation(min_sep=min_sep, max_sep=max_sep, nbins=nbins*2)
    with assert_raises(ValueError):
        nz2 += nz6
    with assert_raises(ValueError):
        nz.process(cat1, cat2, patch_method='nonlocal')


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
    z12 = rng.normal(0,0.2, (ngal,) )
    z22 = rng.normal(0,0.2, (ngal,) )

    ra1, dec1 = coord.CelestialCoord.xyz_to_radec(x1,y1,z1)
    ra2, dec2 = coord.CelestialCoord.xyz_to_radec(x2,y2,z2)

    cat1 = treecorr.Catalog(ra=ra1, dec=dec1, ra_units='rad', dec_units='rad', w=w1)
    cat2 = treecorr.Catalog(ra=ra2, dec=dec2, ra_units='rad', dec_units='rad', w=w2, z1=z12, z2=z22)

    min_sep = 1.
    max_sep = 10.
    nbins = 50
    bin_size = np.log(max_sep/min_sep) / nbins
    nz = treecorr.NZCorrelation(min_sep=min_sep, max_sep=max_sep, nbins=nbins,
                                sep_units='deg', brute=True)
    nz.process(cat1, cat2)

    r1 = np.sqrt(x1**2 + y1**2 + z1**2)
    r2 = np.sqrt(x2**2 + y2**2 + z2**2)
    x1 /= r1;  y1 /= r1;  z1 /= r1
    x2 /= r2;  y2 /= r2;  z2 /= r2

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

            ww = w1[i] * w2[j]
            xi = ww * (z12[j] + 1j * z22[j])

            true_npairs[index] += 1
            true_weight[index] += ww
            true_xi[index] += xi

    true_xi /= true_weight

    print('true_npairs = ',true_npairs)
    print('diff = ',nz.npairs - true_npairs)
    np.testing.assert_array_equal(nz.npairs, true_npairs)

    print('true_weight = ',true_weight)
    print('diff = ',nz.weight - true_weight)
    np.testing.assert_allclose(nz.weight, true_weight, rtol=1.e-6, atol=1.e-8)

    print('true_xi = ',true_xi)
    print('nz.xi = ',nz.xi)
    print('nz.xi_im = ',nz.xi_im)
    np.testing.assert_allclose(nz.xi, true_xi.real, rtol=1.e-6, atol=1.e-8)
    np.testing.assert_allclose(nz.xi_im, true_xi.imag, rtol=1.e-6, atol=1.e-8)

    # Check that running via the corr2 script works correctly.
    config = treecorr.config.read_config('configs/nz_direct_spherical.yaml')
    try:
        import fitsio
    except ImportError:
        pass
    else:
        cat1.write(config['file_name'])
        cat2.write(config['file_name2'])
        treecorr.corr2(config)
        data = fitsio.read(config['nz_file_name'])
        np.testing.assert_allclose(data['r_nom'], nz.rnom)
        np.testing.assert_allclose(data['npairs'], nz.npairs)
        np.testing.assert_allclose(data['weight'], nz.weight)
        np.testing.assert_allclose(data['z_real'], nz.xi)
        np.testing.assert_allclose(data['z_imag'], nz.xi_im)

    # Repeat with binslop = 0
    # And don't do any top-level recursion so we actually test not going to the leaves.
    nz = treecorr.NZCorrelation(min_sep=min_sep, max_sep=max_sep, nbins=nbins,
                                sep_units='deg', bin_slop=0, max_top=0)
    nz.process(cat1, cat2)
    np.testing.assert_array_equal(nz.npairs, true_npairs)
    np.testing.assert_allclose(nz.weight, true_weight, rtol=1.e-6, atol=1.e-8)
    np.testing.assert_allclose(nz.xi, true_xi.real, rtol=1.e-6, atol=1.e-8)
    np.testing.assert_allclose(nz.xi_im, true_xi.imag, rtol=1.e-6, atol=1.e-8)


@timer
def test_single():
    # Use z(r) = z0 exp(-r^2/2r0^2) (1-r^2/2r0^2) around a single lens

    nsource = 300000
    z0 = 0.05 + 1j * 0.02
    r0 = 10.
    L = 5. * r0
    rng = np.random.RandomState(8675309)
    x = (rng.random_sample(nsource)-0.5) * L
    y = (rng.random_sample(nsource)-0.5) * L
    r2 = (x**2 + y**2)
    r = np.sqrt(r2)
    z = z0 * np.exp(-0.5*r2/r0**2) * (1.-0.5*r2/r0**2)
    z1 = np.real(z)
    z2 = np.imag(z)

    lens_cat = treecorr.Catalog(x=[0], y=[0], x_units='arcmin', y_units='arcmin')
    source_cat = treecorr.Catalog(x=x, y=y, z1=z1, z2=z2, x_units='arcmin', y_units='arcmin')
    nz = treecorr.NZCorrelation(bin_size=0.1, min_sep=1., max_sep=20., sep_units='arcmin',
                                verbose=1)
    nz.process(lens_cat, source_cat)

    # log(<R>) != <logR>, but it should be close:
    print('meanlogr - log(meanr) = ',nz.meanlogr - np.log(nz.meanr))
    np.testing.assert_allclose(nz.meanlogr, np.log(nz.meanr), atol=1.e-3)

    r = nz.meanr
    true_z = z0 * np.exp(-0.5*r**2/r0**2) * (1.-0.5*r**2/r0**2)

    print('nz.xi = ',nz.xi)
    print('nz.xi_im = ',nz.xi_im)
    print('true_z = ',true_z)
    print('ratio = ',nz.xi / true_z)
    print('diff = ',nz.xi - true_z)
    print('max diff = ',max(abs(nz.xi - true_z)))
    np.testing.assert_allclose(nz.xi, np.real(true_z), rtol=1.e-2, atol=1.e-4)
    np.testing.assert_allclose(nz.xi_im, np.imag(true_z), rtol=1.e-2, atol=1.e-4)

    # Check that we get the same result using the corr2 function:
    lens_cat.write(os.path.join('data','nz_single_lens.dat'))
    source_cat.write(os.path.join('data','nz_single_source.dat'))
    config = treecorr.read_config('configs/nz_single.yaml')
    config['verbose'] = 0
    treecorr.corr2(config)
    corr2_output = np.genfromtxt(os.path.join('output','nz_single.out'), names=True,
                                 skip_header=1)
    print('nz.xi = ',nz.xi)
    print('from corr2 output = ',corr2_output['z_real'])
    print('ratio = ',corr2_output['z_real']/nz.xi)
    print('diff = ',corr2_output['z_real']-nz.xi)
    np.testing.assert_allclose(corr2_output['z_real'], nz.xi, rtol=1.e-3)
    np.testing.assert_allclose(corr2_output['z_imag'], nz.xi_im, rtol=1.e-3)


@timer
def test_nz():
    # Use z(r) = z0 exp(-r^2/2r0^2) (1-r^2/2r0^2) around a bunch of foreground lenses.

    nlens = 1000
    nsource = 100000
    z0 = 0.05 + 1j*0.02
    r0 = 10.
    L = 100. * r0
    rng = np.random.RandomState(8675309)
    xl = (rng.random_sample(nlens)-0.5) * L
    yl = (rng.random_sample(nlens)-0.5) * L
    xs = (rng.random_sample(nsource)-0.5) * L
    ys = (rng.random_sample(nsource)-0.5) * L
    z1 = np.zeros( (nsource,) )
    z2 = np.zeros( (nsource,) )
    for x,y in zip(xl,yl):
        dx = xs-x
        dy = ys-y
        r2 = dx**2 + dy**2
        zz = z0 * np.exp(-0.5*r2/r0**2) * (1.-0.5*r2/r0**2)
        z1 += np.real(zz)
        z2 += np.imag(zz)

    lens_cat = treecorr.Catalog(x=xl, y=yl, x_units='arcmin', y_units='arcmin')
    source_cat = treecorr.Catalog(x=xs, y=ys, z1=z1, z2=z2, x_units='arcmin', y_units='arcmin')
    nz = treecorr.NZCorrelation(bin_size=0.1, min_sep=1., max_sep=20., sep_units='arcmin',
                                verbose=1)
    nz.process(lens_cat, source_cat)

    # Using nbins=None rather than omitting nbins is equivalent.
    nz2 = treecorr.NZCorrelation(bin_size=0.1, min_sep=1., max_sep=20., nbins=None, sep_units='arcmin')
    nz2.process(lens_cat, source_cat, num_threads=1)
    nz.process(lens_cat, source_cat, num_threads=1)
    assert nz2 == nz

    r = nz.meanr
    true_z = z0 * np.exp(-0.5*r**2/r0**2) * (1.-0.5*r**2/r0**2)

    print('nz.xi = ',nz.xi)
    print('nz.xi_im = ',nz.xi_im)
    print('true_z = ',true_z)
    print('ratio = ',nz.xi / true_z)
    print('diff = ',nz.xi - true_z)
    print('max diff = ',max(abs(nz.xi - true_z)))
    np.testing.assert_allclose(nz.xi, np.real(true_z), rtol=0.1, atol=2.e-3)
    np.testing.assert_allclose(nz.xi_im, np.imag(true_z), rtol=0.1, atol=2.e-3)

    nrand = nlens * 10
    xr = (rng.random_sample(nrand)-0.5) * L
    yr = (rng.random_sample(nrand)-0.5) * L
    rand_cat = treecorr.Catalog(x=xr, y=yr, x_units='arcmin', y_units='arcmin')
    rz = treecorr.NZCorrelation(bin_size=0.1, min_sep=1., max_sep=20., sep_units='arcmin',
                                verbose=1)
    rz.process(rand_cat, source_cat)
    print('rz.xi = ',rz.xi)
    xi, xi_im, varxi = nz.calculateXi(rz=rz)
    print('compensated xi = ',xi)
    print('compensated xi_im = ',xi_im)
    print('true_z = ',true_z)
    np.testing.assert_allclose(xi, np.real(true_z), rtol=0.05, atol=1.e-3)
    np.testing.assert_allclose(xi_im, np.imag(true_z), rtol=0.05, atol=1.e-3)

    # Check that we get the same result using the corr2 function:
    config = treecorr.read_config('configs/nz.yaml')
    try:
        import fitsio
    except ImportError:
        pass
    else:
        lens_cat.write(os.path.join('data','nz_lens.fits'))
        source_cat.write(os.path.join('data','nz_source.fits'))
        rand_cat.write(os.path.join('data','nz_rand.fits'))
        config['verbose'] = 0
        config['precision'] = 8
        treecorr.corr2(config)
        corr2_output = np.genfromtxt(os.path.join('output','nz.out'), names=True, skip_header=1)
        print('nz.xi = ',nz.xi)
        print('xi = ',xi)
        print('from corr2 output = ',corr2_output['z_real'])
        print('ratio = ',corr2_output['z_real']/xi)
        print('diff = ',corr2_output['z_real']-xi)
        np.testing.assert_allclose(corr2_output['z_real'], xi)
        print('xi_im from corr2 output = ',corr2_output['z_imag'])
        np.testing.assert_allclose(corr2_output['z_imag'], xi_im)

        # In the corr2 context, you can turn off the compensated bit, even if there are randoms
        # (e.g. maybe you only want randoms for some nn calculation, but not nz.)
        config['nz_statistic'] = 'simple'
        treecorr.corr2(config)
        corr2_output = np.genfromtxt(os.path.join('output','nz.out'), names=True, skip_header=1)
        xi_simple, _, _ = nz.calculateXi()
        np.testing.assert_equal(xi_simple, nz.xi)
        np.testing.assert_allclose(corr2_output['z_real'], xi_simple)

        # Check the fits write option
        out_file_name1 = os.path.join('output','nz_out1.fits')
        nz.write(out_file_name1)
        data = fitsio.read(out_file_name1)
        np.testing.assert_almost_equal(data['r_nom'], np.exp(nz.logr))
        np.testing.assert_almost_equal(data['meanr'], nz.meanr)
        np.testing.assert_almost_equal(data['meanlogr'], nz.meanlogr)
        np.testing.assert_almost_equal(data['z_real'], nz.xi)
        np.testing.assert_almost_equal(data['z_imag'], nz.xi_im)
        np.testing.assert_almost_equal(data['sigma'], np.sqrt(nz.varxi))
        np.testing.assert_almost_equal(data['weight'], nz.weight)
        np.testing.assert_almost_equal(data['npairs'], nz.npairs)

        out_file_name2 = os.path.join('output','nz_out2.fits')
        nz.write(out_file_name2, rz=rz)
        data = fitsio.read(out_file_name2)
        np.testing.assert_almost_equal(data['r_nom'], np.exp(nz.logr))
        np.testing.assert_almost_equal(data['meanr'], nz.meanr)
        np.testing.assert_almost_equal(data['meanlogr'], nz.meanlogr)
        np.testing.assert_almost_equal(data['z_real'], xi)
        np.testing.assert_almost_equal(data['z_imag'], xi_im)
        np.testing.assert_almost_equal(data['sigma'], np.sqrt(varxi))
        np.testing.assert_almost_equal(data['weight'], nz.weight)
        np.testing.assert_almost_equal(data['npairs'], nz.npairs)

        # Check the read function
        nz2 = treecorr.NZCorrelation.from_file(out_file_name2)
        np.testing.assert_almost_equal(nz2.logr, nz.logr)
        np.testing.assert_almost_equal(nz2.meanr, nz.meanr)
        np.testing.assert_almost_equal(nz2.meanlogr, nz.meanlogr)
        np.testing.assert_almost_equal(nz2.xi, nz.xi)
        np.testing.assert_almost_equal(nz2.xi_im, nz.xi_im)
        np.testing.assert_almost_equal(nz2.varxi, nz.varxi)
        np.testing.assert_almost_equal(nz2.weight, nz.weight)
        np.testing.assert_almost_equal(nz2.npairs, nz.npairs)
        assert nz2.coords == nz.coords
        assert nz2.metric == nz.metric
        assert nz2.sep_units == nz.sep_units
        assert nz2.bin_type == nz.bin_type


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
    z0 = 0.05 + 1j*0.03
    r0 = 10.
    L = 50. * r0
    rng = np.random.RandomState(8675309)
    xl = (rng.random_sample(nlens)-0.5) * L
    yl = (rng.random_sample(nlens)-0.5) * L
    xs = (rng.random_sample( (nsource,ncats) )-0.5) * L
    ys = (rng.random_sample( (nsource,ncats) )-0.5) * L
    z1 = np.zeros( (nsource,ncats) )
    z2 = np.zeros( (nsource,ncats) )
    w = rng.random_sample( (nsource,ncats) ) + 0.5
    for x,y in zip(xl,yl):
        dx = xs-x
        dy = ys-y
        r2 = dx**2 + dy**2
        zz = z0 * np.exp(-0.5*r2/r0**2) * (1.-r2/r0**2)
        z1 += np.real(zz)
        z2 += np.imag(zz)

    lens_cat = treecorr.Catalog(x=xl, y=yl, x_units='arcmin', y_units='arcmin')
    source_cats = [ treecorr.Catalog(x=xs[:,k], y=ys[:,k], z1=z1[:,k], z2=z2[:,k], w=w[:,k],
                                     x_units='arcmin', y_units='arcmin') for k in range(ncats) ]
    full_source_cat = treecorr.Catalog(x=xs.flatten(), y=ys.flatten(), w=w.flatten(),
                                       z1=z1.flatten(), z2=z2.flatten(),
                                       x_units='arcmin', y_units='arcmin')

    t0 = time.time()
    for k in range(ncats):
        # These could each be done on different machines in a real world application.
        nz = treecorr.NZCorrelation(bin_size=0.1, min_sep=1., max_sep=25., sep_units='arcmin',
                                    verbose=1)
        # These should use process_cross, not process, since we don't want to call finalize.
        nz.process_cross(lens_cat, source_cats[k])
        nz.write(os.path.join('output','nz_piece_%d.fits'%k))

    pieces_nz = treecorr.NZCorrelation(bin_size=0.1, min_sep=1., max_sep=25., sep_units='arcmin')
    for k in range(ncats):
        nz = pieces_nz.copy()
        nz.read(os.path.join('output','nz_piece_%d.fits'%k))
        pieces_nz += nz
    varz = treecorr.calculateVarZ(source_cats)
    pieces_nz.finalize(varz)
    t1 = time.time()
    print('time for piece-wise processing (including I/O) = ',t1-t0)

    full_nz = treecorr.NZCorrelation(bin_size=0.1, min_sep=1., max_sep=25., sep_units='arcmin',
                                     verbose=1)
    full_nz.process(lens_cat, full_source_cat)
    t2 = time.time()
    print('time for full processing = ',t2-t1)

    print('max error in meanr = ',np.max(pieces_nz.meanr - full_nz.meanr),)
    print('    max meanr = ',np.max(full_nz.meanr))
    print('max error in meanlogr = ',np.max(pieces_nz.meanlogr - full_nz.meanlogr),)
    print('    max meanlogr = ',np.max(full_nz.meanlogr))
    print('max error in weight = ',np.max(pieces_nz.weight - full_nz.weight),)
    print('    max weight = ',np.max(full_nz.weight))
    print('max error in xi = ',np.max(pieces_nz.xi - full_nz.xi),)
    print('    max xi = ',np.max(full_nz.xi))
    print('max error in xi_im = ',np.max(pieces_nz.xi_im - full_nz.xi_im),)
    print('    max xi_im = ',np.max(full_nz.xi_im))
    print('max error in varxi = ',np.max(pieces_nz.varxi - full_nz.varxi),)
    print('    max varxi = ',np.max(full_nz.varxi))
    np.testing.assert_allclose(pieces_nz.meanr, full_nz.meanr, rtol=2.e-3)
    np.testing.assert_allclose(pieces_nz.meanlogr, full_nz.meanlogr, atol=2.e-3)
    np.testing.assert_allclose(pieces_nz.weight, full_nz.weight, rtol=3.e-2)
    np.testing.assert_allclose(pieces_nz.xi, full_nz.xi, rtol=0.1)
    np.testing.assert_allclose(pieces_nz.xi_im, full_nz.xi_im, atol=2.e-3)
    np.testing.assert_allclose(pieces_nz.varxi, full_nz.varxi, rtol=3.e-2)

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
                                      z1=z1.flatten(), z2=z2.flatten(),
                                      wpos=w.flatten(), w=w2[k].flatten(),
                                      x_units='arcmin', y_units='arcmin') for k in range(ncats) ]

    t3 = time.time()
    nz2 = [ full_nz.copy() for k in range(ncats) ]
    for k in range(ncats):
        nz2[k].clear()
        nz2[k].process_cross(lens_cat, source_cats2[k])

    pieces_nz2 = full_nz.copy()
    pieces_nz2.clear()
    for k in range(ncats):
        pieces_nz2 += nz2[k]
    pieces_nz2.finalize(varz)
    t4 = time.time()
    print('time for zero-weight piece-wise processing = ',t4-t3)

    print('max error in meanr = ',np.max(pieces_nz2.meanr - full_nz.meanr),)
    print('    max meanr = ',np.max(full_nz.meanr))
    print('max error in meanlogr = ',np.max(pieces_nz2.meanlogr - full_nz.meanlogr),)
    print('    max meanlogr = ',np.max(full_nz.meanlogr))
    print('max error in weight = ',np.max(pieces_nz2.weight - full_nz.weight),)
    print('    max weight = ',np.max(full_nz.weight))
    print('max error in xi = ',np.max(pieces_nz2.xi - full_nz.xi),)
    print('    max xi = ',np.max(full_nz.xi))
    print('max error in xi_im = ',np.max(pieces_nz2.xi_im - full_nz.xi_im),)
    print('    max xi_im = ',np.max(full_nz.xi_im))
    print('max error in varxi = ',np.max(pieces_nz2.varxi - full_nz.varxi),)
    print('    max varxi = ',np.max(full_nz.varxi))
    np.testing.assert_allclose(pieces_nz2.meanr, full_nz.meanr, rtol=1.e-7)
    np.testing.assert_allclose(pieces_nz2.meanlogr, full_nz.meanlogr, rtol=1.e-7)
    np.testing.assert_allclose(pieces_nz2.weight, full_nz.weight, rtol=1.e-7)
    np.testing.assert_allclose(pieces_nz2.xi, full_nz.xi, rtol=1.e-7)
    np.testing.assert_allclose(pieces_nz2.xi_im, full_nz.xi_im, atol=1.e-10)
    np.testing.assert_allclose(pieces_nz2.varxi, full_nz.varxi, rtol=1.e-7)

    # Can also do this with initialize/finalize options
    pieces_nz3 = full_nz.copy()
    t3 = time.time()
    for k in range(ncats):
        pieces_nz3.process(lens_cat, source_cats2[k], initialize=(k==0), finalize=(k==ncats-1))
    t4 = time.time()
    print('time for initialize/finalize processing = ',t4-t3)

    np.testing.assert_allclose(pieces_nz3.meanr, full_nz.meanr, rtol=1.e-7)
    np.testing.assert_allclose(pieces_nz3.meanlogr, full_nz.meanlogr, rtol=1.e-7)
    np.testing.assert_allclose(pieces_nz3.weight, full_nz.weight, rtol=1.e-7)
    np.testing.assert_allclose(pieces_nz3.xi, full_nz.xi, rtol=1.e-7)
    np.testing.assert_allclose(pieces_nz3.xi_im, full_nz.xi_im, atol=1.e-10)
    np.testing.assert_allclose(pieces_nz3.varxi, full_nz.varxi, rtol=1.e-7)

    # Try this with corr2
    lens_cat.write(os.path.join('data','nz_wpos_lens.fits'))
    for i, sc in enumerate(source_cats2):
        sc.write(os.path.join('data','nz_wpos_source%d.fits'%i))
    config = treecorr.read_config('configs/nz_wpos.yaml')
    config['verbose'] = 0
    treecorr.corr2(config)
    data = fitsio.read(config['nz_file_name'])
    print('data.dtype = ',data.dtype)
    np.testing.assert_allclose(data['meanr'], pieces_nz3.meanr)
    np.testing.assert_allclose(data['meanlogr'], pieces_nz3.meanlogr)
    np.testing.assert_allclose(data['weight'], pieces_nz3.weight)
    np.testing.assert_allclose(data['z_real'], pieces_nz3.xi)
    np.testing.assert_allclose(data['z_imag'], pieces_nz3.xi_im)
    np.testing.assert_allclose(data['sigma']**2, pieces_nz3.varxi)


@timer
def test_varxi():
    # Test that varxi is correct (or close) based on actual variance of many runs.

    z0 = 0.05 + 1j*0.05
    r0 = 10.
    L = 10 * r0
    rng = np.random.RandomState(8675309)

    nsource = 1000
    nrand = 10
    nruns = 50000
    lens = treecorr.Catalog(x=[0], y=[0])

    file_name = 'data/test_varxi_nz.npz'
    print(file_name)
    if not os.path.isfile(file_name):
        all_nzs = []
        all_rzs = []
        for run in range(nruns):
            print(f'{run}/{nruns}')
            x2 = (rng.random_sample(nsource)-0.5) * L
            y2 = (rng.random_sample(nsource)-0.5) * L
            x3 = (rng.random_sample(nrand)-0.5) * L
            y3 = (rng.random_sample(nrand)-0.5) * L

            r2 = (x2**2 + y2**2)/r0**2
            zz = z0 * np.exp(-r2/2.) * (1.-r2/2)
            z1 = np.real(zz)
            z2 = np.imag(zz)
            # This time, add some shape noise (different each run).
            z1 += rng.normal(0, 0.1, size=nsource)
            z2 += rng.normal(0, 0.1, size=nsource)
            # Varied weights are hard, but at least check that non-unit weights work correctly.
            w = np.ones_like(x2) * 5

            source = treecorr.Catalog(x=x2, y=y2, w=w, z1=z1, z2=z2)
            rand = treecorr.Catalog(x=x3, y=y3)
            nz = treecorr.NZCorrelation(bin_size=0.3, min_sep=6., max_sep=15., angle_slop=0.3)
            rz = treecorr.NZCorrelation(bin_size=0.3, min_sep=6., max_sep=15., angle_slop=0.3)
            nz.process(lens, source)
            rz.process(rand, source)
            all_nzs.append(nz)
            all_rzs.append(rz)

        all_xis = [nz.calculateXi() for nz in all_nzs]
        var_xi_1 = np.var([xi[0] for xi in all_xis], axis=0)
        mean_varxi_1 = np.mean([xi[2] for xi in all_xis], axis=0)

        all_xis = [nz.calculateXi(rz=rz) for (nz,rz) in zip(all_nzs, all_rzs)]
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
    zz = z0 * np.exp(-r2/2.) * (1.-r2/2.)
    z1 = np.real(zz)
    z2 = np.imag(zz)
    z1 += rng.normal(0, 0.1, size=nsource)
    z2 += rng.normal(0, 0.1, size=nsource)
    w = np.ones_like(x2) * 5

    source = treecorr.Catalog(x=x2, y=y2, w=w, z1=z1, z2=z2)
    rand = treecorr.Catalog(x=x3, y=y3)
    nz = treecorr.NZCorrelation(bin_size=0.3, min_sep=6., max_sep=15., angle_slop=0.3)
    rz = treecorr.NZCorrelation(bin_size=0.3, min_sep=6., max_sep=15., angle_slop=0.3)
    nz.process(lens, source)
    rz.process(rand, source)

    print('single run:')
    print('Uncompensated')
    print('ratio = ',nz.varxi / var_xi_1)
    print('max relerr for xi = ',np.max(np.abs((nz.varxi - var_xi_1)/var_xi_1)))
    np.testing.assert_allclose(nz.varxi, var_xi_1, rtol=0.6)

    xi, xi_im, varxi = nz.calculateXi(rz=rz)
    print('Compensated')
    print('ratio = ',varxi / var_xi_2)
    print('max relerr for xi = ',np.max(np.abs((varxi - var_xi_2)/var_xi_2)))
    np.testing.assert_allclose(varxi, var_xi_2, rtol=0.6)

@timer
def test_jk():

    # Similar to the profile we use above, but multiple "lenses".
    z0 = 0.05 + 1j*0.05
    r0 = 30.
    L = 30 * r0
    rng = np.random.RandomState(8675309)

    nsource = 100000
    nrand = 1000
    nlens = 300
    nruns = 1000
    npatch = 64

    corr_params = dict(bin_size=0.3, min_sep=5, max_sep=30, bin_slop=0.1)

    def make_field(rng):
        x1 = (rng.random(nlens)-0.5) * L
        y1 = (rng.random(nlens)-0.5) * L
        w = rng.random(nlens) + 10
        x2 = (rng.random(nsource)-0.5) * L
        y2 = (rng.random(nsource)-0.5) * L
        x3 = (rng.random(nrand)-0.5) * L
        y3 = (rng.random(nrand)-0.5) * L

        # Start with just the noise
        z1 = rng.normal(0, 0.1, size=nsource)
        z2 = rng.normal(0, 0.1, size=nsource)

        # Add in the signal from all lenses
        for i in range(nlens):
            x2i = x2 - x1[i]
            y2i = y2 - y1[i]
            r2 = (x2i**2 + y2i**2)/r0**2
            zz = w[i] * z0 * np.exp(-r2/2.) * (1-r2/2)
            z1 += np.real(zz)
            z2 += np.imag(zz)
        return x1, y1, w, x2, y2, z1, z2, x3, y3

    file_name = 'data/test_nz_jk_{}.npz'.format(nruns)
    print(file_name)
    if not os.path.isfile(file_name):
        all_nzs = []
        all_rzs = []
        rng = np.random.default_rng()
        for run in range(nruns):
            x1, y1, w, x2, y2, z1, z2, x3, y3 = make_field(rng)
            print(run,': ',np.mean(z1),np.std(z1),np.min(z1),np.max(z1))
            cat1 = treecorr.Catalog(x=x1, y=y1, w=w)
            cat2 = treecorr.Catalog(x=x2, y=y2, z1=z1, z2=z2)
            cat3 = treecorr.Catalog(x=x3, y=y3)
            nz = treecorr.NZCorrelation(corr_params)
            rz = treecorr.NZCorrelation(corr_params)
            nz.process(cat1, cat2)
            rz.process(cat3, cat2)
            all_nzs.append(nz)
            all_rzs.append(rz)

        mean_xi = np.mean([nz.xi for nz in all_nzs], axis=0)
        var_xi = np.var([nz.xi for nz in all_nzs], axis=0)
        mean_varxi = np.mean([nz.varxi for nz in all_nzs], axis=0)

        for nz, rz in zip(all_nzs, all_rzs):
            nz.calculateXi(rz=rz)

        mean_xi_r = np.mean([nz.xi for nz in all_nzs], axis=0)
        var_xi_r = np.var([nz.xi for nz in all_nzs], axis=0)
        mean_varxi_r = np.mean([nz.varxi for nz in all_nzs], axis=0)

        np.savez(file_name,
                 mean_xi=mean_xi, var_xi=var_xi, mean_varxi=mean_varxi,
                 mean_xi_r=mean_xi_r, var_xi_r=var_xi_r, mean_varxi_r=mean_varxi_r)

    data = np.load(file_name)
    mean_xi = data['mean_xi']
    mean_varxi = data['mean_varxi']
    var_xi = data['var_xi']

    print('mean_xi = ',mean_xi)
    print('mean_varxi = ',mean_varxi)
    print('var_xi = ',var_xi)
    print('ratio = ',var_xi / mean_varxi)

    rng = np.random.default_rng(1234)
    x1, y1, w, x2, y2, z1, z2, x3, y3 = make_field(rng)

    cat1 = treecorr.Catalog(x=x1, y=y1, w=w)
    cat2 = treecorr.Catalog(x=x2, y=y2, z1=z1, z2=z2)
    nz1 = treecorr.NZCorrelation(corr_params)
    t0 = time.time()
    nz1.process(cat1, cat2)
    t1 = time.time()
    print('Time for non-patch processing = ',t1-t0)

    print('weight = ',nz1.weight)
    print('xi = ',nz1.xi)
    print('varxi = ',nz1.varxi)
    print('pullsq for xi = ',(nz1.xi-mean_xi)**2/var_xi)
    print('max pull for xi = ',np.sqrt(np.max((nz1.xi-mean_xi)**2/var_xi)))
    np.testing.assert_array_less((nz1.xi-mean_xi)**2, 9*var_xi)  # < 3 sigma pull
    np.testing.assert_allclose(nz1.varxi, mean_varxi, rtol=0.1)

    # Now run with patches, but still with shot variance.  Should be basically the same answer.
    try:
        import fitsio
        patch_dir = 'output'
        low_mem = True
    except ImportError:
        # If we cannot write to a fits file, skip the save_patch_dir tests.
        patch_dir = None
        low_mem = False
    cat2p = treecorr.Catalog(x=x2, y=y2, z1=z1, z2=z2, npatch=npatch, save_patch_dir=patch_dir)
    if low_mem:
        cat2p.write_patches()  # Force rewrite of any existing saved patches.
    cat1p = treecorr.Catalog(x=x1, y=y1, w=w, patch_centers=cat2p.patch_centers)
    print('tot w = ',np.sum(w))
    print('Patch\tNlens\tNsource')
    for i in range(npatch):
        print('%d\t%d\t%d'%(i,np.sum(cat1p.w[cat1p.patch==i]),np.sum(cat2p.w[cat2p.patch==i])))
    nz2 = treecorr.NZCorrelation(corr_params)
    t0 = time.time()
    nz2.process(cat1p, cat2p)
    t1 = time.time()
    print('Time for patch processing = ',t1-t0)
    print('weight = ',nz2.weight)
    print('xi = ',nz2.xi)
    print('xi1 = ',nz1.xi)
    print('varxi = ',nz2.varxi)
    print('ratio = ',nz2.xi/nz1.xi)
    np.testing.assert_allclose(nz2.weight, nz1.weight, rtol=1.e-2)
    np.testing.assert_allclose(nz2.xi, nz1.xi, rtol=1.e-2)
    np.testing.assert_allclose(nz2.varxi, nz1.varxi, rtol=1.e-2)

    # estimate_cov with var_method='shot' returns just the diagonal.
    np.testing.assert_allclose(nz2.estimate_cov('shot'), nz2.varxi)
    np.testing.assert_allclose(nz1.estimate_cov('shot'), nz1.varxi)

    # Now try jackknife variance estimate.
    t0 = time.time()
    cov2 = nz2.estimate_cov('jackknife')
    t1 = time.time()
    print('Time to calculate jackknife covariance = ',t1-t0)
    print('varxi = ',np.diagonal(cov2))
    print('cf var_xi = ',var_xi)
    np.testing.assert_allclose(np.diagonal(cov2), var_xi, rtol=0.5)

    # Check only using patches for one of the two catalogs.
    # Not as good as using patches for both, but not much worse.
    nz3 = treecorr.NZCorrelation(corr_params, var_method='jackknife')
    t0 = time.time()
    nz3.process(cat1p, cat2)
    t1 = time.time()
    print('Time for only patches for cat1 processing = ',t1-t0)
    print('varxi = ',nz3.varxi)
    np.testing.assert_allclose(nz3.weight, nz1.weight, rtol=1.e-2)
    np.testing.assert_allclose(nz3.xi, nz1.xi, rtol=1.e-2)
    np.testing.assert_allclose(nz3.varxi, var_xi, rtol=0.5)

    nz4 = treecorr.NZCorrelation(corr_params, var_method='jackknife', rng=rng)
    t0 = time.time()
    nz4.process(cat1, cat2p)
    t1 = time.time()
    print('Time for only patches for cat2 processing = ',t1-t0)
    print('varxi = ',nz4.varxi)
    np.testing.assert_allclose(nz4.weight, nz1.weight, rtol=1.e-2)
    np.testing.assert_allclose(nz4.xi, nz1.xi, rtol=1.e-2)
    np.testing.assert_allclose(nz4.varxi, var_xi, rtol=0.6)

    # Use initialize/finalize
    nz5 = treecorr.NZCorrelation(corr_params)
    for k1, p1 in enumerate(cat1p.get_patches()):
        for k2, p2 in enumerate(cat2p.get_patches()):
            nz5.process(p1, p2, initialize=(k1==k2==0), finalize=(k1==k2==npatch-1))
    np.testing.assert_allclose(nz5.xi, nz2.xi)
    np.testing.assert_allclose(nz5.weight, nz2.weight)
    np.testing.assert_allclose(nz5.varxi, nz2.varxi)

    # Check that these still work after roundtripping through a file.
    try:
        import fitsio
    except ImportError:
        pass
    else:
        file_name = os.path.join('output','test_write_results_nz.fits')
        nz2.write(file_name, write_patch_results=True)
        nz5 = treecorr.NZCorrelation.from_file(file_name)
        cov5 = nz5.estimate_cov('jackknife')
        np.testing.assert_allclose(cov5, cov2)

    # Use a random catalog
    mean_xi_r = data['mean_xi_r']
    mean_varxi_r = data['mean_varxi_r']
    var_xi_r = data['var_xi_r']

    print('mean_xi = ',mean_xi_r)
    print('mean_varxi = ',mean_varxi_r)
    print('var_xi = ',var_xi_r)
    print('ratio = ',var_xi_r / mean_varxi_r)

    cat3 = treecorr.Catalog(x=x3, y=y3)
    rz5 = treecorr.NZCorrelation(corr_params)
    rz5.process(cat3, cat2)
    nz5 = nz1.copy()
    nz5.calculateXi(rz=rz5)
    print('weight = ',nz5.weight)
    print('xi = ',nz5.xi)
    print('varxi = ',nz5.varxi)
    print('ratio = ',nz5.varxi / var_xi_r)
    print('pullsq for xi = ',(nz5.xi-mean_xi_r)**2/var_xi_r)
    print('max pull for xi = ',np.sqrt(np.max((nz5.xi-mean_xi_r)**2/var_xi_r)))
    np.testing.assert_array_less((nz5.xi-mean_xi_r)**2, 9*var_xi_r)  # < 3 sigma pull
    np.testing.assert_allclose(nz5.varxi, mean_varxi_r, rtol=0.1)

    # Repeat with patches
    cat3p = treecorr.Catalog(x=x3, y=y3, patch_centers=cat2p.patch_centers)
    rz6 = treecorr.NZCorrelation(corr_params)
    rz6.process(cat3p, cat2p, low_mem=low_mem)
    nz6 = nz2.copy()
    nz6.calculateXi(rz=rz6)
    cov6 = nz6.estimate_cov('jackknife')
    np.testing.assert_allclose(np.diagonal(cov6), var_xi_r, rtol=0.5)

    # Use a random catalog without patches.
    rz7 = treecorr.NZCorrelation(corr_params)
    rz7.process(cat3, cat2p)
    nz7 = nz4.copy()
    nz7.calculateXi(rz=rz7)
    cov7 = nz7.estimate_cov('jackknife')
    np.testing.assert_allclose(np.diagonal(cov7), var_xi_r, rtol=0.7)

    nz8 = nz2.copy()
    nz8.calculateXi(rz=rz7)
    cov8 = nz8.estimate_cov('jackknife')
    np.testing.assert_allclose(np.diagonal(cov8), var_xi_r, rtol=0.5)

    # Check some invalid actions
    # Bad var_method
    with assert_raises(ValueError):
        nz2.estimate_cov('invalid')
    # Not run on patches, but need patches
    with assert_raises(ValueError):
        nz1.estimate_cov('jackknife')
    with assert_raises(ValueError):
        nz1.estimate_cov('sample')
    with assert_raises(ValueError):
        nz1.estimate_cov('marked_bootstrap')
    with assert_raises(ValueError):
        nz1.estimate_cov('bootstrap')
    # rz also needs patches (at least for the g part).
    with assert_raises(RuntimeError):
        nz2.calculateXi(rz=nz1)

    cat1a = treecorr.Catalog(x=x1[:100], y=y1[:100], npatch=10)
    cat2a = treecorr.Catalog(x=x2[:100], y=y2[:100], z1=z1[:100], z2=z2[:100], npatch=10)
    cat1b = treecorr.Catalog(x=x1[:100], y=y1[:100], npatch=2)
    cat2b = treecorr.Catalog(x=x2[:100], y=y2[:100], z1=z1[:100], z2=z2[:100], npatch=2)
    nz6 = treecorr.NZCorrelation(corr_params)
    nz7 = treecorr.NZCorrelation(corr_params)
    # All catalogs need to have the same number of patches
    with assert_raises(RuntimeError):
        nz6.process(cat1a,cat2b)
    with assert_raises(RuntimeError):
        nz7.process(cat1b,cat2a)


if __name__ == '__main__':
    test_direct()
    test_direct_spherical()
    test_single()
    test_nz()
    test_pieces()
    test_varxi()
    test_jk()
