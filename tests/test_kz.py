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
import coord
import treecorr

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
    z12 = rng.normal(0,0.2, (ngal,) )
    z22 = rng.normal(0,0.2, (ngal,) )

    cat1 = treecorr.Catalog(x=x1, y=y1, w=w1, k=k1)
    cat2 = treecorr.Catalog(x=x2, y=y2, w=w2, z1=z12, z2=z22)

    min_sep = 1.
    max_sep = 50.
    nbins = 50
    bin_size = np.log(max_sep/min_sep) / nbins
    kz = treecorr.KZCorrelation(min_sep=min_sep, max_sep=max_sep, nbins=nbins, brute=True)
    kz.process(cat1, cat2)

    true_npairs = np.zeros(nbins, dtype=int)
    true_weight = np.zeros(nbins, dtype=float)
    true_xi = np.zeros(nbins, dtype=complex)
    for i in range(ngal):
        # It's hard to do all the pairs at once with numpy operations (although maybe possible).
        # But we can at least do all the pairs for each entry in cat1 at once with arrays.
        rsq = (x1[i]-x2)**2 + (y1[i]-y2)**2
        r = np.sqrt(rsq)

        ww = w1[i] * w2
        xi = ww * k1[i] * (z12 + 1j*z22)

        index = np.floor(np.log(r/min_sep) / bin_size).astype(int)
        mask = (index >= 0) & (index < nbins)
        np.add.at(true_npairs, index[mask], 1)
        np.add.at(true_weight, index[mask], ww[mask])
        np.add.at(true_xi, index[mask], xi[mask])

    true_xi /= true_weight

    print('true_npairs = ',true_npairs)
    print('diff = ',kz.npairs - true_npairs)
    np.testing.assert_array_equal(kz.npairs, true_npairs)

    print('true_weight = ',true_weight)
    print('diff = ',kz.weight - true_weight)
    np.testing.assert_allclose(kz.weight, true_weight, rtol=1.e-6, atol=1.e-8)

    print('true_xi = ',true_xi)
    print('kz.xi = ',kz.xi)
    print('kz.xi_im = ',kz.xi_im)
    np.testing.assert_allclose(kz.xi, true_xi.real, rtol=1.e-6, atol=1.e-8)
    np.testing.assert_allclose(kz.xi_im, true_xi.imag, rtol=1.e-6, atol=1.e-8)

    # Check that running via the corr2 script works correctly.
    config = treecorr.config.read_config('configs/kz_direct.yaml')
    try:
        import fitsio
    except ImportError:
        pass
    else:
        cat1.write(config['file_name'])
        cat2.write(config['file_name2'])
        treecorr.corr2(config)
        data = fitsio.read(config['kz_file_name'])
        np.testing.assert_allclose(data['r_nom'], kz.rnom)
        np.testing.assert_allclose(data['npairs'], kz.npairs)
        np.testing.assert_allclose(data['weight'], kz.weight)
        np.testing.assert_allclose(data['xi'], kz.xi)
        np.testing.assert_allclose(data['xi_im'], kz.xi_im)

        # Invalid with only one file_name
        del config['file_name2']
        with assert_raises(TypeError):
            treecorr.corr2(config)

    # Repeat with binslop = 0, since code is different for bin_slop=0 and brute=True.
    # And don't do any top-level recursion so we actually test not going to the leaves.
    kz = treecorr.KZCorrelation(min_sep=min_sep, max_sep=max_sep, nbins=nbins, bin_slop=0,
                                max_top=0)
    kz.process(cat1, cat2)
    np.testing.assert_array_equal(kz.npairs, true_npairs)
    np.testing.assert_allclose(kz.weight, true_weight, rtol=1.e-6, atol=1.e-8)
    np.testing.assert_allclose(kz.xi, true_xi.real, atol=1.e-3)
    np.testing.assert_allclose(kz.xi_im, true_xi.imag, atol=1.e-3)

    # With angle_slop = 0, it goes back to being basically exact (to single precision).
    kz = treecorr.KZCorrelation(min_sep=min_sep, max_sep=max_sep, nbins=nbins, bin_slop=0,
                                angle_slop=0, max_top=0)
    kz.process(cat1, cat2)
    np.testing.assert_array_equal(kz.npairs, true_npairs)
    np.testing.assert_allclose(kz.weight, true_weight, rtol=1.e-6, atol=1.e-8)
    np.testing.assert_allclose(kz.xi, true_xi.real, rtol=1.e-6, atol=1.e-8)
    np.testing.assert_allclose(kz.xi_im, true_xi.imag, rtol=1.e-6, atol=1.e-8)

    # Check a few basic operations with a KZCorrelation object.
    do_pickle(kz)

    kz2 = kz.copy()
    kz2 += kz
    np.testing.assert_allclose(kz2.npairs, 2*kz.npairs)
    np.testing.assert_allclose(kz2.weight, 2*kz.weight)
    np.testing.assert_allclose(kz2.meanr, 2*kz.meanr)
    np.testing.assert_allclose(kz2.meanlogr, 2*kz.meanlogr)
    np.testing.assert_allclose(kz2.xi, 2*kz.xi)
    np.testing.assert_allclose(kz2.xi_im, 2*kz.xi_im)

    kz2.clear()
    kz2 += kz
    np.testing.assert_allclose(kz2.npairs, kz.npairs)
    np.testing.assert_allclose(kz2.weight, kz.weight)
    np.testing.assert_allclose(kz2.meanr, kz.meanr)
    np.testing.assert_allclose(kz2.meanlogr, kz.meanlogr)
    np.testing.assert_allclose(kz2.xi, kz.xi)
    np.testing.assert_allclose(kz2.xi_im, kz.xi_im)

    ascii_name = 'output/kz_ascii.txt'
    kz.write(ascii_name, precision=16)
    kz3 = treecorr.KZCorrelation(min_sep=min_sep, max_sep=max_sep, nbins=nbins, bin_type='Log')
    kz3.read(ascii_name)
    np.testing.assert_allclose(kz3.npairs, kz.npairs)
    np.testing.assert_allclose(kz3.weight, kz.weight)
    np.testing.assert_allclose(kz3.meanr, kz.meanr)
    np.testing.assert_allclose(kz3.meanlogr, kz.meanlogr)
    np.testing.assert_allclose(kz3.xi, kz.xi)
    np.testing.assert_allclose(kz3.xi_im, kz.xi_im)

    # Check that the repr is minimal
    assert repr(kz3) == f'KZCorrelation(min_sep={min_sep}, max_sep={max_sep}, nbins={nbins})'

    # Simpler API using from_file:
    with CaptureLog() as cl:
        kz3b = treecorr.KZCorrelation.from_file(ascii_name, logger=cl.logger)
    assert ascii_name in cl.output
    np.testing.assert_allclose(kz3b.npairs, kz.npairs)
    np.testing.assert_allclose(kz3b.weight, kz.weight)
    np.testing.assert_allclose(kz3b.meanr, kz.meanr)
    np.testing.assert_allclose(kz3b.meanlogr, kz.meanlogr)
    np.testing.assert_allclose(kz3b.xi, kz.xi)
    np.testing.assert_allclose(kz3b.xi_im, kz.xi_im)

    # or using the Corr2 base class
    with CaptureLog() as cl:
        kz3c = treecorr.Corr2.from_file(ascii_name, logger=cl.logger)
    assert ascii_name in cl.output
    np.testing.assert_allclose(kz3c.npairs, kz.npairs)
    np.testing.assert_allclose(kz3c.weight, kz.weight)
    np.testing.assert_allclose(kz3c.meanr, kz.meanr)
    np.testing.assert_allclose(kz3c.meanlogr, kz.meanlogr)
    np.testing.assert_allclose(kz3c.xi, kz.xi)
    np.testing.assert_allclose(kz3c.xi_im, kz.xi_im)

    try:
        import fitsio
    except ImportError:
        pass
    else:
        fits_name = 'output/kz_fits.fits'
        kz.write(fits_name)
        kz4 = treecorr.KZCorrelation(min_sep=min_sep, max_sep=max_sep, nbins=nbins)
        kz4.read(fits_name)
        np.testing.assert_allclose(kz4.npairs, kz.npairs)
        np.testing.assert_allclose(kz4.weight, kz.weight)
        np.testing.assert_allclose(kz4.meanr, kz.meanr)
        np.testing.assert_allclose(kz4.meanlogr, kz.meanlogr)
        np.testing.assert_allclose(kz4.xi, kz.xi)
        np.testing.assert_allclose(kz4.xi_im, kz.xi_im)

        kz4b = treecorr.KZCorrelation.from_file(fits_name)
        np.testing.assert_allclose(kz4b.npairs, kz.npairs)
        np.testing.assert_allclose(kz4b.weight, kz.weight)
        np.testing.assert_allclose(kz4b.meanr, kz.meanr)
        np.testing.assert_allclose(kz4b.meanlogr, kz.meanlogr)
        np.testing.assert_allclose(kz4b.xi, kz.xi)
        np.testing.assert_allclose(kz4b.xi_im, kz.xi_im)

    with assert_raises(TypeError):
        kz2 += config
    kz4 = treecorr.KZCorrelation(min_sep=min_sep/2, max_sep=max_sep, nbins=nbins)
    with assert_raises(ValueError):
        kz2 += kz4
    kz5 = treecorr.KZCorrelation(min_sep=min_sep, max_sep=max_sep*2, nbins=nbins)
    with assert_raises(ValueError):
        kz2 += kz5
    kz6 = treecorr.KZCorrelation(min_sep=min_sep, max_sep=max_sep, nbins=nbins*2)
    with assert_raises(ValueError):
        kz2 += kz6
    with assert_raises(ValueError):
        kz.process(cat1, cat2, patch_method='nonlocal')

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
    z12 = rng.normal(0,0.2, (ngal,) )
    z22 = rng.normal(0,0.2, (ngal,) )

    ra1, dec1 = coord.CelestialCoord.xyz_to_radec(x1,y1,z1)
    ra2, dec2 = coord.CelestialCoord.xyz_to_radec(x2,y2,z2)

    cat1 = treecorr.Catalog(ra=ra1, dec=dec1, ra_units='rad', dec_units='rad', w=w1, k=k1)
    cat2 = treecorr.Catalog(ra=ra2, dec=dec2, ra_units='rad', dec_units='rad', w=w2, z1=z12, z2=z22)

    min_sep = 1.
    max_sep = 10.
    nbins = 50
    bin_size = np.log(max_sep/min_sep) / nbins
    kz = treecorr.KZCorrelation(min_sep=min_sep, max_sep=max_sep, nbins=nbins,
                                sep_units='deg', brute=True)
    kz.process(cat1, cat2)

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
            xi = ww * k1[i] * (z12[j] + 1j * z22[j])

            true_npairs[index] += 1
            true_weight[index] += ww
            true_xi[index] += xi

    true_xi /= true_weight

    print('true_npairs = ',true_npairs)
    print('diff = ',kz.npairs - true_npairs)
    np.testing.assert_array_equal(kz.npairs, true_npairs)

    print('true_weight = ',true_weight)
    print('diff = ',kz.weight - true_weight)
    np.testing.assert_allclose(kz.weight, true_weight, rtol=1.e-6, atol=1.e-8)

    print('true_xi = ',true_xi)
    print('kz.xi = ',kz.xi)
    np.testing.assert_allclose(kz.xi, true_xi.real, rtol=1.e-6, atol=1.e-8)
    np.testing.assert_allclose(kz.xi_im, true_xi.imag, rtol=1.e-6, atol=1.e-8)

    # Check that running via the corr2 script works correctly.
    config = treecorr.config.read_config('configs/kz_direct_spherical.yaml')
    try:
        import fitsio
    except ImportError:
        pass
    else:
        cat1.write(config['file_name'])
        cat2.write(config['file_name2'])
        treecorr.corr2(config)
        data = fitsio.read(config['kz_file_name'])
        np.testing.assert_allclose(data['r_nom'], kz.rnom)
        np.testing.assert_allclose(data['npairs'], kz.npairs)
        np.testing.assert_allclose(data['weight'], kz.weight)
        np.testing.assert_allclose(data['xi'], kz.xi)
        np.testing.assert_allclose(data['xi_im'], kz.xi_im)

    # Repeat with binslop = 0
    # And don't do any top-level recursion so we actually test not going to the leaves.
    kz = treecorr.KZCorrelation(min_sep=min_sep, max_sep=max_sep, nbins=nbins,
                                sep_units='deg', bin_slop=0, max_top=0)
    kz.process(cat1, cat2)
    np.testing.assert_array_equal(kz.npairs, true_npairs)
    np.testing.assert_allclose(kz.weight, true_weight, rtol=1.e-6, atol=1.e-8)
    np.testing.assert_allclose(kz.xi, true_xi.real, atol=1.e-3)
    np.testing.assert_allclose(kz.xi_im, true_xi.imag, atol=1.e-3)

    # With angle_slop = 0, it goes back to being basically exact (to single precision).
    kz = treecorr.KZCorrelation(min_sep=min_sep, max_sep=max_sep, nbins=nbins,
                                sep_units='deg', bin_slop=0, angle_slop=0, max_top=0)
    kz.process(cat1, cat2)
    np.testing.assert_array_equal(kz.npairs, true_npairs)
    np.testing.assert_allclose(kz.weight, true_weight, rtol=1.e-6, atol=1.e-8)
    np.testing.assert_allclose(kz.xi, true_xi.real, rtol=1.e-6, atol=1.e-8)
    np.testing.assert_allclose(kz.xi_im, true_xi.imag, rtol=1.e-6, atol=1.e-8)


@timer
def test_single():
    # Use z(r) = z0 exp(-r^2/2r0^2) (1-r^2/wr0^2) around a single lens

    nsource = 100000
    z0 = 0.05 + 1j * 0.02
    kappa = 0.23
    r0 = 10.
    L = 5. * r0
    rng = np.random.RandomState(8675309)
    x = (rng.random_sample(nsource)-0.5) * L
    y = (rng.random_sample(nsource)-0.5) * L
    r2 = (x**2 + y**2)
    r = np.sqrt(r2)
    z = z0 * np.exp(-0.5*r2/r0**2) * (1-0.5*r2/r0**2)
    z1 = np.real(z)
    z2 = np.imag(z)

    lens_cat = treecorr.Catalog(x=[0], y=[0], k=[kappa],  x_units='arcmin', y_units='arcmin')
    source_cat = treecorr.Catalog(x=x, y=y, z1=z1, z2=z2, x_units='arcmin', y_units='arcmin')
    kz = treecorr.KZCorrelation(bin_size=0.1, min_sep=1., max_sep=20., sep_units='arcmin',
                                verbose=1)
    kz.process(lens_cat, source_cat)

    # log(<R>) != <logR>, but it should be close:
    print('meanlogr - log(meanr) = ',kz.meanlogr - np.log(kz.meanr))
    np.testing.assert_allclose(kz.meanlogr, np.log(kz.meanr), atol=1.e-3)

    r = kz.meanr
    true_kz = kappa * z0 * np.exp(-0.5*r**2/r0**2) * (1-0.5*r**2/r0**2)

    print('kz.xi = ',kz.xi)
    print('kz.xi_im = ',kz.xi_im)
    print('true_kz = ',true_kz)
    print('ratio = ',kz.xi / true_kz)
    print('diff = ',kz.xi - true_kz)
    print('max diff = ',max(abs(kz.xi - true_kz)))
    np.testing.assert_allclose(kz.xi, np.real(true_kz), rtol=1.e-2, atol=1.e-4)
    np.testing.assert_allclose(kz.xi_im, np.imag(true_kz), rtol=1.e-2, atol=1.e-4)

    # Check that we get the same result using the corr2 function:
    lens_cat.write(os.path.join('data','kz_single_lens.dat'))
    source_cat.write(os.path.join('data','kz_single_source.dat'))
    config = treecorr.read_config('configs/kz_single.yaml')
    config['verbose'] = 0
    treecorr.corr2(config)
    corr2_output = np.genfromtxt(os.path.join('output','kz_single.out'), names=True,
                                 skip_header=1)
    print('kz.xi = ',kz.xi)
    print('from corr2 output = ',corr2_output['xi'])
    print('ratio = ',corr2_output['xi']/kz.xi)
    print('diff = ',corr2_output['xi']-kz.xi)
    np.testing.assert_allclose(corr2_output['xi'], kz.xi, rtol=1.e-4)

    print('xi_im from corr2 output = ',corr2_output['xi_im'])
    np.testing.assert_allclose(corr2_output['xi_im'], kz.xi_im, rtol=1.e-4)


@timer
def test_kz():
    # Use z(r) = z0 exp(-r^2/2r0^2) (1-r^2/2r0^2) around a bunch of foreground lenses.
    # For each lens, we divide this by a random kappa value assigned to that lens, so
    # the final kz output shoudl be just z(r).

    nlens = 1000
    nsource = 50000
    r0 = 10.
    L = 100. * r0

    z0 = 0.05 + 1j*0.02
    rng = np.random.RandomState(8675309)
    xl = (rng.random_sample(nlens)-0.5) * L
    yl = (rng.random_sample(nlens)-0.5) * L
    kl = rng.normal(0.23, 0.05, (nlens,) )
    xs = (rng.random_sample(nsource)-0.5) * L
    ys = (rng.random_sample(nsource)-0.5) * L
    z1 = np.zeros( (nsource,) )
    z2 = np.zeros( (nsource,) )
    for x,y,k in zip(xl,yl,kl):
        dx = xs-x
        dy = ys-y
        r2 = dx**2 + dy**2
        r = np.sqrt(r2)
        zz = z0 * np.exp(-0.5*r2/r0**2) * (1-0.5*r2/r0**2) / k
        z1 += np.real(zz)
        z2 += np.imag(zz)

    lens_cat = treecorr.Catalog(x=xl, y=yl, k=kl, x_units='arcmin', y_units='arcmin')
    source_cat = treecorr.Catalog(x=xs, y=ys, z1=z1, z2=z2, x_units='arcmin', y_units='arcmin')
    kz = treecorr.KZCorrelation(bin_size=0.1, min_sep=1., max_sep=20., sep_units='arcmin',
                                verbose=1)
    kz.process(lens_cat, source_cat, num_threads=1)

    # Using nbins=None rather than omiting nbins is equivalent.
    kz2 = treecorr.KZCorrelation(bin_size=0.1, min_sep=1., max_sep=20., nbins=None, sep_units='arcmin')
    kz2.process(lens_cat, source_cat, num_threads=1)
    assert kz2 == kz  # (Only exact == if num_threads=1.)

    r = kz.meanr
    true_kz = z0 * np.exp(-0.5*r**2/r0**2) * (1-0.5*r**2/r0**2)

    print('kz.xi = ',kz.xi)
    print('kz.xi_im = ',kz.xi_im)
    print('true_kz = ',true_kz)
    print('ratio = ',kz.xi / true_kz)
    print('diff = ',kz.xi - true_kz)
    print('max diff = ',max(abs(kz.xi - true_kz)))
    np.testing.assert_allclose(kz.xi, np.real(true_kz), rtol=0.1, atol=2.e-3)
    np.testing.assert_allclose(kz.xi_im, np.imag(true_kz), rtol=0.1, atol=2.e-3)

    # Check that we get the same result using the corr2 function:
    lens_cat.write(os.path.join('data','kz_lens.dat'))
    source_cat.write(os.path.join('data','kz_source.dat'))
    config = treecorr.read_config('configs/kz.yaml')
    config['verbose'] = 0
    config['precision'] = 8
    treecorr.corr2(config)
    corr2_output = np.genfromtxt(os.path.join('output','kz.out'), names=True, skip_header=1)
    print('kz.xi = ',kz.xi)
    print('from corr2 output = ',corr2_output['xi'])
    print('ratio = ',corr2_output['xi']/kz.xi)
    print('diff = ',corr2_output['xi']-kz.xi)
    np.testing.assert_allclose(corr2_output['xi'], kz.xi, rtol=1.e-4)

    print('xi_im from corr2 output = ',corr2_output['xi_im'])
    np.testing.assert_allclose(corr2_output['xi_im'], kz.xi_im, rtol=1.e-4)

    # Check the fits write option
    try:
        import fitsio
    except ImportError:
        pass
    else:
        out_file_name1 = os.path.join('output','kg_out1.fits')
        kz.write(out_file_name1)
        data = fitsio.read(out_file_name1)
        np.testing.assert_almost_equal(data['r_nom'], np.exp(kz.logr))
        np.testing.assert_almost_equal(data['meanr'], kz.meanr)
        np.testing.assert_almost_equal(data['meanlogr'], kz.meanlogr)
        np.testing.assert_almost_equal(data['xi'], kz.xi)
        np.testing.assert_almost_equal(data['xi_im'], kz.xi_im)
        np.testing.assert_almost_equal(data['sigma'], np.sqrt(kz.varxi))
        np.testing.assert_almost_equal(data['weight'], kz.weight)
        np.testing.assert_almost_equal(data['npairs'], kz.npairs)

        # Check the read function
        kz2 = treecorr.KZCorrelation.from_file(out_file_name1)
        np.testing.assert_almost_equal(kz2.logr, kz.logr)
        np.testing.assert_almost_equal(kz2.meanr, kz.meanr)
        np.testing.assert_almost_equal(kz2.meanlogr, kz.meanlogr)
        np.testing.assert_almost_equal(kz2.xi, kz.xi)
        np.testing.assert_almost_equal(kz2.xi_im, kz.xi_im)
        np.testing.assert_almost_equal(kz2.varxi, kz.varxi)
        np.testing.assert_almost_equal(kz2.weight, kz.weight)
        np.testing.assert_almost_equal(kz2.npairs, kz.npairs)
        assert kz2.coords == kz.coords
        assert kz2.metric == kz.metric
        assert kz2.sep_units == kz.sep_units
        assert kz2.bin_type == kz.bin_type


@timer
def test_varxi():
    # Test that varxi is correct (or close) based on actual variance of many runs.

    z0 = 0.05 + 1j*0.05
    kappa0 = 0.03
    r0 = 10.
    L = 50.*r0
    rng = np.random.RandomState(8675309)

    # Note: to get a good estimate of var(xi), you need a lot of runs.  The number of
    # runs matters much more than the number of galaxies for getting this to pass.
    ngal = 1000
    nruns = 50000

    file_name = 'data/test_varxi_kz.npz'
    print(file_name)
    if not os.path.isfile(file_name):
        all_kzs = []
        for run in range(nruns):
            print(f'{run}/{nruns}')
            x = (rng.random_sample(ngal)-0.5) * L
            y = (rng.random_sample(ngal)-0.5) * L
            # Varied weights are hard, but at least check that non-unit weights work correctly.
            w = np.ones_like(x) * 5
            r2 = (x**2 + y**2)/r0**2
            zz = z0 * np.exp(-r2/2.) * (1-r2/2)
            z1 = np.real(zz)
            z2 = np.imag(zz)
            k = kappa0 * np.exp(-r2/2.)
            # This time, add some shape noise (different each run).
            z1 += rng.normal(0, 0.3, size=ngal)
            z2 += rng.normal(0, 0.3, size=ngal)
            k += rng.normal(0, 0.1, size=ngal)

            cat = treecorr.Catalog(x=x, y=y, w=w, z1=z1, z2=z2, k=k)
            kz = treecorr.KZCorrelation(bin_size=0.1, min_sep=5., max_sep=50.)
            kz.process(cat, cat)
            all_kzs.append(kz)

        mean_xi = np.mean([kz.xi for kz in all_kzs], axis=0)
        var_xi = np.var([kz.xi for kz in all_kzs], axis=0)
        mean_varxi = np.mean([kz.varxi for kz in all_kzs], axis=0)

        np.savez(file_name,
                 mean_xi=mean_xi, var_xi=var_xi, mean_varxi=mean_varxi)

    data = np.load(file_name)
    mean_xi = data['mean_xi']
    mean_varxi = data['mean_varxi']
    var_xi = data['var_xi']
    print('nruns = ',nruns)
    print('mean_xi = ',mean_xi)
    print('mean_varxi = ',mean_varxi)
    print('var_xi = ',var_xi)
    print('ratio = ',var_xi / mean_varxi)
    print('max relerr for xi = ',np.max(np.abs((var_xi - mean_varxi)/var_xi)))
    np.testing.assert_allclose(mean_varxi, var_xi, rtol=0.02)

    # Now the actual test that's based on current code, not just from the saved file.
    # There is a bit more noise on a singe run, so the tolerance needs to be somewhat higher.
    x = (rng.random_sample(ngal)-0.5) * L
    y = (rng.random_sample(ngal)-0.5) * L
    # Varied weights are hard, but at least check that non-unit weights work correctly.
    w = np.ones_like(x) * 5
    r2 = (x**2 + y**2)/r0**2
    zz = z0 * np.exp(-r2/2.) * (1-r2/2)
    z1 = np.real(zz)
    z2 = np.imag(zz)
    k = kappa0 * np.exp(-r2/2.)
    # This time, add some shape noise (different each run).
    z1 += rng.normal(0, 0.3, size=ngal)
    z2 += rng.normal(0, 0.3, size=ngal)
    k += rng.normal(0, 0.1, size=ngal)

    cat = treecorr.Catalog(x=x, y=y, w=w, z1=z1, z2=z2, k=k)
    kz = treecorr.KZCorrelation(bin_size=0.1, min_sep=5., max_sep=50.)
    kz.process(cat, cat)

    print('single run:')
    print('ratio = ',kz.varxi / var_xi)
    print('max relerr for xi = ',np.max(np.abs((kz.varxi - var_xi)/var_xi)))
    np.testing.assert_allclose(kz.varxi, var_xi, rtol=0.5)

@timer
def test_jk():

    # Same multi-lens field we used for NZ patch test
    z0 = 0.05 + 1j*0.03
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
        k = rng.random(nlens)*3 + 10
        x2 = (rng.random(nsource)-0.5) * L
        y2 = (rng.random(nsource)-0.5) * L

        # Start with just the noise
        z1 = rng.normal(0, 0.1, size=nsource)
        z2 = rng.normal(0, 0.1, size=nsource)

        # Add in the signal from all lenses
        for i in range(nlens):
            x2i = x2 - x1[i]
            y2i = y2 - y1[i]
            r2 = (x2i**2 + y2i**2)/r0**2
            zz = z0 * np.exp(-r2/2.) * (1-r2/2)
            z1 += np.real(zz)
            z2 += np.imag(zz)
        return x1, y1, k, x2, y2, z1, z2

    file_name = 'data/test_kz_jk_{}.npz'.format(nruns)
    print(file_name)
    if not os.path.isfile(file_name):
        all_kzs = []
        rng = np.random.default_rng()
        for run in range(nruns):
            x1, y1, k, x2, y2, z1, z2 = make_field(rng)
            print(run,': ',np.mean(z1),np.std(z1),np.min(z1),np.max(z1))
            cat1 = treecorr.Catalog(x=x1, y=y1, k=k)
            cat2 = treecorr.Catalog(x=x2, y=y2, z1=z1, z2=z2)
            kz = treecorr.KZCorrelation(corr_params)
            kz.process(cat1, cat2)
            all_kzs.append(kz)

        mean_xi = np.mean([kz.xi for kz in all_kzs], axis=0)
        var_xi = np.var([kz.xi for kz in all_kzs], axis=0)
        mean_varxi = np.mean([kz.varxi for kz in all_kzs], axis=0)

        np.savez(file_name,
                 mean_xi=mean_xi, var_xi=var_xi, mean_varxi=mean_varxi)

    data = np.load(file_name)
    mean_xi = data['mean_xi']
    mean_varxi = data['mean_varxi']
    var_xi = data['var_xi']

    print('mean_xi = ',mean_xi)
    print('mean_varxi = ',mean_varxi)
    print('var_xi = ',var_xi)
    print('ratio = ',var_xi / mean_varxi)

    rng = np.random.default_rng(1234)
    x1, y1, k, x2, y2, z1, z2 = make_field(rng)

    cat1 = treecorr.Catalog(x=x1, y=y1, k=k)
    cat2 = treecorr.Catalog(x=x2, y=y2, z1=z1, z2=z2)
    kz1 = treecorr.KZCorrelation(corr_params)
    t0 = time.time()
    kz1.process(cat1, cat2)
    t1 = time.time()
    print('Time for non-patch processing = ',t1-t0)

    print('weight = ',kz1.weight)
    print('xi = ',kz1.xi)
    print('varxi = ',kz1.varxi)
    print('pullsq for xi = ',(kz1.xi-mean_xi)**2/var_xi)
    print('max pull for xi = ',np.sqrt(np.max((kz1.xi-mean_xi)**2/var_xi)))
    np.testing.assert_array_less((kz1.xi-mean_xi)**2, 9*var_xi)  # < 3 sigma pull
    np.testing.assert_allclose(kz1.varxi, mean_varxi, rtol=0.1)

    # Now run with patches, but still with shot variance.  Should be basically the same answer.
    cat2p = treecorr.Catalog(x=x2, y=y2, z1=z1, z2=z2, npatch=npatch)
    cat1p = treecorr.Catalog(x=x1, y=y1, k=k, patch_centers=cat2p.patch_centers)
    kz2 = treecorr.KZCorrelation(corr_params)
    t0 = time.time()
    kz2.process(cat1p, cat2p)
    t1 = time.time()
    print('Time for patch processing = ',t1-t0)
    print('weight = ',kz2.weight)
    print('xi = ',kz2.xi)
    print('xi1 = ',kz1.xi)
    print('varxi = ',kz2.varxi)
    np.testing.assert_allclose(kz2.weight, kz1.weight, rtol=1.e-2)
    np.testing.assert_allclose(kz2.xi, kz1.xi, rtol=2.e-2)
    np.testing.assert_allclose(kz2.varxi, kz1.varxi, rtol=1.e-2)

    # estimate_cov with var_method='shot' returns just the diagonal.
    np.testing.assert_allclose(kz2.estimate_cov('shot'), kz2.varxi)
    np.testing.assert_allclose(kz1.estimate_cov('shot'), kz1.varxi)

    # Now try jackknife variance estimate.
    t0 = time.time()
    cov2 = kz2.estimate_cov('jackknife')
    t1 = time.time()
    print('Time to calculate jackknife covariance = ',t1-t0)
    print('varxi = ',np.diagonal(cov2))
    print('cf var_xi = ',var_xi)
    np.testing.assert_allclose(np.diagonal(cov2), var_xi, rtol=0.6)

    # Check only using patches for one of the two catalogs.
    kz3 = treecorr.KZCorrelation(corr_params, var_method='jackknife')
    t0 = time.time()
    kz3.process(cat1p, cat2)
    t1 = time.time()
    print('Time for only patches for cat1 processing = ',t1-t0)
    print('varxi = ',kz3.varxi)
    np.testing.assert_allclose(kz3.weight, kz1.weight, rtol=1.e-2)
    np.testing.assert_allclose(kz3.xi, kz1.xi, rtol=1.e-2)
    np.testing.assert_allclose(kz3.varxi, var_xi, rtol=0.5)

    kz4 = treecorr.KZCorrelation(corr_params, var_method='jackknife', rng=rng)
    t0 = time.time()
    kz4.process(cat1, cat2p)
    t1 = time.time()
    print('Time for only patches for cat2 processing = ',t1-t0)
    print('varxi = ',kz4.varxi)
    np.testing.assert_allclose(kz4.weight, kz1.weight, rtol=1.e-2)
    np.testing.assert_allclose(kz4.xi, kz1.xi, rtol=2.e-2)
    np.testing.assert_allclose(kz4.varxi, var_xi, rtol=0.9)

    # Use initialize/finalize
    kz5 = treecorr.KZCorrelation(corr_params)
    for k1, p1 in enumerate(cat1p.get_patches()):
        for k2, p2 in enumerate(cat2p.get_patches()):
            kz5.process(p1, p2, initialize=(k1==k2==0), finalize=(k1==k2==npatch-1))
    np.testing.assert_allclose(kz5.xi, kz2.xi)
    np.testing.assert_allclose(kz5.weight, kz2.weight)
    np.testing.assert_allclose(kz5.varxi, kz2.varxi)

    # Check that these still work after roundtripping through a file.
    try:
        import fitsio
    except ImportError:
        pass
    else:
        file_name = os.path.join('output','test_write_results_kz.fits')
        kz2.write(file_name, write_patch_results=True)
        kz5 = treecorr.KZCorrelation.from_file(file_name)
        cov5 = kz5.estimate_cov('jackknife')
        np.testing.assert_allclose(cov5, cov2)

    # Check some invalid actions
    # Bad var_method
    with assert_raises(ValueError):
        kz2.estimate_cov('invalid')
    # Not run on patches, but need patches
    with assert_raises(ValueError):
        kz1.estimate_cov('jackknife')
    with assert_raises(ValueError):
        kz1.estimate_cov('sample')
    with assert_raises(ValueError):
        kz1.estimate_cov('marked_bootstrap')
    with assert_raises(ValueError):
        kz1.estimate_cov('bootstrap')

    cat1a = treecorr.Catalog(x=x1[:100], y=y1[:100], npatch=10)
    cat2a = treecorr.Catalog(x=x2[:100], y=y2[:100], z1=z1[:100], z2=z2[:100], npatch=10)
    cat1b = treecorr.Catalog(x=x1[:100], y=y1[:100], npatch=2)
    cat2b = treecorr.Catalog(x=x2[:100], y=y2[:100], z1=z1[:100], z2=z2[:100], npatch=2)
    kz6 = treecorr.KZCorrelation(corr_params)
    kz7 = treecorr.KZCorrelation(corr_params)
    # All catalogs need to have the same number of patches
    with assert_raises(RuntimeError):
        kz6.process(cat1a,cat2b)
    with assert_raises(RuntimeError):
        kz7.process(cat1b,cat2a)


if __name__ == '__main__':
    test_direct()
    test_direct_spherical()
    test_single()
    test_kz()
    test_varxi()
    test_jk()
