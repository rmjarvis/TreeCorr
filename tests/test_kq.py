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
    q12 = rng.normal(0,0.2, (ngal,) )
    q22 = rng.normal(0,0.2, (ngal,) )

    cat1 = treecorr.Catalog(x=x1, y=y1, w=w1, k=k1)
    cat2 = treecorr.Catalog(x=x2, y=y2, w=w2, q1=q12, q2=q22)

    min_sep = 1.
    max_sep = 50.
    nbins = 50
    bin_size = np.log(max_sep/min_sep) / nbins
    kq = treecorr.KQCorrelation(min_sep=min_sep, max_sep=max_sep, nbins=nbins, brute=True)
    kq.process(cat1, cat2)

    true_npairs = np.zeros(nbins, dtype=int)
    true_weight = np.zeros(nbins, dtype=float)
    true_xi = np.zeros(nbins, dtype=complex)
    for i in range(ngal):
        # It's hard to do all the pairs at once with numpy operations (although maybe possible).
        # But we can at least do all the pairs for each entry in cat1 at once with arrays.
        rsq = (x1[i]-x2)**2 + (y1[i]-y2)**2
        r = np.sqrt(rsq)
        expmialpha = ((x2-x1[i]) - 1j*(y2-y1[i])) / r

        ww = w1[i] * w2
        xi = ww * k1[i] * (q12 + 1j*q22) * expmialpha**4

        index = np.floor(np.log(r/min_sep) / bin_size).astype(int)
        mask = (index >= 0) & (index < nbins)
        np.add.at(true_npairs, index[mask], 1)
        np.add.at(true_weight, index[mask], ww[mask])
        np.add.at(true_xi, index[mask], xi[mask])

    true_xi /= true_weight

    print('true_npairs = ',true_npairs)
    print('diff = ',kq.npairs - true_npairs)
    np.testing.assert_array_equal(kq.npairs, true_npairs)

    print('true_weight = ',true_weight)
    print('diff = ',kq.weight - true_weight)
    np.testing.assert_allclose(kq.weight, true_weight, rtol=1.e-6, atol=1.e-8)

    print('true_xi = ',true_xi)
    print('kq.xi = ',kq.xi)
    print('kq.xi_im = ',kq.xi_im)
    np.testing.assert_allclose(kq.xi, true_xi.real, rtol=1.e-6, atol=1.e-8)
    np.testing.assert_allclose(kq.xi_im, true_xi.imag, rtol=1.e-6, atol=1.e-8)

    # Check that running via the corr2 script works correctly.
    config = treecorr.config.read_config('configs/kq_direct.yaml')
    try:
        import fitsio
    except ImportError:
        pass
    else:
        cat1.write(config['file_name'])
        cat2.write(config['file_name2'])
        treecorr.corr2(config)
        data = fitsio.read(config['kq_file_name'])
        np.testing.assert_allclose(data['r_nom'], kq.rnom)
        np.testing.assert_allclose(data['npairs'], kq.npairs)
        np.testing.assert_allclose(data['weight'], kq.weight)
        np.testing.assert_allclose(data['xi'], kq.xi)
        np.testing.assert_allclose(data['xi_im'], kq.xi_im)

        # Invalid with only one file_name
        del config['file_name2']
        with assert_raises(TypeError):
            treecorr.corr2(config)

    # Repeat with binslop = 0, since code is different for bin_slop=0 and brute=True.
    # And don't do any top-level recursion so we actually test not going to the leaves.
    kq = treecorr.KQCorrelation(min_sep=min_sep, max_sep=max_sep, nbins=nbins, bin_slop=0,
                                max_top=0)
    kq.process(cat1, cat2)
    np.testing.assert_array_equal(kq.npairs, true_npairs)
    np.testing.assert_allclose(kq.weight, true_weight, rtol=1.e-6, atol=1.e-8)
    np.testing.assert_allclose(kq.xi, true_xi.real, atol=2.e-3)
    np.testing.assert_allclose(kq.xi_im, true_xi.imag, atol=2.e-3)

    # With angle_slop = 0, it goes back to being basically exact (to single precision).
    kq = treecorr.KQCorrelation(min_sep=min_sep, max_sep=max_sep, nbins=nbins, bin_slop=0,
                                angle_slop=0, max_top=0)
    kq.process(cat1, cat2)
    np.testing.assert_array_equal(kq.npairs, true_npairs)
    np.testing.assert_allclose(kq.weight, true_weight, rtol=1.e-6, atol=1.e-8)
    np.testing.assert_allclose(kq.xi, true_xi.real, rtol=1.e-6, atol=1.e-8)
    np.testing.assert_allclose(kq.xi_im, true_xi.imag, rtol=1.e-6, atol=1.e-8)

    # Check a few basic operations with a KQCorrelation object.
    do_pickle(kq)

    kq2 = kq.copy()
    kq2 += kq
    np.testing.assert_allclose(kq2.npairs, 2*kq.npairs)
    np.testing.assert_allclose(kq2.weight, 2*kq.weight)
    np.testing.assert_allclose(kq2.meanr, 2*kq.meanr)
    np.testing.assert_allclose(kq2.meanlogr, 2*kq.meanlogr)
    np.testing.assert_allclose(kq2.xi, 2*kq.xi)
    np.testing.assert_allclose(kq2.xi_im, 2*kq.xi_im)

    kq2.clear()
    kq2 += kq
    np.testing.assert_allclose(kq2.npairs, kq.npairs)
    np.testing.assert_allclose(kq2.weight, kq.weight)
    np.testing.assert_allclose(kq2.meanr, kq.meanr)
    np.testing.assert_allclose(kq2.meanlogr, kq.meanlogr)
    np.testing.assert_allclose(kq2.xi, kq.xi)
    np.testing.assert_allclose(kq2.xi_im, kq.xi_im)

    ascii_name = 'output/kq_ascii.txt'
    kq.write(ascii_name, precision=16)
    kq3 = treecorr.KQCorrelation(min_sep=min_sep, max_sep=max_sep, nbins=nbins, bin_type='Log')
    kq3.read(ascii_name)
    np.testing.assert_allclose(kq3.npairs, kq.npairs)
    np.testing.assert_allclose(kq3.weight, kq.weight)
    np.testing.assert_allclose(kq3.meanr, kq.meanr)
    np.testing.assert_allclose(kq3.meanlogr, kq.meanlogr)
    np.testing.assert_allclose(kq3.xi, kq.xi)
    np.testing.assert_allclose(kq3.xi_im, kq.xi_im)

    # Check that the repr is minimal
    assert repr(kq3) == f'KQCorrelation(min_sep={min_sep}, max_sep={max_sep}, nbins={nbins})'

    # New in version 5.0 is a simpler API for reading
    with CaptureLog() as cl:
        kq3b = treecorr.KQCorrelation.from_file(ascii_name, logger=cl.logger)
    assert ascii_name in cl.output
    np.testing.assert_allclose(kq3b.npairs, kq.npairs)
    np.testing.assert_allclose(kq3b.weight, kq.weight)
    np.testing.assert_allclose(kq3b.meanr, kq.meanr)
    np.testing.assert_allclose(kq3b.meanlogr, kq.meanlogr)
    np.testing.assert_allclose(kq3b.xi, kq.xi)
    np.testing.assert_allclose(kq3b.xi_im, kq.xi_im)

    # or using the Corr2 base class
    with CaptureLog() as cl:
        kq3c = treecorr.Corr2.from_file(ascii_name, logger=cl.logger)
    assert ascii_name in cl.output
    np.testing.assert_allclose(kq3b.npairs, kq.npairs)
    np.testing.assert_allclose(kq3b.weight, kq.weight)
    np.testing.assert_allclose(kq3b.meanr, kq.meanr)
    np.testing.assert_allclose(kq3b.meanlogr, kq.meanlogr)
    np.testing.assert_allclose(kq3b.xi, kq.xi)
    np.testing.assert_allclose(kq3b.xi_im, kq.xi_im)

    try:
        import fitsio
    except ImportError:
        pass
    else:
        fits_name = 'output/kq_fits.fits'
        kq.write(fits_name)
        kq4 = treecorr.KQCorrelation(min_sep=min_sep, max_sep=max_sep, nbins=nbins)
        kq4.read(fits_name)
        np.testing.assert_allclose(kq4.npairs, kq.npairs)
        np.testing.assert_allclose(kq4.weight, kq.weight)
        np.testing.assert_allclose(kq4.meanr, kq.meanr)
        np.testing.assert_allclose(kq4.meanlogr, kq.meanlogr)
        np.testing.assert_allclose(kq4.xi, kq.xi)
        np.testing.assert_allclose(kq4.xi_im, kq.xi_im)

        kq4b = treecorr.KQCorrelation.from_file(fits_name)
        np.testing.assert_allclose(kq4b.npairs, kq.npairs)
        np.testing.assert_allclose(kq4b.weight, kq.weight)
        np.testing.assert_allclose(kq4b.meanr, kq.meanr)
        np.testing.assert_allclose(kq4b.meanlogr, kq.meanlogr)
        np.testing.assert_allclose(kq4b.xi, kq.xi)
        np.testing.assert_allclose(kq4b.xi_im, kq.xi_im)

    with assert_raises(TypeError):
        kq2 += config
    kq4 = treecorr.KQCorrelation(min_sep=min_sep/2, max_sep=max_sep, nbins=nbins)
    with assert_raises(ValueError):
        kq2 += kq4
    kq5 = treecorr.KQCorrelation(min_sep=min_sep, max_sep=max_sep*2, nbins=nbins)
    with assert_raises(ValueError):
        kq2 += kq5
    kq6 = treecorr.KQCorrelation(min_sep=min_sep, max_sep=max_sep, nbins=nbins*2)
    with assert_raises(ValueError):
        kq2 += kq6
    with assert_raises(ValueError):
        kq.process(cat1, cat2, patch_method='nonlocal')


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
    q12 = rng.normal(0,0.2, (ngal,) )
    q22 = rng.normal(0,0.2, (ngal,) )

    ra1, dec1 = coord.CelestialCoord.xyz_to_radec(x1,y1,z1)
    ra2, dec2 = coord.CelestialCoord.xyz_to_radec(x2,y2,z2)

    cat1 = treecorr.Catalog(ra=ra1, dec=dec1, ra_units='rad', dec_units='rad', w=w1, k=k1)
    cat2 = treecorr.Catalog(ra=ra2, dec=dec2, ra_units='rad', dec_units='rad', w=w2, q1=q12, q2=q22)

    min_sep = 1.
    max_sep = 10.
    nbins = 50
    bin_size = np.log(max_sep/min_sep) / nbins
    kq = treecorr.KQCorrelation(min_sep=min_sep, max_sep=max_sep, nbins=nbins,
                                sep_units='deg', brute=True)
    kq.process(cat1, cat2)

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

            # Rotate vectors to coordinates where line connecting is horizontal.
            # Original orientation is where north is up.
            theta2 = 90*coord.degrees + c2[j].angleBetween(c1[i], north_pole)
            exp4theta2 = np.cos(4*theta2) + 1j * np.sin(4*theta2)

            q2 = q12[j] + 1j * q22[j]
            q2 *= exp4theta2

            ww = w1[i] * w2[j]
            xi = ww * k1[i] * q2

            true_npairs[index] += 1
            true_weight[index] += ww
            true_xi[index] += xi

    true_xi /= true_weight

    print('true_npairs = ',true_npairs)
    print('diff = ',kq.npairs - true_npairs)
    np.testing.assert_array_equal(kq.npairs, true_npairs)

    print('true_weight = ',true_weight)
    print('diff = ',kq.weight - true_weight)
    np.testing.assert_allclose(kq.weight, true_weight, rtol=1.e-6, atol=1.e-8)

    print('true_xi = ',true_xi)
    print('kq.xi = ',kq.xi)
    np.testing.assert_allclose(kq.xi, true_xi.real, rtol=1.e-6, atol=1.e-8)
    np.testing.assert_allclose(kq.xi_im, true_xi.imag, rtol=1.e-6, atol=1.e-8)

    # Check that running via the corr2 script works correctly.
    config = treecorr.config.read_config('configs/kq_direct_spherical.yaml')
    try:
        import fitsio
    except ImportError:
        pass
    else:
        cat1.write(config['file_name'])
        cat2.write(config['file_name2'])
        treecorr.corr2(config)
        data = fitsio.read(config['kq_file_name'])
        np.testing.assert_allclose(data['r_nom'], kq.rnom)
        np.testing.assert_allclose(data['npairs'], kq.npairs)
        np.testing.assert_allclose(data['weight'], kq.weight)
        np.testing.assert_allclose(data['xi'], kq.xi)
        np.testing.assert_allclose(data['xi_im'], kq.xi_im)

    # Repeat with binslop = 0
    # And don't do any top-level recursion so we actually test not going to the leaves.
    kq = treecorr.KQCorrelation(min_sep=min_sep, max_sep=max_sep, nbins=nbins,
                                sep_units='deg', bin_slop=0, max_top=0)
    kq.process(cat1, cat2)
    np.testing.assert_array_equal(kq.npairs, true_npairs)
    np.testing.assert_allclose(kq.weight, true_weight, rtol=1.e-6, atol=1.e-8)
    np.testing.assert_allclose(kq.xi, true_xi.real, atol=2.e-3)
    np.testing.assert_allclose(kq.xi_im, true_xi.imag, atol=2.e-3)

    # With angle_slop = 0, it goes back to being basically exact (to single precision).
    kq = treecorr.KQCorrelation(min_sep=min_sep, max_sep=max_sep, nbins=nbins,
                                sep_units='deg', bin_slop=0, angle_slop=0, max_top=0)
    kq.process(cat1, cat2)
    np.testing.assert_array_equal(kq.npairs, true_npairs)
    np.testing.assert_allclose(kq.weight, true_weight, rtol=1.e-6, atol=1.e-8)
    np.testing.assert_allclose(kq.xi, true_xi.real, rtol=1.e-6, atol=1.e-8)
    np.testing.assert_allclose(kq.xi_im, true_xi.imag, rtol=1.e-6, atol=1.e-8)


@timer
def test_single():
    # Use q_radial(r) = q0 exp(-r^2/2r0^2) around a single lens
    # i.e. q(r) = q0 exp(-r^2/2r0^2) (x+iy)^4/r^4

    nsource = 100000
    q0 = 0.05
    kappa = 0.23
    r0 = 10.
    L = 5. * r0
    rng = np.random.RandomState(8675309)
    x = (rng.random_sample(nsource)-0.5) * L
    y = (rng.random_sample(nsource)-0.5) * L
    r2 = (x**2 + y**2)
    r = np.sqrt(r2)
    qrad = q0 * np.exp(-0.5*r2/r0**2)
    theta = np.arctan2(y,x)
    q1 = qrad * np.cos(4*theta)
    q2 = qrad * np.sin(4*theta)

    lens_cat = treecorr.Catalog(x=[0], y=[0], k=[kappa],  x_units='arcmin', y_units='arcmin')
    source_cat = treecorr.Catalog(x=x, y=y, q1=q1, q2=q2, x_units='arcmin', y_units='arcmin')
    kq = treecorr.KQCorrelation(bin_size=0.1, min_sep=1., max_sep=20., sep_units='arcmin',
                                verbose=1)
    kq.process(lens_cat, source_cat)

    # log(<R>) != <logR>, but it should be close:
    print('meanlogr - log(meanr) = ',kq.meanlogr - np.log(kq.meanr))
    np.testing.assert_allclose(kq.meanlogr, np.log(kq.meanr), atol=1.e-3)

    r = kq.meanr
    true_kqr = kappa * q0 * np.exp(-0.5*r**2/r0**2)

    print('kq.xi = ',kq.xi)
    print('kq.xi_im = ',kq.xi_im)
    print('true_kqr = ',true_kqr)
    print('ratio = ',kq.xi / true_kqr)
    print('diff = ',kq.xi - true_kqr)
    print('max diff = ',max(abs(kq.xi - true_kqr)))
    np.testing.assert_allclose(kq.xi, true_kqr, rtol=2.e-2)
    np.testing.assert_allclose(kq.xi_im, 0, atol=1.e-4)

    # Check that we get the same result using the corr2 function:
    lens_cat.write(os.path.join('data','kq_single_lens.dat'))
    source_cat.write(os.path.join('data','kq_single_source.dat'))
    config = treecorr.read_config('configs/kq_single.yaml')
    config['verbose'] = 0
    treecorr.corr2(config)
    corr2_output = np.genfromtxt(os.path.join('output','kq_single.out'), names=True,
                                 skip_header=1)
    print('kq.xi = ',kq.xi)
    print('from corr2 output = ',corr2_output['xi'])
    print('ratio = ',corr2_output['xi']/kq.xi)
    print('diff = ',corr2_output['xi']-kq.xi)
    np.testing.assert_allclose(corr2_output['xi'], kq.xi, rtol=1.e-3)

    print('xi_im from corr2 output = ',corr2_output['xi_im'])
    np.testing.assert_allclose(corr2_output['xi_im'], 0., atol=1.e-4)


@timer
def test_kq():
    # Use q_radial(r) = q0 exp(-r^2/2r0^2) around a bunch of foreground lenses.
    # i.e. q(r) = q0 exp(-r^2/2r0^2) (x+iy)^4/r^4
    # For each lens, we divide this by a random kappa value assigned to that lens, so
    # the final kq output shoudl be just q_radial.

    nlens = 1000
    nsource = 50000
    r0 = 10.
    L = 100. * r0

    q0 = 0.05
    rng = np.random.RandomState(8675309)
    xl = (rng.random_sample(nlens)-0.5) * L
    yl = (rng.random_sample(nlens)-0.5) * L
    kl = rng.normal(0.23, 0.05, (nlens,) )
    xs = (rng.random_sample(nsource)-0.5) * L
    ys = (rng.random_sample(nsource)-0.5) * L
    q1 = np.zeros( (nsource,) )
    q2 = np.zeros( (nsource,) )
    for x,y,k in zip(xl,yl,kl):
        dx = xs-x
        dy = ys-y
        r2 = dx**2 + dy**2
        r = np.sqrt(r2)
        qrad = q0 * np.exp(-0.5*r2/r0**2) / k
        theta = np.arctan2(dy,dx)
        q1 += qrad * np.cos(4*theta)
        q2 += qrad * np.sin(4*theta)

    lens_cat = treecorr.Catalog(x=xl, y=yl, k=kl, x_units='arcmin', y_units='arcmin')
    source_cat = treecorr.Catalog(x=xs, y=ys, q1=q1, q2=q2, x_units='arcmin', y_units='arcmin')
    kq = treecorr.KQCorrelation(bin_size=0.1, min_sep=1., max_sep=20., sep_units='arcmin',
                                verbose=1)
    kq.process(lens_cat, source_cat)

    # Using nbins=None rather than omiting nbins is equivalent.
    kq2 = treecorr.KQCorrelation(bin_size=0.1, min_sep=1., max_sep=20., nbins=None, sep_units='arcmin')
    kq2.process(lens_cat, source_cat, num_threads=1)
    kq.process(lens_cat, source_cat, num_threads=1)
    assert kq2 == kq

    r = kq.meanr
    true_qr = q0 * np.exp(-0.5*r**2/r0**2)

    print('kq.xi = ',kq.xi)
    print('kq.xi_im = ',kq.xi_im)
    print('true_qr = ',true_qr)
    print('ratio = ',kq.xi / true_qr)
    print('diff = ',kq.xi - true_qr)
    print('max diff = ',max(abs(kq.xi - true_qr)))
    np.testing.assert_allclose(kq.xi, true_qr, rtol=0.1)
    np.testing.assert_allclose(kq.xi_im, 0., atol=1.e-2)

    # Check that we get the same result using the corr2 function:
    lens_cat.write(os.path.join('data','kq_lens.dat'))
    source_cat.write(os.path.join('data','kq_source.dat'))
    config = treecorr.read_config('configs/kq.yaml')
    config['verbose'] = 0
    config['precision'] = 8
    treecorr.corr2(config)
    corr2_output = np.genfromtxt(os.path.join('output','kq.out'), names=True, skip_header=1)
    print('kq.xi = ',kq.xi)
    print('from corr2 output = ',corr2_output['xi'])
    print('ratio = ',corr2_output['xi']/kq.xi)
    print('diff = ',corr2_output['xi']-kq.xi)
    np.testing.assert_allclose(corr2_output['xi'], kq.xi, rtol=1.e-3)

    print('xi_im from corr2 output = ',corr2_output['xi_im'])
    np.testing.assert_allclose(corr2_output['xi_im'], 0., atol=1.e-2)

    # Check the fits write option
    try:
        import fitsio
    except ImportError:
        pass
    else:
        out_file_name = os.path.join('output','kg_out.fits')
        kq.write(out_file_name)
        data = fitsio.read(out_file_name)
        np.testing.assert_almost_equal(data['r_nom'], np.exp(kq.logr))
        np.testing.assert_almost_equal(data['meanr'], kq.meanr)
        np.testing.assert_almost_equal(data['meanlogr'], kq.meanlogr)
        np.testing.assert_almost_equal(data['xi'], kq.xi)
        np.testing.assert_almost_equal(data['xi_im'], kq.xi_im)
        np.testing.assert_almost_equal(data['sigma'], np.sqrt(kq.varxi))
        np.testing.assert_almost_equal(data['weight'], kq.weight)
        np.testing.assert_almost_equal(data['npairs'], kq.npairs)

        # Check the read function
        kq2 = treecorr.KQCorrelation.from_file(out_file_name)
        np.testing.assert_almost_equal(kq2.logr, kq.logr)
        np.testing.assert_almost_equal(kq2.meanr, kq.meanr)
        np.testing.assert_almost_equal(kq2.meanlogr, kq.meanlogr)
        np.testing.assert_almost_equal(kq2.xi, kq.xi)
        np.testing.assert_almost_equal(kq2.xi_im, kq.xi_im)
        np.testing.assert_almost_equal(kq2.varxi, kq.varxi)
        np.testing.assert_almost_equal(kq2.weight, kq.weight)
        np.testing.assert_almost_equal(kq2.npairs, kq.npairs)
        assert kq2.coords == kq.coords
        assert kq2.metric == kq.metric
        assert kq2.sep_units == kq.sep_units
        assert kq2.bin_type == kq.bin_type


@timer
def test_varxi():
    # Test that varxi is correct (or close) based on actual variance of many runs.

    # Signal doesn't matter much.  Use the one from test_gg.
    q0 = 0.05
    kappa0 = 0.03
    r0 = 10.
    L = 50.*r0
    rng = np.random.RandomState(8675309)

    # Note: to get a good estimate of var(xi), you need a lot of runs.  The number of
    # runs matters much more than the number of galaxies for getting this to pass.
    ngal = 1000
    nruns = 50000

    file_name = 'data/test_varxi_kq.npz'
    print(file_name)
    if not os.path.isfile(file_name):
        all_kqs = []
        for run in range(nruns):
            print(f'{run}/{nruns}')
            x = (rng.random_sample(ngal)-0.5) * L
            y = (rng.random_sample(ngal)-0.5) * L
            # Varied weights are hard, but at least check that non-unit weights work correctly.
            w = np.ones_like(x) * 5
            r2 = (x**2 + y**2)/r0**2
            theta = np.arctan2(y,x)
            q1 = q0 * np.exp(-r2/2.) * np.cos(4*theta)
            q2 = q0 * np.exp(-r2/2.) * np.sin(4*theta)
            k = kappa0 * np.exp(-r2/2.)
            # This time, add some shape noise (different each run).
            q1 += rng.normal(0, 0.3, size=ngal)
            q2 += rng.normal(0, 0.3, size=ngal)
            k += rng.normal(0, 0.1, size=ngal)

            cat = treecorr.Catalog(x=x, y=y, w=w, q1=q1, q2=q2, k=k)
            kq = treecorr.KQCorrelation(bin_size=0.1, min_sep=10., max_sep=100.)
            kq.process(cat, cat)
            all_kqs.append(kq)

        mean_xi = np.mean([kq.xi for kq in all_kqs], axis=0)
        var_xi = np.var([kq.xi for kq in all_kqs], axis=0)
        mean_varxi = np.mean([kq.varxi for kq in all_kqs], axis=0)

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
    theta = np.arctan2(y,x)
    q1 = q0 * np.exp(-r2/2.) * np.cos(4*theta)
    q2 = q0 * np.exp(-r2/2.) * np.sin(4*theta)
    k = kappa0 * np.exp(-r2/2.)
    # This time, add some shape noise (different each run).
    q1 += rng.normal(0, 0.3, size=ngal)
    q2 += rng.normal(0, 0.3, size=ngal)
    k += rng.normal(0, 0.1, size=ngal)

    cat = treecorr.Catalog(x=x, y=y, w=w, q1=q1, q2=q2, k=k)
    kq = treecorr.KQCorrelation(bin_size=0.1, min_sep=10., max_sep=100.)
    kq.process(cat, cat)

    print('single run:')
    print('ratio = ',kq.varxi / var_xi)
    print('max relerr for xi = ',np.max(np.abs((kq.varxi - var_xi)/var_xi)))
    np.testing.assert_allclose(kq.varxi, var_xi, rtol=0.3)

@timer
def test_jk():

    # Same multi-lens field we used for NQ patch test
    q0 = 0.05
    r0 = 30.
    L = 30 * r0
    rng = np.random.RandomState(8675309)

    nsource = 100000
    nrand = 1000
    nlens = 300
    nruns = 1000
    npatch = 64

    corr_params = dict(bin_size=0.3, min_sep=10, max_sep=50, bin_slop=0.1)

    def make_spin4_field(rng):
        x1 = (rng.random(nlens)-0.5) * L
        y1 = (rng.random(nlens)-0.5) * L
        k = rng.random(nlens)*3 + 10
        x2 = (rng.random(nsource)-0.5) * L
        y2 = (rng.random(nsource)-0.5) * L

        # Start with just the noise
        q1 = rng.normal(0, 0.1, size=nsource)
        q2 = rng.normal(0, 0.1, size=nsource)

        # Also a non-zero background constant field
        q1 += 2*q0
        q2 -= 3*q0

        # Add in the signal from all lenses
        for i in range(nlens):
            x2i = x2 - x1[i]
            y2i = y2 - y1[i]
            r2 = (x2i**2 + y2i**2)/r0**2
            theta = np.arctan2(y2i,x2i)
            q1 += q0 * np.exp(-r2/2.) * np.cos(4*theta)
            q2 += q0 * np.exp(-r2/2.) * np.sin(4*theta)
        return x1, y1, k, x2, y2, q1, q2

    file_name = 'data/test_kq_jk_{}.npz'.format(nruns)
    print(file_name)
    if not os.path.isfile(file_name):
        all_kqs = []
        rng = np.random.default_rng()
        for run in range(nruns):
            x1, y1, k, x2, y2, q1, q2 = make_spin4_field(rng)
            print(run,': ',np.mean(q1),np.std(q1),np.min(q1),np.max(q1))
            cat1 = treecorr.Catalog(x=x1, y=y1, k=k)
            cat2 = treecorr.Catalog(x=x2, y=y2, q1=q1, q2=q2)
            kq = treecorr.KQCorrelation(corr_params)
            kq.process(cat1, cat2)
            all_kqs.append(kq)

        mean_xi = np.mean([kq.xi for kq in all_kqs], axis=0)
        var_xi = np.var([kq.xi for kq in all_kqs], axis=0)
        mean_varxi = np.mean([kq.varxi for kq in all_kqs], axis=0)

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
    x1, y1, k, x2, y2, q1, q2 = make_spin4_field(rng)

    cat1 = treecorr.Catalog(x=x1, y=y1, k=k)
    cat2 = treecorr.Catalog(x=x2, y=y2, q1=q1, q2=q2)
    kq1 = treecorr.KQCorrelation(corr_params)
    kq1.process(cat1, cat2)

    print('weight = ',kq1.weight)
    print('xi = ',kq1.xi)
    print('varxi = ',kq1.varxi)
    print('pullsq for xi = ',(kq1.xi-mean_xi)**2/var_xi)
    print('max pull for xi = ',np.sqrt(np.max((kq1.xi-mean_xi)**2/var_xi)))
    np.testing.assert_array_less((kq1.xi-mean_xi)**2, 9*var_xi)  # < 3 sigma pull
    np.testing.assert_allclose(kq1.varxi, mean_varxi, rtol=0.1)

    # Now run with patches, but still with shot variance.  Should be basically the same answer.
    cat2p = treecorr.Catalog(x=x2, y=y2, q1=q1, q2=q2, npatch=npatch)
    cat1p = treecorr.Catalog(x=x1, y=y1, k=k, patch_centers=cat2p.patch_centers)
    kq2 = treecorr.KQCorrelation(corr_params)
    kq2.process(cat1p, cat2p)
    print('weight = ',kq2.weight)
    print('xi = ',kq2.xi)
    print('xi1 = ',kq1.xi)
    print('varxi = ',kq2.varxi)
    np.testing.assert_allclose(kq2.weight, kq1.weight, rtol=1.e-2)
    np.testing.assert_allclose(kq2.xi, kq1.xi, rtol=2.e-2)
    np.testing.assert_allclose(kq2.varxi, kq1.varxi, rtol=1.e-2)

    # estimate_cov with var_method='shot' returns just the diagonal.
    np.testing.assert_allclose(kq2.estimate_cov('shot'), kq2.varxi)
    np.testing.assert_allclose(kq1.estimate_cov('shot'), kq1.varxi)

    # Now try jackknife variance estimate.
    cov2 = kq2.estimate_cov('jackknife')
    print('varxi = ',np.diagonal(cov2))
    print('cf var_xi = ',var_xi)
    np.testing.assert_allclose(np.diagonal(cov2), var_xi, rtol=0.6)

    # Check only using patches for one of the two catalogs.
    kq3 = treecorr.KQCorrelation(corr_params, var_method='jackknife')
    kq3.process(cat1p, cat2)
    print('varxi = ',kq3.varxi)
    np.testing.assert_allclose(kq3.weight, kq1.weight, rtol=1.e-2)
    np.testing.assert_allclose(kq3.xi, kq1.xi, rtol=1.e-2)
    np.testing.assert_allclose(kq3.varxi, var_xi, rtol=0.5)

    kq4 = treecorr.KQCorrelation(corr_params, var_method='jackknife', rng=rng)
    kq4.process(cat1, cat2p)
    print('varxi = ',kq4.varxi)
    np.testing.assert_allclose(kq4.weight, kq1.weight, rtol=1.e-2)
    np.testing.assert_allclose(kq4.xi, kq1.xi, rtol=2.e-2)
    np.testing.assert_allclose(kq4.varxi, var_xi, rtol=0.9)

    # Use initialize/finalize
    kq5 = treecorr.KQCorrelation(corr_params)
    for k1, p1 in enumerate(cat1p.get_patches()):
        for k2, p2 in enumerate(cat2p.get_patches()):
            kq5.process(p1, p2, initialize=(k1==k2==0), finalize=(k1==k2==npatch-1))
    np.testing.assert_allclose(kq5.xi, kq2.xi)
    np.testing.assert_allclose(kq5.weight, kq2.weight)
    np.testing.assert_allclose(kq5.varxi, kq2.varxi)

    # Check that these still work after roundtripping through a file.
    try:
        import fitsio
    except ImportError:
        pass
    else:
        file_name = os.path.join('output','test_write_results_kq.fits')
        kq2.write(file_name, write_patch_results=True)
        kq5 = treecorr.KQCorrelation.from_file(file_name)
        cov5 = kq5.estimate_cov('jackknife')
        np.testing.assert_allclose(cov5, cov2)

    # Check some invalid actions
    # Bad var_method
    with assert_raises(ValueError):
        kq2.estimate_cov('invalid')
    # Not run on patches, but need patches
    with assert_raises(ValueError):
        kq1.estimate_cov('jackknife')
    with assert_raises(ValueError):
        kq1.estimate_cov('sample')
    with assert_raises(ValueError):
        kq1.estimate_cov('marked_bootstrap')
    with assert_raises(ValueError):
        kq1.estimate_cov('bootstrap')

    cat1a = treecorr.Catalog(x=x1[:100], y=y1[:100], npatch=10)
    cat2a = treecorr.Catalog(x=x2[:100], y=y2[:100], q1=q1[:100], q2=q2[:100], npatch=10)
    cat1b = treecorr.Catalog(x=x1[:100], y=y1[:100], npatch=2)
    cat2b = treecorr.Catalog(x=x2[:100], y=y2[:100], q1=q1[:100], q2=q2[:100], npatch=2)
    kq6 = treecorr.KQCorrelation(corr_params)
    kq7 = treecorr.KQCorrelation(corr_params)
    # All catalogs need to have the same number of patches
    with assert_raises(RuntimeError):
        kq6.process(cat1a,cat2b)
    with assert_raises(RuntimeError):
        kq7.process(cat1b,cat2a)


if __name__ == '__main__':
    test_direct()
    test_direct_spherical()
    test_single()
    test_kq()
    test_varxi()
    test_jk()
