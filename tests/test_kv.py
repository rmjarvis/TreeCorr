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
    v12 = rng.normal(0,0.2, (ngal,) )
    v22 = rng.normal(0,0.2, (ngal,) )

    cat1 = treecorr.Catalog(x=x1, y=y1, w=w1, k=k1)
    cat2 = treecorr.Catalog(x=x2, y=y2, w=w2, v1=v12, v2=v22)

    min_sep = 1.
    max_sep = 50.
    nbins = 50
    bin_size = np.log(max_sep/min_sep) / nbins
    kv = treecorr.KVCorrelation(min_sep=min_sep, max_sep=max_sep, nbins=nbins, brute=True)
    kv.process(cat1, cat2)

    kv2 = kv.copy()
    kv2.process(cat1, cat2, corr_only=True)
    np.testing.assert_allclose(kv2.weight, kv.weight)
    np.testing.assert_allclose(kv2.xi, kv.xi)
    np.testing.assert_allclose(kv2.xi_im, kv.xi_im)
    np.testing.assert_allclose(kv2.npairs, kv.weight / (np.mean(w1) * np.mean(w2)))
    np.testing.assert_allclose(kv2.meanr, kv.rnom)
    np.testing.assert_allclose(kv2.meanlogr, kv.logr)

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
        xi = ww * k1[i] * (v12 + 1j*v22) * expmialpha

        index = np.floor(np.log(r/min_sep) / bin_size).astype(int)
        mask = (index >= 0) & (index < nbins)
        np.add.at(true_npairs, index[mask], 1)
        np.add.at(true_weight, index[mask], ww[mask])
        np.add.at(true_xi, index[mask], xi[mask])

    true_xi /= true_weight

    print('true_npairs = ',true_npairs)
    print('diff = ',kv.npairs - true_npairs)
    np.testing.assert_array_equal(kv.npairs, true_npairs)

    print('true_weight = ',true_weight)
    print('diff = ',kv.weight - true_weight)
    np.testing.assert_allclose(kv.weight, true_weight, rtol=1.e-6, atol=1.e-8)

    print('true_xi = ',true_xi)
    print('kv.xi = ',kv.xi)
    print('kv.xi_im = ',kv.xi_im)
    np.testing.assert_allclose(kv.xi, true_xi.real, rtol=1.e-6, atol=1.e-8)
    np.testing.assert_allclose(kv.xi_im, true_xi.imag, rtol=1.e-6, atol=1.e-8)

    # Check that running via the corr2 script works correctly.
    config = treecorr.config.read_config('configs/kv_direct.yaml')
    try:
        import fitsio
    except ImportError:
        pass
    else:
        cat1.write(config['file_name'])
        cat2.write(config['file_name2'])
        treecorr.corr2(config)
        data = fitsio.read(config['kv_file_name'])
        np.testing.assert_allclose(data['r_nom'], kv.rnom)
        np.testing.assert_allclose(data['npairs'], kv.npairs)
        np.testing.assert_allclose(data['weight'], kv.weight)
        np.testing.assert_allclose(data['xi'], kv.xi)
        np.testing.assert_allclose(data['xi_im'], kv.xi_im)

        # Invalid with only one file_name
        del config['file_name2']
        with assert_raises(TypeError):
            treecorr.corr2(config)

    # Repeat with binslop = 0, since code is different for bin_slop=0 and brute=True.
    # And don't do any top-level recursion so we actually test not going to the leaves.
    kv = treecorr.KVCorrelation(min_sep=min_sep, max_sep=max_sep, nbins=nbins, bin_slop=0,
                                max_top=0)
    kv.process(cat1, cat2)
    np.testing.assert_array_equal(kv.npairs, true_npairs)
    np.testing.assert_allclose(kv.weight, true_weight, rtol=1.e-6, atol=1.e-8)
    np.testing.assert_allclose(kv.xi, true_xi.real, atol=1.e-3)
    np.testing.assert_allclose(kv.xi_im, true_xi.imag, atol=1.e-3)

    # With angle_slop = 0, it goes back to being basically exact (to single precision).
    kv = treecorr.KVCorrelation(min_sep=min_sep, max_sep=max_sep, nbins=nbins, bin_slop=0,
                                angle_slop=0, max_top=0)
    kv.process(cat1, cat2)
    np.testing.assert_array_equal(kv.npairs, true_npairs)
    np.testing.assert_allclose(kv.weight, true_weight, rtol=1.e-6, atol=1.e-8)
    np.testing.assert_allclose(kv.xi, true_xi.real, rtol=1.e-6, atol=1.e-8)
    np.testing.assert_allclose(kv.xi_im, true_xi.imag, rtol=1.e-6, atol=1.e-8)

    # Check a few basic operations with a KVCorrelation object.
    do_pickle(kv)

    kv2 = kv.copy()
    kv2 += kv
    np.testing.assert_allclose(kv2.npairs, 2*kv.npairs)
    np.testing.assert_allclose(kv2.weight, 2*kv.weight)
    np.testing.assert_allclose(kv2.meanr, 2*kv.meanr)
    np.testing.assert_allclose(kv2.meanlogr, 2*kv.meanlogr)
    np.testing.assert_allclose(kv2.xi, 2*kv.xi)
    np.testing.assert_allclose(kv2.xi_im, 2*kv.xi_im)

    kv2.clear()
    kv2 += kv
    np.testing.assert_allclose(kv2.npairs, kv.npairs)
    np.testing.assert_allclose(kv2.weight, kv.weight)
    np.testing.assert_allclose(kv2.meanr, kv.meanr)
    np.testing.assert_allclose(kv2.meanlogr, kv.meanlogr)
    np.testing.assert_allclose(kv2.xi, kv.xi)
    np.testing.assert_allclose(kv2.xi_im, kv.xi_im)

    ascii_name = 'output/kv_ascii.txt'
    kv.write(ascii_name, precision=16)
    kv3 = treecorr.KVCorrelation(min_sep=min_sep, max_sep=max_sep, nbins=nbins, bin_type='Log')
    kv3.read(ascii_name)
    np.testing.assert_allclose(kv3.npairs, kv.npairs)
    np.testing.assert_allclose(kv3.weight, kv.weight)
    np.testing.assert_allclose(kv3.meanr, kv.meanr)
    np.testing.assert_allclose(kv3.meanlogr, kv.meanlogr)
    np.testing.assert_allclose(kv3.xi, kv.xi)
    np.testing.assert_allclose(kv3.xi_im, kv.xi_im)

    # Check that the repr is minimal
    assert repr(kv3) == f'KVCorrelation(min_sep={min_sep}, max_sep={max_sep}, nbins={nbins})'

    # Simpler API using from_file:
    with CaptureLog() as cl:
        kv3b = treecorr.KVCorrelation.from_file(ascii_name, logger=cl.logger)
    assert ascii_name in cl.output
    np.testing.assert_allclose(kv3b.npairs, kv.npairs)
    np.testing.assert_allclose(kv3b.weight, kv.weight)
    np.testing.assert_allclose(kv3b.meanr, kv.meanr)
    np.testing.assert_allclose(kv3b.meanlogr, kv.meanlogr)
    np.testing.assert_allclose(kv3b.xi, kv.xi)
    np.testing.assert_allclose(kv3b.xi_im, kv.xi_im)

    # or using the Corr2 base class
    with CaptureLog() as cl:
        kv3c = treecorr.Corr2.from_file(ascii_name, logger=cl.logger)
    assert ascii_name in cl.output
    np.testing.assert_allclose(kv3c.npairs, kv.npairs)
    np.testing.assert_allclose(kv3c.weight, kv.weight)
    np.testing.assert_allclose(kv3c.meanr, kv.meanr)
    np.testing.assert_allclose(kv3c.meanlogr, kv.meanlogr)
    np.testing.assert_allclose(kv3c.xi, kv.xi)
    np.testing.assert_allclose(kv3c.xi_im, kv.xi_im)

    try:
        import fitsio
    except ImportError:
        pass
    else:
        fits_name = 'output/kv_fits.fits'
        kv.write(fits_name)
        kv4 = treecorr.KVCorrelation(min_sep=min_sep, max_sep=max_sep, nbins=nbins)
        kv4.read(fits_name)
        np.testing.assert_allclose(kv4.npairs, kv.npairs)
        np.testing.assert_allclose(kv4.weight, kv.weight)
        np.testing.assert_allclose(kv4.meanr, kv.meanr)
        np.testing.assert_allclose(kv4.meanlogr, kv.meanlogr)
        np.testing.assert_allclose(kv4.xi, kv.xi)
        np.testing.assert_allclose(kv4.xi_im, kv.xi_im)

        kv4b = treecorr.KVCorrelation.from_file(fits_name)
        np.testing.assert_allclose(kv4b.npairs, kv.npairs)
        np.testing.assert_allclose(kv4b.weight, kv.weight)
        np.testing.assert_allclose(kv4b.meanr, kv.meanr)
        np.testing.assert_allclose(kv4b.meanlogr, kv.meanlogr)
        np.testing.assert_allclose(kv4b.xi, kv.xi)
        np.testing.assert_allclose(kv4b.xi_im, kv.xi_im)

    with assert_raises(TypeError):
        kv2 += config
    kv4 = treecorr.KVCorrelation(min_sep=min_sep/2, max_sep=max_sep, nbins=nbins)
    with assert_raises(ValueError):
        kv2 += kv4
    kv5 = treecorr.KVCorrelation(min_sep=min_sep, max_sep=max_sep*2, nbins=nbins)
    with assert_raises(ValueError):
        kv2 += kv5
    kv6 = treecorr.KVCorrelation(min_sep=min_sep, max_sep=max_sep, nbins=nbins*2)
    with assert_raises(ValueError):
        kv2 += kv6
    with assert_raises(ValueError):
        kv.process(cat1, cat2, patch_method='nonlocal')

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
    v12 = rng.normal(0,0.2, (ngal,) )
    v22 = rng.normal(0,0.2, (ngal,) )

    ra1, dec1 = coord.CelestialCoord.xyz_to_radec(x1,y1,z1)
    ra2, dec2 = coord.CelestialCoord.xyz_to_radec(x2,y2,z2)

    cat1 = treecorr.Catalog(ra=ra1, dec=dec1, ra_units='rad', dec_units='rad', w=w1, k=k1)
    cat2 = treecorr.Catalog(ra=ra2, dec=dec2, ra_units='rad', dec_units='rad', w=w2, v1=v12, v2=v22)

    min_sep = 1.
    max_sep = 10.
    nbins = 50
    bin_size = np.log(max_sep/min_sep) / nbins
    kv = treecorr.KVCorrelation(min_sep=min_sep, max_sep=max_sep, nbins=nbins,
                                sep_units='deg', brute=True)
    kv.process(cat1, cat2)

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
            exptheta2 = np.cos(theta2) + 1j * np.sin(theta2)

            v2 = v12[j] + 1j * v22[j]
            v2 *= exptheta2

            ww = w1[i] * w2[j]
            xi = ww * k1[i] * v2

            true_npairs[index] += 1
            true_weight[index] += ww
            true_xi[index] += xi

    true_xi /= true_weight

    print('true_npairs = ',true_npairs)
    print('diff = ',kv.npairs - true_npairs)
    np.testing.assert_array_equal(kv.npairs, true_npairs)

    print('true_weight = ',true_weight)
    print('diff = ',kv.weight - true_weight)
    np.testing.assert_allclose(kv.weight, true_weight, rtol=1.e-6, atol=1.e-8)

    print('true_xi = ',true_xi)
    print('kv.xi = ',kv.xi)
    np.testing.assert_allclose(kv.xi, true_xi.real, rtol=1.e-6, atol=1.e-8)
    np.testing.assert_allclose(kv.xi_im, true_xi.imag, rtol=1.e-6, atol=1.e-8)

    # Check that running via the corr2 script works correctly.
    config = treecorr.config.read_config('configs/kv_direct_spherical.yaml')
    try:
        import fitsio
    except ImportError:
        pass
    else:
        cat1.write(config['file_name'])
        cat2.write(config['file_name2'])
        treecorr.corr2(config)
        data = fitsio.read(config['kv_file_name'])
        np.testing.assert_allclose(data['r_nom'], kv.rnom)
        np.testing.assert_allclose(data['npairs'], kv.npairs)
        np.testing.assert_allclose(data['weight'], kv.weight)
        np.testing.assert_allclose(data['xi'], kv.xi)
        np.testing.assert_allclose(data['xi_im'], kv.xi_im)

    # Repeat with binslop = 0
    # And don't do any top-level recursion so we actually test not going to the leaves.
    kv = treecorr.KVCorrelation(min_sep=min_sep, max_sep=max_sep, nbins=nbins,
                                sep_units='deg', bin_slop=0, max_top=0)
    kv.process(cat1, cat2)
    np.testing.assert_array_equal(kv.npairs, true_npairs)
    np.testing.assert_allclose(kv.weight, true_weight, rtol=1.e-6, atol=1.e-8)
    np.testing.assert_allclose(kv.xi, true_xi.real, atol=1.e-3)
    np.testing.assert_allclose(kv.xi_im, true_xi.imag, atol=1.e-3)

    # With angle_slop = 0, it goes back to being basically exact (to single precision).
    kv = treecorr.KVCorrelation(min_sep=min_sep, max_sep=max_sep, nbins=nbins,
                                sep_units='deg', bin_slop=0, angle_slop=0, max_top=0)
    kv.process(cat1, cat2)
    np.testing.assert_array_equal(kv.npairs, true_npairs)
    np.testing.assert_allclose(kv.weight, true_weight, rtol=1.e-6, atol=1.e-8)
    np.testing.assert_allclose(kv.xi, true_xi.real, rtol=1.e-6, atol=1.e-8)
    np.testing.assert_allclose(kv.xi_im, true_xi.imag, rtol=1.e-6, atol=1.e-8)


@timer
def test_single():
    # Use v_radial(r) = v0 exp(-r^2/2r0^2) around a single lens
    # i.e. v(r) = v0 exp(-r^2/2r0^2) (x+iy)/r

    nsource = 100000
    v0 = 0.05
    kappa = 0.23
    r0 = 10.
    L = 5. * r0
    rng = np.random.RandomState(8675309)
    x = (rng.random_sample(nsource)-0.5) * L
    y = (rng.random_sample(nsource)-0.5) * L
    r2 = (x**2 + y**2)
    r = np.sqrt(r2)
    vrad = v0 * np.exp(-0.5*r2/r0**2)
    v1 = vrad * x/r
    v2 = vrad * y/r

    lens_cat = treecorr.Catalog(x=[0], y=[0], k=[kappa],  x_units='arcmin', y_units='arcmin')
    source_cat = treecorr.Catalog(x=x, y=y, v1=v1, v2=v2, x_units='arcmin', y_units='arcmin')
    kv = treecorr.KVCorrelation(bin_size=0.1, min_sep=1., max_sep=20., sep_units='arcmin',
                                verbose=1)
    kv.process(lens_cat, source_cat)

    # log(<R>) != <logR>, but it should be close:
    print('meanlogr - log(meanr) = ',kv.meanlogr - np.log(kv.meanr))
    np.testing.assert_allclose(kv.meanlogr, np.log(kv.meanr), atol=1.e-3)

    r = kv.meanr
    true_kvr = kappa * v0 * np.exp(-0.5*r**2/r0**2)

    print('kv.xi = ',kv.xi)
    print('kv.xi_im = ',kv.xi_im)
    print('true_kvr = ',true_kvr)
    print('ratio = ',kv.xi / true_kvr)
    print('diff = ',kv.xi - true_kvr)
    print('max diff = ',max(abs(kv.xi - true_kvr)))
    np.testing.assert_allclose(kv.xi, true_kvr, rtol=1.e-2)
    np.testing.assert_allclose(kv.xi_im, 0, atol=1.e-4)

    # Check that we get the same result using the corr2 function:
    lens_cat.write(os.path.join('data','kv_single_lens.dat'))
    source_cat.write(os.path.join('data','kv_single_source.dat'))
    config = treecorr.read_config('configs/kv_single.yaml')
    config['verbose'] = 0
    treecorr.corr2(config)
    corr2_output = np.genfromtxt(os.path.join('output','kv_single.out'), names=True,
                                 skip_header=1)
    print('kv.xi = ',kv.xi)
    print('from corr2 output = ',corr2_output['xi'])
    print('ratio = ',corr2_output['xi']/kv.xi)
    print('diff = ',corr2_output['xi']-kv.xi)
    np.testing.assert_allclose(corr2_output['xi'], kv.xi, rtol=1.e-3)

    print('xi_im from corr2 output = ',corr2_output['xi_im'])
    np.testing.assert_allclose(corr2_output['xi_im'], 0., atol=1.e-4)


@timer
def test_kv():
    # Use v_radial(r) = v0 exp(-r^2/2r0^2) around a bunch of foreground lenses.
    # i.e. v(r) = v0 exp(-r^2/2r0^2) (x+iy)/r
    # For each lens, we divide this by a random kappa value assigned to that lens, so
    # the final kv output shoudl be just v_radial.

    nlens = 1000
    nsource = 50000
    r0 = 10.
    L = 100. * r0

    v0 = 0.05
    rng = np.random.RandomState(8675309)
    xl = (rng.random_sample(nlens)-0.5) * L
    yl = (rng.random_sample(nlens)-0.5) * L
    kl = rng.normal(0.23, 0.05, (nlens,) )
    xs = (rng.random_sample(nsource)-0.5) * L
    ys = (rng.random_sample(nsource)-0.5) * L
    v1 = np.zeros( (nsource,) )
    v2 = np.zeros( (nsource,) )
    for x,y,k in zip(xl,yl,kl):
        dx = xs-x
        dy = ys-y
        r2 = dx**2 + dy**2
        r = np.sqrt(r2)
        vrad = v0 * np.exp(-0.5*r2/r0**2) / k
        v1 += vrad * dx/r
        v2 += vrad * dy/r

    lens_cat = treecorr.Catalog(x=xl, y=yl, k=kl, x_units='arcmin', y_units='arcmin')
    source_cat = treecorr.Catalog(x=xs, y=ys, v1=v1, v2=v2, x_units='arcmin', y_units='arcmin')
    kv = treecorr.KVCorrelation(bin_size=0.1, min_sep=1., max_sep=20., sep_units='arcmin',
                                verbose=1)
    t0 = time.time()
    kv.process(lens_cat, source_cat, num_threads=1)
    t1 = time.time()
    print('Time for kv process = ',t1-t0)

    # Using nbins=None rather than omiting nbins is equivalent.
    kv2 = treecorr.KVCorrelation(bin_size=0.1, min_sep=1., max_sep=20., nbins=None, sep_units='arcmin')
    kv2.process(lens_cat, source_cat, num_threads=1)
    assert kv2 == kv

    t2 = time.time()
    kv2.process(lens_cat, source_cat, num_threads=1, corr_only=True)
    t3 = time.time()
    print('Time for corr-only kv process = ',t3-t2)
    np.testing.assert_allclose(kv2.xi, kv.xi)
    np.testing.assert_allclose(kv2.xi_im, kv.xi_im)
    np.testing.assert_allclose(kv2.weight, kv.weight)
    np.testing.assert_allclose(kv2.npairs, kv.npairs)
    if __name__ == '__main__':
        assert t3-t2 < t1-t0

    r = kv.meanr
    true_vr = v0 * np.exp(-0.5*r**2/r0**2)

    print('kv.xi = ',kv.xi)
    print('kv.xi_im = ',kv.xi_im)
    print('true_vr = ',true_vr)
    print('ratio = ',kv.xi / true_vr)
    print('diff = ',kv.xi - true_vr)
    print('max diff = ',max(abs(kv.xi - true_vr)))
    np.testing.assert_allclose(kv.xi, true_vr, rtol=0.1)
    np.testing.assert_allclose(kv.xi_im, 0., atol=1.e-2)

    # Check that we get the same result using the corr2 function:
    lens_cat.write(os.path.join('data','kv_lens.dat'))
    source_cat.write(os.path.join('data','kv_source.dat'))
    config = treecorr.read_config('configs/kv.yaml')
    config['verbose'] = 0
    config['precision'] = 8
    treecorr.corr2(config)
    corr2_output = np.genfromtxt(os.path.join('output','kv.out'), names=True, skip_header=1)
    print('kv.xi = ',kv.xi)
    print('from corr2 output = ',corr2_output['xi'])
    print('ratio = ',corr2_output['xi']/kv.xi)
    print('diff = ',corr2_output['xi']-kv.xi)
    np.testing.assert_allclose(corr2_output['xi'], kv.xi, rtol=1.e-3)

    print('xi_im from corr2 output = ',corr2_output['xi_im'])
    np.testing.assert_allclose(corr2_output['xi_im'], 0., atol=1.e-2)

    # Check the fits write option
    try:
        import fitsio
    except ImportError:
        pass
    else:
        out_file_name1 = os.path.join('output','kg_out1.fits')
        kv.write(out_file_name1)
        data = fitsio.read(out_file_name1)
        np.testing.assert_almost_equal(data['r_nom'], np.exp(kv.logr))
        np.testing.assert_almost_equal(data['meanr'], kv.meanr)
        np.testing.assert_almost_equal(data['meanlogr'], kv.meanlogr)
        np.testing.assert_almost_equal(data['xi'], kv.xi)
        np.testing.assert_almost_equal(data['xi_im'], kv.xi_im)
        np.testing.assert_almost_equal(data['sigma'], np.sqrt(kv.varxi))
        np.testing.assert_almost_equal(data['weight'], kv.weight)
        np.testing.assert_almost_equal(data['npairs'], kv.npairs)

        # Check the read function
        kv2 = treecorr.KVCorrelation.from_file(out_file_name1)
        np.testing.assert_almost_equal(kv2.logr, kv.logr)
        np.testing.assert_almost_equal(kv2.meanr, kv.meanr)
        np.testing.assert_almost_equal(kv2.meanlogr, kv.meanlogr)
        np.testing.assert_almost_equal(kv2.xi, kv.xi)
        np.testing.assert_almost_equal(kv2.xi_im, kv.xi_im)
        np.testing.assert_almost_equal(kv2.varxi, kv.varxi)
        np.testing.assert_almost_equal(kv2.weight, kv.weight)
        np.testing.assert_almost_equal(kv2.npairs, kv.npairs)
        assert kv2.coords == kv.coords
        assert kv2.metric == kv.metric
        assert kv2.sep_units == kv.sep_units
        assert kv2.bin_type == kv.bin_type


@timer
def test_varxi():
    # Test that varxi is correct (or close) based on actual variance of many runs.

    # Signal doesn't matter much.  Use the one from test_gg.
    v0 = 0.05
    kappa0 = 0.03
    r0 = 10.
    L = 50.*r0
    rng = np.random.RandomState(8675309)

    # Note: to get a good estimate of var(xi), you need a lot of runs.  The number of
    # runs matters much more than the number of galaxies for getting this to pass.
    ngal = 1000
    nruns = 50000

    file_name = 'data/test_varxi_kv.npz'
    print(file_name)
    if not os.path.isfile(file_name):
        all_kvs = []
        for run in range(nruns):
            print(f'{run}/{nruns}')
            x = (rng.random_sample(ngal)-0.5) * L
            y = (rng.random_sample(ngal)-0.5) * L
            # Varied weights are hard, but at least check that non-unit weights work correctly.
            w = np.ones_like(x) * 5
            r2 = (x**2 + y**2)/r0**2
            v1 = v0 * np.exp(-r2/2.) * x/r0
            v2 = v0 * np.exp(-r2/2.) * y/r0
            k = kappa0 * np.exp(-r2/2.)
            # This time, add some shape noise (different each run).
            v1 += rng.normal(0, 0.3, size=ngal)
            v2 += rng.normal(0, 0.3, size=ngal)
            k += rng.normal(0, 0.1, size=ngal)

            cat = treecorr.Catalog(x=x, y=y, w=w, v1=v1, v2=v2, k=k)
            kv = treecorr.KVCorrelation(bin_size=0.1, min_sep=10., max_sep=100.)
            kv.process(cat, cat)
            all_kvs.append(kv)

        mean_xi = np.mean([kv.xi for kv in all_kvs], axis=0)
        var_xi = np.var([kv.xi for kv in all_kvs], axis=0)
        mean_varxi = np.mean([kv.varxi for kv in all_kvs], axis=0)

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
    v1 = v0 * np.exp(-r2/2.) * x/r0
    v2 = v0 * np.exp(-r2/2.) * y/r0
    k = kappa0 * np.exp(-r2/2.)
    # This time, add some shape noise (different each run).
    v1 += rng.normal(0, 0.3, size=ngal)
    v2 += rng.normal(0, 0.3, size=ngal)
    k += rng.normal(0, 0.1, size=ngal)

    cat = treecorr.Catalog(x=x, y=y, w=w, v1=v1, v2=v2, k=k)
    kv = treecorr.KVCorrelation(bin_size=0.1, min_sep=10., max_sep=100.)
    kv.process(cat, cat)

    print('single run:')
    print('ratio = ',kv.varxi / var_xi)
    print('max relerr for xi = ',np.max(np.abs((kv.varxi - var_xi)/var_xi)))
    np.testing.assert_allclose(kv.varxi, var_xi, rtol=0.3)

@timer
def test_jk():

    # Skip this test on windows, since it is vv slow.
    if os.name == 'nt': return

    # Same multi-lens field we used for NV patch test
    v0 = 0.05
    r0 = 30.
    L = 30 * r0
    rng = np.random.RandomState(8675309)

    nsource = 100000
    nrand = 1000
    nlens = 300
    nruns = 1000
    npatch = 64

    corr_params = dict(bin_size=0.3, min_sep=10, max_sep=50, bin_slop=0.1)

    def make_velocity_field(rng):
        x1 = (rng.random(nlens)-0.5) * L
        y1 = (rng.random(nlens)-0.5) * L
        k = rng.random(nlens)*3 + 10
        x2 = (rng.random(nsource)-0.5) * L
        y2 = (rng.random(nsource)-0.5) * L

        # Start with just the noise
        v1 = rng.normal(0, 0.1, size=nsource)
        v2 = rng.normal(0, 0.1, size=nsource)

        # Also a non-zero background constant velocity
        v1 += 2*v0
        v2 -= 3*v0

        # Add in the signal from all lenses
        for i in range(nlens):
            x2i = x2 - x1[i]
            y2i = y2 - y1[i]
            r2 = (x2i**2 + y2i**2)/r0**2
            v1 += v0 * np.exp(-r2/2.) * x2i/r0
            v2 += v0 * np.exp(-r2/2.) * y2i/r0
        return x1, y1, k, x2, y2, v1, v2

    file_name = 'data/test_kv_jk_{}.npz'.format(nruns)
    print(file_name)
    if not os.path.isfile(file_name):
        all_kvs = []
        rng = np.random.default_rng()
        for run in range(nruns):
            x1, y1, k, x2, y2, v1, v2 = make_velocity_field(rng)
            print(run,': ',np.mean(v1),np.std(v1),np.min(v1),np.max(v1))
            cat1 = treecorr.Catalog(x=x1, y=y1, k=k)
            cat2 = treecorr.Catalog(x=x2, y=y2, v1=v1, v2=v2)
            kv = treecorr.KVCorrelation(corr_params)
            kv.process(cat1, cat2)
            all_kvs.append(kv)

        mean_xi = np.mean([kv.xi for kv in all_kvs], axis=0)
        var_xi = np.var([kv.xi for kv in all_kvs], axis=0)
        mean_varxi = np.mean([kv.varxi for kv in all_kvs], axis=0)

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
    x1, y1, k, x2, y2, v1, v2 = make_velocity_field(rng)

    cat1 = treecorr.Catalog(x=x1, y=y1, k=k)
    cat2 = treecorr.Catalog(x=x2, y=y2, v1=v1, v2=v2)
    kv1 = treecorr.KVCorrelation(corr_params)
    t0 = time.time()
    kv1.process(cat1, cat2)
    t1 = time.time()
    print('Time for non-patch processing = ',t1-t0)

    print('weight = ',kv1.weight)
    print('xi = ',kv1.xi)
    print('varxi = ',kv1.varxi)
    print('pullsq for xi = ',(kv1.xi-mean_xi)**2/var_xi)
    print('max pull for xi = ',np.sqrt(np.max((kv1.xi-mean_xi)**2/var_xi)))
    np.testing.assert_array_less((kv1.xi-mean_xi)**2, 9*var_xi)  # < 3 sigma pull
    np.testing.assert_allclose(kv1.varxi, mean_varxi, rtol=0.1)

    # Now run with patches, but still with shot variance.  Should be basically the same answer.
    cat2p = treecorr.Catalog(x=x2, y=y2, v1=v1, v2=v2, npatch=npatch)
    cat1p = treecorr.Catalog(x=x1, y=y1, k=k, patch_centers=cat2p.patch_centers)
    kv2 = treecorr.KVCorrelation(corr_params)
    t0 = time.time()
    kv2.process(cat1p, cat2p)
    t1 = time.time()
    print('Time for patch processing = ',t1-t0)
    print('weight = ',kv2.weight)
    print('xi = ',kv2.xi)
    print('xi1 = ',kv1.xi)
    print('varxi = ',kv2.varxi)
    np.testing.assert_allclose(kv2.weight, kv1.weight, rtol=1.e-2)
    np.testing.assert_allclose(kv2.xi, kv1.xi, rtol=2.e-2)
    np.testing.assert_allclose(kv2.varxi, kv1.varxi, rtol=1.e-2)

    # estimate_cov with var_method='shot' returns just the diagonal.
    np.testing.assert_allclose(kv2.estimate_cov('shot'), kv2.varxi)
    np.testing.assert_allclose(kv1.estimate_cov('shot'), kv1.varxi)

    # Now try jackknife variance estimate.
    t0 = time.time()
    cov2 = kv2.estimate_cov('jackknife')
    t1 = time.time()
    print('Time to calculate jackknife covariance = ',t1-t0)
    print('varxi = ',np.diagonal(cov2))
    print('cf var_xi = ',var_xi)
    np.testing.assert_allclose(np.diagonal(cov2), var_xi, rtol=0.6)

    # Check only using patches for one of the two catalogs.
    kv3 = treecorr.KVCorrelation(corr_params, var_method='jackknife')
    t0 = time.time()
    kv3.process(cat1p, cat2)
    t1 = time.time()
    print('Time for only patches for cat1 processing = ',t1-t0)
    print('varxi = ',kv3.varxi)
    np.testing.assert_allclose(kv3.weight, kv1.weight, rtol=1.e-2)
    np.testing.assert_allclose(kv3.xi, kv1.xi, rtol=1.e-2)
    np.testing.assert_allclose(kv3.varxi, var_xi, rtol=0.5)

    kv4 = treecorr.KVCorrelation(corr_params, var_method='jackknife', rng=rng)
    t0 = time.time()
    kv4.process(cat1, cat2p)
    t1 = time.time()
    print('Time for only patches for cat2 processing = ',t1-t0)
    print('varxi = ',kv4.varxi)
    np.testing.assert_allclose(kv4.weight, kv1.weight, rtol=1.e-2)
    np.testing.assert_allclose(kv4.xi, kv1.xi, rtol=2.e-2)
    np.testing.assert_allclose(kv4.varxi, var_xi, rtol=0.9)

    # Use initialize/finalize
    kv5 = treecorr.KVCorrelation(corr_params)
    for k1, p1 in enumerate(cat1p.get_patches()):
        for k2, p2 in enumerate(cat2p.get_patches()):
            kv5.process(p1, p2, initialize=(k1==k2==0), finalize=(k1==k2==npatch-1))
    np.testing.assert_allclose(kv5.xi, kv2.xi)
    np.testing.assert_allclose(kv5.weight, kv2.weight)
    np.testing.assert_allclose(kv5.varxi, kv2.varxi)

    # Check that these still work after roundtripping through a file.
    try:
        import fitsio
    except ImportError:
        pass
    else:
        file_name = os.path.join('output','test_write_results_kv.fits')
        kv2.write(file_name, write_patch_results=True)
        kv5 = treecorr.KVCorrelation.from_file(file_name)
        cov5 = kv5.estimate_cov('jackknife')
        np.testing.assert_allclose(cov5, cov2)

    # Check some invalid actions
    # Bad var_method
    with assert_raises(ValueError):
        kv2.estimate_cov('invalid')
    # Not run on patches, but need patches
    with assert_raises(ValueError):
        kv1.estimate_cov('jackknife')
    with assert_raises(ValueError):
        kv1.estimate_cov('sample')
    with assert_raises(ValueError):
        kv1.estimate_cov('marked_bootstrap')
    with assert_raises(ValueError):
        kv1.estimate_cov('bootstrap')

    cat1a = treecorr.Catalog(x=x1[:100], y=y1[:100], npatch=10)
    cat2a = treecorr.Catalog(x=x2[:100], y=y2[:100], v1=v1[:100], v2=v2[:100], npatch=10)
    cat1b = treecorr.Catalog(x=x1[:100], y=y1[:100], npatch=2)
    cat2b = treecorr.Catalog(x=x2[:100], y=y2[:100], v1=v1[:100], v2=v2[:100], npatch=2)
    kv6 = treecorr.KVCorrelation(corr_params)
    kv7 = treecorr.KVCorrelation(corr_params)
    # All catalogs need to have the same number of patches
    with assert_raises(RuntimeError):
        kv6.process(cat1a,cat2b)
    with assert_raises(RuntimeError):
        kv7.process(cat1b,cat2a)


if __name__ == '__main__':
    test_direct()
    test_direct_spherical()
    test_single()
    test_kv()
    test_varxi()
    test_jk()
