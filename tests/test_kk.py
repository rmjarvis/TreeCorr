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
import coord

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
    k1 = rng.normal(10,1, (ngal,) )

    x2 = rng.normal(0,s, (ngal,) )
    y2 = rng.normal(0,s, (ngal,) )
    w2 = rng.random_sample(ngal)
    k2 = rng.normal(0,3, (ngal,) )

    cat1 = treecorr.Catalog(x=x1, y=y1, w=w1, k=k1)
    cat2 = treecorr.Catalog(x=x2, y=y2, w=w2, k=k2)

    min_sep = 1.
    max_sep = 50.
    nbins = 50
    bin_size = np.log(max_sep/min_sep) / nbins
    kk = treecorr.KKCorrelation(min_sep=min_sep, max_sep=max_sep, nbins=nbins, brute=True)
    kk.process(cat1, cat2)

    true_npairs = np.zeros(nbins, dtype=int)
    true_weight = np.zeros(nbins, dtype=float)
    true_xi = np.zeros(nbins, dtype=float)
    for i in range(ngal):
        # It's hard to do all the pairs at once with numpy operations (although maybe possible).
        # But we can at least do all the pairs for each entry in cat1 at once with arrays.
        rsq = (x1[i]-x2)**2 + (y1[i]-y2)**2
        r = np.sqrt(rsq)

        ww = w1[i] * w2
        xi = ww * k1[i] * k2

        index = np.floor(np.log(r/min_sep) / bin_size).astype(int)
        mask = (index >= 0) & (index < nbins)
        np.add.at(true_npairs, index[mask], 1)
        np.add.at(true_weight, index[mask], ww[mask])
        np.add.at(true_xi, index[mask], xi[mask])

    true_xi /= true_weight

    print('true_npairs = ',true_npairs)
    print('diff = ',kk.npairs - true_npairs)
    np.testing.assert_array_equal(kk.npairs, true_npairs)

    print('true_weight = ',true_weight)
    print('diff = ',kk.weight - true_weight)
    np.testing.assert_allclose(kk.weight, true_weight, rtol=1.e-5, atol=1.e-8)

    print('true_xi = ',true_xi)
    print('kk.xi = ',kk.xi)
    np.testing.assert_allclose(kk.xi, true_xi, rtol=1.e-4, atol=1.e-8)

    # Check that running via the corr2 script works correctly.
    config = treecorr.config.read_config('configs/kk_direct.yaml')
    try:
        import fitsio
    except ImportError:
        pass
    else:
        cat1.write(config['file_name'])
        cat2.write(config['file_name2'])
        treecorr.corr2(config)
        data = fitsio.read(config['kk_file_name'])
        np.testing.assert_allclose(data['r_nom'], kk.rnom)
        np.testing.assert_allclose(data['npairs'], kk.npairs)
        np.testing.assert_allclose(data['weight'], kk.weight)
        np.testing.assert_allclose(data['xi'], kk.xi, rtol=1.e-3)

    # Repeat with binslop = 0
    # And don't do any top-level recursion so we actually test not going to the leaves.
    kk = treecorr.KKCorrelation(min_sep=min_sep, max_sep=max_sep, nbins=nbins, bin_slop=0,
                                max_top=0)
    kk.process(cat1, cat2)
    np.testing.assert_array_equal(kk.npairs, true_npairs)
    np.testing.assert_allclose(kk.weight, true_weight, rtol=1.e-5, atol=1.e-8)
    np.testing.assert_allclose(kk.xi, true_xi, rtol=1.e-4, atol=1.e-8)

    # Check a few basic operations with a KKCorrelation object.
    do_pickle(kk)

    kk2 = kk.copy()
    kk2 += kk
    np.testing.assert_allclose(kk2.npairs, 2*kk.npairs)
    np.testing.assert_allclose(kk2.weight, 2*kk.weight)
    np.testing.assert_allclose(kk2.meanr, 2*kk.meanr)
    np.testing.assert_allclose(kk2.meanlogr, 2*kk.meanlogr)
    np.testing.assert_allclose(kk2.xi, 2*kk.xi)

    kk2.clear()
    kk2 += kk
    np.testing.assert_allclose(kk2.npairs, kk.npairs)
    np.testing.assert_allclose(kk2.weight, kk.weight)
    np.testing.assert_allclose(kk2.meanr, kk.meanr)
    np.testing.assert_allclose(kk2.meanlogr, kk.meanlogr)
    np.testing.assert_allclose(kk2.xi, kk.xi)

    ascii_name = 'output/kk_ascii.txt'
    kk.write(ascii_name, precision=16)
    kk3 = treecorr.KKCorrelation(min_sep=min_sep, max_sep=max_sep, nbins=nbins)
    kk3.read(ascii_name)
    np.testing.assert_allclose(kk3.npairs, kk.npairs)
    np.testing.assert_allclose(kk3.weight, kk.weight)
    np.testing.assert_allclose(kk3.meanr, kk.meanr)
    np.testing.assert_allclose(kk3.meanlogr, kk.meanlogr)
    np.testing.assert_allclose(kk3.xi, kk.xi)

    try:
        import fitsio
    except ImportError:
        pass
    else:
        fits_name = 'output/kk_fits.fits'
        kk.write(fits_name)
        kk4 = treecorr.KKCorrelation(min_sep=min_sep, max_sep=max_sep, nbins=nbins)
        kk4.read(fits_name)
        np.testing.assert_allclose(kk4.npairs, kk.npairs)
        np.testing.assert_allclose(kk4.weight, kk.weight)
        np.testing.assert_allclose(kk4.meanr, kk.meanr)
        np.testing.assert_allclose(kk4.meanlogr, kk.meanlogr)
        np.testing.assert_allclose(kk4.xi, kk.xi)

    with assert_raises(TypeError):
        kk2 += config
    kk4 = treecorr.KKCorrelation(min_sep=min_sep/2, max_sep=max_sep, nbins=nbins)
    with assert_raises(ValueError):
        kk2 += kk4
    kk5 = treecorr.KKCorrelation(min_sep=min_sep, max_sep=max_sep*2, nbins=nbins)
    with assert_raises(ValueError):
        kk2 += kk5
    kk6 = treecorr.KKCorrelation(min_sep=min_sep, max_sep=max_sep, nbins=nbins*2)
    with assert_raises(ValueError):
        kk2 += kk6


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
    k1 = rng.normal(10,1, (ngal,) )

    x2 = rng.normal(0,s, (ngal,) )
    y2 = rng.normal(0,s, (ngal,) ) + 200
    z2 = rng.normal(0,s, (ngal,) )
    w2 = rng.random_sample(ngal)
    k2 = rng.normal(0,3, (ngal,) )

    ra1, dec1 = coord.CelestialCoord.xyz_to_radec(x1,y1,z1)
    ra2, dec2 = coord.CelestialCoord.xyz_to_radec(x2,y2,z2)

    cat1 = treecorr.Catalog(ra=ra1, dec=dec1, ra_units='rad', dec_units='rad', w=w1, k=k1)
    cat2 = treecorr.Catalog(ra=ra2, dec=dec2, ra_units='rad', dec_units='rad', w=w2, k=k2)

    min_sep = 1.
    max_sep = 10.
    nbins = 50
    bin_size = np.log(max_sep/min_sep) / nbins
    kk = treecorr.KKCorrelation(min_sep=min_sep, max_sep=max_sep, nbins=nbins,
                                sep_units='deg', brute=True)
    kk.process(cat1, cat2)

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
            xi = ww * k1[i] * k2[j]

            true_npairs[index] += 1
            true_weight[index] += ww
            true_xi[index] += xi

    true_xi /= true_weight

    print('true_npairs = ',true_npairs)
    print('diff = ',kk.npairs - true_npairs)
    np.testing.assert_array_equal(kk.npairs, true_npairs)

    print('true_weight = ',true_weight)
    print('diff = ',kk.weight - true_weight)
    np.testing.assert_allclose(kk.weight, true_weight, rtol=1.e-5, atol=1.e-8)

    print('true_xi = ',true_xi)
    print('kk.xi = ',kk.xi)
    np.testing.assert_allclose(kk.xi, true_xi, rtol=1.e-4, atol=1.e-8)

    # Check that running via the corr2 script works correctly.
    config = treecorr.config.read_config('configs/kk_direct_spherical.yaml')
    try:
        import fitsio
    except ImportError:
        pass
    else:
        cat1.write(config['file_name'])
        cat2.write(config['file_name2'])
        treecorr.corr2(config)
        data = fitsio.read(config['kk_file_name'])
        np.testing.assert_allclose(data['r_nom'], kk.rnom)
        np.testing.assert_allclose(data['npairs'], kk.npairs)
        np.testing.assert_allclose(data['weight'], kk.weight)
        np.testing.assert_allclose(data['xi'], kk.xi, rtol=1.e-3)

    # Repeat with binslop = 0
    # And don't do any top-level recursion so we actually test not going to the leaves.
    kk = treecorr.KKCorrelation(min_sep=min_sep, max_sep=max_sep, nbins=nbins,
                                sep_units='deg', bin_slop=0, max_top=0)
    kk.process(cat1, cat2)
    np.testing.assert_array_equal(kk.npairs, true_npairs)
    np.testing.assert_allclose(kk.weight, true_weight, rtol=1.e-5, atol=1.e-8)
    np.testing.assert_allclose(kk.xi, true_xi, rtol=1.e-3, atol=1.e-6)


@timer
def test_pairwise():
    # Test the pairwise option.

    ngal = 1000
    s = 10.
    rng = np.random.RandomState(8675309)
    x1 = rng.normal(0,s, (ngal,) )
    y1 = rng.normal(0,s, (ngal,) )
    w1 = rng.random_sample(ngal)
    k1 = rng.normal(10,1, (ngal,) )

    x2 = rng.normal(0,s, (ngal,) )
    y2 = rng.normal(0,s, (ngal,) )
    w2 = rng.random_sample(ngal)
    k2 = rng.normal(0,3, (ngal,) )

    w1 = np.ones_like(w1)
    w2 = np.ones_like(w2)

    cat1 = treecorr.Catalog(x=x1, y=y1, w=w1, k=k1)
    cat2 = treecorr.Catalog(x=x2, y=y2, w=w2, k=k2)

    min_sep = 5.
    max_sep = 50.
    nbins = 10
    bin_size = np.log(max_sep/min_sep) / nbins
    kk = treecorr.KKCorrelation(min_sep=min_sep, max_sep=max_sep, nbins=nbins)
    with assert_warns(FutureWarning):
        kk.process_pairwise(cat1, cat2)
    kk.finalize(cat1.vark, cat2.vark)

    true_npairs = np.zeros(nbins, dtype=int)
    true_weight = np.zeros(nbins, dtype=float)
    true_xi = np.zeros(nbins, dtype=float)

    rsq = (x1-x2)**2 + (y1-y2)**2
    r = np.sqrt(rsq)

    ww = w1 * w2
    xi = ww * k1 * k2

    index = np.floor(np.log(r/min_sep) / bin_size).astype(int)
    mask = (index >= 0) & (index < nbins)
    np.add.at(true_npairs, index[mask], 1)
    np.add.at(true_weight, index[mask], ww[mask])
    np.add.at(true_xi, index[mask], xi[mask])

    true_xi /= true_weight

    np.testing.assert_array_equal(kk.npairs, true_npairs)
    np.testing.assert_allclose(kk.weight, true_weight, rtol=1.e-5, atol=1.e-8)
    np.testing.assert_allclose(kk.xi, true_xi, rtol=1.e-4, atol=1.e-8)

    # If cats have names, then the logger will mention them.
    # Also, test running with optional args.
    cat1.name = "first"
    cat2.name = "second"
    with CaptureLog() as cl:
        kk.logger = cl.logger
        with assert_warns(FutureWarning):
            kk.process_pairwise(cat1, cat2, metric='Euclidean', num_threads=2)
    assert "for cats first, second" in cl.output


@timer
def test_constant():
    # A fairly trivial test is to use a constant value of kappa everywhere.

    ngal = 100000
    A = 0.05
    L = 100.
    rng = np.random.RandomState(8675309)
    x = (rng.random_sample(ngal)-0.5) * L
    y = (rng.random_sample(ngal)-0.5) * L
    kappa = A * np.ones(ngal)

    cat = treecorr.Catalog(x=x, y=y, k=kappa, x_units='arcmin', y_units='arcmin')
    kk = treecorr.KKCorrelation(bin_size=0.1, min_sep=0.1, max_sep=10., sep_units='arcmin')
    kk.process(cat)
    print('kk.xi = ',kk.xi)
    np.testing.assert_allclose(kk.xi, A**2, rtol=1.e-6)

    # Now add some noise to the values. It should still work, but at slightly lower accuracy.
    kappa += 0.001 * (rng.random_sample(ngal)-0.5)
    cat = treecorr.Catalog(x=x, y=y, k=kappa, x_units='arcmin', y_units='arcmin')
    kk.process(cat)
    print('kk.xi = ',kk.xi)
    np.testing.assert_allclose(kk.xi, A**2, rtol=1.e-3)


@timer
def test_kk():
    # cf. http://adsabs.harvard.edu/abs/2002A%26A...389..729S for the basic formulae I use here.
    #
    # Use kappa(r) = A exp(-r^2/2s^2)
    #
    # The Fourier transform is: kappa~(k) = 2 pi A s^2 exp(-s^2 k^2/2) / L^2
    # P(k) = (1/2pi) <|kappa~(k)|^2> = 2 pi A^2 (s/L)^4 exp(-s^2 k^2)
    # xi(r) = (1/2pi) int( dk k P(k) J0(kr) )
    #       = pi A^2 (s/L)^2 exp(-r^2/2s^2/4)
    # Note: I'm not sure I handled the L factors correctly, but the units at the end need
    # to be kappa^2, so it needs to be (s/L)^2.

    s = 10.
    if __name__ == '__main__':
        ngal = 1000000
        L = 30. * s  # Not infinity, so this introduces some error.  Our integrals were to infinity.
        tol_factor = 1
    else:
        ngal = 100000
        L = 30. * s
        tol_factor = 2

    A = 0.05
    rng = np.random.RandomState(8675309)
    x = (rng.random_sample(ngal)-0.5) * L
    y = (rng.random_sample(ngal)-0.5) * L
    r2 = (x**2 + y**2)/s**2
    kappa = A * np.exp(-r2/2.)

    cat = treecorr.Catalog(x=x, y=y, k=kappa, x_units='arcmin', y_units='arcmin')
    kk = treecorr.KKCorrelation(bin_size=0.1, min_sep=1., max_sep=20., sep_units='arcmin',
                                verbose=1)
    kk.process(cat)

    # Using nbins=None rather than omiting nbins is equivalent.
    kk2 = treecorr.KKCorrelation(bin_size=0.1, min_sep=1., max_sep=20., nbins=None,
                                 sep_units='arcmin')
    kk2.process(cat, num_threads=1)
    kk.process(cat, num_threads=1)
    assert kk2 == kk

    # log(<R>) != <logR>, but it should be close:
    print('meanlogr - log(meanr) = ',kk.meanlogr - np.log(kk.meanr))
    np.testing.assert_allclose(kk.meanlogr, np.log(kk.meanr), atol=1.e-3)

    r = kk.meanr
    true_xi = np.pi * A**2 * (s/L)**2 * np.exp(-0.25*r**2/s**2)
    print('kk.xi = ',kk.xi)
    print('true_xi = ',true_xi)
    print('ratio = ',kk.xi / true_xi)
    print('diff = ',kk.xi - true_xi)
    print('max diff = ',max(abs(kk.xi - true_xi)))
    print('max rel diff = ',max(abs((kk.xi - true_xi)/true_xi)))
    np.testing.assert_allclose(kk.xi, true_xi, rtol=0.1*tol_factor)

    # It should also work as a cross-correlation of this cat with itself
    kk.process(cat,cat)
    np.testing.assert_allclose(kk.meanlogr, np.log(kk.meanr), atol=1.e-3)
    np.testing.assert_allclose(kk.xi, true_xi, rtol=0.1*tol_factor)

    # Check that we get the same result using the corr2 function
    cat.write(os.path.join('data','kk.dat'))
    config = treecorr.read_config('configs/kk.yaml')
    config['verbose'] = 0
    config['precision'] = 8
    treecorr.corr2(config)
    corr2_output = np.genfromtxt(os.path.join('output','kk.out'), names=True, skip_header=1)
    print('kk.xi = ',kk.xi)
    print('from corr2 output = ',corr2_output['xi'])
    print('ratio = ',corr2_output['xi']/kk.xi)
    print('diff = ',corr2_output['xi']-kk.xi)
    np.testing.assert_allclose(corr2_output['xi'], kk.xi, rtol=1.e-3)

    # Check the fits write option
    try:
        import fitsio
    except ImportError:
        pass
    else:
        out_file_name = os.path.join('output','kk_out.fits')
        kk.write(out_file_name)
        data = fitsio.read(out_file_name)
        np.testing.assert_almost_equal(data['r_nom'], np.exp(kk.logr))
        np.testing.assert_almost_equal(data['meanr'], kk.meanr)
        np.testing.assert_almost_equal(data['meanlogr'], kk.meanlogr)
        np.testing.assert_almost_equal(data['xi'], kk.xi)
        np.testing.assert_almost_equal(data['sigma_xi'], np.sqrt(kk.varxi))
        np.testing.assert_almost_equal(data['weight'], kk.weight)
        np.testing.assert_almost_equal(data['npairs'], kk.npairs)

        # Check the read function
        kk2 = treecorr.KKCorrelation(bin_size=0.1, min_sep=1., max_sep=20., sep_units='arcmin')
        kk2.read(out_file_name)
        np.testing.assert_almost_equal(kk2.logr, kk.logr)
        np.testing.assert_almost_equal(kk2.meanr, kk.meanr)
        np.testing.assert_almost_equal(kk2.meanlogr, kk.meanlogr)
        np.testing.assert_almost_equal(kk2.xi, kk.xi)
        np.testing.assert_almost_equal(kk2.varxi, kk.varxi)
        np.testing.assert_almost_equal(kk2.weight, kk.weight)
        np.testing.assert_almost_equal(kk2.npairs, kk.npairs)
        assert kk2.coords == kk.coords
        assert kk2.metric == kk.metric
        assert kk2.sep_units == kk.sep_units
        assert kk2.bin_type == kk.bin_type

@timer
def test_large_scale():
    # Test very large scales, comparing Arc, Euclidean (spherical), and Euclidean (3d)

    # Distribute points uniformly in all directions.
    if __name__ == '__main__':
        ngal = 100000
        tol = 1
        nbins = 100
        half = 50
    else:
        # Use fewer galaxies when running nosetests, so this is faster
        ngal = 10000
        tol = 3  # A factor by which we scale the tolerances to work with the smaller ngal.
        nbins = 50
        half = 25
    s = 1.
    rng = np.random.RandomState(8675309)
    x = rng.normal(0, s, (ngal,) )
    y = rng.normal(0, s, (ngal,) )
    z = rng.normal(0, s, (ngal,) )
    r = np.sqrt( x*x + y*y + z*z )
    dec = np.arcsin(z/r)
    ra = np.arctan2(y,x)
    r = np.ones_like(x)

    # Use x for "kappa" so there's a strong real correlation function
    cat1 = treecorr.Catalog(ra=ra, dec=dec, k=x, ra_units='rad', dec_units='rad')
    cat2 = treecorr.Catalog(ra=ra, dec=dec, k=x, r=r, ra_units='rad', dec_units='rad')

    config = {
        'min_sep' : 0.01,
        'max_sep' : 1.57,
        'nbins' : nbins,
        'bin_slop' : 0.2,
        'verbose' : 1
    }
    kk_sphere = treecorr.KKCorrelation(config)
    kk_chord = treecorr.KKCorrelation(config)
    kk_euclid = treecorr.KKCorrelation(config)
    kk_euclid.process(cat1, metric='Euclidean')
    kk_sphere.process(cat1, metric='Arc')
    kk_chord.process(cat2, metric='Euclidean')

    for tag in [ 'rnom', 'logr', 'meanr', 'meanlogr', 'npairs', 'xi' ]:
        for name, dd in [ ('Euclid', kk_euclid), ('Sphere', kk_sphere), ('Chord', kk_chord) ]:
            print(name, tag, '=', getattr(dd,tag))

    # rnom and logr should be identical
    np.testing.assert_array_equal(kk_sphere.rnom, kk_euclid.rnom)
    np.testing.assert_array_equal(kk_chord.rnom, kk_euclid.rnom)
    np.testing.assert_array_equal(kk_sphere.logr, kk_euclid.logr)
    np.testing.assert_array_equal(kk_chord.logr, kk_euclid.logr)

    # meanr should be similar for sphere and chord, but euclid is larger, since the chord
    # distances have been scaled up to the real great circle distances
    np.testing.assert_allclose(kk_sphere.meanr, kk_chord.meanr, rtol=1.e-3*tol)
    np.testing.assert_allclose(kk_chord.meanr[:half], kk_euclid.meanr[:half], rtol=1.e-3*tol)
    np.testing.assert_array_less(kk_chord.meanr[half:], kk_euclid.meanr[half:])
    np.testing.assert_allclose(kk_sphere.meanlogr, kk_chord.meanlogr, atol=2.e-2*tol)
    np.testing.assert_allclose(kk_chord.meanlogr[:half], kk_euclid.meanlogr[:half], atol=2.e-2*tol)
    np.testing.assert_array_less(kk_chord.meanlogr[half:], kk_euclid.meanlogr[half:])

    # npairs is basically the same for chord and euclid since the only difference there comes from
    # differences in where they cut off the tree traversal, so the number of pairs is almost equal,
    # even though the separations in each bin are given a different nominal distance.
    # Sphere is smaller than both at all scales, since it is measuring the correlation
    # function on larger real scales at each position.
    print('diff (c-e)/e = ',(kk_chord.npairs-kk_euclid.npairs)/kk_euclid.npairs)
    print('max = ',np.max(np.abs((kk_chord.npairs-kk_euclid.npairs)/kk_euclid.npairs)))
    np.testing.assert_allclose(kk_chord.npairs, kk_euclid.npairs, rtol=1.e-3*tol)
    print('diff (s-e)/e = ',(kk_sphere.npairs-kk_euclid.npairs)/kk_euclid.npairs)
    np.testing.assert_allclose(kk_sphere.npairs[:half], kk_euclid.npairs[:half], rtol=3.e-3*tol)
    np.testing.assert_array_less(kk_sphere.npairs[half:], kk_euclid.npairs[half:])

    # Renormalize by the actual spacing in log(r)
    renorm_euclid = kk_euclid.npairs / np.gradient(kk_euclid.meanlogr)
    renorm_sphere = kk_sphere.npairs / np.gradient(kk_sphere.meanlogr)
    # Then interpolate the euclid results to the values of the sphere distances
    interp_euclid = np.interp(kk_sphere.meanlogr, kk_euclid.meanlogr, renorm_euclid)
    # Matches at 3e-3 over whole range now.
    print('interp_euclid = ',interp_euclid)
    print('renorm_sphere = ',renorm_sphere)
    print('new diff = ',(renorm_sphere-interp_euclid)/renorm_sphere)
    print('max = ',np.max(np.abs((renorm_sphere-interp_euclid)/renorm_sphere)))
    np.testing.assert_allclose(renorm_sphere, interp_euclid, rtol=3.e-3*tol)

    # And almost the full range at the same precision.
    np.testing.assert_allclose(renorm_sphere[:-4], interp_euclid[:-4], rtol=2.e-3*tol)
    np.testing.assert_allclose(renorm_sphere, interp_euclid, rtol=1.e-2*tol)

    # The xi values are similar.  The euclid and chord values start out basically identical,
    # but the distances are different.  The euclid and the sphere are actually the same function
    # so they match when rescaled to have the same distance values.
    print('diff euclid, chord = ',(kk_chord.xi-kk_euclid.xi)/kk_euclid.xi)
    print('max = ',np.max(np.abs((kk_chord.xi-kk_euclid.xi)/kk_euclid.xi)))
    np.testing.assert_allclose(kk_chord.xi[:-8], kk_euclid.xi[:-8], rtol=1.e-3*tol)
    np.testing.assert_allclose(kk_chord.xi, kk_euclid.xi, rtol=3.e-3*tol)

    interp_euclid = np.interp(kk_sphere.meanlogr, kk_euclid.meanlogr, kk_euclid.xi)
    print('interp_euclid = ',interp_euclid)
    print('sphere.xi = ',kk_sphere.xi)
    print('diff interp euclid, sphere = ',(kk_sphere.xi-interp_euclid))
    print('max = ',np.max(np.abs((kk_sphere.xi-interp_euclid))))
    np.testing.assert_allclose(kk_sphere.xi, interp_euclid, atol=1.e-3*tol)

@timer
def test_varxi():
    # Test that varxi is correct (or close) based on actual variance of many runs.

    # Signal doesn't matter much.  Use the one from test_gg.
    kappa0 = 0.03
    r0 = 10.
    L = 50.*r0
    rng = np.random.RandomState(8675309)

    # Note: to get a good estimate of var(xi), you need a lot of runs.  The number of
    # runs matters much more than the number of galaxies for getting this to pass.
    ngal = 1000
    nruns = 50000

    file_name = 'data/test_varxi_kk.npz'
    print(file_name)
    if not os.path.isfile(file_name):
        all_kks = []
        for run in range(nruns):
            print(f'{run}/{nruns}')
            x = (rng.random_sample(ngal)-0.5) * L
            y = (rng.random_sample(ngal)-0.5) * L
            # Varied weights are hard, but at least check that non-unit weights work correctly.
            w = np.ones_like(x) * 5
            r2 = (x**2 + y**2)/r0**2
            k = kappa0 * np.exp(-r2/2.)
            k += rng.normal(0, 0.1, size=ngal)

            cat = treecorr.Catalog(x=x, y=y, w=w, k=k)
            kk = treecorr.KKCorrelation(bin_size=0.1, min_sep=10., max_sep=100.)
            kk.process(cat)
            all_kks.append(kk)

        mean_xi = np.mean([kk.xi for kk in all_kks], axis=0)
        var_xi = np.var([kk.xi for kk in all_kks], axis=0)
        mean_varxi = np.mean([kk.varxi for kk in all_kks], axis=0)

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
    w = np.ones_like(x) * 5
    r2 = (x**2 + y**2)/r0**2
    k = kappa0 * np.exp(-r2/2.)
    k += rng.normal(0, 0.1, size=ngal)

    cat = treecorr.Catalog(x=x, y=y, w=w, k=k)
    kk = treecorr.KKCorrelation(bin_size=0.1, min_sep=10., max_sep=100.)
    kk.process(cat)

    print('single run:')
    print('ratio = ',kk.varxi / var_xi)
    print('max relerr for xi = ',np.max(np.abs((kk.varxi - var_xi)/var_xi)))
    np.testing.assert_allclose(kk.varxi, var_xi, rtol=0.3)



if __name__ == '__main__':
    test_direct()
    test_direct_spherical()
    test_pairwise()
    test_constant()
    test_kk()
    test_large_scale()
    test_varxi()
