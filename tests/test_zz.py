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
import os
import time
import coord
import treecorr

from test_helper import get_from_wiki, do_pickle, CaptureLog
from test_helper import assert_raises, timer, assert_warns


@timer
def test_direct():
    # If the catalogs are small enough, we can do a direct calculation to see if comes out right.
    # This should exactly match the treecorr result if brute_force=True

    ngal = 200
    s = 10.
    rng = np.random.RandomState(8675309)
    x1 = rng.normal(0,s, (ngal,) )
    y1 = rng.normal(0,s, (ngal,) )
    w1 = rng.random_sample(ngal)
    z11 = rng.normal(0,0.2, (ngal,) )
    z21 = rng.normal(0,0.2, (ngal,) )

    x2 = rng.normal(0,s, (ngal,) )
    y2 = rng.normal(0,s, (ngal,) )
    w2 = rng.random_sample(ngal)
    z12 = rng.normal(0,0.2, (ngal,) )
    z22 = rng.normal(0,0.2, (ngal,) )

    cat1 = treecorr.Catalog(x=x1, y=y1, w=w1, z1=z11, z2=z21)
    cat2 = treecorr.Catalog(x=x2, y=y2, w=w2, z1=z12, z2=z22)

    min_sep = 1.
    max_sep = 50.
    nbins = 50
    bin_size = np.log(max_sep/min_sep) / nbins
    zz = treecorr.ZZCorrelation(min_sep=min_sep, max_sep=max_sep, nbins=nbins, brute=True)
    zz.process(cat1, cat2)

    true_npairs = np.zeros(nbins, dtype=int)
    true_weight = np.zeros(nbins, dtype=float)
    true_xip = np.zeros(nbins, dtype=complex)
    true_xim = np.zeros(nbins, dtype=complex)
    for i in range(ngal):
        # It's hard to do all the pairs at once with numpy operations (although maybe possible).
        # But we can at least do all the pairs for each entry in cat1 at once with arrays.
        rsq = (x1[i]-x2)**2 + (y1[i]-y2)**2
        r = np.sqrt(rsq)

        ww = w1[i] * w2
        xip = ww * (z11[i] + 1j*z21[i]) * (z12 - 1j*z22)
        xim = ww * (z11[i] + 1j*z21[i]) * (z12 + 1j*z22)

        index = np.floor(np.log(r/min_sep) / bin_size).astype(int)
        mask = (index >= 0) & (index < nbins)
        np.add.at(true_npairs, index[mask], 1)
        np.add.at(true_weight, index[mask], ww[mask])
        np.add.at(true_xip, index[mask], xip[mask])
        np.add.at(true_xim, index[mask], xim[mask])

    true_xip /= true_weight
    true_xim /= true_weight

    print('true_npairs = ',true_npairs)
    print('diff = ',zz.npairs - true_npairs)
    np.testing.assert_array_equal(zz.npairs, true_npairs)

    print('true_weight = ',true_weight)
    print('diff = ',zz.weight - true_weight)
    np.testing.assert_allclose(zz.weight, true_weight, rtol=1.e-6, atol=1.e-8)

    print('true_xip = ',true_xip)
    print('zz.xip = ',zz.xip)
    print('zz.xip_im = ',zz.xip_im)
    np.testing.assert_allclose(zz.xip, true_xip.real, rtol=1.e-6, atol=1.e-8)
    np.testing.assert_allclose(zz.xip_im, true_xip.imag, rtol=1.e-6, atol=1.e-8)
    print('true_xim = ',true_xim)
    print('zz.xim = ',zz.xim)
    print('zz.xim_im = ',zz.xim_im)
    np.testing.assert_allclose(zz.xim, true_xim.real, rtol=1.e-6, atol=1.e-8)
    np.testing.assert_allclose(zz.xim_im, true_xim.imag, rtol=1.e-6, atol=1.e-8)

    # Check that running via the corr2 script works correctly.
    config = treecorr.config.read_config('configs/zz_direct.yaml')
    try:
        import fitsio
    except ImportError:
        pass
    else:
        cat1.write(config['file_name'])
        cat2.write(config['file_name2'])
        treecorr.corr2(config)
        data = fitsio.read(config['zz_file_name'])
        np.testing.assert_allclose(data['r_nom'], zz.rnom)
        np.testing.assert_allclose(data['npairs'], zz.npairs)
        np.testing.assert_allclose(data['weight'], zz.weight)
        np.testing.assert_allclose(data['xip'], zz.xip)
        np.testing.assert_allclose(data['xip_im'], zz.xip_im)
        np.testing.assert_allclose(data['xim'], zz.xim)
        np.testing.assert_allclose(data['xim_im'], zz.xim_im)

    # Repeat with binslop = 0.
    # And don't do any top-level recursion so we actually test not going to the leaves.
    zz = treecorr.ZZCorrelation(min_sep=min_sep, max_sep=max_sep, nbins=nbins, bin_slop=0,
                                max_top=0)
    zz.process(cat1, cat2)
    np.testing.assert_array_equal(zz.npairs, true_npairs)
    np.testing.assert_allclose(zz.weight, true_weight, rtol=1.e-6, atol=1.e-8)
    np.testing.assert_allclose(zz.xip, true_xip.real, rtol=1.e-6, atol=1.e-8)
    np.testing.assert_allclose(zz.xip_im, true_xip.imag, rtol=1.e-6, atol=1.e-8)
    print('true_xim = ',true_xim)
    print('zz.xim = ',zz.xim)
    print('zz.xim_im = ',zz.xim_im)
    print('diff = ',zz.xim - true_xim.real)
    print('max diff = ',np.max(np.abs(zz.xim - true_xim.real)))
    print('rel diff = ',(zz.xim - true_xim.real)/true_xim.real)
    np.testing.assert_allclose(zz.xim, true_xim.real, atol=3.e-4)
    np.testing.assert_allclose(zz.xim_im, true_xim.imag, atol=1.e-4)

    # With angle_slop = 0, it goes back to being basically exact (to single precision).
    zz = treecorr.ZZCorrelation(min_sep=min_sep, max_sep=max_sep, nbins=nbins, bin_slop=0,
                                angle_slop=0, max_top=0)
    zz.process(cat1, cat2)
    np.testing.assert_array_equal(zz.npairs, true_npairs)
    np.testing.assert_allclose(zz.weight, true_weight, rtol=1.e-6, atol=1.e-8)
    np.testing.assert_allclose(zz.xip, true_xip.real, rtol=1.e-6, atol=1.e-8)
    np.testing.assert_allclose(zz.xip_im, true_xip.imag, rtol=1.e-6, atol=1.e-8)
    np.testing.assert_allclose(zz.xim, true_xim.real, rtol=1.e-6, atol=1.e-8)
    np.testing.assert_allclose(zz.xim_im, true_xim.imag, rtol=1.e-6, atol=1.e-8)

    # Check a few basic operations with a ZZCorrelation object.
    do_pickle(zz)

    zz2 = zz.copy()
    zz2 += zz
    np.testing.assert_allclose(zz2.npairs, 2*zz.npairs)
    np.testing.assert_allclose(zz2.weight, 2*zz.weight)
    np.testing.assert_allclose(zz2.meanr, 2*zz.meanr)
    np.testing.assert_allclose(zz2.meanlogr, 2*zz.meanlogr)
    np.testing.assert_allclose(zz2.xip, 2*zz.xip)
    np.testing.assert_allclose(zz2.xip_im, 2*zz.xip_im)
    np.testing.assert_allclose(zz2.xim, 2*zz.xim)
    np.testing.assert_allclose(zz2.xim_im, 2*zz.xim_im)

    zz2.clear()
    zz2 += zz
    np.testing.assert_allclose(zz2.npairs, zz.npairs)
    np.testing.assert_allclose(zz2.weight, zz.weight)
    np.testing.assert_allclose(zz2.meanr, zz.meanr)
    np.testing.assert_allclose(zz2.meanlogr, zz.meanlogr)
    np.testing.assert_allclose(zz2.xip, zz.xip)
    np.testing.assert_allclose(zz2.xip_im, zz.xip_im)
    np.testing.assert_allclose(zz2.xim, zz.xim)
    np.testing.assert_allclose(zz2.xim_im, zz.xim_im)

    ascii_name = 'output/zz_ascii.txt'
    zz.write(ascii_name, precision=16)
    zz3 = treecorr.ZZCorrelation(min_sep=min_sep, max_sep=max_sep, nbins=nbins, bin_type='Log')
    zz3.read(ascii_name)
    np.testing.assert_allclose(zz3.npairs, zz.npairs)
    np.testing.assert_allclose(zz3.weight, zz.weight)
    np.testing.assert_allclose(zz3.meanr, zz.meanr)
    np.testing.assert_allclose(zz3.meanlogr, zz.meanlogr)
    np.testing.assert_allclose(zz3.xip, zz.xip)
    np.testing.assert_allclose(zz3.xip_im, zz.xip_im)
    np.testing.assert_allclose(zz3.xim, zz.xim)
    np.testing.assert_allclose(zz3.xim_im, zz.xim_im)

    # Check that the repr is minimal
    assert repr(zz3) == f'ZZCorrelation(min_sep={min_sep}, max_sep={max_sep}, nbins={nbins})'

    # Simpler API using from_file:
    with CaptureLog() as cl:
        zz3b = treecorr.ZZCorrelation.from_file(ascii_name, logger=cl.logger)
    assert ascii_name in cl.output
    np.testing.assert_allclose(zz3b.npairs, zz.npairs)
    np.testing.assert_allclose(zz3b.weight, zz.weight)
    np.testing.assert_allclose(zz3b.meanr, zz.meanr)
    np.testing.assert_allclose(zz3b.meanlogr, zz.meanlogr)
    np.testing.assert_allclose(zz3b.xip, zz.xip)
    np.testing.assert_allclose(zz3b.xip_im, zz.xip_im)
    np.testing.assert_allclose(zz3b.xim, zz.xim)
    np.testing.assert_allclose(zz3b.xim_im, zz.xim_im)

    # or using the Corr2 base class
    with CaptureLog() as cl:
        zz3c = treecorr.Corr2.from_file(ascii_name, logger=cl.logger)
    assert ascii_name in cl.output
    np.testing.assert_allclose(zz3c.npairs, zz.npairs)
    np.testing.assert_allclose(zz3c.weight, zz.weight)
    np.testing.assert_allclose(zz3c.meanr, zz.meanr)
    np.testing.assert_allclose(zz3c.meanlogr, zz.meanlogr)
    np.testing.assert_allclose(zz3c.xip, zz.xip)
    np.testing.assert_allclose(zz3c.xip_im, zz.xip_im)
    np.testing.assert_allclose(zz3c.xim, zz.xim)
    np.testing.assert_allclose(zz3c.xim_im, zz.xim_im)

    try:
        import fitsio
    except ImportError:
        pass
    else:
        fits_name = 'output/zz_fits.fits'
        zz.write(fits_name)
        zz4 = treecorr.ZZCorrelation(min_sep=min_sep, max_sep=max_sep, nbins=nbins)
        zz4.read(fits_name)
        np.testing.assert_allclose(zz4.npairs, zz.npairs)
        np.testing.assert_allclose(zz4.weight, zz.weight)
        np.testing.assert_allclose(zz4.meanr, zz.meanr)
        np.testing.assert_allclose(zz4.meanlogr, zz.meanlogr)
        np.testing.assert_allclose(zz4.xip, zz.xip)
        np.testing.assert_allclose(zz4.xip_im, zz.xip_im)
        np.testing.assert_allclose(zz4.xim, zz.xim)
        np.testing.assert_allclose(zz4.xim_im, zz.xim_im)

        zz4b = treecorr.ZZCorrelation.from_file(fits_name)
        np.testing.assert_allclose(zz4b.npairs, zz.npairs)
        np.testing.assert_allclose(zz4b.weight, zz.weight)
        np.testing.assert_allclose(zz4b.meanr, zz.meanr)
        np.testing.assert_allclose(zz4b.meanlogr, zz.meanlogr)
        np.testing.assert_allclose(zz4b.xip, zz.xip)
        np.testing.assert_allclose(zz4b.xip_im, zz.xip_im)
        np.testing.assert_allclose(zz4b.xim, zz.xim)
        np.testing.assert_allclose(zz4b.xim_im, zz.xim_im)

    with assert_raises(TypeError):
        zz2 += config
    zz4 = treecorr.ZZCorrelation(min_sep=min_sep/2, max_sep=max_sep, nbins=nbins)
    with assert_raises(ValueError):
        zz2 += zz4
    zz5 = treecorr.ZZCorrelation(min_sep=min_sep, max_sep=max_sep*2, nbins=nbins)
    with assert_raises(ValueError):
        zz2 += zz5
    zz6 = treecorr.ZZCorrelation(min_sep=min_sep, max_sep=max_sep, nbins=nbins*2)
    with assert_raises(ValueError):
        zz2 += zz6
    with assert_raises(ValueError):
        zz.process(cat1, cat2, patch_method='nonlocal')

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
    z11 = rng.normal(0,0.2, (ngal,) )
    z21 = rng.normal(0,0.2, (ngal,) )

    x2 = rng.normal(0,s, (ngal,) )
    y2 = rng.normal(0,s, (ngal,) ) + 200
    z2 = rng.normal(0,s, (ngal,) )
    w2 = rng.random_sample(ngal)
    z12 = rng.normal(0,0.2, (ngal,) )
    z22 = rng.normal(0,0.2, (ngal,) )

    ra1, dec1 = coord.CelestialCoord.xyz_to_radec(x1,y1,z1)
    ra2, dec2 = coord.CelestialCoord.xyz_to_radec(x2,y2,z2)

    cat1 = treecorr.Catalog(ra=ra1, dec=dec1, ra_units='rad', dec_units='rad', w=w1, z1=z11, z2=z21)
    cat2 = treecorr.Catalog(ra=ra2, dec=dec2, ra_units='rad', dec_units='rad', w=w2, z1=z12, z2=z22)

    min_sep = 1.
    max_sep = 10.
    nbins = 50
    bin_size = np.log(max_sep/min_sep) / nbins
    zz = treecorr.ZZCorrelation(min_sep=min_sep, max_sep=max_sep, nbins=nbins,
                                sep_units='deg', brute=True)
    zz.process(cat1, cat2)

    r1 = np.sqrt(x1**2 + y1**2 + z1**2)
    r2 = np.sqrt(x2**2 + y2**2 + z2**2)
    x1 /= r1;  y1 /= r1;  z1 /= r1
    x2 /= r2;  y2 /= r2;  z2 /= r2

    north_pole = coord.CelestialCoord(0*coord.radians, 90*coord.degrees)

    true_npairs = np.zeros(nbins, dtype=int)
    true_weight = np.zeros(nbins, dtype=float)
    true_xip = np.zeros(nbins, dtype=complex)
    true_xim = np.zeros(nbins, dtype=complex)

    rad_min_sep = min_sep * coord.degrees / coord.radians
    c1 = [coord.CelestialCoord(r*coord.radians, d*coord.radians) for (r,d) in zip(ra1, dec1)]
    c2 = [coord.CelestialCoord(r*coord.radians, d*coord.radians) for (r,d) in zip(ra2, dec2)]
    for i in range(ngal):
        for j in range(ngal):
            rsq = (x1[i]-x2[j])**2 + (y1[i]-y2[j])**2 + (z1[i]-z2[j])**2
            r = np.sqrt(rsq)

            index = np.floor(np.log(r/rad_min_sep) / bin_size).astype(int)
            if index < 0 or index >= nbins:
                continue

            zz1 = z11[i] + 1j * z21[i]
            zz2 = z12[j] + 1j * z22[j]

            ww = w1[i] * w2[j]
            xip = ww * zz1 * np.conjugate(zz2)
            xim = ww * zz1 * zz2

            true_npairs[index] += 1
            true_weight[index] += ww
            true_xip[index] += xip
            true_xim[index] += xim

    true_xip /= true_weight
    true_xim /= true_weight

    print('true_npairs = ',true_npairs)
    print('diff = ',zz.npairs - true_npairs)
    np.testing.assert_array_equal(zz.npairs, true_npairs)

    print('true_weight = ',true_weight)
    print('diff = ',zz.weight - true_weight)
    np.testing.assert_allclose(zz.weight, true_weight, rtol=1.e-6, atol=1.e-8)

    print('true_xip = ',true_xip)
    print('zz.xip = ',zz.xip)
    print('zz.xip_im = ',zz.xip_im)
    np.testing.assert_allclose(zz.xip, true_xip.real, rtol=1.e-6, atol=1.e-8)
    np.testing.assert_allclose(zz.xip_im, true_xip.imag, rtol=1.e-6, atol=1.e-8)
    print('true_xim = ',true_xim)
    print('zz.xim = ',zz.xim)
    print('zz.xim_im = ',zz.xim_im)
    np.testing.assert_allclose(zz.xim, true_xim.real, rtol=1.e-6, atol=1.e-8)
    np.testing.assert_allclose(zz.xim_im, true_xim.imag, rtol=1.e-6, atol=1.e-8)

    # Check that running via the corr2 script works correctly.
    try:
        import fitsio
    except ImportError:
        pass
    else:
        config = treecorr.config.read_config('configs/zz_direct_spherical.yaml')
        cat1.write(config['file_name'])
        cat2.write(config['file_name2'])
        treecorr.corr2(config)
        data = fitsio.read(config['zz_file_name'])
        np.testing.assert_allclose(data['r_nom'], zz.rnom)
        np.testing.assert_allclose(data['npairs'], zz.npairs)
        np.testing.assert_allclose(data['weight'], zz.weight)
        np.testing.assert_allclose(data['xip'], zz.xip)
        np.testing.assert_allclose(data['xip_im'], zz.xip_im)
        np.testing.assert_allclose(data['xim'], zz.xim)
        np.testing.assert_allclose(data['xim_im'], zz.xim_im)

    # Repeat with binslop = 0
    # And don't do any top-level recursion so we actually test not going to the leaves.
    zz = treecorr.ZZCorrelation(min_sep=min_sep, max_sep=max_sep, nbins=nbins,
                                sep_units='deg', bin_slop=0, max_top=0)
    zz.process(cat1, cat2)
    np.testing.assert_array_equal(zz.npairs, true_npairs)
    np.testing.assert_allclose(zz.weight, true_weight, rtol=1.e-6, atol=1.e-8)
    np.testing.assert_allclose(zz.xip, true_xip.real, rtol=1.e-6, atol=3.e-7)
    np.testing.assert_allclose(zz.xip_im, true_xip.imag, rtol=1.e-6, atol=2.e-7)
    np.testing.assert_allclose(zz.xim, true_xim.real, atol=1.e-4)
    np.testing.assert_allclose(zz.xim_im, true_xim.imag, atol=1.e-4)

    # With angle_slop = 0, it goes back to being basically exact (to single precision).
    zz = treecorr.ZZCorrelation(min_sep=min_sep, max_sep=max_sep, nbins=nbins,
                                sep_units='deg', bin_slop=0, angle_slop=0, max_top=0)
    zz.process(cat1, cat2)
    np.testing.assert_array_equal(zz.npairs, true_npairs)
    np.testing.assert_allclose(zz.weight, true_weight, rtol=1.e-6, atol=1.e-8)
    np.testing.assert_allclose(zz.xip, true_xip.real, rtol=1.e-6, atol=1.e-8)
    np.testing.assert_allclose(zz.xip_im, true_xip.imag, rtol=1.e-6, atol=1.e-8)
    np.testing.assert_allclose(zz.xim, true_xim.real, rtol=1.e-6, atol=1.e-8)
    np.testing.assert_allclose(zz.xim_im, true_xim.imag, rtol=1.e-6, atol=1.e-8)


@timer
def test_zz():
    # Similar to the math in test_gg(), but use a functional form that has a radial vector,
    # rather than radial shear pattern.
    # Also, the xi- integral here uses J2, not J4.

    # Use z(r) = z0 exp(-r^2/2r0^2)
    #
    # The Fourier transform is: z~(k) = 2 pi z0 r0^2 exp(-r0^2 k^2/2) / L^2
    # P(k) = (1/2pi) <|z~(k)|^2> = 2 pi |z0|^2 (r0/L)^4 exp(-r0^2 k^2)
    # xi+(r) = (1/2pi) int( dk k P(k) J0(kr) )
    #        = pi |z0|^2 (r0/L)^2 exp(-r^2/4r0^2)
    # xi-(r) = (1/2pi) int( dk k P(k) J2(kr) )
    #        = pi z0^2 (r0/L)^2 exp(-r^2/4r0^2)

    z0 = 0.05 + 1j * 0.03
    r0 = 10.
    if __name__ == "__main__":
        ngal = 1000000
        L = 50.*r0  # Not infinity, so this introduces some error.  Our integrals were to infinity.
        tol_factor = 1
    else:
        ngal = 100000
        L = 50.*r0
        # Rather than have a single set tolerance, we tune the tolerances for the above
        # __main__ setup, but scale up by a factor of 5 for the quicker run.
        tol_factor = 5
    rng = np.random.RandomState(8675309)
    x = (rng.random_sample(ngal)-0.5) * L
    y = (rng.random_sample(ngal)-0.5) * L
    r2 = (x**2 + y**2)/r0**2
    zz = z0 * np.exp(-r2/2.)
    z1 = np.real(zz)
    z2 = np.imag(zz)

    cat = treecorr.Catalog(x=x, y=y, z1=z1, z2=z2, x_units='arcmin', y_units='arcmin')
    zz = treecorr.ZZCorrelation(bin_size=0.1, min_sep=1., max_sep=100., sep_units='arcmin',
                                verbose=1)
    zz.process(cat)

    # log(<R>) != <logR>, but it should be close:
    print('meanlogr - log(meanr) = ',zz.meanlogr - np.log(zz.meanr))
    np.testing.assert_allclose(zz.meanlogr, np.log(zz.meanr), atol=1.e-3)

    r = zz.meanr
    temp = np.pi * (r0/L)**2 * np.exp(-0.25*r**2/r0**2)
    true_xip = temp * np.abs(z0**2)
    true_xim = temp * z0**2

    print('zz.xip = ',zz.xip)
    print('true_xip = ',true_xip)
    print('ratio = ',zz.xip / true_xip)
    print('diff = ',zz.xip - true_xip)
    print('max diff = ',max(abs(zz.xip - true_xip)))
    # It's within 10% everywhere except at the zero crossings.
    np.testing.assert_allclose(zz.xip, true_xip, rtol=0.1 * tol_factor, atol=1.e-7 * tol_factor)
    print('xip_im = ',zz.xip_im)
    np.testing.assert_allclose(zz.xip_im, 0, atol=2.e-7 * tol_factor)

    print('zz.xim = ',zz.xim)
    print('true_xim = ',true_xim)
    print('ratio = ',zz.xim / true_xim)
    print('diff = ',zz.xim - true_xim)
    print('max diff = ',max(abs(zz.xim - true_xim)))
    np.testing.assert_allclose(zz.xim, np.real(true_xim), rtol=0.1 * tol_factor, atol=2.e-7 * tol_factor)
    print('xim_im = ',zz.xim_im)
    np.testing.assert_allclose(zz.xim_im, np.imag(true_xim), rtol=0.1 * tol_factor, atol=2.e-7 * tol_factor)

    # Should also work as a cross-correlation with itself
    zz.process(cat,cat)
    np.testing.assert_allclose(zz.meanlogr, np.log(zz.meanr), atol=1.e-3)
    assert max(abs(zz.xip - true_xip)) < 3.e-7 * tol_factor
    assert max(abs(zz.xip_im)) < 2.e-7 * tol_factor
    assert max(abs(zz.xim - np.real(true_xim))) < 3.e-7 * tol_factor
    assert max(abs(zz.xim_im - np.imag(true_xim))) < 3.e-7 * tol_factor

    # Check that we get the same result using the corr2 function:
    cat.write(os.path.join('data','zz.dat'))
    config = treecorr.read_config('configs/zz.yaml')
    config['verbose'] = 0
    config['precision'] = 8
    treecorr.corr2(config)
    corr2_output = np.genfromtxt(os.path.join('output','zz.out'), names=True, skip_header=1)
    np.testing.assert_allclose(corr2_output['xip'], zz.xip, rtol=1.e-4)
    np.testing.assert_allclose(corr2_output['xim'], zz.xim, rtol=1.e-4)
    np.testing.assert_allclose(corr2_output['xip_im'], zz.xip_im, rtol=1.e-4, atol=1.e-12)
    np.testing.assert_allclose(corr2_output['xim_im'], zz.xim_im, rtol=1.e-4)

    # Check the fits write option
    out_file_name = os.path.join('output','zz_out.dat')
    zz.write(out_file_name, precision=16)
    data = np.genfromtxt(out_file_name, names=True, skip_header=1)
    np.testing.assert_allclose(data['r_nom'], np.exp(zz.logr))
    np.testing.assert_allclose(data['meanr'], zz.meanr)
    np.testing.assert_allclose(data['meanlogr'], zz.meanlogr)
    np.testing.assert_allclose(data['xip'], zz.xip)
    np.testing.assert_allclose(data['xim'], zz.xim)
    np.testing.assert_allclose(data['xip_im'], zz.xip_im)
    np.testing.assert_allclose(data['xim_im'], zz.xim_im)
    np.testing.assert_allclose(data['sigma_xip'], np.sqrt(zz.varxip))
    np.testing.assert_allclose(data['sigma_xim'], np.sqrt(zz.varxim))
    np.testing.assert_allclose(data['weight'], zz.weight)
    np.testing.assert_allclose(data['npairs'], zz.npairs)

    # Check the read function
    zz2 = treecorr.ZZCorrelation.from_file(out_file_name)
    np.testing.assert_allclose(zz2.logr, zz.logr)
    np.testing.assert_allclose(zz2.meanr, zz.meanr)
    np.testing.assert_allclose(zz2.meanlogr, zz.meanlogr)
    np.testing.assert_allclose(zz2.xip, zz.xip)
    np.testing.assert_allclose(zz2.xim, zz.xim)
    np.testing.assert_allclose(zz2.xip_im, zz.xip_im)
    np.testing.assert_allclose(zz2.xim_im, zz.xim_im)
    np.testing.assert_allclose(zz2.varxip, zz.varxip)
    np.testing.assert_allclose(zz2.varxim, zz.varxim)
    np.testing.assert_allclose(zz2.weight, zz.weight)
    np.testing.assert_allclose(zz2.npairs, zz.npairs)
    assert zz2.coords == zz.coords
    assert zz2.metric == zz.metric
    assert zz2.sep_units == zz.sep_units
    assert zz2.bin_type == zz.bin_type


@timer
def test_varxi():
    # Test that varxip, varxim are correct (or close) based on actual variance of many runs.

    # Same v pattern as in test_zz().  Although the signal doesn't actually matter at all here.
    z0 = 0.05 + 1j*0.05
    r0 = 10.
    L = 50.*r0
    rng = np.random.RandomState(8675309)

    # Note: to get a good estimate of var(xi), you need a lot of runs.  The number of
    # runs matters much more than the number of galaxies for getting this to pass.
    ngal = 1000
    nruns = 50000

    file_name = 'data/test_varxi_zz.npz'
    print(file_name)
    if not os.path.isfile(file_name):
        all_zzs = []

        for run in range(nruns):
            print(f'{run}/{nruns}')
            # In addition to the shape noise below, there is shot noise from the random x,y positions.
            x = (rng.random_sample(ngal)-0.5) * L
            y = (rng.random_sample(ngal)-0.5) * L
            # Varied weights are hard, but at least check that non-unit weights work correctly.
            w = np.ones_like(x) * 5
            r2 = (x**2 + y**2)/r0**2
            zz = z0 * np.exp(-r2/2.)
            z1 = np.real(zz)
            z2 = np.imag(zz)
            # This time, add some shape noise (different each run).
            z1 += rng.normal(0, 0.3, size=ngal)
            z2 += rng.normal(0, 0.3, size=ngal)

            cat = treecorr.Catalog(x=x, y=y, w=w, z1=z1, z2=z2, x_units='arcmin', y_units='arcmin')
            zz = treecorr.ZZCorrelation(bin_size=0.1, min_sep=10., max_sep=100., sep_units='arcmin',
                                        verbose=1)
            zz.process(cat)
            all_zzs.append(zz)

        mean_xip = np.mean([zz.xip for zz in all_zzs], axis=0)
        var_xip = np.var([zz.xip for zz in all_zzs], axis=0)
        mean_xim = np.mean([zz.xim for zz in all_zzs], axis=0)
        var_xim = np.var([zz.xim for zz in all_zzs], axis=0)
        mean_varxip = np.mean([zz.varxip for zz in all_zzs], axis=0)
        mean_varxim = np.mean([zz.varxim for zz in all_zzs], axis=0)

        np.savez(file_name,
                 mean_xip=mean_xip, var_xip=var_xip, mean_varxip=mean_varxip,
                 mean_xim=mean_xim, var_xim=var_xim, mean_varxim=mean_varxim)

    data = np.load(file_name)
    mean_xip = data['mean_xip']
    var_xip = data['var_xip']
    mean_varxip = data['mean_varxip']
    mean_xim = data['mean_xim']
    var_xim = data['var_xim']
    mean_varxim = data['mean_varxim']

    print('nruns = ',nruns)
    print('mean_xip = ',mean_xip)
    print('mean_xim = ',mean_xim)
    print('mean_varxip = ',mean_varxip)
    print('mean_varxim = ',mean_varxim)
    print('var_xip = ',var_xip)
    print('ratio = ',var_xip / mean_varxip)
    print('var_xim = ',var_xim)
    print('ratio = ',var_xim / mean_varxim)
    print('max relerr for xip = ',np.max(np.abs((var_xip - mean_varxip)/var_xip)))
    print('max relerr for xim = ',np.max(np.abs((var_xim - mean_varxim)/var_xim)))
    np.testing.assert_allclose(mean_varxip, var_xip, rtol=0.02)
    np.testing.assert_allclose(mean_varxim, var_xim, rtol=0.02)

    # Now the actual test that's based on current code, not just from the saved file.
    # There is a bit more noise on a singe run, so the tolerance needs to be somewhat higher.
    x = (rng.random_sample(ngal)-0.5) * L
    y = (rng.random_sample(ngal)-0.5) * L
    # Varied weights are hard, but at least check that non-unit weights work correctly.
    w = np.ones_like(x) * 5
    r2 = (x**2 + y**2)/r0**2
    zz = z0 * np.exp(-r2/2.)
    z1 = np.real(zz)
    z2 = np.imag(zz)
    # This time, add some shape noise (different each run).
    z1 += rng.normal(0, 0.3, size=ngal)
    z2 += rng.normal(0, 0.3, size=ngal)

    cat = treecorr.Catalog(x=x, y=y, w=w, z1=z1, z2=z2, x_units='arcmin', y_units='arcmin')
    zz = treecorr.ZZCorrelation(bin_size=0.1, min_sep=10., max_sep=100., sep_units='arcmin',
                                verbose=1)
    zz.process(cat)
    print('single run:')
    print('max relerr for xip = ',np.max(np.abs((zz.varxip - var_xip)/var_xip)))
    print('max relerr for xim = ',np.max(np.abs((zz.varxip - var_xim)/var_xim)))
    np.testing.assert_allclose(zz.varxip, var_xip, rtol=0.3)
    np.testing.assert_allclose(zz.varxim, var_xim, rtol=0.3)

@timer
def test_jk():

    # Same multi-lens field we used for NV patch test
    r0 = 30.
    L = 30 * r0
    rng = np.random.RandomState(8675309)

    nsource = 100000
    nlens = 300
    nruns = 1000
    npatch = 64

    corr_params = dict(bin_size=0.3, min_sep=20, max_sep=50, bin_slop=0.1)

    def make_field(rng):
        x1 = (rng.random(nlens)-0.5) * L
        y1 = (rng.random(nlens)-0.5) * L
        w = rng.random(nlens) + 10
        x2 = (rng.random(nsource)-0.5) * L
        y2 = (rng.random(nsource)-0.5) * L

        # Start with just the noise
        z1 = rng.normal(0, 0.2, size=nsource)
        z2 = rng.normal(0, 0.2, size=nsource)

        # Add in the signal from all lenses
        for i in range(nlens):
            x2i = x2 - x1[i]
            y2i = y2 - y1[i]
            r2 = (x2i**2 + y2i**2)/r0**2
            z0 = rng.normal(0, 0.03) + 1j * rng.normal(0, 0.03)
            zz = w[i] * z0 * np.exp(-r2/2.)
            z1 += np.real(zz)
            z2 += np.imag(zz)
        return x1, y1, w, x2, y2, z1, z2

    file_name = 'data/test_zz_jk_{}.npz'.format(nruns)
    print(file_name)
    if not os.path.isfile(file_name):
        all_zzs = []
        rng = np.random.default_rng()
        for run in range(nruns):
            x1, y1, w, x2, y2, z1, z2 = make_field(rng)
            print(run,': ',np.mean(z1),np.std(z1),np.min(z1),np.max(z1))
            cat = treecorr.Catalog(x=x2, y=y2, z1=z1, z2=z2)
            zz = treecorr.ZZCorrelation(corr_params)
            zz.process(cat)
            all_zzs.append(zz)

        print('xip = ',np.array([zz.xip for zz in all_zzs]))
        mean_xip = np.mean([zz.xip for zz in all_zzs], axis=0)
        mean_xim = np.mean([zz.xim for zz in all_zzs], axis=0)
        var_xip = np.var([zz.xip for zz in all_zzs], axis=0)
        var_xim = np.var([zz.xim for zz in all_zzs], axis=0)
        mean_varxip = np.mean([zz.varxip for zz in all_zzs], axis=0)
        mean_varxim = np.mean([zz.varxim for zz in all_zzs], axis=0)

        np.savez(file_name,
                 mean_xip=mean_xip, var_xip=var_xip, mean_varxip=mean_varxip,
                 mean_xim=mean_xim, var_xim=var_xim, mean_varxim=mean_varxim)

    data = np.load(file_name)
    mean_xip = data['mean_xip']
    mean_xim = data['mean_xim']
    mean_varxip = data['mean_varxip']
    mean_varxim = data['mean_varxim']
    var_xip = data['var_xip']
    var_xim = data['var_xim']

    print('mean_xip = ',mean_xip)
    print('mean_varxip = ',mean_varxip)
    print('var_xip = ',var_xip)
    print('ratio = ',var_xip / mean_varxip)
    print('mean_xim = ',mean_xim)
    print('mean_varxim = ',mean_varxim)
    print('var_xim = ',var_xim)
    print('ratio = ',var_xim / mean_varxim)

    rng = np.random.default_rng(1234)
    x1, y1, w, x2, y2, z1, z2 = make_field(rng)

    cat = treecorr.Catalog(x=x2, y=y2, z1=z1, z2=z2)
    zz1 = treecorr.ZZCorrelation(corr_params)
    t0 = time.time()
    zz1.process(cat)
    t1 = time.time()
    print('Time for non-patch processing = ',t1-t0)

    print('weight = ',zz1.weight)
    print('xip = ',zz1.xip)
    print('varxip = ',zz1.varxip)
    print('pullsq for xip = ',(zz1.xip-mean_xip)**2/var_xip)
    print('max pull for xip = ',np.sqrt(np.max((zz1.xip-mean_xip)**2/var_xip)))
    print('max pull for xim = ',np.sqrt(np.max((zz1.xim-mean_xim)**2/var_xim)))
    np.testing.assert_array_less((zz1.xip-mean_xip)**2, 9*var_xip)  # < 3 sigma pull
    np.testing.assert_array_less((zz1.xim-mean_xim)**2, 9*var_xim)  # < 3 sigma pull
    np.testing.assert_allclose(zz1.varxip, mean_varxip, rtol=0.1)
    np.testing.assert_allclose(zz1.varxim, mean_varxim, rtol=0.1)

    # Now run with patches, but still with shot variance.  Should be basically the same answer.
    catp = treecorr.Catalog(x=x2, y=y2, z1=z1, z2=z2, npatch=npatch)
    print('tot w = ',np.sum(w))
    print('Patch\tNsource')
    for i in range(npatch):
        print('%d\t%d'%(i,np.sum(catp.w[catp.patch==i])))
    zz2 = treecorr.ZZCorrelation(corr_params)
    t0 = time.time()
    zz2.process(catp)
    t1 = time.time()
    print('Time for patch processing = ',t1-t0)
    print('weight = ',zz2.weight)
    print('xip = ',zz2.xip)
    print('xip1 = ',zz1.xip)
    print('varxip = ',zz2.varxip)
    print('xim = ',zz2.xim)
    print('xim1 = ',zz1.xim)
    print('varxim = ',zz2.varxim)
    np.testing.assert_allclose(zz2.weight, zz1.weight, rtol=1.e-2)
    np.testing.assert_allclose(zz2.xip, zz1.xip, rtol=1.e-2)
    np.testing.assert_allclose(zz2.xim, zz1.xim, rtol=1.e-2)
    np.testing.assert_allclose(zz2.varxip, zz1.varxip, rtol=1.e-2)
    np.testing.assert_allclose(zz2.varxim, zz1.varxim, rtol=1.e-2)

    # Now try jackknife variance estimate.
    t0 = time.time()
    cov2 = zz2.estimate_cov('jackknife')
    t1 = time.time()
    print('Time to calculate jackknife covariance = ',t1-t0)
    print('cov.diag = ',np.diagonal(cov2))
    print('cf var_xip = ',var_xip)
    print('cf var_xim = ',var_xim)
    np.testing.assert_allclose(np.diagonal(cov2)[:4], var_xip, rtol=0.3)
    np.testing.assert_allclose(np.diagonal(cov2)[4:], var_xim, rtol=0.5)

    # Use initialize/finalize
    zz3 = treecorr.ZZCorrelation(corr_params)
    for k1, p1 in enumerate(catp.get_patches()):
        zz3.process(p1, initialize=(k1==0), finalize=(k1==npatch-1))
        for k2, p2 in enumerate(catp.get_patches()):
            if k2 <= k1: continue
            zz3.process(p1, p2, initialize=False, finalize=False)
    np.testing.assert_allclose(zz3.xip, zz2.xip)
    np.testing.assert_allclose(zz3.xim, zz2.xim)
    np.testing.assert_allclose(zz3.weight, zz2.weight)

    # Check that these still work after roundtripping through a file.
    try:
        import fitsio
    except ImportError:
        pass
    else:
        file_name = os.path.join('output','test_write_results_zz.fits')
        zz2.write(file_name, write_patch_results=True)
        zz3 = treecorr.ZZCorrelation.from_file(file_name)
        cov3 = zz3.estimate_cov('jackknife')
        np.testing.assert_allclose(cov3, cov2)

    # Check some invalid actions
    # Bad var_method
    with assert_raises(ValueError):
        zz2.estimate_cov('invalid')
    # Not run on patches, but need patches
    with assert_raises(ValueError):
        zz1.estimate_cov('jackknife')
    with assert_raises(ValueError):
        zz1.estimate_cov('sample')
    with assert_raises(ValueError):
        zz1.estimate_cov('marked_bootstrap')
    with assert_raises(ValueError):
        zz1.estimate_cov('bootstrap')

    cata = treecorr.Catalog(x=x2[:100], y=y2[:100], z1=z1[:100], z2=z2[:100], npatch=10)
    catb = treecorr.Catalog(x=x2[:100], y=y2[:100], z1=z1[:100], z2=z2[:100], npatch=2)
    zz4 = treecorr.ZZCorrelation(corr_params)
    zz5 = treecorr.ZZCorrelation(corr_params)
    # All catalogs need to have the same number of patches
    with assert_raises(RuntimeError):
        zz4.process(cata,catb)
    with assert_raises(RuntimeError):
        zz5.process(catb,cata)

@timer
def test_twod():
    from test_twod import corr2d
    try:
        from scipy.spatial.distance import pdist, squareform
    except ImportError:
        print('Skipping test_twod, since uses scipy, and scipy is not installed.')
        return

    # N random points in 2 dimensions
    rng = np.random.RandomState(8675309)
    N = 200
    x = rng.uniform(-20, 20, N)
    y = rng.uniform(-20, 20, N)

    # Give the points a multivariate Gaussian random field for v
    L1 = [[0.33, 0.09], [-0.01, 0.26]]  # Some arbitrary correlation matrix
    invL1 = np.linalg.inv(L1)
    dists = pdist(np.array([x,y]).T, metric='mahalanobis', VI=invL1)
    K = np.exp(-0.5 * dists**2)
    K = squareform(K)
    np.fill_diagonal(K, 1.)

    A = 2.3
    sigma = A/10.

    # Make v
    z1 = rng.multivariate_normal(np.zeros(N), K*(A**2))
    z1 += rng.normal(scale=sigma, size=N)
    z2 = rng.multivariate_normal(np.zeros(N), K*(A**2))
    z2 += rng.normal(scale=sigma, size=N)
    z = z1 + 1j * z2

    # Calculate the 2D correlation using brute force
    max_sep = 21.
    nbins = 21
    xi_brut = corr2d(x, y, z, np.conj(z), rmax=max_sep, bins=nbins)

    # And using TreeCorr
    cat = treecorr.Catalog(x=x, y=y, z1=z1, z2=z2)
    zz = treecorr.ZZCorrelation(max_sep=max_sep, bin_size=2., bin_type='TwoD', brute=True)
    zz.process(cat)
    print('max abs diff = ',np.max(np.abs(zz.xip - xi_brut)))
    print('max rel diff = ',np.max(np.abs(zz.xip - xi_brut)/np.abs(zz.xip)))
    np.testing.assert_allclose(zz.xip, xi_brut, atol=2.e-7)

    zz = treecorr.ZZCorrelation(max_sep=max_sep, bin_size=2., bin_type='TwoD', bin_slop=0.05)
    zz.process(cat)
    print('max abs diff = ',np.max(np.abs(zz.xip - xi_brut)))
    print('max rel diff = ',np.max(np.abs(zz.xip - xi_brut)/np.abs(zz.xip)))
    np.testing.assert_allclose(zz.xip, xi_brut, atol=2.e-7)

    # Check I/O
    try:
        import fitsio
    except ImportError:
        pass
    else:
        fits_name = 'output/zz_twod.fits'
        zz.write(fits_name)
        zz2 = treecorr.ZZCorrelation.from_file(fits_name)
        np.testing.assert_allclose(zz2.npairs, zz.npairs)
        np.testing.assert_allclose(zz2.weight, zz.weight)
        np.testing.assert_allclose(zz2.meanr, zz.meanr)
        np.testing.assert_allclose(zz2.meanlogr, zz.meanlogr)
        np.testing.assert_allclose(zz2.xip, zz.xip)
        np.testing.assert_allclose(zz2.xip_im, zz.xip_im)
        np.testing.assert_allclose(zz2.xim, zz.xim)
        np.testing.assert_allclose(zz2.xim_im, zz.xim_im)

    ascii_name = 'output/zz_twod.txt'
    zz.write(ascii_name, precision=16)
    zz3 = treecorr.ZZCorrelation.from_file(ascii_name)
    np.testing.assert_allclose(zz3.npairs, zz.npairs)
    np.testing.assert_allclose(zz3.weight, zz.weight)
    np.testing.assert_allclose(zz3.meanr, zz.meanr)
    np.testing.assert_allclose(zz3.meanlogr, zz.meanlogr)
    np.testing.assert_allclose(zz3.xip, zz.xip)
    np.testing.assert_allclose(zz3.xip_im, zz.xip_im)
    np.testing.assert_allclose(zz3.xim, zz.xim)
    np.testing.assert_allclose(zz3.xim_im, zz.xim_im)


if __name__ == '__main__':
    test_direct()
    test_direct_spherical()
    test_zz()
    test_varxi()
    test_jk()
    test_twod()
