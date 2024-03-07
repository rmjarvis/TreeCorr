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

    ngal = 500
    s = 10.
    rng = np.random.RandomState(8675309)
    x1 = rng.normal(0,s, (ngal,) )
    y1 = rng.normal(0,s, (ngal,) )
    w1 = rng.random_sample(ngal) 
    q11 = rng.normal(0,0.2, (ngal,) )
    q21 = rng.normal(0,0.2, (ngal,) )

    x2 = rng.normal(0,s, (ngal,) )
    y2 = rng.normal(0,s, (ngal,) )
    w2 = rng.random_sample(ngal) 
    q12 = rng.normal(0,0.2, (ngal,) )
    q22 = rng.normal(0,0.2, (ngal,) )

    cat1 = treecorr.Catalog(x=x1, y=y1, w=w1, q1=q11, q2=q21)
    cat2 = treecorr.Catalog(x=x2, y=y2, w=w2, q1=q12, q2=q22)

    min_sep = 1.
    max_sep = 50.
    nbins = 50
    bin_size = np.log(max_sep/min_sep) / nbins
    qq = treecorr.QQCorrelation(min_sep=min_sep, max_sep=max_sep, nbins=nbins, brute=True)
    qq.process(cat1, cat2)

    true_npairs = np.zeros(nbins, dtype=int)
    true_weight = np.zeros(nbins, dtype=float)
    true_xip = np.zeros(nbins, dtype=complex)
    true_xim = np.zeros(nbins, dtype=complex)
    for i in range(ngal):
        # It's hard to do all the pairs at once with numpy operations (although maybe possible).
        # But we can at least do all the pairs for each entry in cat1 at once with arrays.
        rsq = (x1[i]-x2)**2 + (y1[i]-y2)**2
        r = np.sqrt(rsq)
        expmialpha = ((x1[i]-x2) - 1j*(y1[i]-y2)) / r

        ww = w1[i] * w2
        xip = ww * (q11[i] + 1j*q21[i]) * (q12 - 1j*q22)
        xim = ww * (q11[i] + 1j*q21[i]) * (q12 + 1j*q22) * expmialpha**8

        index = np.floor(np.log(r/min_sep) / bin_size).astype(int)
        mask = (index >= 0) & (index < nbins)
        np.add.at(true_npairs, index[mask], 1)
        np.add.at(true_weight, index[mask], ww[mask])
        np.add.at(true_xip, index[mask], xip[mask])
        np.add.at(true_xim, index[mask], xim[mask])

    true_xip /= true_weight
    true_xim /= true_weight

    print('true_npairs = ',true_npairs)
    print('diff = ',qq.npairs - true_npairs)
    np.testing.assert_array_equal(qq.npairs, true_npairs)

    print('true_weight = ',true_weight)
    print('diff = ',qq.weight - true_weight)
    np.testing.assert_allclose(qq.weight, true_weight, rtol=1.e-6, atol=1.e-8)

    print('true_xip = ',true_xip)
    print('qq.xip = ',qq.xip)
    print('qq.xip_im = ',qq.xip_im)
    np.testing.assert_allclose(qq.xip, true_xip.real, rtol=1.e-6, atol=1.e-8)
    np.testing.assert_allclose(qq.xip_im, true_xip.imag, rtol=1.e-6, atol=1.e-8)
    print('true_xim = ',true_xim)
    print('qq.xim = ',qq.xim)
    print('qq.xim_im = ',qq.xim_im)
    np.testing.assert_allclose(qq.xim, true_xim.real, rtol=1.e-6, atol=1.e-8)
    np.testing.assert_allclose(qq.xim_im, true_xim.imag, rtol=1.e-6, atol=1.e-8)

    # Check that running via the corr2 script works correctly.
    config = treecorr.config.read_config('configs/qq_direct.yaml')
    try:
        import fitsio
    except ImportError:
        pass
    else:
        cat1.write(config['file_name'])
        cat2.write(config['file_name2'])
        treecorr.corr2(config)
        data = fitsio.read(config['qq_file_name'])
        np.testing.assert_allclose(data['r_nom'], qq.rnom)
        np.testing.assert_allclose(data['npairs'], qq.npairs)
        np.testing.assert_allclose(data['weight'], qq.weight)
        np.testing.assert_allclose(data['xip'], qq.xip)
        np.testing.assert_allclose(data['xip_im'], qq.xip_im)
        np.testing.assert_allclose(data['xim'], qq.xim)
        np.testing.assert_allclose(data['xim_im'], qq.xim_im)

    # Repeat with binslop = 0.
    # And don't do any top-level recursion so we actually test not going to the leaves.
    qq = treecorr.QQCorrelation(min_sep=min_sep, max_sep=max_sep, nbins=nbins, bin_slop=0,
                                max_top=0)
    qq.process(cat1, cat2)
    np.testing.assert_array_equal(qq.npairs, true_npairs)
    np.testing.assert_allclose(qq.weight, true_weight, rtol=1.e-6, atol=1.e-8)
    np.testing.assert_allclose(qq.xip, true_xip.real, rtol=1.e-6, atol=1.e-8)
    np.testing.assert_allclose(qq.xip_im, true_xip.imag, rtol=1.e-6, atol=1.e-8)
    print('true_xim = ',true_xim)
    print('qq.xim = ',qq.xim)
    print('qq.xim_im = ',qq.xim_im)
    print('diff = ',qq.xim - true_xim.real)
    print('max diff = ',np.max(np.abs(qq.xim - true_xim.real)))
    print('rel diff = ',(qq.xim - true_xim.real)/true_xim.real)
    print('ratio = ',qq.xim/true_xim.real)
    np.testing.assert_allclose(qq.xim, true_xim.real, atol=3.e-4)
    np.testing.assert_allclose(qq.xim_im, true_xim.imag, atol=1.e-4)

    # With angle_slop = 0, it goes back to being basically exact (to single precision).
    qq = treecorr.QQCorrelation(min_sep=min_sep, max_sep=max_sep, nbins=nbins, bin_slop=0,
                                angle_slop=0, max_top=0)
    qq.process(cat1, cat2)
    np.testing.assert_array_equal(qq.npairs, true_npairs)
    np.testing.assert_allclose(qq.weight, true_weight, rtol=1.e-6, atol=1.e-8)
    np.testing.assert_allclose(qq.xip, true_xip.real, rtol=1.e-6, atol=1.e-8)
    np.testing.assert_allclose(qq.xip_im, true_xip.imag, rtol=1.e-6, atol=1.e-8)
    np.testing.assert_allclose(qq.xim, true_xim.real, rtol=1.e-6, atol=1.e-8)
    np.testing.assert_allclose(qq.xim_im, true_xim.imag, rtol=1.e-6, atol=1.e-8)

    # Check a few basic operations with a QQCorrelation object.
    do_pickle(qq)

    qq2 = qq.copy()
    qq2 += qq
    np.testing.assert_allclose(qq2.npairs, 2*qq.npairs)
    np.testing.assert_allclose(qq2.weight, 2*qq.weight)
    np.testing.assert_allclose(qq2.meanr, 2*qq.meanr)
    np.testing.assert_allclose(qq2.meanlogr, 2*qq.meanlogr)
    np.testing.assert_allclose(qq2.xip, 2*qq.xip)
    np.testing.assert_allclose(qq2.xip_im, 2*qq.xip_im)
    np.testing.assert_allclose(qq2.xim, 2*qq.xim)
    np.testing.assert_allclose(qq2.xim_im, 2*qq.xim_im)

    qq2.clear()
    qq2 += qq
    np.testing.assert_allclose(qq2.npairs, qq.npairs)
    np.testing.assert_allclose(qq2.weight, qq.weight)
    np.testing.assert_allclose(qq2.meanr, qq.meanr)
    np.testing.assert_allclose(qq2.meanlogr, qq.meanlogr)
    np.testing.assert_allclose(qq2.xip, qq.xip)
    np.testing.assert_allclose(qq2.xip_im, qq.xip_im)
    np.testing.assert_allclose(qq2.xim, qq.xim)
    np.testing.assert_allclose(qq2.xim_im, qq.xim_im)

    ascii_name = 'output/qq_ascii.txt'
    qq.write(ascii_name, precision=16)
    qq3 = treecorr.QQCorrelation(min_sep=min_sep, max_sep=max_sep, nbins=nbins, bin_type='Log')
    qq3.read(ascii_name)
    np.testing.assert_allclose(qq3.npairs, qq.npairs)
    np.testing.assert_allclose(qq3.weight, qq.weight)
    np.testing.assert_allclose(qq3.meanr, qq.meanr)
    np.testing.assert_allclose(qq3.meanlogr, qq.meanlogr)
    np.testing.assert_allclose(qq3.xip, qq.xip)
    np.testing.assert_allclose(qq3.xip_im, qq.xip_im)
    np.testing.assert_allclose(qq3.xim, qq.xim)
    np.testing.assert_allclose(qq3.xim_im, qq.xim_im)

    # Check that the repr is minimal
    assert repr(qq3) == f'QQCorrelation(min_sep={min_sep}, max_sep={max_sep}, nbins={nbins})'

    # Simpler API using from_file:
    with CaptureLog() as cl:
        qq3b = treecorr.QQCorrelation.from_file(ascii_name, logger=cl.logger)
    assert ascii_name in cl.output
    np.testing.assert_allclose(qq3b.npairs, qq.npairs)
    np.testing.assert_allclose(qq3b.weight, qq.weight)
    np.testing.assert_allclose(qq3b.meanr, qq.meanr)
    np.testing.assert_allclose(qq3b.meanlogr, qq.meanlogr)
    np.testing.assert_allclose(qq3b.xip, qq.xip)
    np.testing.assert_allclose(qq3b.xip_im, qq.xip_im)
    np.testing.assert_allclose(qq3b.xim, qq.xim)
    np.testing.assert_allclose(qq3b.xim_im, qq.xim_im)

    # or using the Corr2 base class
    with CaptureLog() as cl:
        qq3c = treecorr.Corr2.from_file(ascii_name, logger=cl.logger)
    assert ascii_name in cl.output
    np.testing.assert_allclose(qq3c.npairs, qq.npairs)
    np.testing.assert_allclose(qq3c.weight, qq.weight)
    np.testing.assert_allclose(qq3c.meanr, qq.meanr)
    np.testing.assert_allclose(qq3c.meanlogr, qq.meanlogr)
    np.testing.assert_allclose(qq3c.xip, qq.xip)
    np.testing.assert_allclose(qq3c.xip_im, qq.xip_im)
    np.testing.assert_allclose(qq3c.xim, qq.xim)
    np.testing.assert_allclose(qq3c.xim_im, qq.xim_im)

    try:
        import fitsio
    except ImportError:
        pass
    else:
        fits_name = 'output/qq_fits.fits'
        qq.write(fits_name)
        qq4 = treecorr.QQCorrelation(min_sep=min_sep, max_sep=max_sep, nbins=nbins)
        qq4.read(fits_name)
        np.testing.assert_allclose(qq4.npairs, qq.npairs)
        np.testing.assert_allclose(qq4.weight, qq.weight)
        np.testing.assert_allclose(qq4.meanr, qq.meanr)
        np.testing.assert_allclose(qq4.meanlogr, qq.meanlogr)
        np.testing.assert_allclose(qq4.xip, qq.xip)
        np.testing.assert_allclose(qq4.xip_im, qq.xip_im)
        np.testing.assert_allclose(qq4.xim, qq.xim)
        np.testing.assert_allclose(qq4.xim_im, qq.xim_im)

        qq4b = treecorr.QQCorrelation.from_file(fits_name)
        np.testing.assert_allclose(qq4b.npairs, qq.npairs)
        np.testing.assert_allclose(qq4b.weight, qq.weight)
        np.testing.assert_allclose(qq4b.meanr, qq.meanr)
        np.testing.assert_allclose(qq4b.meanlogr, qq.meanlogr)
        np.testing.assert_allclose(qq4b.xip, qq.xip)
        np.testing.assert_allclose(qq4b.xip_im, qq.xip_im)
        np.testing.assert_allclose(qq4b.xim, qq.xim)
        np.testing.assert_allclose(qq4b.xim_im, qq.xim_im)

    with assert_raises(TypeError):
        qq2 += config
    qq4 = treecorr.QQCorrelation(min_sep=min_sep/2, max_sep=max_sep, nbins=nbins)
    with assert_raises(ValueError):
        qq2 += qq4
    qq5 = treecorr.QQCorrelation(min_sep=min_sep, max_sep=max_sep*2, nbins=nbins)
    with assert_raises(ValueError):
        qq2 += qq5
    qq6 = treecorr.QQCorrelation(min_sep=min_sep, max_sep=max_sep, nbins=nbins*2)
    with assert_raises(ValueError):
        qq2 += qq6
    with assert_raises(ValueError):
        qq.process(cat1, cat2, patch_method='nonlocal')

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
    q11 = rng.normal(0,0.2, (ngal,) )
    q21 = rng.normal(0,0.2, (ngal,) )

    x2 = rng.normal(0,s, (ngal,) )
    y2 = rng.normal(0,s, (ngal,) ) + 200
    z2 = rng.normal(0,s, (ngal,) )
    w2 = rng.random_sample(ngal)
    q12 = rng.normal(0,0.2, (ngal,) )
    q22 = rng.normal(0,0.2, (ngal,) )

    ra1, dec1 = coord.CelestialCoord.xyz_to_radec(x1,y1,z1)
    ra2, dec2 = coord.CelestialCoord.xyz_to_radec(x2,y2,z2)

    cat1 = treecorr.Catalog(ra=ra1, dec=dec1, ra_units='rad', dec_units='rad', w=w1, q1=q11, q2=q21)
    cat2 = treecorr.Catalog(ra=ra2, dec=dec2, ra_units='rad', dec_units='rad', w=w2, q1=q12, q2=q22)

    min_sep = 1.
    max_sep = 10.
    nbins = 50
    bin_size = np.log(max_sep/min_sep) / nbins
    qq = treecorr.QQCorrelation(min_sep=min_sep, max_sep=max_sep, nbins=nbins,
                                sep_units='deg', brute=True)
    qq.process(cat1, cat2)

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

            # Rotate vectors to coordinates where line connecting is horizontal.
            # Original orientation is where north is up.
            theta1 = -90*coord.degrees + c1[i].angleBetween(c2[j], north_pole)
            theta2 = 90*coord.degrees + c2[j].angleBetween(c1[i], north_pole)
            exp4theta1 = np.cos(4*theta1) + 1j * np.sin(4*theta1)
            exp4theta2 = np.cos(4*theta2) + 1j * np.sin(4*theta2)

            q1 = q11[i] + 1j * q21[i]
            q2 = q12[j] + 1j * q22[j]
            q1 *= exp4theta1
            q2 *= exp4theta2

            ww = w1[i] * w2[j]
            xip = ww * q1 * np.conjugate(q2)
            xim = ww * q1 * q2

            true_npairs[index] += 1
            true_weight[index] += ww
            true_xip[index] += xip
            true_xim[index] += xim

    true_xip /= true_weight
    true_xim /= true_weight

    print('true_npairs = ',true_npairs)
    print('diff = ',qq.npairs - true_npairs)
    np.testing.assert_array_equal(qq.npairs, true_npairs)

    print('true_weight = ',true_weight)
    print('diff = ',qq.weight - true_weight)
    np.testing.assert_allclose(qq.weight, true_weight, rtol=1.e-6, atol=1.e-8)

    print('true_xip = ',true_xip)
    print('qq.xip = ',qq.xip)
    print('qq.xip_im = ',qq.xip_im)
    np.testing.assert_allclose(qq.xip, true_xip.real, rtol=1.e-6, atol=1.e-8)
    np.testing.assert_allclose(qq.xip_im, true_xip.imag, rtol=1.e-6, atol=1.e-8)
    print('true_xim = ',true_xim)
    print('qq.xim = ',qq.xim)
    print('qq.xim_im = ',qq.xim_im)
    np.testing.assert_allclose(qq.xim, true_xim.real, rtol=1.e-6, atol=1.e-8)
    np.testing.assert_allclose(qq.xim_im, true_xim.imag, rtol=1.e-6, atol=1.e-8)

    # Check that running via the corr2 script works correctly.
    try:
        import fitsio
    except ImportError:
        pass
    else:
        config = treecorr.config.read_config('configs/qq_direct_spherical.yaml')
        cat1.write(config['file_name'])
        cat2.write(config['file_name2'])
        treecorr.corr2(config)
        data = fitsio.read(config['qq_file_name'])
        np.testing.assert_allclose(data['r_nom'], qq.rnom)
        np.testing.assert_allclose(data['npairs'], qq.npairs)
        np.testing.assert_allclose(data['weight'], qq.weight)
        np.testing.assert_allclose(data['xip'], qq.xip)
        np.testing.assert_allclose(data['xip_im'], qq.xip_im)
        np.testing.assert_allclose(data['xim'], qq.xim)
        np.testing.assert_allclose(data['xim_im'], qq.xim_im)

    # Repeat with binslop = 0
    # And don't do any top-level recursion so we actually test not going to the leaves.
    qq = treecorr.QQCorrelation(min_sep=min_sep, max_sep=max_sep, nbins=nbins,
                                sep_units='deg', bin_slop=0, max_top=0)
    qq.process(cat1, cat2)
    np.testing.assert_array_equal(qq.npairs, true_npairs)
    np.testing.assert_allclose(qq.weight, true_weight, rtol=1.e-6, atol=1.e-8)
    np.testing.assert_allclose(qq.xip, true_xip.real, rtol=1.e-6, atol=1.e-6)
    np.testing.assert_allclose(qq.xip_im, true_xip.imag, rtol=1.e-6, atol=1.e-6)
    np.testing.assert_allclose(qq.xim, true_xim.real, atol=2.e-4)
    np.testing.assert_allclose(qq.xim_im, true_xim.imag, atol=2.e-4)

    # With angle_slop = 0, it goes back to being basically exact (to single precision).
    qq = treecorr.QQCorrelation(min_sep=min_sep, max_sep=max_sep, nbins=nbins,
                                sep_units='deg', bin_slop=0, angle_slop=0, max_top=0)
    qq.process(cat1, cat2)
    np.testing.assert_array_equal(qq.npairs, true_npairs)
    np.testing.assert_allclose(qq.weight, true_weight, rtol=1.e-6, atol=1.e-8)
    np.testing.assert_allclose(qq.xip, true_xip.real, rtol=1.e-6, atol=1.e-8)
    np.testing.assert_allclose(qq.xip_im, true_xip.imag, rtol=1.e-6, atol=1.e-8)
    np.testing.assert_allclose(qq.xim, true_xim.real, rtol=1.e-6, atol=1.e-8)
    np.testing.assert_allclose(qq.xim_im, true_xim.imag, rtol=1.e-6, atol=1.e-8)


@timer
def test_qq():
    # Similar to the math in test_gg(), but use a functional form that has a radial vector,
    # rather than radial shear pattern.
    # Also, the xi- integral here uses J2, not J4.

    # Use q_radial(r) = q0 (r/r0)^4 exp(-r^2/2r0^2)
    # i.e. q(r) = q0 (r/r0)^4 exp(-r^2/2r0^2) (x+iy)^4/r^4
    #
    # The Fourier transform is: q~(k) = 2 pi q0 r0^6 k^4 exp(-r0^2 k^2/2) / L^2
    # P(k) = (1/2pi) <|q~(k)|^2> = 2 pi q0^2 r0^12 k^8 / L^4 exp(-r0^2 k^2)
    # xi+(r) = (1/2pi) int( dk k P(k) J0(kr) )
    #        = pi/256 q0^2 (r0/L)^2 exp(-r^2/4r0^2)
    #               (6144r0^8 - 6144r^2r0^6 + 1152r^4r0^4 - 64r^6r0^2 + r^8)/r0^8
    # xi-(r) = (1/2pi) int( dk k P(k) J8(kr) )
    #        = pi/256 q0^2 (r0/L)^2 exp(-r^2/4r0^2) r^8/r0^8

    q0 = 0.05
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
        tol_factor = 7
    rng = np.random.RandomState(8675309)
    x = (rng.random_sample(ngal)-0.5) * L
    y = (rng.random_sample(ngal)-0.5) * L
    r2 = (x**2 + y**2)/r0**2
    theta = np.arctan2(y,x)
    q1 = q0 * r2**2 * np.exp(-r2/2.) * np.cos(4*theta)
    q2 = q0 * r2**2 * np.exp(-r2/2.) * np.sin(4*theta)

    cat = treecorr.Catalog(x=x, y=y, q1=q1, q2=q2, x_units='arcmin', y_units='arcmin')
    qq = treecorr.QQCorrelation(bin_size=0.1, min_sep=10., max_sep=100., sep_units='arcmin',
                                verbose=1)
    qq.process(cat)

    # log(<R>) != <logR>, but it should be close:
    print('meanlogr - log(meanr) = ',qq.meanlogr - np.log(qq.meanr))
    np.testing.assert_allclose(qq.meanlogr, np.log(qq.meanr), atol=1.e-3)

    r = qq.meanr
    temp = np.pi/256. * q0**2 * (r0/L)**2 * np.exp(-0.25*r**2/r0**2)
    true_xip = temp * (6144*r0**8 - 6144*r**2*r0**6 + 1152*r**4*r0**4 - 64*r**6*r0**2 + r**8)/r0**8
    true_xim = temp * r**8/r0**8

    print('qq.xip = ',qq.xip)
    print('true_xip = ',true_xip)
    print('ratio = ',qq.xip / true_xip)
    print('diff = ',qq.xip - true_xip)
    print('max diff = ',max(abs(qq.xip - true_xip)))
    np.testing.assert_allclose(qq.xip, true_xip, rtol=0.2 * tol_factor, atol=3.e-7 * tol_factor)
    print('xip_im = ',qq.xip_im)
    np.testing.assert_allclose(qq.xip_im, 0, atol=4.e-7 * tol_factor)

    print('qq.xim = ',qq.xim)
    print('true_xim = ',true_xim)
    print('ratio = ',qq.xim / true_xim)
    print('diff = ',qq.xim - true_xim)
    print('max diff = ',max(abs(qq.xim - true_xim)))
    np.testing.assert_allclose(qq.xim, true_xim, rtol=0.2 * tol_factor, atol=3.e-7 * tol_factor)
    print('xim_im = ',qq.xim_im)
    np.testing.assert_allclose(qq.xim_im, 0, atol=3.e-7 * tol_factor)

    # Should also work as a cross-correlation with itself
    qq.process(cat,cat)
    np.testing.assert_allclose(qq.meanlogr, np.log(qq.meanr), atol=1.e-3)
    np.testing.assert_allclose(qq.xip, true_xip, rtol=0.2 * tol_factor, atol=3.e-7 * tol_factor)
    np.testing.assert_allclose(qq.xip_im, 0, atol=4.e-7 * tol_factor)
    np.testing.assert_allclose(qq.xim, true_xim, rtol=0.2 * tol_factor, atol=3.e-7 * tol_factor)
    np.testing.assert_allclose(qq.xim_im, 0, atol=3.e-7 * tol_factor)

    # Check that we get the same result using the corr2 function:
    cat.write(os.path.join('data','qq.dat'))
    config = treecorr.read_config('configs/qq.yaml')
    config['verbose'] = 0
    config['precision'] = 8
    treecorr.corr2(config)
    corr2_output = np.genfromtxt(os.path.join('output','qq.out'), names=True, skip_header=1)
    print('qq.xip = ',qq.xip)
    print('from corr2 output = ',corr2_output['xip'])
    print('ratio = ',corr2_output['xip']/qq.xip)
    print('diff = ',corr2_output['xip']-qq.xip)
    np.testing.assert_allclose(corr2_output['xip'], qq.xip)

    print('qq.xim = ',qq.xim)
    print('from corr2 output = ',corr2_output['xim'])
    print('ratio = ',corr2_output['xim']/qq.xim)
    print('diff = ',corr2_output['xim']-qq.xim)
    np.testing.assert_allclose(corr2_output['xim'], qq.xim)

    print('xip_im from corr2 output = ',corr2_output['xip_im'])
    print('max err = ',max(abs(corr2_output['xip_im'])))
    np.testing.assert_allclose(corr2_output['xip_im'], 0, atol=4.e-7 * tol_factor)
    print('xim_im from corr2 output = ',corr2_output['xim_im'])
    print('max err = ',max(abs(corr2_output['xim_im'])))
    np.testing.assert_allclose(corr2_output['xim_im'], 0, atol=2.e-7 * tol_factor)

    # Check the write function
    out_file_name = os.path.join('output','qq_out.dat')
    qq.write(out_file_name, precision=16)
    data = np.genfromtxt(out_file_name, names=True, skip_header=1)
    np.testing.assert_allclose(data['r_nom'], np.exp(qq.logr))
    np.testing.assert_allclose(data['meanr'], qq.meanr)
    np.testing.assert_allclose(data['meanlogr'], qq.meanlogr)
    np.testing.assert_allclose(data['xip'], qq.xip)
    np.testing.assert_allclose(data['xim'], qq.xim)
    np.testing.assert_allclose(data['xip_im'], qq.xip_im)
    np.testing.assert_allclose(data['xim_im'], qq.xim_im)
    np.testing.assert_allclose(data['sigma_xip'], np.sqrt(qq.varxip))
    np.testing.assert_allclose(data['sigma_xim'], np.sqrt(qq.varxim))
    np.testing.assert_allclose(data['weight'], qq.weight)
    np.testing.assert_allclose(data['npairs'], qq.npairs)

    # Check the read function
    qq2 = treecorr.QQCorrelation.from_file(out_file_name)
    np.testing.assert_allclose(qq2.logr, qq.logr)
    np.testing.assert_allclose(qq2.meanr, qq.meanr)
    np.testing.assert_allclose(qq2.meanlogr, qq.meanlogr)
    np.testing.assert_allclose(qq2.xip, qq.xip)
    np.testing.assert_allclose(qq2.xim, qq.xim)
    np.testing.assert_allclose(qq2.xip_im, qq.xip_im)
    np.testing.assert_allclose(qq2.xim_im, qq.xim_im)
    np.testing.assert_allclose(qq2.varxip, qq.varxip)
    np.testing.assert_allclose(qq2.varxim, qq.varxim)
    np.testing.assert_allclose(qq2.weight, qq.weight)
    np.testing.assert_allclose(qq2.npairs, qq.npairs)
    assert qq2.coords == qq.coords
    assert qq2.metric == qq.metric
    assert qq2.sep_units == qq.sep_units
    assert qq2.bin_type == qq.bin_type


@timer
def test_spherical():
    # This is the same field we used for test_qq, but put into spherical coords.
    # We do the spherical trig by hand using the obvious formulae, rather than the clever
    # optimizations that are used by the TreeCorr code, thus serving as a useful test of
    # the latter.

    q0 = 0.05
    r0 = 10. * coord.arcmin / coord.radians
    if __name__ == "__main__":
        nsource = 1000000
        L = 50.*r0  # Not infinity, so this introduces some error.  Our integrals were to infinity.
        tol_factor = 1
    else:
        nsource = 200000
        L = 50.*r0
        tol_factor = 5
    rng = np.random.RandomState(8675309)
    x = (rng.random_sample(nsource)-0.5) * L
    y = (rng.random_sample(nsource)-0.5) * L
    r2 = x**2 + y**2
    r = np.sqrt(r2)
    theta = np.arctan2(y,x)
    q1 = q0 * (r/r0)**4 * np.exp(-r2/2./r0**2) * np.cos(4*theta)
    q2 = q0 * (r/r0)**4 * np.exp(-r2/2./r0**2) * np.sin(4*theta)

    qq = treecorr.QQCorrelation(bin_size=0.1, min_sep=10., max_sep=100., sep_units='arcmin',
                                verbose=1)
    r1 = np.exp(qq.logr) * (coord.arcmin / coord.radians)
    temp = np.pi/256. * q0**2 * (r0/L)**2 * np.exp(-0.25*r1**2/r0**2)
    true_xip = temp * (6144 - 6144*(r1/r0)**2 + 1152*(r1/r0)**4 - 64*(r1/r0)**6 + (r1/r0)**8)
    true_xim = temp * (r1/r0)**8

    # Test this around several central points
    if __name__ == '__main__':
        ra0_list = [ 0., 1., 1.3, 232., 0. ]
        dec0_list = [ 0., -0.3, 1.3, -1.4, np.pi/2.-1.e-6 ]
    else:
        ra0_list = [ 232.]
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

        # Rotate field relative to local west
        # q_sph = exp(4i beta) * q
        # where beta = pi - (A+B) is the angle between north and "up" in the tangent plane.
        beta = np.pi - (A+B)
        beta[x>0] *= -1.
        cos4beta = np.cos(4*beta)
        sin4beta = np.sin(4*beta)
        q1_sph = q1 * cos4beta - q2 * sin4beta
        q2_sph = q2 * cos4beta + q1 * sin4beta

        cat = treecorr.Catalog(ra=ra, dec=dec, q1=q1_sph, q2=q2_sph, ra_units='rad',
                               dec_units='rad')
        qq = treecorr.QQCorrelation(bin_size=0.1, min_sep=10., max_sep=100., sep_units='arcmin',
                                    verbose=1)
        qq.process(cat)

        print('ra0, dec0 = ',ra0,dec0)
        print('qq.xip = ',qq.xip)
        print('true_xip = ',true_xip)
        print('ratio = ',qq.xip / true_xip)
        print('diff = ',qq.xip - true_xip)
        print('max diff = ',max(abs(qq.xip - true_xip)))
        assert max(abs(qq.xip - true_xip)) < 1.6e-6 * tol_factor

        print('qq.xim = ',qq.xim)
        print('true_xim = ',true_xim)
        print('ratio = ',qq.xim / true_xim)
        print('diff = ',qq.xim - true_xim)
        print('max diff = ',max(abs(qq.xim - true_xim)))
        assert max(abs(qq.xim - true_xim)) < 1.8e-6 * tol_factor

    # One more center that can be done very easily.  If the center is the north pole, then all
    # the radial vectors are pure negative q2.
    ra0 = 0
    dec0 = np.pi/2.
    ra = theta
    dec = np.pi/2. - 2.*np.arcsin(r/2.)
    qrad = q0 * (r/r0)**4 * np.exp(-r2/2./r0**2)

    cat = treecorr.Catalog(ra=ra, dec=dec, q1=qrad, q2=np.zeros_like(qrad), ra_units='rad',
                           dec_units='rad')
    qq.process(cat)

    print('qq.xip = ',qq.xip)
    print('qq.xip_im = ',qq.xip_im)
    print('true_xip = ',true_xip)
    print('ratio = ',qq.xip / true_xip)
    print('diff = ',qq.xip - true_xip)
    print('max diff = ',max(abs(qq.xip - true_xip)))
    assert max(abs(qq.xip - true_xip)) < 1.6e-6 * tol_factor
    print('max xip_im = ',max(abs(qq.xip_im)))
    assert max(abs(qq.xip_im)) < 4.e-7 * tol_factor

    print('qq.xim = ',qq.xim)
    print('qq.xim_im = ',qq.xim_im)
    print('true_xim = ',true_xim)
    print('ratio = ',qq.xim / true_xim)
    print('diff = ',qq.xim - true_xim)
    print('max diff = ',max(abs(qq.xim - true_xim)))
    assert max(abs(qq.xim - true_xim)) < 1.8e-6 * tol_factor
    print('max xim_im = ',max(abs(qq.xim_im)))
    assert max(abs(qq.xim_im)) < 2.e-7 * tol_factor

    # Check that we get the same result using the corr2 function
    cat.write(os.path.join('data','qq_spherical.dat'))
    config = treecorr.read_config('configs/qq_spherical.yaml')
    config['verbose'] = 0
    treecorr.corr2(config)
    corr2_output = np.genfromtxt(os.path.join('output','qq_spherical.out'), names=True,
                                 skip_header=1)
    print('qq.xip = ',qq.xip)
    print('from corr2 output = ',corr2_output['xip'])
    print('ratio = ',corr2_output['xip']/qq.xip)
    print('diff = ',corr2_output['xip']-qq.xip)
    np.testing.assert_allclose(corr2_output['xip'], qq.xip)

    print('qq.xim = ',qq.xim)
    print('from corr2 output = ',corr2_output['xim'])
    print('ratio = ',corr2_output['xim']/qq.xim)
    print('diff = ',corr2_output['xim']-qq.xim)
    np.testing.assert_allclose(corr2_output['xim'], qq.xim)

    print('xip_im from corr2 output = ',corr2_output['xip_im'])
    np.testing.assert_allclose(corr2_output['xip_im'], 0, atol=4.e-7 * tol_factor)

    print('xim_im from corr2 output = ',corr2_output['xim_im'])
    np.testing.assert_allclose(corr2_output['xim_im'], 0, atol=2.e-7 * tol_factor)


@timer
def test_varxi():
    # Test that varxip, varxim are correct (or close) based on actual variance of many runs.

    # Same v pattern as in test_qq().  Although the signal doesn't actually matter at all here.
    q0 = 0.05
    r0 = 10.
    L = 50.*r0
    rng = np.random.RandomState(8675309)

    # Note: to get a good estimate of var(xi), you need a lot of runs.  The number of
    # runs matters much more than the number of galaxies for getting this to pass.
    ngal = 1000
    nruns = 50000

    file_name = 'data/test_varxi_qq.npz'
    print(file_name)
    if not os.path.isfile(file_name):
        all_qqs = []

        for run in range(nruns):
            print(f'{run}/{nruns}')
            # In addition to the shape noise below, there is shot noise from the random x,y positions.
            x = (rng.random_sample(ngal)-0.5) * L
            y = (rng.random_sample(ngal)-0.5) * L
            # Varied weights are hard, but at least check that non-unit weights work correctly.
            w = np.ones_like(x) * 5
            r2 = (x**2 + y**2)/r0**2
            theta = np.arctan2(y,x)
            q1 = q0 * np.exp(-r2/2.) * np.cos(4*theta)
            q2 = q0 * np.exp(-r2/2.) * np.sin(4*theta)
            # This time, add some shape noise (different each run).
            q1 += rng.normal(0, 0.3, size=ngal)
            q2 += rng.normal(0, 0.3, size=ngal)

            cat = treecorr.Catalog(x=x, y=y, w=w, q1=q1, q2=q2, x_units='arcmin', y_units='arcmin')
            qq = treecorr.QQCorrelation(bin_size=0.1, min_sep=10., max_sep=100., sep_units='arcmin',
                                        verbose=1)
            qq.process(cat)
            all_qqs.append(qq)

        mean_xip = np.mean([qq.xip for qq in all_qqs], axis=0)
        var_xip = np.var([qq.xip for qq in all_qqs], axis=0)
        mean_xim = np.mean([qq.xim for qq in all_qqs], axis=0)
        var_xim = np.var([qq.xim for qq in all_qqs], axis=0)
        mean_varxip = np.mean([qq.varxip for qq in all_qqs], axis=0)
        mean_varxim = np.mean([qq.varxim for qq in all_qqs], axis=0)

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
    theta = np.arctan2(y,x)
    q1 = q0 * np.exp(-r2/2.) * np.cos(4*theta)
    q2 = q0 * np.exp(-r2/2.) * np.sin(4*theta)
    # This time, add some shape noise (different each run).
    q1 += rng.normal(0, 0.3, size=ngal)
    q2 += rng.normal(0, 0.3, size=ngal)

    cat = treecorr.Catalog(x=x, y=y, w=w, q1=q1, q2=q2, x_units='arcmin', y_units='arcmin')
    qq = treecorr.QQCorrelation(bin_size=0.1, min_sep=10., max_sep=100., sep_units='arcmin',
                                verbose=1)
    qq.process(cat)
    print('single run:')
    print('max relerr for xip = ',np.max(np.abs((qq.varxip - var_xip)/var_xip)))
    print('max relerr for xim = ',np.max(np.abs((qq.varxip - var_xim)/var_xim)))
    np.testing.assert_allclose(qq.varxip, var_xip, rtol=0.2)
    np.testing.assert_allclose(qq.varxim, var_xim, rtol=0.2)

@timer
def test_jk():

    # Same multi-lens field we used for NV patch test
    q0 = 0.05
    r0 = 30.
    L = 30 * r0
    rng = np.random.RandomState(8675309)

    nsource = 100000
    nlens = 300
    nruns = 1000
    npatch = 64

    corr_params = dict(bin_size=0.3, min_sep=80, max_sep=200, bin_slop=0.1)

    def make_spin4_field(rng):
        x1 = (rng.random(nlens)-0.5) * L
        y1 = (rng.random(nlens)-0.5) * L
        w = rng.random(nlens) + 10
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
            q1 += w[i] * q0 * r2**2 * np.exp(-r2/2.) * np.cos(4*theta)
            q2 += w[i] * q0 * r2**2 * np.exp(-r2/2.) * np.sin(4*theta)
        return x1, y1, w, x2, y2, q1, q2

    file_name = 'data/test_qq_jk_{}.npz'.format(nruns)
    print(file_name)
    if not os.path.isfile(file_name):
        all_qqs = []
        rng = np.random.default_rng()
        for run in range(nruns):
            x1, y1, w, x2, y2, q1, q2 = make_spin4_field(rng)
            print(run,': ',np.mean(q1),np.std(q1),np.min(q1),np.max(q1))
            cat = treecorr.Catalog(x=x2, y=y2, q1=q1, q2=q2)
            qq = treecorr.QQCorrelation(corr_params)
            qq.process(cat)
            all_qqs.append(qq)

        mean_xip = np.mean([qq.xip for qq in all_qqs], axis=0)
        mean_xim = np.mean([qq.xim for qq in all_qqs], axis=0)
        var_xip = np.var([qq.xip for qq in all_qqs], axis=0)
        var_xim = np.var([qq.xim for qq in all_qqs], axis=0)
        mean_varxip = np.mean([qq.varxip for qq in all_qqs], axis=0)
        mean_varxim = np.mean([qq.varxim for qq in all_qqs], axis=0)

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
    x1, y1, w, x2, y2, q1, q2 = make_spin4_field(rng)

    cat = treecorr.Catalog(x=x2, y=y2, q1=q1, q2=q2)
    qq1 = treecorr.QQCorrelation(corr_params)
    qq1.process(cat)

    print('weight = ',qq1.weight)
    print('xip = ',qq1.xip)
    print('varxip = ',qq1.varxip)
    print('pullsq for xip = ',(qq1.xip-mean_xip)**2/var_xip)
    print('max pull for xip = ',np.sqrt(np.max((qq1.xip-mean_xip)**2/var_xip)))
    print('max pull for xim = ',np.sqrt(np.max((qq1.xim-mean_xim)**2/var_xim)))
    np.testing.assert_array_less((qq1.xip-mean_xip)**2, 9*var_xip)  # < 3 sigma pull
    np.testing.assert_array_less((qq1.xim-mean_xim)**2, 9*var_xim)  # < 3 sigma pull
    np.testing.assert_allclose(qq1.varxip, mean_varxip, rtol=0.2)
    np.testing.assert_allclose(qq1.varxim, mean_varxim, rtol=0.2)

    # Now run with patches, but still with shot variance.  Should be basically the same answer.
    catp = treecorr.Catalog(x=x2, y=y2, q1=q1, q2=q2, npatch=npatch)
    print('tot w = ',np.sum(w))
    print('Patch\tNsource')
    for i in range(npatch):
        print('%d\t%d'%(i,np.sum(catp.w[catp.patch==i])))
    qq2 = treecorr.QQCorrelation(corr_params)
    qq2.process(catp)
    print('weight = ',qq2.weight)
    print('xip = ',qq2.xip)
    print('xip1 = ',qq1.xip)
    print('varxip = ',qq2.varxip)
    print('xim = ',qq2.xim)
    print('xim1 = ',qq1.xim)
    print('varxim = ',qq2.varxim)
    np.testing.assert_allclose(qq2.weight, qq1.weight, rtol=1.e-2)
    np.testing.assert_allclose(qq2.xip, qq1.xip, rtol=2.e-2)
    np.testing.assert_allclose(qq2.xim, qq1.xim, rtol=2.e-2)
    np.testing.assert_allclose(qq2.varxip, qq1.varxip, rtol=1.e-2)
    np.testing.assert_allclose(qq2.varxim, qq1.varxim, rtol=1.e-2)

    # Now try jackknife variance estimate.
    cov2 = qq2.estimate_cov('jackknife')
    print('cov.diag = ',np.diagonal(cov2))
    print('cf var_xip = ',var_xip)
    print('cf var_xim = ',var_xim)
    np.testing.assert_allclose(np.diagonal(cov2)[:4], var_xip, rtol=0.6)
    np.testing.assert_allclose(np.diagonal(cov2)[4:], var_xim, rtol=0.9)

    # Use initialize/finalize
    qq3 = treecorr.QQCorrelation(corr_params)
    for k1, p1 in enumerate(catp.get_patches()):
        qq3.process(p1, initialize=(k1==0), finalize=(k1==npatch-1))
        for k2, p2 in enumerate(catp.get_patches()):
            if k2 <= k1: continue
            qq3.process(p1, p2, initialize=False, finalize=False)
    np.testing.assert_allclose(qq3.xip, qq2.xip)
    np.testing.assert_allclose(qq3.xim, qq2.xim)
    np.testing.assert_allclose(qq3.weight, qq2.weight)

    # Check that these still work after roundtripping through a file.
    try:
        import fitsio
    except ImportError:
        pass
    else:
        file_name = os.path.join('output','test_write_results_qq.fits')
        qq2.write(file_name, write_patch_results=True)
        qq3 = treecorr.QQCorrelation.from_file(file_name)
        cov3 = qq3.estimate_cov('jackknife')
        np.testing.assert_allclose(cov3, cov2)

    # Check some invalid actions
    # Bad var_method
    with assert_raises(ValueError):
        qq2.estimate_cov('invalid')
    # Not run on patches, but need patches
    with assert_raises(ValueError):
        qq1.estimate_cov('jackknife')
    with assert_raises(ValueError):
        qq1.estimate_cov('sample')
    with assert_raises(ValueError):
        qq1.estimate_cov('marked_bootstrap')
    with assert_raises(ValueError):
        qq1.estimate_cov('bootstrap')

    cata = treecorr.Catalog(x=x2[:100], y=y2[:100], q1=q1[:100], q2=q2[:100], npatch=10)
    catb = treecorr.Catalog(x=x2[:100], y=y2[:100], q1=q1[:100], q2=q2[:100], npatch=2)
    qq4 = treecorr.QQCorrelation(corr_params)
    qq5 = treecorr.QQCorrelation(corr_params)
    # All catalogs need to have the same number of patches
    with assert_raises(RuntimeError):
        qq4.process(cata,catb)
    with assert_raises(RuntimeError):
        qq5.process(catb,cata)

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
    q1 = rng.multivariate_normal(np.zeros(N), K*(A**2))
    q1 += rng.normal(scale=sigma, size=N)
    q2 = rng.multivariate_normal(np.zeros(N), K*(A**2))
    q2 += rng.normal(scale=sigma, size=N)
    q = q1 + 1j * q2

    # Calculate the 2D correlation using brute force
    max_sep = 21.
    nbins = 21
    xi_brut = corr2d(x, y, q, np.conj(q), rmax=max_sep, bins=nbins)

    # And using TreeCorr
    cat = treecorr.Catalog(x=x, y=y, q1=q1, q2=q2)
    qq = treecorr.QQCorrelation(max_sep=max_sep, bin_size=2., bin_type='TwoD', brute=True)
    qq.process(cat)
    print('max abs diff = ',np.max(np.abs(qq.xip - xi_brut)))
    print('max rel diff = ',np.max(np.abs(qq.xip - xi_brut)/np.abs(qq.xip)))
    np.testing.assert_allclose(qq.xip, xi_brut, atol=2.e-7)

    qq = treecorr.QQCorrelation(max_sep=max_sep, bin_size=2., bin_type='TwoD', bin_slop=0.05)
    qq.process(cat)
    print('max abs diff = ',np.max(np.abs(qq.xip - xi_brut)))
    print('max rel diff = ',np.max(np.abs(qq.xip - xi_brut)/np.abs(qq.xip)))
    np.testing.assert_allclose(qq.xip, xi_brut, atol=2.e-7)

    # Check I/O
    try:
        import fitsio
    except ImportError:
        pass
    else:
        fits_name = 'output/qq_twod.fits'
        qq.write(fits_name)
        qq2 = treecorr.QQCorrelation.from_file(fits_name)
        np.testing.assert_allclose(qq2.npairs, qq.npairs)
        np.testing.assert_allclose(qq2.weight, qq.weight)
        np.testing.assert_allclose(qq2.meanr, qq.meanr)
        np.testing.assert_allclose(qq2.meanlogr, qq.meanlogr)
        np.testing.assert_allclose(qq2.xip, qq.xip)
        np.testing.assert_allclose(qq2.xip_im, qq.xip_im)
        np.testing.assert_allclose(qq2.xim, qq.xim)
        np.testing.assert_allclose(qq2.xim_im, qq.xim_im)

    ascii_name = 'output/qq_twod.txt'
    qq.write(ascii_name, precision=16)
    qq3 = treecorr.QQCorrelation.from_file(ascii_name)
    np.testing.assert_allclose(qq3.npairs, qq.npairs)
    np.testing.assert_allclose(qq3.weight, qq.weight)
    np.testing.assert_allclose(qq3.meanr, qq.meanr)
    np.testing.assert_allclose(qq3.meanlogr, qq.meanlogr)
    np.testing.assert_allclose(qq3.xip, qq.xip)
    np.testing.assert_allclose(qq3.xip_im, qq.xip_im)
    np.testing.assert_allclose(qq3.xim, qq.xim)
    np.testing.assert_allclose(qq3.xim_im, qq.xim_im)


if __name__ == '__main__':
    test_direct()
    test_direct_spherical()
    test_qq()
    test_spherical()
    test_varxi()
    test_jk()
    test_twod()
