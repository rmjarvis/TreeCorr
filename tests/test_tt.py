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
    t11 = rng.normal(0,0.2, (ngal,) )
    t21 = rng.normal(0,0.2, (ngal,) )

    x2 = rng.normal(0,s, (ngal,) )
    y2 = rng.normal(0,s, (ngal,) )
    w2 = rng.random_sample(ngal)
    t12 = rng.normal(0,0.2, (ngal,) )
    t22 = rng.normal(0,0.2, (ngal,) )

    cat1 = treecorr.Catalog(x=x1, y=y1, w=w1, t1=t11, t2=t21)
    cat2 = treecorr.Catalog(x=x2, y=y2, w=w2, t1=t12, t2=t22)

    min_sep = 1.
    max_sep = 50.
    nbins = 50
    bin_size = np.log(max_sep/min_sep) / nbins
    tt = treecorr.TTCorrelation(min_sep=min_sep, max_sep=max_sep, nbins=nbins, brute=True)
    tt.process(cat1, cat2)

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
        xip = ww * (t11[i] + 1j*t21[i]) * (t12 - 1j*t22)
        xim = ww * (t11[i] + 1j*t21[i]) * (t12 + 1j*t22) * expmialpha**6

        index = np.floor(np.log(r/min_sep) / bin_size).astype(int)
        mask = (index >= 0) & (index < nbins)
        np.add.at(true_npairs, index[mask], 1)
        np.add.at(true_weight, index[mask], ww[mask])
        np.add.at(true_xip, index[mask], xip[mask])
        np.add.at(true_xim, index[mask], xim[mask])

    true_xip /= true_weight
    true_xim /= true_weight

    print('true_npairs = ',true_npairs)
    print('diff = ',tt.npairs - true_npairs)
    np.testing.assert_array_equal(tt.npairs, true_npairs)

    print('true_weight = ',true_weight)
    print('diff = ',tt.weight - true_weight)
    np.testing.assert_allclose(tt.weight, true_weight, rtol=1.e-6, atol=1.e-8)

    print('true_xip = ',true_xip)
    print('tt.xip = ',tt.xip)
    print('tt.xip_im = ',tt.xip_im)
    np.testing.assert_allclose(tt.xip, true_xip.real, rtol=1.e-6, atol=1.e-8)
    np.testing.assert_allclose(tt.xip_im, true_xip.imag, rtol=1.e-6, atol=1.e-8)
    print('true_xim = ',true_xim)
    print('tt.xim = ',tt.xim)
    print('tt.xim_im = ',tt.xim_im)
    np.testing.assert_allclose(tt.xim, true_xim.real, rtol=1.e-6, atol=1.e-8)
    np.testing.assert_allclose(tt.xim_im, true_xim.imag, rtol=1.e-6, atol=1.e-8)

    # Check that running via the corr2 script works correctly.
    config = treecorr.config.read_config('configs/tt_direct.yaml')
    try:
        import fitsio
    except ImportError:
        pass
    else:
        cat1.write(config['file_name'])
        cat2.write(config['file_name2'])
        treecorr.corr2(config)
        data = fitsio.read(config['tt_file_name'])
        np.testing.assert_allclose(data['r_nom'], tt.rnom)
        np.testing.assert_allclose(data['npairs'], tt.npairs)
        np.testing.assert_allclose(data['weight'], tt.weight)
        np.testing.assert_allclose(data['xip'], tt.xip)
        np.testing.assert_allclose(data['xip_im'], tt.xip_im)
        np.testing.assert_allclose(data['xim'], tt.xim)
        np.testing.assert_allclose(data['xim_im'], tt.xim_im)

    # Repeat with binslop = 0.
    # And don't do any top-level recursion so we actually test not going to the leaves.
    tt = treecorr.TTCorrelation(min_sep=min_sep, max_sep=max_sep, nbins=nbins, bin_slop=0,
                                max_top=0)
    tt.process(cat1, cat2)
    np.testing.assert_array_equal(tt.npairs, true_npairs)
    np.testing.assert_allclose(tt.weight, true_weight, rtol=1.e-6, atol=1.e-8)
    np.testing.assert_allclose(tt.xip, true_xip.real, rtol=1.e-6, atol=1.e-8)
    np.testing.assert_allclose(tt.xip_im, true_xip.imag, rtol=1.e-6, atol=1.e-8)
    print('true_xim = ',true_xim)
    print('tt.xim = ',tt.xim)
    print('tt.xim_im = ',tt.xim_im)
    print('diff = ',tt.xim - true_xim.real)
    print('max diff = ',np.max(np.abs(tt.xim - true_xim.real)))
    print('rel diff = ',(tt.xim - true_xim.real)/true_xim.real)
    np.testing.assert_allclose(tt.xim, true_xim.real, atol=3.e-4)
    np.testing.assert_allclose(tt.xim_im, true_xim.imag, atol=5.e-4)

    # With angle_slop = 0, it goes back to being basically exact (to single precision).
    tt = treecorr.TTCorrelation(min_sep=min_sep, max_sep=max_sep, nbins=nbins, bin_slop=0,
                                angle_slop=0, max_top=0)
    tt.process(cat1, cat2)
    np.testing.assert_array_equal(tt.npairs, true_npairs)
    np.testing.assert_allclose(tt.weight, true_weight, rtol=1.e-6, atol=1.e-8)
    np.testing.assert_allclose(tt.xip, true_xip.real, rtol=1.e-6, atol=1.e-8)
    np.testing.assert_allclose(tt.xip_im, true_xip.imag, rtol=1.e-6, atol=1.e-8)
    np.testing.assert_allclose(tt.xim, true_xim.real, rtol=1.e-6, atol=1.e-8)
    np.testing.assert_allclose(tt.xim_im, true_xim.imag, rtol=1.e-6, atol=1.e-8)

    # Check a few basic operations with a TTCorrelation object.
    do_pickle(tt)

    tt2 = tt.copy()
    tt2 += tt
    np.testing.assert_allclose(tt2.npairs, 2*tt.npairs)
    np.testing.assert_allclose(tt2.weight, 2*tt.weight)
    np.testing.assert_allclose(tt2.meanr, 2*tt.meanr)
    np.testing.assert_allclose(tt2.meanlogr, 2*tt.meanlogr)
    np.testing.assert_allclose(tt2.xip, 2*tt.xip)
    np.testing.assert_allclose(tt2.xip_im, 2*tt.xip_im)
    np.testing.assert_allclose(tt2.xim, 2*tt.xim)
    np.testing.assert_allclose(tt2.xim_im, 2*tt.xim_im)

    tt2.clear()
    tt2 += tt
    np.testing.assert_allclose(tt2.npairs, tt.npairs)
    np.testing.assert_allclose(tt2.weight, tt.weight)
    np.testing.assert_allclose(tt2.meanr, tt.meanr)
    np.testing.assert_allclose(tt2.meanlogr, tt.meanlogr)
    np.testing.assert_allclose(tt2.xip, tt.xip)
    np.testing.assert_allclose(tt2.xip_im, tt.xip_im)
    np.testing.assert_allclose(tt2.xim, tt.xim)
    np.testing.assert_allclose(tt2.xim_im, tt.xim_im)

    ascii_name = 'output/tt_ascii.txt'
    tt.write(ascii_name, precision=16)
    tt3 = treecorr.TTCorrelation(min_sep=min_sep, max_sep=max_sep, nbins=nbins, bin_type='Log')
    tt3.read(ascii_name)
    np.testing.assert_allclose(tt3.npairs, tt.npairs)
    np.testing.assert_allclose(tt3.weight, tt.weight)
    np.testing.assert_allclose(tt3.meanr, tt.meanr)
    np.testing.assert_allclose(tt3.meanlogr, tt.meanlogr)
    np.testing.assert_allclose(tt3.xip, tt.xip)
    np.testing.assert_allclose(tt3.xip_im, tt.xip_im)
    np.testing.assert_allclose(tt3.xim, tt.xim)
    np.testing.assert_allclose(tt3.xim_im, tt.xim_im)

    # Check that the repr is minimal
    assert repr(tt3) == f'TTCorrelation(min_sep={min_sep}, max_sep={max_sep}, nbins={nbins})'

    # Simpler API using from_file:
    with CaptureLog() as cl:
        tt3b = treecorr.TTCorrelation.from_file(ascii_name, logger=cl.logger)
    assert ascii_name in cl.output
    np.testing.assert_allclose(tt3b.npairs, tt.npairs)
    np.testing.assert_allclose(tt3b.weight, tt.weight)
    np.testing.assert_allclose(tt3b.meanr, tt.meanr)
    np.testing.assert_allclose(tt3b.meanlogr, tt.meanlogr)
    np.testing.assert_allclose(tt3b.xip, tt.xip)
    np.testing.assert_allclose(tt3b.xip_im, tt.xip_im)
    np.testing.assert_allclose(tt3b.xim, tt.xim)
    np.testing.assert_allclose(tt3b.xim_im, tt.xim_im)

    # or using the Corr2 base class
    with CaptureLog() as cl:
        tt3c = treecorr.Corr2.from_file(ascii_name, logger=cl.logger)
    assert ascii_name in cl.output
    np.testing.assert_allclose(tt3c.npairs, tt.npairs)
    np.testing.assert_allclose(tt3c.weight, tt.weight)
    np.testing.assert_allclose(tt3c.meanr, tt.meanr)
    np.testing.assert_allclose(tt3c.meanlogr, tt.meanlogr)
    np.testing.assert_allclose(tt3c.xip, tt.xip)
    np.testing.assert_allclose(tt3c.xip_im, tt.xip_im)
    np.testing.assert_allclose(tt3c.xim, tt.xim)
    np.testing.assert_allclose(tt3c.xim_im, tt.xim_im)

    try:
        import fitsio
    except ImportError:
        pass
    else:
        fits_name = 'output/tt_fits.fits'
        tt.write(fits_name)
        tt4 = treecorr.TTCorrelation(min_sep=min_sep, max_sep=max_sep, nbins=nbins)
        tt4.read(fits_name)
        np.testing.assert_allclose(tt4.npairs, tt.npairs)
        np.testing.assert_allclose(tt4.weight, tt.weight)
        np.testing.assert_allclose(tt4.meanr, tt.meanr)
        np.testing.assert_allclose(tt4.meanlogr, tt.meanlogr)
        np.testing.assert_allclose(tt4.xip, tt.xip)
        np.testing.assert_allclose(tt4.xip_im, tt.xip_im)
        np.testing.assert_allclose(tt4.xim, tt.xim)
        np.testing.assert_allclose(tt4.xim_im, tt.xim_im)

        tt4b = treecorr.TTCorrelation.from_file(fits_name)
        np.testing.assert_allclose(tt4b.npairs, tt.npairs)
        np.testing.assert_allclose(tt4b.weight, tt.weight)
        np.testing.assert_allclose(tt4b.meanr, tt.meanr)
        np.testing.assert_allclose(tt4b.meanlogr, tt.meanlogr)
        np.testing.assert_allclose(tt4b.xip, tt.xip)
        np.testing.assert_allclose(tt4b.xip_im, tt.xip_im)
        np.testing.assert_allclose(tt4b.xim, tt.xim)
        np.testing.assert_allclose(tt4b.xim_im, tt.xim_im)

    with assert_raises(TypeError):
        tt2 += config
    tt4 = treecorr.TTCorrelation(min_sep=min_sep/2, max_sep=max_sep, nbins=nbins)
    with assert_raises(ValueError):
        tt2 += tt4
    tt5 = treecorr.TTCorrelation(min_sep=min_sep, max_sep=max_sep*2, nbins=nbins)
    with assert_raises(ValueError):
        tt2 += tt5
    tt6 = treecorr.TTCorrelation(min_sep=min_sep, max_sep=max_sep, nbins=nbins*2)
    with assert_raises(ValueError):
        tt2 += tt6
    with assert_raises(ValueError):
        tt.process(cat1, cat2, patch_method='nonlocal')

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
    t11 = rng.normal(0,0.2, (ngal,) )
    t21 = rng.normal(0,0.2, (ngal,) )

    x2 = rng.normal(0,s, (ngal,) )
    y2 = rng.normal(0,s, (ngal,) ) + 200
    z2 = rng.normal(0,s, (ngal,) )
    w2 = rng.random_sample(ngal)
    t12 = rng.normal(0,0.2, (ngal,) )
    t22 = rng.normal(0,0.2, (ngal,) )

    ra1, dec1 = coord.CelestialCoord.xyz_to_radec(x1,y1,z1)
    ra2, dec2 = coord.CelestialCoord.xyz_to_radec(x2,y2,z2)

    cat1 = treecorr.Catalog(ra=ra1, dec=dec1, ra_units='rad', dec_units='rad', w=w1, t1=t11, t2=t21)
    cat2 = treecorr.Catalog(ra=ra2, dec=dec2, ra_units='rad', dec_units='rad', w=w2, t1=t12, t2=t22)

    min_sep = 1.
    max_sep = 10.
    nbins = 50
    bin_size = np.log(max_sep/min_sep) / nbins
    tt = treecorr.TTCorrelation(min_sep=min_sep, max_sep=max_sep, nbins=nbins,
                                sep_units='deg', brute=True)
    tt.process(cat1, cat2)

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

            # Rotate field to coordinates where line connecting is horizontal.
            # Original orientation is where north is up.
            theta1 = -90*coord.degrees + c1[i].angleBetween(c2[j], north_pole)
            theta2 = 90*coord.degrees + c2[j].angleBetween(c1[i], north_pole)
            exp3theta1 = np.cos(3*theta1) + 1j * np.sin(3*theta1)
            exp3theta2 = np.cos(3*theta2) + 1j * np.sin(3*theta2)

            t1 = t11[i] + 1j * t21[i]
            t2 = t12[j] + 1j * t22[j]
            t1 *= exp3theta1
            t2 *= exp3theta2

            ww = w1[i] * w2[j]
            xip = ww * t1 * np.conjugate(t2)
            xim = ww * t1 * t2

            true_npairs[index] += 1
            true_weight[index] += ww
            true_xip[index] += xip
            true_xim[index] += xim

    true_xip /= true_weight
    true_xim /= true_weight

    print('true_npairs = ',true_npairs)
    print('diff = ',tt.npairs - true_npairs)
    np.testing.assert_array_equal(tt.npairs, true_npairs)

    print('true_weight = ',true_weight)
    print('diff = ',tt.weight - true_weight)
    np.testing.assert_allclose(tt.weight, true_weight, rtol=1.e-6, atol=1.e-8)

    print('true_xip = ',true_xip)
    print('tt.xip = ',tt.xip)
    print('tt.xip_im = ',tt.xip_im)
    np.testing.assert_allclose(tt.xip, true_xip.real, rtol=1.e-6, atol=1.e-8)
    np.testing.assert_allclose(tt.xip_im, true_xip.imag, rtol=1.e-6, atol=1.e-8)
    print('true_xim = ',true_xim)
    print('tt.xim = ',tt.xim)
    print('tt.xim_im = ',tt.xim_im)
    np.testing.assert_allclose(tt.xim, true_xim.real, rtol=1.e-6, atol=1.e-8)
    np.testing.assert_allclose(tt.xim_im, true_xim.imag, rtol=1.e-6, atol=1.e-8)

    # Check that running via the corr2 script works correctly.
    try:
        import fitsio
    except ImportError:
        pass
    else:
        config = treecorr.config.read_config('configs/tt_direct_spherical.yaml')
        cat1.write(config['file_name'])
        cat2.write(config['file_name2'])
        treecorr.corr2(config)
        data = fitsio.read(config['tt_file_name'])
        np.testing.assert_allclose(data['r_nom'], tt.rnom)
        np.testing.assert_allclose(data['npairs'], tt.npairs)
        np.testing.assert_allclose(data['weight'], tt.weight)
        np.testing.assert_allclose(data['xip'], tt.xip)
        np.testing.assert_allclose(data['xip_im'], tt.xip_im)
        np.testing.assert_allclose(data['xim'], tt.xim)
        np.testing.assert_allclose(data['xim_im'], tt.xim_im)

    # Repeat with binslop = 0
    # And don't do any top-level recursion so we actually test not going to the leaves.
    tt = treecorr.TTCorrelation(min_sep=min_sep, max_sep=max_sep, nbins=nbins,
                                sep_units='deg', bin_slop=0, max_top=0)
    tt.process(cat1, cat2)
    np.testing.assert_array_equal(tt.npairs, true_npairs)
    np.testing.assert_allclose(tt.weight, true_weight, rtol=1.e-6, atol=1.e-8)
    np.testing.assert_allclose(tt.xip, true_xip.real, rtol=1.e-6, atol=1.e-6)
    np.testing.assert_allclose(tt.xip_im, true_xip.imag, rtol=1.e-6, atol=1.e-6)
    np.testing.assert_allclose(tt.xim, true_xim.real, atol=1.e-4)
    np.testing.assert_allclose(tt.xim_im, true_xim.imag, atol=1.e-4)

    # With angle_slop = 0, it goes back to being basically exact (to single precision).
    tt = treecorr.TTCorrelation(min_sep=min_sep, max_sep=max_sep, nbins=nbins,
                                sep_units='deg', bin_slop=0, angle_slop=0, max_top=0)
    tt.process(cat1, cat2)
    np.testing.assert_array_equal(tt.npairs, true_npairs)
    np.testing.assert_allclose(tt.weight, true_weight, rtol=1.e-6, atol=1.e-8)
    np.testing.assert_allclose(tt.xip, true_xip.real, rtol=1.e-6, atol=1.e-8)
    np.testing.assert_allclose(tt.xip_im, true_xip.imag, rtol=1.e-6, atol=1.e-8)
    np.testing.assert_allclose(tt.xim, true_xim.real, rtol=1.e-6, atol=1.e-8)
    np.testing.assert_allclose(tt.xim_im, true_xim.imag, rtol=1.e-6, atol=1.e-8)


@timer
def test_tt():
    # Similar to the math in test_gg(), but use a functional form that has a radial spin-3 value,
    # rather than radial shear pattern.
    # Also, the xi- integral here uses J6, not J4.

    # Use t_radial(r) = t0 r^3/r0^3 exp(-r^2/2r0^2)
    # i.e. t(r) = t0 r^3/r0^3 exp(-r^2/2r0^2) (x+iy)^3/r^3
    #
    # The Fourier transform is: t~(k) = -2 pi i t0 r0^5 k^3 exp(-r0^2 k^2/2) / L^2
    # P(k) = (1/2pi) <|t~(k)|^2> = 2 pi t0^2 r0^10 k^6 / L^4 exp(-r0^2 k^2)
    # xi+(r) = (1/2pi) int( dk k P(k) J0(kr) )
    #        = pi/64 t0^2 (r0/L)^2 exp(-r^2/4r0^2) (384r0^6 - 288r0^4 r^2 + 36 r0^2 r^4 - r^6)/r0^6
    # Note: as with VV, the - sign is empirical, but probably comes from t~ being imaginary.
    # xi-(r) = -(1/2pi) int( dk k P(k) J6(kr) )
    #        = -pi/64 t0^2 (r0/L)^2 exp(-r^2/4r0^2) r^6/r0^6

    t0 = 0.05
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
    theta = np.arctan2(y,x)
    t1 = t0 * r2**1.5 * np.exp(-r2/2.) * np.cos(3*theta)
    t2 = t0 * r2**1.5 * np.exp(-r2/2.) * np.sin(3*theta)

    cat = treecorr.Catalog(x=x, y=y, t1=t1, t2=t2, x_units='arcmin', y_units='arcmin')
    tt = treecorr.TTCorrelation(bin_size=0.1, min_sep=10., max_sep=100., sep_units='arcmin',
                                verbose=1)
    tt.process(cat)

    # log(<R>) != <logR>, but it should be close:
    print('meanlogr - log(meanr) = ',tt.meanlogr - np.log(tt.meanr))
    np.testing.assert_allclose(tt.meanlogr, np.log(tt.meanr), atol=1.e-3)

    r = tt.meanr
    temp = np.pi/64. * t0**2 * (r0/L)**2 * np.exp(-0.25*r**2/r0**2)
    true_xip = temp * (384 - 288*(r/r0)**2 + 36*(r/r0)**4 - (r/r0)**6)
    true_xim = -temp * (r/r0)**6

    print('tt.xip = ',tt.xip)
    print('true_xip = ',true_xip)
    print('ratio = ',tt.xip / true_xip)
    print('diff = ',tt.xip - true_xip)
    print('max diff = ',max(abs(tt.xip - true_xip)))
    # It's within 15% everywhere except at the zero crossings.
    np.testing.assert_allclose(tt.xip, true_xip, rtol=0.15 * tol_factor, atol=3.e-7 * tol_factor)
    print('xip_im = ',tt.xip_im)
    np.testing.assert_allclose(tt.xip_im, 0, atol=3.e-7 * tol_factor)

    print('tt.xim = ',tt.xim)
    print('true_xim = ',true_xim)
    print('ratio = ',tt.xim / true_xim)
    print('diff = ',tt.xim - true_xim)
    print('max diff = ',max(abs(tt.xim - true_xim)))
    np.testing.assert_allclose(tt.xim, true_xim, rtol=0.2 * tol_factor, atol=3.e-7 * tol_factor)
    print('xim_im = ',tt.xim_im)
    np.testing.assert_allclose(tt.xim_im, 0, atol=3.e-7 * tol_factor)

    # Should also work as a cross-correlation with itself
    tt.process(cat,cat)
    np.testing.assert_allclose(tt.meanlogr, np.log(tt.meanr), atol=1.e-3)
    np.testing.assert_allclose(tt.xip, true_xip, rtol=0.15 * tol_factor, atol=3.e-7 * tol_factor)
    np.testing.assert_allclose(tt.xip_im, 0, atol=3.e-7 * tol_factor)
    np.testing.assert_allclose(tt.xim, true_xim, rtol=0.2 * tol_factor, atol=3.e-7 * tol_factor)
    np.testing.assert_allclose(tt.xim_im, 0, atol=3.e-7 * tol_factor)

    # Check that we get the same result using the corr2 function:
    cat.write(os.path.join('data','tt.dat'))
    config = treecorr.read_config('configs/tt.yaml')
    config['verbose'] = 0
    config['precision'] = 8
    treecorr.corr2(config)
    corr2_output = np.genfromtxt(os.path.join('output','tt.out'), names=True, skip_header=1)
    print('tt.xip = ',tt.xip)
    print('from corr2 output = ',corr2_output['xip'])
    print('ratio = ',corr2_output['xip']/tt.xip)
    print('diff = ',corr2_output['xip']-tt.xip)
    np.testing.assert_allclose(corr2_output['xip'], tt.xip, rtol=1.e-4)

    print('tt.xim = ',tt.xim)
    print('from corr2 output = ',corr2_output['xim'])
    print('ratio = ',corr2_output['xim']/tt.xim)
    print('diff = ',corr2_output['xim']-tt.xim)
    np.testing.assert_allclose(corr2_output['xim'], tt.xim, rtol=1.e-4)

    print('xip_im from corr2 output = ',corr2_output['xip_im'])
    print('max err = ',max(abs(corr2_output['xip_im'])))
    np.testing.assert_allclose(corr2_output['xip_im'], 0, atol=3.e-7 * tol_factor)
    print('xim_im from corr2 output = ',corr2_output['xim_im'])
    print('max err = ',max(abs(corr2_output['xim_im'])))
    np.testing.assert_allclose(corr2_output['xim_im'], 0, atol=3.e-7 * tol_factor)

    # Check the fits write option
    out_file_name = os.path.join('output','tt_out.dat')
    tt.write(out_file_name, precision=16)
    data = np.genfromtxt(out_file_name, names=True, skip_header=1)
    np.testing.assert_allclose(data['r_nom'], np.exp(tt.logr))
    np.testing.assert_allclose(data['meanr'], tt.meanr)
    np.testing.assert_allclose(data['meanlogr'], tt.meanlogr)
    np.testing.assert_allclose(data['xip'], tt.xip)
    np.testing.assert_allclose(data['xim'], tt.xim)
    np.testing.assert_allclose(data['xip_im'], tt.xip_im)
    np.testing.assert_allclose(data['xim_im'], tt.xim_im)
    np.testing.assert_allclose(data['sigma_xip'], np.sqrt(tt.varxip))
    np.testing.assert_allclose(data['sigma_xim'], np.sqrt(tt.varxim))
    np.testing.assert_allclose(data['weight'], tt.weight)
    np.testing.assert_allclose(data['npairs'], tt.npairs)

    # Check the read function
    tt2 = treecorr.TTCorrelation.from_file(out_file_name)
    np.testing.assert_allclose(tt2.logr, tt.logr)
    np.testing.assert_allclose(tt2.meanr, tt.meanr)
    np.testing.assert_allclose(tt2.meanlogr, tt.meanlogr)
    np.testing.assert_allclose(tt2.xip, tt.xip)
    np.testing.assert_allclose(tt2.xim, tt.xim)
    np.testing.assert_allclose(tt2.xip_im, tt.xip_im)
    np.testing.assert_allclose(tt2.xim_im, tt.xim_im)
    np.testing.assert_allclose(tt2.varxip, tt.varxip)
    np.testing.assert_allclose(tt2.varxim, tt.varxim)
    np.testing.assert_allclose(tt2.weight, tt.weight)
    np.testing.assert_allclose(tt2.npairs, tt.npairs)
    assert tt2.coords == tt.coords
    assert tt2.metric == tt.metric
    assert tt2.sep_units == tt.sep_units
    assert tt2.bin_type == tt.bin_type


@timer
def test_spherical():
    # This is the same field we used for test_tt, but put into spherical coords.
    # We do the spherical trig by hand using the obvious formulae, rather than the clever
    # optimizations that are used by the TreeCorr code, thus serving as a useful test of
    # the latter.

    t0 = 0.05
    r0 = 10. * coord.arcmin / coord.radians
    if __name__ == "__main__":
        nsource = 1000000
        L = 50.*r0  # Not infinity, so this introduces some error.  Our integrals were to infinity.
        tol_factor = 1
    else:
        nsource = 100000
        L = 50.*r0
        tol_factor = 5
    rng = np.random.RandomState(8675309)
    x = (rng.random_sample(nsource)-0.5) * L
    y = (rng.random_sample(nsource)-0.5) * L
    r2 = x**2 + y**2
    r = np.sqrt(r2)
    theta = np.arctan2(y,x)
    t1 = t0 * (r/r0)**3 * np.exp(-r2/2./r0**2) * np.cos(3*theta)
    t2 = t0 * (r/r0)**3 * np.exp(-r2/2./r0**2) * np.sin(3*theta)

    tt = treecorr.TTCorrelation(bin_size=0.1, min_sep=10., max_sep=100., sep_units='arcmin',
                                verbose=1)
    r1 = np.exp(tt.logr) * (coord.arcmin / coord.radians)
    temp = np.pi/64. * t0**2 * (r0/L)**2 * np.exp(-0.25*r1**2/r0**2)
    true_xip = temp * (384 - 288*(r1/r0)**2 + 36*(r1/r0)**4 - (r1/r0)**6)
    true_xim = -temp * (r1/r0)**6

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
        # t_sph = exp(3i beta) * v
        # where beta = pi - (A+B) is the angle between north and "up" in the tangent plane.
        beta = np.pi - (A+B)
        beta[x>0] *= -1.
        cos3beta = np.cos(3*beta)
        sin3beta = np.sin(3*beta)
        t1_sph = t1 * cos3beta - t2 * sin3beta
        t2_sph = t2 * cos3beta + t1 * sin3beta

        cat = treecorr.Catalog(ra=ra, dec=dec, t1=t1_sph, t2=t2_sph, ra_units='rad',
                               dec_units='rad')
        tt = treecorr.TTCorrelation(bin_size=0.1, min_sep=10., max_sep=100., sep_units='arcmin',
                                    verbose=1)
        tt.process(cat)

        print('ra0, dec0 = ',ra0,dec0)
        print('tt.xip = ',tt.xip)
        print('true_xip = ',true_xip)
        print('ratio = ',tt.xip / true_xip)
        print('diff = ',tt.xip - true_xip)
        print('max diff = ',max(abs(tt.xip - true_xip)))
        assert max(abs(tt.xip - true_xip)) < 4.e-7 * tol_factor

        print('tt.xim = ',tt.xim)
        print('true_xim = ',true_xim)
        print('ratio = ',tt.xim / true_xim)
        print('diff = ',tt.xim - true_xim)
        print('max diff = ',max(abs(tt.xim - true_xim)))
        assert max(abs(tt.xim - true_xim)) < 5.e-7 * tol_factor

    # One more center that can be done very easily.  If the center is the north pole, then all
    # the radial spin-3 values are pure positive t1.
    ra0 = 0
    dec0 = np.pi/2.
    ra = theta
    dec = np.pi/2. - 2.*np.arcsin(r/2.)
    trad = t0 * (r/r0)**3 * np.exp(-r2/2./r0**2)

    cat = treecorr.Catalog(ra=ra, dec=dec, t1=np.zeros_like(trad), t2=trad, ra_units='rad',
                           dec_units='rad')
    tt.process(cat)

    print('tt.xip = ',tt.xip)
    print('tt.xip_im = ',tt.xip_im)
    print('true_xip = ',true_xip)
    print('ratio = ',tt.xip / true_xip)
    print('diff = ',tt.xip - true_xip)
    print('max diff = ',max(abs(tt.xip - true_xip)))
    assert max(abs(tt.xip - true_xip)) < 4.e-7 * tol_factor
    assert max(abs(tt.xip_im)) < 3.e-7 * tol_factor

    print('tt.xim = ',tt.xim)
    print('tt.xim_im = ',tt.xim_im)
    print('true_xim = ',true_xim)
    print('ratio = ',tt.xim / true_xim)
    print('diff = ',tt.xim - true_xim)
    print('max diff = ',max(abs(tt.xim - true_xim)))
    assert max(abs(tt.xim - true_xim)) < 5.e-7 * tol_factor
    assert max(abs(tt.xim_im)) < 2.e-7 * tol_factor

    # Check that we get the same result using the corr2 function
    cat.write(os.path.join('data','tt_spherical.dat'))
    config = treecorr.read_config('configs/tt_spherical.yaml')
    config['verbose'] = 0
    treecorr.corr2(config)
    corr2_output = np.genfromtxt(os.path.join('output','tt_spherical.out'), names=True,
                                 skip_header=1)
    print('tt.xip = ',tt.xip)
    print('from corr2 output = ',corr2_output['xip'])
    print('ratio = ',corr2_output['xip']/tt.xip)
    print('diff = ',corr2_output['xip']-tt.xip)
    np.testing.assert_allclose(corr2_output['xip'], tt.xip)

    print('tt.xim = ',tt.xim)
    print('from corr2 output = ',corr2_output['xim'])
    print('ratio = ',corr2_output['xim']/tt.xim)
    print('diff = ',corr2_output['xim']-tt.xim)
    np.testing.assert_allclose(corr2_output['xim'], tt.xim)

    print('xip_im from corr2 output = ',corr2_output['xip_im'])
    assert max(abs(corr2_output['xip_im'])) < 3.e-7 * tol_factor

    print('xim_im from corr2 output = ',corr2_output['xim_im'])
    assert max(abs(corr2_output['xim_im'])) < 2.e-7 * tol_factor


@timer
def test_varxi():
    # Test that varxip, varxim are correct (or close) based on actual variance of many runs.

    # Same t pattern as in test_tt().  Although the signal doesn't actually matter at all here.
    t0 = 0.05
    r0 = 10.
    L = 50.*r0
    rng = np.random.RandomState(8675309)

    # Note: to get a good estimate of var(xi), you need a lot of runs.  The number of
    # runs matters much more than the number of galaxies for getting this to pass.
    ngal = 1000
    nruns = 50000

    file_name = 'data/test_varxi_tt.npz'
    print(file_name)
    if not os.path.isfile(file_name):
        all_tts = []

        for run in range(nruns):
            print(f'{run}/{nruns}')
            # In addition to the shape noise below, there is shot noise from the random x,y positions.
            x = (rng.random_sample(ngal)-0.5) * L
            y = (rng.random_sample(ngal)-0.5) * L
            # Varied weights are hard, but at least check that non-unit weights work correctly.
            w = np.ones_like(x) * 5
            r2 = (x**2 + y**2)/r0**2
            theta = np.arctan2(y,x)
            t1 = t0 * np.exp(-r2/2.) * np.cos(3*theta)
            t2 = t0 * np.exp(-r2/2.) * np.sin(3*theta)
            # This time, add some shape noise (different each run).
            t1 += rng.normal(0, 0.3, size=ngal)
            t2 += rng.normal(0, 0.3, size=ngal)

            cat = treecorr.Catalog(x=x, y=y, w=w, t1=t1, t2=t2, x_units='arcmin', y_units='arcmin')
            tt = treecorr.TTCorrelation(bin_size=0.1, min_sep=10., max_sep=100., sep_units='arcmin',
                                        verbose=1)
            tt.process(cat)
            all_tts.append(tt)

        mean_xip = np.mean([tt.xip for tt in all_tts], axis=0)
        var_xip = np.var([tt.xip for tt in all_tts], axis=0)
        mean_xim = np.mean([tt.xim for tt in all_tts], axis=0)
        var_xim = np.var([tt.xim for tt in all_tts], axis=0)
        mean_varxip = np.mean([tt.varxip for tt in all_tts], axis=0)
        mean_varxim = np.mean([tt.varxim for tt in all_tts], axis=0)

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
    t1 = t0 * np.exp(-r2/2.) * np.cos(3*theta)
    t2 = t0 * np.exp(-r2/2.) * np.sin(3*theta)
    # This time, add some shape noise (different each run).
    t1 += rng.normal(0, 0.3, size=ngal)
    t2 += rng.normal(0, 0.3, size=ngal)

    cat = treecorr.Catalog(x=x, y=y, w=w, t1=t1, t2=t2, x_units='arcmin', y_units='arcmin')
    tt = treecorr.TTCorrelation(bin_size=0.1, min_sep=10., max_sep=100., sep_units='arcmin',
                                verbose=1)
    tt.process(cat)
    print('single run:')
    print('max relerr for xip = ',np.max(np.abs((tt.varxip - var_xip)/var_xip)))
    print('max relerr for xim = ',np.max(np.abs((tt.varxip - var_xim)/var_xim)))
    np.testing.assert_allclose(tt.varxip, var_xip, rtol=0.2)
    np.testing.assert_allclose(tt.varxim, var_xim, rtol=0.2)

@timer
def test_jk():

    # Same multi-lens field we used for NT patch test
    t0 = 0.05
    r0 = 30.
    L = 30 * r0
    rng = np.random.RandomState(8675309)

    nsource = 100000
    nlens = 300
    nruns = 1000
    npatch = 64

    corr_params = dict(bin_size=0.3, min_sep=60, max_sep=200, bin_slop=0.1)

    def make_spin3_field(rng):
        x1 = (rng.random(nlens)-0.5) * L
        y1 = (rng.random(nlens)-0.5) * L
        w = rng.random(nlens) + 10
        x2 = (rng.random(nsource)-0.5) * L
        y2 = (rng.random(nsource)-0.5) * L

        # Start with just the noise
        t1 = rng.normal(0, 0.1, size=nsource)
        t2 = rng.normal(0, 0.1, size=nsource)

        # Also a non-zero background constant field
        t1 += 2*t0
        t2 -= 3*t0

        # Add in the signal from all lenses
        for i in range(nlens):
            x2i = x2 - x1[i]
            y2i = y2 - y1[i]
            r2 = (x2i**2 + y2i**2)/r0**2
            theta = np.arctan2(y2i,x2i)
            t1 += w[i] * t0 * r2**1.5 * np.exp(-r2/2) * np.cos(3*theta)
            t2 += w[i] * t0 * r2**1.5 * np.exp(-r2/2) * np.sin(3*theta)
        return x1, y1, w, x2, y2, t1, t2

    file_name = 'data/test_tt_jk_{}.npz'.format(nruns)
    print(file_name)
    if not os.path.isfile(file_name):
        all_tts = []
        rng = np.random.default_rng()
        for run in range(nruns):
            x1, y1, w, x2, y2, t1, t2 = make_spin3_field(rng)
            print(run,': ',np.mean(t1),np.std(t1),np.min(t1),np.max(t1))
            cat = treecorr.Catalog(x=x2, y=y2, t1=t1, t2=t2)
            tt = treecorr.TTCorrelation(corr_params)
            tt.process(cat)
            all_tts.append(tt)

        mean_xip = np.mean([tt.xip for tt in all_tts], axis=0)
        mean_xim = np.mean([tt.xim for tt in all_tts], axis=0)
        var_xip = np.var([tt.xip for tt in all_tts], axis=0)
        var_xim = np.var([tt.xim for tt in all_tts], axis=0)
        mean_varxip = np.mean([tt.varxip for tt in all_tts], axis=0)
        mean_varxim = np.mean([tt.varxim for tt in all_tts], axis=0)

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
    x1, y1, w, x2, y2, t1, t2 = make_spin3_field(rng)

    cat = treecorr.Catalog(x=x2, y=y2, t1=t1, t2=t2)
    tt1 = treecorr.TTCorrelation(corr_params)
    tt1.process(cat)

    print('weight = ',tt1.weight)
    print('xip = ',tt1.xip)
    print('varxip = ',tt1.varxip)
    print('pullsq for xip = ',(tt1.xip-mean_xip)**2/var_xip)
    print('max pull for xip = ',np.sqrt(np.max((tt1.xip-mean_xip)**2/var_xip)))
    print('max pull for xim = ',np.sqrt(np.max((tt1.xim-mean_xim)**2/var_xim)))
    np.testing.assert_array_less((tt1.xip-mean_xip)**2, 9*var_xip)  # < 3 sigma pull
    np.testing.assert_array_less((tt1.xim-mean_xim)**2, 9*var_xim)  # < 3 sigma pull
    np.testing.assert_allclose(tt1.varxip, mean_varxip, rtol=0.2)
    np.testing.assert_allclose(tt1.varxim, mean_varxim, rtol=0.2)

    # Now run with patches, but still with shot variance.  Should be basically the same answer.
    catp = treecorr.Catalog(x=x2, y=y2, t1=t1, t2=t2, npatch=npatch)
    print('tot w = ',np.sum(w))
    print('Patch\tNsource')
    for i in range(npatch):
        print('%d\t%d'%(i,np.sum(catp.w[catp.patch==i])))
    tt2 = treecorr.TTCorrelation(corr_params)
    tt2.process(catp)
    print('weight = ',tt2.weight)
    print('xip = ',tt2.xip)
    print('xip1 = ',tt1.xip)
    print('varxip = ',tt2.varxip)
    print('xim = ',tt2.xim)
    print('xim1 = ',tt1.xim)
    print('varxim = ',tt2.varxim)
    np.testing.assert_allclose(tt2.weight, tt1.weight, rtol=1.e-2)
    np.testing.assert_allclose(tt2.xip, tt1.xip, rtol=3.e-2)
    np.testing.assert_allclose(tt2.xim, tt1.xim, rtol=3.e-2)
    np.testing.assert_allclose(tt2.varxip, tt1.varxip, rtol=1.e-2)
    np.testing.assert_allclose(tt2.varxim, tt1.varxim, rtol=1.e-2)

    # Now try jackknife variance estimate.
    cov2 = tt2.estimate_cov('jackknife')
    print('cov.diag = ',np.diagonal(cov2))
    print('cf var_xip = ',var_xip)
    print('cf var_xim = ',var_xim)
    np.testing.assert_allclose(np.diagonal(cov2)[:5], var_xip, rtol=0.6)
    np.testing.assert_allclose(np.diagonal(cov2)[5:], var_xim, rtol=0.5)

    # Use initialize/finalize
    tt3 = treecorr.TTCorrelation(corr_params)
    for k1, p1 in enumerate(catp.get_patches()):
        tt3.process(p1, initialize=(k1==0), finalize=(k1==npatch-1))
        for k2, p2 in enumerate(catp.get_patches()):
            if k2 <= k1: continue
            tt3.process(p1, p2, initialize=False, finalize=False)
    np.testing.assert_allclose(tt3.xip, tt2.xip)
    np.testing.assert_allclose(tt3.xim, tt2.xim)
    np.testing.assert_allclose(tt3.weight, tt2.weight)

    # Check that these still work after roundtripping through a file.
    try:
        import fitsio
    except ImportError:
        pass
    else:
        file_name = os.path.join('output','test_write_results_tt.fits')
        tt2.write(file_name, write_patch_results=True)
        tt3 = treecorr.TTCorrelation.from_file(file_name)
        cov3 = tt3.estimate_cov('jackknife')
        np.testing.assert_allclose(cov3, cov2)

    # Check some invalid actions
    # Bad var_method
    with assert_raises(ValueError):
        tt2.estimate_cov('invalid')
    # Not run on patches, but need patches
    with assert_raises(ValueError):
        tt1.estimate_cov('jackknife')
    with assert_raises(ValueError):
        tt1.estimate_cov('sample')
    with assert_raises(ValueError):
        tt1.estimate_cov('marked_bootstrap')
    with assert_raises(ValueError):
        tt1.estimate_cov('bootstrap')

    cata = treecorr.Catalog(x=x2[:100], y=y2[:100], t1=t1[:100], t2=t2[:100], npatch=10)
    catb = treecorr.Catalog(x=x2[:100], y=y2[:100], t1=t1[:100], t2=t2[:100], npatch=2)
    tt4 = treecorr.TTCorrelation(corr_params)
    tt5 = treecorr.TTCorrelation(corr_params)
    # All catalogs need to have the same number of patches
    with assert_raises(RuntimeError):
        tt4.process(cata,catb)
    with assert_raises(RuntimeError):
        tt5.process(catb,cata)

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
    t1 = rng.multivariate_normal(np.zeros(N), K*(A**2))
    t1 += rng.normal(scale=sigma, size=N)
    t2 = rng.multivariate_normal(np.zeros(N), K*(A**2))
    t2 += rng.normal(scale=sigma, size=N)
    t = t1 + 1j * t2

    # Calculate the 2D correlation using brute force
    max_sep = 21.
    nbins = 21
    xi_brut = corr2d(x, y, t, np.conj(t), rmax=max_sep, bins=nbins)

    # And using TreeCorr
    cat = treecorr.Catalog(x=x, y=y, t1=t1, t2=t2)
    tt = treecorr.TTCorrelation(max_sep=max_sep, bin_size=2., bin_type='TwoD', brute=True)
    tt.process(cat)
    print('max abs diff = ',np.max(np.abs(tt.xip - xi_brut)))
    print('max rel diff = ',np.max(np.abs(tt.xip - xi_brut)/np.abs(tt.xip)))
    np.testing.assert_allclose(tt.xip, xi_brut, atol=2.e-7)

    tt = treecorr.TTCorrelation(max_sep=max_sep, bin_size=2., bin_type='TwoD', bin_slop=0.05)
    tt.process(cat)
    print('max abs diff = ',np.max(np.abs(tt.xip - xi_brut)))
    print('max rel diff = ',np.max(np.abs(tt.xip - xi_brut)/np.abs(tt.xip)))
    np.testing.assert_allclose(tt.xip, xi_brut, atol=2.e-7)

    # Check I/O
    try:
        import fitsio
    except ImportError:
        pass
    else:
        fits_name = 'output/tt_twod.fits'
        tt.write(fits_name)
        tt2 = treecorr.TTCorrelation.from_file(fits_name)
        np.testing.assert_allclose(tt2.npairs, tt.npairs)
        np.testing.assert_allclose(tt2.weight, tt.weight)
        np.testing.assert_allclose(tt2.meanr, tt.meanr)
        np.testing.assert_allclose(tt2.meanlogr, tt.meanlogr)
        np.testing.assert_allclose(tt2.xip, tt.xip)
        np.testing.assert_allclose(tt2.xip_im, tt.xip_im)
        np.testing.assert_allclose(tt2.xim, tt.xim)
        np.testing.assert_allclose(tt2.xim_im, tt.xim_im)

    ascii_name = 'output/tt_twod.txt'
    tt.write(ascii_name, precision=16)
    tt3 = treecorr.TTCorrelation.from_file(ascii_name)
    np.testing.assert_allclose(tt3.npairs, tt.npairs)
    np.testing.assert_allclose(tt3.weight, tt.weight)
    np.testing.assert_allclose(tt3.meanr, tt.meanr)
    np.testing.assert_allclose(tt3.meanlogr, tt.meanlogr)
    np.testing.assert_allclose(tt3.xip, tt.xip)
    np.testing.assert_allclose(tt3.xip_im, tt.xip_im)
    np.testing.assert_allclose(tt3.xim, tt.xim)
    np.testing.assert_allclose(tt3.xim_im, tt.xim_im)


if __name__ == '__main__':
    test_direct()
    test_direct_spherical()
    test_tt()
    test_spherical()
    test_varxi()
    test_jk()
    test_twod()
