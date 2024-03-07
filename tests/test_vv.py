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
    v11 = rng.normal(0,0.2, (ngal,) )
    v21 = rng.normal(0,0.2, (ngal,) )

    x2 = rng.normal(0,s, (ngal,) )
    y2 = rng.normal(0,s, (ngal,) )
    w2 = rng.random_sample(ngal)
    v12 = rng.normal(0,0.2, (ngal,) )
    v22 = rng.normal(0,0.2, (ngal,) )

    cat1 = treecorr.Catalog(x=x1, y=y1, w=w1, v1=v11, v2=v21)
    cat2 = treecorr.Catalog(x=x2, y=y2, w=w2, v1=v12, v2=v22)

    min_sep = 1.
    max_sep = 50.
    nbins = 50
    bin_size = np.log(max_sep/min_sep) / nbins
    vv = treecorr.VVCorrelation(min_sep=min_sep, max_sep=max_sep, nbins=nbins, brute=True)
    vv.process(cat1, cat2)

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
        xip = ww * (v11[i] + 1j*v21[i]) * (v12 - 1j*v22)
        xim = ww * (v11[i] + 1j*v21[i]) * (v12 + 1j*v22) * expmialpha**2

        index = np.floor(np.log(r/min_sep) / bin_size).astype(int)
        mask = (index >= 0) & (index < nbins)
        np.add.at(true_npairs, index[mask], 1)
        np.add.at(true_weight, index[mask], ww[mask])
        np.add.at(true_xip, index[mask], xip[mask])
        np.add.at(true_xim, index[mask], xim[mask])

    true_xip /= true_weight
    true_xim /= true_weight

    print('true_npairs = ',true_npairs)
    print('diff = ',vv.npairs - true_npairs)
    np.testing.assert_array_equal(vv.npairs, true_npairs)

    print('true_weight = ',true_weight)
    print('diff = ',vv.weight - true_weight)
    np.testing.assert_allclose(vv.weight, true_weight, rtol=1.e-6, atol=1.e-8)

    print('true_xip = ',true_xip)
    print('vv.xip = ',vv.xip)
    print('vv.xip_im = ',vv.xip_im)
    np.testing.assert_allclose(vv.xip, true_xip.real, rtol=1.e-6, atol=1.e-8)
    np.testing.assert_allclose(vv.xip_im, true_xip.imag, rtol=1.e-6, atol=1.e-8)
    print('true_xim = ',true_xim)
    print('vv.xim = ',vv.xim)
    print('vv.xim_im = ',vv.xim_im)
    np.testing.assert_allclose(vv.xim, true_xim.real, rtol=1.e-6, atol=1.e-8)
    np.testing.assert_allclose(vv.xim_im, true_xim.imag, rtol=1.e-6, atol=1.e-8)

    # Check that running via the corr2 script works correctly.
    config = treecorr.config.read_config('configs/vv_direct.yaml')
    try:
        import fitsio
    except ImportError:
        pass
    else:
        cat1.write(config['file_name'])
        cat2.write(config['file_name2'])
        treecorr.corr2(config)
        data = fitsio.read(config['vv_file_name'])
        np.testing.assert_allclose(data['r_nom'], vv.rnom)
        np.testing.assert_allclose(data['npairs'], vv.npairs)
        np.testing.assert_allclose(data['weight'], vv.weight)
        np.testing.assert_allclose(data['xip'], vv.xip)
        np.testing.assert_allclose(data['xip_im'], vv.xip_im)
        np.testing.assert_allclose(data['xim'], vv.xim)
        np.testing.assert_allclose(data['xim_im'], vv.xim_im)

    # Repeat with binslop = 0.
    # And don't do any top-level recursion so we actually test not going to the leaves.
    vv = treecorr.VVCorrelation(min_sep=min_sep, max_sep=max_sep, nbins=nbins, bin_slop=0,
                                max_top=0)
    vv.process(cat1, cat2)
    np.testing.assert_array_equal(vv.npairs, true_npairs)
    np.testing.assert_allclose(vv.weight, true_weight, rtol=1.e-6, atol=1.e-8)
    np.testing.assert_allclose(vv.xip, true_xip.real, rtol=1.e-6, atol=1.e-8)
    np.testing.assert_allclose(vv.xip_im, true_xip.imag, rtol=1.e-6, atol=1.e-8)
    print('true_xim = ',true_xim)
    print('vv.xim = ',vv.xim)
    print('vv.xim_im = ',vv.xim_im)
    print('diff = ',vv.xim - true_xim.real)
    print('max diff = ',np.max(np.abs(vv.xim - true_xim.real)))
    print('rel diff = ',(vv.xim - true_xim.real)/true_xim.real)
    np.testing.assert_allclose(vv.xim, true_xim.real, atol=3.e-4)
    np.testing.assert_allclose(vv.xim_im, true_xim.imag, atol=1.e-4)

    # With angle_slop = 0, it goes back to being basically exact (to single precision).
    vv = treecorr.VVCorrelation(min_sep=min_sep, max_sep=max_sep, nbins=nbins, bin_slop=0,
                                angle_slop=0, max_top=0)
    vv.process(cat1, cat2)
    np.testing.assert_array_equal(vv.npairs, true_npairs)
    np.testing.assert_allclose(vv.weight, true_weight, rtol=1.e-6, atol=1.e-8)
    np.testing.assert_allclose(vv.xip, true_xip.real, rtol=1.e-6, atol=1.e-8)
    np.testing.assert_allclose(vv.xip_im, true_xip.imag, rtol=1.e-6, atol=1.e-8)
    np.testing.assert_allclose(vv.xim, true_xim.real, rtol=1.e-6, atol=1.e-8)
    np.testing.assert_allclose(vv.xim_im, true_xim.imag, rtol=1.e-6, atol=1.e-8)

    # Check a few basic operations with a VVCorrelation object.
    do_pickle(vv)

    vv2 = vv.copy()
    vv2 += vv
    np.testing.assert_allclose(vv2.npairs, 2*vv.npairs)
    np.testing.assert_allclose(vv2.weight, 2*vv.weight)
    np.testing.assert_allclose(vv2.meanr, 2*vv.meanr)
    np.testing.assert_allclose(vv2.meanlogr, 2*vv.meanlogr)
    np.testing.assert_allclose(vv2.xip, 2*vv.xip)
    np.testing.assert_allclose(vv2.xip_im, 2*vv.xip_im)
    np.testing.assert_allclose(vv2.xim, 2*vv.xim)
    np.testing.assert_allclose(vv2.xim_im, 2*vv.xim_im)

    vv2.clear()
    vv2 += vv
    np.testing.assert_allclose(vv2.npairs, vv.npairs)
    np.testing.assert_allclose(vv2.weight, vv.weight)
    np.testing.assert_allclose(vv2.meanr, vv.meanr)
    np.testing.assert_allclose(vv2.meanlogr, vv.meanlogr)
    np.testing.assert_allclose(vv2.xip, vv.xip)
    np.testing.assert_allclose(vv2.xip_im, vv.xip_im)
    np.testing.assert_allclose(vv2.xim, vv.xim)
    np.testing.assert_allclose(vv2.xim_im, vv.xim_im)

    ascii_name = 'output/vv_ascii.txt'
    vv.write(ascii_name, precision=16)
    vv3 = treecorr.VVCorrelation(min_sep=min_sep, max_sep=max_sep, nbins=nbins, bin_type='Log')
    vv3.read(ascii_name)
    np.testing.assert_allclose(vv3.npairs, vv.npairs)
    np.testing.assert_allclose(vv3.weight, vv.weight)
    np.testing.assert_allclose(vv3.meanr, vv.meanr)
    np.testing.assert_allclose(vv3.meanlogr, vv.meanlogr)
    np.testing.assert_allclose(vv3.xip, vv.xip)
    np.testing.assert_allclose(vv3.xip_im, vv.xip_im)
    np.testing.assert_allclose(vv3.xim, vv.xim)
    np.testing.assert_allclose(vv3.xim_im, vv.xim_im)

    # Check that the repr is minimal
    assert repr(vv3) == f'VVCorrelation(min_sep={min_sep}, max_sep={max_sep}, nbins={nbins})'

    # Simpler API using from_file:
    with CaptureLog() as cl:
        vv3b = treecorr.VVCorrelation.from_file(ascii_name, logger=cl.logger)
    assert ascii_name in cl.output
    np.testing.assert_allclose(vv3b.npairs, vv.npairs)
    np.testing.assert_allclose(vv3b.weight, vv.weight)
    np.testing.assert_allclose(vv3b.meanr, vv.meanr)
    np.testing.assert_allclose(vv3b.meanlogr, vv.meanlogr)
    np.testing.assert_allclose(vv3b.xip, vv.xip)
    np.testing.assert_allclose(vv3b.xip_im, vv.xip_im)
    np.testing.assert_allclose(vv3b.xim, vv.xim)
    np.testing.assert_allclose(vv3b.xim_im, vv.xim_im)

    # or using the Corr2 base class
    with CaptureLog() as cl:
        vv3c = treecorr.Corr2.from_file(ascii_name, logger=cl.logger)
    assert ascii_name in cl.output
    np.testing.assert_allclose(vv3c.npairs, vv.npairs)
    np.testing.assert_allclose(vv3c.weight, vv.weight)
    np.testing.assert_allclose(vv3c.meanr, vv.meanr)
    np.testing.assert_allclose(vv3c.meanlogr, vv.meanlogr)
    np.testing.assert_allclose(vv3c.xip, vv.xip)
    np.testing.assert_allclose(vv3c.xip_im, vv.xip_im)
    np.testing.assert_allclose(vv3c.xim, vv.xim)
    np.testing.assert_allclose(vv3c.xim_im, vv.xim_im)

    try:
        import fitsio
    except ImportError:
        pass
    else:
        fits_name = 'output/vv_fits.fits'
        vv.write(fits_name)
        vv4 = treecorr.VVCorrelation(min_sep=min_sep, max_sep=max_sep, nbins=nbins)
        vv4.read(fits_name)
        np.testing.assert_allclose(vv4.npairs, vv.npairs)
        np.testing.assert_allclose(vv4.weight, vv.weight)
        np.testing.assert_allclose(vv4.meanr, vv.meanr)
        np.testing.assert_allclose(vv4.meanlogr, vv.meanlogr)
        np.testing.assert_allclose(vv4.xip, vv.xip)
        np.testing.assert_allclose(vv4.xip_im, vv.xip_im)
        np.testing.assert_allclose(vv4.xim, vv.xim)
        np.testing.assert_allclose(vv4.xim_im, vv.xim_im)

        vv4b = treecorr.VVCorrelation.from_file(fits_name)
        np.testing.assert_allclose(vv4b.npairs, vv.npairs)
        np.testing.assert_allclose(vv4b.weight, vv.weight)
        np.testing.assert_allclose(vv4b.meanr, vv.meanr)
        np.testing.assert_allclose(vv4b.meanlogr, vv.meanlogr)
        np.testing.assert_allclose(vv4b.xip, vv.xip)
        np.testing.assert_allclose(vv4b.xip_im, vv.xip_im)
        np.testing.assert_allclose(vv4b.xim, vv.xim)
        np.testing.assert_allclose(vv4b.xim_im, vv.xim_im)

    with assert_raises(TypeError):
        vv2 += config
    vv4 = treecorr.VVCorrelation(min_sep=min_sep/2, max_sep=max_sep, nbins=nbins)
    with assert_raises(ValueError):
        vv2 += vv4
    vv5 = treecorr.VVCorrelation(min_sep=min_sep, max_sep=max_sep*2, nbins=nbins)
    with assert_raises(ValueError):
        vv2 += vv5
    vv6 = treecorr.VVCorrelation(min_sep=min_sep, max_sep=max_sep, nbins=nbins*2)
    with assert_raises(ValueError):
        vv2 += vv6
    with assert_raises(ValueError):
        vv.process(cat1, cat2, patch_method='nonlocal')

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
    v11 = rng.normal(0,0.2, (ngal,) )
    v21 = rng.normal(0,0.2, (ngal,) )

    x2 = rng.normal(0,s, (ngal,) )
    y2 = rng.normal(0,s, (ngal,) ) + 200
    z2 = rng.normal(0,s, (ngal,) )
    w2 = rng.random_sample(ngal)
    v12 = rng.normal(0,0.2, (ngal,) )
    v22 = rng.normal(0,0.2, (ngal,) )

    ra1, dec1 = coord.CelestialCoord.xyz_to_radec(x1,y1,z1)
    ra2, dec2 = coord.CelestialCoord.xyz_to_radec(x2,y2,z2)

    cat1 = treecorr.Catalog(ra=ra1, dec=dec1, ra_units='rad', dec_units='rad', w=w1, v1=v11, v2=v21)
    cat2 = treecorr.Catalog(ra=ra2, dec=dec2, ra_units='rad', dec_units='rad', w=w2, v1=v12, v2=v22)

    min_sep = 1.
    max_sep = 10.
    nbins = 50
    bin_size = np.log(max_sep/min_sep) / nbins
    vv = treecorr.VVCorrelation(min_sep=min_sep, max_sep=max_sep, nbins=nbins,
                                sep_units='deg', brute=True)
    vv.process(cat1, cat2)

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
            exptheta1 = np.cos(theta1) + 1j * np.sin(theta1)
            exptheta2 = np.cos(theta2) + 1j * np.sin(theta2)

            v1 = v11[i] + 1j * v21[i]
            v2 = v12[j] + 1j * v22[j]
            v1 *= exptheta1
            v2 *= exptheta2

            ww = w1[i] * w2[j]
            xip = ww * v1 * np.conjugate(v2)
            xim = ww * v1 * v2

            true_npairs[index] += 1
            true_weight[index] += ww
            true_xip[index] += xip
            true_xim[index] += xim

    true_xip /= true_weight
    true_xim /= true_weight

    print('true_npairs = ',true_npairs)
    print('diff = ',vv.npairs - true_npairs)
    np.testing.assert_array_equal(vv.npairs, true_npairs)

    print('true_weight = ',true_weight)
    print('diff = ',vv.weight - true_weight)
    np.testing.assert_allclose(vv.weight, true_weight, rtol=1.e-6, atol=1.e-8)

    print('true_xip = ',true_xip)
    print('vv.xip = ',vv.xip)
    print('vv.xip_im = ',vv.xip_im)
    np.testing.assert_allclose(vv.xip, true_xip.real, rtol=1.e-6, atol=1.e-8)
    np.testing.assert_allclose(vv.xip_im, true_xip.imag, rtol=1.e-6, atol=1.e-8)
    print('true_xim = ',true_xim)
    print('vv.xim = ',vv.xim)
    print('vv.xim_im = ',vv.xim_im)
    np.testing.assert_allclose(vv.xim, true_xim.real, rtol=1.e-6, atol=1.e-8)
    np.testing.assert_allclose(vv.xim_im, true_xim.imag, rtol=1.e-6, atol=1.e-8)

    # Check that running via the corr2 script works correctly.
    try:
        import fitsio
    except ImportError:
        pass
    else:
        config = treecorr.config.read_config('configs/vv_direct_spherical.yaml')
        cat1.write(config['file_name'])
        cat2.write(config['file_name2'])
        treecorr.corr2(config)
        data = fitsio.read(config['vv_file_name'])
        np.testing.assert_allclose(data['r_nom'], vv.rnom)
        np.testing.assert_allclose(data['npairs'], vv.npairs)
        np.testing.assert_allclose(data['weight'], vv.weight)
        np.testing.assert_allclose(data['xip'], vv.xip)
        np.testing.assert_allclose(data['xip_im'], vv.xip_im)
        np.testing.assert_allclose(data['xim'], vv.xim)
        np.testing.assert_allclose(data['xim_im'], vv.xim_im)

    # Repeat with binslop = 0
    # And don't do any top-level recursion so we actually test not going to the leaves.
    vv = treecorr.VVCorrelation(min_sep=min_sep, max_sep=max_sep, nbins=nbins,
                                sep_units='deg', bin_slop=0, max_top=0)
    vv.process(cat1, cat2)
    np.testing.assert_array_equal(vv.npairs, true_npairs)
    np.testing.assert_allclose(vv.weight, true_weight, rtol=1.e-6, atol=1.e-8)
    np.testing.assert_allclose(vv.xip, true_xip.real, rtol=1.e-6, atol=3.e-7)
    np.testing.assert_allclose(vv.xip_im, true_xip.imag, rtol=1.e-6, atol=2.e-7)
    np.testing.assert_allclose(vv.xim, true_xim.real, atol=1.e-4)
    np.testing.assert_allclose(vv.xim_im, true_xim.imag, atol=1.e-4)

    # With angle_slop = 0, it goes back to being basically exact (to single precision).
    vv = treecorr.VVCorrelation(min_sep=min_sep, max_sep=max_sep, nbins=nbins,
                                sep_units='deg', bin_slop=0, angle_slop=0, max_top=0)
    vv.process(cat1, cat2)
    np.testing.assert_array_equal(vv.npairs, true_npairs)
    np.testing.assert_allclose(vv.weight, true_weight, rtol=1.e-6, atol=1.e-8)
    np.testing.assert_allclose(vv.xip, true_xip.real, rtol=1.e-6, atol=1.e-8)
    np.testing.assert_allclose(vv.xip_im, true_xip.imag, rtol=1.e-6, atol=1.e-8)
    np.testing.assert_allclose(vv.xim, true_xim.real, rtol=1.e-6, atol=1.e-8)
    np.testing.assert_allclose(vv.xim_im, true_xim.imag, rtol=1.e-6, atol=1.e-8)


@timer
def test_vv():
    # Similar to the math in test_gg(), but use a functional form that has a radial vector,
    # rather than radial shear pattern.
    # Also, the xi- integral here uses J2, not J4.

    # Use v_radial(r) = v0 r/r0 exp(-r^2/2r0^2)
    # i.e. v(r) = v0 r/r0 exp(-r^2/2r0^2) (x+iy)/r
    #
    # The Fourier transform is: v~(k) = 2 pi i v0 r0^3 k exp(-r0^2 k^2/2) / L^2
    # P(k) = (1/2pi) <|v~(k)|^2> = 2 pi v0^2 r0^6 k^2 / L^4 exp(-r0^2 k^2)
    # xi+(r) = (1/2pi) int( dk k P(k) J0(kr) )
    #        = pi/4 v0^2 (r0/L)^2 exp(-r^2/4r0^2) (4r0^2 - r^2)/r0^2
    # Note: The - sign in the next line is somewhat empirical.  I'm not quite sure where it
    #       comes from, and rederiving Peter's formula for spin-1 confuses me.  I see why J4
    #       changed to J2, but I'm not sure about the extra - sign.  My best guess is that it
    #       comes from v~(k) being imaginary, so v~(k)^2 = -P(k), not P(k).
    # xi-(r) = -(1/2pi) int( dk k P(k) J2(kr) )
    #        = -pi/4 v0^2 (r0/L)^2 exp(-r^2/4r0^2) r^2/r0^2

    v0 = 0.05
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
    v1 = v0 * np.exp(-r2/2.) * x/r0
    v2 = v0 * np.exp(-r2/2.) * y/r0

    cat = treecorr.Catalog(x=x, y=y, v1=v1, v2=v2, x_units='arcmin', y_units='arcmin')
    vv = treecorr.VVCorrelation(bin_size=0.1, min_sep=1., max_sep=100., sep_units='arcmin',
                                verbose=1)
    vv.process(cat)

    # log(<R>) != <logR>, but it should be close:
    print('meanlogr - log(meanr) = ',vv.meanlogr - np.log(vv.meanr))
    np.testing.assert_allclose(vv.meanlogr, np.log(vv.meanr), atol=1.e-3)

    r = vv.meanr
    temp = np.pi/4. * v0**2 * (r0/L)**2 * np.exp(-0.25*r**2/r0**2)
    true_xip = temp * (4*r0**2 - r**2)/r0**2
    true_xim = -temp * r**2/r0**2

    print('vv.xip = ',vv.xip)
    print('true_xip = ',true_xip)
    print('ratio = ',vv.xip / true_xip)
    print('diff = ',vv.xip - true_xip)
    print('max diff = ',max(abs(vv.xip - true_xip)))
    # It's within 10% everywhere except at the zero crossings.
    np.testing.assert_allclose(vv.xip, true_xip, rtol=0.1 * tol_factor, atol=1.e-7 * tol_factor)
    print('xip_im = ',vv.xip_im)
    np.testing.assert_allclose(vv.xip_im, 0, atol=2.e-7 * tol_factor)

    print('vv.xim = ',vv.xim)
    print('true_xim = ',true_xim)
    print('ratio = ',vv.xim / true_xim)
    print('diff = ',vv.xim - true_xim)
    print('max diff = ',max(abs(vv.xim - true_xim)))
    np.testing.assert_allclose(vv.xim, true_xim, rtol=0.1 * tol_factor, atol=2.e-7 * tol_factor)
    print('xim_im = ',vv.xim_im)
    np.testing.assert_allclose(vv.xim_im, 0, atol=1.e-7 * tol_factor)

    # Should also work as a cross-correlation with itself
    vv.process(cat,cat)
    np.testing.assert_allclose(vv.meanlogr, np.log(vv.meanr), atol=1.e-3)
    assert max(abs(vv.xip - true_xip)) < 3.e-7 * tol_factor
    assert max(abs(vv.xip_im)) < 2.e-7 * tol_factor
    assert max(abs(vv.xim - true_xim)) < 3.e-7 * tol_factor
    assert max(abs(vv.xim_im)) < 1.e-7 * tol_factor

    # Check that we get the same result using the corr2 function:
    cat.write(os.path.join('data','vv.dat'))
    config = treecorr.read_config('configs/vv.yaml')
    config['verbose'] = 0
    config['precision'] = 8
    treecorr.corr2(config)
    corr2_output = np.genfromtxt(os.path.join('output','vv.out'), names=True, skip_header=1)
    print('vv.xip = ',vv.xip)
    print('from corr2 output = ',corr2_output['xip'])
    print('ratio = ',corr2_output['xip']/vv.xip)
    print('diff = ',corr2_output['xip']-vv.xip)
    np.testing.assert_allclose(corr2_output['xip'], vv.xip, rtol=1.e-4)

    print('vv.xim = ',vv.xim)
    print('from corr2 output = ',corr2_output['xim'])
    print('ratio = ',corr2_output['xim']/vv.xim)
    print('diff = ',corr2_output['xim']-vv.xim)
    np.testing.assert_allclose(corr2_output['xim'], vv.xim, rtol=1.e-4)

    print('xip_im from corr2 output = ',corr2_output['xip_im'])
    print('max err = ',max(abs(corr2_output['xip_im'])))
    np.testing.assert_allclose(corr2_output['xip_im'], 0, atol=2.e-7 * tol_factor)
    print('xim_im from corr2 output = ',corr2_output['xim_im'])
    print('max err = ',max(abs(corr2_output['xim_im'])))
    np.testing.assert_allclose(corr2_output['xim_im'], 0, atol=2.e-7 * tol_factor)

    # Check the fits write option
    out_file_name = os.path.join('output','vv_out.dat')
    vv.write(out_file_name, precision=16)
    data = np.genfromtxt(out_file_name, names=True, skip_header=1)
    np.testing.assert_allclose(data['r_nom'], np.exp(vv.logr))
    np.testing.assert_allclose(data['meanr'], vv.meanr)
    np.testing.assert_allclose(data['meanlogr'], vv.meanlogr)
    np.testing.assert_allclose(data['xip'], vv.xip)
    np.testing.assert_allclose(data['xim'], vv.xim)
    np.testing.assert_allclose(data['xip_im'], vv.xip_im)
    np.testing.assert_allclose(data['xim_im'], vv.xim_im)
    np.testing.assert_allclose(data['sigma_xip'], np.sqrt(vv.varxip))
    np.testing.assert_allclose(data['sigma_xim'], np.sqrt(vv.varxim))
    np.testing.assert_allclose(data['weight'], vv.weight)
    np.testing.assert_allclose(data['npairs'], vv.npairs)

    # Check the read function
    vv2 = treecorr.VVCorrelation.from_file(out_file_name)
    np.testing.assert_allclose(vv2.logr, vv.logr)
    np.testing.assert_allclose(vv2.meanr, vv.meanr)
    np.testing.assert_allclose(vv2.meanlogr, vv.meanlogr)
    np.testing.assert_allclose(vv2.xip, vv.xip)
    np.testing.assert_allclose(vv2.xim, vv.xim)
    np.testing.assert_allclose(vv2.xip_im, vv.xip_im)
    np.testing.assert_allclose(vv2.xim_im, vv.xim_im)
    np.testing.assert_allclose(vv2.varxip, vv.varxip)
    np.testing.assert_allclose(vv2.varxim, vv.varxim)
    np.testing.assert_allclose(vv2.weight, vv.weight)
    np.testing.assert_allclose(vv2.npairs, vv.npairs)
    assert vv2.coords == vv.coords
    assert vv2.metric == vv.metric
    assert vv2.sep_units == vv.sep_units
    assert vv2.bin_type == vv.bin_type


@timer
def test_spherical():
    # This is the same field we used for test_vv, but put into spherical coords.
    # We do the spherical trig by hand using the obvious formulae, rather than the clever
    # optimizations that are used by the TreeCorr code, thus serving as a useful test of
    # the latter.

    v0 = 0.05
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
    v1 = v0 * np.exp(-r2/2./r0**2) * x/r0
    v2 = v0 * np.exp(-r2/2./r0**2) * y/r0
    r = np.sqrt(r2)
    theta = np.arctan2(y,x)

    vv = treecorr.VVCorrelation(bin_size=0.1, min_sep=1., max_sep=100., sep_units='arcmin',
                                verbose=1)
    r1 = np.exp(vv.logr) * (coord.arcmin / coord.radians)
    temp = np.pi/4. * v0**2 * (r0/L)**2 * np.exp(-0.25*r1**2/r0**2)
    true_xip = temp * (4*r0**2 - r1**2)/r0**2
    true_xim = -temp * r1**2/r0**2

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
        # v_sph = exp(i beta) * v
        # where beta = pi - (A+B) is the angle between north and "up" in the tangent plane.
        beta = np.pi - (A+B)
        beta[x>0] *= -1.
        cosbeta = np.cos(beta)
        sinbeta = np.sin(beta)
        v1_sph = v1 * cosbeta - v2 * sinbeta
        v2_sph = v2 * cosbeta + v1 * sinbeta

        cat = treecorr.Catalog(ra=ra, dec=dec, v1=v1_sph, v2=v2_sph, ra_units='rad',
                               dec_units='rad')
        vv = treecorr.VVCorrelation(bin_size=0.1, min_sep=1., max_sep=100., sep_units='arcmin',
                                    verbose=1)
        vv.process(cat)

        print('ra0, dec0 = ',ra0,dec0)
        print('vv.xip = ',vv.xip)
        print('true_xip = ',true_xip)
        print('ratio = ',vv.xip / true_xip)
        print('diff = ',vv.xip - true_xip)
        print('max diff = ',max(abs(vv.xip - true_xip)))
        assert max(abs(vv.xip - true_xip)) < 3.e-7 * tol_factor

        print('vv.xim = ',vv.xim)
        print('true_xim = ',true_xim)
        print('ratio = ',vv.xim / true_xim)
        print('diff = ',vv.xim - true_xim)
        print('max diff = ',max(abs(vv.xim - true_xim)))
        assert max(abs(vv.xim - true_xim)) < 2.e-7 * tol_factor

    # One more center that can be done very easily.  If the center is the north pole, then all
    # the radial vectors are pure negative v2.
    ra0 = 0
    dec0 = np.pi/2.
    ra = theta
    dec = np.pi/2. - 2.*np.arcsin(r/2.)
    vrad = v0 * np.exp(-r2/2./r0**2) * r/r0

    cat = treecorr.Catalog(ra=ra, dec=dec, v1=np.zeros_like(vrad), v2=-vrad, ra_units='rad',
                           dec_units='rad')
    vv.process(cat)

    print('vv.xip = ',vv.xip)
    print('vv.xip_im = ',vv.xip_im)
    print('true_xip = ',true_xip)
    print('ratio = ',vv.xip / true_xip)
    print('diff = ',vv.xip - true_xip)
    print('max diff = ',max(abs(vv.xip - true_xip)))
    assert max(abs(vv.xip - true_xip)) < 3.e-7 * tol_factor
    assert max(abs(vv.xip_im)) < 3.e-7 * tol_factor

    print('vv.xim = ',vv.xim)
    print('vv.xim_im = ',vv.xim_im)
    print('true_xim = ',true_xim)
    print('ratio = ',vv.xim / true_xim)
    print('diff = ',vv.xim - true_xim)
    print('max diff = ',max(abs(vv.xim - true_xim)))
    assert max(abs(vv.xim - true_xim)) < 2.e-7 * tol_factor
    assert max(abs(vv.xim_im)) < 2.e-7 * tol_factor

    # Check that we get the same result using the corr2 function
    cat.write(os.path.join('data','vv_spherical.dat'))
    config = treecorr.read_config('configs/vv_spherical.yaml')
    config['verbose'] = 0
    treecorr.corr2(config)
    corr2_output = np.genfromtxt(os.path.join('output','vv_spherical.out'), names=True,
                                 skip_header=1)
    print('vv.xip = ',vv.xip)
    print('from corr2 output = ',corr2_output['xip'])
    print('ratio = ',corr2_output['xip']/vv.xip)
    print('diff = ',corr2_output['xip']-vv.xip)
    np.testing.assert_allclose(corr2_output['xip'], vv.xip, rtol=1.e-3)

    print('vv.xim = ',vv.xim)
    print('from corr2 output = ',corr2_output['xim'])
    print('ratio = ',corr2_output['xim']/vv.xim)
    print('diff = ',corr2_output['xim']-vv.xim)
    np.testing.assert_allclose(corr2_output['xim'], vv.xim, rtol=1.e-3)

    print('xip_im from corr2 output = ',corr2_output['xip_im'])
    assert max(abs(corr2_output['xip_im'])) < 3.e-7 * tol_factor

    print('xim_im from corr2 output = ',corr2_output['xim_im'])
    assert max(abs(corr2_output['xim_im'])) < 2.e-7 * tol_factor


@timer
def test_varxi():
    # Test that varxip, varxim are correct (or close) based on actual variance of many runs.

    # Same v pattern as in test_vv().  Although the signal doesn't actually matter at all here.
    v0 = 0.05
    r0 = 10.
    L = 50.*r0
    rng = np.random.RandomState(8675309)

    # Note: to get a good estimate of var(xi), you need a lot of runs.  The number of
    # runs matters much more than the number of galaxies for getting this to pass.
    ngal = 1000
    nruns = 50000

    file_name = 'data/test_varxi_vv.npz'
    print(file_name)
    if not os.path.isfile(file_name):
        all_vvs = []

        for run in range(nruns):
            print(f'{run}/{nruns}')
            # In addition to the shape noise below, there is shot noise from the random x,y positions.
            x = (rng.random_sample(ngal)-0.5) * L
            y = (rng.random_sample(ngal)-0.5) * L
            # Varied weights are hard, but at least check that non-unit weights work correctly.
            w = np.ones_like(x) * 5
            r2 = (x**2 + y**2)/r0**2
            v1 = v0 * np.exp(-r2/2.) * x/r0
            v2 = v0 * np.exp(-r2/2.) * y/r0
            # This time, add some shape noise (different each run).
            v1 += rng.normal(0, 0.3, size=ngal)
            v2 += rng.normal(0, 0.3, size=ngal)

            cat = treecorr.Catalog(x=x, y=y, w=w, v1=v1, v2=v2, x_units='arcmin', y_units='arcmin')
            vv = treecorr.VVCorrelation(bin_size=0.1, min_sep=10., max_sep=100., sep_units='arcmin',
                                        verbose=1)
            vv.process(cat)
            all_vvs.append(vv)

        mean_xip = np.mean([vv.xip for vv in all_vvs], axis=0)
        var_xip = np.var([vv.xip for vv in all_vvs], axis=0)
        mean_xim = np.mean([vv.xim for vv in all_vvs], axis=0)
        var_xim = np.var([vv.xim for vv in all_vvs], axis=0)
        mean_varxip = np.mean([vv.varxip for vv in all_vvs], axis=0)
        mean_varxim = np.mean([vv.varxim for vv in all_vvs], axis=0)

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
    v1 = v0 * np.exp(-r2/2.) * x/r0
    v2 = v0 * np.exp(-r2/2.) * y/r0
    # This time, add some shape noise (different each run).
    v1 += rng.normal(0, 0.3, size=ngal)
    v2 += rng.normal(0, 0.3, size=ngal)

    cat = treecorr.Catalog(x=x, y=y, w=w, v1=v1, v2=v2, x_units='arcmin', y_units='arcmin')
    vv = treecorr.VVCorrelation(bin_size=0.1, min_sep=10., max_sep=100., sep_units='arcmin',
                                verbose=1)
    vv.process(cat)
    print('single run:')
    print('max relerr for xip = ',np.max(np.abs((vv.varxip - var_xip)/var_xip)))
    print('max relerr for xim = ',np.max(np.abs((vv.varxip - var_xim)/var_xim)))
    np.testing.assert_allclose(vv.varxip, var_xip, rtol=0.2)
    np.testing.assert_allclose(vv.varxim, var_xim, rtol=0.2)

@timer
def test_jk():

    # Same multi-lens field we used for NV patch test
    v0 = 0.05
    r0 = 30.
    L = 30 * r0
    rng = np.random.RandomState(8675309)

    nsource = 100000
    nlens = 300
    nruns = 1000
    npatch = 64

    corr_params = dict(bin_size=0.3, min_sep=10, max_sep=50, bin_slop=0.1)

    def make_velocity_field(rng):
        x1 = (rng.random(nlens)-0.5) * L
        y1 = (rng.random(nlens)-0.5) * L
        w = rng.random(nlens) + 10
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
            v1 += w[i] * v0 * np.exp(-r2/2.) * x2i/r0
            v2 += w[i] * v0 * np.exp(-r2/2.) * y2i/r0
        return x1, y1, w, x2, y2, v1, v2

    file_name = 'data/test_vv_jk_{}.npz'.format(nruns)
    print(file_name)
    if not os.path.isfile(file_name):
        all_vvs = []
        rng = np.random.default_rng()
        for run in range(nruns):
            x1, y1, w, x2, y2, v1, v2 = make_velocity_field(rng)
            print(run,': ',np.mean(v1),np.std(v1),np.min(v1),np.max(v1))
            cat = treecorr.Catalog(x=x2, y=y2, v1=v1, v2=v2)
            vv = treecorr.VVCorrelation(corr_params)
            vv.process(cat)
            all_vvs.append(vv)

        mean_xip = np.mean([vv.xip for vv in all_vvs], axis=0)
        mean_xim = np.mean([vv.xim for vv in all_vvs], axis=0)
        var_xip = np.var([vv.xip for vv in all_vvs], axis=0)
        var_xim = np.var([vv.xim for vv in all_vvs], axis=0)
        mean_varxip = np.mean([vv.varxip for vv in all_vvs], axis=0)
        mean_varxim = np.mean([vv.varxim for vv in all_vvs], axis=0)

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
    x1, y1, w, x2, y2, v1, v2 = make_velocity_field(rng)

    cat = treecorr.Catalog(x=x2, y=y2, v1=v1, v2=v2)
    vv1 = treecorr.VVCorrelation(corr_params)
    t0 = time.time()
    vv1.process(cat)
    t1 = time.time()
    print('Time for non-patch processing = ',t1-t0)

    print('weight = ',vv1.weight)
    print('xip = ',vv1.xip)
    print('varxip = ',vv1.varxip)
    print('pullsq for xip = ',(vv1.xip-mean_xip)**2/var_xip)
    print('max pull for xip = ',np.sqrt(np.max((vv1.xip-mean_xip)**2/var_xip)))
    print('max pull for xim = ',np.sqrt(np.max((vv1.xim-mean_xim)**2/var_xim)))
    np.testing.assert_array_less((vv1.xip-mean_xip)**2, 9*var_xip)  # < 3 sigma pull
    np.testing.assert_array_less((vv1.xim-mean_xim)**2, 9*var_xim)  # < 3 sigma pull
    np.testing.assert_allclose(vv1.varxip, mean_varxip, rtol=0.1)
    np.testing.assert_allclose(vv1.varxim, mean_varxim, rtol=0.1)

    # Now run with patches, but still with shot variance.  Should be basically the same answer.
    catp = treecorr.Catalog(x=x2, y=y2, v1=v1, v2=v2, npatch=npatch)
    print('tot w = ',np.sum(w))
    print('Patch\tNsource')
    for i in range(npatch):
        print('%d\t%d'%(i,np.sum(catp.w[catp.patch==i])))
    vv2 = treecorr.VVCorrelation(corr_params)
    t0 = time.time()
    vv2.process(catp)
    t1 = time.time()
    print('Time for patch processing = ',t1-t0)
    print('weight = ',vv2.weight)
    print('xip = ',vv2.xip)
    print('xip1 = ',vv1.xip)
    print('varxip = ',vv2.varxip)
    print('xim = ',vv2.xim)
    print('xim1 = ',vv1.xim)
    print('varxim = ',vv2.varxim)
    np.testing.assert_allclose(vv2.weight, vv1.weight, rtol=1.e-2)
    np.testing.assert_allclose(vv2.xip, vv1.xip, rtol=1.e-2)
    np.testing.assert_allclose(vv2.xim, vv1.xim, rtol=1.e-2)
    np.testing.assert_allclose(vv2.varxip, vv1.varxip, rtol=1.e-2)
    np.testing.assert_allclose(vv2.varxim, vv1.varxim, rtol=1.e-2)

    # Now try jackknife variance estimate.
    t0 = time.time()
    cov2 = vv2.estimate_cov('jackknife')
    t1 = time.time()
    print('Time to calculate jackknife covariance = ',t1-t0)
    print('cov.diag = ',np.diagonal(cov2))
    print('cf var_xip = ',var_xip)
    print('cf var_xim = ',var_xim)
    np.testing.assert_allclose(np.diagonal(cov2)[:6], var_xip, rtol=0.4)
    np.testing.assert_allclose(np.diagonal(cov2)[6:], var_xim, rtol=0.5)

    # Use initialize/finalize
    vv3 = treecorr.VVCorrelation(corr_params)
    for k1, p1 in enumerate(catp.get_patches()):
        vv3.process(p1, initialize=(k1==0), finalize=(k1==npatch-1))
        for k2, p2 in enumerate(catp.get_patches()):
            if k2 <= k1: continue
            vv3.process(p1, p2, initialize=False, finalize=False)
    np.testing.assert_allclose(vv3.xip, vv2.xip)
    np.testing.assert_allclose(vv3.xim, vv2.xim)
    np.testing.assert_allclose(vv3.weight, vv2.weight)

    # Check that these still work after roundtripping through a file.
    try:
        import fitsio
    except ImportError:
        pass
    else:
        file_name = os.path.join('output','test_write_results_vv.fits')
        vv2.write(file_name, write_patch_results=True)
        vv3 = treecorr.VVCorrelation.from_file(file_name)
        cov3 = vv3.estimate_cov('jackknife')
        np.testing.assert_allclose(cov3, cov2)

    # Check some invalid actions
    # Bad var_method
    with assert_raises(ValueError):
        vv2.estimate_cov('invalid')
    # Not run on patches, but need patches
    with assert_raises(ValueError):
        vv1.estimate_cov('jackknife')
    with assert_raises(ValueError):
        vv1.estimate_cov('sample')
    with assert_raises(ValueError):
        vv1.estimate_cov('marked_bootstrap')
    with assert_raises(ValueError):
        vv1.estimate_cov('bootstrap')

    cata = treecorr.Catalog(x=x2[:100], y=y2[:100], v1=v1[:100], v2=v2[:100], npatch=10)
    catb = treecorr.Catalog(x=x2[:100], y=y2[:100], v1=v1[:100], v2=v2[:100], npatch=2)
    vv4 = treecorr.VVCorrelation(corr_params)
    vv5 = treecorr.VVCorrelation(corr_params)
    # All catalogs need to have the same number of patches
    with assert_raises(RuntimeError):
        vv4.process(cata,catb)
    with assert_raises(RuntimeError):
        vv5.process(catb,cata)

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
    v1 = rng.multivariate_normal(np.zeros(N), K*(A**2))
    v1 += rng.normal(scale=sigma, size=N)
    v2 = rng.multivariate_normal(np.zeros(N), K*(A**2))
    v2 += rng.normal(scale=sigma, size=N)
    v = v1 + 1j * v2

    # Calculate the 2D correlation using brute force
    max_sep = 21.
    nbins = 21
    xi_brut = corr2d(x, y, v, np.conj(v), rmax=max_sep, bins=nbins)

    # And using TreeCorr
    cat = treecorr.Catalog(x=x, y=y, v1=v1, v2=v2)
    vv = treecorr.VVCorrelation(max_sep=max_sep, bin_size=2., bin_type='TwoD', brute=True)
    vv.process(cat)
    print('max abs diff = ',np.max(np.abs(vv.xip - xi_brut)))
    print('max rel diff = ',np.max(np.abs(vv.xip - xi_brut)/np.abs(vv.xip)))
    np.testing.assert_allclose(vv.xip, xi_brut, atol=2.e-7)

    vv = treecorr.VVCorrelation(max_sep=max_sep, bin_size=2., bin_type='TwoD', bin_slop=0.05)
    vv.process(cat)
    print('max abs diff = ',np.max(np.abs(vv.xip - xi_brut)))
    print('max rel diff = ',np.max(np.abs(vv.xip - xi_brut)/np.abs(vv.xip)))
    np.testing.assert_allclose(vv.xip, xi_brut, atol=2.e-7)

    # Check I/O
    try:
        import fitsio
    except ImportError:
        pass
    else:
        fits_name = 'output/vv_twod.fits'
        vv.write(fits_name)
        vv2 = treecorr.VVCorrelation.from_file(fits_name)
        np.testing.assert_allclose(vv2.npairs, vv.npairs)
        np.testing.assert_allclose(vv2.weight, vv.weight)
        np.testing.assert_allclose(vv2.meanr, vv.meanr)
        np.testing.assert_allclose(vv2.meanlogr, vv.meanlogr)
        np.testing.assert_allclose(vv2.xip, vv.xip)
        np.testing.assert_allclose(vv2.xip_im, vv.xip_im)
        np.testing.assert_allclose(vv2.xim, vv.xim)
        np.testing.assert_allclose(vv2.xim_im, vv.xim_im)

    ascii_name = 'output/vv_twod.txt'
    vv.write(ascii_name, precision=16)
    vv3 = treecorr.VVCorrelation.from_file(ascii_name)
    np.testing.assert_allclose(vv3.npairs, vv.npairs)
    np.testing.assert_allclose(vv3.weight, vv.weight)
    np.testing.assert_allclose(vv3.meanr, vv.meanr)
    np.testing.assert_allclose(vv3.meanlogr, vv.meanlogr)
    np.testing.assert_allclose(vv3.xip, vv.xip)
    np.testing.assert_allclose(vv3.xip_im, vv.xip_im)
    np.testing.assert_allclose(vv3.xim, vv.xim)
    np.testing.assert_allclose(vv3.xim_im, vv.xim_im)


if __name__ == '__main__':
    test_direct()
    test_direct_spherical()
    test_vv()
    test_spherical()
    test_varxi()
    test_jk()
    test_twod()
