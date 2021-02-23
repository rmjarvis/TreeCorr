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
import os
import coord
import fitsio
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
    g11 = rng.normal(0,0.2, (ngal,) )
    g21 = rng.normal(0,0.2, (ngal,) )

    x2 = rng.normal(0,s, (ngal,) )
    y2 = rng.normal(0,s, (ngal,) )
    w2 = rng.random_sample(ngal)
    g12 = rng.normal(0,0.2, (ngal,) )
    g22 = rng.normal(0,0.2, (ngal,) )

    cat1 = treecorr.Catalog(x=x1, y=y1, w=w1, g1=g11, g2=g21)
    cat2 = treecorr.Catalog(x=x2, y=y2, w=w2, g1=g12, g2=g22)

    min_sep = 1.
    max_sep = 50.
    nbins = 50
    bin_size = np.log(max_sep/min_sep) / nbins
    gg = treecorr.GGCorrelation(min_sep=min_sep, max_sep=max_sep, nbins=nbins, brute=True)
    gg.process(cat1, cat2)

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
        xip = ww * (g11[i] + 1j*g21[i]) * (g12 - 1j*g22)
        xim = ww * (g11[i] + 1j*g21[i]) * (g12 + 1j*g22) * expmialpha**4

        index = np.floor(np.log(r/min_sep) / bin_size).astype(int)
        mask = (index >= 0) & (index < nbins)
        np.add.at(true_npairs, index[mask], 1)
        np.add.at(true_weight, index[mask], ww[mask])
        np.add.at(true_xip, index[mask], xip[mask])
        np.add.at(true_xim, index[mask], xim[mask])

    true_xip /= true_weight
    true_xim /= true_weight

    print('true_npairs = ',true_npairs)
    print('diff = ',gg.npairs - true_npairs)
    np.testing.assert_array_equal(gg.npairs, true_npairs)

    print('true_weight = ',true_weight)
    print('diff = ',gg.weight - true_weight)
    np.testing.assert_allclose(gg.weight, true_weight, rtol=1.e-5, atol=1.e-8)

    print('true_xip = ',true_xip)
    print('gg.xip = ',gg.xip)
    print('gg.xip_im = ',gg.xip_im)
    np.testing.assert_allclose(gg.xip, true_xip.real, rtol=1.e-4, atol=1.e-8)
    np.testing.assert_allclose(gg.xip_im, true_xip.imag, rtol=1.e-4, atol=1.e-8)
    print('true_xim = ',true_xim)
    print('gg.xim = ',gg.xim)
    print('gg.xim_im = ',gg.xim_im)
    np.testing.assert_allclose(gg.xim, true_xim.real, rtol=1.e-4, atol=1.e-8)
    np.testing.assert_allclose(gg.xim_im, true_xim.imag, rtol=1.e-4, atol=1.e-8)

    # Check that running via the corr2 script works correctly.
    config = treecorr.config.read_config('configs/gg_direct.yaml')
    cat1.write(config['file_name'])
    cat2.write(config['file_name2'])
    treecorr.corr2(config)
    data = fitsio.read(config['gg_file_name'])
    np.testing.assert_allclose(data['r_nom'], gg.rnom)
    np.testing.assert_allclose(data['npairs'], gg.npairs)
    np.testing.assert_allclose(data['weight'], gg.weight)
    np.testing.assert_allclose(data['xip'], gg.xip, rtol=1.e-3)
    np.testing.assert_allclose(data['xip_im'], gg.xip_im, rtol=1.e-3)
    np.testing.assert_allclose(data['xim'], gg.xim, rtol=1.e-3)
    np.testing.assert_allclose(data['xim_im'], gg.xim_im, rtol=1.e-3)

    # Repeat with binslop = 0.
    # And don't do any top-level recursion so we actually test not going to the leaves.
    gg = treecorr.GGCorrelation(min_sep=min_sep, max_sep=max_sep, nbins=nbins, bin_slop=0,
                                max_top=0)
    gg.process(cat1, cat2)
    np.testing.assert_array_equal(gg.npairs, true_npairs)
    np.testing.assert_allclose(gg.weight, true_weight, rtol=1.e-5, atol=1.e-8)
    np.testing.assert_allclose(gg.xip, true_xip.real, rtol=1.e-4, atol=1.e-8)
    np.testing.assert_allclose(gg.xip_im, true_xip.imag, rtol=1.e-4, atol=1.e-8)
    print('true_xim = ',true_xim)
    print('gg.xim = ',gg.xim)
    print('gg.xim_im = ',gg.xim_im)
    print('diff = ',gg.xim - true_xim.real)
    print('max diff = ',np.max(np.abs(gg.xim - true_xim.real)))
    print('rel diff = ',(gg.xim - true_xim.real)/true_xim.real)
    # This is the one that is highly affected by the approximation from averaging the shears
    # before projecting, rather than averaging each shear projected to its own connecting line.
    np.testing.assert_allclose(gg.xim, true_xim.real, rtol=1.e-3, atol=3.e-4)
    np.testing.assert_allclose(gg.xim_im, true_xim.imag, atol=1.e-3)

    # Check a few basic operations with a GGCorrelation object.
    do_pickle(gg)

    gg2 = gg.copy()
    gg2 += gg
    np.testing.assert_allclose(gg2.npairs, 2*gg.npairs)
    np.testing.assert_allclose(gg2.weight, 2*gg.weight)
    np.testing.assert_allclose(gg2.meanr, 2*gg.meanr)
    np.testing.assert_allclose(gg2.meanlogr, 2*gg.meanlogr)
    np.testing.assert_allclose(gg2.xip, 2*gg.xip)
    np.testing.assert_allclose(gg2.xip_im, 2*gg.xip_im)
    np.testing.assert_allclose(gg2.xim, 2*gg.xim)
    np.testing.assert_allclose(gg2.xim_im, 2*gg.xim_im)

    gg2.clear()
    gg2 += gg
    np.testing.assert_allclose(gg2.npairs, gg.npairs)
    np.testing.assert_allclose(gg2.weight, gg.weight)
    np.testing.assert_allclose(gg2.meanr, gg.meanr)
    np.testing.assert_allclose(gg2.meanlogr, gg.meanlogr)
    np.testing.assert_allclose(gg2.xip, gg.xip)
    np.testing.assert_allclose(gg2.xip_im, gg.xip_im)
    np.testing.assert_allclose(gg2.xim, gg.xim)
    np.testing.assert_allclose(gg2.xim_im, gg.xim_im)

    ascii_name = 'output/gg_ascii.txt'
    gg.write(ascii_name, precision=16)
    gg3 = treecorr.GGCorrelation(min_sep=min_sep, max_sep=max_sep, nbins=nbins)
    gg3.read(ascii_name)
    np.testing.assert_allclose(gg3.npairs, gg.npairs)
    np.testing.assert_allclose(gg3.weight, gg.weight)
    np.testing.assert_allclose(gg3.meanr, gg.meanr)
    np.testing.assert_allclose(gg3.meanlogr, gg.meanlogr)
    np.testing.assert_allclose(gg3.xip, gg.xip)
    np.testing.assert_allclose(gg3.xip_im, gg.xip_im)
    np.testing.assert_allclose(gg3.xim, gg.xim)
    np.testing.assert_allclose(gg3.xim_im, gg.xim_im)

    fits_name = 'output/gg_fits.fits'
    gg.write(fits_name)
    gg4 = treecorr.GGCorrelation(min_sep=min_sep, max_sep=max_sep, nbins=nbins)
    gg4.read(fits_name)
    np.testing.assert_allclose(gg4.npairs, gg.npairs)
    np.testing.assert_allclose(gg4.weight, gg.weight)
    np.testing.assert_allclose(gg4.meanr, gg.meanr)
    np.testing.assert_allclose(gg4.meanlogr, gg.meanlogr)
    np.testing.assert_allclose(gg4.xip, gg.xip)
    np.testing.assert_allclose(gg4.xip_im, gg.xip_im)
    np.testing.assert_allclose(gg4.xim, gg.xim)
    np.testing.assert_allclose(gg4.xim_im, gg.xim_im)

    with assert_raises(TypeError):
        gg2 += config
    gg4 = treecorr.GGCorrelation(min_sep=min_sep/2, max_sep=max_sep, nbins=nbins)
    with assert_raises(ValueError):
        gg2 += gg4
    gg5 = treecorr.GGCorrelation(min_sep=min_sep, max_sep=max_sep*2, nbins=nbins)
    with assert_raises(ValueError):
        gg2 += gg5
    gg6 = treecorr.GGCorrelation(min_sep=min_sep, max_sep=max_sep, nbins=nbins*2)
    with assert_raises(ValueError):
        gg2 += gg6

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
    g11 = rng.normal(0,0.2, (ngal,) )
    g21 = rng.normal(0,0.2, (ngal,) )

    x2 = rng.normal(0,s, (ngal,) )
    y2 = rng.normal(0,s, (ngal,) ) + 200
    z2 = rng.normal(0,s, (ngal,) )
    w2 = rng.random_sample(ngal)
    g12 = rng.normal(0,0.2, (ngal,) )
    g22 = rng.normal(0,0.2, (ngal,) )

    ra1, dec1 = coord.CelestialCoord.xyz_to_radec(x1,y1,z1)
    ra2, dec2 = coord.CelestialCoord.xyz_to_radec(x2,y2,z2)

    cat1 = treecorr.Catalog(ra=ra1, dec=dec1, ra_units='rad', dec_units='rad', w=w1, g1=g11, g2=g21)
    cat2 = treecorr.Catalog(ra=ra2, dec=dec2, ra_units='rad', dec_units='rad', w=w2, g1=g12, g2=g22)

    min_sep = 1.
    max_sep = 10.
    nbins = 50
    bin_size = np.log(max_sep/min_sep) / nbins
    gg = treecorr.GGCorrelation(min_sep=min_sep, max_sep=max_sep, nbins=nbins,
                                sep_units='deg', brute=True)
    gg.process(cat1, cat2)

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

            # Rotate shears to coordinates where line connecting is horizontal.
            # Original orientation is where north is up.
            theta1 = 90*coord.degrees - c1[i].angleBetween(north_pole, c2[j])
            theta2 = 90*coord.degrees - c2[j].angleBetween(north_pole, c1[i])
            exp2theta1 = np.cos(2*theta1) + 1j * np.sin(2*theta1)
            exp2theta2 = np.cos(2*theta2) + 1j * np.sin(2*theta2)

            g1 = g11[i] + 1j * g21[i]
            g2 = g12[j] + 1j * g22[j]
            g1 *= exp2theta1
            g2 *= exp2theta2

            ww = w1[i] * w2[j]
            xip = ww * g1 * np.conjugate(g2)
            xim = ww * g1 * g2

            true_npairs[index] += 1
            true_weight[index] += ww
            true_xip[index] += xip
            true_xim[index] += xim

    true_xip /= true_weight
    true_xim /= true_weight

    print('true_npairs = ',true_npairs)
    print('diff = ',gg.npairs - true_npairs)
    np.testing.assert_array_equal(gg.npairs, true_npairs)

    print('true_weight = ',true_weight)
    print('diff = ',gg.weight - true_weight)
    np.testing.assert_allclose(gg.weight, true_weight, rtol=1.e-5, atol=1.e-8)

    print('true_xip = ',true_xip)
    print('gg.xip = ',gg.xip)
    print('gg.xip_im = ',gg.xip_im)
    np.testing.assert_allclose(gg.xip, true_xip.real, rtol=1.e-4, atol=1.e-8)
    np.testing.assert_allclose(gg.xip_im, true_xip.imag, rtol=1.e-4, atol=1.e-8)
    print('true_xim = ',true_xim)
    print('gg.xim = ',gg.xim)
    print('gg.xim_im = ',gg.xim_im)
    np.testing.assert_allclose(gg.xim, true_xim.real, rtol=1.e-4, atol=1.e-8)
    np.testing.assert_allclose(gg.xim_im, true_xim.imag, rtol=1.e-4, atol=1.e-8)

    # Check that running via the corr2 script works correctly.
    config = treecorr.config.read_config('configs/gg_direct_spherical.yaml')
    cat1.write(config['file_name'])
    cat2.write(config['file_name2'])
    treecorr.corr2(config)
    data = fitsio.read(config['gg_file_name'])
    np.testing.assert_allclose(data['r_nom'], gg.rnom)
    np.testing.assert_allclose(data['npairs'], gg.npairs)
    np.testing.assert_allclose(data['weight'], gg.weight)
    np.testing.assert_allclose(data['xip'], gg.xip, rtol=1.e-3)
    np.testing.assert_allclose(data['xip_im'], gg.xip_im, rtol=1.e-3)
    np.testing.assert_allclose(data['xim'], gg.xim, rtol=1.e-3)
    np.testing.assert_allclose(data['xim_im'], gg.xim_im, rtol=1.e-3)

    # Repeat with binslop = 0
    # And don't do any top-level recursion so we actually test not going to the leaves.
    gg = treecorr.GGCorrelation(min_sep=min_sep, max_sep=max_sep, nbins=nbins,
                                sep_units='deg', bin_slop=0, max_top=0)
    gg.process(cat1, cat2)
    np.testing.assert_array_equal(gg.npairs, true_npairs)
    np.testing.assert_allclose(gg.weight, true_weight, rtol=1.e-5, atol=1.e-8)
    np.testing.assert_allclose(gg.xip, true_xip.real, rtol=1.e-3, atol=1.e-6)
    np.testing.assert_allclose(gg.xip_im, true_xip.imag, rtol=1.e-3, atol=1.e-6)
    np.testing.assert_allclose(gg.xim, true_xim.real, rtol=1.e-3, atol=2.e-4)
    np.testing.assert_allclose(gg.xim_im, true_xim.imag, rtol=1.e-3, atol=2.e-4)


@timer
def test_pairwise():
    # Test the pairwise option.

    ngal = 1000
    s = 10.
    rng = np.random.RandomState(8675309)
    x1 = rng.normal(0,s, (ngal,) )
    y1 = rng.normal(0,s, (ngal,) )
    w1 = rng.random_sample(ngal)
    g11 = rng.normal(0,0.2, (ngal,) )
    g21 = rng.normal(0,0.2, (ngal,) )

    x2 = rng.normal(0,s, (ngal,) )
    y2 = rng.normal(0,s, (ngal,) )
    w2 = rng.random_sample(ngal)
    g12 = rng.normal(0,0.2, (ngal,) )
    g22 = rng.normal(0,0.2, (ngal,) )

    w1 = np.ones_like(w1)
    w2 = np.ones_like(w2)

    cat1 = treecorr.Catalog(x=x1, y=y1, w=w1, g1=g11, g2=g21)
    cat2 = treecorr.Catalog(x=x2, y=y2, w=w2, g1=g12, g2=g22)

    min_sep = 5.
    max_sep = 50.
    nbins = 10
    bin_size = np.log(max_sep/min_sep) / nbins
    gg = treecorr.GGCorrelation(min_sep=min_sep, max_sep=max_sep, nbins=nbins)
    with assert_warns(FutureWarning):
        gg.process_pairwise(cat1, cat2)
    gg.finalize(cat1.varg, cat2.varg)

    true_npairs = np.zeros(nbins, dtype=int)
    true_weight = np.zeros(nbins, dtype=float)
    true_xip = np.zeros(nbins, dtype=complex)
    true_xim = np.zeros(nbins, dtype=complex)

    rsq = (x1-x2)**2 + (y1-y2)**2
    r = np.sqrt(rsq)
    expmialpha = ((x1-x2) - 1j*(y1-y2)) / r

    ww = w1 * w2
    xip = ww * (g11 + 1j*g21) * (g12 - 1j*g22)
    xim = ww * (g11 + 1j*g21) * (g12 + 1j*g22) * expmialpha**4

    index = np.floor(np.log(r/min_sep) / bin_size).astype(int)
    mask = (index >= 0) & (index < nbins)
    np.add.at(true_npairs, index[mask], 1)
    np.add.at(true_weight, index[mask], ww[mask])
    np.add.at(true_xip, index[mask], xip[mask])
    np.add.at(true_xim, index[mask], xim[mask])

    true_xip /= true_weight
    true_xim /= true_weight

    np.testing.assert_array_equal(gg.npairs, true_npairs)
    np.testing.assert_allclose(gg.weight, true_weight, rtol=1.e-5, atol=1.e-8)
    np.testing.assert_allclose(gg.xip, true_xip.real, rtol=1.e-4, atol=1.e-8)
    np.testing.assert_allclose(gg.xip_im, true_xip.imag, rtol=1.e-4, atol=1.e-8)
    np.testing.assert_allclose(gg.xim, true_xim.real, rtol=1.e-4, atol=1.e-8)
    np.testing.assert_allclose(gg.xim_im, true_xim.imag, rtol=1.e-4, atol=1.e-8)

    # If cats have names, then the logger will mention them.
    # Also, test running with optional args.
    cat1.name = "first"
    cat2.name = "second"
    with CaptureLog() as cl:
        gg.logger = cl.logger
        with assert_warns(FutureWarning):
            gg.process_pairwise(cat1, cat2, metric='Euclidean', num_threads=2)
    assert "for cats first, second" in cl.output


@timer
def test_gg():
    # cf. http://adsabs.harvard.edu/abs/2002A%26A...389..729S for the basic formulae I use here.
    #
    # Use gamma_t(r) = gamma0 r^2/r0^2 exp(-r^2/2r0^2)
    # i.e. gamma(r) = -gamma0 exp(-r^2/2r0^2) (x+iy)^2 / r0^2
    #
    # The Fourier transform is: gamma~(k) = -2 pi gamma0 r0^4 k^2 exp(-r0^2 k^2/2) / L^2
    # P(k) = (1/2pi) <|gamma~(k)|^2> = 2 pi gamma0^2 r0^8 k^4 / L^4 exp(-r0^2 k^2)
    # xi+(r) = (1/2pi) int( dk k P(k) J0(kr) )
    #        = pi/16 gamma0^2 (r0/L)^2 exp(-r^2/4r0^2) (r^4 - 16r^2r0^2 + 32r0^4)/r0^4
    # xi-(r) = (1/2pi) int( dk k P(k) J4(kr) )
    #        = pi/16 gamma0^2 (r0/L)^2 exp(-r^2/4r0^2) r^4/r0^4
    # Note: I'm not sure I handled the L factors correctly, but the units at the end need
    # to be gamma^2, so it needs to be (r0/L)^2.

    gamma0 = 0.05
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
    g1 = -gamma0 * np.exp(-r2/2.) * (x**2-y**2)/r0**2
    g2 = -gamma0 * np.exp(-r2/2.) * (2.*x*y)/r0**2

    cat = treecorr.Catalog(x=x, y=y, g1=g1, g2=g2, x_units='arcmin', y_units='arcmin')
    gg = treecorr.GGCorrelation(bin_size=0.1, min_sep=1., max_sep=100., sep_units='arcmin',
                                verbose=1)
    gg.process(cat)

    # log(<R>) != <logR>, but it should be close:
    print('meanlogr - log(meanr) = ',gg.meanlogr - np.log(gg.meanr))
    np.testing.assert_allclose(gg.meanlogr, np.log(gg.meanr), atol=1.e-3)

    r = gg.meanr
    temp = np.pi/16. * gamma0**2 * (r0/L)**2 * np.exp(-0.25*r**2/r0**2)
    true_xip = temp * (r**4 - 16.*r**2*r0**2 + 32.*r0**4)/r0**4
    true_xim = temp * r**4/r0**4

    print('gg.xip = ',gg.xip)
    print('true_xip = ',true_xip)
    print('ratio = ',gg.xip / true_xip)
    print('diff = ',gg.xip - true_xip)
    print('max diff = ',max(abs(gg.xip - true_xip)))
    # It's within 10% everywhere except at the zero crossings.
    np.testing.assert_allclose(gg.xip, true_xip, rtol=0.1 * tol_factor, atol=1.e-7 * tol_factor)
    print('xip_im = ',gg.xip_im)
    np.testing.assert_allclose(gg.xip_im, 0, atol=2.e-7 * tol_factor)

    print('gg.xim = ',gg.xim)
    print('true_xim = ',true_xim)
    print('ratio = ',gg.xim / true_xim)
    print('diff = ',gg.xim - true_xim)
    print('max diff = ',max(abs(gg.xim - true_xim)))
    np.testing.assert_allclose(gg.xim, true_xim, rtol=0.1 * tol_factor, atol=2.e-7 * tol_factor)
    print('xim_im = ',gg.xim_im)
    np.testing.assert_allclose(gg.xim_im, 0, atol=1.e-7 * tol_factor)

    # Should also work as a cross-correlation with itself
    gg.process(cat,cat)
    np.testing.assert_allclose(gg.meanlogr, np.log(gg.meanr), atol=1.e-3)
    assert max(abs(gg.xip - true_xip)) < 3.e-7 * tol_factor
    assert max(abs(gg.xip_im)) < 2.e-7 * tol_factor
    assert max(abs(gg.xim - true_xim)) < 3.e-7 * tol_factor
    assert max(abs(gg.xim_im)) < 1.e-7 * tol_factor

    # We check the accuracy of the MapSq calculation below in test_mapsq.
    # Here we just check that it runs, round trips correctly through an output file,
    # and gives the same answer when run through corr2.

    mapsq, mapsq_im, mxsq, mxsq_im, varmapsq = gg.calculateMapSq()
    print('mapsq = ',mapsq)
    print('mxsq = ',mxsq)

    mapsq_file = 'output/gg_m2.txt'
    gg.writeMapSq(mapsq_file, precision=16)
    data = np.genfromtxt(os.path.join('output','gg_m2.txt'), names=True)
    np.testing.assert_allclose(data['Mapsq'], mapsq)
    np.testing.assert_allclose(data['Mxsq'], mxsq)

    # Check that we get the same result using the corr2 function:
    cat.write(os.path.join('data','gg.dat'))
    config = treecorr.read_config('configs/gg.yaml')
    config['verbose'] = 0
    config['precision'] = 8
    treecorr.corr2(config)
    corr2_output = np.genfromtxt(os.path.join('output','gg.out'), names=True, skip_header=1)
    print('gg.xip = ',gg.xip)
    print('from corr2 output = ',corr2_output['xip'])
    print('ratio = ',corr2_output['xip']/gg.xip)
    print('diff = ',corr2_output['xip']-gg.xip)
    np.testing.assert_allclose(corr2_output['xip'], gg.xip, rtol=1.e-4)

    print('gg.xim = ',gg.xim)
    print('from corr2 output = ',corr2_output['xim'])
    print('ratio = ',corr2_output['xim']/gg.xim)
    print('diff = ',corr2_output['xim']-gg.xim)
    np.testing.assert_allclose(corr2_output['xim'], gg.xim, rtol=1.e-4)

    print('xip_im from corr2 output = ',corr2_output['xip_im'])
    print('max err = ',max(abs(corr2_output['xip_im'])))
    np.testing.assert_allclose(corr2_output['xip_im'], 0, atol=2.e-7 * tol_factor)
    print('xim_im from corr2 output = ',corr2_output['xim_im'])
    print('max err = ',max(abs(corr2_output['xim_im'])))
    np.testing.assert_allclose(corr2_output['xim_im'], 0, atol=2.e-7 * tol_factor)

    # Check m2 output
    corr2_output2 = np.genfromtxt(os.path.join('output','gg_m2.out'), names=True)
    print('mapsq = ',mapsq)
    print('from corr2 output = ',corr2_output2['Mapsq'])
    print('ratio = ',corr2_output2['Mapsq']/mapsq)
    print('diff = ',corr2_output2['Mapsq']-mapsq)
    np.testing.assert_allclose(corr2_output2['Mapsq'], mapsq, rtol=1.e-4)

    print('mxsq = ',mxsq)
    print('from corr2 output = ',corr2_output2['Mxsq'])
    print('ratio = ',corr2_output2['Mxsq']/mxsq)
    print('diff = ',corr2_output2['Mxsq']-mxsq)
    np.testing.assert_allclose(corr2_output2['Mxsq'], mxsq, rtol=1.e-4)

    # OK to have m2 output, but not gg
    del config['gg_file_name']
    treecorr.corr2(config)
    corr2_output2 = np.genfromtxt(os.path.join('output','gg_m2.out'), names=True)
    np.testing.assert_allclose(corr2_output2['Mapsq'], mapsq, rtol=1.e-4)
    np.testing.assert_allclose(corr2_output2['Mxsq'], mxsq, rtol=1.e-4)

    # Check the fits write option
    out_file_name = os.path.join('output','gg_out.fits')
    gg.write(out_file_name)
    data = fitsio.read(out_file_name)
    np.testing.assert_allclose(data['r_nom'], np.exp(gg.logr))
    np.testing.assert_allclose(data['meanr'], gg.meanr)
    np.testing.assert_allclose(data['meanlogr'], gg.meanlogr)
    np.testing.assert_allclose(data['xip'], gg.xip)
    np.testing.assert_allclose(data['xim'], gg.xim)
    np.testing.assert_allclose(data['xip_im'], gg.xip_im)
    np.testing.assert_allclose(data['xim_im'], gg.xim_im)
    np.testing.assert_allclose(data['sigma_xip'], np.sqrt(gg.varxip))
    np.testing.assert_allclose(data['sigma_xim'], np.sqrt(gg.varxim))
    np.testing.assert_allclose(data['weight'], gg.weight)
    np.testing.assert_allclose(data['npairs'], gg.npairs)

    # Check the read function
    gg2 = treecorr.GGCorrelation(bin_size=0.1, min_sep=1., max_sep=100., sep_units='arcmin')
    gg2.read(out_file_name)
    np.testing.assert_allclose(gg2.logr, gg.logr)
    np.testing.assert_allclose(gg2.meanr, gg.meanr)
    np.testing.assert_allclose(gg2.meanlogr, gg.meanlogr)
    np.testing.assert_allclose(gg2.xip, gg.xip)
    np.testing.assert_allclose(gg2.xim, gg.xim)
    np.testing.assert_allclose(gg2.xip_im, gg.xip_im)
    np.testing.assert_allclose(gg2.xim_im, gg.xim_im)
    np.testing.assert_allclose(gg2.varxip, gg.varxip)
    np.testing.assert_allclose(gg2.varxim, gg.varxim)
    np.testing.assert_allclose(gg2.weight, gg.weight)
    np.testing.assert_allclose(gg2.npairs, gg.npairs)
    assert gg2.coords == gg.coords
    assert gg2.metric == gg.metric
    assert gg2.sep_units == gg.sep_units
    assert gg2.bin_type == gg.bin_type

    # Also check the Schneider version.
    mapsq, mapsq_im, mxsq, mxsq_im, varmapsq = gg.calculateMapSq(m2_uform='Schneider')
    print('Schneider mapsq = ',mapsq)
    print('mxsq = ',mxsq)
    print('max = ',max(abs(mxsq)))

    # And GamSq.
    gamsq, vargamsq = gg.calculateGamSq()
    print('gamsq = ',gamsq)

    gamsq, vargamsq, gamsq_e, gamsq_b, vargamsq_eb = gg.calculateGamSq(eb=True)
    print('gamsq_e = ',gamsq_e)
    print('gamsq_b = ',gamsq_b)

    # The Gamsq columns were already output in the above m2_output run of corr2.
    np.testing.assert_allclose(corr2_output2['Gamsq'], gamsq, rtol=1.e-4)


@timer
def test_mapsq():
    # Use the same gamma(r) as in test_gg.
    # This time, rather than use a smaller catalog in the nosetests run, we skip the run
    # in that case and just read in the output file.  This way we can test the Map^2 formulae
    # on the more precise output.
    # When running from the command line, the output file is made from scratch.

    gamma0 = 0.05
    r0 = 10.
    L = 50.*r0
    cat_name = os.path.join('data','gg_map.dat')
    out_name = os.path.join('data','gg_map.out')
    gg = treecorr.GGCorrelation(bin_size=0.1, min_sep=1, nbins=47, sep_units='arcmin',
                                verbose=1)
    if __name__ == "__main__":
        ngal = 1000000

        rng = np.random.RandomState(8675309)
        x = (rng.random_sample(ngal)-0.5) * L
        y = (rng.random_sample(ngal)-0.5) * L
        r2 = (x**2 + y**2)/r0**2
        g1 = -gamma0 * np.exp(-r2/2.) * (x**2-y**2)/r0**2
        g2 = -gamma0 * np.exp(-r2/2.) * (2.*x*y)/r0**2

        cat = treecorr.Catalog(x=x, y=y, g1=g1, g2=g2, x_units='arcmin', y_units='arcmin')
        cat.write(cat_name)
        gg.process(cat)
        gg.write(out_name, precision=16)
    else:
        gg.read(out_name)

    # Check MapSq calculation:
    # cf. http://adsabs.harvard.edu/abs/2004MNRAS.352..338J
    # Use Crittenden formulation, since the analytic result is simpler:
    # Map^2(R) = int 1/2 r/R^2 (T+(r/R) xi+(r) + T-(r/R) xi-(r)) dr
    #          = 6 pi gamma0^2 r0^8 R^4 / (L^2 (r0^2+R^2)^5)
    # Mx^2(R)  = int 1/2 r/R^2 (T+(r/R) xi+(r) - T-(r/R) xi-(r)) dr
    #          = 0
    # where T+(s) = (s^4-16s^2+32)/128 exp(-s^2/4)
    #       T-(s) = s^4/128 exp(-s^2/4)
    #
    # Note: Another way to calculate this, which will turn out to be helpful when we do the
    #       Map^3 calculation in test_ggg.py is as follows:
    # Map(u,v) = int( g(x,y) * ((u-x) -I(v-y))^2 / ((u-x)^2 + (v-y)^2) * Q(u-x, v-y) )
    #          = 1/2 gamma0 r0^4 R^2 / (R^2+r0^2)^5 x
    #                 ((u^2+v^2)^2 - 8 (u^2+v^2) (R^2+r0^2) + 8 (R^2+r0^2)^2) x
    #                 exp(-1/2 (u^2+v^2) / (R^2+r0^2))
    # Then, you can directly compute <Map^2>:
    # <Map^2> = int(Map(u,v)^2, u=-inf..inf, v=-inf..inf) / L^2
    #         = 6 pi gamma0^2 r0^8 R^4 / (r0^2+R^2)^5 / L^2   (i.e. the same answer as above.)
    r = gg.meanr
    true_mapsq = 6.*np.pi * gamma0**2 * r0**8 * r**4 / (L**2 * (r**2+r0**2)**5)

    mapsq, mapsq_im, mxsq, mxsq_im, varmapsq = gg.calculateMapSq()
    print('mapsq = ',mapsq)
    print('true_mapsq = ',true_mapsq)
    print('ratio = ',mapsq/true_mapsq)
    print('diff = ',mapsq-true_mapsq)
    print('max diff = ',max(abs(mapsq - true_mapsq)))
    print('max diff[16:] = ',max(abs(mapsq[16:] - true_mapsq[16:])))
    # It's pretty ratty near the start where the integral is poorly evaluated, but the
    # agreement is pretty good if we skip the first 16 elements.
    # Well, it gets bad again at the end, but those values are small enough that they still
    # pass this test.
    np.testing.assert_allclose(mapsq[16:], true_mapsq[16:], rtol=0.1, atol=1.e-9)
    print('mxsq = ',mxsq)
    print('max = ',max(abs(mxsq)))
    print('max[16:] = ',max(abs(mxsq[16:])))
    np.testing.assert_allclose(mxsq[16:], 0., atol=3.e-8)

    mapsq_file = 'output/gg_m2.txt'
    gg.writeMapSq(mapsq_file, precision=16)
    data = np.genfromtxt(os.path.join('output','gg_m2.txt'), names=True)
    np.testing.assert_allclose(data['Mapsq'], mapsq)
    np.testing.assert_allclose(data['Mxsq'], mxsq)

    # Check providing a specific range of R values
    # (We provide the range where the results worked out well above.)
    R = gg.rnom[16::2]
    print('R = ',R)
    mapsq, mapsq_im, mxsq, mxsq_im, varmapsq = gg.calculateMapSq(R)
    true_mapsq = true_mapsq[16::2]
    print('mapsq = ',mapsq)
    print('true_mapsq = ',true_mapsq)
    print('ratio = ',mapsq/true_mapsq)
    print('diff = ',mapsq-true_mapsq)
    print('max diff = ',max(abs(mapsq - true_mapsq)))
    np.testing.assert_allclose(mapsq, true_mapsq, rtol=0.1, atol=1.e-9)
    print('mxsq = ',mxsq)
    print('max = ',max(abs(mxsq)))
    np.testing.assert_allclose(mxsq, 0., atol=3.e-8)

    mapsq_file = 'output/gg_m2b.txt'
    gg.writeMapSq(mapsq_file, R=R, precision=16)
    data = np.genfromtxt(mapsq_file, names=True)
    np.testing.assert_allclose(data['Mapsq'], mapsq)
    np.testing.assert_allclose(data['Mxsq'], mxsq)

    # Also check the Schneider version.  The math isn't quite as nice here, but it is tractable
    # using a different formula than I used above:
    # Map^2(R) = int k P(k) W(kR) dk
    #          = 576 pi gamma0^2 r0^6/(L^2 R^4) exp(-R^2/2r0^2) (I4(R^2/2r0^2)
    # where I4 is the modified Bessel function with nu=4.
    try:
        from scipy.special import iv
    except ImportError:
        # Don't require scipy if the user doesn't have it.
        print('Skipping tests of Schneider aperture mass, since scipy.special not available.')
        return
    x = 0.5*r**2/r0**2
    true_mapsq = 144.*np.pi * gamma0**2 * r0**2 / (L**2 * x**2) * np.exp(-x) * iv(4,x)

    mapsq, mapsq_im, mxsq, mxsq_im, varmapsq = gg.calculateMapSq(m2_uform='Schneider')
    print('Schneider mapsq = ',mapsq)
    print('true_mapsq = ',true_mapsq)
    print('ratio = ',mapsq/true_mapsq)
    print('diff = ',mapsq-true_mapsq)
    print('max diff = ',max(abs(mapsq - true_mapsq)))
    print('max diff[26:] = ',max(abs(mapsq[26:] - true_mapsq[26:])))
    # This one stays ratty longer, so we need to skip the first 26.
    np.testing.assert_allclose(mapsq[26:], true_mapsq[26:], rtol=0.1, atol=1.e-9)
    print('mxsq = ',mxsq)
    print('max = ',max(abs(mxsq)))
    print('max[26:] = ',max(abs(mxsq[26:])))
    np.testing.assert_allclose(mxsq[26:], 0, atol=3.e-8)

    # Finally, check the <gamma^2>(R) calculation.
    # Gam^2(R) = int k P(k) Wth(kR) dk
    #          = 2pi gamma0^2 (r0/L)^2 exp(-r^2/2r0^2)  *
    #               (BesselI(0, r^2/2r0^2) - BesselI(1, r^2/2r0^2))
    x = 0.5*r**2/r0**2
    true_gamsq = 2.*np.pi*gamma0**2 * r0**2 / L**2 * np.exp(-x) * (iv(0,x) - iv(1,x))

    gamsq, vargamsq = gg.calculateGamSq()
    print('gamsq = ',gamsq)
    print('true_gamsq = ',true_gamsq)
    print('ratio = ',gamsq/true_gamsq)
    print('diff = ',gamsq-true_gamsq)
    print('max diff = ',max(abs(gamsq - true_gamsq)))
    print('max rel diff[12:33] = ',max(abs((gamsq[12:33] - true_gamsq[12:33])/true_gamsq[12:33])))
    # This is only close in a narrow range of scales
    np.testing.assert_allclose(gamsq[12:33], true_gamsq[12:33], rtol=0.1)
    # Everywhere else it is less (since integral misses unmeasured power at both ends).
    np.testing.assert_array_less(gamsq, true_gamsq)

    # With E/B decomposition, it's ok over a larger range of scales.
    gamsq, vargamsq, gamsq_e, gamsq_b, vargamsq_eb = gg.calculateGamSq(eb=True)
    print('gamsq_e = ',gamsq_e)
    print('true_gamsq = ',true_gamsq)
    print('ratio = ',gamsq_e/true_gamsq)
    print('diff = ',gamsq_e-true_gamsq)
    print('max diff = ',max(abs(gamsq_e - true_gamsq)))
    print('rel diff[6:41] = ',(gamsq_e[6:41] - true_gamsq[6:41])/true_gamsq[6:41])
    print('max rel diff[6:41] = ',max(abs((gamsq_e[6:41] - true_gamsq[6:41])/true_gamsq[6:41])))
    # This is only close in a narrow range of scales
    np.testing.assert_allclose(gamsq_e[6:41], true_gamsq[6:41], rtol=0.1)
    print('gamsq_b = ',gamsq_b)
    np.testing.assert_allclose(gamsq_b[6:41], 0, atol=1.e-6)

    # Check providing a specific range of R values
    # (We provide the range where the results worked out well above.)
    R = gg.rnom[6:40:4]
    print('R = ',R)
    gamsq, vargamsq, gamsq_e, gamsq_b, vargamsq_eb = gg.calculateGamSq(R, eb=True)
    true_gamsq = true_gamsq[6:40:4]
    print('gamsq_e = ',gamsq_e)
    print('true_gamsq = ',true_gamsq)
    print('ratio = ',gamsq_e/true_gamsq)
    print('diff = ',gamsq_e-true_gamsq)
    print('max diff = ',max(abs(gamsq_e - true_gamsq)))
    np.testing.assert_allclose(gamsq_e, true_gamsq, rtol=0.1)
    print('gamsq_b = ',gamsq_b)
    np.testing.assert_allclose(gamsq_b, 0, atol=1.e-6)

    # Not valid with TwoD or Linear binning
    gg2 = treecorr.GGCorrelation(bin_size=0.1, min_sep=1., max_sep=100., sep_units='arcmin',
                                 bin_type='Linear')
    with assert_raises(ValueError):
        gg2.calculateMapSq()
    with assert_raises(ValueError):
        gg2.calculateGamSq()
    gg3 = treecorr.GGCorrelation(bin_size=0.1, min_sep=1., max_sep=100., sep_units='arcmin',
                                 bin_type='TwoD')
    with assert_raises(ValueError):
        gg3.calculateMapSq()
    with assert_raises(ValueError):
        gg3.calculateGamSq()



@timer
def test_spherical():
    # This is the same field we used for test_gg, but put into spherical coords.
    # We do the spherical trig by hand using the obvious formulae, rather than the clever
    # optimizations that are used by the TreeCorr code, thus serving as a useful test of
    # the latter.

    gamma0 = 0.05
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
    g1 = -gamma0 * np.exp(-r2/2./r0**2) * (x**2-y**2)/r0**2
    g2 = -gamma0 * np.exp(-r2/2./r0**2) * (2.*x*y)/r0**2
    r = np.sqrt(r2)
    theta = np.arctan2(y,x)

    gg = treecorr.GGCorrelation(bin_size=0.1, min_sep=1., max_sep=100., sep_units='arcmin',
                                verbose=1)
    r1 = np.exp(gg.logr) * (coord.arcmin / coord.radians)
    temp = np.pi/16. * gamma0**2 * (r0/L)**2 * np.exp(-0.25*r1**2/r0**2)
    true_xip = temp * (r1**4 - 16.*r1**2*r0**2 + 32.*r0**4)/r0**4
    true_xim = temp * r1**4/r0**4

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

        # Rotate shear relative to local west
        # gamma_sph = exp(2i beta) * gamma
        # where beta = pi - (A+B) is the angle between north and "up" in the tangent plane.
        beta = np.pi - (A+B)
        beta[x>0] *= -1.
        cos2beta = np.cos(2.*beta)
        sin2beta = np.sin(2.*beta)
        g1_sph = g1 * cos2beta - g2 * sin2beta
        g2_sph = g2 * cos2beta + g1 * sin2beta

        cat = treecorr.Catalog(ra=ra, dec=dec, g1=g1_sph, g2=g2_sph, ra_units='rad',
                               dec_units='rad')
        gg = treecorr.GGCorrelation(bin_size=0.1, min_sep=1., max_sep=100., sep_units='arcmin',
                                    verbose=1)
        gg.process(cat)

        print('ra0, dec0 = ',ra0,dec0)
        print('gg.xip = ',gg.xip)
        print('true_xip = ',true_xip)
        print('ratio = ',gg.xip / true_xip)
        print('diff = ',gg.xip - true_xip)
        print('max diff = ',max(abs(gg.xip - true_xip)))
        # The 3rd and 4th centers are somewhat less accurate.  Not sure why.
        # The math seems to be right, since the last one that gets all the way to the pole
        # works, so I'm not sure what is going on.  It's just a few bins that get a bit less
        # accurate.  Possibly worth investigating further at some point...
        assert max(abs(gg.xip - true_xip)) < 3.e-7 * tol_factor

        print('gg.xim = ',gg.xim)
        print('true_xim = ',true_xim)
        print('ratio = ',gg.xim / true_xim)
        print('diff = ',gg.xim - true_xim)
        print('max diff = ',max(abs(gg.xim - true_xim)))
        assert max(abs(gg.xim - true_xim)) < 2.e-7 * tol_factor

    # One more center that can be done very easily.  If the center is the north pole, then all
    # the tangential shears are pure (positive) g1.
    ra0 = 0
    dec0 = np.pi/2.
    ra = theta
    dec = np.pi/2. - 2.*np.arcsin(r/2.)
    gammat = -gamma0 * r2/r0**2 * np.exp(-r2/2./r0**2)

    cat = treecorr.Catalog(ra=ra, dec=dec, g1=gammat, g2=np.zeros_like(gammat), ra_units='rad',
                           dec_units='rad')
    gg.process(cat)

    print('gg.xip = ',gg.xip)
    print('gg.xip_im = ',gg.xip_im)
    print('true_xip = ',true_xip)
    print('ratio = ',gg.xip / true_xip)
    print('diff = ',gg.xip - true_xip)
    print('max diff = ',max(abs(gg.xip - true_xip)))
    assert max(abs(gg.xip - true_xip)) < 3.e-7 * tol_factor
    assert max(abs(gg.xip_im)) < 3.e-7 * tol_factor

    print('gg.xim = ',gg.xim)
    print('gg.xim_im = ',gg.xim_im)
    print('true_xim = ',true_xim)
    print('ratio = ',gg.xim / true_xim)
    print('diff = ',gg.xim - true_xim)
    print('max diff = ',max(abs(gg.xim - true_xim)))
    assert max(abs(gg.xim - true_xim)) < 2.e-7 * tol_factor
    assert max(abs(gg.xim_im)) < 2.e-7 * tol_factor

    # Check that we get the same result using the corr2 function
    cat.write(os.path.join('data','gg_spherical.dat'))
    config = treecorr.read_config('configs/gg_spherical.yaml')
    config['verbose'] = 0
    treecorr.corr2(config)
    corr2_output = np.genfromtxt(os.path.join('output','gg_spherical.out'), names=True,
                                 skip_header=1)
    print('gg.xip = ',gg.xip)
    print('from corr2 output = ',corr2_output['xip'])
    print('ratio = ',corr2_output['xip']/gg.xip)
    print('diff = ',corr2_output['xip']-gg.xip)
    np.testing.assert_allclose(corr2_output['xip'], gg.xip, rtol=1.e-3)

    print('gg.xim = ',gg.xim)
    print('from corr2 output = ',corr2_output['xim'])
    print('ratio = ',corr2_output['xim']/gg.xim)
    print('diff = ',corr2_output['xim']-gg.xim)
    np.testing.assert_allclose(corr2_output['xim'], gg.xim, rtol=1.e-3)

    print('xip_im from corr2 output = ',corr2_output['xip_im'])
    assert max(abs(corr2_output['xip_im'])) < 3.e-7 * tol_factor

    print('xim_im from corr2 output = ',corr2_output['xim_im'])
    assert max(abs(corr2_output['xim_im'])) < 2.e-7 * tol_factor


@timer
def test_aardvark():

    # Eric Suchyta did a brute force calculation of the Aardvark catalog, so it is useful to
    # compare the output from my code with that.

    get_from_wiki('Aardvark.fit')
    file_name = os.path.join('data','Aardvark.fit')
    config = treecorr.read_config('Aardvark.yaml')
    config['verbose'] = 1
    cat1 = treecorr.Catalog(file_name, config)
    gg = treecorr.GGCorrelation(config)
    gg.process(cat1)

    direct_file_name = os.path.join('data','Aardvark.direct')
    direct_data = np.genfromtxt(direct_file_name)
    direct_xip = direct_data[:,3]
    direct_xim = direct_data[:,4]

    #print('gg.xip = ',gg.xip)
    #print('direct.xip = ',direct_xip)

    xip_err = gg.xip - direct_xip
    print('xip_err = ',xip_err)
    print('max = ',max(abs(xip_err)))
    assert max(abs(xip_err)) < 2.e-7
    print('xip_im = ',gg.xip_im)
    print('max = ',max(abs(gg.xip_im)))
    assert max(abs(gg.xip_im)) < 3.e-7

    xim_err = gg.xim - direct_xim
    print('xim_err = ',xim_err)
    print('max = ',max(abs(xim_err)))
    assert max(abs(xim_err)) < 1.e-7
    print('xim_im = ',gg.xim_im)
    print('max = ',max(abs(gg.xim_im)))
    assert max(abs(gg.xim_im)) < 1.e-7

    # However, after some back and forth about the calculation, we concluded that Eric hadn't
    # done the spherical trig correctly to get the shears relative to the great circle joining
    # the two positions.  So let's compare with my own brute force calculation.
    # This also has the advantage that the radial bins are done the same way -- uniformly
    # spaced in log of the chord distance, rather than the great circle distance.

    bs0_file_name = os.path.join('data','Aardvark.bs0')
    bs0_data = np.genfromtxt(bs0_file_name)
    bs0_xip = bs0_data[:,2]
    bs0_xim = bs0_data[:,3]

    #print('gg.xip = ',gg.xip)
    #print('bs0.xip = ',bs0_xip)

    xip_err = gg.xip - bs0_xip
    print('xip_err = ',xip_err)
    print('max = ',max(abs(xip_err)))
    assert max(abs(xip_err)) < 1.e-7

    xim_err = gg.xim - bs0_xim
    print('xim_err = ',xim_err)
    print('max = ',max(abs(xim_err)))
    assert max(abs(xim_err)) < 5.e-8

    # Check that we get the same result using the corr2 function
    # There's nothing new here coverage-wise, so only do this when running from command line.
    if __name__ == '__main__':
        treecorr.corr2(config)
        corr2_output = np.genfromtxt(os.path.join('output','Aardvark.out'), names=True,
                                     skip_header=1)
        print('gg.xip = ',gg.xip)
        print('from corr2 output = ',corr2_output['xip'])
        print('ratio = ',corr2_output['xip']/gg.xip)
        print('diff = ',corr2_output['xip']-gg.xip)
        np.testing.assert_allclose(corr2_output['xip'], gg.xip, rtol=1.e-3)

        print('gg.xim = ',gg.xim)
        print('from corr2 output = ',corr2_output['xim'])
        print('ratio = ',corr2_output['xim']/gg.xim)
        print('diff = ',corr2_output['xim']-gg.xim)
        np.testing.assert_allclose(corr2_output['xim'], gg.xim, rtol=1.e-3)

        print('xip_im from corr2 output = ',corr2_output['xip_im'])
        print('max err = ',max(abs(corr2_output['xip_im'])))
        assert max(abs(corr2_output['xip_im'])) < 3.e-7
        print('xim_im from corr2 output = ',corr2_output['xim_im'])
        print('max err = ',max(abs(corr2_output['xim_im'])))
        assert max(abs(corr2_output['xim_im'])) < 1.e-7

    # As bin_slop decreases, the agreement should get even better.
    # This test is slow, so only do it if running test_gg.py directly.
    if __name__ == '__main__':
        config['bin_slop'] = 0.2
        gg = treecorr.GGCorrelation(config)
        gg.process(cat1)

        #print('gg.xip = ',gg.xip)
        #print('bs0.xip = ',bs0_xip)

        xip_err = gg.xip - bs0_xip
        print('xip_err = ',xip_err)
        print('max = ',max(abs(xip_err)))
        assert max(abs(xip_err)) < 2.e-8

        xim_err = gg.xim - bs0_xim
        print('xim_err = ',xim_err)
        print('max = ',max(abs(xim_err)))
        assert max(abs(xim_err)) < 3.e-8


@timer
def test_shuffle():
    # Check that the code is insensitive to shuffling the input data vectors.

    # Might as well use the same function as above, although I reduce L a bit.
    ngal = 10000
    gamma0 = 0.05
    r0 = 10.
    L = 5. * r0
    rng = np.random.RandomState(8675309)
    x = (rng.random_sample(ngal)-0.5) * L
    y = (rng.random_sample(ngal)-0.5) * L
    r2 = (x**2 + y**2)/r0**2
    g1 = -gamma0 * np.exp(-r2/2.) * (x**2-y**2)/r0**2
    g2 = -gamma0 * np.exp(-r2/2.) * (2.*x*y)/r0**2

    cat_u = treecorr.Catalog(x=x, y=y, g1=g1, g2=g2)
    gg_u = treecorr.GGCorrelation(bin_size=0.1, min_sep=1., max_sep=30., verbose=1)
    gg_u.process(cat_u)

    # Put these in a single 2d array so we can easily use np.random.shuffle
    data = np.array( [x, y, g1, g2] ).T
    print('data = ',data)
    rng.shuffle(data)

    cat_s = treecorr.Catalog(x=data[:,0], y=data[:,1], g1=data[:,2], g2=data[:,3])
    gg_s = treecorr.GGCorrelation(bin_size=0.1, min_sep=1., max_sep=30., verbose=1)
    gg_s.process(cat_s)

    print('gg_u.xip = ',gg_u.xip)
    print('gg_s.xip = ',gg_s.xip)
    print('ratio = ',gg_u.xip / gg_s.xip)
    print('diff = ',gg_u.xip - gg_s.xip)
    print('max diff = ',max(abs(gg_u.xip - gg_s.xip)))
    assert max(abs(gg_u.xip - gg_s.xip)) < 1.e-14

@timer
def test_haloellip():
    """Test that the constant and quadrupole versions of the Clampitt halo ellipticity calculation
    are equivalent to xi+ and xi- (respectively) of the shear-shear cross correlation, where
    the halo ellipticities are normalized to |g_lens|=1.

    Joseph's original formulation: (cf. Issue #36, although I correct what I believe is an error
    in his gamma_Qx formula.)

    gamma_Q = Sum_i (w_i * g1_i * cos(4theta) + w_i * g2_i * sin(4theta)) / Sum_i (w_i)
    gamma_C = Sum_i (w_i * g1_i) / Sum_i (w_i)

    gamma_Qx = Sum_i (w_i * g2_i * cos(4theta) - w_i * g1_i * sin(4theta)) / Sum_i (w_i)
    gamma_Cx = Sum_i (w_i * g2_i) / Sum_i (w_i)

    where g1,g2 and theta are measured w.r.t. the coordinate system where the halo ellitpicity
    is along the x-axis.  Converting this to complex notation, we obtain:

    gamma_C + i gamma_Cx = < g1 + i g2 >
                         = < gobs exp(-2iphi) >
                         = < gobs elens* >
    gamma_Q + i gamma_Qx = < (g1 + i g2) (cos(4t) - isin(4t) >
                         = < gobs exp(-2iphi) exp(-4itheta) >
                         = < gobs exp(2iphi) exp(-4i(theta+phi)) >
                         = < gobs elens exp(-4i(theta+phi)) >

    where gobs is the observed shape of the source in the normal world coordinate system, and
    elens = exp(2iphi) is the unit-normalized shape of the lens in that same coordinate system.
    Note that the combination theta+phi is the angle between the line joining the two points
    and the E-W coordinate, which means that

    gamma_C + i gamma_Cx = xi+(elens, gobs)
    gamma_Q + i gamma_Qx = xi-(elens, gobs)

    We test this result here using the above formulation with both unit weights and weights
    proportional to the halo ellitpicity.  We also try keeping the magnitude of elens rather
    than normalizing it.
    """

    if __name__ == '__main__':
        # It's hard to get enough sources/lenses to get very high precision on these tests.
        # We settle on a number that lead to 3% accuracy.  Increasing nlens and nsource
        # lead to high accuracy.
        nlens = 1000
        nsource = 10000  # sources per lens
        tol = 3.e-2
    else:
        # For nosetests runs, use 10x fewer lenses and 2x larger tolerance
        nlens = 100
        nsource = 10000
        tol = 6.e-2

    ntot = nsource * nlens
    L = 100000.  # The side length in which the lenses are placed
    R = 10.      # The (rms) radius of the associated sources from the lenses
                 # In this case, we want L >> R so that most sources are only associated
                 # with the one lens we used for assigning its shear value.

    # Lenses are randomly located with random shapes.
    rng = np.random.RandomState(8675309)
    lens_g1 = rng.normal(0., 0.1, (nlens,))
    lens_g2 = rng.normal(0., 0.1, (nlens,))
    lens_g = lens_g1 + 1j * lens_g2
    lens_absg = np.abs(lens_g)
    lens_x = (rng.random_sample(nlens)-0.5) * L
    lens_y = (rng.random_sample(nlens)-0.5) * L
    print('Made lenses')

    e_a = 0.17  # The amplitude of the constant part of the signal
    e_b = 0.23  # The amplitude of the quadrupole part of the signal
    source_g1 = np.empty(ntot)
    source_g2 = np.empty(ntot)
    source_x = np.empty(ntot)
    source_y = np.empty(ntot)
    # For the sources, place 100 galaxies around each lens with the expected azimuthal pattern
    # I just use a constant |g| for the amplitude, not a real radial pattern.
    for i in range(nlens):
        # First build the signal as it appears in the coordinate system where the halo
        # is oriented along the x-axis
        dx = rng.normal(0., R, (nsource,))
        dy = rng.normal(0., R, (nsource,))
        z = dx + 1j * dy
        exp2iphi = z**2 / np.abs(z)**2
        source_g = e_a + e_b * exp2iphi**2
        # Now rotate the whole system by the phase of the lens ellipticity.
        exp2ialpha = lens_g[i] / lens_absg[i]
        expialpha = np.sqrt(exp2ialpha)
        source_g *= exp2ialpha
        z *= expialpha
        # Also scale the signal by |lens_g|
        source_g *= lens_absg[i]
        # Place the source galaxies at this dx,dy with this shape
        source_x[i*nsource: (i+1)*nsource] = lens_x[i] + z.real
        source_y[i*nsource: (i+1)*nsource] = lens_y[i] + z.imag
        source_g1[i*nsource: (i+1)*nsource] = source_g.real
        source_g2[i*nsource: (i+1)*nsource] = source_g.imag
    print('Made sources')

    source_cat = treecorr.Catalog(x=source_x, y=source_y, g1=source_g1, g2=source_g2)
    gg = treecorr.GGCorrelation(min_sep=1, bin_size=0.1, nbins=35)
    lens_mean_absg = np.mean(lens_absg)
    print('mean_absg = ',lens_mean_absg)

    # First the original version where we only use the phase of the lens ellipticities:
    lens_cat1 = treecorr.Catalog(x=lens_x, y=lens_y, g1=lens_g1/lens_absg, g2=lens_g2/lens_absg)
    gg.process(lens_cat1, source_cat)
    print('gg.xim = ',gg.xim)
    # The net signal here is just <absg> * e_b
    print('expected signal = ',e_b * lens_mean_absg)
    np.testing.assert_allclose(gg.xim, e_b * lens_mean_absg, rtol=tol)
    print('gg.xip = ',gg.xip)
    print('expected signal = ',e_a * lens_mean_absg)
    np.testing.assert_allclose(gg.xip, e_a * lens_mean_absg, rtol=tol)

    # Next weight the lenses by their absg.
    lens_cat2 = treecorr.Catalog(x=lens_x, y=lens_y, g1=lens_g1/lens_absg, g2=lens_g2/lens_absg,
                                w=lens_absg)
    gg.process(lens_cat2, source_cat)
    print('gg.xim = ',gg.xim)
    # Now the net signal is
    # sum(w * e_b*absg[i]) / sum(w)
    # = sum(absg[i]^2 * e_b) / sum(absg[i])
    # = <absg^2> * e_b / <absg>
    lens_mean_gsq = np.mean(lens_absg**2)
    print('expected signal = ',e_b * lens_mean_gsq / lens_mean_absg)
    np.testing.assert_allclose(gg.xim, e_b * lens_mean_gsq / lens_mean_absg, rtol=tol)
    print('gg.xip = ',gg.xip)
    print('expected signal = ',e_a * lens_mean_gsq / lens_mean_absg)
    np.testing.assert_allclose(gg.xip, e_a * lens_mean_gsq / lens_mean_absg, rtol=tol)

    # Finally, use the unnormalized lens_g for the lens ellipticities
    lens_cat3 = treecorr.Catalog(x=lens_x, y=lens_y, g1=lens_g1, g2=lens_g2)
    gg.process(lens_cat3, source_cat)
    print('gg.xim = ',gg.xim)
    # Now the net signal is
    # sum(absg[i] * e_b*absg[i]) / N
    # = sum(absg[i]^2 * e_b) / N
    # = <absg^2> * e_b
    print('expected signal = ',e_b * lens_mean_gsq)
    # This one is slightly less accurate.  But easily passes at 4% accuracy.
    np.testing.assert_allclose(gg.xim, e_b * lens_mean_gsq, rtol=tol*1.5)
    print('gg.xip = ',gg.xip)
    print('expected signal = ',e_a * lens_mean_gsq)
    np.testing.assert_allclose(gg.xip, e_a * lens_mean_gsq, rtol=tol*1.5)

    # It's worth noting that exactly half the signal is in each of g1, g2, so for things
    # like SDSS, you can use only g2, for instance, which avoids some insidious systematic
    # errors related to the scan direction.
    source_cat2 = treecorr.Catalog(x=source_x, y=source_y,
                                   g1=np.zeros_like(source_g2), g2=source_g2)
    gg.process(lens_cat1, source_cat2)
    print('gg.xim = ',gg.xim)
    print('expected signal = ',e_b * lens_mean_absg / 2.)
    # The precision of this is a bit less though, since we now have more shape noise.
    # Naively, I would expect sqrt(2) worse, but since the agreement in this test is largely
    # artificial, as I placed the exact signal down with no shape noise, the increased shape
    # noise is a lot more than previously here.  So I had to drop the precision by a factor of
    # 5 relative to what I did above.
    np.testing.assert_allclose(gg.xim, e_b * lens_mean_absg/2., rtol=tol*5)
    print('gg.xip = ',gg.xip)
    print('expected signal = ',e_a * lens_mean_absg / 2.)
    np.testing.assert_allclose(gg.xip, e_a * lens_mean_absg/2., rtol=tol*5)

@timer
def test_varxi():
    # Test that varxip, varxim are correct (or close) based on actual variance of many runs.

    # Same gamma pattern as in test_gg().  Although the signal doesn't actually matter at all here.
    gamma0 = 0.05
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


    all_ggs = []
    for run in range(nruns):
        # In addition to the shape noise below, there is shot noise from the random x,y positions.
        x = (rng.random_sample(ngal)-0.5) * L
        y = (rng.random_sample(ngal)-0.5) * L
        # Varied weights are hard, but at least check that non-unit weights work correctly.
        w = np.ones_like(x) * 5
        r2 = (x**2 + y**2)/r0**2
        g1 = -gamma0 * np.exp(-r2/2.) * (x**2-y**2)/r0**2
        g2 = -gamma0 * np.exp(-r2/2.) * (2.*x*y)/r0**2
        # This time, add some shape noise (different each run).
        g1 += rng.normal(0, 0.3, size=ngal)
        g2 += rng.normal(0, 0.3, size=ngal)

        cat = treecorr.Catalog(x=x, y=y, w=w, g1=g1, g2=g2, x_units='arcmin', y_units='arcmin')
        gg = treecorr.GGCorrelation(bin_size=0.1, min_sep=10., max_sep=100., sep_units='arcmin',
                                    verbose=1)
        gg.process(cat)
        all_ggs.append(gg)

    mean_xip = np.mean([gg.xip for gg in all_ggs], axis=0)
    var_xip = np.var([gg.xip for gg in all_ggs], axis=0)
    mean_xim = np.mean([gg.xim for gg in all_ggs], axis=0)
    var_xim = np.var([gg.xim for gg in all_ggs], axis=0)
    mean_varxip = np.mean([gg.varxip for gg in all_ggs], axis=0)
    mean_varxim = np.mean([gg.varxim for gg in all_ggs], axis=0)

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
    np.testing.assert_allclose(mean_varxip, var_xip, rtol=0.02 * tol_factor)
    np.testing.assert_allclose(mean_varxim, var_xim, rtol=0.02 * tol_factor)


if __name__ == '__main__':
    test_direct()
    test_direct_spherical()
    test_pairwise()
    test_gg()
    test_mapsq()
    test_spherical()
    test_aardvark()
    test_shuffle()
    test_haloellip()
    test_varxi
