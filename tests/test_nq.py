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
    q12 = rng.normal(0,0.2, (ngal,) )
    q22 = rng.normal(0,0.2, (ngal,) )

    cat1 = treecorr.Catalog(x=x1, y=y1, w=w1)
    cat2 = treecorr.Catalog(x=x2, y=y2, w=w2, q1=q12, q2=q22)

    min_sep = 1.
    max_sep = 50.
    nbins = 50
    bin_size = np.log(max_sep/min_sep) / nbins
    nq = treecorr.NQCorrelation(min_sep=min_sep, max_sep=max_sep, nbins=nbins, brute=True)
    nq.process(cat1, cat2)

    true_npairs = np.zeros(nbins, dtype=int)
    true_weight = np.zeros(nbins, dtype=float)
    true_xi = np.zeros(nbins, dtype=complex)
    for i in range(ngal):
        # It's hard to do all the pairs at once with numpy operations (although maybe possible).
        # But we can at least do all the pairs for each entry in cat1 at once with arrays.
        rsq = (x1[i]-x2)**2 + (y1[i]-y2)**2
        r = np.sqrt(rsq)
        expmialpha = ((x2 - x1[i]) - 1j*(y2 - y1[i])) / r

        ww = w1[i] * w2
        xi = ww * (q12 + 1j*q22) * expmialpha**4

        index = np.floor(np.log(r/min_sep) / bin_size).astype(int)
        mask = (index >= 0) & (index < nbins)
        np.add.at(true_npairs, index[mask], 1)
        np.add.at(true_weight, index[mask], ww[mask])
        np.add.at(true_xi, index[mask], xi[mask])

    true_xi /= true_weight

    print('true_npairs = ',true_npairs)
    print('diff = ',nq.npairs - true_npairs)
    np.testing.assert_array_equal(nq.npairs, true_npairs)

    print('true_weight = ',true_weight)
    print('diff = ',nq.weight - true_weight)
    np.testing.assert_allclose(nq.weight, true_weight, rtol=1.e-6, atol=1.e-8)

    print('true_xi = ',true_xi)
    print('nq.xi = ',nq.xi)
    print('nq.xi_im = ',nq.xi_im)
    np.testing.assert_allclose(nq.xi, true_xi.real, rtol=1.e-6, atol=1.e-8)
    np.testing.assert_allclose(nq.xi_im, true_xi.imag, rtol=1.e-6, atol=1.e-8)

    # Check that running via the corr2 script works correctly.
    config = treecorr.config.read_config('configs/nq_direct.yaml')
    try:
        import fitsio
    except ImportError:
        pass
    else:
        cat1.write(config['file_name'])
        cat2.write(config['file_name2'])
        with CaptureLog() as cl:
            treecorr.corr2(config, logger=cl.logger)
        assert "skipping q1_col" in cl.output
        data = fitsio.read(config['nq_file_name'])
        np.testing.assert_allclose(data['r_nom'], nq.rnom)
        np.testing.assert_allclose(data['npairs'], nq.npairs)
        np.testing.assert_allclose(data['weight'], nq.weight)
        np.testing.assert_allclose(data['qR'], nq.xi)
        np.testing.assert_allclose(data['qR_im'], nq.xi_im)

        # When not using corr2, it's invalid to specify invalid q1_col, q2_col
        with assert_raises(ValueError):
            cat = treecorr.Catalog(config['file_name'], config)

        # Invalid with only one file_name
        del config['file_name2']
        with assert_raises(TypeError):
            treecorr.corr2(config)
        config['file_name2'] = 'data/nq_direct_cat2.fits'
        # Invalid to request compoensated if no rand_file
        config['nq_statistic'] = 'compensated'
        with assert_raises(TypeError):
            treecorr.corr2(config)

    # Repeat with binslop = 0
    # And don't do any top-level recursion so we actually test not going to the leaves.
    nq = treecorr.NQCorrelation(min_sep=min_sep, max_sep=max_sep, nbins=nbins, bin_slop=0,
                                max_top=0)
    nq.process(cat1, cat2)
    np.testing.assert_array_equal(nq.npairs, true_npairs)
    np.testing.assert_allclose(nq.weight, true_weight, rtol=1.e-6, atol=1.e-8)
    np.testing.assert_allclose(nq.xi, true_xi.real, atol=5.e-4)
    np.testing.assert_allclose(nq.xi_im, true_xi.imag, atol=3.e-4)

    # With angle_slop = 0, it goes back to being basically exact (to single precision).
    nq = treecorr.NQCorrelation(min_sep=min_sep, max_sep=max_sep, nbins=nbins, bin_slop=0,
                                angle_slop=0, max_top=0)
    nq.process(cat1, cat2)
    np.testing.assert_array_equal(nq.npairs, true_npairs)
    np.testing.assert_allclose(nq.weight, true_weight, rtol=1.e-6, atol=1.e-8)
    np.testing.assert_allclose(nq.xi, true_xi.real, rtol=1.e-6, atol=1.e-8)
    np.testing.assert_allclose(nq.xi_im, true_xi.imag, rtol=1.e-6, atol=1.e-8)

    # Check a few basic operations with a NQCorrelation object.
    do_pickle(nq)

    nq2 = nq.copy()
    nq2 += nq
    np.testing.assert_allclose(nq2.npairs, 2*nq.npairs)
    np.testing.assert_allclose(nq2.weight, 2*nq.weight)
    np.testing.assert_allclose(nq2.meanr, 2*nq.meanr)
    np.testing.assert_allclose(nq2.meanlogr, 2*nq.meanlogr)
    np.testing.assert_allclose(nq2.xi, 2*nq.xi)
    np.testing.assert_allclose(nq2.xi_im, 2*nq.xi_im)

    nq2.clear()
    nq2 += nq
    np.testing.assert_allclose(nq2.npairs, nq.npairs)
    np.testing.assert_allclose(nq2.weight, nq.weight)
    np.testing.assert_allclose(nq2.meanr, nq.meanr)
    np.testing.assert_allclose(nq2.meanlogr, nq.meanlogr)
    np.testing.assert_allclose(nq2.xi, nq.xi)
    np.testing.assert_allclose(nq2.xi_im, nq.xi_im)

    ascii_name = 'output/nq_ascii.txt'
    nq.write(ascii_name, precision=16)
    nq3 = treecorr.NQCorrelation(min_sep=min_sep, max_sep=max_sep, nbins=nbins, bin_type='Log')
    nq3.read(ascii_name)
    np.testing.assert_allclose(nq3.npairs, nq.npairs)
    np.testing.assert_allclose(nq3.weight, nq.weight)
    np.testing.assert_allclose(nq3.meanr, nq.meanr)
    np.testing.assert_allclose(nq3.meanlogr, nq.meanlogr)
    np.testing.assert_allclose(nq3.xi, nq.xi)
    np.testing.assert_allclose(nq3.xi_im, nq.xi_im)

    # Check that the repr is minimal
    assert repr(nq3) == f'NQCorrelation(min_sep={min_sep}, max_sep={max_sep}, nbins={nbins})'

    # Simpler API using from_file:
    with CaptureLog() as cl:
        nq3b = treecorr.NQCorrelation.from_file(ascii_name, logger=cl.logger)
    assert ascii_name in cl.output
    np.testing.assert_allclose(nq3b.npairs, nq.npairs)
    np.testing.assert_allclose(nq3b.weight, nq.weight)
    np.testing.assert_allclose(nq3b.meanr, nq.meanr)
    np.testing.assert_allclose(nq3b.meanlogr, nq.meanlogr)
    np.testing.assert_allclose(nq3b.xi, nq.xi)
    np.testing.assert_allclose(nq3b.xi_im, nq.xi_im)

    # or using the Corr2 base class
    with CaptureLog() as cl:
        nq3c = treecorr.Corr2.from_file(ascii_name, logger=cl.logger)
    assert ascii_name in cl.output
    np.testing.assert_allclose(nq3c.npairs, nq.npairs)
    np.testing.assert_allclose(nq3c.weight, nq.weight)
    np.testing.assert_allclose(nq3c.meanr, nq.meanr)
    np.testing.assert_allclose(nq3c.meanlogr, nq.meanlogr)
    np.testing.assert_allclose(nq3c.xi, nq.xi)
    np.testing.assert_allclose(nq3c.xi_im, nq.xi_im)

    try:
        import fitsio
    except ImportError:
        pass
    else:
        fits_name = 'output/nq_fits.fits'
        nq.write(fits_name)
        nq4 = treecorr.NQCorrelation(min_sep=min_sep, max_sep=max_sep, nbins=nbins)
        nq4.read(fits_name)
        np.testing.assert_allclose(nq4.npairs, nq.npairs)
        np.testing.assert_allclose(nq4.weight, nq.weight)
        np.testing.assert_allclose(nq4.meanr, nq.meanr)
        np.testing.assert_allclose(nq4.meanlogr, nq.meanlogr)
        np.testing.assert_allclose(nq4.xi, nq.xi)
        np.testing.assert_allclose(nq4.xi_im, nq.xi_im)

        nq4b = treecorr.NQCorrelation.from_file(fits_name)
        np.testing.assert_allclose(nq4b.npairs, nq.npairs)
        np.testing.assert_allclose(nq4b.weight, nq.weight)
        np.testing.assert_allclose(nq4b.meanr, nq.meanr)
        np.testing.assert_allclose(nq4b.meanlogr, nq.meanlogr)
        np.testing.assert_allclose(nq4b.xi, nq.xi)
        np.testing.assert_allclose(nq4b.xi_im, nq.xi_im)

    with assert_raises(TypeError):
        nq2 += config
    nq4 = treecorr.NQCorrelation(min_sep=min_sep/2, max_sep=max_sep, nbins=nbins)
    with assert_raises(ValueError):
        nq2 += nq4
    nq5 = treecorr.NQCorrelation(min_sep=min_sep, max_sep=max_sep*2, nbins=nbins)
    with assert_raises(ValueError):
        nq2 += nq5
    nq6 = treecorr.NQCorrelation(min_sep=min_sep, max_sep=max_sep, nbins=nbins*2)
    with assert_raises(ValueError):
        nq2 += nq6
    with assert_raises(ValueError):
        nq.process(cat1, cat2, patch_method='nonlocal')


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
    q12 = rng.normal(0,0.2, (ngal,) )
    q22 = rng.normal(0,0.2, (ngal,) )

    ra1, dec1 = coord.CelestialCoord.xyz_to_radec(x1,y1,z1)
    ra2, dec2 = coord.CelestialCoord.xyz_to_radec(x2,y2,z2)

    cat1 = treecorr.Catalog(ra=ra1, dec=dec1, ra_units='rad', dec_units='rad', w=w1)
    cat2 = treecorr.Catalog(ra=ra2, dec=dec2, ra_units='rad', dec_units='rad', w=w2, q1=q12, q2=q22)

    min_sep = 1.
    max_sep = 10.
    nbins = 50
    bin_size = np.log(max_sep/min_sep) / nbins
    nq = treecorr.NQCorrelation(min_sep=min_sep, max_sep=max_sep, nbins=nbins,
                                sep_units='deg', brute=True)
    nq.process(cat1, cat2)

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

            # Rotate feild to coordinates where line connecting is horizontal.
            # Original orientation is where north is up.
            theta2 = 90*coord.degrees + c2[j].angleBetween(c1[i], north_pole)
            exp4theta2 = np.cos(4*theta2) + 1j * np.sin(4*theta2)

            q2 = q12[j] + 1j * q22[j]
            q2 *= exp4theta2

            ww = w1[i] * w2[j]
            xi = w1[i] * w2[j] * q2

            true_npairs[index] += 1
            true_weight[index] += ww
            true_xi[index] += xi

    true_xi /= true_weight

    print('true_npairs = ',true_npairs)
    print('diff = ',nq.npairs - true_npairs)
    np.testing.assert_array_equal(nq.npairs, true_npairs)

    print('true_weight = ',true_weight)
    print('diff = ',nq.weight - true_weight)
    np.testing.assert_allclose(nq.weight, true_weight, rtol=1.e-6, atol=1.e-8)

    print('true_xi = ',true_xi)
    print('nq.xi = ',nq.xi)
    print('nq.xi_im = ',nq.xi_im)
    np.testing.assert_allclose(nq.xi, true_xi.real, rtol=1.e-6, atol=1.e-8)
    np.testing.assert_allclose(nq.xi_im, true_xi.imag, rtol=1.e-6, atol=1.e-8)

    # Check that running via the corr2 script works correctly.
    config = treecorr.config.read_config('configs/nq_direct_spherical.yaml')
    try:
        import fitsio
    except ImportError:
        pass
    else:
        cat1.write(config['file_name'])
        cat2.write(config['file_name2'])
        treecorr.corr2(config)
        data = fitsio.read(config['nq_file_name'])
        np.testing.assert_allclose(data['r_nom'], nq.rnom)
        np.testing.assert_allclose(data['npairs'], nq.npairs)
        np.testing.assert_allclose(data['weight'], nq.weight)
        np.testing.assert_allclose(data['qR'], nq.xi)
        np.testing.assert_allclose(data['qR_im'], nq.xi_im)

    # Repeat with binslop = 0
    # And don't do any top-level recursion so we actually test not going to the leaves.
    nq = treecorr.NQCorrelation(min_sep=min_sep, max_sep=max_sep, nbins=nbins,
                                sep_units='deg', bin_slop=0, max_top=0)
    nq.process(cat1, cat2)
    np.testing.assert_array_equal(nq.npairs, true_npairs)
    np.testing.assert_allclose(nq.weight, true_weight, rtol=1.e-6, atol=1.e-8)
    np.testing.assert_allclose(nq.xi, true_xi.real, atol=2.e-4)
    np.testing.assert_allclose(nq.xi_im, true_xi.imag, atol=2.e-4)

    # With angle_slop = 0, it goes back to being basically exact (to single precision).
    nq = treecorr.NQCorrelation(min_sep=min_sep, max_sep=max_sep, nbins=nbins,
                                sep_units='deg', bin_slop=0, angle_slop=0, max_top=0)
    nq.process(cat1, cat2)
    np.testing.assert_array_equal(nq.npairs, true_npairs)
    np.testing.assert_allclose(nq.weight, true_weight, rtol=1.e-6, atol=1.e-8)
    np.testing.assert_allclose(nq.xi, true_xi.real, rtol=1.e-6, atol=1.e-8)
    np.testing.assert_allclose(nq.xi_im, true_xi.imag, rtol=1.e-6, atol=1.e-8)


@timer
def test_single():
    # Use q_radial(r) = q0 exp(-r^2/2r0^2) around a single lens
    # i.e. q(r) = q0 exp(-r^2/2r0^2) (x+iy)^4/r^4

    nsource = 300000
    q0 = 0.05
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

    lens_cat = treecorr.Catalog(x=[0], y=[0], x_units='arcmin', y_units='arcmin')
    source_cat = treecorr.Catalog(x=x, y=y, q1=q1, q2=q2, x_units='arcmin', y_units='arcmin')
    nq = treecorr.NQCorrelation(bin_size=0.1, min_sep=1., max_sep=20., sep_units='arcmin',
                                verbose=1)
    nq.process(lens_cat, source_cat)

    # log(<R>) != <logR>, but it should be close:
    print('meanlogr - log(meanr) = ',nq.meanlogr - np.log(nq.meanr))
    np.testing.assert_allclose(nq.meanlogr, np.log(nq.meanr), atol=1.e-3)

    r = nq.meanr
    true_qr = q0 * np.exp(-0.5*r**2/r0**2)

    print('nq.xi = ',nq.xi)
    print('nq.xi_im = ',nq.xi_im)
    print('true_qrad = ',true_qr)
    print('ratio = ',nq.xi / true_qr)
    print('diff = ',nq.xi - true_qr)
    print('max diff = ',max(abs(nq.xi - true_qr)))
    np.testing.assert_allclose(nq.xi, true_qr, rtol=3.e-2)
    np.testing.assert_allclose(nq.xi_im, 0, atol=1.e-4)

    # Check that we get the same result using the corr2 function:
    lens_cat.write(os.path.join('data','nq_single_lens.dat'))
    source_cat.write(os.path.join('data','nq_single_source.dat'))
    config = treecorr.read_config('configs/nq_single.yaml')
    config['verbose'] = 0
    treecorr.corr2(config)
    corr2_output = np.genfromtxt(os.path.join('output','nq_single.out'), names=True,
                                 skip_header=1)
    print('nq.xi = ',nq.xi)
    print('from corr2 output = ',corr2_output['qR'])
    print('ratio = ',corr2_output['qR']/nq.xi)
    print('diff = ',corr2_output['qR']-nq.xi)
    print('xi_im from corr2 output = ',corr2_output['qR_im'])
    np.testing.assert_allclose(corr2_output['qR'], nq.xi, rtol=1.e-3)
    np.testing.assert_allclose(corr2_output['qR_im'], 0, atol=1.e-4)

    # Check that adding results with different coords or metric emits a warning.
    lens_cat2 = treecorr.Catalog(x=[0], y=[0], z=[0])
    source_cat2 = treecorr.Catalog(x=x, y=y, z=x, q1=q1, q2=q2)
    with CaptureLog() as cl:
        nq2 = treecorr.NQCorrelation(bin_size=0.1, min_sep=1., max_sep=20., logger=cl.logger)
        nq2.process_cross(lens_cat2, source_cat2)
        nq2 += nq
    assert "Detected a change in catalog coordinate systems" in cl.output

    with CaptureLog() as cl:
        nq3 = treecorr.NQCorrelation(bin_size=0.1, min_sep=1., max_sep=20., logger=cl.logger)
        nq3.process_cross(lens_cat2, source_cat2, metric='Rperp')
        nq3 += nq2
    assert "Detected a change in metric" in cl.output

    # There is special handling for single-row catalogs when using np.genfromtxt rather
    # than pandas.  So mock it up to make sure we test it.
    treecorr.Catalog._emitted_pandas_warning = False  # Reset this, in case already triggered.
    with mock.patch.dict(sys.modules, {'pandas':None}):
        with CaptureLog() as cl:
            treecorr.corr2(config, logger=cl.logger)
        assert "Unable to import pandas" in cl.output
    corr2_output = np.genfromtxt(os.path.join('output','nq_single.out'), names=True,
                                 skip_header=1)
    np.testing.assert_allclose(corr2_output['qR'], nq.xi, rtol=1.e-3)


@timer
def test_spherical():
    # This is the same profile we used for test_single, but put into spherical coords.
    # We do the spherical trig by hand using the obvious formulae, rather than the clever
    # optimizations that are used by the TreeCorr code, thus serving as a useful test of
    # the latter.

    nsource = 400000
    q0 = 0.05
    r0 = 10. * coord.degrees / coord.radians
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

    nq = treecorr.NQCorrelation(bin_size=0.1, min_sep=1., max_sep=20., sep_units='deg',
                                verbose=1)
    r1 = np.exp(nq.logr) * (coord.degrees / coord.radians)
    true_qr = q0 * np.exp(-0.5*r1**2/r0**2)

    # Test this around several central points
    if __name__ == '__main__':
        ra0_list = [ 0., 1., 1.3, 232., 0. ]
        dec0_list = [ 0., -0.3, 1.3, -1.4, np.pi/2.-1.e-6 ]
    else:
        ra0_list = [ 232 ]
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

        # Rotate vector relative to local west
        # v_sph = exp(i beta) * v
        # where beta = pi - (A+B) is the angle between north and "up" in the tangent plane.
        beta = np.pi - (A+B)
        beta[x>0] *= -1.
        cos4beta = np.cos(4*beta)
        sin4beta = np.sin(4*beta)
        q1_sph = q1 * cos4beta - q2 * sin4beta
        q2_sph = q2 * cos4beta + q1 * sin4beta

        lens_cat = treecorr.Catalog(ra=[ra0], dec=[dec0], ra_units='rad', dec_units='rad')
        source_cat = treecorr.Catalog(ra=ra, dec=dec, q1=q1_sph, q2=q2_sph,
                                      ra_units='rad', dec_units='rad')
        nq.process(lens_cat, source_cat)

        print('ra0, dec0 = ',ra0,dec0)
        print('nq.xi = ',nq.xi)
        print('true_qrad = ',true_qr)
        print('ratio = ',nq.xi / true_qr)
        print('diff = ',nq.xi - true_qr)
        print('max diff = ',max(abs(nq.xi - true_qr)))
        np.testing.assert_allclose(nq.xi, true_qr, rtol=0.15)

    # One more center that can be done very easily.  If the center is the north pole, then all
    # the radial vectors are pure (positive) q1.
    ra0 = 0
    dec0 = np.pi/2.
    ra = theta
    dec = np.pi/2. - 2.*np.arcsin(r/2.)

    lens_cat = treecorr.Catalog(ra=[ra0], dec=[dec0], ra_units='rad', dec_units='rad')
    source_cat = treecorr.Catalog(ra=ra, dec=dec, q1=qrad, q2=np.zeros_like(qrad),
                                  ra_units='rad', dec_units='rad')
    nq.process(lens_cat, source_cat)

    print('nq.xi = ',nq.xi)
    print('nq.xi_im = ',nq.xi_im)
    print('true_qrad = ',true_qr)
    print('ratio = ',nq.xi / true_qr)
    print('diff = ',nq.xi - true_qr)
    print('max diff = ',max(abs(nq.xi - true_qr)))
    np.testing.assert_allclose(nq.xi, true_qr, rtol=0.1)
    np.testing.assert_allclose(nq.xi_im, 0, atol=1.e-4)

    # Check that we get the same result using the corr2 function:
    lens_cat.write(os.path.join('data','nq_spherical_lens.dat'))
    source_cat.write(os.path.join('data','nq_spherical_source.dat'))
    config = treecorr.read_config('configs/nq_spherical.yaml')
    config['verbose'] = 0
    treecorr.corr2(config)
    corr2_output = np.genfromtxt(os.path.join('output','nq_spherical.out'), names=True,
                                 skip_header=1)
    print('nq.xi = ',nq.xi)
    print('from corr2 output = ',corr2_output['qR'])
    print('ratio = ',corr2_output['qR']/nq.xi)
    print('diff = ',corr2_output['qR']-nq.xi)
    np.testing.assert_allclose(corr2_output['qR'], nq.xi, rtol=1.e-3)

    print('xi_im from corr2 output = ',corr2_output['qR_im'])
    np.testing.assert_allclose(corr2_output['qR_im'], 0., atol=4.e-5)


@timer
def test_nq():
    # Use q_radial(r) = q0 exp(-r^2/2r0^2) around a bunch of foreground lenses.
    # i.e. q(r) = q0 exp(-r^2/2r0^2) (x+iy)^4/r^4

    nlens = 1000
    nsource = 100000
    q0 = 0.05
    r0 = 10.
    L = 100. * r0
    rng = np.random.RandomState(8675309)
    xl = (rng.random_sample(nlens)-0.5) * L
    yl = (rng.random_sample(nlens)-0.5) * L
    xs = (rng.random_sample(nsource)-0.5) * L
    ys = (rng.random_sample(nsource)-0.5) * L
    q1 = np.zeros( (nsource,) )
    q2 = np.zeros( (nsource,) )
    for x,y in zip(xl,yl):
        dx = xs-x
        dy = ys-y
        r2 = dx**2 + dy**2
        r = np.sqrt(r2)
        qrad = q0 * np.exp(-0.5*r2/r0**2)
        theta = np.arctan2(dy,dx)
        q1 += qrad * np.cos(4*theta)
        q2 += qrad * np.sin(4*theta)

    lens_cat = treecorr.Catalog(x=xl, y=yl, x_units='arcmin', y_units='arcmin')
    source_cat = treecorr.Catalog(x=xs, y=ys, q1=q1, q2=q2, x_units='arcmin', y_units='arcmin')
    nq = treecorr.NQCorrelation(bin_size=0.1, min_sep=1., max_sep=20., sep_units='arcmin',
                                verbose=1)
    nq.process(lens_cat, source_cat)

    # Using nbins=None rather than omitting nbins is equivalent.
    nq2 = treecorr.NQCorrelation(bin_size=0.1, min_sep=1., max_sep=20., nbins=None, sep_units='arcmin')
    nq2.process(lens_cat, source_cat, num_threads=1)
    nq.process(lens_cat, source_cat, num_threads=1)
    assert nq2 == nq

    r = nq.meanr
    true_qr = q0 * np.exp(-0.5*r**2/r0**2)

    print('nq.xi = ',nq.xi)
    print('nq.xi_im = ',nq.xi_im)
    print('true_qrad = ',true_qr)
    print('ratio = ',nq.xi / true_qr)
    print('diff = ',nq.xi - true_qr)
    print('max diff = ',max(abs(nq.xi - true_qr)))
    np.testing.assert_allclose(nq.xi, true_qr, rtol=0.1)
    np.testing.assert_allclose(nq.xi_im, 0, atol=6.e-3)

    nrand = nlens * 3
    xr = (rng.random_sample(nrand)-0.5) * L
    yr = (rng.random_sample(nrand)-0.5) * L
    rand_cat = treecorr.Catalog(x=xr, y=yr, x_units='arcmin', y_units='arcmin')
    rq = treecorr.NQCorrelation(bin_size=0.1, min_sep=1., max_sep=20., sep_units='arcmin',
                                verbose=1)
    rq.process(rand_cat, source_cat)
    print('rq.xi = ',rq.xi)
    xi, xi_im, varxi = nq.calculateXi(rq=rq)
    print('compensated xi = ',xi)
    print('compensated xi_im = ',xi_im)
    print('true_qrad = ',true_qr)
    print('ratio = ',xi / true_qr)
    print('diff = ',xi - true_qr)
    print('max diff = ',max(abs(xi - true_qr)))
    # It turns out this doesn't come out much better.  I think the imprecision is mostly just due
    # to the smallish number of lenses, not to edge effects
    np.testing.assert_allclose(xi, true_qr, rtol=0.1)
    np.testing.assert_allclose(xi_im, 0, atol=5.e-3)

    # Check that we get the same result using the corr2 function:
    config = treecorr.read_config('configs/nq.yaml')
    try:
        import fitsio
    except ImportError:
        pass
    else:
        lens_cat.write(os.path.join('data','nq_lens.fits'))
        source_cat.write(os.path.join('data','nq_source.fits'))
        rand_cat.write(os.path.join('data','nq_rand.fits'))
        config['verbose'] = 0
        config['precision'] = 8
        treecorr.corr2(config)
        corr2_output = np.genfromtxt(os.path.join('output','nq.out'), names=True, skip_header=1)
        print('nq.xi = ',nq.xi)
        print('xi = ',xi)
        print('from corr2 output = ',corr2_output['qR'])
        print('ratio = ',corr2_output['qR']/xi)
        print('diff = ',corr2_output['qR']-xi)
        np.testing.assert_allclose(corr2_output['qR'], xi)
        print('xi_im from corr2 output = ',corr2_output['qR_im'])
        np.testing.assert_allclose(corr2_output['qR_im'], xi_im)

        # In the corr2 context, you can turn off the compensated bit, even if there are randoms
        # (e.g. maybe you only want randoms for some nn calculation, but not nq.)
        config['nq_statistic'] = 'simple'
        treecorr.corr2(config)
        corr2_output = np.genfromtxt(os.path.join('output','nq.out'), names=True, skip_header=1)
        xi_simple, _, _ = nq.calculateXi()
        np.testing.assert_equal(xi_simple, nq.xi)
        np.testing.assert_allclose(corr2_output['qR'], xi_simple, rtol=1.e-3)

    # Check the fits write option
    try:
        import fitsio
    except ImportError:
        pass
    else:
        out_file_name1 = os.path.join('output','nq_out1.fits')
        nq.write(out_file_name1)
        data = fitsio.read(out_file_name1)
        np.testing.assert_almost_equal(data['r_nom'], np.exp(nq.logr))
        np.testing.assert_almost_equal(data['meanr'], nq.meanr)
        np.testing.assert_almost_equal(data['meanlogr'], nq.meanlogr)
        np.testing.assert_almost_equal(data['qR'], nq.xi)
        np.testing.assert_almost_equal(data['qR_im'], nq.xi_im)
        np.testing.assert_almost_equal(data['sigma'], np.sqrt(nq.varxi))
        np.testing.assert_almost_equal(data['weight'], nq.weight)
        np.testing.assert_almost_equal(data['npairs'], nq.npairs)

        out_file_name2 = os.path.join('output','nq_out2.fits')
        nq.write(out_file_name2, rq=rq)
        data = fitsio.read(out_file_name2)
        np.testing.assert_almost_equal(data['r_nom'], np.exp(nq.logr))
        np.testing.assert_almost_equal(data['meanr'], nq.meanr)
        np.testing.assert_almost_equal(data['meanlogr'], nq.meanlogr)
        np.testing.assert_almost_equal(data['qR'], xi)
        np.testing.assert_almost_equal(data['qR_im'], xi_im)
        np.testing.assert_almost_equal(data['sigma'], np.sqrt(varxi))
        np.testing.assert_almost_equal(data['weight'], nq.weight)
        np.testing.assert_almost_equal(data['npairs'], nq.npairs)

        # Check the read function
        nq2 = treecorr.NQCorrelation.from_file(out_file_name2)
        np.testing.assert_almost_equal(nq2.logr, nq.logr)
        np.testing.assert_almost_equal(nq2.meanr, nq.meanr)
        np.testing.assert_almost_equal(nq2.meanlogr, nq.meanlogr)
        np.testing.assert_almost_equal(nq2.xi, nq.xi)
        np.testing.assert_almost_equal(nq2.xi_im, nq.xi_im)
        np.testing.assert_almost_equal(nq2.varxi, nq.varxi)
        np.testing.assert_almost_equal(nq2.weight, nq.weight)
        np.testing.assert_almost_equal(nq2.npairs, nq.npairs)
        assert nq2.coords == nq.coords
        assert nq2.metric == nq.metric
        assert nq2.sep_units == nq.sep_units
        assert nq2.bin_type == nq.bin_type


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
    q0 = 0.05
    r0 = 10.
    L = 50. * r0
    rng = np.random.RandomState(8675309)
    xl = (rng.random_sample(nlens)-0.5) * L
    yl = (rng.random_sample(nlens)-0.5) * L
    xs = (rng.random_sample( (nsource,ncats) )-0.5) * L
    ys = (rng.random_sample( (nsource,ncats) )-0.5) * L
    q1 = np.zeros( (nsource,ncats) )
    q2 = np.zeros( (nsource,ncats) )
    w = rng.random_sample( (nsource,ncats) ) + 0.5
    for x,y in zip(xl,yl):
        dx = xs-x
        dy = ys-y
        r2 = dx**2 + dy**2
        r = np.sqrt(r2)
        qrad = q0 * np.exp(-0.5*r2/r0**2)
        theta = np.arctan2(dy,dx)
        q1 += qrad * np.cos(4*theta)
        q2 += qrad * np.sin(4*theta)

    lens_cat = treecorr.Catalog(x=xl, y=yl, x_units='arcmin', y_units='arcmin')
    source_cats = [ treecorr.Catalog(x=xs[:,k], y=ys[:,k], q1=q1[:,k], q2=q2[:,k], w=w[:,k],
                                     x_units='arcmin', y_units='arcmin') for k in range(ncats) ]
    full_source_cat = treecorr.Catalog(x=xs.flatten(), y=ys.flatten(), w=w.flatten(),
                                       q1=q1.flatten(), q2=q2.flatten(),
                                       x_units='arcmin', y_units='arcmin')

    t0 = time.time()
    for k in range(ncats):
        # These could each be done on different machines in a real world application.
        nq = treecorr.NQCorrelation(bin_size=0.1, min_sep=1., max_sep=25., sep_units='arcmin',
                                    verbose=1)
        # These should use process_cross, not process, since we don't want to call finalize.
        nq.process_cross(lens_cat, source_cats[k])
        nq.write(os.path.join('output','nq_piece_%d.fits'%k))

    pieces_nq = treecorr.NQCorrelation(bin_size=0.1, min_sep=1., max_sep=25., sep_units='arcmin')
    for k in range(ncats):
        nq = pieces_nq.copy()
        nq.read(os.path.join('output','nq_piece_%d.fits'%k))
        pieces_nq += nq
    varq = treecorr.calculateVarQ(source_cats)
    pieces_nq.finalize(varq)
    t1 = time.time()
    print('time for piece-wise processing (including I/O) = ',t1-t0)

    full_nq = treecorr.NQCorrelation(bin_size=0.1, min_sep=1., max_sep=25., sep_units='arcmin',
                                     verbose=1)
    full_nq.process(lens_cat, full_source_cat)
    t2 = time.time()
    print('time for full processing = ',t2-t1)

    print('max error in meanr = ',np.max(pieces_nq.meanr - full_nq.meanr),)
    print('    max meanr = ',np.max(full_nq.meanr))
    print('max error in meanlogr = ',np.max(pieces_nq.meanlogr - full_nq.meanlogr),)
    print('    max meanlogr = ',np.max(full_nq.meanlogr))
    print('max error in weight = ',np.max(pieces_nq.weight - full_nq.weight),)
    print('    max weight = ',np.max(full_nq.weight))
    print('max error in xi = ',np.max(pieces_nq.xi - full_nq.xi),)
    print('    max xi = ',np.max(full_nq.xi))
    print('max error in xi_im = ',np.max(pieces_nq.xi_im - full_nq.xi_im),)
    print('    max xi_im = ',np.max(full_nq.xi_im))
    print('max error in varxi = ',np.max(pieces_nq.varxi - full_nq.varxi),)
    print('    max varxi = ',np.max(full_nq.varxi))
    np.testing.assert_allclose(pieces_nq.meanr, full_nq.meanr, rtol=2.e-3)
    np.testing.assert_allclose(pieces_nq.meanlogr, full_nq.meanlogr, atol=2.e-3)
    np.testing.assert_allclose(pieces_nq.weight, full_nq.weight, rtol=3.e-2)
    np.testing.assert_allclose(pieces_nq.xi, full_nq.xi, rtol=0.1)
    np.testing.assert_allclose(pieces_nq.xi_im, full_nq.xi_im, atol=2.e-3)
    np.testing.assert_allclose(pieces_nq.varxi, full_nq.varxi, rtol=3.e-2)

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
                                      q1=q1.flatten(), q2=q2.flatten(),
                                      wpos=w.flatten(), w=w2[k].flatten(),
                                      x_units='arcmin', y_units='arcmin') for k in range(ncats) ]

    t3 = time.time()
    nq2 = [ full_nq.copy() for k in range(ncats) ]
    for k in range(ncats):
        nq2[k].clear()
        nq2[k].process_cross(lens_cat, source_cats2[k])

    pieces_nq2 = full_nq.copy()
    pieces_nq2.clear()
    for k in range(ncats):
        pieces_nq2 += nq2[k]
    pieces_nq2.finalize(varq)
    t4 = time.time()
    print('time for zero-weight piece-wise processing = ',t4-t3)

    print('max error in meanr = ',np.max(pieces_nq2.meanr - full_nq.meanr),)
    print('    max meanr = ',np.max(full_nq.meanr))
    print('max error in meanlogr = ',np.max(pieces_nq2.meanlogr - full_nq.meanlogr),)
    print('    max meanlogr = ',np.max(full_nq.meanlogr))
    print('max error in weight = ',np.max(pieces_nq2.weight - full_nq.weight),)
    print('    max weight = ',np.max(full_nq.weight))
    print('max error in xi = ',np.max(pieces_nq2.xi - full_nq.xi),)
    print('    max xi = ',np.max(full_nq.xi))
    print('max error in xi_im = ',np.max(pieces_nq2.xi_im - full_nq.xi_im),)
    print('    max xi_im = ',np.max(full_nq.xi_im))
    print('max error in varxi = ',np.max(pieces_nq2.varxi - full_nq.varxi),)
    print('    max varxi = ',np.max(full_nq.varxi))
    np.testing.assert_allclose(pieces_nq2.meanr, full_nq.meanr, rtol=1.e-7)
    np.testing.assert_allclose(pieces_nq2.meanlogr, full_nq.meanlogr, rtol=1.e-7)
    np.testing.assert_allclose(pieces_nq2.weight, full_nq.weight, rtol=1.e-7)
    np.testing.assert_allclose(pieces_nq2.xi, full_nq.xi, rtol=1.e-7)
    np.testing.assert_allclose(pieces_nq2.xi_im, full_nq.xi_im, atol=1.e-10)
    np.testing.assert_allclose(pieces_nq2.varxi, full_nq.varxi, rtol=1.e-7)

    # Can also do this with initialize/finalize options
    pieces_nq3 = full_nq.copy()
    t3 = time.time()
    for k in range(ncats):
        pieces_nq3.process(lens_cat, source_cats2[k], initialize=(k==0), finalize=(k==ncats-1))
    t4 = time.time()
    print('time for initialize/finalize processing = ',t4-t3)

    np.testing.assert_allclose(pieces_nq3.meanr, full_nq.meanr, rtol=1.e-7)
    np.testing.assert_allclose(pieces_nq3.meanlogr, full_nq.meanlogr, rtol=1.e-7)
    np.testing.assert_allclose(pieces_nq3.weight, full_nq.weight, rtol=1.e-7)
    np.testing.assert_allclose(pieces_nq3.xi, full_nq.xi, rtol=1.e-7)
    np.testing.assert_allclose(pieces_nq3.xi_im, full_nq.xi_im, atol=1.e-10)
    np.testing.assert_allclose(pieces_nq3.varxi, full_nq.varxi, rtol=1.e-7)

    # Try this with corr2
    lens_cat.write(os.path.join('data','nq_wpos_lens.fits'))
    for i, sc in enumerate(source_cats2):
        sc.write(os.path.join('data','nq_wpos_source%d.fits'%i))
    config = treecorr.read_config('configs/nq_wpos.yaml')
    config['verbose'] = 0
    treecorr.corr2(config)
    data = fitsio.read(config['nq_file_name'])
    print('data.dtype = ',data.dtype)
    np.testing.assert_allclose(data['meanr'], pieces_nq3.meanr)
    np.testing.assert_allclose(data['meanlogr'], pieces_nq3.meanlogr)
    np.testing.assert_allclose(data['weight'], pieces_nq3.weight)
    np.testing.assert_allclose(data['qR'], pieces_nq3.xi)
    np.testing.assert_allclose(data['qR_im'], pieces_nq3.xi_im)
    np.testing.assert_allclose(data['sigma']**2, pieces_nq3.varxi)


@timer
def test_varxi():
    # Test that varxi is correct (or close) based on actual variance of many runs.

    # Signal doesn't matter much.  Use the one from test_gg.
    q0 = 0.05
    r0 = 10.
    L = 10 * r0
    rng = np.random.RandomState(8675309)

    nsource = 1000
    nrand = 10
    nruns = 50000
    lens = treecorr.Catalog(x=[0], y=[0])

    file_name = 'data/test_varxi_nq.npz'
    print(file_name)
    if not os.path.isfile(file_name):
        all_nqs = []
        all_rqs = []
        for run in range(nruns):
            print(f'{run}/{nruns}')
            x2 = (rng.random_sample(nsource)-0.5) * L
            y2 = (rng.random_sample(nsource)-0.5) * L
            x3 = (rng.random_sample(nrand)-0.5) * L
            y3 = (rng.random_sample(nrand)-0.5) * L

            r2 = (x2**2 + y2**2)/r0**2
            theta = np.arctan2(y2,x2)
            q1 = q0 * np.exp(-r2/2.) * np.cos(4*theta)
            q2 = q0 * np.exp(-r2/2.) * np.sin(4*theta)
            # This time, add some shape noise (different each run).
            q1 += rng.normal(0, 0.1, size=nsource)
            q2 += rng.normal(0, 0.1, size=nsource)
            # Varied weights are hard, but at least check that non-unit weights work correctly.
            w = np.ones_like(x2) * 5

            source = treecorr.Catalog(x=x2, y=y2, w=w, q1=q1, q2=q2)
            rand = treecorr.Catalog(x=x3, y=y3)
            nq = treecorr.NQCorrelation(bin_size=0.2, min_sep=10., max_sep=30.)
            rq = treecorr.NQCorrelation(bin_size=0.2, min_sep=10., max_sep=30.)
            nq.process(lens, source)
            rq.process(rand, source)
            all_nqs.append(nq)
            all_rqs.append(rq)

        all_xis = [nq.calculateXi() for nq in all_nqs]
        var_xi_1 = np.var([xi[0] for xi in all_xis], axis=0)
        mean_varxi_1 = np.mean([xi[2] for xi in all_xis], axis=0)

        all_xis = [nq.calculateXi(rq=rq) for (nq,rq) in zip(all_nqs, all_rqs)]
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
    theta = np.arctan2(y2,x2)
    q1 = q0 * np.exp(-r2/2.) * np.cos(4*theta)
    q2 = q0 * np.exp(-r2/2.) * np.sin(4*theta)
    q1 += rng.normal(0, 0.1, size=nsource)
    q2 += rng.normal(0, 0.1, size=nsource)
    # Varied weights are hard, but at least check that non-unit weights work correctly.
    w = np.ones_like(x2) * 5

    source = treecorr.Catalog(x=x2, y=y2, w=w, q1=q1, q2=q2)
    rand = treecorr.Catalog(x=x3, y=y3)
    nq = treecorr.NQCorrelation(bin_size=0.2, min_sep=10., max_sep=30.)
    rq = treecorr.NQCorrelation(bin_size=0.2, min_sep=10., max_sep=30.)
    nq.process(lens, source)
    rq.process(rand, source)

    print('single run:')
    print('Uncompensated')
    print('ratio = ',nq.varxi / var_xi_1)
    print('max relerr for xi = ',np.max(np.abs((nq.varxi - var_xi_1)/var_xi_1)))
    np.testing.assert_allclose(nq.varxi, var_xi_1, rtol=0.6)

    xi, xi_im, varxi = nq.calculateXi(rq=rq)
    print('Compensated')
    print('ratio = ',varxi / var_xi_2)
    print('max relerr for xi = ',np.max(np.abs((varxi - var_xi_2)/var_xi_2)))
    np.testing.assert_allclose(varxi, var_xi_2, rtol=0.5)

@timer
def test_jk():

    # Similar to the profile we use above, but multiple "lenses".
    q0 = 0.05
    r0 = 30.
    L = 30 * r0
    rng = np.random.RandomState(8675309)

    nsource = 100000
    nrand = 1000
    nlens = 300
    nruns = 1000
    npatch = 32

    corr_params = dict(bin_size=0.3, min_sep=10, max_sep=50, bin_slop=0.1)

    def make_spin4_field(rng):
        x1 = (rng.random(nlens)-0.5) * L
        y1 = (rng.random(nlens)-0.5) * L
        w = rng.random(nlens) + 10
        x2 = (rng.random(nsource)-0.5) * L
        y2 = (rng.random(nsource)-0.5) * L
        x3 = (rng.random(nrand)-0.5) * L
        y3 = (rng.random(nrand)-0.5) * L

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
            q1 += w[i] * q0 * np.exp(-r2/2.) * np.cos(4*theta)
            q2 += w[i] * q0 * np.exp(-r2/2.) * np.sin(4*theta)
        return x1, y1, w, x2, y2, q1, q2, x3, y3

    file_name = 'data/test_nq_jk_{}.npz'.format(nruns)
    print(file_name)
    if not os.path.isfile(file_name):
        all_nqs = []
        all_rqs = []
        rng = np.random.default_rng()
        for run in range(nruns):
            x1, y1, w, x2, y2, q1, q2, x3, y3 = make_spin4_field(rng)
            print(run,': ',np.mean(q1),np.std(q1),np.min(q1),np.max(q1))
            cat1 = treecorr.Catalog(x=x1, y=y1, w=w)
            cat2 = treecorr.Catalog(x=x2, y=y2, q1=q1, q2=q2)
            cat3 = treecorr.Catalog(x=x3, y=y3)
            nq = treecorr.NQCorrelation(corr_params)
            rq = treecorr.NQCorrelation(corr_params)
            nq.process(cat1, cat2)
            rq.process(cat3, cat2)
            all_nqs.append(nq)
            all_rqs.append(rq)

        mean_xi = np.mean([nq.xi for nq in all_nqs], axis=0)
        var_xi = np.var([nq.xi for nq in all_nqs], axis=0)
        mean_varxi = np.mean([nq.varxi for nq in all_nqs], axis=0)

        for nq, rq in zip(all_nqs, all_rqs):
            nq.calculateXi(rq=rq)

        mean_xi_r = np.mean([nq.xi for nq in all_nqs], axis=0)
        var_xi_r = np.var([nq.xi for nq in all_nqs], axis=0)
        mean_varxi_r = np.mean([nq.varxi for nq in all_nqs], axis=0)

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
    x1, y1, w, x2, y2, q1, q2, x3, y3 = make_spin4_field(rng)

    cat1 = treecorr.Catalog(x=x1, y=y1, w=w)
    cat2 = treecorr.Catalog(x=x2, y=y2, q1=q1, q2=q2)
    nq1 = treecorr.NQCorrelation(corr_params)
    t0 = time.time()
    nq1.process(cat1, cat2)
    t1 = time.time()
    print('Time for non-patch processing = ',t1-t0)

    print('weight = ',nq1.weight)
    print('xi = ',nq1.xi)
    print('varxi = ',nq1.varxi)
    print('pullsq for xi = ',(nq1.xi-mean_xi)**2/var_xi)
    print('max pull for xi = ',np.sqrt(np.max((nq1.xi-mean_xi)**2/var_xi)))
    np.testing.assert_array_less((nq1.xi-mean_xi)**2, 9*var_xi)  # < 3 sigma pull
    np.testing.assert_allclose(nq1.varxi, mean_varxi, rtol=0.1)

    # Now run with patches, but still with shot variance.  Should be basically the same answer.
    try:
        import fitsio
        patch_dir = 'output'
        low_mem = True
    except ImportError:
        # If we cannot write to a fits file, skip the save_patch_dir tests.
        patch_dir = None
        low_mem = False
    cat2p = treecorr.Catalog(x=x2, y=y2, q1=q1, q2=q2, npatch=npatch, save_patch_dir=patch_dir)
    if low_mem:
        cat2p.write_patches()  # Force rewrite of any existing saved patches.
    cat1p = treecorr.Catalog(x=x1, y=y1, w=w, patch_centers=cat2p.patch_centers)
    print('tot w = ',np.sum(w))
    print('Patch\tNlens\tNsource')
    for i in range(npatch):
        print('%d\t%d\t%d'%(i,np.sum(cat1p.w[cat1p.patch==i]),np.sum(cat2p.w[cat2p.patch==i])))
    nq2 = treecorr.NQCorrelation(corr_params)
    t0 = time.time()
    nq2.process(cat1p, cat2p)
    t1 = time.time()
    print('Time for patch processing = ',t1-t0)
    print('weight = ',nq2.weight)
    print('xi = ',nq2.xi)
    print('xi1 = ',nq1.xi)
    print('varxi = ',nq2.varxi)
    np.testing.assert_allclose(nq2.weight, nq1.weight, rtol=1.e-2)
    np.testing.assert_allclose(nq2.xi, nq1.xi, rtol=1.e-2)
    np.testing.assert_allclose(nq2.varxi, nq1.varxi, rtol=1.e-2)

    # estimate_cov with var_method='shot' returns just the diagonal.
    np.testing.assert_allclose(nq2.estimate_cov('shot'), nq2.varxi)
    np.testing.assert_allclose(nq1.estimate_cov('shot'), nq1.varxi)

    # Now try jackknife variance estimate.
    t0 = time.time()
    cov2 = nq2.estimate_cov('jackknife')
    t1 = time.time()
    print('Time to calculate jackknife covariance = ',t1-t0)
    print('varxi = ',np.diagonal(cov2))
    print('cf var_xi = ',var_xi)
    np.testing.assert_allclose(np.diagonal(cov2), var_xi, rtol=0.5)

    # Check only using patches for one of the two catalogs.
    # Not as good as using patches for both, but not much worse.
    nq3 = treecorr.NQCorrelation(corr_params, var_method='jackknife')
    t0 = time.time()
    nq3.process(cat1p, cat2)
    t1 = time.time()
    print('Time for only patches for cat1 processing = ',t1-t0)
    print('varxi = ',nq3.varxi)
    np.testing.assert_allclose(nq3.weight, nq1.weight, rtol=1.e-2)
    np.testing.assert_allclose(nq3.xi, nq1.xi, rtol=1.e-2)
    np.testing.assert_allclose(nq3.varxi, var_xi, rtol=0.6)

    nq4 = treecorr.NQCorrelation(corr_params, var_method='jackknife', rng=rng)
    t0 = time.time()
    nq4.process(cat1, cat2p)
    t1 = time.time()
    print('Time for only patches for cat2 processing = ',t1-t0)
    print('varxi = ',nq4.varxi)
    np.testing.assert_allclose(nq4.weight, nq1.weight, rtol=1.e-2)
    np.testing.assert_allclose(nq4.xi, nq1.xi, rtol=1.e-2)
    np.testing.assert_allclose(nq4.varxi, var_xi, rtol=0.6)

    # Use initialize/finalize
    nq5 = treecorr.NQCorrelation(corr_params)
    for k1, p1 in enumerate(cat1p.get_patches()):
        for k2, p2 in enumerate(cat2p.get_patches()):
            nq5.process(p1, p2, initialize=(k1==k2==0), finalize=(k1==k2==npatch-1))
    np.testing.assert_allclose(nq5.xi, nq2.xi)
    np.testing.assert_allclose(nq5.weight, nq2.weight)
    np.testing.assert_allclose(nq5.varxi, nq2.varxi)

    # Check that these still work after roundtripping through a file.
    try:
        import fitsio
    except ImportError:
        pass
    else:
        file_name = os.path.join('output','test_write_results_nq.fits')
        nq2.write(file_name, write_patch_results=True)
        nq5 = treecorr.NQCorrelation.from_file(file_name)
        cov5 = nq5.estimate_cov('jackknife')
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
    rq5 = treecorr.NQCorrelation(corr_params)
    rq5.process(cat3, cat2)
    nq5 = nq1.copy()
    nq5.calculateXi(rq=rq5)
    print('weight = ',nq5.weight)
    print('xi = ',nq5.xi)
    print('varxi = ',nq5.varxi)
    print('ratio = ',nq5.varxi / var_xi_r)
    print('pullsq for xi = ',(nq5.xi-mean_xi_r)**2/var_xi_r)
    print('max pull for xi = ',np.sqrt(np.max((nq5.xi-mean_xi_r)**2/var_xi_r)))
    np.testing.assert_array_less((nq5.xi-mean_xi_r)**2, 9*var_xi_r)  # < 3 sigma pull
    np.testing.assert_allclose(nq5.varxi, mean_varxi_r, rtol=0.1)

    # Repeat with patches
    cat3p = treecorr.Catalog(x=x3, y=y3, patch_centers=cat2p.patch_centers)
    rq6 = treecorr.NQCorrelation(corr_params)
    rq6.process(cat3p, cat2p, low_mem=low_mem)
    nq6 = nq2.copy()
    nq6.calculateXi(rq=rq6)
    cov6 = nq6.estimate_cov('jackknife')
    np.testing.assert_allclose(np.diagonal(cov6), var_xi_r, rtol=0.6)

    # Use a random catalog without patches.
    rq7 = treecorr.NQCorrelation(corr_params)
    rq7.process(cat3, cat2p)
    nq7 = nq4.copy()
    nq7.calculateXi(rq=rq7)
    cov7 = nq7.estimate_cov('jackknife')
    np.testing.assert_allclose(np.diagonal(cov7), var_xi_r, rtol=0.7)

    nq8 = nq2.copy()
    nq8.calculateXi(rq=rq7)
    cov8 = nq8.estimate_cov('jackknife')
    np.testing.assert_allclose(np.diagonal(cov8), var_xi_r, rtol=0.6)

    # Check some invalid actions
    # Bad var_method
    with assert_raises(ValueError):
        nq2.estimate_cov('invalid')
    # Not run on patches, but need patches
    with assert_raises(ValueError):
        nq1.estimate_cov('jackknife')
    with assert_raises(ValueError):
        nq1.estimate_cov('sample')
    with assert_raises(ValueError):
        nq1.estimate_cov('marked_bootstrap')
    with assert_raises(ValueError):
        nq1.estimate_cov('bootstrap')
    # rq also needs patches (at least for the g part).
    with assert_raises(RuntimeError):
        nq2.calculateXi(rq=nq1)

    cat1a = treecorr.Catalog(x=x1[:100], y=y1[:100], npatch=10)
    cat2a = treecorr.Catalog(x=x2[:100], y=y2[:100], q1=q1[:100], q2=q2[:100], npatch=10)
    cat1b = treecorr.Catalog(x=x1[:100], y=y1[:100], npatch=2)
    cat2b = treecorr.Catalog(x=x2[:100], y=y2[:100], q1=q1[:100], q2=q2[:100], npatch=2)
    nq6 = treecorr.NQCorrelation(corr_params)
    nq7 = treecorr.NQCorrelation(corr_params)
    # All catalogs need to have the same number of patches
    with assert_raises(RuntimeError):
        nq6.process(cat1a,cat2b)
    with assert_raises(RuntimeError):
        nq7.process(cat1b,cat2a)

@timer
def test_matrix_r():
    """Test using Q to model a 2x2 matrix for R, a la BFD shear estimation.
    """
    # The Bayesian Fourier Domain method for shear estimation involves expanding out
    # the first few terms of a Taylor approximation of the likelihoood.
    # The net result is that the expectation for the mean shear of an ensemble of galaxies is:
    #
    # <g> = (Sum_k R_k)^-1 Sum_k Q_k
    #
    # where Q_k is a 2x1 vector, and R_k is a 2x2 matrix.  (N.B. Q here is a spin-2 quantity,
    # not spin-4 like the TreeCorr Q notation.)
    #
    # The tricky thing about this is when doing projections for e.g. galaxy-galaxy lensing.
    # Then Q_k rotates as normal for a spin-2 field.  And R_k also needs to rotate correspondingly.
    #
    # For any particular galaxy with terms Q_k and R_k, when rotating the coordinate system by
    # an angle theta, we need
    #
    # [ g_1' ] = [ cos(2theta)  -sin(2theta) ] [ g_1 ]
    # [ g_2' ]   [ sin(2theta)   cos(2theta) ] [ g_2 ]
    #
    # For compactness of notation in what folows, use C = cos(2theta) and S = sin(2theta).
    #
    # [ g_1' ] = [ C -S ] [ g_1 ]
    # [ g_2' ] = [ S  C ] [ g_2 ]
    #          = [ C -S ] [ R_11 R_12 ]^-1 [ Q_1 ]
    #            [ S  C ] [ R_21 R_22 ]    [ Q_2 ]
    #          = [ C -S ] [ R_11 R_12 ]^-1 [  C S ] [ C -S ] [ Q_1 ]
    #            [ S  C ] [ R_21 R_22 ]    [ -S C ] [ S  C ] [ Q_2 ]
    #          = ( [ C -S ] [ R_11 R_12 ] [  C S ] )^-1 ( [ C -S ] [ Q_1 ] )
    #            ( [ S  C ] [ R_21 R_22 ] [ -S C ] )    ( [ S  C ] [ Q_2 ] )
    #
    # In other words, the Q values transform in the normal way that we normally transform the
    # shear values.  And R transforms with a rotation matrix on both sides.
    # Also, in the BFD context, R is symmetric, so R_12 = R_21, but we'll ignore that fact
    # and continue with the general case of any 2x2 R matrix, in case there are related
    # applications where it is not true.
    #
    # Let's try to turn all this matrix math into complex numbers.
    # To this end, we redefine the R matrix as:
    # R = [  r1+q1  r2+q2 ]
    #     [ -r2+q2  r1-q1 ]
    #
    # And we'll consider the complex numbers r = r1 + i r2 and q = q1 + i q2.
    #
    # This implies r1 = (R_11 + R_22)/2
    #              r2 = (R_12 - R_21)/2
    #              q1 = (R_11 - R_22)/2
    #              q2 = (R_12 + R_21)/2
    # R^-1 = (|r|^2-|q|^2)^-1 [ r1-q1  -r2-q2 ]
    #                         [ r2-q2   r1+q1 ]
    #
    # g = R^-1 Q = (|r|^2-|q|^2)^-1 [ r1-q1  -r2-q2 ] [ Q1 ]
    #                               [ r2-q2   r1+q1 ] [ Q2 ]
    #   = (|r|^2-|q|^2)^-1 [ r1Q1 - q1Q1 - r2Q2 - q2Q2 ]
    #                      [ r2Q1 - q2Q1 + r1Q2 + q1Q2 ]
    #
    # If we treat [Q1 Q2] as a complex number Q = Q1 + i Q2, then this becomes:
    #
    # g = (|r|^2-|q|^2)^-1 [ Re( rQ - qQ* ) ]
    #                      [ Im( rQ - qQ* ) ]
    #
    # which implies that the complex value g can be written as
    #
    # g = (rQ - qQ*) / (|r|^2-|q|^2)
    #
    # Now we just need to figure out how r and q transform under coordinate rotations.
    #
    # R' = [ C -S ] [  r1+q1  r2+q2 ] [  C S ]
    #      [ S  C ] [ -r2+q2  r1-q1 ] [ -S C ]
    #    = [ C -S ] [  (r1+q1)C - (r2+q2)S    (r1+q1)S + (r2+q2) C ]
    #      [ S  C ] [ -(r2-q2)C - (r1-q1)S   -(r2-q2)S + (r1-q1) C ]
    #    = [ (r1+q1)C^2 - (r2+q2)CS + (r2-q2)CS + (r1-q1)S^2
    #                                       (r1+q1)CS + (r2+q2)C^2 + (r2-q2) S^2 - (r1-q1) CS ]
    #      [ (r1+q1)CS - (r2+q2)S^2 - (r2-q2)C^2 - (r1-q1)CS
    #                                       (r1+q1)S^2 + (r2+q2)CS - (r2-q2) CS + (r1-q1) C^2 ]
    #    = [  r1 + q1 (C^2-S^2) - q2 (2CS)      r2 + q1 (2CS) + q2 (C^2-S^2) ]
    #    = [ -r2 + q1 (2CS) + q2 (C^2-S^2)      r1 - q1 (C^2-S^2) + q2 (2CS) ]
    #
    # This implies that r and q transform as
    #
    # r' = r
    # q' = q (cos(4 theta) + i sin(4 theta)) = q exp(4 i theta)
    #
    # I.e. r is a spin-0 quantity, and q is a spin-4 quantity.
    # So we can compute the properly rotated Sum_k R_k by converting the R matrices into
    # r and q complex numbers and computing NZ and NQ correlation functions of those.
    # This realization was in fact the impetus to add spin-4 correlations to TreeCorr.
    #
    # The following test confirms that this calculation is equivalent to doing the direct
    # transformations of the R matrix.

    nlens = 200
    nsource = 200
    s = 10.
    rng = np.random.default_rng(8675309)
    xlens = rng.normal(0,s, (nlens,) )
    ylens = rng.normal(0,s, (nlens,) )
    wlens = rng.random(nlens)

    x = rng.normal(0,s, (nsource,) )
    y = rng.normal(0,s, (nsource,) )
    w = rng.random(nlens)
    Q1 = rng.normal(0,0.2, (nsource,) )
    Q2 = rng.normal(0,0.2, (nsource,) )
    R11 = rng.random(nsource) + 1
    R22 = rng.random(nsource) + 1
    R12 = rng.random(nsource)
    R21 = rng.random(nsource)  # Again, BFD has R12=R21, but for this test we ignore that.

    min_sep = 1.
    max_sep = 50.
    nbins = 50
    bin_size = np.log(max_sep/min_sep) / nbins

    # First the more direct calculation using matrix R and vector Q.
    true_npairs = np.zeros(nbins, dtype=int)
    true_weight = np.zeros(nbins, dtype=float)
    true_Qt = np.zeros(nbins, dtype=float)
    true_Qx = np.zeros(nbins, dtype=float)
    true_Rtt = np.zeros(nbins, dtype=float)
    true_Rxx = np.zeros(nbins, dtype=float)
    true_Rtx = np.zeros(nbins, dtype=float)
    true_Rxt = np.zeros(nbins, dtype=float)

    for i in range(nlens):
        rsq = (x-xlens[i])**2 + (y-ylens[i])**2
        r = np.sqrt(rsq)
        theta = np.arctan2(y-ylens[i], x-xlens[i])
        c = np.cos(2*theta)
        s = np.sin(2*theta)

        rot = np.array([[c, -s], [s, c]])

        Q = np.array([Q1, Q2])
        R = np.array([[R11, R12], [R21, R22]])

        # In 1-d, this is:
        #   Qt, Qx = np.dot(rot.T, Q)
        #   (Rtt,Rtx), (Rxt,Rxx) = np.dot(rot.T, np.dot(R, rot))
        # With Q and R holding many values, this is easiest to do using einsum:
        Qt, Qx = np.einsum('jik,jk->ik', rot, [Q1,Q2])
        (Rtt, Rtx), (Rxt, Rxx) = np.einsum('jik,jlk->ilk', rot, np.einsum('ijk,jlk->ilk', R, rot))

        # Times -1 so Qt is tangential rather than radial.  Convention is to do the same to Qx.
        Qt *= -1
        Qx *= -1

        ww = wlens[i] * w
        index = np.floor(np.log(r/min_sep) / bin_size).astype(int)
        mask = (index >= 0) & (index < nbins)
        np.add.at(true_npairs, index[mask], 1)
        np.add.at(true_weight, index[mask], ww[mask])
        np.add.at(true_Qt, index[mask], ww[mask] * Qt[mask])
        np.add.at(true_Qx, index[mask], ww[mask] * Qx[mask])
        np.add.at(true_Rtt, index[mask], ww[mask] * Rtt[mask])
        np.add.at(true_Rtx, index[mask], ww[mask] * Rtx[mask])
        np.add.at(true_Rxt, index[mask], ww[mask] * Rxt[mask])
        np.add.at(true_Rxx, index[mask], ww[mask] * Rxx[mask])

    # Dividing by weight isn't required for BFD, since it will cancel in the eventual division.
    # Vut TreeCorr will do it, so we do too to make the comparisons easier.
    true_Qt /= true_weight
    true_Qx /= true_weight
    true_Rtt /= true_weight
    true_Rxx /= true_weight
    true_Rtx /= true_weight
    true_Rxt /= true_weight

    # Now finish the calculation by calculating g = R^-1 Q
    true_gt = np.zeros(nbins, dtype=float)
    true_gx = np.zeros(nbins, dtype=float)
    for k in range(nbins):
        R = np.array([[true_Rtt[k], true_Rtx[k]],
                      [true_Rxt[k], true_Rxx[k]]])
        Rinv = np.linalg.inv(R)
        Q = np.array([true_Qt[k], true_Qx[k]])
        g = Rinv.dot(Q)
        true_gt[k] = g[0]
        true_gx[k] = g[1]

    # Now use TreeCorr
    # cat1 = lenses
    cat1 = treecorr.Catalog(x=xlens, y=ylens, w=wlens)

    # Convert R matrix into r,q complex numbers.
    r = (R11 + R22)/2 + 1j * (R12 - R21)/2
    q = (R11 - R22)/2 + 1j * (R12 + R21)/2

    # cat2 = sources
    # Note: BFD can use k, rather than z1,z2, since r is real in that use case.
    # And would use NKCorrelation below rather than NZ.
    cat2 = treecorr.Catalog(x=x, y=y, w=w, g1=Q1, g2=Q2,
                            z1=np.real(r), z2=np.imag(r), q1=np.real(q), q2=np.imag(q))

    # Perform all the correlations
    ng = treecorr.NGCorrelation(min_sep=min_sep, max_sep=max_sep, nbins=nbins, brute=True)
    ng.process(cat1, cat2)
    nz = treecorr.NZCorrelation(min_sep=min_sep, max_sep=max_sep, nbins=nbins, brute=True)
    nz.process(cat1, cat2)
    nq = treecorr.NQCorrelation(min_sep=min_sep, max_sep=max_sep, nbins=nbins, brute=True)
    nq.process(cat1, cat2)

    # First check that the raw outputs match the matrix calculation.
    print('true_npairs = ',true_npairs)
    print('diff ng = ',ng.npairs - true_npairs)
    print('diff nz = ',nz.npairs - true_npairs)
    print('diff nq = ',nq.npairs - true_npairs)
    np.testing.assert_array_equal(ng.npairs, true_npairs)
    np.testing.assert_array_equal(nz.npairs, true_npairs)
    np.testing.assert_array_equal(nq.npairs, true_npairs)

    print('true_weight = ',true_weight)
    print('diff ng = ',ng.weight - true_weight)
    print('diff nz = ',nz.weight - true_weight)
    print('diff nq = ',nq.weight - true_weight)
    np.testing.assert_allclose(ng.weight, true_weight)
    np.testing.assert_allclose(nz.weight, true_weight)
    np.testing.assert_allclose(nq.weight, true_weight)

    print('true_Qt = ',true_Qt)
    print('ng.xi = ',ng.xi)
    np.testing.assert_allclose(ng.xi, true_Qt, atol=1.e-8)
    print('true_Qx = ',true_Qx)
    print('ng.xi_im = ',ng.xi_im)
    np.testing.assert_allclose(ng.xi_im, true_Qx, atol=1.e-8)

    print('true_Rtt = ',true_Rtt)
    print('nz.xi + nq.xi = ',nz.xi + nq.xi)
    np.testing.assert_allclose(nz.xi + nq.xi, true_Rtt, atol=1.e-8)
    print('true_Rtx = ',true_Rtx)
    np.testing.assert_allclose(nz.xi_im + nq.xi_im, true_Rtx, atol=1.e-8)
    print('true_Rxt = ',true_Rxt)
    print('-nz.xi_im + nq.xi_im = ',-nz.xi_im + nq.xi_im)
    np.testing.assert_allclose(-nz.xi_im + nq.xi_im, true_Rxt, atol=1.e-8)
    print('true_Rxx = ',true_Rxx)
    print('nz.xi - nq.xi = ',nz.xi - nq.xi)
    np.testing.assert_allclose(nz.xi - nq.xi, true_Rxx, atol=1.e-8)

    # Now finish the calculation using r,q.
    # g = (rQ - qQ*) / (|r|^2-|q|^2)
    r = nz.xi + 1j * nz.xi_im    # Again, for BFD, r = nz.xi, since nz_im is 0.
    q = nq.xi + 1j * nq.xi_im
    Q = ng.xi + 1j * ng.xi_im
    g = (r * Q - q * np.conj(Q)) / (np.abs(r)**2 - np.abs(q)**2)

    print('true_gt = ',true_gt)
    print('gt = ',np.real(g))
    print('diff = ',np.real(g) - true_gt)
    print('true_gx = ',true_gx)
    print('gx = ',np.imag(g))
    print('diff = ',np.imag(g) - true_gx)
    np.testing.assert_allclose(np.real(g), true_gt, atol=1.e-8)
    np.testing.assert_allclose(np.imag(g), true_gx, atol=1.e-8)


if __name__ == '__main__':
    test_direct()
    test_direct_spherical()
    test_single()
    test_spherical()
    test_nq()
    test_pieces()
    test_varxi()
    test_jk()
    test_matrix_r()
