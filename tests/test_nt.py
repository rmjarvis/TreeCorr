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
    t12 = rng.normal(0,0.2, (ngal,) )
    t22 = rng.normal(0,0.2, (ngal,) )

    cat1 = treecorr.Catalog(x=x1, y=y1, w=w1)
    cat2 = treecorr.Catalog(x=x2, y=y2, w=w2, t1=t12, t2=t22)

    min_sep = 1.
    max_sep = 50.
    nbins = 50
    bin_size = np.log(max_sep/min_sep) / nbins
    nt = treecorr.NTCorrelation(min_sep=min_sep, max_sep=max_sep, nbins=nbins, brute=True)
    nt.process(cat1, cat2)

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
        xi = ww * (t12 + 1j*t22) * expmialpha**3

        index = np.floor(np.log(r/min_sep) / bin_size).astype(int)
        mask = (index >= 0) & (index < nbins)
        np.add.at(true_npairs, index[mask], 1)
        np.add.at(true_weight, index[mask], ww[mask])
        np.add.at(true_xi, index[mask], xi[mask])

    true_xi /= true_weight

    print('true_npairs = ',true_npairs)
    print('diff = ',nt.npairs - true_npairs)
    np.testing.assert_array_equal(nt.npairs, true_npairs)

    print('true_weight = ',true_weight)
    print('diff = ',nt.weight - true_weight)
    np.testing.assert_allclose(nt.weight, true_weight, rtol=1.e-6, atol=1.e-8)

    print('true_xi = ',true_xi)
    print('nt.xi = ',nt.xi)
    print('nt.xi_im = ',nt.xi_im)
    np.testing.assert_allclose(nt.xi, true_xi.real, rtol=1.e-6, atol=1.e-8)
    np.testing.assert_allclose(nt.xi_im, true_xi.imag, rtol=1.e-6, atol=1.e-8)

    # Check that running via the corr2 script works correctly.
    config = treecorr.config.read_config('configs/nt_direct.yaml')
    try:
        import fitsio
    except ImportError:
        pass
    else:
        cat1.write(config['file_name'])
        cat2.write(config['file_name2'])
        with CaptureLog() as cl:
            treecorr.corr2(config, logger=cl.logger)
        assert "skipping t1_col" in cl.output
        data = fitsio.read(config['nt_file_name'])
        np.testing.assert_allclose(data['r_nom'], nt.rnom)
        np.testing.assert_allclose(data['npairs'], nt.npairs)
        np.testing.assert_allclose(data['weight'], nt.weight)
        np.testing.assert_allclose(data['tR'], nt.xi)
        np.testing.assert_allclose(data['tR_im'], nt.xi_im)

        # When not using corr2, it's invalid to specify invalid t1_col, t2_col
        with assert_raises(ValueError):
            cat = treecorr.Catalog(config['file_name'], config)

        # Invalid with only one file_name
        del config['file_name2']
        with assert_raises(TypeError):
            treecorr.corr2(config)
        config['file_name2'] = 'data/nt_direct_cat2.fits'
        # Invalid to request compoensated if no rand_file
        config['nt_statistic'] = 'compensated'
        with assert_raises(TypeError):
            treecorr.corr2(config)

    # Repeat with binslop = 0
    # And don't do any top-level recursion so we actually test not going to the leaves.
    nt = treecorr.NTCorrelation(min_sep=min_sep, max_sep=max_sep, nbins=nbins, bin_slop=0,
                                max_top=0)
    nt.process(cat1, cat2)
    np.testing.assert_array_equal(nt.npairs, true_npairs)
    np.testing.assert_allclose(nt.weight, true_weight, rtol=1.e-6, atol=1.e-8)
    np.testing.assert_allclose(nt.xi, true_xi.real, atol=4.e-4)
    np.testing.assert_allclose(nt.xi_im, true_xi.imag, atol=2.e-4)

    # With angle_slop = 0, it goes back to being basically exact (to single precision).
    nt = treecorr.NTCorrelation(min_sep=min_sep, max_sep=max_sep, nbins=nbins, bin_slop=0,
                                angle_slop=0, max_top=0)
    nt.process(cat1, cat2)
    np.testing.assert_array_equal(nt.npairs, true_npairs)
    np.testing.assert_allclose(nt.weight, true_weight, rtol=1.e-6, atol=1.e-8)
    np.testing.assert_allclose(nt.xi, true_xi.real, rtol=1.e-6, atol=1.e-8)
    np.testing.assert_allclose(nt.xi_im, true_xi.imag, rtol=1.e-6, atol=1.e-8)

    # Check a few basic operations with a NTCorrelation object.
    do_pickle(nt)

    nt2 = nt.copy()
    nt2 += nt
    np.testing.assert_allclose(nt2.npairs, 2*nt.npairs)
    np.testing.assert_allclose(nt2.weight, 2*nt.weight)
    np.testing.assert_allclose(nt2.meanr, 2*nt.meanr)
    np.testing.assert_allclose(nt2.meanlogr, 2*nt.meanlogr)
    np.testing.assert_allclose(nt2.xi, 2*nt.xi)
    np.testing.assert_allclose(nt2.xi_im, 2*nt.xi_im)

    nt2.clear()
    nt2 += nt
    np.testing.assert_allclose(nt2.npairs, nt.npairs)
    np.testing.assert_allclose(nt2.weight, nt.weight)
    np.testing.assert_allclose(nt2.meanr, nt.meanr)
    np.testing.assert_allclose(nt2.meanlogr, nt.meanlogr)
    np.testing.assert_allclose(nt2.xi, nt.xi)
    np.testing.assert_allclose(nt2.xi_im, nt.xi_im)

    ascii_name = 'output/nt_ascii.txt'
    nt.write(ascii_name, precision=16)
    nt3 = treecorr.NTCorrelation(min_sep=min_sep, max_sep=max_sep, nbins=nbins, bin_type='Log')
    nt3.read(ascii_name)
    np.testing.assert_allclose(nt3.npairs, nt.npairs)
    np.testing.assert_allclose(nt3.weight, nt.weight)
    np.testing.assert_allclose(nt3.meanr, nt.meanr)
    np.testing.assert_allclose(nt3.meanlogr, nt.meanlogr)
    np.testing.assert_allclose(nt3.xi, nt.xi)
    np.testing.assert_allclose(nt3.xi_im, nt.xi_im)

    # Check that the repr is minimal
    assert repr(nt3) == f'NTCorrelation(min_sep={min_sep}, max_sep={max_sep}, nbins={nbins})'

    # Simpler API using from_file:
    with CaptureLog() as cl:
        nt3b = treecorr.NTCorrelation.from_file(ascii_name, logger=cl.logger)
    assert ascii_name in cl.output
    np.testing.assert_allclose(nt3b.npairs, nt.npairs)
    np.testing.assert_allclose(nt3b.weight, nt.weight)
    np.testing.assert_allclose(nt3b.meanr, nt.meanr)
    np.testing.assert_allclose(nt3b.meanlogr, nt.meanlogr)
    np.testing.assert_allclose(nt3b.xi, nt.xi)
    np.testing.assert_allclose(nt3b.xi_im, nt.xi_im)

    # or using the Corr2 base class
    with CaptureLog() as cl:
        nt3c = treecorr.Corr2.from_file(ascii_name, logger=cl.logger)
    assert ascii_name in cl.output
    np.testing.assert_allclose(nt3c.npairs, nt.npairs)
    np.testing.assert_allclose(nt3c.weight, nt.weight)
    np.testing.assert_allclose(nt3c.meanr, nt.meanr)
    np.testing.assert_allclose(nt3c.meanlogr, nt.meanlogr)
    np.testing.assert_allclose(nt3c.xi, nt.xi)
    np.testing.assert_allclose(nt3c.xi_im, nt.xi_im)

    try:
        import fitsio
    except ImportError:
        pass
    else:
        fits_name = 'output/nt_fits.fits'
        nt.write(fits_name)
        nt4 = treecorr.NTCorrelation(min_sep=min_sep, max_sep=max_sep, nbins=nbins)
        nt4.read(fits_name)
        np.testing.assert_allclose(nt4.npairs, nt.npairs)
        np.testing.assert_allclose(nt4.weight, nt.weight)
        np.testing.assert_allclose(nt4.meanr, nt.meanr)
        np.testing.assert_allclose(nt4.meanlogr, nt.meanlogr)
        np.testing.assert_allclose(nt4.xi, nt.xi)
        np.testing.assert_allclose(nt4.xi_im, nt.xi_im)

        nt4b = treecorr.NTCorrelation.from_file(fits_name)
        np.testing.assert_allclose(nt4b.npairs, nt.npairs)
        np.testing.assert_allclose(nt4b.weight, nt.weight)
        np.testing.assert_allclose(nt4b.meanr, nt.meanr)
        np.testing.assert_allclose(nt4b.meanlogr, nt.meanlogr)
        np.testing.assert_allclose(nt4b.xi, nt.xi)
        np.testing.assert_allclose(nt4b.xi_im, nt.xi_im)

    with assert_raises(TypeError):
        nt2 += config
    nt4 = treecorr.NTCorrelation(min_sep=min_sep/2, max_sep=max_sep, nbins=nbins)
    with assert_raises(ValueError):
        nt2 += nt4
    nt5 = treecorr.NTCorrelation(min_sep=min_sep, max_sep=max_sep*2, nbins=nbins)
    with assert_raises(ValueError):
        nt2 += nt5
    nt6 = treecorr.NTCorrelation(min_sep=min_sep, max_sep=max_sep, nbins=nbins*2)
    with assert_raises(ValueError):
        nt2 += nt6
    with assert_raises(ValueError):
        nt.process(cat1, cat2, patch_method='nonlocal')



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
    t12 = rng.normal(0,0.2, (ngal,) )
    t22 = rng.normal(0,0.2, (ngal,) )

    ra1, dec1 = coord.CelestialCoord.xyz_to_radec(x1,y1,z1)
    ra2, dec2 = coord.CelestialCoord.xyz_to_radec(x2,y2,z2)

    cat1 = treecorr.Catalog(ra=ra1, dec=dec1, ra_units='rad', dec_units='rad', w=w1)
    cat2 = treecorr.Catalog(ra=ra2, dec=dec2, ra_units='rad', dec_units='rad', w=w2, t1=t12, t2=t22)

    min_sep = 1.
    max_sep = 10.
    nbins = 50
    bin_size = np.log(max_sep/min_sep) / nbins
    nt = treecorr.NTCorrelation(min_sep=min_sep, max_sep=max_sep, nbins=nbins,
                                sep_units='deg', brute=True)
    nt.process(cat1, cat2)

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

            # Rotate field to coordinates where line connecting is horizontal.
            # Original orientation is where north is up.
            theta2 = 90*coord.degrees + c2[j].angleBetween(c1[i], north_pole)
            exp3theta2 = np.cos(3*theta2) + 1j * np.sin(3*theta2)

            t2 = t12[j] + 1j * t22[j]
            t2 *= exp3theta2

            ww = w1[i] * w2[j]
            xi = w1[i] * w2[j] * t2

            true_npairs[index] += 1
            true_weight[index] += ww
            true_xi[index] += xi

    true_xi /= true_weight

    print('true_npairs = ',true_npairs)
    print('diff = ',nt.npairs - true_npairs)
    np.testing.assert_array_equal(nt.npairs, true_npairs)

    print('true_weight = ',true_weight)
    print('diff = ',nt.weight - true_weight)
    np.testing.assert_allclose(nt.weight, true_weight, rtol=1.e-6, atol=1.e-8)

    print('true_xi = ',true_xi)
    print('nt.xi = ',nt.xi)
    print('nt.xi_im = ',nt.xi_im)
    np.testing.assert_allclose(nt.xi, true_xi.real, rtol=1.e-6, atol=1.e-8)
    np.testing.assert_allclose(nt.xi_im, true_xi.imag, rtol=1.e-6, atol=1.e-8)

    # Check that running via the corr2 script works correctly.
    config = treecorr.config.read_config('configs/nt_direct_spherical.yaml')
    try:
        import fitsio
    except ImportError:
        pass
    else:
        cat1.write(config['file_name'])
        cat2.write(config['file_name2'])
        treecorr.corr2(config)
        data = fitsio.read(config['nt_file_name'])
        np.testing.assert_allclose(data['r_nom'], nt.rnom)
        np.testing.assert_allclose(data['npairs'], nt.npairs)
        np.testing.assert_allclose(data['weight'], nt.weight)
        np.testing.assert_allclose(data['tR'], nt.xi)
        np.testing.assert_allclose(data['tR_im'], nt.xi_im)

    # Repeat with binslop = 0
    # And don't do any top-level recursion so we actually test not going to the leaves.
    nt = treecorr.NTCorrelation(min_sep=min_sep, max_sep=max_sep, nbins=nbins,
                                sep_units='deg', bin_slop=0, max_top=0)
    nt.process(cat1, cat2)
    np.testing.assert_array_equal(nt.npairs, true_npairs)
    np.testing.assert_allclose(nt.weight, true_weight, rtol=1.e-6, atol=1.e-8)
    np.testing.assert_allclose(nt.xi, true_xi.real, atol=1.e-4)
    np.testing.assert_allclose(nt.xi_im, true_xi.imag, atol=1.e-4)

    # With angle_slop = 0, it goes back to being basically exact (to single precision).
    nt = treecorr.NTCorrelation(min_sep=min_sep, max_sep=max_sep, nbins=nbins,
                                sep_units='deg', bin_slop=0, angle_slop=0, max_top=0)
    nt.process(cat1, cat2)
    np.testing.assert_array_equal(nt.npairs, true_npairs)
    np.testing.assert_allclose(nt.weight, true_weight, rtol=1.e-6, atol=1.e-8)
    np.testing.assert_allclose(nt.xi, true_xi.real, rtol=1.e-6, atol=1.e-8)
    np.testing.assert_allclose(nt.xi_im, true_xi.imag, rtol=1.e-6, atol=1.e-8)


@timer
def test_single():
    # Use t_radial(r) = t0 exp(-r^2/2r0^2) around a single lens
    # i.e. t(r) = t0 exp(-r^2/2r0^2) (x+iy)^3/r^3

    nsource = 300000
    t0 = 0.05
    r0 = 10.
    L = 5. * r0
    rng = np.random.RandomState(8675309)
    x = (rng.random_sample(nsource)-0.5) * L
    y = (rng.random_sample(nsource)-0.5) * L
    r2 = (x**2 + y**2)
    r = np.sqrt(r2)
    trad = t0 * np.exp(-0.5*r2/r0**2)
    theta = np.arctan2(y,x)
    t1 = trad * np.cos(3*theta)
    t2 = trad * np.sin(3*theta)

    lens_cat = treecorr.Catalog(x=[0], y=[0], x_units='arcmin', y_units='arcmin')
    source_cat = treecorr.Catalog(x=x, y=y, t1=t1, t2=t2, x_units='arcmin', y_units='arcmin')
    nt = treecorr.NTCorrelation(bin_size=0.1, min_sep=1., max_sep=20., sep_units='arcmin',
                                verbose=1)
    nt.process(lens_cat, source_cat)

    # log(<R>) != <logR>, but it should be close:
    print('meanlogr - log(meanr) = ',nt.meanlogr - np.log(nt.meanr))
    np.testing.assert_allclose(nt.meanlogr, np.log(nt.meanr), atol=1.e-3)

    r = nt.meanr
    true_tr = t0 * np.exp(-0.5*r**2/r0**2)

    print('nt.xi = ',nt.xi)
    print('nt.xi_im = ',nt.xi_im)
    print('true_trad = ',true_tr)
    print('ratio = ',nt.xi / true_tr)
    print('diff = ',nt.xi - true_tr)
    print('max diff = ',max(abs(nt.xi - true_tr)))
    np.testing.assert_allclose(nt.xi, true_tr, rtol=3.e-2)
    np.testing.assert_allclose(nt.xi_im, 0, atol=1.e-4)

    # Check that we get the same result using the corr2 function:
    lens_cat.write(os.path.join('data','nt_single_lens.dat'))
    source_cat.write(os.path.join('data','nt_single_source.dat'))
    config = treecorr.read_config('configs/nt_single.yaml')
    config['verbose'] = 0
    treecorr.corr2(config)
    corr2_output = np.genfromtxt(os.path.join('output','nt_single.out'), names=True,
                                 skip_header=1)
    print('nt.xi = ',nt.xi)
    print('from corr2 output = ',corr2_output['tR'])
    print('ratio = ',corr2_output['tR']/nt.xi)
    print('diff = ',corr2_output['tR']-nt.xi)
    print('xi_im from corr2 output = ',corr2_output['tR_im'])
    np.testing.assert_allclose(corr2_output['tR'], nt.xi, rtol=1.e-3)
    np.testing.assert_allclose(corr2_output['tR_im'], 0, atol=1.e-4)

    # Check that adding results with different coords or metric emits a warning.
    lens_cat2 = treecorr.Catalog(x=[0], y=[0], z=[0])
    source_cat2 = treecorr.Catalog(x=x, y=y, z=x, t1=t1, t2=t2)
    with CaptureLog() as cl:
        nt2 = treecorr.NTCorrelation(bin_size=0.1, min_sep=1., max_sep=20., logger=cl.logger)
        nt2.process_cross(lens_cat2, source_cat2)
        nt2 += nt
    assert "Detected a change in catalog coordinate systems" in cl.output

    with CaptureLog() as cl:
        nt3 = treecorr.NTCorrelation(bin_size=0.1, min_sep=1., max_sep=20., logger=cl.logger)
        nt3.process_cross(lens_cat2, source_cat2, metric='Rperp')
        nt3 += nt2
    assert "Detected a change in metric" in cl.output

    # There is special handling for single-row catalogs when using np.genfromtxt rather
    # than pandas.  So mock it up to make sure we test it.
    treecorr.Catalog._emitted_pandas_warning = False  # Reset this, in case already triggered.
    with mock.patch.dict(sys.modules, {'pandas':None}):
        with CaptureLog() as cl:
            treecorr.corr2(config, logger=cl.logger)
        assert "Unable to import pandas" in cl.output
    corr2_output = np.genfromtxt(os.path.join('output','nt_single.out'), names=True,
                                 skip_header=1)
    np.testing.assert_allclose(corr2_output['tR'], nt.xi, rtol=1.e-3)


@timer
def test_spherical():
    # This is the same profile we used for test_single, but put into spherical coords.
    # We do the spherical trig by hand using the obvious formulae, rather than the clever
    # optimizations that are used by the TreeCorr code, thus serving as a useful test of
    # the latter.

    nsource = 400000
    t0 = 0.05
    r0 = 10. * coord.degrees / coord.radians
    L = 5. * r0
    rng = np.random.RandomState(8675309)
    x = (rng.random_sample(nsource)-0.5) * L
    y = (rng.random_sample(nsource)-0.5) * L
    r2 = (x**2 + y**2)
    r = np.sqrt(r2)
    trad = t0 * np.exp(-0.5*r2/r0**2)
    theta = np.arctan2(y,x)
    t1 = trad * np.cos(3*theta)
    t2 = trad * np.sin(3*theta)

    nt = treecorr.NTCorrelation(bin_size=0.1, min_sep=1., max_sep=20., sep_units='deg',
                                verbose=1)
    r1 = np.exp(nt.logr) * (coord.degrees / coord.radians)
    true_tr = t0 * np.exp(-0.5*r1**2/r0**2)

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

        # Rotate field relative to local west
        # t_sph = exp(i beta) * t
        # where beta = pi - (A+B) is the angle between north and "up" in the tangent plane.
        beta = np.pi - (A+B)
        beta[x>0] *= -1.
        cos3beta = np.cos(3*beta)
        sin3beta = np.sin(3*beta)
        t1_sph = t1 * cos3beta - t2 * sin3beta
        t2_sph = t2 * cos3beta + t1 * sin3beta

        lens_cat = treecorr.Catalog(ra=[ra0], dec=[dec0], ra_units='rad', dec_units='rad')
        source_cat = treecorr.Catalog(ra=ra, dec=dec, t1=t1_sph, t2=t2_sph,
                                      ra_units='rad', dec_units='rad')
        nt.process(lens_cat, source_cat)

        print('ra0, dec0 = ',ra0,dec0)
        print('nt.xi = ',nt.xi)
        print('true_trad = ',true_tr)
        print('ratio = ',nt.xi / true_tr)
        print('diff = ',nt.xi - true_tr)
        print('max diff = ',max(abs(nt.xi - true_tr)))
        # The 3rd and 4th centers are somewhat less accurate.  Not sure why.
        # The math seems to be right, since the last one that gets all the way to the pole
        # works, so I'm not sure what is going on.  It's just a few bins that get a bit less
        # accurate.  Possibly worth investigating further at some point...
        np.testing.assert_allclose(nt.xi, true_tr, rtol=0.1)

    # One more center that can be done very easily.  If the center is the north pole, then all
    # the radial vectors are pure (positive) t1.
    ra0 = 0
    dec0 = np.pi/2.
    ra = theta
    dec = np.pi/2. - 2.*np.arcsin(r/2.)

    lens_cat = treecorr.Catalog(ra=[ra0], dec=[dec0], ra_units='rad', dec_units='rad')
    source_cat = treecorr.Catalog(ra=ra, dec=dec, t1=np.zeros_like(trad), t2=trad,
                                  ra_units='rad', dec_units='rad')
    nt.process(lens_cat, source_cat)

    print('nt.xi = ',nt.xi)
    print('nt.xi_im = ',nt.xi_im)
    print('true_trad = ',true_tr)
    print('ratio = ',nt.xi / true_tr)
    print('diff = ',nt.xi - true_tr)
    print('max diff = ',max(abs(nt.xi - true_tr)))
    np.testing.assert_allclose(nt.xi, true_tr, rtol=0.1)
    np.testing.assert_allclose(nt.xi_im, 0, atol=1.e-4)

    # Check that we get the same result using the corr2 function:
    lens_cat.write(os.path.join('data','nt_spherical_lens.dat'))
    source_cat.write(os.path.join('data','nt_spherical_source.dat'))
    config = treecorr.read_config('configs/nt_spherical.yaml')
    config['verbose'] = 0
    treecorr.corr2(config)
    corr2_output = np.genfromtxt(os.path.join('output','nt_spherical.out'), names=True,
                                 skip_header=1)
    print('nt.xi = ',nt.xi)
    print('from corr2 output = ',corr2_output['tR'])
    print('ratio = ',corr2_output['tR']/nt.xi)
    print('diff = ',corr2_output['tR']-nt.xi)
    np.testing.assert_allclose(corr2_output['tR'], nt.xi, rtol=1.e-3)

    print('xi_im from corr2 output = ',corr2_output['tR_im'])
    np.testing.assert_allclose(corr2_output['tR_im'], 0., atol=3.e-5)


@timer
def test_nt():
    # Use t_radial(r) = t0 exp(-r^2/2r0^2) around a bunch of foreground lenses.
    # i.e. t(r) = t0 exp(-r^2/2r0^2) (x+iy)^3/r^3

    nlens = 1000
    nsource = 100000
    t0 = 0.05
    r0 = 10.
    L = 100. * r0
    rng = np.random.RandomState(8675309)
    xl = (rng.random_sample(nlens)-0.5) * L
    yl = (rng.random_sample(nlens)-0.5) * L
    xs = (rng.random_sample(nsource)-0.5) * L
    ys = (rng.random_sample(nsource)-0.5) * L
    t1 = np.zeros( (nsource,) )
    t2 = np.zeros( (nsource,) )
    for x,y in zip(xl,yl):
        dx = xs-x
        dy = ys-y
        r2 = dx**2 + dy**2
        r = np.sqrt(r2)
        trad = t0 * np.exp(-0.5*r2/r0**2)
        theta = np.arctan2(dy,dx)
        t1 += trad * np.cos(3*theta)
        t2 += trad * np.sin(3*theta)

    lens_cat = treecorr.Catalog(x=xl, y=yl, x_units='arcmin', y_units='arcmin')
    source_cat = treecorr.Catalog(x=xs, y=ys, t1=t1, t2=t2, x_units='arcmin', y_units='arcmin')
    nt = treecorr.NTCorrelation(bin_size=0.1, min_sep=1., max_sep=20., sep_units='arcmin',
                                verbose=1)
    nt.process(lens_cat, source_cat)

    # Using nbins=None rather than omitting nbins is equivalent.
    nt2 = treecorr.NTCorrelation(bin_size=0.1, min_sep=1., max_sep=20., nbins=None, sep_units='arcmin')
    nt2.process(lens_cat, source_cat, num_threads=1)
    nt.process(lens_cat, source_cat, num_threads=1)
    assert nt2 == nt

    r = nt.meanr
    true_tr = t0 * np.exp(-0.5*r**2/r0**2)

    print('nt.xi = ',nt.xi)
    print('nt.xi_im = ',nt.xi_im)
    print('true_trad = ',true_tr)
    print('ratio = ',nt.xi / true_tr)
    print('diff = ',nt.xi - true_tr)
    print('max diff = ',max(abs(nt.xi - true_tr)))
    np.testing.assert_allclose(nt.xi, true_tr, rtol=0.1)
    np.testing.assert_allclose(nt.xi_im, 0, atol=5.e-3)

    nrand = nlens * 10
    xr = (rng.random_sample(nrand)-0.5) * L
    yr = (rng.random_sample(nrand)-0.5) * L
    rand_cat = treecorr.Catalog(x=xr, y=yr, x_units='arcmin', y_units='arcmin')
    rt = treecorr.NTCorrelation(bin_size=0.1, min_sep=1., max_sep=20., sep_units='arcmin',
                                verbose=1)
    rt.process(rand_cat, source_cat)
    print('rt.xi = ',rt.xi)
    xi, xi_im, varxi = nt.calculateXi(rt=rt)
    print('compensated xi = ',xi)
    print('compensated xi_im = ',xi_im)
    print('true_trad = ',true_tr)
    print('ratio = ',xi / true_tr)
    print('diff = ',xi - true_tr)
    print('max diff = ',max(abs(xi - true_tr)))
    # It turns out this doesn't come out much better.  I think the imprecision is mostly just due
    # to the smallish number of lenses, not to edge effects
    np.testing.assert_allclose(xi, true_tr, rtol=0.1)
    np.testing.assert_allclose(xi_im, 0, atol=5.e-3)

    # Check that we get the same result using the corr2 function:
    config = treecorr.read_config('configs/nt.yaml')
    try:
        import fitsio
    except ImportError:
        pass
    else:
        lens_cat.write(os.path.join('data','nt_lens.fits'))
        source_cat.write(os.path.join('data','nt_source.fits'))
        rand_cat.write(os.path.join('data','nt_rand.fits'))
        config['verbose'] = 0
        config['precision'] = 8
        treecorr.corr2(config)
        corr2_output = np.genfromtxt(os.path.join('output','nt.out'), names=True, skip_header=1)
        print('nt.xi = ',nt.xi)
        print('xi = ',xi)
        print('from corr2 output = ',corr2_output['tR'])
        print('ratio = ',corr2_output['tR']/xi)
        print('diff = ',corr2_output['tR']-xi)
        np.testing.assert_allclose(corr2_output['tR'], xi)
        print('xi_im from corr2 output = ',corr2_output['tR_im'])
        np.testing.assert_allclose(corr2_output['tR_im'], xi_im)

        # In the corr2 context, you can turn off the compensated bit, even if there are randoms
        # (e.g. maybe you only want randoms for some nn calculation, but not nt.)
        config['nt_statistic'] = 'simple'
        treecorr.corr2(config)
        corr2_output = np.genfromtxt(os.path.join('output','nt.out'), names=True, skip_header=1)
        xi_simple, _, _ = nt.calculateXi()
        np.testing.assert_equal(xi_simple, nt.xi)
        np.testing.assert_allclose(corr2_output['tR'], xi_simple, rtol=1.e-3)

    # Check the fits write option
    try:
        import fitsio
    except ImportError:
        pass
    else:
        out_file_name1 = os.path.join('output','nt_out1.fits')
        nt.write(out_file_name1)
        data = fitsio.read(out_file_name1)
        np.testing.assert_almost_equal(data['r_nom'], np.exp(nt.logr))
        np.testing.assert_almost_equal(data['meanr'], nt.meanr)
        np.testing.assert_almost_equal(data['meanlogr'], nt.meanlogr)
        np.testing.assert_almost_equal(data['tR'], nt.xi)
        np.testing.assert_almost_equal(data['tR_im'], nt.xi_im)
        np.testing.assert_almost_equal(data['sigma'], np.sqrt(nt.varxi))
        np.testing.assert_almost_equal(data['weight'], nt.weight)
        np.testing.assert_almost_equal(data['npairs'], nt.npairs)

        out_file_name2 = os.path.join('output','nt_out2.fits')
        nt.write(out_file_name2, rt=rt)
        data = fitsio.read(out_file_name2)
        np.testing.assert_almost_equal(data['r_nom'], np.exp(nt.logr))
        np.testing.assert_almost_equal(data['meanr'], nt.meanr)
        np.testing.assert_almost_equal(data['meanlogr'], nt.meanlogr)
        np.testing.assert_almost_equal(data['tR'], xi)
        np.testing.assert_almost_equal(data['tR_im'], xi_im)
        np.testing.assert_almost_equal(data['sigma'], np.sqrt(varxi))
        np.testing.assert_almost_equal(data['weight'], nt.weight)
        np.testing.assert_almost_equal(data['npairs'], nt.npairs)

        # Check the read function
        nt2 = treecorr.NTCorrelation.from_file(out_file_name2)
        np.testing.assert_almost_equal(nt2.logr, nt.logr)
        np.testing.assert_almost_equal(nt2.meanr, nt.meanr)
        np.testing.assert_almost_equal(nt2.meanlogr, nt.meanlogr)
        np.testing.assert_almost_equal(nt2.xi, nt.xi)
        np.testing.assert_almost_equal(nt2.xi_im, nt.xi_im)
        np.testing.assert_almost_equal(nt2.varxi, nt.varxi)
        np.testing.assert_almost_equal(nt2.weight, nt.weight)
        np.testing.assert_almost_equal(nt2.npairs, nt.npairs)
        assert nt2.coords == nt.coords
        assert nt2.metric == nt.metric
        assert nt2.sep_units == nt.sep_units
        assert nt2.bin_type == nt.bin_type


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
    t0 = 0.05
    r0 = 10.
    L = 50. * r0
    rng = np.random.RandomState(8675309)
    xl = (rng.random_sample(nlens)-0.5) * L
    yl = (rng.random_sample(nlens)-0.5) * L
    xs = (rng.random_sample( (nsource,ncats) )-0.5) * L
    ys = (rng.random_sample( (nsource,ncats) )-0.5) * L
    t1 = np.zeros( (nsource,ncats) )
    t2 = np.zeros( (nsource,ncats) )
    w = rng.random_sample( (nsource,ncats) ) + 0.5
    for x,y in zip(xl,yl):
        dx = xs-x
        dy = ys-y
        r2 = dx**2 + dy**2
        r = np.sqrt(r2)
        trad = t0 * np.exp(-0.5*r2/r0**2)
        theta = np.arctan2(dy,dx)
        t1 += trad * np.cos(3*theta)
        t2 += trad * np.sin(3*theta)

    lens_cat = treecorr.Catalog(x=xl, y=yl, x_units='arcmin', y_units='arcmin')
    source_cats = [ treecorr.Catalog(x=xs[:,k], y=ys[:,k], t1=t1[:,k], t2=t2[:,k], w=w[:,k],
                                     x_units='arcmin', y_units='arcmin') for k in range(ncats) ]
    full_source_cat = treecorr.Catalog(x=xs.flatten(), y=ys.flatten(), w=w.flatten(),
                                       t1=t1.flatten(), t2=t2.flatten(),
                                       x_units='arcmin', y_units='arcmin')

    for k in range(ncats):
        # These could each be done on different machines in a real world application.
        nt = treecorr.NTCorrelation(bin_size=0.1, min_sep=1., max_sep=25., sep_units='arcmin',
                                    verbose=1)
        # These should use process_cross, not process, since we don't want to call finalize.
        nt.process_cross(lens_cat, source_cats[k])
        nt.write(os.path.join('output','nt_piece_%d.fits'%k))

    pieces_nt = treecorr.NTCorrelation(bin_size=0.1, min_sep=1., max_sep=25., sep_units='arcmin')
    for k in range(ncats):
        nt = pieces_nt.copy()
        nt.read(os.path.join('output','nt_piece_%d.fits'%k))
        pieces_nt += nt
    vart = treecorr.calculateVarT(source_cats)
    pieces_nt.finalize(vart)

    full_nt = treecorr.NTCorrelation(bin_size=0.1, min_sep=1., max_sep=25., sep_units='arcmin',
                                     verbose=1)
    full_nt.process(lens_cat, full_source_cat)

    print('max error in meanr = ',np.max(pieces_nt.meanr - full_nt.meanr),)
    print('    max meanr = ',np.max(full_nt.meanr))
    print('max error in meanlogr = ',np.max(pieces_nt.meanlogr - full_nt.meanlogr),)
    print('    max meanlogr = ',np.max(full_nt.meanlogr))
    print('max error in weight = ',np.max(pieces_nt.weight - full_nt.weight),)
    print('    max weight = ',np.max(full_nt.weight))
    print('max error in xi = ',np.max(pieces_nt.xi - full_nt.xi),)
    print('    max xi = ',np.max(full_nt.xi))
    print('max error in xi_im = ',np.max(pieces_nt.xi_im - full_nt.xi_im),)
    print('    max xi_im = ',np.max(full_nt.xi_im))
    print('max error in varxi = ',np.max(pieces_nt.varxi - full_nt.varxi),)
    print('    max varxi = ',np.max(full_nt.varxi))
    np.testing.assert_allclose(pieces_nt.meanr, full_nt.meanr, rtol=2.e-3)
    np.testing.assert_allclose(pieces_nt.meanlogr, full_nt.meanlogr, atol=2.e-3)
    np.testing.assert_allclose(pieces_nt.weight, full_nt.weight, rtol=3.e-2)
    np.testing.assert_allclose(pieces_nt.xi, full_nt.xi, rtol=0.1)
    np.testing.assert_allclose(pieces_nt.xi_im, full_nt.xi_im, atol=2.e-3)
    np.testing.assert_allclose(pieces_nt.varxi, full_nt.varxi, rtol=3.e-2)

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
                                      t1=t1.flatten(), t2=t2.flatten(),
                                      wpos=w.flatten(), w=w2[k].flatten(),
                                      x_units='arcmin', y_units='arcmin') for k in range(ncats) ]

    nt2 = [ full_nt.copy() for k in range(ncats) ]
    for k in range(ncats):
        nt2[k].clear()
        nt2[k].process_cross(lens_cat, source_cats2[k])

    pieces_nt2 = full_nt.copy()
    pieces_nt2.clear()
    for k in range(ncats):
        pieces_nt2 += nt2[k]
    pieces_nt2.finalize(vart)

    print('max error in meanr = ',np.max(pieces_nt2.meanr - full_nt.meanr),)
    print('    max meanr = ',np.max(full_nt.meanr))
    print('max error in meanlogr = ',np.max(pieces_nt2.meanlogr - full_nt.meanlogr),)
    print('    max meanlogr = ',np.max(full_nt.meanlogr))
    print('max error in weight = ',np.max(pieces_nt2.weight - full_nt.weight),)
    print('    max weight = ',np.max(full_nt.weight))
    print('max error in xi = ',np.max(pieces_nt2.xi - full_nt.xi),)
    print('    max xi = ',np.max(full_nt.xi))
    print('max error in xi_im = ',np.max(pieces_nt2.xi_im - full_nt.xi_im),)
    print('    max xi_im = ',np.max(full_nt.xi_im))
    print('max error in varxi = ',np.max(pieces_nt2.varxi - full_nt.varxi),)
    print('    max varxi = ',np.max(full_nt.varxi))
    np.testing.assert_allclose(pieces_nt2.meanr, full_nt.meanr, rtol=1.e-7)
    np.testing.assert_allclose(pieces_nt2.meanlogr, full_nt.meanlogr, rtol=1.e-7)
    np.testing.assert_allclose(pieces_nt2.weight, full_nt.weight, rtol=1.e-7)
    np.testing.assert_allclose(pieces_nt2.xi, full_nt.xi, rtol=1.e-7)
    np.testing.assert_allclose(pieces_nt2.xi_im, full_nt.xi_im, atol=1.e-10)
    np.testing.assert_allclose(pieces_nt2.varxi, full_nt.varxi, rtol=1.e-7)

    # Can also do this with initialize/finalize options
    pieces_nt3 = full_nt.copy()
    for k in range(ncats):
        pieces_nt3.process(lens_cat, source_cats2[k], initialize=(k==0), finalize=(k==ncats-1))

    np.testing.assert_allclose(pieces_nt3.meanr, full_nt.meanr, rtol=1.e-7)
    np.testing.assert_allclose(pieces_nt3.meanlogr, full_nt.meanlogr, rtol=1.e-7)
    np.testing.assert_allclose(pieces_nt3.weight, full_nt.weight, rtol=1.e-7)
    np.testing.assert_allclose(pieces_nt3.xi, full_nt.xi, rtol=1.e-7)
    np.testing.assert_allclose(pieces_nt3.xi_im, full_nt.xi_im, atol=1.e-10)
    np.testing.assert_allclose(pieces_nt3.varxi, full_nt.varxi, rtol=1.e-7)

    # Try this with corr2
    lens_cat.write(os.path.join('data','nt_wpos_lens.fits'))
    for i, sc in enumerate(source_cats2):
        sc.write(os.path.join('data','nt_wpos_source%d.fits'%i))
    config = treecorr.read_config('configs/nt_wpos.yaml')
    config['verbose'] = 0
    treecorr.corr2(config)
    data = fitsio.read(config['nt_file_name'])
    print('data.dtype = ',data.dtype)
    np.testing.assert_allclose(data['meanr'], pieces_nt3.meanr)
    np.testing.assert_allclose(data['meanlogr'], pieces_nt3.meanlogr)
    np.testing.assert_allclose(data['weight'], pieces_nt3.weight)
    np.testing.assert_allclose(data['tR'], pieces_nt3.xi)
    np.testing.assert_allclose(data['tR_im'], pieces_nt3.xi_im)
    np.testing.assert_allclose(data['sigma']**2, pieces_nt3.varxi)


@timer
def test_varxi():
    # Test that varxi is correct (or close) based on actual variance of many runs.

    # Signal doesn't matter much.  Use the one from test_gg.
    t0 = 0.05
    r0 = 10.
    L = 10 * r0
    rng = np.random.RandomState(8675309)

    nsource = 1000
    nrand = 10
    nruns = 50000
    lens = treecorr.Catalog(x=[0], y=[0])

    file_name = 'data/test_varxi_nt.npz'
    print(file_name)
    if not os.path.isfile(file_name):
        all_nts = []
        all_rts = []
        for run in range(nruns):
            print(f'{run}/{nruns}')
            x2 = (rng.random_sample(nsource)-0.5) * L
            y2 = (rng.random_sample(nsource)-0.5) * L
            x3 = (rng.random_sample(nrand)-0.5) * L
            y3 = (rng.random_sample(nrand)-0.5) * L

            r2 = (x2**2 + y2**2)/r0**2
            theta = np.arctan2(y2,x2)
            t1 = t0 * np.exp(-r2/2.) * np.cos(3*theta)
            t2 = t0 * np.exp(-r2/2.) * np.sin(3*theta)
            # This time, add some shape noise (different each run).
            t1 += rng.normal(0, 0.1, size=nsource)
            t2 += rng.normal(0, 0.1, size=nsource)
            # Varied weights are hard, but at least check that non-unit weights work correctly.
            w = np.ones_like(x2) * 5

            source = treecorr.Catalog(x=x2, y=y2, w=w, t1=t1, t2=t2)
            rand = treecorr.Catalog(x=x3, y=y3)
            nt = treecorr.NTCorrelation(bin_size=0.2, min_sep=10., max_sep=30.)
            rt = treecorr.NTCorrelation(bin_size=0.2, min_sep=10., max_sep=30.)
            nt.process(lens, source)
            rt.process(rand, source)
            all_nts.append(nt)
            all_rts.append(rt)

        all_xis = [nt.calculateXi() for nt in all_nts]
        var_xi_1 = np.var([xi[0] for xi in all_xis], axis=0)
        mean_varxi_1 = np.mean([xi[2] for xi in all_xis], axis=0)

        all_xis = [nt.calculateXi(rt=rt) for (nt,rt) in zip(all_nts, all_rts)]
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
    t1 = t0 * np.exp(-r2/2.) * np.cos(3*theta)
    t2 = t0 * np.exp(-r2/2.) * np.sin(3*theta)
    t1 += rng.normal(0, 0.1, size=nsource)
    t2 += rng.normal(0, 0.1, size=nsource)
    # Varied weights are hard, but at least check that non-unit weights work correctly.
    w = np.ones_like(x2) * 5

    source = treecorr.Catalog(x=x2, y=y2, w=w, t1=t1, t2=t2)
    rand = treecorr.Catalog(x=x3, y=y3)
    nt = treecorr.NTCorrelation(bin_size=0.2, min_sep=10., max_sep=30.)
    rt = treecorr.NTCorrelation(bin_size=0.2, min_sep=10., max_sep=30.)
    nt.process(lens, source)
    rt.process(rand, source)

    print('single run:')
    print('Uncompensated')
    print('ratio = ',nt.varxi / var_xi_1)
    print('max relerr for xi = ',np.max(np.abs((nt.varxi - var_xi_1)/var_xi_1)))
    np.testing.assert_allclose(nt.varxi, var_xi_1, rtol=0.6)

    xi, xi_im, varxi = nt.calculateXi(rt=rt)
    print('Compensated')
    print('ratio = ',varxi / var_xi_2)
    print('max relerr for xi = ',np.max(np.abs((varxi - var_xi_2)/var_xi_2)))
    np.testing.assert_allclose(varxi, var_xi_2, rtol=0.5)

@timer
def test_jk():

    # Similar to the profile we use above, but multiple "lenses".
    t0 = 0.05
    r0 = 30.
    L = 30 * r0
    rng = np.random.RandomState(8675309)

    nsource = 100000
    nrand = 1000
    nlens = 300
    nruns = 1000
    npatch = 64

    corr_params = dict(bin_size=0.3, min_sep=10, max_sep=40, bin_slop=0.1)

    def make_spin3_field(rng):
        x1 = (rng.random(nlens)-0.5) * L
        y1 = (rng.random(nlens)-0.5) * L
        w = rng.random(nlens) + 10
        x2 = (rng.random(nsource)-0.5) * L
        y2 = (rng.random(nsource)-0.5) * L
        x3 = (rng.random(nrand)-0.5) * L
        y3 = (rng.random(nrand)-0.5) * L

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
            t1 += w[i] * t0 * np.exp(-r2/2.) * np.cos(3*theta)
            t2 += w[i] * t0 * np.exp(-r2/2.) * np.sin(3*theta)
        return x1, y1, w, x2, y2, t1, t2, x3, y3

    file_name = 'data/test_nt_jk_{}.npz'.format(nruns)
    print(file_name)
    if not os.path.isfile(file_name):
        all_nts = []
        all_rts = []
        rng = np.random.default_rng()
        for run in range(nruns):
            x1, y1, w, x2, y2, t1, t2, x3, y3 = make_spin3_field(rng)
            print(run,': ',np.mean(t1),np.std(t1),np.min(t1),np.max(t1))
            cat1 = treecorr.Catalog(x=x1, y=y1, w=w)
            cat2 = treecorr.Catalog(x=x2, y=y2, t1=t1, t2=t2)
            cat3 = treecorr.Catalog(x=x3, y=y3)
            nt = treecorr.NTCorrelation(corr_params)
            rt = treecorr.NTCorrelation(corr_params)
            nt.process(cat1, cat2)
            rt.process(cat3, cat2)
            all_nts.append(nt)
            all_rts.append(rt)

        mean_xi = np.mean([nt.xi for nt in all_nts], axis=0)
        var_xi = np.var([nt.xi for nt in all_nts], axis=0)
        mean_varxi = np.mean([nt.varxi for nt in all_nts], axis=0)

        for nt, rt in zip(all_nts, all_rts):
            nt.calculateXi(rt=rt)

        mean_xi_r = np.mean([nt.xi for nt in all_nts], axis=0)
        var_xi_r = np.var([nt.xi for nt in all_nts], axis=0)
        mean_varxi_r = np.mean([nt.varxi for nt in all_nts], axis=0)

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
    x1, y1, w, x2, y2, t1, t2, x3, y3 = make_spin3_field(rng)

    cat1 = treecorr.Catalog(x=x1, y=y1, w=w)
    cat2 = treecorr.Catalog(x=x2, y=y2, t1=t1, t2=t2)
    nt1 = treecorr.NTCorrelation(corr_params)
    nt1.process(cat1, cat2)

    print('weight = ',nt1.weight)
    print('xi = ',nt1.xi)
    print('varxi = ',nt1.varxi)
    print('pullsq for xi = ',(nt1.xi-mean_xi)**2/var_xi)
    print('max pull for xi = ',np.sqrt(np.max((nt1.xi-mean_xi)**2/var_xi)))
    np.testing.assert_array_less((nt1.xi-mean_xi)**2, 9*var_xi)  # < 3 sigma pull
    np.testing.assert_allclose(nt1.varxi, mean_varxi, rtol=0.1)

    # Now run with patches, but still with shot variance.  Should be basically the same answer.
    try:
        import fitsio
        patch_dir = 'output'
        low_mem = True
    except ImportError:
        # If we cannot write to a fits file, skip the save_patch_dir tests.
        patch_dir = None
        low_mem = False
    cat2p = treecorr.Catalog(x=x2, y=y2, t1=t1, t2=t2, npatch=npatch, save_patch_dir=patch_dir)
    if low_mem:
        cat2p.write_patches()  # Force rewrite of any existing saved patches.
    cat1p = treecorr.Catalog(x=x1, y=y1, w=w, patch_centers=cat2p.patch_centers)
    print('tot w = ',np.sum(w))
    print('Patch\tNlens\tNsource')
    for i in range(npatch):
        print('%d\t%d\t%d'%(i,np.sum(cat1p.w[cat1p.patch==i]),np.sum(cat2p.w[cat2p.patch==i])))
    nt2 = treecorr.NTCorrelation(corr_params)
    nt2.process(cat1p, cat2p)
    print('weight = ',nt2.weight)
    print('xi = ',nt2.xi)
    print('xi1 = ',nt1.xi)
    print('varxi = ',nt2.varxi)
    np.testing.assert_allclose(nt2.weight, nt1.weight, rtol=1.e-2)
    np.testing.assert_allclose(nt2.xi, nt1.xi, rtol=1.e-2)
    np.testing.assert_allclose(nt2.varxi, nt1.varxi, rtol=1.e-2)

    # estimate_cov with var_method='shot' returns just the diagonal.
    np.testing.assert_allclose(nt2.estimate_cov('shot'), nt2.varxi)
    np.testing.assert_allclose(nt1.estimate_cov('shot'), nt1.varxi)

    # Now try jackknife variance estimate.
    cov2 = nt2.estimate_cov('jackknife')
    print('varxi = ',np.diagonal(cov2))
    print('cf var_xi = ',var_xi)
    np.testing.assert_allclose(np.diagonal(cov2), var_xi, rtol=0.6)

    # Check only using patches for one of the two catalogs.
    # Not as good as using patches for both, but not much worse.
    nt3 = treecorr.NTCorrelation(corr_params, var_method='jackknife')
    nt3.process(cat1p, cat2)
    print('varxi = ',nt3.varxi)
    np.testing.assert_allclose(nt3.weight, nt1.weight, rtol=1.e-2)
    np.testing.assert_allclose(nt3.xi, nt1.xi, rtol=1.e-2)
    np.testing.assert_allclose(nt3.varxi, var_xi, rtol=0.5)

    nt4 = treecorr.NTCorrelation(corr_params, var_method='jackknife', rng=rng)
    nt4.process(cat1, cat2p)
    print('varxi = ',nt4.varxi)
    np.testing.assert_allclose(nt4.weight, nt1.weight, rtol=1.e-2)
    np.testing.assert_allclose(nt4.xi, nt1.xi, rtol=1.e-2)
    np.testing.assert_allclose(nt4.varxi, var_xi, rtol=0.6)

    # Use initialize/finalize
    nt5 = treecorr.NTCorrelation(corr_params)
    for k1, p1 in enumerate(cat1p.get_patches()):
        for k2, p2 in enumerate(cat2p.get_patches()):
            nt5.process(p1, p2, initialize=(k1==k2==0), finalize=(k1==k2==npatch-1))
    np.testing.assert_allclose(nt5.xi, nt2.xi)
    np.testing.assert_allclose(nt5.weight, nt2.weight)
    np.testing.assert_allclose(nt5.varxi, nt2.varxi)

    # Check that these still work after roundtripping through a file.
    try:
        import fitsio
    except ImportError:
        pass
    else:
        file_name = os.path.join('output','test_write_results_nt.fits')
        nt2.write(file_name, write_patch_results=True)
        nt5 = treecorr.NTCorrelation.from_file(file_name)
        cov5 = nt5.estimate_cov('jackknife')
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
    rt5 = treecorr.NTCorrelation(corr_params)
    rt5.process(cat3, cat2)
    nt5 = nt1.copy()
    nt5.calculateXi(rt=rt5)
    print('weight = ',nt5.weight)
    print('xi = ',nt5.xi)
    print('varxi = ',nt5.varxi)
    print('ratio = ',nt5.varxi / var_xi_r)
    print('pullsq for xi = ',(nt5.xi-mean_xi_r)**2/var_xi_r)
    print('max pull for xi = ',np.sqrt(np.max((nt5.xi-mean_xi_r)**2/var_xi_r)))
    np.testing.assert_array_less((nt5.xi-mean_xi_r)**2, 9*var_xi_r)  # < 3 sigma pull
    np.testing.assert_allclose(nt5.varxi, mean_varxi_r, rtol=0.1)

    # Repeat with patches
    cat3p = treecorr.Catalog(x=x3, y=y3, patch_centers=cat2p.patch_centers)
    rt6 = treecorr.NTCorrelation(corr_params)
    rt6.process(cat3p, cat2p, low_mem=low_mem)
    nt6 = nt2.copy()
    nt6.calculateXi(rt=rt6)
    cov6 = nt6.estimate_cov('jackknife')
    np.testing.assert_allclose(np.diagonal(cov6), var_xi_r, rtol=0.7)

    # Use a random catalog without patches.
    rt7 = treecorr.NTCorrelation(corr_params)
    rt7.process(cat3, cat2p)
    nt7 = nt4.copy()
    nt7.calculateXi(rt=rt7)
    cov7 = nt7.estimate_cov('jackknife')
    np.testing.assert_allclose(np.diagonal(cov7), var_xi_r, rtol=0.7)

    nt8 = nt2.copy()
    nt8.calculateXi(rt=rt7)
    cov8 = nt8.estimate_cov('jackknife')
    np.testing.assert_allclose(np.diagonal(cov8), var_xi_r, rtol=0.6)

    # Check some intalid actions
    # Bad var_method
    with assert_raises(ValueError):
        nt2.estimate_cov('intalid')
    # Not run on patches, but need patches
    with assert_raises(ValueError):
        nt1.estimate_cov('jackknife')
    with assert_raises(ValueError):
        nt1.estimate_cov('sample')
    with assert_raises(ValueError):
        nt1.estimate_cov('marked_bootstrap')
    with assert_raises(ValueError):
        nt1.estimate_cov('bootstrap')
    # rt also needs patches (at least for the g part).
    with assert_raises(RuntimeError):
        nt2.calculateXi(rt=nt1)

    cat1a = treecorr.Catalog(x=x1[:100], y=y1[:100], npatch=10)
    cat2a = treecorr.Catalog(x=x2[:100], y=y2[:100], t1=t1[:100], t2=t2[:100], npatch=10)
    cat1b = treecorr.Catalog(x=x1[:100], y=y1[:100], npatch=2)
    cat2b = treecorr.Catalog(x=x2[:100], y=y2[:100], t1=t1[:100], t2=t2[:100], npatch=2)
    nt6 = treecorr.NTCorrelation(corr_params)
    nt7 = treecorr.NTCorrelation(corr_params)
    # All catalogs need to have the same number of patches
    with assert_raises(RuntimeError):
        nt6.process(cat1a,cat2b)
    with assert_raises(RuntimeError):
        nt7.process(cat1b,cat2a)


if __name__ == '__main__':
    test_direct()
    test_direct_spherical()
    test_single()
    test_spherical()
    test_nt()
    test_pieces()
    test_varxi()
    test_jk()
