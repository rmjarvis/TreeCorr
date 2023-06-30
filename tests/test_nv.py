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
    v12 = rng.normal(0,0.2, (ngal,) )
    v22 = rng.normal(0,0.2, (ngal,) )

    cat1 = treecorr.Catalog(x=x1, y=y1, w=w1)
    cat2 = treecorr.Catalog(x=x2, y=y2, w=w2, v1=v12, v2=v22)

    min_sep = 1.
    max_sep = 50.
    nbins = 50
    bin_size = np.log(max_sep/min_sep) / nbins
    nv = treecorr.NVCorrelation(min_sep=min_sep, max_sep=max_sep, nbins=nbins, brute=True)
    nv.process(cat1, cat2)

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
        xi = ww * (v12 + 1j*v22) * expmialpha

        index = np.floor(np.log(r/min_sep) / bin_size).astype(int)
        mask = (index >= 0) & (index < nbins)
        np.add.at(true_npairs, index[mask], 1)
        np.add.at(true_weight, index[mask], ww[mask])
        np.add.at(true_xi, index[mask], xi[mask])

    true_xi /= true_weight

    print('true_npairs = ',true_npairs)
    print('diff = ',nv.npairs - true_npairs)
    np.testing.assert_array_equal(nv.npairs, true_npairs)

    print('true_weight = ',true_weight)
    print('diff = ',nv.weight - true_weight)
    np.testing.assert_allclose(nv.weight, true_weight, rtol=1.e-5, atol=1.e-8)

    print('true_xi = ',true_xi)
    print('nv.xi = ',nv.xi)
    print('nv.xi_im = ',nv.xi_im)
    np.testing.assert_allclose(nv.xi, true_xi.real, rtol=1.e-4, atol=1.e-8)
    np.testing.assert_allclose(nv.xi_im, true_xi.imag, rtol=1.e-4, atol=1.e-8)

    # Check that running via the corr2 script works correctly.
    config = treecorr.config.read_config('configs/nv_direct.yaml')
    try:
        import fitsio
    except ImportError:
        pass
    else:
        cat1.write(config['file_name'])
        cat2.write(config['file_name2'])
        treecorr.corr2(config)
        data = fitsio.read(config['nv_file_name'])
        np.testing.assert_allclose(data['r_nom'], nv.rnom)
        np.testing.assert_allclose(data['npairs'], nv.npairs)
        np.testing.assert_allclose(data['weight'], nv.weight)
        np.testing.assert_allclose(data['vR'], nv.xi, rtol=1.e-3)
        np.testing.assert_allclose(data['vT'], nv.xi_im, rtol=1.e-3)

        # Invalid with only one file_name
        del config['file_name2']
        with assert_raises(TypeError):
            treecorr.corr2(config)
        config['file_name2'] = 'data/nv_direct_cat2.fits'
        # Invalid to request compoensated if no rand_file
        config['nv_statistic'] = 'compensated'
        with assert_raises(TypeError):
            treecorr.corr2(config)

    # Repeat with binslop = 0
    # And don't do any top-level recursion so we actually test not going to the leaves.
    nv = treecorr.NVCorrelation(min_sep=min_sep, max_sep=max_sep, nbins=nbins, bin_slop=0,
                                max_top=0)
    nv.process(cat1, cat2)
    np.testing.assert_array_equal(nv.npairs, true_npairs)
    np.testing.assert_allclose(nv.weight, true_weight, rtol=1.e-5, atol=1.e-8)
    np.testing.assert_allclose(nv.xi, true_xi.real, rtol=1.e-3, atol=3.e-4)
    np.testing.assert_allclose(nv.xi_im, true_xi.imag, atol=3.e-4)

    # Check a few basic operations with a NVCorrelation object.
    do_pickle(nv)

    nv2 = nv.copy()
    nv2 += nv
    np.testing.assert_allclose(nv2.npairs, 2*nv.npairs)
    np.testing.assert_allclose(nv2.weight, 2*nv.weight)
    np.testing.assert_allclose(nv2.meanr, 2*nv.meanr)
    np.testing.assert_allclose(nv2.meanlogr, 2*nv.meanlogr)
    np.testing.assert_allclose(nv2.xi, 2*nv.xi)
    np.testing.assert_allclose(nv2.xi_im, 2*nv.xi_im)

    nv2.clear()
    nv2 += nv
    np.testing.assert_allclose(nv2.npairs, nv.npairs)
    np.testing.assert_allclose(nv2.weight, nv.weight)
    np.testing.assert_allclose(nv2.meanr, nv.meanr)
    np.testing.assert_allclose(nv2.meanlogr, nv.meanlogr)
    np.testing.assert_allclose(nv2.xi, nv.xi)
    np.testing.assert_allclose(nv2.xi_im, nv.xi_im)

    ascii_name = 'output/nv_ascii.txt'
    nv.write(ascii_name, precision=16)
    nv3 = treecorr.NVCorrelation(min_sep=min_sep, max_sep=max_sep, nbins=nbins)
    nv3.read(ascii_name)
    np.testing.assert_allclose(nv3.npairs, nv.npairs)
    np.testing.assert_allclose(nv3.weight, nv.weight)
    np.testing.assert_allclose(nv3.meanr, nv.meanr)
    np.testing.assert_allclose(nv3.meanlogr, nv.meanlogr)
    np.testing.assert_allclose(nv3.xi, nv.xi)
    np.testing.assert_allclose(nv3.xi_im, nv.xi_im)

    try:
        import fitsio
    except ImportError:
        pass
    else:
        fits_name = 'output/nv_fits.fits'
        nv.write(fits_name)
        nv4 = treecorr.NVCorrelation(min_sep=min_sep, max_sep=max_sep, nbins=nbins)
        nv4.read(fits_name)
        np.testing.assert_allclose(nv4.npairs, nv.npairs)
        np.testing.assert_allclose(nv4.weight, nv.weight)
        np.testing.assert_allclose(nv4.meanr, nv.meanr)
        np.testing.assert_allclose(nv4.meanlogr, nv.meanlogr)
        np.testing.assert_allclose(nv4.xi, nv.xi)
        np.testing.assert_allclose(nv4.xi_im, nv.xi_im)

    with assert_raises(TypeError):
        nv2 += config
    nv4 = treecorr.NVCorrelation(min_sep=min_sep/2, max_sep=max_sep, nbins=nbins)
    with assert_raises(ValueError):
        nv2 += nv4
    nv5 = treecorr.NVCorrelation(min_sep=min_sep, max_sep=max_sep*2, nbins=nbins)
    with assert_raises(ValueError):
        nv2 += nv5
    nv6 = treecorr.NVCorrelation(min_sep=min_sep, max_sep=max_sep, nbins=nbins*2)
    with assert_raises(ValueError):
        nv2 += nv6



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
    v12 = rng.normal(0,0.2, (ngal,) )
    v22 = rng.normal(0,0.2, (ngal,) )

    ra1, dec1 = coord.CelestialCoord.xyz_to_radec(x1,y1,z1)
    ra2, dec2 = coord.CelestialCoord.xyz_to_radec(x2,y2,z2)

    cat1 = treecorr.Catalog(ra=ra1, dec=dec1, ra_units='rad', dec_units='rad', w=w1)
    cat2 = treecorr.Catalog(ra=ra2, dec=dec2, ra_units='rad', dec_units='rad', w=w2, v1=v12, v2=v22)

    min_sep = 1.
    max_sep = 10.
    nbins = 50
    bin_size = np.log(max_sep/min_sep) / nbins
    nv = treecorr.NVCorrelation(min_sep=min_sep, max_sep=max_sep, nbins=nbins,
                                sep_units='deg', brute=True)
    nv.process(cat1, cat2)

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
            theta2 = 90*coord.degrees - c2[j].angleBetween(c1[i], north_pole)
            expmtheta2 = np.cos(theta2) - 1j * np.sin(theta2)

            v2 = v12[j] + 1j * v22[j]
            v2 *= expmtheta2

            ww = w1[i] * w2[j]
            xi = -w1[i] * w2[j] * v2

            true_npairs[index] += 1
            true_weight[index] += ww
            true_xi[index] += xi

    true_xi /= true_weight

    print('true_npairs = ',true_npairs)
    print('diff = ',nv.npairs - true_npairs)
    np.testing.assert_array_equal(nv.npairs, true_npairs)

    print('true_weight = ',true_weight)
    print('diff = ',nv.weight - true_weight)
    np.testing.assert_allclose(nv.weight, true_weight, rtol=1.e-5, atol=1.e-8)

    print('true_xi = ',true_xi)
    print('nv.xi = ',nv.xi)
    print('nv.xi_im = ',nv.xi_im)
    np.testing.assert_allclose(nv.xi, true_xi.real, rtol=1.e-4, atol=1.e-8)
    np.testing.assert_allclose(nv.xi_im, true_xi.imag, rtol=1.e-4, atol=1.e-8)

    # Check that running via the corr2 script works correctly.
    config = treecorr.config.read_config('configs/nv_direct_spherical.yaml')
    try:
        import fitsio
    except ImportError:
        pass
    else:
        cat1.write(config['file_name'])
        cat2.write(config['file_name2'])
        treecorr.corr2(config)
        data = fitsio.read(config['nv_file_name'])
        np.testing.assert_allclose(data['r_nom'], nv.rnom)
        np.testing.assert_allclose(data['npairs'], nv.npairs)
        np.testing.assert_allclose(data['weight'], nv.weight)
        np.testing.assert_allclose(data['vR'], nv.xi, rtol=1.e-3)
        np.testing.assert_allclose(data['vT'], nv.xi_im, rtol=1.e-3)

    # Repeat with binslop = 0
    # And don't do any top-level recursion so we actually test not going to the leaves.
    nv = treecorr.NVCorrelation(min_sep=min_sep, max_sep=max_sep, nbins=nbins,
                                sep_units='deg', bin_slop=0, max_top=0)
    nv.process(cat1, cat2)
    np.testing.assert_array_equal(nv.npairs, true_npairs)
    np.testing.assert_allclose(nv.weight, true_weight, rtol=1.e-5, atol=1.e-8)
    np.testing.assert_allclose(nv.xi, true_xi.real, rtol=1.e-3, atol=2.e-4)
    np.testing.assert_allclose(nv.xi_im, true_xi.imag, atol=2.e-4)


@timer
def test_single():
    # Use v_radial(r) = v0 exp(-r^2/2r0^2) around a single lens
    # i.e. v(r) = v0 exp(-r^2/2r0^2) (x+iy)/r

    nsource = 300000
    v0 = 0.05
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

    lens_cat = treecorr.Catalog(x=[0], y=[0], x_units='arcmin', y_units='arcmin')
    source_cat = treecorr.Catalog(x=x, y=y, v1=v1, v2=v2, x_units='arcmin', y_units='arcmin')
    nv = treecorr.NVCorrelation(bin_size=0.1, min_sep=1., max_sep=20., sep_units='arcmin',
                                verbose=1)
    nv.process(lens_cat, source_cat)

    # log(<R>) != <logR>, but it should be close:
    print('meanlogr - log(meanr) = ',nv.meanlogr - np.log(nv.meanr))
    np.testing.assert_allclose(nv.meanlogr, np.log(nv.meanr), atol=1.e-3)

    r = nv.meanr
    true_vr = v0 * np.exp(-0.5*r**2/r0**2)

    print('nv.xi = ',nv.xi)
    print('nv.xi_im = ',nv.xi_im)
    print('true_vrad = ',true_vr)
    print('ratio = ',nv.xi / true_vr)
    print('diff = ',nv.xi - true_vr)
    print('max diff = ',max(abs(nv.xi - true_vr)))
    np.testing.assert_allclose(nv.xi, true_vr, rtol=3.e-2)
    np.testing.assert_allclose(nv.xi_im, 0, atol=1.e-4)

    # Check that we get the same result using the corr2 function:
    lens_cat.write(os.path.join('data','nv_single_lens.dat'))
    source_cat.write(os.path.join('data','nv_single_source.dat'))
    config = treecorr.read_config('configs/nv_single.yaml')
    config['verbose'] = 0
    treecorr.corr2(config)
    corr2_output = np.genfromtxt(os.path.join('output','nv_single.out'), names=True,
                                 skip_header=1)
    print('nv.xi = ',nv.xi)
    print('from corr2 output = ',corr2_output['vR'])
    print('ratio = ',corr2_output['vR']/nv.xi)
    print('diff = ',corr2_output['vR']-nv.xi)
    print('xi_im from corr2 output = ',corr2_output['vT'])
    np.testing.assert_allclose(corr2_output['vR'], nv.xi, rtol=1.e-3)
    np.testing.assert_allclose(corr2_output['vT'], 0, atol=1.e-4)

    # Check that adding results with different coords or metric emits a warning.
    lens_cat2 = treecorr.Catalog(x=[0], y=[0], z=[0])
    source_cat2 = treecorr.Catalog(x=x, y=y, z=x, v1=v1, v2=v2)
    with CaptureLog() as cl:
        nv2 = treecorr.NVCorrelation(bin_size=0.1, min_sep=1., max_sep=20., logger=cl.logger)
        nv2.process_cross(lens_cat2, source_cat2)
        nv2 += nv
    assert "Detected a change in catalog coordinate systems" in cl.output

    with CaptureLog() as cl:
        nv3 = treecorr.NVCorrelation(bin_size=0.1, min_sep=1., max_sep=20., logger=cl.logger)
        nv3.process_cross(lens_cat2, source_cat2, metric='Rperp')
        nv3 += nv2
    assert "Detected a change in metric" in cl.output

    # There is special handling for single-row catalogs when using np.genfromtxt rather
    # than pandas.  So mock it up to make sure we test it.
    treecorr.Catalog._emitted_pandas_warning = False  # Reset this, in case already triggered.
    with mock.patch.dict(sys.modules, {'pandas':None}):
        with CaptureLog() as cl:
            treecorr.corr2(config, logger=cl.logger)
        assert "Unable to import pandas" in cl.output
    corr2_output = np.genfromtxt(os.path.join('output','nv_single.out'), names=True,
                                 skip_header=1)
    np.testing.assert_allclose(corr2_output['vR'], nv.xi, rtol=1.e-3)


@timer
def test_spherical():
    # This is the same profile we used for test_single, but put into spherical coords.
    # We do the spherical trig by hand using the obvious formulae, rather than the clever
    # optimizations that are used by the TreeCorr code, thus serving as a useful test of
    # the latter.

    nsource = 300000
    v0 = 0.05
    r0 = 10. * coord.degrees / coord.radians
    L = 5. * r0
    rng = np.random.RandomState(8675309)
    x = (rng.random_sample(nsource)-0.5) * L
    y = (rng.random_sample(nsource)-0.5) * L
    r2 = (x**2 + y**2)
    r = np.sqrt(r2)
    vrad = v0 * np.exp(-0.5*r2/r0**2)
    v1 = vrad * x/r
    v2 = vrad * y/r
    theta = np.arctan2(y,x)

    nv = treecorr.NVCorrelation(bin_size=0.1, min_sep=1., max_sep=20., sep_units='deg',
                                verbose=1)
    r1 = np.exp(nv.logr) * (coord.degrees / coord.radians)
    true_vr = v0 * np.exp(-0.5*r1**2/r0**2)

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
        cosbeta = np.cos(beta)
        sinbeta = np.sin(beta)
        v1_sph = v1 * cosbeta - v2 * sinbeta
        v2_sph = v2 * cosbeta + v1 * sinbeta

        lens_cat = treecorr.Catalog(ra=[ra0], dec=[dec0], ra_units='rad', dec_units='rad')
        source_cat = treecorr.Catalog(ra=ra, dec=dec, v1=v1_sph, v2=v2_sph,
                                      ra_units='rad', dec_units='rad')
        nv.process(lens_cat, source_cat)

        print('ra0, dec0 = ',ra0,dec0)
        print('nv.xi = ',nv.xi)
        print('true_vrad = ',true_vr)
        print('ratio = ',nv.xi / true_vr)
        print('diff = ',nv.xi - true_vr)
        print('max diff = ',max(abs(nv.xi - true_vr)))
        # The 3rd and 4th centers are somewhat less accurate.  Not sure why.
        # The math seems to be right, since the last one that gets all the way to the pole
        # works, so I'm not sure what is going on.  It's just a few bins that get a bit less
        # accurate.  Possibly worth investigating further at some point...
        np.testing.assert_allclose(nv.xi, true_vr, rtol=0.1)

    # One more center that can be done very easily.  If the center is the north pole, then all
    # the radial vectors are pure (positive) v1.
    ra0 = 0
    dec0 = np.pi/2.
    ra = theta
    dec = np.pi/2. - 2.*np.arcsin(r/2.)

    lens_cat = treecorr.Catalog(ra=[ra0], dec=[dec0], ra_units='rad', dec_units='rad')
    source_cat = treecorr.Catalog(ra=ra, dec=dec, v1=np.zeros_like(vrad), v2=-vrad,
                                  ra_units='rad', dec_units='rad')
    nv.process(lens_cat, source_cat)

    print('nv.xi = ',nv.xi)
    print('nv.xi_im = ',nv.xi_im)
    print('true_vrad = ',true_vr)
    print('ratio = ',nv.xi / true_vr)
    print('diff = ',nv.xi - true_vr)
    print('max diff = ',max(abs(nv.xi - true_vr)))
    np.testing.assert_allclose(nv.xi, true_vr, rtol=0.1)
    np.testing.assert_allclose(nv.xi_im, 0, atol=1.e-4)

    # Check that we get the same result using the corr2 function:
    lens_cat.write(os.path.join('data','nv_spherical_lens.dat'))
    source_cat.write(os.path.join('data','nv_spherical_source.dat'))
    config = treecorr.read_config('configs/nv_spherical.yaml')
    config['verbose'] = 0
    treecorr.corr2(config)
    corr2_output = np.genfromtxt(os.path.join('output','nv_spherical.out'), names=True,
                                 skip_header=1)
    print('nv.xi = ',nv.xi)
    print('from corr2 output = ',corr2_output['vR'])
    print('ratio = ',corr2_output['vR']/nv.xi)
    print('diff = ',corr2_output['vR']-nv.xi)
    np.testing.assert_allclose(corr2_output['vR'], nv.xi, rtol=1.e-3)

    print('xi_im from corr2 output = ',corr2_output['vT'])
    np.testing.assert_allclose(corr2_output['vT'], 0., atol=3.e-5)


@timer
def test_nv():
    # Use v_radial(r) = v0 exp(-r^2/2r0^2) around a bunch of foreground lenses.
    # i.e. v(r) = v0 exp(-r^2/2r0^2) (x+iy)/r

    nlens = 1000
    nsource = 100000
    v0 = 0.05
    r0 = 10.
    L = 100. * r0
    rng = np.random.RandomState(8675309)
    xl = (rng.random_sample(nlens)-0.5) * L
    yl = (rng.random_sample(nlens)-0.5) * L
    xs = (rng.random_sample(nsource)-0.5) * L
    ys = (rng.random_sample(nsource)-0.5) * L
    v1 = np.zeros( (nsource,) )
    v2 = np.zeros( (nsource,) )
    for x,y in zip(xl,yl):
        dx = xs-x
        dy = ys-y
        r2 = dx**2 + dy**2
        r = np.sqrt(r2)
        vrad = v0 * np.exp(-0.5*r2/r0**2)
        v1 += vrad * dx/r
        v2 += vrad * dy/r

    lens_cat = treecorr.Catalog(x=xl, y=yl, x_units='arcmin', y_units='arcmin')
    source_cat = treecorr.Catalog(x=xs, y=ys, v1=v1, v2=v2, x_units='arcmin', y_units='arcmin')
    nv = treecorr.NVCorrelation(bin_size=0.1, min_sep=1., max_sep=20., sep_units='arcmin',
                                verbose=1)
    nv.process(lens_cat, source_cat)

    # Using nbins=None rather than omitting nbins is equivalent.
    nv2 = treecorr.NVCorrelation(bin_size=0.1, min_sep=1., max_sep=20., nbins=None, sep_units='arcmin')
    nv2.process(lens_cat, source_cat, num_threads=1)
    nv.process(lens_cat, source_cat, num_threads=1)
    assert nv2 == nv

    r = nv.meanr
    true_vr = v0 * np.exp(-0.5*r**2/r0**2)

    print('nv.xi = ',nv.xi)
    print('nv.xi_im = ',nv.xi_im)
    print('true_vrad = ',true_vr)
    print('ratio = ',nv.xi / true_vr)
    print('diff = ',nv.xi - true_vr)
    print('max diff = ',max(abs(nv.xi - true_vr)))
    np.testing.assert_allclose(nv.xi, true_vr, rtol=0.1)
    np.testing.assert_allclose(nv.xi_im, 0, atol=5.e-3)

    nrand = nlens * 3
    xr = (rng.random_sample(nrand)-0.5) * L
    yr = (rng.random_sample(nrand)-0.5) * L
    rand_cat = treecorr.Catalog(x=xr, y=yr, x_units='arcmin', y_units='arcmin')
    rv = treecorr.NVCorrelation(bin_size=0.1, min_sep=1., max_sep=20., sep_units='arcmin',
                                verbose=1)
    rv.process(rand_cat, source_cat)
    print('rv.xi = ',rv.xi)
    xi, xi_im, varxi = nv.calculateXi(rv=rv)
    print('compensated xi = ',xi)
    print('compensated xi_im = ',xi_im)
    print('true_vrad = ',true_vr)
    print('ratio = ',xi / true_vr)
    print('diff = ',xi - true_vr)
    print('max diff = ',max(abs(xi - true_vr)))
    # It turns out this doesn't come out much better.  I think the imprecision is mostly just due
    # to the smallish number of lenses, not to edge effects
    np.testing.assert_allclose(xi, true_vr, rtol=0.1)
    np.testing.assert_allclose(xi_im, 0, atol=5.e-3)

    # Check that we get the same result using the corr2 function:
    config = treecorr.read_config('configs/nv.yaml')
    try:
        import fitsio
    except ImportError:
        pass
    else:
        lens_cat.write(os.path.join('data','nv_lens.fits'))
        source_cat.write(os.path.join('data','nv_source.fits'))
        rand_cat.write(os.path.join('data','nv_rand.fits'))
        config['verbose'] = 0
        config['precision'] = 8
        treecorr.corr2(config)
        corr2_output = np.genfromtxt(os.path.join('output','nv.out'), names=True, skip_header=1)
        print('nv.xi = ',nv.xi)
        print('xi = ',xi)
        print('from corr2 output = ',corr2_output['vR'])
        print('ratio = ',corr2_output['vR']/xi)
        print('diff = ',corr2_output['vR']-xi)
        np.testing.assert_allclose(corr2_output['vR'], xi, rtol=1.e-3)
        print('xi_im from corr2 output = ',corr2_output['vT'])
        np.testing.assert_allclose(corr2_output['vT'], 0., atol=4.e-3)

        # In the corr2 context, you can turn off the compensated bit, even if there are randoms
        # (e.g. maybe you only want randoms for some nn calculation, but not nv.)
        config['nv_statistic'] = 'simple'
        treecorr.corr2(config)
        corr2_output = np.genfromtxt(os.path.join('output','nv.out'), names=True, skip_header=1)
        xi_simple, _, _ = nv.calculateXi()
        np.testing.assert_equal(xi_simple, nv.xi)
        np.testing.assert_allclose(corr2_output['vR'], xi_simple, rtol=1.e-3)

    # Check the fits write option
    try:
        import fitsio
    except ImportError:
        pass
    else:
        out_file_name1 = os.path.join('output','nv_out1.fits')
        nv.write(out_file_name1)
        data = fitsio.read(out_file_name1)
        np.testing.assert_almost_equal(data['r_nom'], np.exp(nv.logr))
        np.testing.assert_almost_equal(data['meanr'], nv.meanr)
        np.testing.assert_almost_equal(data['meanlogr'], nv.meanlogr)
        np.testing.assert_almost_equal(data['vR'], nv.xi)
        np.testing.assert_almost_equal(data['vT'], nv.xi_im)
        np.testing.assert_almost_equal(data['sigma'], np.sqrt(nv.varxi))
        np.testing.assert_almost_equal(data['weight'], nv.weight)
        np.testing.assert_almost_equal(data['npairs'], nv.npairs)

        out_file_name2 = os.path.join('output','nv_out2.fits')
        nv.write(out_file_name2, rv=rv)
        data = fitsio.read(out_file_name2)
        np.testing.assert_almost_equal(data['r_nom'], np.exp(nv.logr))
        np.testing.assert_almost_equal(data['meanr'], nv.meanr)
        np.testing.assert_almost_equal(data['meanlogr'], nv.meanlogr)
        np.testing.assert_almost_equal(data['vR'], xi)
        np.testing.assert_almost_equal(data['vT'], xi_im)
        np.testing.assert_almost_equal(data['sigma'], np.sqrt(varxi))
        np.testing.assert_almost_equal(data['weight'], nv.weight)
        np.testing.assert_almost_equal(data['npairs'], nv.npairs)

        # Check the read function
        nv2 = treecorr.NVCorrelation(bin_size=0.1, min_sep=1., max_sep=20., sep_units='arcmin')
        nv2.read(out_file_name2)
        np.testing.assert_almost_equal(nv2.logr, nv.logr)
        np.testing.assert_almost_equal(nv2.meanr, nv.meanr)
        np.testing.assert_almost_equal(nv2.meanlogr, nv.meanlogr)
        np.testing.assert_almost_equal(nv2.xi, nv.xi)
        np.testing.assert_almost_equal(nv2.xi_im, nv.xi_im)
        np.testing.assert_almost_equal(nv2.varxi, nv.varxi)
        np.testing.assert_almost_equal(nv2.weight, nv.weight)
        np.testing.assert_almost_equal(nv2.npairs, nv.npairs)
        assert nv2.coords == nv.coords
        assert nv2.metric == nv.metric
        assert nv2.sep_units == nv.sep_units
        assert nv2.bin_type == nv.bin_type


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
    v0 = 0.05
    r0 = 10.
    L = 50. * r0
    rng = np.random.RandomState(8675309)
    xl = (rng.random_sample(nlens)-0.5) * L
    yl = (rng.random_sample(nlens)-0.5) * L
    xs = (rng.random_sample( (nsource,ncats) )-0.5) * L
    ys = (rng.random_sample( (nsource,ncats) )-0.5) * L
    v1 = np.zeros( (nsource,ncats) )
    v2 = np.zeros( (nsource,ncats) )
    w = rng.random_sample( (nsource,ncats) ) + 0.5
    for x,y in zip(xl,yl):
        dx = xs-x
        dy = ys-y
        r2 = dx**2 + dy**2
        r = np.sqrt(r2)
        vrad = v0 * np.exp(-0.5*r2/r0**2)
        v1 += vrad * dx/r
        v2 += vrad * dy/r

    lens_cat = treecorr.Catalog(x=xl, y=yl, x_units='arcmin', y_units='arcmin')
    source_cats = [ treecorr.Catalog(x=xs[:,k], y=ys[:,k], v1=v1[:,k], v2=v2[:,k], w=w[:,k],
                                     x_units='arcmin', y_units='arcmin') for k in range(ncats) ]
    full_source_cat = treecorr.Catalog(x=xs.flatten(), y=ys.flatten(), w=w.flatten(),
                                       v1=v1.flatten(), v2=v2.flatten(),
                                       x_units='arcmin', y_units='arcmin')

    t0 = time.time()
    for k in range(ncats):
        # These could each be done on different machines in a real world application.
        nv = treecorr.NVCorrelation(bin_size=0.1, min_sep=1., max_sep=25., sep_units='arcmin',
                                    verbose=1)
        # These should use process_cross, not process, since we don't want to call finalize.
        nv.process_cross(lens_cat, source_cats[k])
        nv.write(os.path.join('output','nv_piece_%d.fits'%k))

    pieces_nv = treecorr.NVCorrelation(bin_size=0.1, min_sep=1., max_sep=25., sep_units='arcmin')
    for k in range(ncats):
        nv = pieces_nv.copy()
        nv.read(os.path.join('output','nv_piece_%d.fits'%k))
        pieces_nv += nv
    varv = treecorr.calculateVarV(source_cats)
    pieces_nv.finalize(varv)
    t1 = time.time()
    print('time for piece-wise processing (including I/O) = ',t1-t0)

    full_nv = treecorr.NVCorrelation(bin_size=0.1, min_sep=1., max_sep=25., sep_units='arcmin',
                                     verbose=1)
    full_nv.process(lens_cat, full_source_cat)
    t2 = time.time()
    print('time for full processing = ',t2-t1)

    print('max error in meanr = ',np.max(pieces_nv.meanr - full_nv.meanr),)
    print('    max meanr = ',np.max(full_nv.meanr))
    print('max error in meanlogr = ',np.max(pieces_nv.meanlogr - full_nv.meanlogr),)
    print('    max meanlogr = ',np.max(full_nv.meanlogr))
    print('max error in weight = ',np.max(pieces_nv.weight - full_nv.weight),)
    print('    max weight = ',np.max(full_nv.weight))
    print('max error in xi = ',np.max(pieces_nv.xi - full_nv.xi),)
    print('    max xi = ',np.max(full_nv.xi))
    print('max error in xi_im = ',np.max(pieces_nv.xi_im - full_nv.xi_im),)
    print('    max xi_im = ',np.max(full_nv.xi_im))
    print('max error in varxi = ',np.max(pieces_nv.varxi - full_nv.varxi),)
    print('    max varxi = ',np.max(full_nv.varxi))
    np.testing.assert_allclose(pieces_nv.meanr, full_nv.meanr, rtol=2.e-3)
    np.testing.assert_allclose(pieces_nv.meanlogr, full_nv.meanlogr, atol=2.e-3)
    np.testing.assert_allclose(pieces_nv.weight, full_nv.weight, rtol=3.e-2)
    np.testing.assert_allclose(pieces_nv.xi, full_nv.xi, rtol=0.1)
    np.testing.assert_allclose(pieces_nv.xi_im, full_nv.xi_im, atol=2.e-3)
    np.testing.assert_allclose(pieces_nv.varxi, full_nv.varxi, rtol=3.e-2)

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
                                      v1=v1.flatten(), v2=v2.flatten(),
                                      wpos=w.flatten(), w=w2[k].flatten(),
                                      x_units='arcmin', y_units='arcmin') for k in range(ncats) ]

    t3 = time.time()
    nv2 = [ full_nv.copy() for k in range(ncats) ]
    for k in range(ncats):
        nv2[k].clear()
        nv2[k].process_cross(lens_cat, source_cats2[k])

    pieces_nv2 = full_nv.copy()
    pieces_nv2.clear()
    for k in range(ncats):
        pieces_nv2 += nv2[k]
    pieces_nv2.finalize(varv)
    t4 = time.time()
    print('time for zero-weight piece-wise processing = ',t4-t3)

    print('max error in meanr = ',np.max(pieces_nv2.meanr - full_nv.meanr),)
    print('    max meanr = ',np.max(full_nv.meanr))
    print('max error in meanlogr = ',np.max(pieces_nv2.meanlogr - full_nv.meanlogr),)
    print('    max meanlogr = ',np.max(full_nv.meanlogr))
    print('max error in weight = ',np.max(pieces_nv2.weight - full_nv.weight),)
    print('    max weight = ',np.max(full_nv.weight))
    print('max error in xi = ',np.max(pieces_nv2.xi - full_nv.xi),)
    print('    max xi = ',np.max(full_nv.xi))
    print('max error in xi_im = ',np.max(pieces_nv2.xi_im - full_nv.xi_im),)
    print('    max xi_im = ',np.max(full_nv.xi_im))
    print('max error in varxi = ',np.max(pieces_nv2.varxi - full_nv.varxi),)
    print('    max varxi = ',np.max(full_nv.varxi))
    np.testing.assert_allclose(pieces_nv2.meanr, full_nv.meanr, rtol=1.e-7)
    np.testing.assert_allclose(pieces_nv2.meanlogr, full_nv.meanlogr, rtol=1.e-7)
    np.testing.assert_allclose(pieces_nv2.weight, full_nv.weight, rtol=1.e-7)
    np.testing.assert_allclose(pieces_nv2.xi, full_nv.xi, rtol=1.e-7)
    np.testing.assert_allclose(pieces_nv2.xi_im, full_nv.xi_im, atol=1.e-10)
    np.testing.assert_allclose(pieces_nv2.varxi, full_nv.varxi, rtol=1.e-7)

    # Try this with corr2
    lens_cat.write(os.path.join('data','nv_wpos_lens.fits'))
    for i, sc in enumerate(source_cats2):
        sc.write(os.path.join('data','nv_wpos_source%d.fits'%i))
    config = treecorr.read_config('configs/nv_wpos.yaml')
    config['verbose'] = 0
    treecorr.corr2(config)
    data = fitsio.read(config['nv_file_name'])
    print('data.dtype = ',data.dtype)
    np.testing.assert_allclose(data['meanr'], full_nv.meanr, rtol=1.e-7)
    np.testing.assert_allclose(data['meanlogr'], full_nv.meanlogr, rtol=1.e-7)
    np.testing.assert_allclose(data['weight'], full_nv.weight, rtol=1.e-7)
    np.testing.assert_allclose(data['vR'], full_nv.xi, rtol=1.e-7)
    np.testing.assert_allclose(data['vT'], full_nv.xi_im, atol=1.e-10)
    np.testing.assert_allclose(data['sigma']**2, full_nv.varxi, rtol=1.e-7)


@timer
def test_varxi():
    # Test that varxi is correct (or close) based on actual variance of many runs.

    # Signal doesn't matter much.  Use the one from test_gg.
    v0 = 0.05
    r0 = 10.
    L = 10 * r0
    rng = np.random.RandomState(8675309)

    # Note: to get a good estimate of var(xi), you need a lot of runs.  The number of
    # runs matters much more than the number of galaxies for getting this to pass.
    # In addition, I found that the variance was significantly underestimated when there
    # were lots of lenses.  I guess because there were multiple lenses that paired with the
    # same sources in a given bin, which increased the variance of the mean <g>.
    # So there might be some adjustment that would help improve the estimate of varxi,
    # but at least this unit test shows that it's fairly accurate for *some* scenario.
    nsource = 1000
    nrand = 10
    nruns = 50000
    lens = treecorr.Catalog(x=[0], y=[0])

    file_name = 'data/test_varxi_nv.npz'
    print(file_name)
    if not os.path.isfile(file_name):
        all_nvs = []
        all_rvs = []
        for run in range(nruns):
            print(f'{run}/{nruns}')
            x2 = (rng.random_sample(nsource)-0.5) * L
            y2 = (rng.random_sample(nsource)-0.5) * L
            x3 = (rng.random_sample(nrand)-0.5) * L
            y3 = (rng.random_sample(nrand)-0.5) * L

            r2 = (x2**2 + y2**2)/r0**2
            v1 = v0 * np.exp(-r2/2.) * x2/r0
            v2 = v0 * np.exp(-r2/2.) * y2/r0
            # This time, add some shape noise (different each run).
            v1 += rng.normal(0, 0.1, size=nsource)
            v2 += rng.normal(0, 0.1, size=nsource)
            # Varied weights are hard, but at least check that non-unit weights work correctly.
            w = np.ones_like(x2) * 5

            source = treecorr.Catalog(x=x2, y=y2, w=w, v1=v1, v2=v2)
            rand = treecorr.Catalog(x=x3, y=y3)
            nv = treecorr.NVCorrelation(bin_size=0.2, min_sep=10., max_sep=30.)
            rv = treecorr.NVCorrelation(bin_size=0.2, min_sep=10., max_sep=30.)
            nv.process(lens, source)
            rv.process(rand, source)
            all_nvs.append(nv)
            all_rvs.append(rv)

        all_xis = [nv.calculateXi() for nv in all_nvs]
        var_xi_1 = np.var([xi[0] for xi in all_xis], axis=0)
        mean_varxi_1 = np.mean([xi[2] for xi in all_xis], axis=0)

        all_xis = [nv.calculateXi(rv=rv) for (nv,rv) in zip(all_nvs, all_rvs)]
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
    v1 = v0 * np.exp(-r2/2.) * x2/r0
    v2 = v0 * np.exp(-r2/2.) * y2/r0
    # This time, add some shape noise (different each run).
    v1 += rng.normal(0, 0.1, size=nsource)
    v2 += rng.normal(0, 0.1, size=nsource)
    # Varied weights are hard, but at least check that non-unit weights work correctly.
    w = np.ones_like(x2) * 5

    source = treecorr.Catalog(x=x2, y=y2, w=w, v1=v1, v2=v2)
    rand = treecorr.Catalog(x=x3, y=y3)
    nv = treecorr.NVCorrelation(bin_size=0.2, min_sep=10., max_sep=30.)
    rv = treecorr.NVCorrelation(bin_size=0.2, min_sep=10., max_sep=30.)
    nv.process(lens, source)
    rv.process(rand, source)

    print('single run:')
    print('Uncompensated')
    print('ratio = ',nv.varxi / var_xi_1)
    print('max relerr for xi = ',np.max(np.abs((nv.varxi - var_xi_1)/var_xi_1)))
    np.testing.assert_allclose(nv.varxi, var_xi_1, rtol=0.6)

    xi, xi_im, varxi = nv.calculateXi(rv=rv)
    print('Compensated')
    print('ratio = ',varxi / var_xi_2)
    print('max relerr for xi = ',np.max(np.abs((varxi - var_xi_2)/var_xi_2)))
    np.testing.assert_allclose(varxi, var_xi_2, rtol=0.5)


if __name__ == '__main__':
    test_direct()
    test_direct_spherical()
    test_single()
    test_spherical()
    test_nv()
    test_pieces()
    test_varxi()