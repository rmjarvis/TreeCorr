# Copyright (c) 2003-2015 by Mike Jarvis
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
import numpy
import treecorr
import os
import fitsio

from test_helper import get_script_name

def test_constant():
    # A fairly trivial test is to use a constant value of kappa everywhere.

    ngal = 100000
    A = 0.05
    L = 100.
    numpy.random.seed(8675309)
    x = (numpy.random.random_sample(ngal)-0.5) * L
    y = (numpy.random.random_sample(ngal)-0.5) * L
    kappa = A * numpy.ones(ngal)

    cat = treecorr.Catalog(x=x, y=y, k=kappa, x_units='arcmin', y_units='arcmin')
    kk = treecorr.KKCorrelation(bin_size=0.1, min_sep=0.1, max_sep=10., sep_units='arcmin')
    kk.process(cat)
    print('kk.xi = ',kk.xi)
    numpy.testing.assert_almost_equal(kk.xi, A**2, decimal=10)

    # Now add some noise to the values. It should still work, but at slightly lower accuracy.
    kappa += 0.001 * (numpy.random.random_sample(ngal)-0.5)
    cat = treecorr.Catalog(x=x, y=y, k=kappa, x_units='arcmin', y_units='arcmin')
    kk.process(cat)
    print('kk.xi = ',kk.xi)
    numpy.testing.assert_almost_equal(kk.xi, A**2, decimal=6)


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

    ngal = 1000000
    A = 0.05
    s = 10.
    L = 30. * s  # Not infinity, so this introduces some error.  Our integrals were to infinity.
    numpy.random.seed(8675309)
    x = (numpy.random.random_sample(ngal)-0.5) * L
    y = (numpy.random.random_sample(ngal)-0.5) * L
    r2 = (x**2 + y**2)/s**2
    kappa = A * numpy.exp(-r2/2.)

    cat = treecorr.Catalog(x=x, y=y, k=kappa, x_units='arcmin', y_units='arcmin')
    kk = treecorr.KKCorrelation(bin_size=0.1, min_sep=1., max_sep=50., sep_units='arcmin',
                                verbose=1)
    kk.process(cat)

    # log(<R>) != <logR>, but it should be close:
    print('meanlogr - log(meanr) = ',kk.meanlogr - numpy.log(kk.meanr))
    numpy.testing.assert_almost_equal(kk.meanlogr, numpy.log(kk.meanr), decimal=3)

    r = kk.meanr
    true_xi = numpy.pi * A**2 * (s/L)**2 * numpy.exp(-0.25*r**2/s**2)
    print('kk.xi = ',kk.xi)
    print('true_xi = ',true_xi)
    print('ratio = ',kk.xi / true_xi)
    print('diff = ',kk.xi - true_xi)
    print('max diff = ',max(abs(kk.xi - true_xi)))
    assert max(abs(kk.xi - true_xi)) < 5.e-7

    # It should also work as a cross-correlation of this cat with itself
    kk.process(cat,cat)
    numpy.testing.assert_almost_equal(kk.meanlogr, numpy.log(kk.meanr), decimal=3)
    assert max(abs(kk.xi - true_xi)) < 5.e-7

    # Check that we get the same result using the corr2 executable:
    if __name__ == '__main__':
        cat.write(os.path.join('data','kk.dat'))
        import subprocess
        corr2_exe = get_script_name('corr2')
        p = subprocess.Popen( [corr2_exe,"kk.yaml"] )
        p.communicate()
        corr2_output = numpy.genfromtxt(os.path.join('output','kk.out'), names=True)
        print('kk.xi = ',kk.xi)
        print('from corr2 output = ',corr2_output['xi'])
        print('ratio = ',corr2_output['xi']/kk.xi)
        print('diff = ',corr2_output['xi']-kk.xi)
        numpy.testing.assert_almost_equal(corr2_output['xi']/kk.xi, 1., decimal=3)

    # Check the fits write option
    out_file_name = os.path.join('output','kk_out.fits')
    kk.write(out_file_name)
    data = fitsio.read(out_file_name)
    numpy.testing.assert_almost_equal(data['R_nom'], numpy.exp(kk.logr))
    numpy.testing.assert_almost_equal(data['meanR'], kk.meanr)
    numpy.testing.assert_almost_equal(data['meanlogR'], kk.meanlogr)
    numpy.testing.assert_almost_equal(data['xi'], kk.xi)
    numpy.testing.assert_almost_equal(data['sigma_xi'], numpy.sqrt(kk.varxi))
    numpy.testing.assert_almost_equal(data['weight'], kk.weight)
    numpy.testing.assert_almost_equal(data['npairs'], kk.npairs)

    # Check the read function
    kk2 = treecorr.KKCorrelation(bin_size=0.1, min_sep=1., max_sep=100., sep_units='arcmin')
    kk2.read(out_file_name)
    numpy.testing.assert_almost_equal(kk2.logr, kk.logr)
    numpy.testing.assert_almost_equal(kk2.meanr, kk.meanr)
    numpy.testing.assert_almost_equal(kk2.meanlogr, kk.meanlogr)
    numpy.testing.assert_almost_equal(kk2.xi, kk.xi)
    numpy.testing.assert_almost_equal(kk2.varxi, kk.varxi)
    numpy.testing.assert_almost_equal(kk2.weight, kk.weight)
    numpy.testing.assert_almost_equal(kk2.npairs, kk.npairs)

def test_large_scale():
    # Test very large scales, comparing Arc, Euclidean (spherical), and Euclidean (3d)

    # Distribute points uniformly in all directions.
    ngal = 50000
    s = 1.
    numpy.random.seed(8675309)
    x = numpy.random.normal(0, s, (ngal,) )
    y = numpy.random.normal(0, s, (ngal,) )
    z = numpy.random.normal(0, s, (ngal,) )
    r = numpy.sqrt( x*x + y*y + z*z )
    dec = numpy.arcsin(z/r)
    ra = numpy.arctan2(y,x)
    r = numpy.ones_like(x)  # Overwrite with all r=1

    # Use x for "kappa" so there's a strong real correlation function
    cat1 = treecorr.Catalog(ra=ra, dec=dec, k=x, ra_units='rad', dec_units='rad')
    cat2 = treecorr.Catalog(ra=ra, dec=dec, k=x, r=r, ra_units='rad', dec_units='rad')

    config = {
        'min_sep' : 0.01,
        'max_sep' : 1.57,
        'nbins' : 50,
        'bin_slop' : 0.3,
        'verbose' : 1
    }
    dd_sphere = treecorr.KKCorrelation(config)
    dd_chord = treecorr.KKCorrelation(config)
    dd_euclid = treecorr.KKCorrelation(config)
    dd_euclid.process(cat1, metric='Euclidean')
    dd_sphere.process(cat1, metric='Arc')
    dd_chord.process(cat2, metric='Euclidean')

    for tag in [ 'rnom', 'logr', 'meanr', 'meanlogr', 'npairs', 'xi' ]:
        for name, dd in [ ('Euclid', dd_euclid), ('Sphere', dd_sphere), ('Chord', dd_chord) ]:
            print(name, tag, '=', getattr(dd,tag))

    # rnom and logr should be identical
    numpy.testing.assert_array_equal(dd_sphere.rnom, dd_euclid.rnom)
    numpy.testing.assert_array_equal(dd_chord.rnom, dd_euclid.rnom)
    numpy.testing.assert_array_equal(dd_sphere.logr, dd_euclid.logr)
    numpy.testing.assert_array_equal(dd_chord.logr, dd_euclid.logr)

    # meanr should be similar for sphere and chord, but euclid is larger, since the chord
    # distances have been scaled up to the real great circle distances
    numpy.testing.assert_allclose(dd_sphere.meanr, dd_chord.meanr, rtol=1.e-3)
    numpy.testing.assert_allclose(dd_chord.meanr[:24], dd_euclid.meanr[:24], rtol=1.e-3)
    numpy.testing.assert_array_less(dd_chord.meanr[24:], dd_euclid.meanr[24:])
    numpy.testing.assert_allclose(dd_sphere.meanlogr, dd_chord.meanlogr, atol=2.e-2)
    numpy.testing.assert_allclose(dd_chord.meanlogr[:24], dd_euclid.meanlogr[:24], atol=2.e-2)
    numpy.testing.assert_array_less(dd_chord.meanlogr[24:], dd_euclid.meanlogr[24:])

    # npairs is basically the same for chord and euclid since the only difference there comes from
    # differences in where they cut off the tree traversal, so the number of pairs is almost equal,
    # even though the separations in each bin are given a different nominal distance.
    # Sphere is smaller than both at all scales, since it is measuring the correlation
    # function on larger real scales at each position.
    print('diff = ',(dd_chord.npairs-dd_euclid.npairs)/dd_euclid.npairs)
    print('max = ',numpy.max(numpy.abs((dd_chord.npairs-dd_euclid.npairs)/dd_euclid.npairs)))
    numpy.testing.assert_allclose(dd_chord.npairs, dd_euclid.npairs, rtol=1.e-3)
    numpy.testing.assert_allclose(dd_sphere.npairs[:24], dd_euclid.npairs[:24], rtol=2.e-3)
    numpy.testing.assert_array_less(dd_sphere.npairs[24:], dd_euclid.npairs[24:])

    # Renormalize by the actual spacing in log(r)
    renorm_euclid = dd_euclid.npairs / numpy.gradient(dd_euclid.meanlogr)
    renorm_sphere = dd_sphere.npairs / numpy.gradient(dd_sphere.meanlogr)
    # Then interpolate the euclid results to the values of the sphere distances
    interp_euclid = numpy.interp(dd_sphere.meanlogr, dd_euclid.meanlogr, renorm_euclid)
    # Matches at higher precision now over the same range
    print('interp_euclid = ',interp_euclid)
    print('renorm_sphere = ',renorm_sphere)
    print('new diff = ',(renorm_sphere-interp_euclid)/renorm_sphere)
    print('max = ',numpy.max(numpy.abs((renorm_sphere-interp_euclid)/renorm_sphere)))
    numpy.testing.assert_allclose(renorm_sphere[:24], interp_euclid[:24], rtol=4.e-4)

    # And almost the full range at the same precision.
    numpy.testing.assert_allclose(renorm_sphere[:43], interp_euclid[:43], rtol=2.e-3)
    numpy.testing.assert_allclose(renorm_sphere, interp_euclid, rtol=1.e-2)

    # The xi values are similar.  The euclid and chord values start out basically identical,
    # but the distances are different.  The euclid and the sphere are actually the same function
    # so they match when rescaled to have the same distance values.
    print('diff euclid, chord = ',(dd_chord.xi-dd_euclid.xi)/dd_euclid.xi)
    print('max = ',numpy.max(numpy.abs((dd_chord.xi-dd_euclid.xi)/dd_euclid.xi)))
    numpy.testing.assert_allclose(dd_chord.xi[:-1], dd_euclid.xi[:-1], rtol=1.e-3)
    numpy.testing.assert_allclose(dd_chord.xi, dd_euclid.xi, rtol=3.e-3)

    interp_euclid = numpy.interp(dd_sphere.meanlogr, dd_euclid.meanlogr, dd_euclid.xi)
    print('interp_euclid = ',interp_euclid)
    print('sphere.xi = ',dd_sphere.xi)
    print('diff interp euclid, sphere = ',(dd_sphere.xi-interp_euclid))
    print('max = ',numpy.max(numpy.abs((dd_sphere.xi-interp_euclid))))
    numpy.testing.assert_allclose(dd_sphere.xi[:36], interp_euclid[:36], atol=3.e-4)
    numpy.testing.assert_allclose(dd_sphere.xi[:44], interp_euclid[:44], atol=1.e-3)
    numpy.testing.assert_allclose(dd_sphere.xi, interp_euclid, atol=3.e-3)


if __name__ == '__main__':
    test_constant()
    test_kk()
    test_large_scale()
