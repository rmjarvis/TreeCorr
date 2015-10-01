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

from numpy import sin, cos, tan, arcsin, arccos, arctan, arctan2, pi

def test_single():
    # Use kappa(r) = kappa0 exp(-r^2/2r0^2) (1-r^2/2r0^2) around a single lens

    nsource = 1000000
    kappa0 = 0.05
    r0 = 10.
    L = 5. * r0
    numpy.random.seed(8675309)
    x = (numpy.random.random_sample(nsource)-0.5) * L
    y = (numpy.random.random_sample(nsource)-0.5) * L
    r2 = (x**2 + y**2)
    k = kappa0 * numpy.exp(-0.5*r2/r0**2) * (1.-0.5*r2/r0**2)

    lens_cat = treecorr.Catalog(x=[0], y=[0], x_units='arcmin', y_units='arcmin')
    source_cat = treecorr.Catalog(x=x, y=y, k=k, x_units='arcmin', y_units='arcmin')
    nk = treecorr.NKCorrelation(bin_size=0.1, min_sep=1., max_sep=25., sep_units='arcmin',
                                verbose=2)
    nk.process(lens_cat, source_cat)

    r = nk.meanr
    true_k = kappa0 * numpy.exp(-0.5*r**2/r0**2) * (1.-0.5*r**2/r0**2)

    print('nk.xi = ',nk.xi)
    print('true_kappa = ',true_k)
    print('ratio = ',nk.xi / true_k)
    print('diff = ',nk.xi - true_k)
    print('max diff = ',max(abs(nk.xi - true_k)))
    assert max(abs(nk.xi - true_k)) < 4.e-4

    # Check that we get the same result using the corr2 executable:
    if __name__ == '__main__':
        lens_cat.write(os.path.join('data','nk_single_lens.dat'))
        source_cat.write(os.path.join('data','nk_single_source.dat'))
        import subprocess
        p = subprocess.Popen( ["corr2","nk_single.params"] )
        p.communicate()
        corr2_output = numpy.genfromtxt(os.path.join('output','nk_single.out'), names=True)
        print('nk.xi = ',nk.xi)
        print('from corr2 output = ',corr2_output['kappa'])
        print('ratio = ',corr2_output['kappa']/nk.xi)
        print('diff = ',corr2_output['kappa']-nk.xi)
        numpy.testing.assert_almost_equal(corr2_output['kappa']/nk.xi, 1., decimal=3)


def test_nk():
    # Use kappa(r) = kappa0 exp(-r^2/2r0^2) (1-r^2/2r0^2) around many lenses.

    nlens = 1000
    nsource = 100000
    kappa0 = 0.05
    r0 = 10.
    L = 50. * r0
    numpy.random.seed(8675309)
    xl = (numpy.random.random_sample(nlens)-0.5) * L
    yl = (numpy.random.random_sample(nlens)-0.5) * L
    xs = (numpy.random.random_sample(nsource)-0.5) * L
    ys = (numpy.random.random_sample(nsource)-0.5) * L
    k = numpy.zeros( (nsource,) )
    for x,y in zip(xl,yl):
        dx = xs-x
        dy = ys-y
        r2 = dx**2 + dy**2
        k += kappa0 * numpy.exp(-0.5*r2/r0**2) * (1.-0.5*r2/r0**2)

    lens_cat = treecorr.Catalog(x=xl, y=yl, x_units='arcmin', y_units='arcmin')
    source_cat = treecorr.Catalog(x=xs, y=ys, k=k, x_units='arcmin', y_units='arcmin')
    nk = treecorr.NKCorrelation(bin_size=0.1, min_sep=1., max_sep=25., sep_units='arcmin',
                                verbose=2)
    nk.process(lens_cat, source_cat)

    # log(<R>) != <logR>, but it should be close:
    print('meanlogr - log(meanr) = ',nk.meanlogr - numpy.log(nk.meanr))
    numpy.testing.assert_almost_equal(nk.meanlogr, numpy.log(nk.meanr), decimal=3)

    r = nk.meanr
    true_k = kappa0 * numpy.exp(-0.5*r**2/r0**2) * (1.-0.5*r**2/r0**2)

    print('nk.xi = ',nk.xi)
    print('true_kappa = ',true_k)
    print('ratio = ',nk.xi / true_k)
    print('diff = ',nk.xi - true_k)
    print('max diff = ',max(abs(nk.xi - true_k)))
    assert max(abs(nk.xi - true_k)) < 5.e-3

    nrand = nlens * 13
    xr = (numpy.random.random_sample(nrand)-0.5) * L
    yr = (numpy.random.random_sample(nrand)-0.5) * L
    rand_cat = treecorr.Catalog(x=xr, y=yr, x_units='arcmin', y_units='arcmin')
    rk = treecorr.NKCorrelation(bin_size=0.1, min_sep=1., max_sep=25., sep_units='arcmin',
                                verbose=2)
    rk.process(rand_cat, source_cat)
    print('rk.xi = ',rk.xi)
    xi, varxi = nk.calculateXi(rk)
    print('compensated xi = ',xi)
    print('true_kappa = ',true_k)
    print('ratio = ',xi / true_k)
    print('diff = ',xi - true_k)
    print('max diff = ',max(abs(xi - true_k)))
    # It turns out this doesn't come out much better.  I think the imprecision is mostly just due
    # to the smallish number of lenses, not to edge effects
    assert max(abs(xi - true_k)) < 5.e-3

    # Check that we get the same result using the corr2 executable:
    if __name__ == '__main__':
        lens_cat.write(os.path.join('data','nk_lens.dat'))
        source_cat.write(os.path.join('data','nk_source.dat'))
        rand_cat.write(os.path.join('data','nk_rand.dat'))
        import subprocess
        p = subprocess.Popen( ["corr2","nk.params"] )
        p.communicate()
        corr2_output = numpy.genfromtxt(os.path.join('output','nk.out'), names=True)
        print('nk.xi = ',nk.xi)
        print('xi = ',xi)
        print('from corr2 output = ',corr2_output['kappa'])
        print('ratio = ',corr2_output['kappa']/xi)
        print('diff = ',corr2_output['kappa']-xi)
        numpy.testing.assert_almost_equal(corr2_output['kappa']/xi, 1., decimal=3)

    # Check the fits write option
    out_file_name1 = os.path.join('output','nk_out1.fits')
    nk.write(out_file_name1)
    try:
        import fitsio
        data = fitsio.read(out_file_name1)
        numpy.testing.assert_almost_equal(data['R_nom'], numpy.exp(nk.logr))
        numpy.testing.assert_almost_equal(data['meanR'], nk.meanr)
        numpy.testing.assert_almost_equal(data['meanlogR'], nk.meanlogr)
        numpy.testing.assert_almost_equal(data['kappa'], nk.xi)
        numpy.testing.assert_almost_equal(data['sigma'], numpy.sqrt(nk.varxi))
        numpy.testing.assert_almost_equal(data['weight'], nk.weight)
        numpy.testing.assert_almost_equal(data['npairs'], nk.npairs)
    except ImportError:
        print('Unable to import fitsio.  Skipping fits tests.')

    out_file_name2 = os.path.join('output','nk_out2.fits')
    nk.write(out_file_name2, rk)
    try:
        import fitsio
        data = fitsio.read(out_file_name2)
        numpy.testing.assert_almost_equal(data['R_nom'], numpy.exp(nk.logr))
        numpy.testing.assert_almost_equal(data['meanR'], nk.meanr)
        numpy.testing.assert_almost_equal(data['meanlogR'], nk.meanlogr)
        numpy.testing.assert_almost_equal(data['kappa'], xi)
        numpy.testing.assert_almost_equal(data['sigma'], numpy.sqrt(varxi))
        numpy.testing.assert_almost_equal(data['weight'], nk.weight)
        numpy.testing.assert_almost_equal(data['npairs'], nk.npairs)
    except ImportError:
        print('Unable to import fitsio.  Skipping fits tests.')
    
    # Check the read function
    nk2 = treecorr.NKCorrelation(bin_size=0.1, min_sep=1., max_sep=25., sep_units='arcmin')
    nk2.read(out_file_name1)
    numpy.testing.assert_almost_equal(nk2.logr, nk.logr)
    numpy.testing.assert_almost_equal(nk2.meanr, nk.meanr)
    numpy.testing.assert_almost_equal(nk2.meanlogr, nk.meanlogr)
    numpy.testing.assert_almost_equal(nk2.xi, nk.xi)
    numpy.testing.assert_almost_equal(nk2.varxi, nk.varxi)
    numpy.testing.assert_almost_equal(nk2.weight, nk.weight)
    numpy.testing.assert_almost_equal(nk2.npairs, nk.npairs)


if __name__ == '__main__':
    test_single()
    test_nk()
