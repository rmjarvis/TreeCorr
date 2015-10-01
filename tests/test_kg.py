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

def test_single():
    # Use gamma_t(r) = gamma0 exp(-r^2/2r0^2) around a single lens
    # i.e. gamma(r) = -gamma0 exp(-r^2/2r0^2) (x+iy)^2/r^2

    nsource = 1000000
    gamma0 = 0.05
    kappa = 0.23
    r0 = 10.
    L = 5. * r0
    numpy.random.seed(8675309)
    x = (numpy.random.random_sample(nsource)-0.5) * L
    y = (numpy.random.random_sample(nsource)-0.5) * L
    r2 = (x**2 + y**2)
    gammat = gamma0 * numpy.exp(-0.5*r2/r0**2)
    g1 = -gammat * (x**2-y**2)/r2
    g2 = -gammat * (2.*x*y)/r2

    lens_cat = treecorr.Catalog(x=[0], y=[0], k=[kappa],  x_units='arcmin', y_units='arcmin')
    source_cat = treecorr.Catalog(x=x, y=y, g1=g1, g2=g2, x_units='arcmin', y_units='arcmin')
    kg = treecorr.KGCorrelation(bin_size=0.1, min_sep=1., max_sep=25., sep_units='arcmin',
                                verbose=2)
    kg.process(lens_cat, source_cat)

    # log(<R>) != <logR>, but it should be close:
    print('meanlogr - log(meanr) = ',kg.meanlogr - numpy.log(kg.meanr))
    numpy.testing.assert_almost_equal(kg.meanlogr, numpy.log(kg.meanr), decimal=3)

    r = kg.meanr
    true_kgt = kappa * gamma0 * numpy.exp(-0.5*r**2/r0**2)

    print('kg.xi = ',kg.xi)
    print('kg.xi_im = ',kg.xi_im)
    print('true_gammat = ',true_kgt)
    print('ratio = ',kg.xi / true_kgt)
    print('diff = ',kg.xi - true_kgt)
    print('max diff = ',max(abs(kg.xi - true_kgt)))
    assert max(abs(kg.xi - true_kgt)) < 4.e-4
    assert max(abs(kg.xi_im)) < 3.e-5

    # Check that we get the same result using the corr2 executable:
    if __name__ == '__main__':
        lens_cat.write(os.path.join('data','kg_single_lens.dat'))
        source_cat.write(os.path.join('data','kg_single_source.dat'))
        import subprocess
        p = subprocess.Popen( ["corr2","kg_single.params"] )
        p.communicate()
        corr2_output = numpy.genfromtxt(os.path.join('output','kg_single.out'),names=True)
        print('kg.xi = ',kg.xi)
        print('from corr2 output = ',corr2_output['kgamT'])
        print('ratio = ',corr2_output['kgamT']/kg.xi)
        print('diff = ',corr2_output['kgamT']-kg.xi)
        numpy.testing.assert_almost_equal(corr2_output['kgamT']/kg.xi, 1., decimal=3)

        print('xi_im from corr2 output = ',corr2_output['kgamX'])
        assert max(abs(corr2_output['kgamX'])) < 3.e-5


def test_pairwise():
    # Test the same profile, but with the pairwise calcualtion:
    nsource = 1000000
    gamma0 = 0.05
    kappa = 0.23
    r0 = 10.
    L = 5. * r0
    numpy.random.seed(8675309)
    x = (numpy.random.random_sample(nsource)-0.5) * L
    y = (numpy.random.random_sample(nsource)-0.5) * L
    r2 = (x**2 + y**2)
    gammat = gamma0 * numpy.exp(-0.5*r2/r0**2)
    g1 = -gammat * (x**2-y**2)/r2
    g2 = -gammat * (2.*x*y)/r2

    dx = (numpy.random.random_sample(nsource)-0.5) * L
    dx = (numpy.random.random_sample(nsource)-0.5) * L
    k = kappa * numpy.ones(nsource)

    lens_cat = treecorr.Catalog(x=dx, y=dx, k=k, x_units='arcmin', y_units='arcmin')
    source_cat = treecorr.Catalog(x=x+dx, y=y+dx, g1=g1, g2=g2, x_units='arcmin', y_units='arcmin')
    kg = treecorr.KGCorrelation(bin_size=0.1, min_sep=1., max_sep=25., sep_units='arcmin',
                                verbose=2, pairwise=True)
    kg.process(lens_cat, source_cat)

    r = kg.meanr
    true_kgt = kappa * gamma0 * numpy.exp(-0.5*r**2/r0**2)

    print('kg.xi = ',kg.xi)
    print('kg.xi_im = ',kg.xi_im)
    print('true_gammat = ',true_kgt)
    print('ratio = ',kg.xi / true_kgt)
    print('diff = ',kg.xi - true_kgt)
    print('max diff = ',max(abs(kg.xi - true_kgt)))
    assert max(abs(kg.xi - true_kgt)) < 4.e-4
    assert max(abs(kg.xi_im)) < 3.e-5

    # Check that we get the same result using the corr2 executable:
    if __name__ == '__main__':
        lens_cat.write(os.path.join('data','kg_pairwise_lens.dat'))
        source_cat.write(os.path.join('data','kg_pairwise_source.dat'))
        import subprocess
        p = subprocess.Popen( ["corr2","kg_pairwise.params"] )
        p.communicate()
        corr2_output = numpy.genfromtxt(os.path.join('output','kg_pairwise.out'),names=True)
        print('kg.xi = ',kg.xi)
        print('from corr2 output = ',corr2_output['kgamT'])
        print('ratio = ',corr2_output['kgamT']/kg.xi)
        print('diff = ',corr2_output['kgamT']-kg.xi)
        numpy.testing.assert_almost_equal(corr2_output['kgamT']/kg.xi, 1., decimal=3)

        print('xi_im from corr2 output = ',corr2_output['kgamX'])
        assert max(abs(corr2_output['kgamX'])) < 3.e-5


def test_kg():
    # Use gamma_t(r) = gamma0 exp(-r^2/2r0^2) around a bunch of foreground lenses.
    # i.e. gamma(r) = -gamma0 exp(-r^2/2r0^2) (x+iy)^2/r^2

    nlens = 1000
    nsource = 100000
    gamma0 = 0.05
    r0 = 10.
    L = 50. * r0
    numpy.random.seed(8675309)
    xl = (numpy.random.random_sample(nlens)-0.5) * L
    yl = (numpy.random.random_sample(nlens)-0.5) * L
    xs = (numpy.random.random_sample(nsource)-0.5) * L
    ys = (numpy.random.random_sample(nsource)-0.5) * L
    g1 = numpy.zeros( (nsource,) )
    g2 = numpy.zeros( (nsource,) )
    kl = numpy.random.normal(0.23, 0.05, (nlens,) )
    for x,y,k in zip(xl,yl,kl):
        dx = xs-x
        dy = ys-y
        r2 = dx**2 + dy**2
        gammat = gamma0 * numpy.exp(-0.5*r2/r0**2) / k
        g1 += -gammat * (dx**2-dy**2)/r2
        g2 += -gammat * (2.*dx*dy)/r2

    lens_cat = treecorr.Catalog(x=xl, y=yl, k=kl, x_units='arcmin', y_units='arcmin')
    source_cat = treecorr.Catalog(x=xs, y=ys, g1=g1, g2=g2, x_units='arcmin', y_units='arcmin')
    kg = treecorr.KGCorrelation(bin_size=0.1, min_sep=1., max_sep=25., sep_units='arcmin',
                                verbose=2)
    kg.process(lens_cat, source_cat)

    r = kg.meanr
    true_gt = gamma0 * numpy.exp(-0.5*r**2/r0**2)

    print('kg.xi = ',kg.xi)
    print('kg.xi_im = ',kg.xi_im)
    print('true_gammat = ',true_gt)
    print('ratio = ',kg.xi / true_gt)
    print('diff = ',kg.xi - true_gt)
    print('max diff = ',max(abs(kg.xi - true_gt)))
    assert max(abs(kg.xi - true_gt)) < 4.e-3
    assert max(abs(kg.xi_im)) < 4.e-3

    # Check that we get the same result using the corr2 executable:
    if __name__ == '__main__':
        lens_cat.write(os.path.join('data','kg_lens.dat'))
        source_cat.write(os.path.join('data','kg_source.dat'))
        import subprocess
        p = subprocess.Popen( ["corr2","kg.params"] )
        p.communicate()
        corr2_output = numpy.genfromtxt(os.path.join('output','kg.out'),names=True)
        print('kg.xi = ',kg.xi)
        print('from corr2 output = ',corr2_output['kgamT'])
        print('ratio = ',corr2_output['kgamT']/kg.xi)
        print('diff = ',corr2_output['kgamT']-kg.xi)
        numpy.testing.assert_almost_equal(corr2_output['kgamT']/kg.xi, 1., decimal=3)

        print('xi_im from corr2 output = ',corr2_output['kgamX'])
        assert max(abs(corr2_output['kgamX'])) < 4.e-3

    # Check the fits write option
    out_file_name1 = os.path.join('output','kg_out1.fits')
    kg.write(out_file_name1)
    try:
        import fitsio
        data = fitsio.read(out_file_name1)
        numpy.testing.assert_almost_equal(data['R_nom'], numpy.exp(kg.logr))
        numpy.testing.assert_almost_equal(data['meanR'], kg.meanr)
        numpy.testing.assert_almost_equal(data['meanlogR'], kg.meanlogr)
        numpy.testing.assert_almost_equal(data['kgamT'], kg.xi)
        numpy.testing.assert_almost_equal(data['kgamX'], kg.xi_im)
        numpy.testing.assert_almost_equal(data['sigma'], numpy.sqrt(kg.varxi))
        numpy.testing.assert_almost_equal(data['weight'], kg.weight)
        numpy.testing.assert_almost_equal(data['npairs'], kg.npairs)
    except ImportError:
        print('Unable to import fitsio.  Skipping fits tests.')

    # Check the read function
    kg2 = treecorr.KGCorrelation(bin_size=0.1, min_sep=1., max_sep=25., sep_units='arcmin')
    kg2.read(out_file_name1)
    numpy.testing.assert_almost_equal(kg2.logr, kg.logr)
    numpy.testing.assert_almost_equal(kg2.meanr, kg.meanr)
    numpy.testing.assert_almost_equal(kg2.meanlogr, kg.meanlogr)
    numpy.testing.assert_almost_equal(kg2.xi, kg.xi)
    numpy.testing.assert_almost_equal(kg2.xi_im, kg.xi_im)
    numpy.testing.assert_almost_equal(kg2.varxi, kg.varxi)
    numpy.testing.assert_almost_equal(kg2.weight, kg.weight)
    numpy.testing.assert_almost_equal(kg2.npairs, kg.npairs)

if __name__ == '__main__':
    test_single()
    test_pairwise()
    test_kg()
