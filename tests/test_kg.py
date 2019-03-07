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
import numpy as np
import treecorr
import os
import fitsio

from test_helper import get_script_name

def test_single():
    # Use gamma_t(r) = gamma0 exp(-r^2/2r0^2) around a single lens
    # i.e. gamma(r) = -gamma0 exp(-r^2/2r0^2) (x+iy)^2/r^2

    nsource = 100000
    gamma0 = 0.05
    kappa = 0.23
    r0 = 10.
    L = 5. * r0
    np.random.seed(8675309)
    x = (np.random.random_sample(nsource)-0.5) * L
    y = (np.random.random_sample(nsource)-0.5) * L
    r2 = (x**2 + y**2)
    gammat = gamma0 * np.exp(-0.5*r2/r0**2)
    g1 = -gammat * (x**2-y**2)/r2
    g2 = -gammat * (2.*x*y)/r2

    lens_cat = treecorr.Catalog(x=[0], y=[0], k=[kappa],  x_units='arcmin', y_units='arcmin')
    source_cat = treecorr.Catalog(x=x, y=y, g1=g1, g2=g2, x_units='arcmin', y_units='arcmin')
    kg = treecorr.KGCorrelation(bin_size=0.1, min_sep=1., max_sep=20., sep_units='arcmin',
                                verbose=1)
    kg.process(lens_cat, source_cat)

    # log(<R>) != <logR>, but it should be close:
    print('meanlogr - log(meanr) = ',kg.meanlogr - np.log(kg.meanr))
    np.testing.assert_allclose(kg.meanlogr, np.log(kg.meanr), atol=1.e-3)

    r = kg.meanr
    true_kgt = kappa * gamma0 * np.exp(-0.5*r**2/r0**2)

    print('kg.xi = ',kg.xi)
    print('kg.xi_im = ',kg.xi_im)
    print('true_gammat = ',true_kgt)
    print('ratio = ',kg.xi / true_kgt)
    print('diff = ',kg.xi - true_kgt)
    print('max diff = ',max(abs(kg.xi - true_kgt)))
    np.testing.assert_allclose(kg.xi, true_kgt, rtol=1.e-2)
    np.testing.assert_allclose(kg.xi_im, 0, atol=1.e-4)

    # Check that we get the same result using the corr2 function:
    lens_cat.write(os.path.join('data','kg_single_lens.dat'))
    source_cat.write(os.path.join('data','kg_single_source.dat'))
    config = treecorr.read_config('kg_single.yaml')
    config['verbose'] = 0
    treecorr.corr2(config)
    corr2_output = np.genfromtxt(os.path.join('output','kg_single.out'), names=True,
                                    skip_header=1)
    print('kg.xi = ',kg.xi)
    print('from corr2 output = ',corr2_output['kgamT'])
    print('ratio = ',corr2_output['kgamT']/kg.xi)
    print('diff = ',corr2_output['kgamT']-kg.xi)
    np.testing.assert_allclose(corr2_output['kgamT'], kg.xi, rtol=1.e-3)

    print('xi_im from corr2 output = ',corr2_output['kgamX'])
    np.testing.assert_allclose(corr2_output['kgamX'], 0., atol=1.e-4)


def test_pairwise():
    # Test the same profile, but with the pairwise calcualtion:
    nsource = 100000
    gamma0 = 0.05
    kappa = 0.23
    r0 = 10.
    L = 5. * r0
    np.random.seed(8675309)
    x = (np.random.random_sample(nsource)-0.5) * L
    y = (np.random.random_sample(nsource)-0.5) * L
    r2 = (x**2 + y**2)
    gammat = gamma0 * np.exp(-0.5*r2/r0**2)
    g1 = -gammat * (x**2-y**2)/r2
    g2 = -gammat * (2.*x*y)/r2

    dx = (np.random.random_sample(nsource)-0.5) * L
    dx = (np.random.random_sample(nsource)-0.5) * L
    k = kappa * np.ones(nsource)

    lens_cat = treecorr.Catalog(x=dx, y=dx, k=k, x_units='arcmin', y_units='arcmin')
    source_cat = treecorr.Catalog(x=x+dx, y=y+dx, g1=g1, g2=g2, x_units='arcmin', y_units='arcmin')
    kg = treecorr.KGCorrelation(bin_size=0.1, min_sep=1., max_sep=20., sep_units='arcmin',
                                verbose=1, pairwise=True)
    kg.process(lens_cat, source_cat)

    r = kg.meanr
    true_kgt = kappa * gamma0 * np.exp(-0.5*r**2/r0**2)

    print('kg.xi = ',kg.xi)
    print('kg.xi_im = ',kg.xi_im)
    print('true_gammat = ',true_kgt)
    print('ratio = ',kg.xi / true_kgt)
    print('diff = ',kg.xi - true_kgt)
    print('max diff = ',max(abs(kg.xi - true_kgt)))
    np.testing.assert_allclose(kg.xi, true_kgt, rtol=1.e-2)
    np.testing.assert_allclose(kg.xi_im, 0, atol=1.e-4)

    # Check that we get the same result using the corr2 function
    lens_cat.write(os.path.join('data','kg_pairwise_lens.dat'))
    source_cat.write(os.path.join('data','kg_pairwise_source.dat'))
    config = treecorr.read_config('kg_pairwise.yaml')
    config['verbose'] = 0
    treecorr.corr2(config)
    corr2_output = np.genfromtxt(os.path.join('output','kg_pairwise.out'), names=True,
                                    skip_header=1)
    print('kg.xi = ',kg.xi)
    print('from corr2 output = ',corr2_output['kgamT'])
    print('ratio = ',corr2_output['kgamT']/kg.xi)
    print('diff = ',corr2_output['kgamT']-kg.xi)
    np.testing.assert_allclose(corr2_output['kgamT'], kg.xi, rtol=1.e-3)

    print('xi_im from corr2 output = ',corr2_output['kgamX'])
    np.testing.assert_allclose(corr2_output['kgamX'], 0., atol=1.e-4)


def test_kg():
    # Use gamma_t(r) = gamma0 exp(-r^2/2r0^2) around a bunch of foreground lenses.
    # i.e. gamma(r) = -gamma0 exp(-r^2/2r0^2) (x+iy)^2/r^2
    # For each lens, we divide this by a random kappa value assigned to that lens, so
    # the final kg output shoudl be just gamma_t.

    nlens = 1000
    nsource = 30000
    r0 = 10.
    L = 50. * r0

    gamma0 = 0.05
    np.random.seed(8675309)
    xl = (np.random.random_sample(nlens)-0.5) * L
    yl = (np.random.random_sample(nlens)-0.5) * L
    kl = np.random.normal(0.23, 0.05, (nlens,) )
    xs = (np.random.random_sample(nsource)-0.5) * L
    ys = (np.random.random_sample(nsource)-0.5) * L
    g1 = np.zeros( (nsource,) )
    g2 = np.zeros( (nsource,) )
    for x,y,k in zip(xl,yl,kl):
        dx = xs-x
        dy = ys-y
        r2 = dx**2 + dy**2
        gammat = gamma0 * np.exp(-0.5*r2/r0**2) / k
        g1 += -gammat * (dx**2-dy**2)/r2
        g2 += -gammat * (2.*dx*dy)/r2

    lens_cat = treecorr.Catalog(x=xl, y=yl, k=kl, x_units='arcmin', y_units='arcmin')
    source_cat = treecorr.Catalog(x=xs, y=ys, g1=g1, g2=g2, x_units='arcmin', y_units='arcmin')
    kg = treecorr.KGCorrelation(bin_size=0.1, min_sep=1., max_sep=20., sep_units='arcmin',
                                verbose=1)
    kg.process(lens_cat, source_cat)

    r = kg.meanr
    true_gt = gamma0 * np.exp(-0.5*r**2/r0**2)

    print('kg.xi = ',kg.xi)
    print('kg.xi_im = ',kg.xi_im)
    print('true_gammat = ',true_gt)
    print('ratio = ',kg.xi / true_gt)
    print('diff = ',kg.xi - true_gt)
    print('max diff = ',max(abs(kg.xi - true_gt)))
    np.testing.assert_allclose(kg.xi, true_gt, rtol=0.1)
    np.testing.assert_allclose(kg.xi_im, 0., atol=1.e-2)

    # Check that we get the same result using the corr2 function:
    lens_cat.write(os.path.join('data','kg_lens.dat'))
    source_cat.write(os.path.join('data','kg_source.dat'))
    config = treecorr.read_config('kg.yaml')
    config['verbose'] = 0
    config['precision'] = 8
    treecorr.corr2(config)
    corr2_output = np.genfromtxt(os.path.join('output','kg.out'), names=True, skip_header=1)
    print('kg.xi = ',kg.xi)
    print('from corr2 output = ',corr2_output['kgamT'])
    print('ratio = ',corr2_output['kgamT']/kg.xi)
    print('diff = ',corr2_output['kgamT']-kg.xi)
    np.testing.assert_allclose(corr2_output['kgamT'], kg.xi, rtol=1.e-3)

    print('xi_im from corr2 output = ',corr2_output['kgamX'])
    np.testing.assert_allclose(corr2_output['kgamX'], 0., atol=1.e-2)

    # Check the fits write option
    out_file_name1 = os.path.join('output','kg_out1.fits')
    kg.write(out_file_name1)
    data = fitsio.read(out_file_name1)
    np.testing.assert_almost_equal(data['R_nom'], np.exp(kg.logr))
    np.testing.assert_almost_equal(data['meanR'], kg.meanr)
    np.testing.assert_almost_equal(data['meanlogR'], kg.meanlogr)
    np.testing.assert_almost_equal(data['kgamT'], kg.xi)
    np.testing.assert_almost_equal(data['kgamX'], kg.xi_im)
    np.testing.assert_almost_equal(data['sigma'], np.sqrt(kg.varxi))
    np.testing.assert_almost_equal(data['weight'], kg.weight)
    np.testing.assert_almost_equal(data['npairs'], kg.npairs)

    # Check the read function
    kg2 = treecorr.KGCorrelation(bin_size=0.1, min_sep=1., max_sep=20., sep_units='arcmin')
    kg2.read(out_file_name1)
    np.testing.assert_almost_equal(kg2.logr, kg.logr)
    np.testing.assert_almost_equal(kg2.meanr, kg.meanr)
    np.testing.assert_almost_equal(kg2.meanlogr, kg.meanlogr)
    np.testing.assert_almost_equal(kg2.xi, kg.xi)
    np.testing.assert_almost_equal(kg2.xi_im, kg.xi_im)
    np.testing.assert_almost_equal(kg2.varxi, kg.varxi)
    np.testing.assert_almost_equal(kg2.weight, kg.weight)
    np.testing.assert_almost_equal(kg2.npairs, kg.npairs)
    assert kg2.coords == kg.coords
    assert kg2.metric == kg.metric
    assert kg2.sep_units == kg.sep_units
    assert kg2.bin_type == kg.bin_type

if __name__ == '__main__':
    test_single()
    test_pairwise()
    test_kg()
