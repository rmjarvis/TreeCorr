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

def test_constant():
    # A fairly trivial test is to use a constant value of kappa everywhere.

    ngal = 500
    A = 0.05
    L = 100.
    np.random.seed(8675309)
    x = (np.random.random_sample(ngal)-0.5) * L
    y = (np.random.random_sample(ngal)-0.5) * L
    kappa = A * np.ones(ngal)

    cat = treecorr.Catalog(x=x, y=y, k=kappa, x_units='arcmin', y_units='arcmin')

    min_sep = 10.
    max_sep = 25.
    nbins = 5
    min_u = 0.6
    max_u = 0.9
    nubins = 3
    min_v = 0.5
    max_v = 0.9
    nvbins = 5
    kkk = treecorr.KKKCorrelation(min_sep=min_sep, max_sep=max_sep, nbins=nbins,
                                  min_u=min_u, max_u=max_u, min_v=min_v, max_v=max_v,
                                  nubins=nubins, nvbins=nvbins,
                                  sep_units='arcmin', verbose=1)
    kkk.process(cat)
    print('kkk.zeta = ',kkk.zeta.flatten())
    np.testing.assert_allclose(kkk.zeta, A**3, rtol=1.e-5)

    # Should also work as a cross-correlation
    kkk.process(cat, cat, cat)
    print('as cross-correlation: kkk.zeta = ',kkk.zeta.flatten())
    np.testing.assert_allclose(kkk.zeta, A**3, rtol=1.e-5)

    # Now add some noise to the values. It should still work, but at slightly lower accuracy.
    kappa += 0.001 * (np.random.random_sample(ngal)-0.5)
    cat = treecorr.Catalog(x=x, y=y, k=kappa, x_units='arcmin', y_units='arcmin')
    kkk.process(cat)
    print('with noise: kkk.zeta = ',kkk.zeta.flatten())
    np.testing.assert_allclose(kkk.zeta, A**3, rtol=3.e-3)


def test_kkk():
    # Use kappa(r) = A exp(-r^2/2s^2)
    #
    # The Fourier transform is: kappa~(k) = 2 pi A s^2 exp(-s^2 k^2/2) / L^2
    #
    # B(k1,k2) = <k~(k1) k~(k2) k~(-k1-k2)>
    #          = (2 pi A (s/L)^2)^3 exp(-s^2 (|k1|^2 + |k2|^2 - k1.k2))
    #          = (2 pi A (s/L)^2)^3 exp(-s^2 (|k1|^2 + |k2|^2 + |k3|^2)/2)
    #
    # zeta(r1,r2) = (1/2pi)^4 int(d^2k1 int(d^2k2 exp(ik1.x1) exp(ik2.x2) B(k1,k2) ))
    #             = 2/3 pi A^3 (s/L)^2 exp(-(x1^2 + y1^2 + x2^2 + y2^2 - x1x2 - y1y2)/3s^2)
    #             = 2/3 pi A^3 (s/L)^2 exp(-(d1^2 + d2^2 + d3^2)/6s^2)

    A = 0.05
    s = 10.
    if __name__ == '__main__':
        ngal = 200000
        L = 30. * s  # Not infinity, so this introduces some error.  Our integrals were to infinity.
        tol_factor = 1
    else:
        # Looser tests from nosetests that don't take so long to run.
        ngal = 10000
        L = 20. * s
        tol_factor = 5
    np.random.seed(8675309)
    x = (np.random.random_sample(ngal)-0.5) * L
    y = (np.random.random_sample(ngal)-0.5) * L
    r2 = (x**2 + y**2)/s**2
    kappa = A * np.exp(-r2/2.)

    min_sep = 11.
    max_sep = 15.
    nbins = 3
    min_u = 0.7
    max_u = 1.0
    nubins = 3
    min_v = -0.1
    max_v = 0.3
    nvbins = 4

    cat = treecorr.Catalog(x=x, y=y, k=kappa, x_units='arcmin', y_units='arcmin')
    kkk = treecorr.KKKCorrelation(min_sep=min_sep, max_sep=max_sep, nbins=nbins,
                                  min_u=min_u, max_u=max_u, min_v=min_v, max_v=max_v,
                                  nubins=nubins, nvbins=nvbins,
                                  sep_units='arcmin', verbose=1)
    kkk.process(cat)

    # log(<d>) != <logd>, but it should be close:
    print('meanlogd1 - log(meand1) = ',kkk.meanlogd1 - np.log(kkk.meand1))
    print('meanlogd2 - log(meand2) = ',kkk.meanlogd2 - np.log(kkk.meand2))
    print('meanlogd3 - log(meand3) = ',kkk.meanlogd3 - np.log(kkk.meand3))
    print('meand3 / meand2 = ',kkk.meand3 / kkk.meand2)
    print('meanu = ',kkk.meanu)
    print('max diff = ',np.max(np.abs(kkk.meand3/kkk.meand2 -kkk.meanu)))
    print('max rel diff = ',np.max(np.abs((kkk.meand3/kkk.meand2 -kkk.meanu)/kkk.meanu)))
    print('(meand1 - meand2)/meand3 = ',(kkk.meand1-kkk.meand2) / kkk.meand3)
    print('meanv = ',kkk.meanv)
    print('max diff = ',np.max(np.abs((kkk.meand1-kkk.meand2)/kkk.meand3 -np.abs(kkk.meanv))))
    print('max rel diff = ',np.max(np.abs(((kkk.meand1-kkk.meand2)/kkk.meand3-np.abs(kkk.meanv))/kkk.meanv)))
    np.testing.assert_allclose(kkk.meanlogd1, np.log(kkk.meand1), rtol=1.e-3)
    np.testing.assert_allclose(kkk.meanlogd2, np.log(kkk.meand2), rtol=1.e-3)
    np.testing.assert_allclose(kkk.meanlogd3, np.log(kkk.meand3), rtol=1.e-3)
    np.testing.assert_allclose(kkk.meand3/kkk.meand2, kkk.meanu, rtol=1.e-5 * tol_factor)
    np.testing.assert_allclose((kkk.meand1-kkk.meand2)/kkk.meand3, np.abs(kkk.meanv),
                                  rtol=1.e-5 * tol_factor, atol=1.e-5 * tol_factor)
    np.testing.assert_allclose(kkk.meanlogd3-kkk.meanlogd2, np.log(kkk.meanu),
                                  atol=1.e-3 * tol_factor)
    np.testing.assert_allclose(np.log(kkk.meand1-kkk.meand2)-kkk.meanlogd3,
                                  np.log(np.abs(kkk.meanv)), atol=2.e-3 * tol_factor)

    d1 = kkk.meand1
    d2 = kkk.meand2
    d3 = kkk.meand3
    #print('rnom = ',np.exp(kkk.logr))
    #print('unom = ',kkk.u)
    #print('vnom = ',kkk.v)
    #print('d1 = ',d1)
    #print('d2 = ',d2)
    #print('d3 = ',d3)
    # The L^2 term in the denominator of true_zeta is the area over which the integral is done.
    # Since the centers of the triangles don't go to the edge of the box, we approximate the
    # correct area by subtracting off 2d2 from L, which should give a slightly better estimate
    # of the correct area to use here.
    L = L - 2.*d2
    true_zeta = (2.*np.pi/3) * A**3 * (s/L)**2 * np.exp(-(d1**2+d2**2+d3**2)/(6.*s**2))

    #print('ntri = ',kkk.ntri)
    print('zeta = ',kkk.zeta)
    print('true_zeta = ',true_zeta)
    #print('ratio = ',kkk.zeta / true_zeta)
    #print('diff = ',kkk.zeta - true_zeta)
    print('max rel diff = ',np.max(np.abs((kkk.zeta - true_zeta)/true_zeta)))
    np.testing.assert_allclose(kkk.zeta, true_zeta, rtol=0.1 * tol_factor)
    np.testing.assert_allclose(np.log(np.abs(kkk.zeta)), np.log(np.abs(true_zeta)),
                                  atol=0.1 * tol_factor)

    # Check that we get the same result using the corr3 functin:
    cat.write(os.path.join('data','kkk_data.dat'))
    config = treecorr.config.read_config('configs/kkk.yaml')
    config['verbose'] = 0
    treecorr.corr3(config)
    corr3_output = np.genfromtxt(os.path.join('output','kkk.out'), names=True, skip_header=1)
    np.testing.assert_almost_equal(corr3_output['zeta'], kkk.zeta.flatten())

    # Check the fits write option
    out_file_name = os.path.join('output','kkk_out.fits')
    kkk.write(out_file_name)
    data = fitsio.read(out_file_name)
    np.testing.assert_almost_equal(data['R_nom'], np.exp(kkk.logr).flatten())
    np.testing.assert_almost_equal(data['u_nom'], kkk.u.flatten())
    np.testing.assert_almost_equal(data['v_nom'], kkk.v.flatten())
    np.testing.assert_almost_equal(data['meand1'], kkk.meand1.flatten())
    np.testing.assert_almost_equal(data['meanlogd1'], kkk.meanlogd1.flatten())
    np.testing.assert_almost_equal(data['meand2'], kkk.meand2.flatten())
    np.testing.assert_almost_equal(data['meanlogd2'], kkk.meanlogd2.flatten())
    np.testing.assert_almost_equal(data['meand3'], kkk.meand3.flatten())
    np.testing.assert_almost_equal(data['meanlogd3'], kkk.meanlogd3.flatten())
    np.testing.assert_almost_equal(data['meanu'], kkk.meanu.flatten())
    np.testing.assert_almost_equal(data['meanv'], kkk.meanv.flatten())
    np.testing.assert_almost_equal(data['zeta'], kkk.zeta.flatten())
    np.testing.assert_almost_equal(data['sigma_zeta'], np.sqrt(kkk.varzeta.flatten()))
    np.testing.assert_almost_equal(data['weight'], kkk.weight.flatten())
    np.testing.assert_almost_equal(data['ntri'], kkk.ntri.flatten())

    # Check the read function
    # Note: These don't need the flatten. The read function should reshape them to the right shape.
    kkk2 = treecorr.KKKCorrelation(min_sep=min_sep, max_sep=max_sep, nbins=nbins,
                                   min_u=min_u, max_u=max_u, min_v=min_v, max_v=max_v,
                                   nubins=nubins, nvbins=nvbins,
                                   sep_units='arcmin', verbose=1)
    kkk2.read(out_file_name)
    np.testing.assert_almost_equal(kkk2.logr, kkk.logr)
    np.testing.assert_almost_equal(kkk2.u, kkk.u)
    np.testing.assert_almost_equal(kkk2.v, kkk.v)
    np.testing.assert_almost_equal(kkk2.meand1, kkk.meand1)
    np.testing.assert_almost_equal(kkk2.meanlogd1, kkk.meanlogd1)
    np.testing.assert_almost_equal(kkk2.meand2, kkk.meand2)
    np.testing.assert_almost_equal(kkk2.meanlogd2, kkk.meanlogd2)
    np.testing.assert_almost_equal(kkk2.meand3, kkk.meand3)
    np.testing.assert_almost_equal(kkk2.meanlogd3, kkk.meanlogd3)
    np.testing.assert_almost_equal(kkk2.meanu, kkk.meanu)
    np.testing.assert_almost_equal(kkk2.meanv, kkk.meanv)
    np.testing.assert_almost_equal(kkk2.zeta, kkk.zeta)
    np.testing.assert_almost_equal(kkk2.varzeta, kkk.varzeta)
    np.testing.assert_almost_equal(kkk2.weight, kkk.weight)
    np.testing.assert_almost_equal(kkk2.ntri, kkk.ntri)
    assert kkk2.coords == kkk.coords
    assert kkk2.metric == kkk.metric
    assert kkk2.sep_units == kkk.sep_units
    assert kkk2.bin_type == kkk.bin_type

if __name__ == '__main__':
    test_constant()
    test_kkk()
