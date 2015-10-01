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

def test_constant():
    # A fairly trivial test is to use a constant value of kappa everywhere.

    ngal = 2000
    A = 0.05
    L = 100.
    numpy.random.seed(8675309)
    x = (numpy.random.random_sample(ngal)-0.5) * L
    y = (numpy.random.random_sample(ngal)-0.5) * L
    kappa = A * numpy.ones(ngal)

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
                                  sep_units='arcmin', verbose=2)
    kkk.process(cat)
    print('kkk.zeta = ',kkk.zeta.flatten())
    numpy.testing.assert_almost_equal(kkk.zeta, A**3, decimal=10)

    # Should also work as a cross-correlation
    kkk.process(cat, cat, cat)
    print('as cross-correlation: kkk.zeta = ',kkk.zeta.flatten())
    numpy.testing.assert_almost_equal(kkk.zeta, A**3, decimal=10)

    # Now add some noise to the values. It should still work, but at slightly lower accuracy.
    kappa += 0.001 * (numpy.random.random_sample(ngal)-0.5)
    cat = treecorr.Catalog(x=x, y=y, k=kappa, x_units='arcmin', y_units='arcmin')
    kkk.process(cat)
    print('with noise: kkk.zeta = ',kkk.zeta.flatten())
    numpy.testing.assert_almost_equal(kkk.zeta, A**3, decimal=6)


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
        req_factor = 1
    else:
        # Looser tests from nosetests that don't take so long to run.
        ngal = 5000
        L = 10. * s
        req_factor = 5
    numpy.random.seed(8675309)
    x = (numpy.random.random_sample(ngal)-0.5) * L
    y = (numpy.random.random_sample(ngal)-0.5) * L
    r2 = (x**2 + y**2)/s**2
    kappa = A * numpy.exp(-r2/2.)

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
                                  sep_units='arcmin', verbose=2)
    kkk.process(cat)

    # log(<d>) != <logd>, but it should be close:
    #print('meanlogd1 - log(meand1) = ',kkk.meanlogd1 - numpy.log(kkk.meand1))
    #print('meanlogd2 - log(meand2) = ',kkk.meanlogd2 - numpy.log(kkk.meand2))
    #print('meanlogd3 - log(meand3) = ',kkk.meanlogd3 - numpy.log(kkk.meand3))
    #print('meanlogd3 - meanlogd2 - log(meanu) = ',kkk.meanlogd3 - kkk.meanlogd2 - numpy.log(kkk.meanu))
    #print('log(meand1-meand2) - meanlogd3 - log(meanv) = ',numpy.log(kkk.meand1-kkk.meand2) - kkk.meanlogd3 - numpy.log(numpy.abs(kkk.meanv)))
    numpy.testing.assert_almost_equal(kkk.meanlogd1, numpy.log(kkk.meand1), decimal=3)
    numpy.testing.assert_almost_equal(kkk.meanlogd2, numpy.log(kkk.meand2), decimal=3)
    numpy.testing.assert_almost_equal(kkk.meanlogd3, numpy.log(kkk.meand3), decimal=3)
    numpy.testing.assert_almost_equal(kkk.meanlogd3-kkk.meanlogd2, numpy.log(kkk.meanu), decimal=3)
    numpy.testing.assert_almost_equal(numpy.log(kkk.meand1-kkk.meand2)-kkk.meanlogd3, 
                                      numpy.log(numpy.abs(kkk.meanv)), decimal=3)

    d1 = kkk.meand1
    d2 = kkk.meand2
    d3 = kkk.meand3
    #print('rnom = ',numpy.exp(kkk.logr))
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
    true_zeta = (2.*numpy.pi/3) * A**3 * (s/L)**2 * numpy.exp(-(d1**2+d2**2+d3**2)/(6.*s**2))

    #print('ntri = ',kkk.ntri)
    print('zeta = ',kkk.zeta)
    print('true_zeta = ',true_zeta)
    #print('ratio = ',kkk.zeta / true_zeta)
    #print('diff = ',kkk.zeta - true_zeta)
    print('max rel diff = ',numpy.max(numpy.abs((kkk.zeta - true_zeta)/true_zeta)))
    assert numpy.max(numpy.abs((kkk.zeta - true_zeta)/true_zeta)) / req_factor < 0.1
    numpy.testing.assert_almost_equal(numpy.log(numpy.abs(kkk.zeta)) / req_factor, 
                                      numpy.log(numpy.abs(true_zeta)) / req_factor, decimal=1)

    # Check that we get the same result using the corr3 executable:
    if __name__ == '__main__':
        cat.write(os.path.join('data','kkk_data.dat'))
        import subprocess
        p = subprocess.Popen( ["corr3","kkk.params"] )
        p.communicate()
        corr3_output = numpy.genfromtxt(os.path.join('output','kkk.out'), names=True)
        #print('zeta = ',kkk.zeta)
        #print('from corr3 output = ',corr3_output['zeta'])
        #print('ratio = ',corr3_output['zeta']/kkk.zeta.flatten())
        #print('diff = ',corr3_output['zeta']-kkk.zeta.flatten())
        numpy.testing.assert_almost_equal(corr3_output['zeta']/kkk.zeta.flatten(), 1., decimal=3)

    # Check the fits write option
    out_file_name = os.path.join('output','kkk_out.fits')
    kkk.write(out_file_name)
    try:
        import fitsio
        data = fitsio.read(out_file_name)
        numpy.testing.assert_almost_equal(data['R_nom'], numpy.exp(kkk.logr).flatten())
        numpy.testing.assert_almost_equal(data['u_nom'], kkk.u.flatten())
        numpy.testing.assert_almost_equal(data['v_nom'], kkk.v.flatten())
        numpy.testing.assert_almost_equal(data['meand1'], kkk.meand1.flatten())
        numpy.testing.assert_almost_equal(data['meanlogd1'], kkk.meanlogd1.flatten())
        numpy.testing.assert_almost_equal(data['meand2'], kkk.meand2.flatten())
        numpy.testing.assert_almost_equal(data['meanlogd2'], kkk.meanlogd2.flatten())
        numpy.testing.assert_almost_equal(data['meand3'], kkk.meand3.flatten())
        numpy.testing.assert_almost_equal(data['meanlogd3'], kkk.meanlogd3.flatten())
        numpy.testing.assert_almost_equal(data['meanu'], kkk.meanu.flatten())
        numpy.testing.assert_almost_equal(data['meanv'], kkk.meanv.flatten())
        numpy.testing.assert_almost_equal(data['zeta'], kkk.zeta.flatten())
        numpy.testing.assert_almost_equal(data['sigma_zeta'], numpy.sqrt(kkk.varzeta.flatten()))
        numpy.testing.assert_almost_equal(data['weight'], kkk.weight.flatten())
        numpy.testing.assert_almost_equal(data['ntri'], kkk.ntri.flatten())
    except ImportError:
        print('Unable to import fitsio.  Skipping fits tests.')

    # Check the read function
    # Note: These don't need the flatten. The read function should reshape them to the right shape.
    kkk2 = treecorr.KKKCorrelation(min_sep=min_sep, max_sep=max_sep, nbins=nbins,
                                   min_u=min_u, max_u=max_u, min_v=min_v, max_v=max_v,
                                   nubins=nubins, nvbins=nvbins,
                                   sep_units='arcmin', verbose=2)
    kkk2.read(out_file_name)
    numpy.testing.assert_almost_equal(kkk2.logr, kkk.logr)
    numpy.testing.assert_almost_equal(kkk2.u, kkk.u)
    numpy.testing.assert_almost_equal(kkk2.v, kkk.v)
    numpy.testing.assert_almost_equal(kkk2.meand1, kkk.meand1)
    numpy.testing.assert_almost_equal(kkk2.meanlogd1, kkk.meanlogd1)
    numpy.testing.assert_almost_equal(kkk2.meand2, kkk.meand2)
    numpy.testing.assert_almost_equal(kkk2.meanlogd2, kkk.meanlogd2)
    numpy.testing.assert_almost_equal(kkk2.meand3, kkk.meand3)
    numpy.testing.assert_almost_equal(kkk2.meanlogd3, kkk.meanlogd3)
    numpy.testing.assert_almost_equal(kkk2.meanu, kkk.meanu)
    numpy.testing.assert_almost_equal(kkk2.meanv, kkk.meanv)
    numpy.testing.assert_almost_equal(kkk2.zeta, kkk.zeta)
    numpy.testing.assert_almost_equal(kkk2.varzeta, kkk.varzeta)
    numpy.testing.assert_almost_equal(kkk2.weight, kkk.weight)
    numpy.testing.assert_almost_equal(kkk2.ntri, kkk.ntri)

if __name__ == '__main__':
    test_constant()
    test_kkk()
