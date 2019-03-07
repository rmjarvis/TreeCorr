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

def test_ggg():
    # Use gamma_t(r) = gamma0 r^2/r0^2 exp(-r^2/2r0^2)
    # i.e. gamma(r) = -gamma0 exp(-r^2/2r0^2) (x+iy)^2 / r0^2
    #
    # Rather than go through the bispectrum, I found it easier to just directly do the
    # integral:
    #
    # Gamma0 = int(int( g(x+x1,y+y1) g(x+x2,y+y2) g(x-x1-x2,y-y1-y2) (x1-Iy1)^2/(x1^2+y1^2)
    #                       (x2-Iy2)^2/(x2^2+y2^2) (x1+x2-I(y1+y2))^2/((x1+x2)^2+(y1+y2)^2)))
    #
    # where the positions are measured relative to the centroid (x,y).
    # If we call the positions relative to the centroid:
    #    q1 = x1 + I y1
    #    q2 = x2 + I y2
    #    q3 = -(x1+x2) - I (y1+y2)
    # then the result comes out as
    #
    # Gamma0 = -2/3 gamma0^3/L^2r0^4 Pi |q1|^2 |q2|^2 |q3|^2 exp(-(|q1|^2+|q2|^2+|q3|^2)/2r0^2)
    #
    # The other three are a bit more complicated.
    #
    # Gamma1 = int(int( g(x+x1,y+y1)* g(x+x2,y+y2) g(x-x1-x2,y-y1-y2) (x1+Iy1)^2/(x1^2+y1^2)
    #                       (x2-Iy2)^2/(x2^2+y2^2) (x1+x2-I(y1+y2))^2/((x1+x2)^2+(y1+y2)^2)))
    #
    #        = -2/3 gamma0^3/L^2r0^4 Pi exp(-(|q1|^2+|q2|^2+|q3|^2)/2r0^2) *
    #             ( |q1|^2 |q2|^2 |q3|^2 - 8/3 r0^2 q1^2 q2* q3*
    #               + 8/9 r0^4 (q1^2 q2*^2 q3*^2)/(|q1|^2 |q2|^2 |q3|^2) (2q1^2-q2^2-q3^2) )
    #
    # Gamm2 and Gamma3 are found from cyclic rotations of q1,q2,q3.

    gamma0 = 0.05
    r0 = 10.
    if __name__ == '__main__':
        ngal = 200000
        L = 30.*r0  # Not infinity, so this introduces some error.  Our integrals were to infinity.
        tol_factor = 1
    else:
        # Looser tests that don't take so long to run.
        ngal = 10000
        L = 20.*r0
        tol_factor = 5
    np.random.seed(8675309)
    x = (np.random.random_sample(ngal)-0.5) * L
    y = (np.random.random_sample(ngal)-0.5) * L
    r2 = (x**2 + y**2)/r0**2
    g1 = -gamma0 * np.exp(-r2/2.) * (x**2-y**2)/r0**2
    g2 = -gamma0 * np.exp(-r2/2.) * (2.*x*y)/r0**2

    min_sep = 11.
    max_sep = 15.
    nbins = 3
    min_u = 0.7
    max_u = 1.0
    nubins = 3
    min_v = -0.1
    max_v = 0.3
    nvbins = 4

    cat = treecorr.Catalog(x=x, y=y, g1=g1, g2=g2, x_units='arcmin', y_units='arcmin')
    ggg = treecorr.GGGCorrelation(min_sep=min_sep, max_sep=max_sep, nbins=nbins,
                                  min_u=min_u, max_u=max_u, min_v=min_v, max_v=max_v,
                                  nubins=nubins, nvbins=nvbins,
                                  sep_units='arcmin', verbose=1)
    ggg.process(cat)

    # log(<d>) != <logd>, but it should be close:
    print('meanlogd1 - log(meand1) = ',ggg.meanlogd1 - np.log(ggg.meand1))
    print('meanlogd2 - log(meand2) = ',ggg.meanlogd2 - np.log(ggg.meand2))
    print('meanlogd3 - log(meand3) = ',ggg.meanlogd3 - np.log(ggg.meand3))
    print('meand3 / meand2 = ',ggg.meand3 / ggg.meand2)
    print('meanu = ',ggg.meanu)
    print('max diff = ',np.max(np.abs(ggg.meand3/ggg.meand2 -ggg.meanu)))
    print('max rel diff = ',np.max(np.abs((ggg.meand3/ggg.meand2 -ggg.meanu)/ggg.meanu)))
    print('(meand1 - meand2)/meand3 = ',(ggg.meand1-ggg.meand2) / ggg.meand3)
    print('meanv = ',ggg.meanv)
    print('max diff = ',np.max(np.abs((ggg.meand1-ggg.meand2)/ggg.meand3 -np.abs(ggg.meanv))))
    print('max rel diff = ',np.max(np.abs(((ggg.meand1-ggg.meand2)/ggg.meand3-np.abs(ggg.meanv))/ggg.meanv)))
    np.testing.assert_allclose(ggg.meanlogd1, np.log(ggg.meand1), rtol=1.e-3)
    np.testing.assert_allclose(ggg.meanlogd2, np.log(ggg.meand2), rtol=1.e-3)
    np.testing.assert_allclose(ggg.meanlogd3, np.log(ggg.meand3), rtol=1.e-3)
    np.testing.assert_allclose(ggg.meand3/ggg.meand2, ggg.meanu, rtol=1.e-5 * tol_factor)
    np.testing.assert_allclose((ggg.meand1-ggg.meand2)/ggg.meand3, np.abs(ggg.meanv),
                                  rtol=1.e-5 * tol_factor, atol=1.e-5 * tol_factor)
    np.testing.assert_allclose(ggg.meanlogd3-ggg.meanlogd2, np.log(ggg.meanu),
                                  atol=1.e-3 * tol_factor)
    np.testing.assert_allclose(np.log(ggg.meand1-ggg.meand2)-ggg.meanlogd3,
                                  np.log(np.abs(ggg.meanv)), atol=2.e-3 * tol_factor)

    d1 = ggg.meand1
    d2 = ggg.meand2
    d3 = ggg.meand3
    #print('rnom = ',np.exp(ggg.logr))
    #print('unom = ',ggg.u)
    #print('vnom = ',ggg.v)
    #print('d1 = ',d1)
    #print('d2 = ',d2)
    #print('d3 = ',d3)

    # For q1,q2,q3, we can choose an orientation where c1 is at the origin, and d2 is horizontal.
    # Then let s be the "complex vector" from c1 to c3, which is just real.
    s = d2
    # And let t be from c1 to c2. t = |t| e^Iphi
    # |t| = d3
    # cos(phi) = (d2^2+d3^2-d1^2)/(2d2 d3)
    # |t| cos(phi) = (d2^2+d3^2-d1^2)/2d2
    # |t| sin(phi) = sqrt(|t|^2 - (|t|cos(phi))^2)
    tx = (d2**2 + d3**2 - d1**2)/(2.*d2)
    ty = np.sqrt(d3**2 - tx**2)
    # As arranged, if ty > 0, points 1,2,3 are clockwise, which is negative v.
    # So for bins with positive v, we need to flip the direction of ty.
    ty[ggg.meanv > 0] *= -1.
    t = tx + 1j * ty

    q1 = (s + t)/3.
    q2 = q1 - t
    q3 = q1 - s
    nq1 = np.abs(q1)**2
    nq2 = np.abs(q2)**2
    nq3 = np.abs(q3)**2
    #print('q1 = ',q1)
    #print('q2 = ',q2)
    #print('q3 = ',q3)

    # The L^2 term in the denominator of true_zeta is the area over which the integral is done.
    # Since the centers of the triangles don't go to the edge of the box, we approximate the
    # correct area by subtracting off 2*mean(qi) from L, which should give a slightly better
    # estimate of the correct area to use here.  (We used 2d2 for the kkk calculation, but this
    # is probably a slightly better estimate of the distance the triangles can get to the edges.)
    L = L - 2.*(q1 + q2 + q3)/3.

    # Gamma0 = -2/3 gamma0^3/L^2r0^4 Pi |q1|^2 |q2|^2 |q3|^2 exp(-(|q1|^2+|q2|^2+|q3|^2)/2r0^2)
    true_gam0 = ((-2.*np.pi * gamma0**3)/(3. * L**2 * r0**4) *
                    np.exp(-(nq1+nq2+nq3)/(2.*r0**2)) * (nq1*nq2*nq3) )

    # Gamma1 = -2/3 gamma0^3/L^2r0^4 Pi exp(-(|q1|^2+|q2|^2+|q3|^2)/2r0^2) *
    #             ( |q1|^2 |q2|^2 |q3|^2 - 8/3 r0^2 q1^2 q2* q3*
    #               + 8/9 r0^4 (q1^2 q2*^2 q3*^2)/(|q1|^2 |q2|^2 |q3|^2) (2q1^2-q2^2-q3^2) )
    true_gam1 = ((-2.*np.pi * gamma0**3)/(3. * L**2 * r0**4) *
                    np.exp(-(nq1+nq2+nq3)/(2.*r0**2)) *
                    (nq1*nq2*nq3 - 8./3. * r0**2 * q1**2*nq2*nq3/(q2*q3)
                     + (8./9. * r0**4 * (q1**2 * nq2 * nq3)/(nq1 * q2**2 * q3**2) *
                         (2.*q1**2 - q2**2 - q3**2)) ))

    true_gam2 = ((-2.*np.pi * gamma0**3)/(3. * L**2 * r0**4) *
                    np.exp(-(nq1+nq2+nq3)/(2.*r0**2)) *
                    (nq1*nq2*nq3 - 8./3. * r0**2 * nq1*q2**2*nq3/(q1*q3)
                     + (8./9. * r0**4 * (nq1 * q2**2 * nq3)/(q1**2 * nq2 * q3**2) *
                         (2.*q2**2 - q1**2 - q3**2)) ))

    true_gam3 = ((-2.*np.pi * gamma0**3)/(3. * L**2 * r0**4) *
                    np.exp(-(nq1+nq2+nq3)/(2.*r0**2)) *
                    (nq1*nq2*nq3 - 8./3. * r0**2 * nq1*nq2*q3**2/(q1*q2)
                     + (8./9. * r0**4 * (nq1 * nq2 * q3**2)/(q1**2 * q2**2 * nq3) *
                         (2.*q3**2 - q1**2 - q2**2)) ))

    print('ntri = ',ggg.ntri)
    print('gam0 = ',ggg.gam0)
    print('true_gam0 = ',true_gam0)
    print('ratio = ',ggg.gam0 / true_gam0)
    print('diff = ',ggg.gam0 - true_gam0)
    print('max rel diff = ',np.max(np.abs((ggg.gam0 - true_gam0)/true_gam0)))
    # The Gamma0 term is a bit worse than the others.  The accurracy improves as I increase the
    # number of objects, so I think it's just because of the smallish number of galaxies being
    # not super accurate.
    np.testing.assert_allclose(ggg.gam0, true_gam0, rtol=0.2 * tol_factor, atol=1.e-7)
    np.testing.assert_allclose(np.log(np.abs(ggg.gam0)),
                                  np.log(np.abs(true_gam0)), atol=0.2 * tol_factor)

    print('gam1 = ',ggg.gam1)
    print('true_gam1 = ',true_gam1)
    print('ratio = ',ggg.gam1 / true_gam1)
    print('diff = ',ggg.gam1 - true_gam1)
    print('max rel diff = ',np.max(np.abs((ggg.gam1 - true_gam1)/true_gam1)))
    np.testing.assert_allclose(ggg.gam1, true_gam1, rtol=0.1 * tol_factor)
    np.testing.assert_allclose(np.log(np.abs(ggg.gam1)),
                                  np.log(np.abs(true_gam1)), atol=0.1 * tol_factor)

    print('gam2 = ',ggg.gam2)
    print('true_gam2 = ',true_gam2)
    print('ratio = ',ggg.gam2 / true_gam2)
    print('diff = ',ggg.gam2 - true_gam2)
    print('max rel diff = ',np.max(np.abs((ggg.gam2 - true_gam2)/true_gam2)))
    np.testing.assert_allclose(ggg.gam2, true_gam2, rtol=0.1 * tol_factor)
    print('max rel diff for log = ',np.max(np.abs((ggg.gam2 - true_gam2)/true_gam2)))
    np.testing.assert_allclose(np.log(np.abs(ggg.gam2)),
                                  np.log(np.abs(true_gam2)), atol=0.1 * tol_factor)

    print('gam3 = ',ggg.gam3)
    print('true_gam3 = ',true_gam3)
    print('ratio = ',ggg.gam3 / true_gam3)
    print('diff = ',ggg.gam3 - true_gam3)
    print('max rel diff = ',np.max(np.abs((ggg.gam3 - true_gam3)/true_gam3)))
    np.testing.assert_allclose(ggg.gam3, true_gam3, rtol=0.1 * tol_factor)
    np.testing.assert_allclose(np.log(np.abs(ggg.gam3)),
                                  np.log(np.abs(true_gam3)), atol=0.1 * tol_factor)

    # Check that we get the same result using the corr3 function:
    cat.write(os.path.join('data','ggg_data.dat'))
    config = treecorr.config.read_config('ggg.yaml')
    config['verbose'] = 0
    treecorr.corr3(config)
    corr3_output = np.genfromtxt(os.path.join('output','ggg.out'), names=True, skip_header=1)
    np.testing.assert_allclose(corr3_output['gam0r'], ggg.gam0r.flatten(), rtol=1.e-3)
    np.testing.assert_allclose(corr3_output['gam0i'], ggg.gam0i.flatten(), rtol=1.e-3)
    np.testing.assert_allclose(corr3_output['gam1r'], ggg.gam1r.flatten(), rtol=1.e-3)
    np.testing.assert_allclose(corr3_output['gam1i'], ggg.gam1i.flatten(), rtol=1.e-3)
    np.testing.assert_allclose(corr3_output['gam2r'], ggg.gam2r.flatten(), rtol=1.e-3)
    np.testing.assert_allclose(corr3_output['gam2i'], ggg.gam2i.flatten(), rtol=1.e-3)
    np.testing.assert_allclose(corr3_output['gam3r'], ggg.gam3r.flatten(), rtol=1.e-3)
    np.testing.assert_allclose(corr3_output['gam3i'], ggg.gam3i.flatten(), rtol=1.e-3)

    # Check the fits write option
    out_file_name1 = os.path.join('output','ggg_out1.fits')
    ggg.write(out_file_name1)
    data = fitsio.read(out_file_name1)
    np.testing.assert_almost_equal(data['R_nom'], np.exp(ggg.logr).flatten())
    np.testing.assert_almost_equal(data['u_nom'], ggg.u.flatten())
    np.testing.assert_almost_equal(data['v_nom'], ggg.v.flatten())
    np.testing.assert_almost_equal(data['meand1'], ggg.meand1.flatten())
    np.testing.assert_almost_equal(data['meanlogd1'], ggg.meanlogd1.flatten())
    np.testing.assert_almost_equal(data['meand2'], ggg.meand2.flatten())
    np.testing.assert_almost_equal(data['meanlogd2'], ggg.meanlogd2.flatten())
    np.testing.assert_almost_equal(data['meand3'], ggg.meand3.flatten())
    np.testing.assert_almost_equal(data['meanlogd3'], ggg.meanlogd3.flatten())
    np.testing.assert_almost_equal(data['meanu'], ggg.meanu.flatten())
    np.testing.assert_almost_equal(data['meanv'], ggg.meanv.flatten())
    np.testing.assert_almost_equal(data['gam0r'], ggg.gam0.real.flatten())
    np.testing.assert_almost_equal(data['gam1r'], ggg.gam1.real.flatten())
    np.testing.assert_almost_equal(data['gam2r'], ggg.gam2.real.flatten())
    np.testing.assert_almost_equal(data['gam3r'], ggg.gam3.real.flatten())
    np.testing.assert_almost_equal(data['gam0i'], ggg.gam0.imag.flatten())
    np.testing.assert_almost_equal(data['gam1i'], ggg.gam1.imag.flatten())
    np.testing.assert_almost_equal(data['gam2i'], ggg.gam2.imag.flatten())
    np.testing.assert_almost_equal(data['gam3i'], ggg.gam3.imag.flatten())
    np.testing.assert_almost_equal(data['sigma_gam'], np.sqrt(ggg.vargam.flatten()))
    np.testing.assert_almost_equal(data['weight'], ggg.weight.flatten())
    np.testing.assert_almost_equal(data['ntri'], ggg.ntri.flatten())

    # Check the read function
    # Note: These don't need the flatten. The read function should reshape them to the right shape.
    ggg2 = treecorr.GGGCorrelation(min_sep=min_sep, max_sep=max_sep, nbins=nbins,
                                   min_u=min_u, max_u=max_u, min_v=min_v, max_v=max_v,
                                   nubins=nubins, nvbins=nvbins,
                                   sep_units='arcmin', verbose=1)
    ggg2.read(out_file_name1)
    np.testing.assert_almost_equal(ggg2.logr, ggg.logr)
    np.testing.assert_almost_equal(ggg2.u, ggg.u)
    np.testing.assert_almost_equal(ggg2.v, ggg.v)
    np.testing.assert_almost_equal(ggg2.meand1, ggg.meand1)
    np.testing.assert_almost_equal(ggg2.meanlogd1, ggg.meanlogd1)
    np.testing.assert_almost_equal(ggg2.meand2, ggg.meand2)
    np.testing.assert_almost_equal(ggg2.meanlogd2, ggg.meanlogd2)
    np.testing.assert_almost_equal(ggg2.meand3, ggg.meand3)
    np.testing.assert_almost_equal(ggg2.meanlogd3, ggg.meanlogd3)
    np.testing.assert_almost_equal(ggg2.meanu, ggg.meanu)
    np.testing.assert_almost_equal(ggg2.meanv, ggg.meanv)
    np.testing.assert_almost_equal(ggg2.gam0, ggg.gam0)
    np.testing.assert_almost_equal(ggg2.gam1, ggg.gam1)
    np.testing.assert_almost_equal(ggg2.gam2, ggg.gam2)
    np.testing.assert_almost_equal(ggg2.gam3, ggg.gam3)
    np.testing.assert_almost_equal(ggg2.vargam, ggg.vargam)
    np.testing.assert_almost_equal(ggg2.weight, ggg.weight)
    np.testing.assert_almost_equal(ggg2.ntri, ggg.ntri)
    assert ggg2.coords == ggg.coords
    assert ggg2.metric == ggg.metric
    assert ggg2.sep_units == ggg.sep_units
    assert ggg2.bin_type == ggg.bin_type

if __name__ == '__main__':
    test_ggg()
