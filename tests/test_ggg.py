# Copyright (c) 2003-2014 by Mike Jarvis
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


import numpy
import treecorr
import os

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

    ngal = 300000
    gamma0 = 0.05
    r0 = 10.
    L = 30. * r0  # Not infinity, so this introduces some error.  Our integrals were to infinity.
    numpy.random.seed(8675309)
    x = (numpy.random.random_sample(ngal)-0.5) * L
    y = (numpy.random.random_sample(ngal)-0.5) * L
    r2 = (x**2 + y**2)/r0**2
    g1 = -gamma0 * numpy.exp(-r2/2.) * (x**2-y**2)/r0**2
    g2 = -gamma0 * numpy.exp(-r2/2.) * (2.*x*y)/r0**2

    min_sep = 15.
    max_sep = 30.
    nbins = 7
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
                                  sep_units='arcmin', verbose=3)
    ggg.process(cat)

    # log(<d>) != <logd>, but it should be close:
    #print 'meanlogd1 - log(meand1) = ',ggg.meanlogd1 - numpy.log(ggg.meand1)
    #print 'meanlogd2 - log(meand2) = ',ggg.meanlogd2 - numpy.log(ggg.meand2)
    #print 'meanlogd3 - log(meand3) = ',ggg.meanlogd3 - numpy.log(ggg.meand3)
    #print 'meanlogd3 - meanlogd2 - log(meanu) = ',ggg.meanlogd3 - ggg.meanlogd2 - numpy.log(ggg.meanu)
    #print 'log(meand1-meand2) - meanlogd3 - log(meanv) = ',numpy.log(ggg.meand1-ggg.meand2) - ggg.meanlogd3 - numpy.log(numpy.abs(ggg.meanv))
    numpy.testing.assert_almost_equal(ggg.meanlogd1, numpy.log(ggg.meand1), decimal=3)
    numpy.testing.assert_almost_equal(ggg.meanlogd2, numpy.log(ggg.meand2), decimal=3)
    numpy.testing.assert_almost_equal(ggg.meanlogd3, numpy.log(ggg.meand3), decimal=3)
    numpy.testing.assert_almost_equal(ggg.meanlogd3-ggg.meanlogd2, numpy.log(ggg.meanu), decimal=3)
    numpy.testing.assert_almost_equal(numpy.log(ggg.meand1-ggg.meand2)-ggg.meanlogd3, 
                                      numpy.log(numpy.abs(ggg.meanv)), decimal=3)

    d1 = ggg.meand1
    d2 = ggg.meand2
    d3 = ggg.meand3
    #print 'rnom = ',numpy.exp(ggg.logr)
    #print 'unom = ',ggg.u
    #print 'vnom = ',ggg.v
    #print 'd1 = ',d1
    #print 'd2 = ',d2
    #print 'd3 = ',d3

    # For q1,q2,q3, we can choose an orientation where c1 is at the origin, and d2 is horizontal.
    # Then let s be the "complex vector" from c1 to c3, which is just real.
    s = d2
    # And let t be from c1 to c2. t = |t| e^Iphi
    # |t| = d3. 
    # cos(phi) = (d2^2+d3^2-d1^2)/(2d2 d3)
    # |t| cos(phi) = (d2^2+d3^2-d1^2)/2d2
    # |t| sin(phi) = sqrt(|t|^2 - (|t|cos(phi))^2)
    tx = (d2**2 + d3**2 - d1**2)/(2.*d2)
    ty = numpy.sqrt(d3**2 - tx**2)
    # As arranged, if ty > 0, points 1,2,3 are clockwise, which is negative v.
    # So for bins with positive v, we need to flip the direction of ty.
    ty[ggg.meanv > 0] *= -1.
    t = tx + 1j * ty

    q1 = (s + t)/3.
    q2 = q1 - s
    q3 = q1 - t
    nq1 = numpy.abs(q1)**2
    nq2 = numpy.abs(q2)**2
    nq3 = numpy.abs(q3)**2

    # Gamma0 = -2/3 gamma0^3/L^2r0^4 Pi |q1|^2 |q2|^2 |q3|^2 exp(-(|q1|^2+|q2|^2+|q3|^2)/2r0^2)
    true_gam0 = ((-2.*numpy.pi * gamma0**3)/(3. * L**2 * r0**4) * 
                    numpy.exp(-(nq1+nq2+nq3)/(2.*r0**2)) * (nq1*nq2*nq3) )

    # Gamma1 = -2/3 gamma0^3/L^2r0^4 Pi exp(-(|q1|^2+|q2|^2+|q3|^2)/2r0^2) *
    #             ( |q1|^2 |q2|^2 |q3|^2 - 8/3 r0^2 q1^2 q2* q3*
    #               + 8/9 r0^4 (q1^2 q2*^2 q3*^2)/(|q1|^2 |q2|^2 |q3|^2) (2q1^2-q2^2-q3^2) )
    true_gam1 = ((-2.*numpy.pi * gamma0**3)/(3. * L**2 * r0**4) * 
                    numpy.exp(-(nq1+nq2+nq3)/(2.*r0**2)) *
                    (nq1*nq2*nq3 - 8./3. * r0**2 * q1**2*nq2*nq3/(q2*q3) 
                     + 8./9. * r0**4 * (q1**2 * nq2**2 * nq3**2)/(nq1**2 * q2**2 * q3**2) *
                         (2.*q1**2 - q2**2 - q3**2) ))

    print 'ntri = ',ggg.ntri
    print 'gam0 = ',ggg.gam0
    print 'true_gam0 = ',true_gam0
    print 'ratio = ',ggg.gam0 / true_gam0
    print 'diff = ',ggg.gam0 - true_gam0
    print 'max rel diff = ',numpy.max(numpy.abs((ggg.gam0 - true_gam0)/true_gam0))
    print 'gam1 = ',ggg.gam1
    print 'true_gam1 = ',true_gam1
    print 'ratio = ',ggg.gam1 / true_gam1
    print 'diff = ',ggg.gam1 - true_gam1
    print 'max rel diff = ',numpy.max(numpy.abs((ggg.gam1 - true_gam1)/true_gam1))

    assert numpy.max(numpy.abs((ggg.gam0 - true_gam0)/true_gam0)) < 0.5
    numpy.testing.assert_almost_equal(numpy.log(numpy.abs(ggg.gam0)), 
                                      numpy.log(numpy.abs(true_gam0)), decimal=1)

    assert numpy.max(numpy.abs((ggg.gam1 - true_gam1)/true_gam1)) < 0.5
    numpy.testing.assert_almost_equal(numpy.log(numpy.abs(ggg.gam1)), 
                                      numpy.log(numpy.abs(true_gam1)), decimal=1)

    # Check that we get the same result using the corr3 executable:
    if __name__ == '__main__':
        cat.write(os.path.join('data','ggg_data.dat'))
        import subprocess
        p = subprocess.Popen( ["corr3","ggg.params"] )
        p.communicate()
        corr3_output = numpy.genfromtxt(os.path.join('output','ggg.out'), names=True)
        print 'gam0 = ',ggg.gam0
        print 'from corr3 output = ',corr3_output['gam0']
        print 'ratio = ',corr3_output['gam0']/ggg.gam0.flatten()
        print 'diff = ',corr3_output['gam0']-ggg.gam0.flatten()
        numpy.testing.assert_almost_equal(corr3_output['gam0']/ggg.gam0.flatten(), 1., decimal=3)
        print 'gam1 = ',ggg.gam1
        print 'from corr3 output = ',corr3_output['gam1']
        print 'ratio = ',corr3_output['gam1']/ggg.gam1.flatten()
        print 'diff = ',corr3_output['gam1']-ggg.gam1.flatten()
        numpy.testing.assert_almost_equal(corr3_output['gam1']/ggg.gam1.flatten(), 1., decimal=3)
        print 'gam2 = ',ggg.gam2
        print 'from corr3 output = ',corr3_output['gam2']
        print 'ratio = ',corr3_output['gam2']/ggg.gam2.flatten()
        print 'diff = ',corr3_output['gam2']-ggg.gam2.flatten()
        numpy.testing.assert_almost_equal(corr3_output['gam2']/ggg.gam2.flatten(), 1., decimal=3)
        print 'gam3 = ',ggg.gam3
        print 'from corr3 output = ',corr3_output['gam3']
        print 'ratio = ',corr3_output['gam3']/ggg.gam3.flatten()
        print 'diff = ',corr3_output['gam3']-ggg.gam3.flatten()
        numpy.testing.assert_almost_equal(corr3_output['gam3']/ggg.gam3.flatten(), 1., decimal=3)

    # Check the fits write option
    out_file_name1 = os.path.join('output','ggg_out1.fits')
    ggg.write(out_file_name1)
    import fitsio
    data = fitsio.read(out_file_name1)
    numpy.testing.assert_almost_equal(data['R_nom'], numpy.exp(ggg.logr).flatten())
    numpy.testing.assert_almost_equal(data['u_nom'], ggg.u.flatten())
    numpy.testing.assert_almost_equal(data['v_nom'], ggg.v.flatten())
    numpy.testing.assert_almost_equal(data['<d1>'], ggg.meand1.flatten())
    numpy.testing.assert_almost_equal(data['<logd1>'], ggg.meanlogd1.flatten())
    numpy.testing.assert_almost_equal(data['<d2>'], ggg.meand2.flatten())
    numpy.testing.assert_almost_equal(data['<logd2>'], ggg.meanlogd2.flatten())
    numpy.testing.assert_almost_equal(data['<d3>'], ggg.meand3.flatten())
    numpy.testing.assert_almost_equal(data['<logd3>'], ggg.meanlogd3.flatten())
    numpy.testing.assert_almost_equal(data['<u>'], ggg.meanu.flatten())
    numpy.testing.assert_almost_equal(data['<v>'], ggg.meanv.flatten())
    numpy.testing.assert_almost_equal(data['gam0r'], ggg.gam0.real.flatten())
    numpy.testing.assert_almost_equal(data['gam1r'], ggg.gam1.real.flatten())
    numpy.testing.assert_almost_equal(data['gam2r'], ggg.gam2.real.flatten())
    numpy.testing.assert_almost_equal(data['gam3r'], ggg.gam3.real.flatten())
    numpy.testing.assert_almost_equal(data['gam0i'], ggg.gam0.imag.flatten())
    numpy.testing.assert_almost_equal(data['gam1i'], ggg.gam1.imag.flatten())
    numpy.testing.assert_almost_equal(data['gam2i'], ggg.gam2.imag.flatten())
    numpy.testing.assert_almost_equal(data['gam3i'], ggg.gam3.imag.flatten())
    numpy.testing.assert_almost_equal(data['sigma_gam'], numpy.sqrt(ggg.vargam.flatten()))
    numpy.testing.assert_almost_equal(data['weight'], ggg.weight.flatten())
    numpy.testing.assert_almost_equal(data['ntri'], ggg.ntri.flatten())

    # Check the read function
    # Note: These don't need the flatten. The read function should reshape them to the right shape.
    ggg2 = treecorr.GGGCorrelation(min_sep=min_sep, max_sep=max_sep, nbins=nbins,
                                   min_u=min_u, max_u=max_u, min_v=min_v, max_v=max_v,
                                   nubins=nubins, nvbins=nvbins,
                                   sep_units='arcmin', verbose=3)
    ggg2.read(out_file_name1)
    numpy.testing.assert_almost_equal(ggg2.logr, ggg.logr)
    numpy.testing.assert_almost_equal(ggg2.u, ggg.u)
    numpy.testing.assert_almost_equal(ggg2.v, ggg.v)
    numpy.testing.assert_almost_equal(ggg2.meand1, ggg.meand1)
    numpy.testing.assert_almost_equal(ggg2.meanlogd1, ggg.meanlogd1)
    numpy.testing.assert_almost_equal(ggg2.meand2, ggg.meand2)
    numpy.testing.assert_almost_equal(ggg2.meanlogd2, ggg.meanlogd2)
    numpy.testing.assert_almost_equal(ggg2.meand3, ggg.meand3)
    numpy.testing.assert_almost_equal(ggg2.meanlogd3, ggg.meanlogd3)
    numpy.testing.assert_almost_equal(ggg2.meanu, ggg.meanu)
    numpy.testing.assert_almost_equal(ggg2.meanv, ggg.meanv)
    numpy.testing.assert_almost_equal(ggg2.gam0, ggg.gam0)
    numpy.testing.assert_almost_equal(ggg2.gam1, ggg.gam1)
    numpy.testing.assert_almost_equal(ggg2.gam2, ggg.gam2)
    numpy.testing.assert_almost_equal(ggg2.gam3, ggg.gam3)
    numpy.testing.assert_almost_equal(ggg2.gam0_im, ggg.gam0_im)
    numpy.testing.assert_almost_equal(ggg2.gam1_im, ggg.gam1_im)
    numpy.testing.assert_almost_equal(ggg2.gam2_im, ggg.gam2_im)
    numpy.testing.assert_almost_equal(ggg2.gam3_im, ggg.gam3_im)
    numpy.testing.assert_almost_equal(ggg2.vargam, ggg.vargam)
    numpy.testing.assert_almost_equal(ggg2.weight, ggg.weight)
    numpy.testing.assert_almost_equal(ggg2.ntri, ggg.ntri)

if __name__ == '__main__':
    test_ggg()
