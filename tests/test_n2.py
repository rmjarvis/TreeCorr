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

def test_n2():
    # Use n(r) = A exp(-r^2/2s^2)
    #
    # The Fourier transform is: n~(k) = 2 pi A s^2 exp(-s^2 k^2/2) / L^2
    # P(k) = (1/2pi) <|n~(k)|^2> = 2 pi A^2 (s/L)^4 exp(-s^2 k^2)
    # xi(r) = (1/2pi) int( dk k P(k) J0(kr) ) 
    #       = pi A^2 (s/L)^2 exp(-r^2/2s^2/4)
    #
    # Note that this time, A is not arbitrary.  n(r) needs to integrate to L^2.
    # So A = (L/s)^2 / 2pi
    # xi(r) = 1/4pi (L/s)^2 exp(-r^2/2s^2/4)
    # 
    # Also, we need to correct for the uniform density background, so the real result
    # is this minus 1.
    #
    # xi(r) = 1/4pi (L/s)^2 exp(-r^2/2s^2/4) - 1

    ngal = 1000000
    s = 10.
    L = 50. * s  # Not infinity, so this introduces some error.  Our integrals were to infinity.
    numpy.random.seed(8675309)
    x = numpy.random.normal(0,s, (ngal,) )
    y = numpy.random.normal(0,s, (ngal,) )

    cat = treecorr.Catalog(x=x, y=y, x_units='arcmin', y_units='arcmin')
    dd = treecorr.N2Correlation(bin_size=0.1, min_sep=1., max_sep=25., sep_units='arcmin',
                                verbose=2)
    dd.process(cat)
    print 'dd.npairs = ',dd.npairs

    nrand = 5 * ngal
    rx = (numpy.random.random_sample(nrand)-0.5) * L
    ry = (numpy.random.random_sample(nrand)-0.5) * L
    rand = treecorr.Catalog(x=rx,y=ry, x_units='arcmin', y_units='arcmin')
    rr = treecorr.N2Correlation(bin_size=0.1, min_sep=1., max_sep=25., sep_units='arcmin',
                                verbose=2)
    rr.process(rand)
    print 'rr.npairs = ',rr.npairs

    dr = treecorr.N2Correlation(bin_size=0.1, min_sep=1., max_sep=25., sep_units='arcmin',
                                verbose=2)
    dr.process(cat,rand)
    print 'dr.npairs = ',dr.npairs

    r = numpy.exp(dd.meanlogr)
    true_xi = 0.25/numpy.pi * (L/s)**2 * numpy.exp(-0.25*r**2/s**2) - 1.

    xi, varxi = dd.calculateXi(rr,dr)
    print 'xi = ',xi
    print 'true_xi = ',true_xi
    print 'ratio = ',xi / true_xi
    print 'diff = ',xi - true_xi
    print 'max rel diff = ',max(abs((xi - true_xi)/true_xi))
    # This isn't super accurate.  But the agreement improves as L increase, so I think it is 
    # merely a matter of the finite field and the integrals going to infinity.  (Sort of, since
    # we still have L in there.)
    assert max(abs(xi - true_xi)/true_xi) < 0.1

    simple_xi, varxi = dd.calculateXi(rr)
    print 'simple xi = ',simple_xi
    #print 'true_xi = ',true_xi
    #print 'ratio = ',simple_xi / true_xi
    #print 'diff = ',simple_xi - true_xi
    print 'max rel diff = ',max(abs((simple_xi - true_xi)/true_xi))
    # The simple calculation (i.e. dd/rr-1, rather than (dd-2dr+rr)/rr as above) is only 
    # slightly less accurate in this case.  Probably because the mask is simple (a box), so
    # the difference is relatively minor.  The error is slightly higher in this case, but testing
    # that it is everywhere < 0.1 is still appropriate.
    assert max(abs(simple_xi - true_xi)/true_xi) < 0.1

    # Check that we get the same result using the corr2 executable:
    if __name__ == '__main__':
        cat.write(os.path.join('data','n2_data.dat'))
        rand.write(os.path.join('data','n2_rand.dat'))
        import subprocess
        p = subprocess.Popen( ["corr2","n2.params"] )
        p.communicate()
        corr2_output = numpy.loadtxt(os.path.join('output','n2.out'))
        print 'xi = ',xi
        print 'from corr2 output = ',corr2_output[:,2]
        print 'ratio = ',corr2_output[:,2]/xi
        print 'diff = ',corr2_output[:,2]-xi
        numpy.testing.assert_almost_equal(corr2_output[:,2]/xi, 1., decimal=3)


if __name__ == '__main__':
    test_n2()
