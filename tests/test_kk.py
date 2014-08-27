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
    k2 = treecorr.K2Correlation(bin_size=0.1, min_sep=0.1, max_sep=10., sep_units='arcmin')
    k2.process(cat)
    print 'k2.xi = ',k2.xi
    numpy.testing.assert_almost_equal(k2.xi, A**2, decimal=10)

    # Now add some noise to the values. It should still work, but at slightly lower accuracy.
    kappa += 0.001 * (numpy.random.random_sample(ngal)-0.5)
    cat = treecorr.Catalog(x=x, y=y, k=kappa, x_units='arcmin', y_units='arcmin')
    k2.process(cat)
    print 'k2.xi = ',k2.xi
    numpy.testing.assert_almost_equal(k2.xi, A**2, decimal=6)


def test_k2():
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
    k2 = treecorr.K2Correlation(bin_size=0.1, min_sep=1., max_sep=50., sep_units='arcmin',
                                verbose=2)
    k2.process(cat)
    r = numpy.exp(k2.meanlogr)
    true_xi = numpy.pi * A**2 * (s/L)**2 * numpy.exp(-0.25*r**2/s**2)
    print 'k2.xi = ',k2.xi
    print 'true_xi = ',true_xi
    print 'ratio = ',k2.xi / true_xi
    print 'diff = ',k2.xi - true_xi
    print 'max diff = ',max(abs(k2.xi - true_xi))
    assert max(abs(k2.xi - true_xi)) < 5.e-7

    # Check that we get the same result using the corr2 executable:
    if __name__ == '__main__':
        cat.write(os.path.join('data','k2.dat'))
        import subprocess
        p = subprocess.Popen( ["corr2","k2.params"] )
        p.communicate()
        corr2_output = numpy.loadtxt(os.path.join('output','k2.out'))
        print 'k2.xi = ',k2.xi
        print 'from corr2 output = ',corr2_output[:,2]
        print 'ratio = ',corr2_output[:,2]/k2.xi
        print 'diff = ',corr2_output[:,2]-k2.xi
        numpy.testing.assert_almost_equal(corr2_output[:,2]/k2.xi, 1., decimal=3)


if __name__ == '__main__':
    test_constant()
    test_k2()
