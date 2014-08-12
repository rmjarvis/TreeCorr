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
# 3. Neither the name of the {organization} nor the names of its
#    contributors may be used to endorse or promote products derived from
#    this software without specific prior written permission.


import numpy
import treecorr
import os

from test_helper import get_aardvark

def test_g2():
    # cf. http://adsabs.harvard.edu/abs/2002A%26A...389..729S for the basic formulae I use here.
    #
    # Use gamma_t(r) = A r^2/s^2 exp(-r^2/2s^2)
    # i.e. gamma(r) = -A exp(-r^2/2s^2) (x+iy)^2 / s^2
    #
    # The Fourier transform is: gamma~(k) = -2 pi A s^4 k^2 exp(-s^2 k^2/2) / L^2
    # P(k) = (1/2pi) <|kappa~(k)|^2> = 2 pi A^2 s^8 k^4 / L^4 exp(-s^2 k^2)
    # xi+(r) = (1/2pi) int( dk k P(k) J0(kr) ) 
    #        = pi/16 A^2 (s/L)^2 exp(-r^2/4s^2) (r^4 - 16r^2s^2 + 32s^4)/s^4
    # xi-(r) = (1/2pi) int( dk k P(k) J4(kr) ) 
    #        = pi/16 A^2 (s/L)^2 exp(-r^2/4s^2)
    # Note: I'm not sure I handled the L factors correctly, but the units at the end need
    # to be kappa^2, so it needs to be (s/L)^2. 

    ngal = 1000000
    A = 0.05
    s = 10. * treecorr.angle_units['arcmin']
    L = 50. * s  # Not infinity, so this introduces some error.  Our integrals were to infinity.
    numpy.random.seed(8675309)
    x = (numpy.random.random_sample(ngal)-0.5) * L
    y = (numpy.random.random_sample(ngal)-0.5) * L
    r2 = (x**2 + y**2)/s**2
    g1 = -A * numpy.exp(-r2/2.) * (x**2-y**2)/s**2
    g2 = -A * numpy.exp(-r2/2.) * (2.*x*y)/s**2

    cat = treecorr.Catalog(x=x, y=y, g1=g1, g2=g2)
    gg = treecorr.G2Correlation(bin_size=0.1, min_sep=1., max_sep=100., sep_units='arcmin',
                                verbose=2)
    gg.process(cat)
    r = numpy.exp(gg.meanlogr) * treecorr.angle_units['arcmin']
    temp = numpy.pi/16. * A**2 * (s/L)**2 * numpy.exp(-0.25*r**2/s**2)
    true_xip = temp * (r**4 - 16.*r**2*s**2 + 32.*s**4)/s**4
    true_xim = temp * r**4/s**4

    print 'gg.xim = ',gg.xim
    print 'true_xim = ',true_xim
    print 'ratio = ',gg.xim / true_xim
    print 'diff = ',gg.xim - true_xim
    print 'max diff = ',max(abs(gg.xim - true_xim))
    assert max(abs(gg.xim - true_xim)) < 3.e-7

    print 'gg.xip = ',gg.xip
    print 'true_xip = ',true_xip
    print 'ratio = ',gg.xip / true_xip
    print 'diff = ',gg.xip - true_xip
    print 'max diff = ',max(abs(gg.xip - true_xip))
    assert max(abs(gg.xip - true_xip)) < 3.e-7


def test_aardvark():

    # Eric Suchyta did a brute force calculation of the Aardvark catalog, so it is useful to
    # compare the output from my code with that.

    get_aardvark()
    file_name = os.path.join('data','Aardvark.fit')
    config = treecorr.read_config('Aardvark.params')
    cat1 = treecorr.Catalog(file_name, config)
    gg = treecorr.G2Correlation(config)
    gg.process(cat1)

    direct_file_name = os.path.join('data','Aardvark.direct')
    direct_data = numpy.loadtxt(direct_file_name)
    direct_xip = direct_data[:,3]
    direct_xim = direct_data[:,4]

    #print 'gg.xip = ',gg.xip
    #print 'direct.xip = ',direct_xip

    xip_err = gg.xip - direct_xip
    print 'xip_err = ',xip_err
    print 'max = ',max(abs(xip_err))
    assert max(abs(xip_err)) < 2.e-7

    xim_err = gg.xim - direct_xim
    print 'xim_err = ',xim_err
    print 'max = ',max(abs(xim_err))
    assert max(abs(xim_err)) < 1.e-7

    # However, after some back and forth about the calculation, we concluded that Eric hadn't
    # done the spherical trig correctly to get the shears relative to the great circle joining
    # the two positions.  So let's compare with my own brute force calculation (i.e. using
    # bin_slop = 0):
    # This also has the advantage that the radial bins are done the same way -- uniformly 
    # spaced in log of the chord distance, rather than the great circle distance.

    bs0_file_name = os.path.join('data','Aardvark.bs0')
    bs0_data = numpy.loadtxt(bs0_file_name)
    bs0_xip = bs0_data[:,2]
    bs0_xim = bs0_data[:,3]

    #print 'gg.xip = ',gg.xip
    #print 'bs0.xip = ',bs0_xip

    xip_err = gg.xip - bs0_xip
    print 'xip_err = ',xip_err
    print 'max = ',max(abs(xip_err))
    assert max(abs(xip_err)) < 1.e-7

    xim_err = gg.xim - bs0_xim
    print 'xim_err = ',xim_err
    print 'max = ',max(abs(xim_err))
    assert max(abs(xim_err)) < 5.e-8

    # As bin_slop decreases, the agreement should get even better.
    if __name__ == '__main__':
        # This test is slow, so only do it if running test_g2.py directly.
        config['bin_slop'] = 0.2
        gg = treecorr.G2Correlation(config)
        gg.process(cat1)

        #print 'gg.xip = ',gg.xip
        #print 'bs0.xip = ',bs0_xip

        xip_err = gg.xip - bs0_xip
        print 'xip_err = ',xip_err
        print 'max = ',max(abs(xip_err))
        assert max(abs(xip_err)) < 1.e-8

        xim_err = gg.xim - bs0_xim
        print 'xim_err = ',xim_err
        print 'max = ',max(abs(xim_err))
        assert max(abs(xim_err)) < 1.e-8

 
if __name__ == '__main__':
    test_g2()
    test_aardvark()
