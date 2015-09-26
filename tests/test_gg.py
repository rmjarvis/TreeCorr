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

from __future__ import print_function
import numpy
import treecorr
import os

from test_helper import get_aardvark
from numpy import sin, cos, tan, arcsin, arccos, arctan, arctan2, pi

def test_gg():
    # cf. http://adsabs.harvard.edu/abs/2002A%26A...389..729S for the basic formulae I use here.
    #
    # Use gamma_t(r) = gamma0 r^2/r0^2 exp(-r^2/2r0^2)
    # i.e. gamma(r) = -gamma0 exp(-r^2/2r0^2) (x+iy)^2 / r0^2
    #
    # The Fourier transform is: gamma~(k) = -2 pi gamma0 r0^4 k^2 exp(-r0^2 k^2/2) / L^2
    # P(k) = (1/2pi) <|gamma~(k)|^2> = 2 pi gamma0^2 r0^8 k^4 / L^4 exp(-r0^2 k^2)
    # xi+(r) = (1/2pi) int( dk k P(k) J0(kr) ) 
    #        = pi/16 gamma0^2 (r0/L)^2 exp(-r^2/4r0^2) (r^4 - 16r^2r0^2 + 32r0^4)/r0^4
    # xi-(r) = (1/2pi) int( dk k P(k) J4(kr) ) 
    #        = pi/16 gamma0^2 (r0/L)^2 exp(-r^2/4r0^2) r^4/r0^4
    # Note: I'm not sure I handled the L factors correctly, but the units at the end need
    # to be gamma^2, so it needs to be (r0/L)^2. 

    ngal = 1000000
    gamma0 = 0.05
    r0 = 10.
    L = 50. * r0  # Not infinity, so this introduces some error.  Our integrals were to infinity.
    numpy.random.seed(8675309)
    x = (numpy.random.random_sample(ngal)-0.5) * L
    y = (numpy.random.random_sample(ngal)-0.5) * L
    r2 = (x**2 + y**2)/r0**2
    g1 = -gamma0 * numpy.exp(-r2/2.) * (x**2-y**2)/r0**2
    g2 = -gamma0 * numpy.exp(-r2/2.) * (2.*x*y)/r0**2

    cat = treecorr.Catalog(x=x, y=y, g1=g1, g2=g2, x_units='arcmin', y_units='arcmin')
    gg = treecorr.GGCorrelation(bin_size=0.1, min_sep=1., max_sep=100., sep_units='arcmin',
                                verbose=2)
    gg.process(cat)
    r = numpy.exp(gg.meanlogr)
    temp = numpy.pi/16. * gamma0**2 * (r0/L)**2 * numpy.exp(-0.25*r**2/r0**2)
    true_xip = temp * (r**4 - 16.*r**2*r0**2 + 32.*r0**4)/r0**4
    true_xim = temp * r**4/r0**4

    print('gg.xip = ',gg.xip)
    print('true_xip = ',true_xip)
    print('ratio = ',gg.xip / true_xip)
    print('diff = ',gg.xip - true_xip)
    print('max diff = ',max(abs(gg.xip - true_xip)))
    assert max(abs(gg.xip - true_xip)) < 3.e-7
    print('xip_im = ',gg.xip_im)
    assert max(abs(gg.xip_im)) < 2.e-7

    print('gg.xim = ',gg.xim)
    print('true_xim = ',true_xim)
    print('ratio = ',gg.xim / true_xim)
    print('diff = ',gg.xim - true_xim)
    print('max diff = ',max(abs(gg.xim - true_xim)))
    assert max(abs(gg.xim - true_xim)) < 3.e-7
    print('xim_im = ',gg.xim_im)
    assert max(abs(gg.xim_im)) < 1.e-7

    # Check MapSq calculation:
    # cf. http://adsabs.harvard.edu/abs/2004MNRAS.352..338J
    # Use Crittenden formulation, since the analytic result is simpler:
    # Map^2(R) = int 1/2 r/R^2 (T+(r/R) xi+(r) + T-(r/R) xi-(r)) dr
    #          = 6 pi gamma0^2 r0^8 R^4 / (L^2 (r0^2+R^2)^5)
    # Mx^2(R)  = int 1/2 r/R^2 (T+(r/R) xi+(r) - T-(r/R) xi-(r)) dr
    #          = 0
    true_mapsq = 6.*numpy.pi * gamma0**2 * r0**8 * r**4 / (L**2 * (r**2+r0**2)**5)

    mapsq, mapsq_im, mxsq, mxsq_im, varmapsq = gg.calculateMapSq('Crittenden')
    print('mapsq = ',mapsq)
    print('true_mapsq = ',true_mapsq)
    print('ratio = ',mapsq/true_mapsq)
    print('diff = ',mapsq-true_mapsq)
    print('max diff = ',max(abs(mapsq - true_mapsq)))
    print('max diff[16:] = ',max(abs(mapsq[16:] - true_mapsq[16:])))
    # It's pretty ratty near the start where the integral is poorly evaluated, but the 
    # agreement is pretty good if we skip the first 16 elements.
    # Well, it gets bad again at the end, but those values are small enough that they still
    # pass this test.
    assert max(abs(mapsq[16:]-true_mapsq[16:])) < 3.e-8
    print('mxsq = ',mxsq)
    print('max = ',max(abs(mxsq)))
    print('max[16:] = ',max(abs(mxsq[16:])))
    assert max(abs(mxsq[16:])) < 3.e-8

    # Check that we get the same result using the corr2 executable:
    if __name__ == '__main__':
        cat.write(os.path.join('data','gg.dat'))
        import subprocess
        p = subprocess.Popen( ["corr2","gg.params"] )
        p.communicate()
        corr2_output = numpy.loadtxt(os.path.join('output','gg.out'))
        print('gg.xip = ',gg.xip)
        print('from corr2 output = ',corr2_output[:,2])
        print('ratio = ',corr2_output[:,2]/gg.xip)
        print('diff = ',corr2_output[:,2]-gg.xip)
        numpy.testing.assert_almost_equal(corr2_output[:,2]/gg.xip, 1., decimal=3)

        print('gg.xim = ',gg.xim)
        print('from corr2 output = ',corr2_output[:,3])
        print('ratio = ',corr2_output[:,3]/gg.xim)
        print('diff = ',corr2_output[:,3]-gg.xim)
        numpy.testing.assert_almost_equal(corr2_output[:,3]/gg.xim, 1., decimal=3)

        print('xip_im from corr2 output = ',corr2_output[:,4])
        print('max err = ',max(abs(corr2_output[:,4])))
        assert max(abs(corr2_output[:,4])) < 2.e-7
        print('xim_im from corr2 output = ',corr2_output[:,5])
        print('max err = ',max(abs(corr2_output[:,5])))
        assert max(abs(corr2_output[:,5])) < 1.e-7

        corr2_output2 = numpy.loadtxt(os.path.join('output','gg_m2.out'))
        print('mapsq = ',mapsq)
        print('from corr2 output = ',corr2_output2[:,1])
        print('ratio = ',corr2_output2[:,1]/mapsq)
        print('diff = ',corr2_output2[:,1]-mapsq)
        numpy.testing.assert_almost_equal(corr2_output2[:,1]/mapsq, 1., decimal=3)

        print('mxsq = ',mxsq)
        print('from corr2 output = ',corr2_output2[:,2])
        print('ratio = ',corr2_output2[:,2]/mxsq)
        print('diff = ',corr2_output2[:,2]-mxsq)
        numpy.testing.assert_almost_equal(corr2_output2[:,2]/mxsq, 1., decimal=3)

    # Also check the Schneider version.  The math isn't quite as nice here, but it is tractable
    # using a different formula than I used above:
    # Map^2(R) = int k P(k) W(kR) dk
    #          = 576 pi gamma0^2 r0^6/(L^2 R^10) exp(-R^2/2r0^2)
    #            x (I0(R^2/2r0^2) R^2 (R^4 + 96 r0^4) - 16 I1(R^2/2r0^2) r0^2 (R^4 + 24 r0^4)
    try:
        from scipy.special import i0,i1
        x = 0.5*r**2/r0**2
        true_mapsq = 576.*numpy.pi * gamma0**2 * r0**6 / (L**2 * r**10) * numpy.exp(-x)
        true_mapsq *= i0(x) * r**2 * (r**4 + 96.*r0**4) - 16.*i1(x) * r0**2 * (r**4 + 24.*r0**4)

        mapsq, mapsq_im, mxsq, mxsq_im, varmapsq = gg.calculateMapSq('Schneider')
        print('Schneider mapsq = ',mapsq)
        print('true_mapsq = ',true_mapsq)
        print('ratio = ',mapsq/true_mapsq)
        print('diff = ',mapsq-true_mapsq)
        print('max diff = ',max(abs(mapsq - true_mapsq)))
        print('max diff[20:] = ',max(abs(mapsq[20:] - true_mapsq[20:])))
        # This one stay ratty longer, so we need to skip the first 20 and also loosen the
        # test a bit.
        assert max(abs(mapsq[20:]-true_mapsq[20:])) < 7.e-8
        print('mxsq = ',mxsq)
        print('max = ',max(abs(mxsq)))
        print('max[20:] = ',max(abs(mxsq[20:])))
        assert max(abs(mxsq[20:])) < 7.e-8

    except ImportError:
        # Don't require scipy if the user doesn't have it.
        print('Skipping tests of Schneider aperture mass, since scipy.special not available.')


def test_spherical():
    # This is the same field we used for test_gg, but put into spherical coords.
    # We do the spherical trig by hand using the obvious formulae, rather than the clever
    # optimizations that are used by the TreeCorr code, thus serving as a useful test of
    # the latter.

    nsource = 1000000
    gamma0 = 0.05
    r0 = 10. * treecorr.arcmin
    L = 50. * r0
    numpy.random.seed(8675309)
    x = (numpy.random.random_sample(nsource)-0.5) * L
    y = (numpy.random.random_sample(nsource)-0.5) * L
    r2 = x**2 + y**2
    g1 = -gamma0 * numpy.exp(-r2/2./r0**2) * (x**2-y**2)/r0**2
    g2 = -gamma0 * numpy.exp(-r2/2./r0**2) * (2.*x*y)/r0**2
    r = numpy.sqrt(r2)
    theta = arctan2(y,x)

    gg = treecorr.GGCorrelation(bin_size=0.1, min_sep=1., max_sep=100., sep_units='arcmin',
                                verbose=2)
    r1 = numpy.exp(gg.logr) * treecorr.arcmin
    temp = numpy.pi/16. * gamma0**2 * (r0/L)**2 * numpy.exp(-0.25*r1**2/r0**2)
    true_xip = temp * (r1**4 - 16.*r1**2*r0**2 + 32.*r0**4)/r0**4
    true_xim = temp * r1**4/r0**4

    # Test this around several central points
    if __name__ == '__main__':
        ra0_list = [ 0., 1., 1.3, 232., 0. ]
        dec0_list = [ 0., -0.3, 1.3, -1.4, pi/2.-1.e-6 ]
    else:
        ra0_list = [ 0., 1.]
        dec0_list = [ 0., -0.3]

    for ra0, dec0 in zip(ra0_list, dec0_list):

        # Use spherical triangle with A = point, B = (ra0,dec0), C = N. pole
        # a = Pi/2-dec0
        # c = 2*asin(r/2)  (lambert projection)
        # B = Pi/2 - theta

        c = 2.*arcsin(r/2.)
        a = pi/2. - dec0
        B = pi/2. - theta
        B[x<0] *= -1.
        B[B<-pi] += 2.*pi
        B[B>pi] -= 2.*pi

        # Solve the rest of the triangle with spherical trig:
        cosb = cos(a)*cos(c) + sin(a)*sin(c)*cos(B)
        b = arccos(cosb)
        cosA = (cos(a) - cos(b)*cos(c)) / (sin(b)*sin(c))
        #A = arccos(cosA)
        A = numpy.zeros_like(cosA)
        A[abs(cosA)<1] = arccos(cosA[abs(cosA)<1])
        A[cosA<=-1] = pi
        cosC = (cos(c) - cos(a)*cos(b)) / (sin(a)*sin(b))
        #C = arccos(cosC)
        C = numpy.zeros_like(cosC)
        C[abs(cosC)<1] = arccos(cosC[abs(cosC)<1])
        C[cosC<=-1] = pi
        C[x<0] *= -1.

        ra = ra0 - C
        dec = pi/2. - b

        # Rotate shear relative to local west
        # gamma_sph = exp(2i beta) * gamma
        # where beta = pi - (A+B) is the angle between north and "up" in the tangent plane.
        beta = pi - (A+B)
        beta[x>0] *= -1.
        cos2beta = cos(2.*beta)
        sin2beta = sin(2.*beta)
        g1_sph = g1 * cos2beta - g2 * sin2beta
        g2_sph = g2 * cos2beta + g1 * sin2beta

        cat = treecorr.Catalog(ra=ra, dec=dec, g1=g1_sph, g2=g2_sph, ra_units='rad', 
                               dec_units='rad')
        gg = treecorr.GGCorrelation(bin_size=0.1, min_sep=1., max_sep=100., sep_units='arcmin',
                                    verbose=2)
        gg.process(cat)

        print('ra0, dec0 = ',ra0,dec0)
        print('gg.xip = ',gg.xip)
        print('true_xip = ',true_xip)
        print('ratio = ',gg.xip / true_xip)
        print('diff = ',gg.xip - true_xip)
        print('max diff = ',max(abs(gg.xip - true_xip)))
        # The 3rd and 4th centers are somewhat less accurate.  Not sure why.
        # The math seems to be right, since the last one that gets all the way to the pole
        # works, so I'm not sure what is going on.  It's just a few bins that get a bit less
        # accurate.  Possibly worth investigating further at some point...
        assert max(abs(gg.xip - true_xip)) < 3.e-7

        print('gg.xim = ',gg.xim)
        print('true_xim = ',true_xim)
        print('ratio = ',gg.xim / true_xim)
        print('diff = ',gg.xim - true_xim)
        print('max diff = ',max(abs(gg.xim - true_xim)))
        assert max(abs(gg.xim - true_xim)) < 2.e-7

    # One more center that can be done very easily.  If the center is the north pole, then all
    # the tangential shears are pure (positive) g1.
    ra0 = 0
    dec0 = pi/2.
    ra = theta
    dec = pi/2. - 2.*arcsin(r/2.)
    gammat = -gamma0 * r2/r0**2 * numpy.exp(-r2/2./r0**2)

    cat = treecorr.Catalog(ra=ra, dec=dec, g1=gammat, g2=numpy.zeros_like(gammat), ra_units='rad',
                           dec_units='rad')
    gg.process(cat)

    print('gg.xip = ',gg.xip)
    print('gg.xip_im = ',gg.xip_im)
    print('true_xip = ',true_xip)
    print('ratio = ',gg.xip / true_xip)
    print('diff = ',gg.xip - true_xip)
    print('max diff = ',max(abs(gg.xip - true_xip)))
    assert max(abs(gg.xip - true_xip)) < 3.e-7
    assert max(abs(gg.xip_im)) < 3.e-7

    print('gg.xim = ',gg.xim)
    print('gg.xim_im = ',gg.xim_im)
    print('true_xim = ',true_xim)
    print('ratio = ',gg.xim / true_xim)
    print('diff = ',gg.xim - true_xim)
    print('max diff = ',max(abs(gg.xim - true_xim)))
    assert max(abs(gg.xim - true_xim)) < 2.e-7
    assert max(abs(gg.xim_im)) < 2.e-7

    # Check that we get the same result using the corr2 executable:
    if __name__ == '__main__':
        cat.write(os.path.join('data','gg_spherical.dat'))
        import subprocess
        p = subprocess.Popen( ["corr2","gg_spherical.params"] )
        p.communicate()
        corr2_output = numpy.loadtxt(os.path.join('output','gg_spherical.out'))
        print('gg.xip = ',gg.xip)
        print('from corr2 output = ',corr2_output[:,2])
        print('ratio = ',corr2_output[:,2]/gg.xip)
        print('diff = ',corr2_output[:,2]-gg.xip)
        numpy.testing.assert_almost_equal(corr2_output[:,2]/gg.xip, 1., decimal=3)

        print('gg.xim = ',gg.xim)
        print('from corr2 output = ',corr2_output[:,3])
        print('ratio = ',corr2_output[:,3]/gg.xim)
        print('diff = ',corr2_output[:,3]-gg.xim)
        numpy.testing.assert_almost_equal(corr2_output[:,3]/gg.xim, 1., decimal=3)

        print('xip_im from corr2 output = ',corr2_output[:,4])
        assert max(abs(corr2_output[:,4])) < 3.e-7

        print('xim_im from corr2 output = ',corr2_output[:,5])
        assert max(abs(corr2_output[:,5])) < 2.e-7



def test_aardvark():

    # Eric Suchyta did a brute force calculation of the Aardvark catalog, so it is useful to
    # compare the output from my code with that.

    get_aardvark()
    file_name = os.path.join('data','Aardvark.fit')
    config = treecorr.read_config('Aardvark.params')
    cat1 = treecorr.Catalog(file_name, config)
    gg = treecorr.GGCorrelation(config)
    gg.process(cat1)

    direct_file_name = os.path.join('data','Aardvark.direct')
    direct_data = numpy.loadtxt(direct_file_name)
    direct_xip = direct_data[:,3]
    direct_xim = direct_data[:,4]

    #print 'gg.xip = ',gg.xip
    #print 'direct.xip = ',direct_xip

    xip_err = gg.xip - direct_xip
    print('xip_err = ',xip_err)
    print('max = ',max(abs(xip_err)))
    assert max(abs(xip_err)) < 2.e-7
    print('xip_im = ',gg.xip_im)
    print('max = ',max(abs(gg.xip_im)))
    assert max(abs(gg.xip_im)) < 3.e-7

    xim_err = gg.xim - direct_xim
    print('xim_err = ',xim_err)
    print('max = ',max(abs(xim_err)))
    assert max(abs(xim_err)) < 1.e-7
    print('xim_im = ',gg.xim_im)
    print('max = ',max(abs(gg.xim_im)))
    assert max(abs(gg.xim_im)) < 1.e-7

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
    print('xip_err = ',xip_err)
    print('max = ',max(abs(xip_err)))
    assert max(abs(xip_err)) < 1.e-7

    xim_err = gg.xim - bs0_xim
    print('xim_err = ',xim_err)
    print('max = ',max(abs(xim_err)))
    assert max(abs(xim_err)) < 5.e-8

    # Check that we get the same result using the corr2 executable:
    # Note: This is the only test of the corr2 executable that we do with nosetests.
    # The other similar tests are blocked out with: if __name__ == '__main__':
    import subprocess
    p = subprocess.Popen( ["corr2","Aardvark.params"] )
    p.communicate()
    corr2_output = numpy.loadtxt(os.path.join('output','Aardvark.out'))
    print('gg.xip = ',gg.xip)
    print('from corr2 output = ',corr2_output[:,2])
    print('ratio = ',corr2_output[:,2]/gg.xip)
    print('diff = ',corr2_output[:,2]-gg.xip)
    numpy.testing.assert_almost_equal(corr2_output[:,2]/gg.xip, 1., decimal=3)

    print('gg.xim = ',gg.xim)
    print('from corr2 output = ',corr2_output[:,3])
    print('ratio = ',corr2_output[:,3]/gg.xim)
    print('diff = ',corr2_output[:,3]-gg.xim)
    numpy.testing.assert_almost_equal(corr2_output[:,3]/gg.xim, 1., decimal=3)

    print('xip_im from corr2 output = ',corr2_output[:,4])
    print('max err = ',max(abs(corr2_output[:,4])))
    assert max(abs(corr2_output[:,4])) < 3.e-7
    print('xim_im from corr2 output = ',corr2_output[:,5])
    print('max err = ',max(abs(corr2_output[:,5])))
    assert max(abs(corr2_output[:,5])) < 1.e-7

    # As bin_slop decreases, the agreement should get even better.
    # This test is slow, so only do it if running test_gg.py directly.
    if __name__ == '__main__':
        config['bin_slop'] = 0.2
        gg = treecorr.GGCorrelation(config)
        gg.process(cat1)

        #print 'gg.xip = ',gg.xip
        #print 'bs0.xip = ',bs0_xip

        xip_err = gg.xip - bs0_xip
        print('xip_err = ',xip_err)
        print('max = ',max(abs(xip_err)))
        assert max(abs(xip_err)) < 1.e-8

        xim_err = gg.xim - bs0_xim
        print('xim_err = ',xim_err)
        print('max = ',max(abs(xim_err)))
        assert max(abs(xim_err)) < 1.e-8

 
if __name__ == '__main__':
    test_gg()
    test_spherical()
    test_aardvark()
