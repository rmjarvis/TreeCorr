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

def test_single():
    # Use gamma_t(r) = gamma0 exp(-r^2/2r0^2) around a single lens
    # i.e. gamma(r) = -gamma0 exp(-r^2/2r0^2) (x+iy)^2/r^2

    nsource = 1000000
    gamma0 = 0.05
    r0 = 10.
    L = 5. * r0
    numpy.random.seed(8675309)
    x = (numpy.random.random_sample(nsource)-0.5) * L
    y = (numpy.random.random_sample(nsource)-0.5) * L
    r2 = (x**2 + y**2)
    gammat = gamma0 * numpy.exp(-0.5*r2/r0**2)
    g1 = -gammat * (x**2-y**2)/r2
    g2 = -gammat * (2.*x*y)/r2

    lens_cat = treecorr.Catalog(x=[0], y=[0], x_units='arcmin', y_units='arcmin')
    source_cat = treecorr.Catalog(x=x, y=y, g1=g1, g2=g2, x_units='arcmin', y_units='arcmin')
    ng = treecorr.NGCorrelation(bin_size=0.1, min_sep=1., max_sep=25., sep_units='arcmin',
                                verbose=2)
    ng.process(lens_cat, source_cat)

    r = numpy.exp(ng.meanlogr)
    true_gt = gamma0 * numpy.exp(-0.5*r**2/r0**2)

    print('ng.xi = ',ng.xi)
    print('ng.xi_im = ',ng.xi_im)
    print('true_gammat = ',true_gt)
    print('ratio = ',ng.xi / true_gt)
    print('diff = ',ng.xi - true_gt)
    print('max diff = ',max(abs(ng.xi - true_gt)))
    assert max(abs(ng.xi - true_gt)) < 4.e-4
    assert max(abs(ng.xi_im)) < 3.e-5

    # Check that we get the same result using the corr2 executable:
    if __name__ == '__main__':
        lens_cat.write(os.path.join('data','ng_single_lens.dat'))
        source_cat.write(os.path.join('data','ng_single_source.dat'))
        import subprocess
        p = subprocess.Popen( ["corr2","ng_single.params"] )
        p.communicate()
        corr2_output = numpy.loadtxt(os.path.join('output','ng_single.out'))
        print('ng.xi = ',ng.xi)
        print('from corr2 output = ',corr2_output[:,2])
        print('ratio = ',corr2_output[:,2]/ng.xi)
        print('diff = ',corr2_output[:,2]-ng.xi)
        numpy.testing.assert_almost_equal(corr2_output[:,2]/ng.xi, 1., decimal=3)

        print('xi_im from corr2 output = ',corr2_output[:,3])
        assert max(abs(corr2_output[:,3])) < 3.e-5


def test_pairwise():
    # Test the same profile, but with the pairwise calcualtion:
    nsource = 1000000
    gamma0 = 0.05
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

    lens_cat = treecorr.Catalog(x=dx, y=dx, x_units='arcmin', y_units='arcmin')
    source_cat = treecorr.Catalog(x=x+dx, y=y+dx, g1=g1, g2=g2, x_units='arcmin', y_units='arcmin')
    ng = treecorr.NGCorrelation(bin_size=0.1, min_sep=1., max_sep=25., sep_units='arcmin',
                                verbose=2, pairwise=True)
    ng.process(lens_cat, source_cat)

    r = numpy.exp(ng.meanlogr)
    true_gt = gamma0 * numpy.exp(-0.5*r**2/r0**2)

    print('ng.xi = ',ng.xi)
    print('ng.xi_im = ',ng.xi_im)
    print('true_gammat = ',true_gt)
    print('ratio = ',ng.xi / true_gt)
    print('diff = ',ng.xi - true_gt)
    print('max diff = ',max(abs(ng.xi - true_gt)))
    # I don't really understand why this comes out slightly less accurate.
    # I would have thought it would be slightly more accurate because it doesn't use the
    # approximations intrinsic to the tree calculation.
    assert max(abs(ng.xi - true_gt)) < 4.e-4
    assert max(abs(ng.xi_im)) < 3.e-5

    # Check that we get the same result using the corr2 executable:
    if __name__ == '__main__':
        lens_cat.write(os.path.join('data','ng_pairwise_lens.dat'))
        source_cat.write(os.path.join('data','ng_pairwise_source.dat'))
        import subprocess
        p = subprocess.Popen( ["corr2","ng_pairwise.params"] )
        p.communicate()
        corr2_output = numpy.loadtxt(os.path.join('output','ng_pairwise.out'))
        print('ng.xi = ',ng.xi)
        print('from corr2 output = ',corr2_output[:,2])
        print('ratio = ',corr2_output[:,2]/ng.xi)
        print('diff = ',corr2_output[:,2]-ng.xi)
        numpy.testing.assert_almost_equal(corr2_output[:,2]/ng.xi, 1., decimal=3)

        print('xi_im from corr2 output = ',corr2_output[:,3])
        assert max(abs(corr2_output[:,3])) < 3.e-5


def test_spherical():
    # This is the same profile we used for test_single, but put into spherical coords.
    # We do the spherical trig by hand using the obvious formulae, rather than the clever
    # optimizations that are used by the TreeCorr code, thus serving as a useful test of
    # the latter.

    nsource = 1000000
    gamma0 = 0.05
    r0 = 10. * treecorr.degrees
    L = 5. * r0
    numpy.random.seed(8675309)
    x = (numpy.random.random_sample(nsource)-0.5) * L
    y = (numpy.random.random_sample(nsource)-0.5) * L
    r2 = (x**2 + y**2)
    gammat = gamma0 * numpy.exp(-0.5*r2/r0**2)
    g1 = -gammat * (x**2-y**2)/r2
    g2 = -gammat * (2.*x*y)/r2
    r = numpy.sqrt(r2)
    theta = arctan2(y,x)

    ng = treecorr.NGCorrelation(bin_size=0.1, min_sep=1., max_sep=25., sep_units='deg',
                                verbose=2)
    r1 = numpy.exp(ng.logr) * treecorr.degrees
    true_gt = gamma0 * numpy.exp(-0.5*r1**2/r0**2)

    # Test this around several central points
    # (For now just one -- on the equator)
    ra0_list = [ 0., 1., 1.3, 232., 0. ]
    dec0_list = [ 0., -0.3, 1.3, -1.4, pi/2.-1.e-6 ]
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

        lens_cat = treecorr.Catalog(ra=[ra0], dec=[dec0], ra_units='rad', dec_units='rad')
        source_cat = treecorr.Catalog(ra=ra, dec=dec, g1=g1_sph, g2=g2_sph,
                                      ra_units='rad', dec_units='rad')
        ng.process(lens_cat, source_cat)

        print('ra0, dec0 = ',ra0,dec0)
        print('ng.xi = ',ng.xi)
        print('true_gammat = ',true_gt)
        print('ratio = ',ng.xi / true_gt)
        print('diff = ',ng.xi - true_gt)
        print('max diff = ',max(abs(ng.xi - true_gt)))
        # The 3rd and 4th centers are somewhat less accurate.  Not sure why.
        # The math seems to be right, since the last one that gets all the way to the pole
        # works, so I'm not sure what is going on.  It's just a few bins that get a bit less
        # accurate.  Possibly worth investigating further at some point...
        assert max(abs(ng.xi - true_gt)) < 2.e-3

    # One more center that can be done very easily.  If the center is the north pole, then all
    # the tangential shears are pure (positive) g1.
    ra0 = 0
    dec0 = pi/2.
    ra = theta
    dec = pi/2. - 2.*arcsin(r/2.)

    lens_cat = treecorr.Catalog(ra=[ra0], dec=[dec0], ra_units='rad', dec_units='rad')
    source_cat = treecorr.Catalog(ra=ra, dec=dec, g1=gammat, g2=numpy.zeros_like(gammat),
                                  ra_units='rad', dec_units='rad')
    ng.process(lens_cat, source_cat)

    print('ng.xi = ',ng.xi)
    print('ng.xi_im = ',ng.xi_im)
    print('true_gammat = ',true_gt)
    print('ratio = ',ng.xi / true_gt)
    print('diff = ',ng.xi - true_gt)
    print('max diff = ',max(abs(ng.xi - true_gt)))
    assert max(abs(ng.xi - true_gt)) < 1.e-3
    assert max(abs(ng.xi_im)) < 3.e-5

    # Check that we get the same result using the corr2 executable:
    if __name__ == '__main__':
        lens_cat.write(os.path.join('data','ng_spherical_lens.dat'))
        source_cat.write(os.path.join('data','ng_spherical_source.dat'))
        import subprocess
        p = subprocess.Popen( ["corr2","ng_spherical.params"] )
        p.communicate()
        corr2_output = numpy.loadtxt(os.path.join('output','ng_spherical.out'))
        print('ng.xi = ',ng.xi)
        print('from corr2 output = ',corr2_output[:,2])
        print('ratio = ',corr2_output[:,2]/ng.xi)
        print('diff = ',corr2_output[:,2]-ng.xi)
        numpy.testing.assert_almost_equal(corr2_output[:,2]/ng.xi, 1., decimal=3)

        print('xi_im from corr2 output = ',corr2_output[:,3])
        assert max(abs(corr2_output[:,3])) < 3.e-5


def test_ng():
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
    for x,y in zip(xl,yl):
        dx = xs-x
        dy = ys-y
        r2 = dx**2 + dy**2
        gammat = gamma0 * numpy.exp(-0.5*r2/r0**2)
        g1 += -gammat * (dx**2-dy**2)/r2
        g2 += -gammat * (2.*dx*dy)/r2

    lens_cat = treecorr.Catalog(x=xl, y=yl, x_units='arcmin', y_units='arcmin')
    source_cat = treecorr.Catalog(x=xs, y=ys, g1=g1, g2=g2, x_units='arcmin', y_units='arcmin')
    ng = treecorr.NGCorrelation(bin_size=0.1, min_sep=1., max_sep=25., sep_units='arcmin',
                                verbose=2)
    ng.process(lens_cat, source_cat)

    r = numpy.exp(ng.meanlogr)
    true_gt = gamma0 * numpy.exp(-0.5*r**2/r0**2)

    print('ng.xi = ',ng.xi)
    print('ng.xi_im = ',ng.xi_im)
    print('true_gammat = ',true_gt)
    print('ratio = ',ng.xi / true_gt)
    print('diff = ',ng.xi - true_gt)
    print('max diff = ',max(abs(ng.xi - true_gt)))
    assert max(abs(ng.xi - true_gt)) < 4.e-3
    assert max(abs(ng.xi_im)) < 4.e-3

    nrand = nlens * 13
    xr = (numpy.random.random_sample(nrand)-0.5) * L
    yr = (numpy.random.random_sample(nrand)-0.5) * L
    rand_cat = treecorr.Catalog(x=xr, y=yr, x_units='arcmin', y_units='arcmin')
    rg = treecorr.NGCorrelation(bin_size=0.1, min_sep=1., max_sep=25., sep_units='arcmin',
                                verbose=2)
    rg.process(rand_cat, source_cat)
    print('rg.xi = ',rg.xi)
    xi, xi_im, varxi = ng.calculateXi(rg)
    print('compensated xi = ',xi)
    print('compensated xi_im = ',xi_im)
    print('true_gammat = ',true_gt)
    print('ratio = ',xi / true_gt)
    print('diff = ',xi - true_gt)
    print('max diff = ',max(abs(xi - true_gt)))
    # It turns out this doesn't come out much better.  I think the imprecision is mostly just due
    # to the smallish number of lenses, not to edge effects
    assert max(abs(xi - true_gt)) < 4.e-3
    assert max(abs(xi_im)) < 4.e-3

    # Check that we get the same result using the corr2 executable:
    if __name__ == '__main__':
        lens_cat.write(os.path.join('data','ng_lens.dat'))
        source_cat.write(os.path.join('data','ng_source.dat'))
        rand_cat.write(os.path.join('data','ng_rand.dat'))
        import subprocess
        p = subprocess.Popen( ["corr2","ng.params"] )
        p.communicate()
        corr2_output = numpy.loadtxt(os.path.join('output','ng.out'))
        print('ng.xi = ',ng.xi)
        print('xi = ',xi)
        print('from corr2 output = ',corr2_output[:,2])
        print('ratio = ',corr2_output[:,2]/xi)
        print('diff = ',corr2_output[:,2]-xi)
        numpy.testing.assert_almost_equal(corr2_output[:,2]/xi, 1., decimal=3)

        print('xi_im from corr2 output = ',corr2_output[:,3])
        assert max(abs(corr2_output[:,3])) < 4.e-3


if __name__ == '__main__':
    test_single()
    test_pairwise()
    test_spherical()
    test_ng()
