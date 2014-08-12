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
from numpy import sin, cos, tan, arcsin, arccos, arctan, arctan2, pi

def test_single():
    # Use gamma_t(r) = gamma0 exp(-r^2/2r0^2) around a single lens
    # i.e. gamma(r) = -gamma0 exp(-r^2/2r0^2) (x+iy)^2/r^2

    nsource = 1000000
    gamma0 = 0.05
    r0 = 10. * treecorr.angle_units['arcmin']
    L = 5. * r0
    numpy.random.seed(8675309)
    x = (numpy.random.random_sample(nsource)-0.5) * L
    y = (numpy.random.random_sample(nsource)-0.5) * L
    r2 = (x**2 + y**2)
    gammat = gamma0 * numpy.exp(-0.5*r2/r0**2)
    g1 = -gammat * (x**2-y**2)/r2
    g2 = -gammat * (2.*x*y)/r2

    lens_cat = treecorr.Catalog(x=[0], y=[0])
    source_cat = treecorr.Catalog(x=x, y=y, g1=g1, g2=g2)
    ng = treecorr.NGCorrelation(bin_size=0.1, min_sep=1., max_sep=30., sep_units='arcmin',
                                verbose=2)
    ng.process(lens_cat, source_cat)

    r = numpy.exp(ng.meanlogr) * treecorr.angle_units['arcmin']
    true_gt = gamma0 * numpy.exp(-0.5*r**2/r0**2)

    print 'ng.xi = ',ng.xi
    print 'true_gammat = ',true_gt
    print 'ratio = ',ng.xi / true_gt
    print 'diff = ',ng.xi - true_gt
    print 'max diff = ',max(abs(ng.xi - true_gt))
    assert max(abs(ng.xi - true_gt)) < 3.e-4

    # Test the same calculation, but with the pairwise calcualtion:
    dx = (numpy.random.random_sample(nsource)-0.5) * L
    dx = (numpy.random.random_sample(nsource)-0.5) * L

    lens_cat2 = treecorr.Catalog(x=dx, y=dx)
    source_cat2 = treecorr.Catalog(x=x+dx, y=y+dx, g1=g1, g2=g2)
    ng2 = treecorr.NGCorrelation(bin_size=0.1, min_sep=1., max_sep=30., sep_units='arcmin',
                                verbose=2, pairwise=True)
    ng2.process(lens_cat2, source_cat2)

    print 'ng.xi = ',ng.xi
    print 'true_gammat = ',true_gt
    print 'ratio = ',ng2.xi / true_gt
    print 'diff = ',ng2.xi - true_gt
    print 'max diff = ',max(abs(ng2.xi - true_gt))
    # I don't really understand why this comes out slightly less accurate.
    # I would have thought it would be slightly more accurate because it doesn't use the
    # approximations intrinsic to the tree calculation.
    assert max(abs(ng2.xi - true_gt)) < 4.e-4


def test_spherical():
    # This is the same profile we used for test_single, but put into spherical coords.
    # We do the spherical trig by hand using the obvious formulae, rather than the clever
    # optimizations that are used by the TreeCorr code, thus serving as a useful test of
    # the latter.

    nsource = 1000000
    gamma0 = 0.05
    r0 = 10. * treecorr.angle_units['deg']
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

    ng = treecorr.NGCorrelation(bin_size=0.1, min_sep=1., max_sep=40., sep_units='deg',
                                verbose=2)
    r1 = numpy.exp(ng.logr) * treecorr.angle_units['deg']
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

        lens_cat = treecorr.Catalog(ra=[ra0], dec=[dec0])
        source_cat = treecorr.Catalog(ra=ra, dec=dec, g1=g1_sph, g2=g2_sph)
        ng.process(lens_cat, source_cat)

        print 'ra0, dec0 = ',ra0,dec0
        print 'ng.xi = ',ng.xi
        print 'true_gammat = ',true_gt
        print 'ratio = ',ng.xi / true_gt
        print 'diff = ',ng.xi - true_gt
        print 'max diff = ',max(abs(ng.xi - true_gt))
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

    lens_cat = treecorr.Catalog(ra=[ra0], dec=[dec0])
    source_cat = treecorr.Catalog(ra=ra, dec=dec, g1=gammat, g2=numpy.zeros_like(gammat))
    ng.process(lens_cat, source_cat)

    print 'ng.xi = ',ng.xi
    print 'true_gammat = ',true_gt
    print 'ratio = ',ng.xi / true_gt
    print 'diff = ',ng.xi - true_gt
    print 'max diff = ',max(abs(ng.xi - true_gt))
    assert max(abs(ng.xi - true_gt)) < 4.e-4


def test_ng():
    # Use gamma_t(r) = gamma0 exp(-r^2/2r0^2) around a bunch of foregroung lenses.
    # i.e. gamma(r) = -gamma0 exp(-r^2/2r0^2) (x+iy)^2/r^2

    nlens = 1000
    nsource = 100000
    gamma0 = 0.05
    r0 = 10. * treecorr.angle_units['arcmin']
    L = 5. * r0
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

    lens_cat = treecorr.Catalog(x=xl, y=yl)
    source_cat = treecorr.Catalog(x=xs, y=ys, g1=g1, g2=g2)
    ng = treecorr.NGCorrelation(bin_size=0.1, min_sep=1., max_sep=30., sep_units='arcmin',
                                verbose=2)
    ng.process(lens_cat, source_cat)

    r = numpy.exp(ng.meanlogr) * treecorr.angle_units['arcmin']
    true_gt = gamma0 * numpy.exp(-0.5*r**2/r0**2)

    print 'ng.xi = ',ng.xi
    print 'true_gammat = ',true_gt
    print 'ratio = ',ng.xi / true_gt
    print 'diff = ',ng.xi - true_gt
    print 'max diff = ',max(abs(ng.xi - true_gt))
    assert max(abs(ng.xi - true_gt)) < 3.e-3

if __name__ == '__main__':
    test_single()
    test_spherical()
    test_ng()
