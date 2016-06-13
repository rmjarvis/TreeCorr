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
import fitsio

from test_helper import get_script_name
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
                                verbose=1)
    ng.process(lens_cat, source_cat)

    # log(<R>) != <logR>, but it should be close:
    print('meanlogr - log(meanr) = ',ng.meanlogr - numpy.log(ng.meanr))
    numpy.testing.assert_almost_equal(ng.meanlogr, numpy.log(ng.meanr), decimal=3)

    r = ng.meanr
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
        corr2_exe = get_script_name('corr2')
        p = subprocess.Popen( [corr2_exe,"ng_single.yaml"] )
        p.communicate()
        corr2_output = numpy.genfromtxt(os.path.join('output','ng_single.out'),names=True)
        print('ng.xi = ',ng.xi)
        print('from corr2 output = ',corr2_output['gamT'])
        print('ratio = ',corr2_output['gamT']/ng.xi)
        print('diff = ',corr2_output['gamT']-ng.xi)
        numpy.testing.assert_almost_equal(corr2_output['gamT']/ng.xi, 1., decimal=3)

        print('xi_im from corr2 output = ',corr2_output['gamX'])
        assert max(abs(corr2_output['gamX'])) < 3.e-5


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
                                verbose=1, pairwise=True)
    ng.process(lens_cat, source_cat)

    r = ng.meanr
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
        corr2_exe = get_script_name('corr2')
        p = subprocess.Popen( [corr2_exe,"ng_pairwise.yaml"] )
        p.communicate()
        corr2_output = numpy.genfromtxt(os.path.join('output','ng_pairwise.out'),names=True)
        print('ng.xi = ',ng.xi)
        print('from corr2 output = ',corr2_output['gamT'])
        print('ratio = ',corr2_output['gamT']/ng.xi)
        print('diff = ',corr2_output['gamT']-ng.xi)
        numpy.testing.assert_almost_equal(corr2_output['gamT']/ng.xi, 1., decimal=3)

        print('xi_im from corr2 output = ',corr2_output['gamX'])
        assert max(abs(corr2_output['gamX'])) < 3.e-5


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
                                verbose=1)
    r1 = numpy.exp(ng.logr) * treecorr.degrees
    true_gt = gamma0 * numpy.exp(-0.5*r1**2/r0**2)

    # Test this around several central points
    if __name__ == '__main__':
        ra0_list = [ 0., 1., 1.3, 232., 0. ]
        dec0_list = [ 0., -0.3, 1.3, -1.4, pi/2.-1.e-6 ]
    else:
        ra0_list = [ 232 ]
        dec0_list = [ -1.4 ]
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
        corr2_exe = get_script_name('corr2')
        p = subprocess.Popen( [corr2_exe,"ng_spherical.yaml"] )
        p.communicate()
        corr2_output = numpy.genfromtxt(os.path.join('output','ng_spherical.out'),names=True)
        print('ng.xi = ',ng.xi)
        print('from corr2 output = ',corr2_output['gamT'])
        print('ratio = ',corr2_output['gamT']/ng.xi)
        print('diff = ',corr2_output['gamT']-ng.xi)
        numpy.testing.assert_almost_equal(corr2_output['gamT']/ng.xi, 1., decimal=3)

        print('xi_im from corr2 output = ',corr2_output['gamX'])
        assert max(abs(corr2_output['gamX'])) < 3.e-5


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
                                verbose=1)
    ng.process(lens_cat, source_cat)

    r = ng.meanr
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
                                verbose=1)
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
        corr2_exe = get_script_name('corr2')
        p = subprocess.Popen( [corr2_exe,"ng.yaml"] )
        p.communicate()
        corr2_output = numpy.genfromtxt(os.path.join('output','ng.out'),names=True)
        print('ng.xi = ',ng.xi)
        print('xi = ',xi)
        print('from corr2 output = ',corr2_output['gamT'])
        print('ratio = ',corr2_output['gamT']/xi)
        print('diff = ',corr2_output['gamT']-xi)
        numpy.testing.assert_almost_equal(corr2_output['gamT']/xi, 1., decimal=3)

        print('xi_im from corr2 output = ',corr2_output['gamX'])
        assert max(abs(corr2_output['gamX'])) < 4.e-3

    # Check the fits write option
    out_file_name1 = os.path.join('output','ng_out1.fits')
    ng.write(out_file_name1)
    data = fitsio.read(out_file_name1)
    numpy.testing.assert_almost_equal(data['R_nom'], numpy.exp(ng.logr))
    numpy.testing.assert_almost_equal(data['meanR'], ng.meanr)
    numpy.testing.assert_almost_equal(data['meanlogR'], ng.meanlogr)
    numpy.testing.assert_almost_equal(data['gamT'], ng.xi)
    numpy.testing.assert_almost_equal(data['gamX'], ng.xi_im)
    numpy.testing.assert_almost_equal(data['sigma'], numpy.sqrt(ng.varxi))
    numpy.testing.assert_almost_equal(data['weight'], ng.weight)
    numpy.testing.assert_almost_equal(data['npairs'], ng.npairs)

    out_file_name2 = os.path.join('output','ng_out2.fits')
    ng.write(out_file_name2, rg)
    data = fitsio.read(out_file_name2)
    numpy.testing.assert_almost_equal(data['R_nom'], numpy.exp(ng.logr))
    numpy.testing.assert_almost_equal(data['meanR'], ng.meanr)
    numpy.testing.assert_almost_equal(data['meanlogR'], ng.meanlogr)
    numpy.testing.assert_almost_equal(data['gamT'], xi)
    numpy.testing.assert_almost_equal(data['gamX'], xi_im)
    numpy.testing.assert_almost_equal(data['sigma'], numpy.sqrt(varxi))
    numpy.testing.assert_almost_equal(data['weight'], ng.weight)
    numpy.testing.assert_almost_equal(data['npairs'], ng.npairs)

    # Check the read function
    ng2 = treecorr.NGCorrelation(bin_size=0.1, min_sep=1., max_sep=25., sep_units='arcmin')
    ng2.read(out_file_name1)
    numpy.testing.assert_almost_equal(ng2.logr, ng.logr)
    numpy.testing.assert_almost_equal(ng2.meanr, ng.meanr)
    numpy.testing.assert_almost_equal(ng2.meanlogr, ng.meanlogr)
    numpy.testing.assert_almost_equal(ng2.xi, ng.xi)
    numpy.testing.assert_almost_equal(ng2.xi_im, ng.xi_im)
    numpy.testing.assert_almost_equal(ng2.varxi, ng.varxi)
    numpy.testing.assert_almost_equal(ng2.weight, ng.weight)
    numpy.testing.assert_almost_equal(ng2.npairs, ng.npairs)


def test_pieces():
    # Test that we can do the calculation in pieces and recombine the results

    import time

    ncats = 3
    data_cats = []

    nlens = 1000
    nsource = 30000
    gamma0 = 0.05
    r0 = 10.
    L = 50. * r0
    numpy.random.seed(8675309)
    xl = (numpy.random.random_sample(nlens)-0.5) * L
    yl = (numpy.random.random_sample(nlens)-0.5) * L
    xs = (numpy.random.random_sample( (nsource,ncats) )-0.5) * L
    ys = (numpy.random.random_sample( (nsource,ncats) )-0.5) * L
    g1 = numpy.zeros( (nsource,ncats) )
    g2 = numpy.zeros( (nsource,ncats) )
    w = numpy.random.random_sample( (nsource,ncats) ) + 0.5
    for x,y in zip(xl,yl):
        dx = xs-x
        dy = ys-y
        r2 = dx**2 + dy**2
        gammat = gamma0 * numpy.exp(-0.5*r2/r0**2)
        g1 += -gammat * (dx**2-dy**2)/r2
        g2 += -gammat * (2.*dx*dy)/r2

    lens_cat = treecorr.Catalog(x=xl, y=yl, x_units='arcmin', y_units='arcmin')
    source_cats = [ treecorr.Catalog(x=xs[:,k], y=ys[:,k], g1=g1[:,k], g2=g2[:,k], w=w[:,k],
                                     x_units='arcmin', y_units='arcmin') for k in range(ncats) ]
    full_source_cat = treecorr.Catalog(x=xs.flatten(), y=ys.flatten(), w=w.flatten(),
                                       g1=g1.flatten(), g2=g2.flatten(),
                                       x_units='arcmin', y_units='arcmin')

    t0 = time.time()
    for k in range(ncats):
        # These could each be done on different machines in a real world application.
        ng = treecorr.NGCorrelation(bin_size=0.1, min_sep=1., max_sep=25., sep_units='arcmin',
                                    verbose=1)
        # These should use process_cross, not process, since we don't want to call finalize.
        ng.process_cross(lens_cat, source_cats[k])
        ng.write(os.path.join('output','ng_piece_%d.fits'%k))

    pieces_ng = treecorr.NGCorrelation(bin_size=0.1, min_sep=1., max_sep=25., sep_units='arcmin')
    for k in range(ncats):
        ng = pieces_ng.copy()
        ng.read(os.path.join('output','ng_piece_%d.fits'%k))
        pieces_ng += ng
    varg = treecorr.calculateVarG(source_cats)
    pieces_ng.finalize(varg)
    t1 = time.time()
    print ('time for piece-wise processing (including I/O) = ',t1-t0)

    full_ng = treecorr.NGCorrelation(bin_size=0.1, min_sep=1., max_sep=25., sep_units='arcmin',
                                     verbose=1)
    full_ng.process(lens_cat, full_source_cat)
    t2 = time.time()
    print ('time for full processing = ',t2-t1)

    print('max error in meanr = ',numpy.max(pieces_ng.meanr - full_ng.meanr),)
    print('    max meanr = ',numpy.max(full_ng.meanr))
    print('max error in meanlogr = ',numpy.max(pieces_ng.meanlogr - full_ng.meanlogr),)
    print('    max meanlogr = ',numpy.max(full_ng.meanlogr))
    print('max error in npairs = ',numpy.max(pieces_ng.npairs - full_ng.npairs),)
    print('    max npairs = ',numpy.max(full_ng.npairs))
    print('max error in weight = ',numpy.max(pieces_ng.weight - full_ng.weight),)
    print('    max weight = ',numpy.max(full_ng.weight))
    print('max error in xi = ',numpy.max(pieces_ng.xi - full_ng.xi),)
    print('    max xi = ',numpy.max(full_ng.xi))
    print('max error in xi_im = ',numpy.max(pieces_ng.xi_im - full_ng.xi_im),)
    print('    max xi_im = ',numpy.max(full_ng.xi_im))
    print('max error in varxi = ',numpy.max(pieces_ng.varxi - full_ng.varxi),)
    print('    max varxi = ',numpy.max(full_ng.varxi))
    numpy.testing.assert_almost_equal(pieces_ng.meanr, full_ng.meanr, decimal=2)
    numpy.testing.assert_almost_equal(pieces_ng.meanlogr, full_ng.meanlogr, decimal=2)
    numpy.testing.assert_almost_equal(pieces_ng.npairs*1.e-5, full_ng.npairs*1.e-5, decimal=2)
    numpy.testing.assert_almost_equal(pieces_ng.weight*1.e-5, full_ng.weight*1.e-5, decimal=2)
    numpy.testing.assert_almost_equal(pieces_ng.xi*1.e1, full_ng.xi*1.e1, decimal=2)
    numpy.testing.assert_almost_equal(pieces_ng.xi_im*1.e1, full_ng.xi_im*1.e1, decimal=2)
    numpy.testing.assert_almost_equal(pieces_ng.varxi*1.e5, full_ng.varxi*1.e5, decimal=2)

    # A different way to do this can produce results that are essentially identical to the
    # full calculation.  We can use wpos = w, but set w = 0 for the items in the pieces catalogs
    # that we don't want to include.  This will force the tree to be built identically in each
    # case, but only use the subset of items in the calculation.  The sum of all these should
    # be identical to the full calculation aside from order of calculation differences.
    # However, we lose some to speed, since there are a lot more wasted calculations along the
    # way that have to be duplicated in each piece.
    w2 = [ numpy.empty(w.shape) for k in range(ncats) ]
    for k in range(ncats):
        w2[k][:,:] = 0.
        w2[k][:,k] = w[:,k]
    source_cats2 = [ treecorr.Catalog(x=xs.flatten(), y=ys.flatten(),
                                      g1=g1.flatten(), g2=g2.flatten(),
                                      wpos=w.flatten(), w=w2[k].flatten(),
                                      x_units='arcmin', y_units='arcmin') for k in range(ncats) ]

    t3 = time.time()
    ng2 = [ full_ng.copy() for k in range(ncats) ]
    for k in range(ncats):
        ng2[k].clear()
        ng2[k].process_cross(lens_cat, source_cats2[k])

    pieces_ng2 = full_ng.copy()
    pieces_ng2.clear()
    for k in range(ncats):
        pieces_ng2 += ng2[k]
    pieces_ng2.finalize(varg)
    t4 = time.time()
    print ('time for zero-weight piece-wise processing = ',t4-t3)

    print('max error in meanr = ',numpy.max(pieces_ng2.meanr - full_ng.meanr),)
    print('    max meanr = ',numpy.max(full_ng.meanr))
    print('max error in meanlogr = ',numpy.max(pieces_ng2.meanlogr - full_ng.meanlogr),)
    print('    max meanlogr = ',numpy.max(full_ng.meanlogr))
    print('max error in npairs = ',numpy.max(pieces_ng2.npairs - full_ng.npairs),)
    print('    max npairs = ',numpy.max(full_ng.npairs))
    print('max error in weight = ',numpy.max(pieces_ng2.weight - full_ng.weight),)
    print('    max weight = ',numpy.max(full_ng.weight))
    print('max error in xi = ',numpy.max(pieces_ng2.xi - full_ng.xi),)
    print('    max xi = ',numpy.max(full_ng.xi))
    print('max error in xi_im = ',numpy.max(pieces_ng2.xi_im - full_ng.xi_im),)
    print('    max xi_im = ',numpy.max(full_ng.xi_im))
    print('max error in varxi = ',numpy.max(pieces_ng2.varxi - full_ng.varxi),)
    print('    max varxi = ',numpy.max(full_ng.varxi))
    numpy.testing.assert_almost_equal(pieces_ng2.meanr, full_ng.meanr, decimal=8)
    numpy.testing.assert_almost_equal(pieces_ng2.meanlogr, full_ng.meanlogr, decimal=8)
    numpy.testing.assert_almost_equal(pieces_ng2.npairs*1.e-5, full_ng.npairs*1.e-5, decimal=8)
    numpy.testing.assert_almost_equal(pieces_ng2.weight*1.e-5, full_ng.weight*1.e-5, decimal=8)
    numpy.testing.assert_almost_equal(pieces_ng2.xi*1.e1, full_ng.xi*1.e1, decimal=8)
    numpy.testing.assert_almost_equal(pieces_ng2.xi_im*1.e1, full_ng.xi_im*1.e1, decimal=8)
    numpy.testing.assert_almost_equal(pieces_ng2.varxi*1.e5, full_ng.varxi*1.e5, decimal=8)


def test_rlens():
    # Same as above, except use R_lens for separation.
    # Use gamma_t(r) = gamma0 exp(-R^2/2R0^2) around a bunch of foreground lenses.

    nlens = 100
    nsource = 200000
    gamma0 = 0.05
    R0 = 10.
    L = 50. * R0
    numpy.random.seed(8675309)
    xl = (numpy.random.random_sample(nlens)-0.5) * L  # -250 < x < 250
    zl = (numpy.random.random_sample(nlens)-0.5) * L  # -250 < y < 250
    yl = numpy.random.random_sample(nlens) * 4*L + 10*L  # 5000 < z < 7000
    rl = numpy.sqrt(xl**2 + yl**2 + zl**2)
    xs = (numpy.random.random_sample(nsource)-0.5) * L
    zs = (numpy.random.random_sample(nsource)-0.5) * L
    ys = numpy.random.random_sample(nsource) * 8*L + 160*L  # 80000 < z < 84000
    rs = numpy.sqrt(xs**2 + ys**2 + zs**2)
    g1 = numpy.zeros( (nsource,) )
    g2 = numpy.zeros( (nsource,) )
    bin_size = 0.1
    # min_sep is set so the first bin doesn't have 0 pairs.
    min_sep = 1.3*R0
    # max_sep can't be too large, since the measured value starts to have shape noise for larger
    # values of separation.  We're not adding any shape noise directly, but the shear from other
    # lenses is effectively a shape noise, and that comes to dominate the measurement above ~4R0.
    max_sep = 4.*R0
    nbins = int(numpy.ceil(numpy.log(max_sep/min_sep)/bin_size))
    true_gt = numpy.zeros( (nbins,) )
    true_npairs = numpy.zeros((nbins,), dtype=int)
    print('Making shear vectors')
    for x,y,z,r in zip(xl,yl,zl,rl):
        # Use |r1 x r2| = |r1| |r2| sin(theta)
        xcross = ys * z - zs * y
        ycross = zs * x - xs * z
        zcross = xs * y - ys * x
        sintheta = numpy.sqrt(xcross**2 + ycross**2 + zcross**2) / (rs * r)
        Rlens = 2. * r * numpy.sin(numpy.arcsin(sintheta)/2)

        gammat = gamma0 * numpy.exp(-0.5*Rlens**2/R0**2)
        # For the rotation, approximate that the x,z coords are approx the perpendicular plane.
        # So just normalize back to the unit sphere and do the 2d projection calculation.
        # It's not exactly right, but it should be good enough for this unit test.
        dx = xs/rs-x/r
        dz = zs/rs-z/r
        drsq = dx**2 + dz**2
        g1 += -gammat * (dx**2-dz**2)/drsq
        g2 += -gammat * (2.*dx*dz)/drsq
        index = numpy.floor( numpy.log(Rlens/min_sep) / bin_size).astype(int)
        mask = (index >= 0) & (index < nbins)
        numpy.add.at(true_gt, index[mask], gammat[mask])
        numpy.add.at(true_npairs, index[mask], 1)
    true_gt /= true_npairs

    # Start with bin_slop == 0.  With only 100 lenses, this still runs very fast.
    lens_cat = treecorr.Catalog(x=xl, y=yl, z=zl)
    source_cat = treecorr.Catalog(x=xs, y=ys, z=zs, g1=g1, g2=g2)
    ng0 = treecorr.NGCorrelation(bin_size=bin_size, min_sep=min_sep, max_sep=max_sep, verbose=1,
                                 metric='Rlens', bin_slop=0)
    ng0.process(lens_cat, source_cat)

    Rlens = ng0.meanr
    theory_gt = gamma0 * numpy.exp(-0.5*Rlens**2/R0**2)

    print('Results with bin_slop = 0:')
    print('ng.npairs = ',ng0.npairs)
    print('true_npairs = ',true_npairs)
    print('ng.xi = ',ng0.xi)
    print('true_gammat = ',true_gt)
    print('ratio = ',ng0.xi / true_gt)
    print('diff = ',ng0.xi - true_gt)
    print('max diff = ',max(abs(ng0.xi - true_gt)))
    assert max(abs(ng0.xi - true_gt)) < 2.e-6
    print('ng.xi_im = ',ng0.xi_im)
    assert max(abs(ng0.xi_im)) < 1.e-6

    print('ng.xi = ',ng0.xi)
    print('theory_gammat = ',theory_gt)
    print('ratio = ',ng0.xi / theory_gt)
    print('diff = ',ng0.xi - theory_gt)
    print('max diff = ',max(abs(ng0.xi - theory_gt)))
    assert max(abs(ng0.xi - theory_gt)) < 4.e-5

    # Now use a more normal value for bin_slop.
    ng1 = treecorr.NGCorrelation(bin_size=bin_size, min_sep=min_sep, max_sep=max_sep, verbose=1,
                                 metric='Rlens', bin_slop=0.5)
    ng1.process(lens_cat, source_cat)
    Rlens = ng1.meanr
    theory_gt = gamma0 * numpy.exp(-0.5*Rlens**2/R0**2)

    print('Results with bin_slop = 0.5')
    print('ng.npairs = ',ng1.npairs)
    print('ng.xi = ',ng1.xi)
    print('theory_gammat = ',theory_gt)
    print('ratio = ',ng1.xi / theory_gt)
    print('diff = ',ng1.xi - theory_gt)
    print('max diff = ',max(abs(ng1.xi - theory_gt)))
    assert max(abs(ng1.xi - theory_gt)) < 5.e-5
    print('ng.xi_im = ',ng1.xi_im)
    assert max(abs(ng1.xi_im)) < 3.e-6

    # Check that we get the same result using the corr2 executable:
    if __name__ == '__main__':
        lens_cat.write(os.path.join('data','ng_rlens_lens.dat'))
        source_cat.write(os.path.join('data','ng_rlens_source.dat'))
        import subprocess
        corr2_exe = get_script_name('corr2')
        p = subprocess.Popen( [corr2_exe,"ng_rlens.yaml"] )
        p.communicate()
        corr2_output = numpy.genfromtxt(os.path.join('output','ng_rlens.out'),names=True)
        print('ng.xi = ',ng1.xi)
        print('from corr2 output = ',corr2_output['gamT'])
        print('ratio = ',corr2_output['gamT']/ng1.xi)
        print('diff = ',corr2_output['gamT']-ng1.xi)
        numpy.testing.assert_almost_equal(corr2_output['gamT'], ng1.xi, decimal=6)
        numpy.testing.assert_almost_equal(corr2_output['gamX'], ng1.xi_im, decimal=6)

    # Repeat with the sources being given as RA/Dec only.
    ral, decl = treecorr.CelestialCoord.xyz_to_radec(xl,yl,zl)
    ras, decs = treecorr.CelestialCoord.xyz_to_radec(xs,ys,zs)
    lens_cat = treecorr.Catalog(ra=ral, dec=decl, ra_units='radians', dec_units='radians', r=rl)
    source_cat = treecorr.Catalog(ra=ras, dec=decs, ra_units='radians', dec_units='radians',
                                  g1=g1, g2=g2)

    # Again, start with bin_slop == 0.
    # This version should be identical to the 3D version.  When bin_slop != 0, it won't be
    # exactly identical, since the tree construction will have different decisions along the
    # way (since everything is at the same radius here), but the results are consistent.
    ng0s = treecorr.NGCorrelation(bin_size=bin_size, min_sep=min_sep, max_sep=max_sep, verbose=1,
                                  metric='Rlens', bin_slop=0)
    ng0s.process(lens_cat, source_cat)

    Rlens = ng0s.meanr
    theory_gt = gamma0 * numpy.exp(-0.5*Rlens**2/R0**2)

    print('Results when sources have no radius information, first bin_slop=0')
    print('ng.npairs = ',ng0s.npairs)
    print('true_npairs = ',true_npairs)
    print('ng.xi = ',ng0s.xi)
    print('true_gammat = ',true_gt)
    print('ratio = ',ng0s.xi / true_gt)
    print('diff = ',ng0s.xi - true_gt)
    print('max diff = ',max(abs(ng0s.xi - true_gt)))
    assert max(abs(ng0s.xi - true_gt)) < 2.e-6
    print('ng.xi_im = ',ng0s.xi_im)
    assert max(abs(ng0s.xi_im)) < 1.e-6

    print('ng.xi = ',ng0s.xi)
    print('theory_gammat = ',theory_gt)
    print('ratio = ',ng0s.xi / theory_gt)
    print('diff = ',ng0s.xi - theory_gt)
    print('max diff = ',max(abs(ng0s.xi - theory_gt)))
    assert max(abs(ng0s.xi - theory_gt)) < 4.e-5

    assert max(abs(ng0s.xi - ng0.xi)) < 1.e-7
    assert max(abs(ng0s.xi_im - ng0.xi_im)) < 1.e-7
    assert max(abs(ng0s.npairs - ng0.npairs)) < 1.e-7

    # Now use a more normal value for bin_slop.
    ng1s = treecorr.NGCorrelation(bin_size=bin_size, min_sep=min_sep, max_sep=max_sep, verbose=1,
                                  metric='Rlens', bin_slop=0.5)
    ng1s.process(lens_cat, source_cat)
    Rlens = ng1s.meanr
    theory_gt = gamma0 * numpy.exp(-0.5*Rlens**2/R0**2)

    print('Results with bin_slop = 0.5')
    print('ng.npairs = ',ng1s.npairs)
    print('ng.xi = ',ng1s.xi)
    print('theory_gammat = ',theory_gt)
    print('ratio = ',ng1s.xi / theory_gt)
    print('diff = ',ng1s.xi - theory_gt)
    print('max diff = ',max(abs(ng1s.xi - theory_gt)))
    assert max(abs(ng1s.xi - theory_gt)) < 5.e-5
    print('ng.xi_im = ',ng1s.xi_im)
    assert max(abs(ng1s.xi_im)) < 3.e-6


def test_rlens_bkg():
    # Same as above, except limit the sources to be in the background of the lens.

    nlens = 100
    nsource = 200000
    gamma0 = 0.05
    R0 = 10.
    L = 50. * R0
    numpy.random.seed(8675309)
    xl = (numpy.random.random_sample(nlens)-0.5) * L  # -250 < x < 250
    zl = (numpy.random.random_sample(nlens)-0.5) * L  # -250 < y < 250
    yl = numpy.random.random_sample(nlens) * 4*L + 10*L  # 5000 < z < 7000
    rl = numpy.sqrt(xl**2 + yl**2 + zl**2)
    xs = (numpy.random.random_sample(nsource)-0.5) * L
    zs = (numpy.random.random_sample(nsource)-0.5) * L
    ys = numpy.random.random_sample(nsource) * 12*L + 8*L  # 4000 < z < 10000
    rs = numpy.sqrt(xs**2 + ys**2 + zs**2)
    print('xl = ',numpy.min(xl),numpy.max(xl))
    print('yl = ',numpy.min(yl),numpy.max(yl))
    print('zl = ',numpy.min(zl),numpy.max(zl))
    print('xs = ',numpy.min(xs),numpy.max(xs))
    print('ys = ',numpy.min(ys),numpy.max(ys))
    print('zs = ',numpy.min(zs),numpy.max(zs))
    g1 = numpy.zeros( (nsource,) )
    g2 = numpy.zeros( (nsource,) )
    bin_size = 0.1
    # min_sep is set so the first bin doesn't have 0 pairs.
    min_sep = 1.3*R0
    # max_sep can't be too large, since the measured value starts to have shape noise for larger
    # values of separation.  We're not adding any shape noise directly, but the shear from other
    # lenses is effectively a shape noise, and that comes to dominate the measurement above ~4R0.
    max_sep = 4.*R0
    nbins = int(numpy.ceil(numpy.log(max_sep/min_sep)/bin_size))
    print('Making shear vectors')
    for x,y,z,r in zip(xl,yl,zl,rl):
        # This time, only give the true shear to the background galaxies.
        bkg = (rs > r)

        # Use |r1 x r2| = |r1| |r2| sin(theta)
        xcross = ys[bkg] * z - zs[bkg] * y
        ycross = zs[bkg] * x - xs[bkg] * z
        zcross = xs[bkg] * y - ys[bkg] * x
        sintheta = numpy.sqrt(xcross**2 + ycross**2 + zcross**2) / (rs[bkg] * r)
        Rlens = 2. * r * numpy.sin(numpy.arcsin(sintheta)/2)

        gammat = gamma0 * numpy.exp(-0.5*Rlens**2/R0**2)
        # For the rotation, approximate that the x,z coords are approx the perpendicular plane.
        # So just normalize back to the unit sphere and do the 2d projection calculation.
        # It's not exactly right, but it should be good enough for this unit test.
        dx = (xs/rs)[bkg]-x/r
        dz = (zs/rs)[bkg]-z/r
        drsq = dx**2 + dz**2

        g1[bkg] += -gammat * (dx**2-dz**2)/drsq
        g2[bkg] += -gammat * (2.*dx*dz)/drsq

    # Slight subtlety in this test vs the previous one.  We need to build up the full g1,g2
    # arrays first before calculating the true_gt value, since we need to include the background
    # galaxies for each lens regardless of whether they had signal or not.
    true_gt = numpy.zeros( (nbins,) )
    true_npairs = numpy.zeros((nbins,), dtype=int)
    for x,y,z,r in zip(xl,yl,zl,rl):
        # Use |r1 x r2| = |r1| |r2| sin(theta)
        xcross = ys * z - zs * y
        ycross = zs * x - xs * z
        zcross = xs * y - ys * x
        sintheta = numpy.sqrt(xcross**2 + ycross**2 + zcross**2) / (rs * r)
        Rlens = 2. * r * numpy.sin(numpy.arcsin(sintheta)/2)
        dx = xs/rs-x/r
        dz = zs/rs-z/r
        drsq = dx**2 + dz**2
        gt = -g1 * (dx**2-dz**2)/drsq - g2 * (2.*dx*dz)/drsq
        bkg = (rs > r)
        index = numpy.floor( numpy.log(Rlens/min_sep) / bin_size).astype(int)
        mask = (index >= 0) & (index < nbins) & bkg
        numpy.add.at(true_gt, index[mask], gt[mask])
        numpy.add.at(true_npairs, index[mask], 1)

    true_gt /= true_npairs

    # Start with bin_slop == 0.  With only 100 lenses, this still runs very fast.
    lens_cat = treecorr.Catalog(x=xl, y=yl, z=zl)
    source_cat = treecorr.Catalog(x=xs, y=ys, z=zs, g1=g1, g2=g2)
    ng0 = treecorr.NGCorrelation(bin_size=bin_size, min_sep=min_sep, max_sep=max_sep, verbose=1,
                                 metric='Rlens', bin_slop=0, min_rpar=0)
    ng0.process(lens_cat, source_cat)

    Rlens = ng0.meanr
    theory_gt = gamma0 * numpy.exp(-0.5*Rlens**2/R0**2)

    print('Results with bin_slop = 0:')
    print('ng.npairs = ',ng0.npairs)
    print('true_npairs = ',true_npairs)
    print('ng.xi = ',ng0.xi)
    print('true_gammat = ',true_gt)
    print('ratio = ',ng0.xi / true_gt)
    print('diff = ',ng0.xi - true_gt)
    print('max diff = ',max(abs(ng0.xi - true_gt)))
    assert max(abs(ng0.xi - true_gt)) < 2.e-6

    print('ng.xi = ',ng0.xi)
    print('theory_gammat = ',theory_gt)
    print('ratio = ',ng0.xi / theory_gt)
    print('diff = ',ng0.xi - theory_gt)
    print('max diff = ',max(abs(ng0.xi - theory_gt)))
    assert max(abs(ng0.xi - theory_gt)) < 1.e-3
    print('ng.xi_im = ',ng0.xi_im)
    assert max(abs(ng0.xi_im)) < 1.e-3

    # Without min_rpar, this should fail.
    lens_cat = treecorr.Catalog(x=xl, y=yl, z=zl)
    source_cat = treecorr.Catalog(x=xs, y=ys, z=zs, g1=g1, g2=g2)
    ng0 = treecorr.NGCorrelation(bin_size=bin_size, min_sep=min_sep, max_sep=max_sep, verbose=1,
                                 metric='Rlens', bin_slop=0)
    ng0.process(lens_cat, source_cat)
    Rlens = ng0.meanr

    print('Results without min_rpar')
    print('ng.xi = ',ng0.xi)
    print('true_gammat = ',true_gt)
    print('max diff = ',max(abs(ng0.xi - true_gt)))
    assert max(abs(ng0.xi - true_gt)) > 5.e-3

    # Now use a more normal value for bin_slop.
    ng1 = treecorr.NGCorrelation(bin_size=bin_size, min_sep=min_sep, max_sep=max_sep, verbose=1,
                                 metric='Rlens', bin_slop=0.5, min_rpar=0)
    ng1.process(lens_cat, source_cat)
    Rlens = ng1.meanr
    theory_gt = gamma0 * numpy.exp(-0.5*Rlens**2/R0**2)

    print('Results with bin_slop = 0.5')
    print('ng.npairs = ',ng1.npairs)
    print('ng.xi = ',ng1.xi)
    print('theory_gammat = ',theory_gt)
    print('ratio = ',ng1.xi / theory_gt)
    print('diff = ',ng1.xi - theory_gt)
    print('max diff = ',max(abs(ng1.xi - theory_gt)))
    assert max(abs(ng1.xi - theory_gt)) < 1.e-3
    print('ng.xi_im = ',ng1.xi_im)
    assert max(abs(ng1.xi_im)) < 1.e-3

    # Check that we get the same result using the corr2 executable:
    if __name__ == '__main__':
        lens_cat.write(os.path.join('data','ng_rlens_bkg_lens.dat'))
        source_cat.write(os.path.join('data','ng_rlens_bkg_source.dat'))
        import subprocess
        corr2_exe = get_script_name('corr2')
        p = subprocess.Popen( [corr2_exe,"ng_rlens_bkg.yaml"] )
        p.communicate()
        corr2_output = numpy.genfromtxt(os.path.join('output','ng_rlens_bkg.out'),names=True)
        print('ng.xi = ',ng1.xi)
        print('from corr2 output = ',corr2_output['gamT'])
        print('ratio = ',corr2_output['gamT']/ng1.xi)
        print('diff = ',corr2_output['gamT']-ng1.xi)
        numpy.testing.assert_almost_equal(corr2_output['gamT'], ng1.xi, decimal=6)
        numpy.testing.assert_almost_equal(corr2_output['gamX'], ng1.xi_im, decimal=6)


def test_haloellip():
    """This is similar to the Clampitt halo ellipticity measurement, but using counts for the
    background galaxies rather than shears.

    w_aligned = Sum_i (w_i * cos(2theta)) / Sum_i (w_i)
    w_cross = Sum_i (w_i * sin(2theta)) / Sum_i (w_i)

    where theta is measured w.r.t. the coordinate system where the halo ellitpicity
    is along the x-axis.  Converting this to complex notation, we obtain:

    w_a - i w_c = < exp(-2itheta) >
                = < exp(2iphi) exp(-2i(theta+phi)) >
                = < ehalo exp(-2i(theta+phi)) >

    where ehalo = exp(2iphi) is the unit-normalized shape of the halo in the normal world
    coordinate system.  Note that the combination theta+phi is the angle between the line joining
    the two points and the E-W coordinate, which means that

    w_a - i w_c = -gamma_t(n_bg, ehalo)

    so the reverse of the usual galaxy-galaxy lensing order.  The N is the background galaxies
    and G is the halo shapes (normalized to have |ehalo| = 1).
    """

    nhalo = 10
    nsource = 1000000  # sources per halo
    ntot = nsource * nhalo
    L = 100000.  # The side length in which the halos are placed
    R = 10.      # The (rms) radius of the associated sources from the halos
                 # In this case, we want L >> R so that most sources are only associated
                 # with the one halo we used for assigning its shear value.

    # Lenses are randomly located with random shapes.
    numpy.random.seed(8675309)
    halo_g1 = numpy.random.normal(0., 0.3, (nhalo,))
    halo_g2 = numpy.random.normal(0., 0.3, (nhalo,))
    halo_g = halo_g1 + 1j * halo_g2
    # The interpretation is simpler if they all have the same |g|, so just make them all 0.3.
    halo_g *= 0.3 / numpy.abs(halo_g)
    halo_absg = numpy.abs(halo_g)
    halo_x = (numpy.random.random_sample(nhalo)-0.5) * L
    halo_y = (numpy.random.random_sample(nhalo)-0.5) * L
    print('Made halos')

    # For the sources, place nsource galaxies around each halo with the expected azimuthal pattern
    source_x = numpy.empty(ntot)
    source_y = numpy.empty(ntot)
    for i in range(nhalo):
        absg = halo_absg[i]
        # First position the sources in a Gaussian cloud around the halo center.
        dx = numpy.random.normal(0., 10., (nsource,))
        dy = numpy.random.normal(0., 10., (nsource,))
        r = numpy.sqrt(dx*dx + dy*dy)
        t = numpy.arctan2(dy,dx)
        # z = dx + idy = r exp(it)

        # Reposition the sources azimuthally so p(theta) ~ 1 + |g_halo| * cos(2 theta)
        # Currently t has p(t) = 1/2pi.
        # Let u be the new azimuthal angle with p(u) = (1/2pi) (1 + |g| cos(2u))
        # p(u) = |dt/du| p(t)
        # 1 + |g| cos(2u) = dt/du
        # t = int( (1 + |g| cos(2u)) du = u + 1/2 |g| sin(2u)

        # This doesn't have an analytic solution, but a few iterations of Newton-Raphson
        # should work well enough.
        u = t.copy()
        for i in range(4):
            u -= (u - t + 0.5 * absg * numpy.sin(2.*u)) / (1. + absg * numpy.cos(2.*u))

        z = r * numpy.exp(1j * u)
        exp2iphi = z**2 / numpy.abs(z)**2

        # Now rotate the whole system by the phase of the halo ellipticity.
        exp2ialpha = halo_g[i] / absg
        expialpha = numpy.sqrt(exp2ialpha)
        z *= expialpha
        # Place the source galaxies at this dx,dy with this shape
        source_x[i*nsource: (i+1)*nsource] = halo_x[i] + z.real
        source_y[i*nsource: (i+1)*nsource] = halo_y[i] + z.imag
    print('Made sources')

    source_cat = treecorr.Catalog(x=source_x, y=source_y)
    # Big fat bin to increase S/N.  The way I set it up, the signal is the same in all
    # radial bins, so just combine them together for higher S/N.
    ng = treecorr.NGCorrelation(min_sep=5, max_sep=10, nbins=1)
    halo_mean_absg = numpy.mean(halo_absg)
    print('mean_absg = ',halo_mean_absg)

    # First the original version where we only use the phase of the halo ellipticities:
    halo_cat1 = treecorr.Catalog(x=halo_x, y=halo_y,
                                 g1=halo_g.real/halo_absg, g2=halo_g.imag/halo_absg)
    ng.process(source_cat, halo_cat1)
    print('ng.npairs = ',ng.npairs)
    print('ng.xi = ',ng.xi)
    # The expected signal is
    # E(ng) = - < int( p(t) cos(2t) ) >
    #       = - < int( (1 + e_halo cos(2t)) cos(2t) ) >
    #       = -0.5 <e_halo>
    print('expected signal = ',-0.5 * halo_mean_absg)
    # These tests don't quite work at the 1% level of accuracy, but 2% seems to work for most.
    # This is effected by checking that 1/2 the value matches 0.5 to 2 decimal places.
    numpy.testing.assert_almost_equal(ng.xi, -0.5 * halo_mean_absg, decimal=2)

    # Next weight the halos by their absg.
    halo_cat2 = treecorr.Catalog(x=halo_x, y=halo_y, w=halo_absg,
                                 g1=halo_g.real/halo_absg, g2=halo_g.imag/halo_absg)
    ng.process(source_cat, halo_cat2)
    print('ng.xi = ',ng.xi)
    # Now the net signal is
    # sum(w * p*cos(2t)) / sum(w)
    # = 0.5 * <absg^2> / <absg>
    halo_mean_gsq = numpy.mean(halo_absg**2)
    print('expected signal = ',0.5 * halo_mean_gsq / halo_mean_absg)
    numpy.testing.assert_almost_equal(ng.xi, -0.5 * halo_mean_gsq / halo_mean_absg, decimal=2)

    # Finally, use the unnormalized halo_g for the halo ellipticities
    halo_cat3 = treecorr.Catalog(x=halo_x, y=halo_y, g1=halo_g.real, g2=halo_g.imag)
    ng.process(source_cat, halo_cat3)
    print('ng.xi = ',ng.xi)
    # Now the net signal is
    # sum(absg * p*cos(2t)) / N
    # = 0.5 * <absg^2>
    print('expected signal = ',0.5 * halo_mean_gsq)
    numpy.testing.assert_almost_equal(ng.xi, -0.5 * halo_mean_gsq, decimal=2)


if __name__ == '__main__':
    test_single()
    test_pairwise()
    test_spherical()
    test_ng()
    test_pieces()
    test_rlens()
    test_rlens_bkg()
    test_haloellip()
