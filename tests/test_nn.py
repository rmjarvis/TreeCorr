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

def test_direct_count():
    # If the catalogs are small enough, we can do a direct count of the number of pairs
    # to see if comes out right.  This should exactly match the treecorr code if bin_slop=0.

    ngal = 100
    s = 10.
    numpy.random.seed(8675309)
    x1 = numpy.random.normal(0,s, (ngal,) )
    y1 = numpy.random.normal(0,s, (ngal,) )
    cat1 = treecorr.Catalog(x=x1, y=y1)
    x2 = numpy.random.normal(0,s, (ngal,) )
    y2 = numpy.random.normal(0,s, (ngal,) )
    cat2 = treecorr.Catalog(x=x2, y=y2)

    min_sep = 1.
    max_sep = 50.
    nbins = 50
    dd = treecorr.NNCorrelation(min_sep=min_sep, max_sep=max_sep, nbins=nbins, bin_slop=0.)
    dd.process(cat1, cat2)
    print('dd.npairs = ',dd.npairs)

    log_min_sep = numpy.log(min_sep)
    log_max_sep = numpy.log(max_sep)
    true_npairs = numpy.zeros(nbins)
    bin_size = (log_max_sep - log_min_sep) / nbins
    for i in range(ngal):
        for j in range(ngal):
            rsq = (x1[i]-x2[j])**2 + (y1[i]-y2[j])**2
            logr = 0.5 * numpy.log(rsq)
            k = int(numpy.floor( (logr-log_min_sep) / bin_size ))
            if k < 0: continue
            if k >= nbins: continue
            true_npairs[k] += 1

    print('true_npairs = ',true_npairs)
    print('diff = ',dd.npairs - true_npairs)
    numpy.testing.assert_array_equal(dd.npairs, true_npairs)

def test_direct_3d():
    # This is the same as the above test, but using the 3d correlations

    ngal = 100
    s = 10.
    numpy.random.seed(8675309)
    x1 = numpy.random.normal(312, s, (ngal,) )
    y1 = numpy.random.normal(728, s, (ngal,) )
    z1 = numpy.random.normal(-932, s, (ngal,) )
    r1 = numpy.sqrt( x1*x1 + y1*y1 + z1*z1 )
    dec1 = numpy.arcsin(z1/r1)
    ra1 = numpy.arctan2(y1,x1)
    cat1 = treecorr.Catalog(ra=ra1, dec=dec1, r=r1, ra_units='rad', dec_units='rad')

    x2 = numpy.random.normal(312, s, (ngal,) )
    y2 = numpy.random.normal(728, s, (ngal,) )
    z2 = numpy.random.normal(-932, s, (ngal,) )
    r2 = numpy.sqrt( x2*x2 + y2*y2 + z2*z2 )
    dec2 = numpy.arcsin(z2/r2)
    ra2 = numpy.arctan2(y2,x2)
    cat2 = treecorr.Catalog(ra=ra2, dec=dec2, r=r2, ra_units='rad', dec_units='rad')

    min_sep = 1.
    max_sep = 50.
    nbins = 50
    dd = treecorr.NNCorrelation(min_sep=min_sep, max_sep=max_sep, nbins=nbins, bin_slop=0.)
    dd.process(cat1, cat2)
    print('dd.npairs = ',dd.npairs)

    log_min_sep = numpy.log(min_sep)
    log_max_sep = numpy.log(max_sep)
    true_npairs = numpy.zeros(nbins)
    bin_size = (log_max_sep - log_min_sep) / nbins
    for i in range(ngal):
        for j in range(ngal):
            rsq = (x1[i]-x2[j])**2 + (y1[i]-y2[j])**2 + (z1[i]-z2[j])**2
            logr = 0.5 * numpy.log(rsq)
            k = int(numpy.floor( (logr-log_min_sep) / bin_size ))
            if k < 0: continue
            if k >= nbins: continue
            true_npairs[k] += 1

    print('true_npairs = ',true_npairs)
    print('diff = ',dd.npairs - true_npairs)
    numpy.testing.assert_array_equal(dd.npairs, true_npairs)

def test_nn():
    # Use a simple probability distribution for the galaxies:
    #
    # n(r) = (2pi s^2)^-1 exp(-r^2/2s^2)
    #
    # The Fourier transform is: n~(k) = exp(-s^2 k^2/2)
    # P(k) = <|n~(k)|^2> = exp(-s^2 k^2)
    # xi(r) = (1/2pi) int( dk k P(k) J0(kr) ) 
    #       = 1/(4 pi s^2) exp(-r^2/4s^2)
    #
    # However, we need to correct for the uniform density background, so the real result
    # is this minus 1/L^2 divided by 1/L^2.  So:
    #
    # xi(r) = 1/4pi (L/s)^2 exp(-r^2/4s^2) - 1

    ngal = 1000000
    s = 10.
    L = 50. * s  # Not infinity, so this introduces some error.  Our integrals were to infinity.
    numpy.random.seed(8675309)
    x = numpy.random.normal(0,s, (ngal,) )
    y = numpy.random.normal(0,s, (ngal,) )

    cat = treecorr.Catalog(x=x, y=y, x_units='arcmin', y_units='arcmin')
    dd = treecorr.NNCorrelation(bin_size=0.1, min_sep=1., max_sep=25., sep_units='arcmin',
                                verbose=2)
    dd.process(cat)
    print('dd.npairs = ',dd.npairs)

    nrand = 5 * ngal
    rx = (numpy.random.random_sample(nrand)-0.5) * L
    ry = (numpy.random.random_sample(nrand)-0.5) * L
    rand = treecorr.Catalog(x=rx,y=ry, x_units='arcmin', y_units='arcmin')
    rr = treecorr.NNCorrelation(bin_size=0.1, min_sep=1., max_sep=25., sep_units='arcmin',
                                verbose=2)
    rr.process(rand)
    print('rr.npairs = ',rr.npairs)

    dr = treecorr.NNCorrelation(bin_size=0.1, min_sep=1., max_sep=25., sep_units='arcmin',
                                verbose=2)
    dr.process(cat,rand)
    print('dr.npairs = ',dr.npairs)

    r = numpy.exp(dd.meanlogr)
    true_xi = 0.25/numpy.pi * (L/s)**2 * numpy.exp(-0.25*r**2/s**2) - 1.

    xi, varxi = dd.calculateXi(rr,dr)
    print('xi = ',xi)
    print('true_xi = ',true_xi)
    print('ratio = ',xi / true_xi)
    print('diff = ',xi - true_xi)
    print('max rel diff = ',max(abs((xi - true_xi)/true_xi)))
    # This isn't super accurate.  But the agreement improves as L increase, so I think it is 
    # merely a matter of the finite field and the integrals going to infinity.  (Sort of, since
    # we still have L in there.)
    assert max(abs(xi - true_xi)/true_xi) < 0.1

    simple_xi, varxi = dd.calculateXi(rr)
    print('simple xi = ',simple_xi)
    print('max rel diff = ',max(abs((simple_xi - true_xi)/true_xi)))
    # The simple calculation (i.e. dd/rr-1, rather than (dd-2dr+rr)/rr as above) is only 
    # slightly less accurate in this case.  Probably because the mask is simple (a box), so
    # the difference is relatively minor.  The error is slightly higher in this case, but testing
    # that it is everywhere < 0.1 is still appropriate.
    assert max(abs(simple_xi - true_xi)/true_xi) < 0.1

    # Check that we get the same result using the corr2 executable:
    if __name__ == '__main__':
        cat.write(os.path.join('data','nn_data.dat'))
        rand.write(os.path.join('data','nn_rand.dat'))
        import subprocess
        p = subprocess.Popen( ["corr2","nn.params"] )
        p.communicate()
        corr2_output = numpy.loadtxt(os.path.join('output','nn.out'))
        print('xi = ',xi)
        print('from corr2 output = ',corr2_output[:,2])
        print('ratio = ',corr2_output[:,2]/xi)
        print('diff = ',corr2_output[:,2]-xi)
        numpy.testing.assert_almost_equal(corr2_output[:,2]/xi, 1., decimal=3)


def test_3d():
    # For this one, build a Gaussian cloud around some random point in 3D space and do the 
    # correlation function in 3D.
    #
    # Use n(r) = (2pi s^2)^-3/2 exp(-r^2/2s^2)
    #
    # The 3D Fourier transform is: n~(k) = exp(-s^2 k^2/2)
    # P(k) = <|n~(k)|^2> = exp(-s^2 k^2)
    # xi(r) = 1/2pi^2 int( dk k^2 P(k) j0(kr) )
    #       = 1/(8 pi^3/2) 1/s^3 exp(-r^2/4s^2)
    #
    # And as before, we need to correct for the randoms, so the final xi(r) is
    #
    # xi(r) = 1/(8 pi^3/2) (L/s)^3 exp(-r^2/4s^2) - 1

    ngal = 100000
    xcen = 823  # Mpc maybe?
    ycen = 342
    zcen = -672
    s = 10.
    L = 50. * s  # Not infinity, so this introduces some error.  Our integrals were to infinity.
    numpy.random.seed(8675309)
    x = numpy.random.normal(xcen, s, (ngal,) )
    y = numpy.random.normal(ycen, s, (ngal,) )
    z = numpy.random.normal(zcen, s, (ngal,) )

    r = numpy.sqrt(x*x+y*y+z*z)
    dec = numpy.arcsin(z/r) / treecorr.degrees
    ra = numpy.arctan2(y,x) / treecorr.degrees

    cat = treecorr.Catalog(ra=ra, dec=dec, r=r, ra_units='deg', dec_units='deg')
    dd = treecorr.NNCorrelation(bin_size=0.1, min_sep=1., max_sep=25., verbose=2)
    dd.process(cat)
    print('dd.npairs = ',dd.npairs)

    nrand = 5 * ngal
    rx = (numpy.random.random_sample(nrand)-0.5) * L + xcen
    ry = (numpy.random.random_sample(nrand)-0.5) * L + ycen
    rz = (numpy.random.random_sample(nrand)-0.5) * L + zcen
    rr = numpy.sqrt(rx*rx+ry*ry+rz*rz)
    rdec = numpy.arcsin(rz/rr) / treecorr.degrees
    rra = numpy.arctan2(ry,rx) / treecorr.degrees
    rand = treecorr.Catalog(ra=rra, dec=rdec, r=rr, ra_units='deg', dec_units='deg')
    rr = treecorr.NNCorrelation(bin_size=0.1, min_sep=1., max_sep=25., verbose=2)
    rr.process(rand)
    print('rr.npairs = ',rr.npairs)

    dr = treecorr.NNCorrelation(bin_size=0.1, min_sep=1., max_sep=25., verbose=2)
    dr.process(cat,rand)
    print('dr.npairs = ',dr.npairs)

    r = numpy.exp(dd.meanlogr)
    true_xi = 1./(8.*numpy.pi**1.5) * (L/s)**3 * numpy.exp(-0.25*r**2/s**2) - 1.

    xi, varxi = dd.calculateXi(rr,dr)
    print('xi = ',xi)
    print('true_xi = ',true_xi)
    print('ratio = ',xi / true_xi)
    print('diff = ',xi - true_xi)
    print('max rel diff = ',max(abs((xi - true_xi)/true_xi)))
    assert max(abs(xi - true_xi)/true_xi) < 0.1

    simple_xi, varxi = dd.calculateXi(rr)
    print('simple xi = ',simple_xi)
    print('max rel diff = ',max(abs((simple_xi - true_xi)/true_xi)))
    assert max(abs(simple_xi - true_xi)/true_xi) < 0.1

    # Check that we get the same result using the corr2 executable:
    if __name__ == '__main__':
        cat.write(os.path.join('data','nn_3d_data.dat'))
        rand.write(os.path.join('data','nn_3d_rand.dat'))
        import subprocess
        p = subprocess.Popen( ["corr2","nn_3d.params"] )
        p.communicate()
        corr2_output = numpy.loadtxt(os.path.join('output','nn_3d.out'))
        print('xi = ',xi)
        print('from corr2 output = ',corr2_output[:,2])
        print('ratio = ',corr2_output[:,2]/xi)
        print('diff = ',corr2_output[:,2]-xi)
        numpy.testing.assert_almost_equal(corr2_output[:,2]/xi, 1., decimal=3)


def test_list():
    # Test that we can use a list of files for either data or rand or both.

    nobj = 5000
    numpy.random.seed(8675309)

    ncats = 3
    data_cats = []
    rand_cats = []

    s = 10.
    L = 50. * s
    numpy.random.seed(8675309)

    x = numpy.random.normal(0,s, (nobj,ncats) )
    y = numpy.random.normal(0,s, (nobj,ncats) )
    data_cats = [ treecorr.Catalog(x=x[:,k],y=y[:,k]) for k in range(ncats) ]
    rx = (numpy.random.random_sample((nobj,ncats))-0.5) * L
    ry = (numpy.random.random_sample((nobj,ncats))-0.5) * L
    rand_cats = [ treecorr.Catalog(x=rx[:,k],y=ry[:,k]) for k in range(ncats) ]

    dd = treecorr.NNCorrelation(bin_size=0.1, min_sep=1., max_sep=25., verbose=2)
    dd.process(data_cats)
    print('dd.npairs = ',dd.npairs)

    rr = treecorr.NNCorrelation(bin_size=0.1, min_sep=1., max_sep=25., verbose=2)
    rr.process(rand_cats)
    print('rr.npairs = ',rr.npairs)

    xi, varxi = dd.calculateXi(rr)
    print('xi = ',xi)

    # Now do the same thing with one big catalog for each.
    ddx = treecorr.NNCorrelation(bin_size=0.1, min_sep=1., max_sep=25., verbose=2)
    rrx = treecorr.NNCorrelation(bin_size=0.1, min_sep=1., max_sep=25., verbose=2)
    data_catx = treecorr.Catalog(x=x.reshape( (nobj*ncats,) ), y=y.reshape( (nobj*ncats,) ))
    rand_catx = treecorr.Catalog(x=rx.reshape( (nobj*ncats,) ), y=ry.reshape( (nobj*ncats,) ))
    ddx.process(data_catx)
    rrx.process(rand_catx)
    xix, varxix = ddx.calculateXi(rrx)

    print('ddx.npairs = ',ddx.npairs)
    print('rrx.npairs = ',rrx.npairs)
    print('xix = ',xix)
    print('ratio = ',xi/xix)
    print('diff = ',xi-xix)
    numpy.testing.assert_almost_equal(xix/xi, 1., decimal=2)

    # Check that we get the same result using the corr2 executable:
    file_list = []
    rand_file_list = []
    for k in range(ncats):
        file_name = os.path.join('data','nn_list_data%d.dat'%k)
        with open(file_name, 'w') as fid:
            for i in range(nobj):
                fid.write(('%.8f %.8f\n')%(x[i,k],y[i,k]))
        file_list.append(file_name)

        rand_file_name = os.path.join('data','nn_list_rand%d.dat'%k)
        with open(rand_file_name, 'w') as fid:
            for i in range(nobj):
                fid.write(('%.8f %.8f\n')%(rx[i,k],ry[i,k]))
        rand_file_list.append(rand_file_name)

    list_name = os.path.join('data','nn_list_data_files.txt')
    with open(list_name, 'w') as fid:
        for file_name in file_list:
            fid.write('%s\n'%file_name)
    rand_list_name = os.path.join('data','nn_list_rand_files.txt')
    with open(rand_list_name, 'w') as fid:
        for file_name in rand_file_list:
            fid.write('%s\n'%file_name)

    file_namex = os.path.join('data','nn_list_datax.dat')
    with open(file_namex, 'w') as fid:
        for k in range(ncats):
            for i in range(nobj):
                fid.write(('%.8f %.8f\n')%(x[i,k],y[i,k]))

    rand_file_namex = os.path.join('data','nn_list_randx.dat')
    with open(rand_file_namex, 'w') as fid:
        for k in range(ncats):
            for i in range(nobj):
                fid.write(('%.8f %.8f\n')%(rx[i,k],ry[i,k]))

    import subprocess
    p = subprocess.Popen( ["corr2","nn_list1.params"] )
    p.communicate()
    corr2_output = numpy.loadtxt(os.path.join('output','nn_list1.out'))
    print('xi = ',xi)
    print('from corr2 output = ',corr2_output[:,2])
    print('ratio = ',corr2_output[:,2]/xi)
    print('diff = ',corr2_output[:,2]-xi)
    numpy.testing.assert_almost_equal(corr2_output[:,2]/xi, 1., decimal=3)

    import subprocess
    p = subprocess.Popen( ["corr2","nn_list2.params"] )
    p.communicate()
    corr2_output = numpy.loadtxt(os.path.join('output','nn_list2.out'))
    print('xi = ',xi)
    print('from corr2 output = ',corr2_output[:,2])
    print('ratio = ',corr2_output[:,2]/xi)
    print('diff = ',corr2_output[:,2]-xi)
    numpy.testing.assert_almost_equal(corr2_output[:,2]/xi, 1., decimal=2)

    import subprocess
    p = subprocess.Popen( ["corr2","nn_list3.params"] )
    p.communicate()
    corr2_output = numpy.loadtxt(os.path.join('output','nn_list3.out'))
    print('xi = ',xi)
    print('from corr2 output = ',corr2_output[:,2])
    print('ratio = ',corr2_output[:,2]/xi)
    print('diff = ',corr2_output[:,2]-xi)
    numpy.testing.assert_almost_equal(corr2_output[:,2]/xi, 1., decimal=2)


if __name__ == '__main__':
    test_direct_count()
    test_direct_3d()
    test_nn()
    test_3d()
    test_list()
