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

def test_single():
    # Use gamma_t(r) = A exp(-r^2/2s^2) around a single lens
    # i.e. gamma(r) = -A exp(-r^2/2s^2) (x+iy)^2/r^2

    nsource = 1000000
    A = 0.05
    s = 10. * treecorr.angle_units['arcmin']
    L = 50. * s
    numpy.random.seed(8675309)
    xs = (numpy.random.random_sample(nsource)-0.5) * L
    ys = (numpy.random.random_sample(nsource)-0.5) * L
    r2 = (xs**2 + ys**2)
    g1 = -A * numpy.exp(-0.5*r2/s**2) * (xs**2-ys**2)/r2
    g2 = -A * numpy.exp(-0.5*r2/s**2) * (2.*xs*ys)/r2

    lens_cat = treecorr.Catalog(x=[0], y=[0])
    source_cat = treecorr.Catalog(x=xs, y=ys, g1=g1, g2=g2)
    ng = treecorr.NGCorrelation(bin_size=0.1, min_sep=1., max_sep=100., sep_units='arcmin',
                                verbose=2)
    ng.process(lens_cat, source_cat)

    r = numpy.exp(ng.meanlogr) * treecorr.angle_units['arcmin']
    true_gt = A * numpy.exp(-0.5*r**2/s**2)

    print 'ng.xi = ',ng.xi
    print 'true_gammat = ',true_gt
    print 'ratio = ',ng.xi / true_gt
    print 'diff = ',ng.xi - true_gt
    print 'max diff = ',max(abs(ng.xi - true_gt))
    assert max(abs(ng.xi - true_gt)) < 3.e-4


def test_ng():
    # Use gamma_t(r) = A exp(-r^2/2s^2) around a bunch of foregroung lenses.
    # i.e. gamma(r) = -A exp(-r^2/2s^2) (x+iy)^2/r^2

    nlens = 1000
    nsource = 100000
    A = 0.05
    s = 10. * treecorr.angle_units['arcmin']
    L = 50. * s
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
        g1 += -A * numpy.exp(-0.5*r2/s**2) * (dx**2-dy**2)/r2
        g2 += -A * numpy.exp(-0.5*r2/s**2) * (2.*dx*dy)/r2

    lens_cat = treecorr.Catalog(x=xl, y=yl)
    source_cat = treecorr.Catalog(x=xs, y=ys, g1=g1, g2=g2)
    ng = treecorr.NGCorrelation(bin_size=0.1, min_sep=1., max_sep=100., sep_units='arcmin',
                                verbose=2)
    ng.process(lens_cat, source_cat)

    r = numpy.exp(ng.meanlogr) * treecorr.angle_units['arcmin']
    true_gt = A * numpy.exp(-0.5*r**2/s**2)

    print 'ng.xi = ',ng.xi
    print 'true_gammat = ',true_gt
    print 'ratio = ',ng.xi / true_gt
    print 'diff = ',ng.xi - true_gt
    print 'max diff = ',max(abs(ng.xi - true_gt))
    assert max(abs(ng.xi - true_gt)) < 3.e-3

if __name__ == '__main__':
    test_single()
    test_ng()
