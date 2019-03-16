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
import numpy as np
import time
import coord
import treecorr

from test_helper import CaptureLog, assert_raises

def test_count_near():

    nobj = 100000
    np.random.seed(8675309)
    x = np.random.random_sample(nobj)   # All from 0..1
    y = np.random.random_sample(nobj)
    z = np.random.random_sample(nobj)
    w = np.random.random_sample(nobj)

    # Some elements have w = 0.
    use = np.random.randint(30, size=nobj).astype(float)
    w[use == 0] = 0

    # Start with flat coords

    cat = treecorr.Catalog(x=x, y=y, w=w, g1=w, g2=w, k=w)
    field = cat.getNField()

    t0 = time.time()
    n1 = np.sum((x[w>0]-0.5)**2 + (y[w>0]-0.8)**2 < 0.1**2)
    t1 = time.time()
    n2 = field.count_near(x=0.5, y=0.8, sep=0.1)
    t2 = time.time()
    n3 = field.count_near(x=0.5, y=0.8, sep=0.1)
    t3 = time.time()
    print('n1 = ',n1,'  time = ',t1-t0)
    print('n2 = ',n2,'  time = ',t2-t1)
    print('n3 = ',n3,'  time = ',t3-t2)
    assert n2 == n1
    assert n3 == n1
    assert t2-t1 < t1-t0
    assert t3-t2 < t1-t0

    # Check G and K with other allowed argument patterns.
    kfield = cat.getKField()
    gfield = cat.getGField()
    n4 = kfield.count_near(0.5, 0.8, sep=0.1)
    n5 = gfield.count_near(0.5, 0.8, 0.1)
    assert n4 == n1
    assert n5 == n1

    # 3D coords

    r = np.sqrt(x*x+y*y+z*z)
    dec = np.arcsin(z/r) * coord.radians / coord.degrees
    ra = np.arctan2(y,x) * coord.radians / coord.degrees

    cat = treecorr.Catalog(ra=ra, dec=dec, r=r, ra_units='deg', dec_units='deg',
                           w=w, g1=w, g2=w, k=w)
    field = cat.getNField()

    t0 = time.time()
    n1 = np.sum((x[w>0]-0.5)**2 + (y[w>0]-0.8)**2 + (z[w>0]-0.3)**2 < 0.1**2)
    t1 = time.time()
    n2 = field.count_near(x=0.5, y=0.8, z=0.3, sep=0.1)
    t2 = time.time()
    c = coord.CelestialCoord.from_xyz(0.5,0.8,0.3)
    r0 = np.sqrt(0.5**2+0.8**2+0.3**2)
    n3 = field.count_near(ra=c.ra, dec=c.dec, r=r0, sep=0.1)
    t3 = time.time()
    print('n1 = ',n1,'  time = ',t1-t0)
    print('n2 = ',n2,'  time = ',t2-t1)
    print('n3 = ',n3,'  time = ',t3-t2)
    assert n2 == n1
    assert n3 == n1
    assert t2-t1 < t1-t0
    assert t3-t2 < t1-t0

    # Check G and K with other allowed argument patterns.
    kfield = cat.getKField()
    gfield = cat.getGField()
    n4 = kfield.count_near(0.5, 0.8, 0.3, sep=0.1)
    n5 = gfield.count_near(0.5, 0.8, 0.3, 0.1)
    n6 = kfield.count_near(c, r0, 0.1)
    n7 = kfield.count_near(c.ra, c.dec, r0, 0.1)
    n8 = gfield.count_near(c.ra.rad, c.dec.rad, r0, 0.1, ra_units='rad', dec_units='rad')
    n9 = gfield.count_near(c, r0, sep=0.1)
    n10 = kfield.count_near(c.ra, c.dec, r0, sep=0.1)
    n11 = kfield.count_near(c.ra.rad, c.dec.rad, r0, ra_units='rad', dec_units='rad', sep=0.1)
    n12 = gfield.count_near(c, r=r0, sep=0.1)
    n13 = gfield.count_near(c.ra/coord.hours, c.dec/coord.degrees, r=r0, sep=0.1,
                            ra_units='hrs', dec_units='degrees')
    assert n4 == n1
    assert n5 == n1
    assert n6 == n1
    assert n7 == n1
    assert n8 == n1
    assert n9 == n1
    assert n10 == n1
    assert n11 == n1
    assert n12 == n1
    assert n13 == n1

    # Spherical
    cat = treecorr.Catalog(ra=ra, dec=dec, ra_units='deg', dec_units='deg',
                           w=w, g1=w, g2=w, k=w)
    field = cat.getNField()

    x /= r
    y /= r
    z /= r
    c = coord.CelestialCoord.from_xyz(0.5,0.8,0.3)
    x0,y0,z0 = c.get_xyz()
    t0 = time.time()
    n1 = np.sum((x[w>0]-x0)**2 + (y[w>0]-y0)**2 + (z[w>0]-z0)**2 < 0.1**2)
    t1 = time.time()
    n2 = field.count_near(c, sep=0.1*coord.radians)
    t2 = time.time()
    n3 = field.count_near(ra=c.ra.rad, dec=c.dec.rad, ra_units='radians', dec_units='radians',
                          sep=0.1 * coord.radians)
    t3 = time.time()
    print('n1 = ',n1,'  time = ',t1-t0)
    print('n2 = ',n2,'  time = ',t2-t1)
    print('n3 = ',n3,'  time = ',t3-t2)
    assert n2 == n1
    assert n3 == n1
    assert t2-t1 < t1-t0
    assert t3-t2 < t1-t0

    # Check G and K with other allowed argument patterns.
    kfield = cat.getKField()
    gfield = cat.getGField()
    n4 = kfield.count_near(ra=c.ra/coord.degrees, dec=c.dec/coord.degrees,
                           ra_units='deg', dec_units='deg', sep=0.1, sep_units='rad')
    n5 = gfield.count_near(ra=c.ra, dec=c.dec, sep=0.1*coord.radians/coord.degrees, sep_units='deg')
    n6 = gfield.count_near(c, 0.1*coord.radians/coord.degrees, sep_units='deg')
    n7 = kfield.count_near(c.ra, c.dec, sep=0.1*coord.radians)
    n8 = kfield.count_near(c.ra, c.dec, 18./np.pi*coord.degrees)
    assert n4 == n1
    assert n5 == n1
    assert n6 == n1
    assert n7 == n1
    assert n8 == n1


if __name__ == '__main__':
    test_count_near()
