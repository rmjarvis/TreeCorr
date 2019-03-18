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
import gc
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

    x0 = 0.5
    y0 = 0.8
    z0 = 0.3
    sep = 0.03

    # Start with flat coords

    cat = treecorr.Catalog(x=x, y=y, w=w, g1=w, g2=w, k=w)
    field = cat.getNField()

    t0 = time.time()
    n1 = np.sum((x[w>0]-x0)**2 + (y[w>0]-y0)**2 < sep**2)
    t1 = time.time()
    n2 = field.count_near(x=x0, y=y0, sep=sep)
    t2 = time.time()
    n3 = field.count_near(x=x0, y=y0, sep=sep)
    t3 = time.time()
    print('n1 = ',n1,'  time = ',t1-t0)
    print('n2 = ',n2,'  time = ',t2-t1)
    print('n3 = ',n3,'  time = ',t3-t2)
    assert n2 == n1
    assert n3 == n1
    assert t2-t1 < t1-t0
    assert t3-t2 < t1-t0

    # Check G and K with other allowed argument patterns.
    kfield = cat.getKField(min_size=0.01, max_size=sep)
    gfield = cat.getGField(min_size=0.05, max_size=sep, max_top=2)
    n4 = kfield.count_near(x0, y0, sep=sep)
    n5 = gfield.count_near(x0, y0, sep)
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
    n1 = np.sum((x[w>0]-x0)**2 + (y[w>0]-y0)**2 + (z[w>0]-z0)**2 < sep**2)
    t1 = time.time()
    n2 = field.count_near(x=x0, y=y0, z=z0, sep=sep)
    t2 = time.time()
    c = coord.CelestialCoord.from_xyz(x0,y0,z0)
    r0 = np.sqrt(x0**2+y0**2+z0**2)
    n3 = field.count_near(ra=c.ra, dec=c.dec, r=r0, sep=sep)
    t3 = time.time()
    print('n1 = ',n1,'  time = ',t1-t0)
    print('n2 = ',n2,'  time = ',t2-t1)
    print('n3 = ',n3,'  time = ',t3-t2)
    assert n2 == n1
    assert n3 == n1
    assert t2-t1 < t1-t0
    assert t3-t2 < t1-t0

    # Check G and K with other allowed argument patterns.
    kfield = cat.getKField(min_size=0.01, max_size=sep)
    gfield = cat.getGField(min_size=0.05, max_size=sep, max_top=2)
    n4 = kfield.count_near(x0, y0, z0, sep=sep)
    n5 = gfield.count_near(x0, y0, z0, sep)
    n6 = kfield.count_near(c, r0, sep)
    n7 = kfield.count_near(c.ra, c.dec, r0, sep)
    n8 = gfield.count_near(c.ra.rad, c.dec.rad, r0, sep, ra_units='rad', dec_units='rad')
    n9 = gfield.count_near(c, r0, sep=sep)
    n10 = kfield.count_near(c.ra, c.dec, r0, sep=sep)
    n11 = kfield.count_near(c.ra.rad, c.dec.rad, r0, ra_units='rad', dec_units='rad', sep=sep)
    n12 = gfield.count_near(c, r=r0, sep=sep)
    n13 = gfield.count_near(c.ra/coord.hours, c.dec/coord.degrees, r=r0, sep=sep,
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
    c = coord.CelestialCoord.from_xyz(x0,y0,z0)
    x0,y0,z0 = c.get_xyz()
    r0 = 2 * np.sin(sep/2)  # length of chord subtending 0.1 radians.
    t0 = time.time()
    n1 = np.sum((x[w>0]-x0)**2 + (y[w>0]-y0)**2 + (z[w>0]-z0)**2 < r0**2)
    t1 = time.time()
    n2 = field.count_near(c, sep=sep*coord.radians)
    t2 = time.time()
    n3 = field.count_near(ra=c.ra.rad, dec=c.dec.rad, ra_units='radians', dec_units='radians',
                          sep=sep * coord.radians)
    t3 = time.time()
    print('n1 = ',n1,'  time = ',t1-t0)
    print('n2 = ',n2,'  time = ',t2-t1)
    print('n3 = ',n3,'  time = ',t3-t2)
    assert n2 == n1
    assert n3 == n1
    assert t2-t1 < t1-t0
    assert t3-t2 < t1-t0

    # Check G and K with other allowed argument patterns.
    kfield = cat.getKField(min_size=0.01, max_size=0.1)
    gfield = cat.getGField(max_size=0.05, max_top=2)
    n4 = kfield.count_near(ra=c.ra/coord.degrees, dec=c.dec/coord.degrees,
                           ra_units='deg', dec_units='deg', sep=sep, sep_units='rad')
    n5 = gfield.count_near(ra=c.ra, dec=c.dec, sep=sep*coord.radians/coord.degrees, sep_units='deg')
    n6 = gfield.count_near(c, sep*coord.radians/coord.degrees, sep_units='deg')
    n7 = kfield.count_near(c.ra, c.dec, sep=sep*coord.radians)
    n8 = kfield.count_near(c.ra, c.dec, sep*180./np.pi*coord.degrees)
    assert n4 == n1
    assert n5 == n1
    assert n6 == n1
    assert n7 == n1
    assert n8 == n1

    # I haven't figured this out yet, but in python 3.7, if I omit this, then
    # things eventually hang when garbage collecting the fields.
    gc.collect()


def test_get_near():

    nobj = 100000
    np.random.seed(8675309)
    x = np.random.random_sample(nobj)   # All from 0..1
    y = np.random.random_sample(nobj)
    z = np.random.random_sample(nobj)
    w = np.random.random_sample(nobj)
    use = np.random.randint(30, size=nobj).astype(float)
    w[use == 0] = 0

    x0 = 0.5
    y0 = 0.8
    z0 = 0.3
    sep = 0.03

    # Put a small cluster inside our search radius
    x[100:130] = np.random.normal(x0+0.03, 0.001, 30)
    y[100:130] = np.random.normal(y0-0.02, 0.001, 30)
    z[100:130] = np.random.normal(z0+0.01, 0.001, 30)

    # Put another small cluster right on the edge of our search radius
    x[500:550] = np.random.normal(x0+sep, 0.001, 50)
    y[500:550] = np.random.normal(y0, 0.001, 50)
    z[500:550] = np.random.normal(z0, 0.001, 50)

    # Start with flat coords

    cat = treecorr.Catalog(x=x, y=y, w=w, g1=w, g2=w, k=w)
    field = cat.getNField()

    t0 = time.time()
    i1 = np.where(((x-x0)**2 + (y-y0)**2 < sep**2) & (w > 0))[0]
    t1 = time.time()
    i2 = field.get_near(x=x0, y=y0, sep=sep)
    t2 = time.time()
    i3 = field.get_near(x0, y0, sep)
    t3 = time.time()
    print('i1 = ',i1[:20],'  time = ',t1-t0)
    print('i2 = ',i2[:20],'  time = ',t2-t1)
    print('i3 = ',i3[:20],'  time = ',t3-t2)
    np.testing.assert_array_equal(i2, i1)
    np.testing.assert_array_equal(i3, i1)
    #assert t2-t1 < t1-t0  # These don't always pass.  The tree version is usually a bit faster,
    #assert t3-t2 < t1-t0  # but not by much and not always.  So don't require it in unit test.

    # Check G and K
    kfield = cat.getKField(min_size=0.01, max_size=sep)
    gfield = cat.getGField(min_size=0.05, max_size=sep, max_top=2)
    i4 = kfield.get_near(x0, y0, sep=sep)
    i5 = gfield.get_near(x0, y0, sep=sep)
    np.testing.assert_array_equal(i4, i1)
    np.testing.assert_array_equal(i5, i1)

    # 3D coords

    r = np.sqrt(x*x+y*y+z*z)
    dec = np.arcsin(z/r) * coord.radians / coord.degrees
    ra = np.arctan2(y,x) * coord.radians / coord.degrees

    cat = treecorr.Catalog(ra=ra, dec=dec, r=r, ra_units='deg', dec_units='deg',
                           w=w, g1=w, g2=w, k=w)
    field = cat.getNField()

    t0 = time.time()
    i1 = np.where(((x-x0)**2 + (y-y0)**2 + (z-z0)**2 < sep**2) & (w > 0))[0]
    t1 = time.time()
    i2 = field.get_near(x=x0, y=y0, z=z0, sep=sep)
    t2 = time.time()
    c = coord.CelestialCoord.from_xyz(x0,y0,z0)
    r0 = np.sqrt(x0**2+y0**2+z0**2)
    i3 = field.get_near(ra=c.ra, dec=c.dec, r=r0, sep=sep)
    t3 = time.time()
    print('i1 = ',i1[:20],'  time = ',t1-t0)
    print('i2 = ',i2[:20],'  time = ',t2-t1)
    print('i3 = ',i3[:20],'  time = ',t3-t2)
    np.testing.assert_array_equal(i2, i1)
    np.testing.assert_array_equal(i3, i1)
    #assert t2-t1 < t1-t0
    #assert t3-t2 < t1-t0

    # Check G and K
    kfield = cat.getKField(min_size=0.01, max_size=sep)
    gfield = cat.getGField(min_size=0.05, max_size=sep, max_top=2)
    i4 = kfield.get_near(c, r0, sep)
    i5 = gfield.get_near(c.ra, c.dec, r0, sep=sep)
    np.testing.assert_array_equal(i4, i1)
    np.testing.assert_array_equal(i5, i1)

    # Spherical
    cat = treecorr.Catalog(ra=ra, dec=dec, ra_units='deg', dec_units='deg',
                           w=w, g1=w, g2=w, k=w)
    field = cat.getNField()

    x /= r
    y /= r
    z /= r
    c = coord.CelestialCoord.from_xyz(x0,y0,z0)
    x0,y0,z0 = c.get_xyz()
    r0 = 2 * np.sin(sep / 2)  # length of chord subtending sep radians.
    t0 = time.time()
    i1 = np.where(((x-x0)**2 + (y-y0)**2 + (z-z0)**2 < r0**2) & (w > 0))[0]
    t1 = time.time()
    i2 = field.get_near(c, sep=sep, sep_units='rad')
    t2 = time.time()
    i3 = field.get_near(ra=c.ra.rad, dec=c.dec.rad, ra_units='radians', dec_units='radians',
                        sep=sep * coord.radians)
    t3 = time.time()
    print('i1 = ',i1[:20],'  time = ',t1-t0)
    print('i2 = ',i2[:20],'  time = ',t2-t1)
    print('i3 = ',i3[:20],'  time = ',t3-t2)
    np.testing.assert_array_equal(i2, i1)
    np.testing.assert_array_equal(i3, i1)
    #assert t2-t1 < t1-t0
    #assert t3-t2 < t1-t0

    # Check G and K with other allowed argument patterns.
    kfield = cat.getKField(min_size=0.01, max_size=sep)
    gfield = cat.getGField(min_size=0.05, max_size=sep, max_top=2)
    i4 = gfield.get_near(c, sep*coord.radians/coord.degrees, sep_units='deg')
    i5 = kfield.get_near(c.ra, c.dec, sep*coord.radians)
    np.testing.assert_array_equal(i4, i1)
    np.testing.assert_array_equal(i5, i1)


if __name__ == '__main__':
    test_count_near()
    test_get_near()
