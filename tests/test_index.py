# Copyright (c) 2003-2019 by Mike Jarvis
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

import numpy as np
import coord
import gc
import treecorr
import platform
import timeit

from test_helper import CaptureLog, assert_raises, timer

@timer
def test_count_near():

    nobj = 100000
    rng = np.random.RandomState(8675309)
    x = rng.random_sample(nobj)   # All from 0..1
    y = rng.random_sample(nobj)
    z = rng.random_sample(nobj)
    w = rng.random_sample(nobj)

    # Some elements have w = 0.  These aren't special, but make sure it works with some w=0.
    use = rng.randint(30, size=nobj).astype(float)
    w[use == 0] = 0

    x0 = 0.5
    y0 = 0.8
    z0 = 0.3
    sep = 0.03

    # Start with flat coords

    cat = treecorr.Catalog(x=x, y=y, w=w, g1=w, g2=w, k=w, keep_zero_weight=True)
    field = cat.getNField()

    # The count_near code is only faster if the tree has been built.
    # Which is deferred until the first time you need it.
    # The simplest way to trigger the build is to ask for ntop
    field.nTopLevelNodes

    n1 = np.sum((x-x0)**2 + (y-y0)**2 < sep**2)
    t1 = min(timeit.repeat(lambda: np.sum((x-x0)**2 + (y-y0)**2 < sep**2),number=100))
    n2 = field.count_near(x=x0, y=y0, sep=sep)
    t2 = min(timeit.repeat(lambda: field.count_near(x=x0, y=y0, sep=sep), number=100))
    print('n1 = ',n1,'  time = ',t1)
    print('n2 = ',n2,'  time = ',t2)
    assert n1 == n2
    print('implementation is ',platform.python_implementation())
    if platform.python_implementation() != 'PyPy':
        # The JIT in PyPy can sometimes beat the tree code.
        # Do best of two timing test to avoid random occasional slowness.
        assert t2 < t1

    # Check G and K with other allowed argument patterns.
    kfield = cat.getKField(min_size=0.01, max_size=sep, min_top=5)
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
                           w=w, g1=w, g2=w, k=w, keep_zero_weight=True)
    field = cat.getNField()
    field.nTopLevelNodes

    n1 = np.sum((x-x0)**2 + (y-y0)**2 + (z-z0)**2 < sep**2)
    t1 = min(timeit.repeat(lambda: np.sum((x-x0)**2 + (y-y0)**2 + (z-z0)**2 < sep**2), number=100))
    n2 = field.count_near(x=x0, y=y0, z=z0, sep=sep)
    t3 = min(timeit.repeat(lambda: field.count_near(x=x0, y=y0, z=z0, sep=sep), number=100))
    c = coord.CelestialCoord.from_xyz(x0,y0,z0)
    r0 = np.sqrt(x0**2+y0**2+z0**2)
    n3 = field.count_near(ra=c.ra, dec=c.dec, r=r0, sep=sep)
    t3 = min(timeit.repeat(lambda: field.count_near(ra=c.ra, dec=c.dec, r=r0, sep=sep), number=100))
    print('n1 = ',n1,'  time = ',t1)
    print('n2 = ',n2,'  time = ',t2)
    print('n3 = ',n3,'  time = ',t3)
    assert n2 == n1
    assert n3 == n1
    if platform.python_implementation() != 'PyPy':
        assert t2 < t1
        assert t3 < t1

    # Check G and K with other allowed argument patterns.
    kfield = cat.getKField(min_size=0.01, max_size=sep, min_top=5)
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
                           w=w, g1=w, g2=w, k=w, keep_zero_weight=True)
    field = cat.getNField()
    field.nTopLevelNodes

    x /= r
    y /= r
    z /= r
    c = coord.CelestialCoord.from_xyz(x0,y0,z0)
    x0,y0,z0 = c.get_xyz()
    r0 = 2 * np.sin(sep/2)  # length of chord subtending 0.1 radians.
    n1 = np.sum((x-x0)**2 + (y-y0)**2 + (z-z0)**2 < r0**2)
    t1 = min(timeit.repeat(lambda: np.sum((x-x0)**2 + (y-y0)**2 + (z-z0)**2 < r0**2), number=100))
    n2 = field.count_near(c, sep=sep*coord.radians)
    t2 = min(timeit.repeat(lambda: field.count_near(c, sep=sep*coord.radians), number=100))
    n3 = field.count_near(ra=c.ra.rad, dec=c.dec.rad, ra_units='radians', dec_units='radians',
                          sep=sep * coord.radians)
    t3 = min(timeit.repeat(lambda: field.count_near(ra=c.ra.rad, dec=c.dec.rad, ra_units='radians',
                                                    dec_units='radians', sep=sep * coord.radians),
                           number=100))
    print('n1 = ',n1,'  time = ',t1)
    print('n2 = ',n2,'  time = ',t2)
    print('n3 = ',n3,'  time = ',t3)
    assert n2 == n1
    assert n3 == n1
    if platform.python_implementation() != 'PyPy':
        assert t2 < t1
        assert t3 < t1

    # Check G and K with other allowed argument patterns.
    kfield = cat.getKField(min_size=0.01, max_size=0.1, min_top=5)
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


@timer
def test_get_near():

    nobj = 100000
    rng = np.random.RandomState(8675309)
    x = rng.random_sample(nobj)   # All from 0..1
    y = rng.random_sample(nobj)
    z = rng.random_sample(nobj)
    w = rng.random_sample(nobj)
    use = rng.randint(30, size=nobj).astype(float)
    w[use == 0] = 0

    x0 = 0.5
    y0 = 0.8
    z0 = 0.3
    sep = 0.03

    # Put a small cluster inside our search radius
    x[100:130] = rng.normal(x0+0.03, 0.001, 30)
    y[100:130] = rng.normal(y0-0.02, 0.001, 30)
    z[100:130] = rng.normal(z0+0.01, 0.001, 30)

    # Put another small cluster right on the edge of our search radius
    x[500:550] = rng.normal(x0+sep, 0.001, 50)
    y[500:550] = rng.normal(y0, 0.001, 50)
    z[500:550] = rng.normal(z0, 0.001, 50)

    # Start with flat coords

    cat = treecorr.Catalog(x=x, y=y, w=w, g1=w, g2=w, k=w, keep_zero_weight=True)
    field = cat.getNField()
    field.nTopLevelNodes

    i1 = np.where(((x-x0)**2 + (y-y0)**2 < sep**2))[0]
    t1 = min(timeit.repeat(lambda: np.where(((x-x0)**2 + (y-y0)**2 < sep**2))[0], number=100))
    i2 = field.get_near(x=x0, y=y0, sep=sep)
    t2 = min(timeit.repeat(lambda: field.get_near(x=x0, y=y0, sep=sep), number=100))
    i3 = field.get_near(x0, y0, sep)
    t3 = min(timeit.repeat(lambda: field.get_near(x0, y0, sep), number=100))
    print('i1 = ',i1[:20],'  time = ',t1)
    print('i2 = ',i2[:20],'  time = ',t2)
    print('i3 = ',i3[:20],'  time = ',t3)
    np.testing.assert_array_equal(i2, i1)
    np.testing.assert_array_equal(i3, i1)
    #assert t2 < t1    # These don't always pass.  The tree version is usually a faster,
    #assert t3 < t1    # but not always.  So don't require it in unit test.

    # Invalid ways to specify x,y,sep
    assert_raises(TypeError, field.get_near)
    assert_raises(TypeError, field.get_near, x0)
    assert_raises(TypeError, field.get_near, x0, y0)
    assert_raises(TypeError, field.get_near, x0, y0, sep, sep)
    assert_raises(TypeError, field.get_near, x=x0, y=y0)
    assert_raises(TypeError, field.get_near, x=x0, sep=sep)
    assert_raises(TypeError, field.get_near, y=y0, sep=sep)
    assert_raises(TypeError, field.get_near, x=x0, y=y0, z=x0, sep=sep)
    assert_raises(TypeError, field.get_near, ra=x0, dec=y0, sep=sep)
    assert_raises(TypeError, field.get_near, coord.CelestialCoord.from_xyz(x0,y0,x0), sep=sep)

    # Check G and K
    kfield = cat.getKField(min_size=0.01, max_size=sep, min_top=5)
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
                           w=w, g1=w, g2=w, k=w, keep_zero_weight=True)
    field = cat.getNField()
    field.nTopLevelNodes

    i1 = np.where(((x-x0)**2 + (y-y0)**2 + (z-z0)**2 < sep**2))[0]
    t1 = min(timeit.repeat(lambda: np.where(((x-x0)**2 + (y-y0)**2 + (z-z0)**2 < sep**2))[0],
                           number=100))
    i2 = field.get_near(x=x0, y=y0, z=z0, sep=sep)
    t2 = min(timeit.repeat(lambda: field.get_near(x=x0, y=y0, z=z0, sep=sep), number=100))
    c = coord.CelestialCoord.from_xyz(x0,y0,z0)
    r0 = np.sqrt(x0**2+y0**2+z0**2)
    i3 = field.get_near(ra=c.ra, dec=c.dec, r=r0, sep=sep)
    t3 = min(timeit.repeat(lambda: field.get_near(ra=c.ra, dec=c.dec, r=r0, sep=sep), number=100))
    print('i1 = ',i1[:20],'  time = ',t1)
    print('i2 = ',i2[:20],'  time = ',t2)
    print('i3 = ',i3[:20],'  time = ',t3)
    np.testing.assert_array_equal(i2, i1)
    np.testing.assert_array_equal(i3, i1)
    #assert t2 < t1
    #assert t3 < t1

    # Invalid ways to specify x,y,z,sep
    ra0 = c.ra / coord.degrees
    dec0 = c.dec / coord.degrees
    assert_raises(TypeError, field.get_near)
    assert_raises(TypeError, field.get_near, x0)
    assert_raises(TypeError, field.get_near, x0, y0)
    assert_raises(TypeError, field.get_near, x0, y0, z0)
    assert_raises(TypeError, field.get_near, x=x0)
    assert_raises(TypeError, field.get_near, x=x0, y=y0)
    assert_raises(TypeError, field.get_near, x=x0, y=y0, z=z0)
    assert_raises(TypeError, field.get_near, ra=ra0)
    assert_raises(TypeError, field.get_near, ra=ra0, dec=dec0)
    assert_raises(TypeError, field.get_near, ra=ra0, dec=dec0, r=r0)
    assert_raises(TypeError, field.get_near, ra0, dec0, sep, ra_units='deg')
    assert_raises(TypeError, field.get_near, ra0, dec0, sep, dec_units='deg')
    assert_raises(TypeError, field.get_near, ra0, dec0, sep, sep_units='rad')
    assert_raises(TypeError, field.get_near, ra0, dec0, sep, ra_units='deg', dec_units='deg')
    assert_raises(TypeError, field.get_near, ra0, dec0, sep, ra_units='deg', sep_units='rad')
    assert_raises(TypeError, field.get_near, ra0, dec0, sep, dec_units='deg', sep_units='rad')
    assert_raises(TypeError, field.get_near, ra0, dec0, sep, sep,
                  ra_units='deg', dec_units='deg', sep_units='rad')
    assert_raises(TypeError, field.get_near, ra=ra0)
    assert_raises(TypeError, field.get_near, dec=dec0)
    assert_raises(TypeError, field.get_near, ra=ra0, dec=dec0)
    assert_raises(TypeError, field.get_near, ra=ra0, dec=dec0, sep=sep)
    assert_raises(TypeError, field.get_near, ra=ra0, dec=dec0, sep=sep, ra_units='deg')
    assert_raises(TypeError, field.get_near, ra=ra0, dec=dec0, sep=sep, dec_units='deg')
    assert_raises(TypeError, field.get_near, ra=ra0, dec=dec0, sep=sep,
                  ra_units='deg', dec_units='deg')
    assert_raises(TypeError, field.get_near, ra, dec=dec, sep=sep,
                  ra_units='deg', dec_units='deg', sep_units='rad')
    assert_raises(TypeError, field.get_near, ra, dec=dec, sep=sep,
                  ra_units='deg', dec_units='deg', sep_units='rad')
    assert_raises(TypeError, field.get_near, c)
    assert_raises(TypeError, field.get_near, c, r=r0)
    assert_raises(TypeError, field.get_near, c, r=r0, sep=sep, sep_units='rad')
    assert_raises(TypeError, field.get_near, c, r0)
    assert_raises(TypeError, field.get_near, c, r0, sep=sep, sep_units='rad')
    assert_raises(TypeError, field.get_near, c, r0, sep, sep_units='rad')
    assert_raises(TypeError, field.get_near, c, r0, sep, 'deg')
    assert_raises(TypeError, field.get_near, c, r0, sep, sep_unit='deg')
    assert_raises(TypeError, field.get_near, c, r0, sep, sep_units='deg',
                  ra_units='deg', dec_units='deg')
    assert_raises(TypeError, field.get_near, c.ra)
    assert_raises(TypeError, field.get_near, c.ra, c.dec)
    assert_raises(TypeError, field.get_near, c.ra, c.dec, r=r0)
    assert_raises(TypeError, field.get_near, c.ra, c.dec, r0)
    assert_raises(TypeError, field.get_near, c.ra, c.dec, r0, sep=sep, extra=4)
    assert_raises(TypeError, field.get_near, c.ra, c.dec, r0, sep, extra=4)
    assert_raises(TypeError, field.get_near, c.ra, c.dec, r0, sep, sep)

    # Check G and K
    kfield = cat.getKField(min_size=0.01, max_size=sep, min_top=5)
    gfield = cat.getGField(min_size=0.05, max_size=sep, max_top=2)
    i4 = kfield.get_near(c, r0, sep)
    i5 = gfield.get_near(c.ra, c.dec, r0, sep=sep)
    np.testing.assert_array_equal(i4, i1)
    np.testing.assert_array_equal(i5, i1)

    # Spherical
    cat = treecorr.Catalog(ra=ra, dec=dec, ra_units='deg', dec_units='deg',
                           w=w, g1=w, g2=w, k=w, keep_zero_weight=True)
    field = cat.getNField()
    field.nTopLevelNodes

    x /= r
    y /= r
    z /= r
    c = coord.CelestialCoord.from_xyz(x0,y0,z0)
    x0,y0,z0 = c.get_xyz()
    r0 = 2 * np.sin(sep / 2)  # length of chord subtending sep radians.
    i1 = np.where(((x-x0)**2 + (y-y0)**2 + (z-z0)**2 < r0**2))[0]
    t1 = min(timeit.repeat(lambda: np.where(((x-x0)**2 + (y-y0)**2 + (z-z0)**2 < r0**2))[0],
                           number=100))
    i2 = field.get_near(c, sep=sep, sep_units='rad')
    t2 = min(timeit.repeat(lambda: field.get_near(c, sep=sep, sep_units='rad'), number=100))
    i3 = field.get_near(ra=c.ra.rad, dec=c.dec.rad, ra_units='radians', dec_units='radians',
                        sep=sep * coord.radians)
    t3 = min(timeit.repeat(lambda: field.get_near(ra=c.ra.rad, dec=c.dec.rad, ra_units='radians',
                                                  dec_units='radians', sep=sep * coord.radians),
                           number=100))
    print('i1 = ',i1[:20],'  time = ',t1)
    print('i2 = ',i2[:20],'  time = ',t2)
    print('i3 = ',i3[:20],'  time = ',t3)
    np.testing.assert_array_equal(i2, i1)
    np.testing.assert_array_equal(i3, i1)
    #assert t2 < t1
    #assert t3 < t1

    # Invalid ways to specify ra,dec,sep
    assert_raises(TypeError, field.get_near)
    assert_raises(TypeError, field.get_near, ra0)
    assert_raises(TypeError, field.get_near, ra0, dec0)
    assert_raises(TypeError, field.get_near, ra0, dec0, sep)
    assert_raises(TypeError, field.get_near, ra0, dec0, sep, ra_units='deg')
    assert_raises(TypeError, field.get_near, ra0, dec0, sep, dec_units='deg')
    assert_raises(TypeError, field.get_near, ra0, dec0, sep, sep_units='rad')
    assert_raises(TypeError, field.get_near, ra0, dec0, sep, ra_units='deg', dec_units='deg')
    assert_raises(TypeError, field.get_near, ra0, dec0, sep, ra_units='deg', sep_units='rad')
    assert_raises(TypeError, field.get_near, ra0, dec0, sep, dec_units='deg', sep_units='rad')
    assert_raises(TypeError, field.get_near, ra0, dec0, sep, sep,
                  ra_units='deg', dec_units='deg', sep_units='rad')
    assert_raises(TypeError, field.get_near, ra=ra0)
    assert_raises(TypeError, field.get_near, dec=dec0)
    assert_raises(TypeError, field.get_near, ra=ra0, dec=dec0)
    assert_raises(TypeError, field.get_near, ra=ra0, dec=dec0, sep=sep)
    assert_raises(TypeError, field.get_near, ra=ra0, dec=dec0, sep=sep, ra_units='deg')
    assert_raises(TypeError, field.get_near, ra=ra0, dec=dec0, sep=sep, dec_units='deg')
    assert_raises(TypeError, field.get_near, ra=ra0, dec=dec0, sep=sep,
                  ra_units='deg', dec_units='deg')
    assert_raises(TypeError, field.get_near, ra0, dec=dec0, sep=sep,
                  ra_units='deg', dec_units='deg', sep_units='rad')
    assert_raises(TypeError, field.get_near, ra0, dec=dec0, sep=sep,
                  ra_units='deg', dec_units='deg', sep_units='rad')
    assert_raises(TypeError, field.get_near, c)
    assert_raises(TypeError, field.get_near, c, sep)
    assert_raises(TypeError, field.get_near, c, sep, 'deg')
    assert_raises(TypeError, field.get_near, c, sep, sep_unit='deg')
    assert_raises(TypeError, field.get_near, c, sep, sep_units='deg',
                  ra_units='deg', dec_units='deg')

    # Check G and K with other allowed argument patterns.
    kfield = cat.getKField(min_size=0.01, max_size=sep, min_top=5)
    gfield = cat.getGField(min_size=0.05, max_size=sep, max_top=2)
    i4 = gfield.get_near(c, sep*coord.radians/coord.degrees, sep_units='deg')
    i5 = kfield.get_near(c.ra, c.dec, sep*coord.radians)
    np.testing.assert_array_equal(i4, i1)
    np.testing.assert_array_equal(i5, i1)


@timer
def test_sample_pairs():

    nobj = 10000
    rng = np.random.RandomState(8675309)
    x1 = rng.random_sample(nobj)   # All from 0..1
    y1 = rng.random_sample(nobj)
    z1 = rng.random_sample(nobj)
    w1 = rng.random_sample(nobj)
    use = rng.randint(30, size=nobj).astype(float)
    w1[use == 0] = 0
    g11 = rng.random_sample(nobj)
    g21 = rng.random_sample(nobj)
    k1 = rng.random_sample(nobj)

    x2 = rng.random_sample(nobj)   # All from 0..1
    y2 = rng.random_sample(nobj)
    z2 = rng.random_sample(nobj)
    w2 = rng.random_sample(nobj)
    use = rng.randint(30, size=nobj).astype(float)
    w2[use == 0] = 0
    g12 = rng.random_sample(nobj)
    g22 = rng.random_sample(nobj)
    k2 = rng.random_sample(nobj)

    # Start with flat coords

    cat1 = treecorr.Catalog(x=x1, y=y1, w=w1, g1=g11, g2=g21, k=k1, keep_zero_weight=True)
    cat2 = treecorr.Catalog(x=x2, y=y2, w=w2, g1=g12, g2=g22, k=k2, keep_zero_weight=True)

    # Note: extend range low enough that some bins have < 100 pairs.
    nn = treecorr.NNCorrelation(min_sep=0.001, max_sep=0.01, bin_size=0.1, max_top=0)
    nn.process(cat1, cat2)
    print('rnom = ',nn.rnom)
    print('npairs = ',nn.npairs.astype(int))

    # Start with a bin near the bottom with < 100 pairs
    # This only exercises case 1 in the sampleFrom function.
    b = 1
    i1, i2, sep = nn.sample_pairs(100, cat1, cat2,
                                  min_sep=nn.left_edges[b], max_sep=nn.right_edges[b])

    print('i1 = ',i1)
    print('i2 = ',i2)
    print('sep = ',sep)
    assert nn.npairs[b] <= 100  # i.e. make sure these next tests are what we want to do.
    assert len(i1) == nn.npairs[b]
    assert len(i2) == nn.npairs[b]
    assert len(sep) == nn.npairs[b]
    actual_sep = ((x1[i1]-x2[i2])**2 + (y1[i1]-y2[i2])**2)**0.5
    np.testing.assert_allclose(sep, actual_sep, rtol=0.1)  # half bin size with slop.
    np.testing.assert_array_less(sep, nn.right_edges[b])
    np.testing.assert_array_less(nn.left_edges[b], sep)

    # Next one that still isn't too many pairs, but more than 100
    # This exercises cases 1,2 in the sampleFrom function.
    b = 10
    i1, i2, sep = nn.sample_pairs(100, cat1, cat2,
                                  min_sep=nn.left_edges[b], max_sep=nn.right_edges[b])

    print('i1 = ',i1)
    print('i2 = ',i2)
    print('sep = ',sep)
    assert nn.npairs[b] > 100
    assert len(i1) == 100
    assert len(i2) == 100
    assert len(sep) == 100
    actual_sep = ((x1[i1]-x2[i2])**2 + (y1[i1]-y2[i2])**2)**0.5
    np.testing.assert_allclose(sep, actual_sep, rtol=0.1)
    np.testing.assert_array_less(sep, nn.right_edges[b])
    np.testing.assert_array_less(nn.left_edges[b], sep)

    # To exercise case 3, we need to go to larger separations, so the recursion
    # more often stops before getting to the leaves.
    # Also switch to 3d coordinates.

    cat1 = treecorr.Catalog(x=x1, y=y1, z=z1, w=w1, g1=g11, g2=g21, k=k1, keep_zero_weight=True)
    cat2 = treecorr.Catalog(x=x2, y=y2, z=z2, w=w2, g1=g12, g2=g22, k=k2, keep_zero_weight=True)

    gg = treecorr.GGCorrelation(min_sep=0.4, nbins=10, bin_size=0.1, max_top=0)
    gg.process(cat1, cat2)
    print('rnom = ',gg.rnom)
    print('npairs = ',gg.npairs.astype(int))
    for b in [0,5]:
        i1, i2, sep = gg.sample_pairs(100, cat1, cat2,
                                      min_sep=gg.left_edges[b], max_sep=gg.right_edges[b])

        print('len(npairs) = ',len(gg.npairs))
        print('npairs = ',gg.npairs)
        print('i1 = ',i1)
        print('i2 = ',i2)
        print('sep = ',sep)
        assert len(i1) == 100
        assert len(i2) == 100
        assert len(sep) == 100
        actual_sep = ((x1[i1]-x2[i2])**2 + (y1[i1]-y2[i2])**2 + (z1[i1]-z2[i2])**2)**0.5
        np.testing.assert_allclose(sep, actual_sep, rtol=0.2)
        np.testing.assert_array_less(sep, gg.right_edges[b])
        np.testing.assert_array_less(gg.left_edges[b], sep)

    # Check a different metric.
    # Also ability to generate the field automatically.
    cat1.clear_cache()  # Clears the previously made cat1.field
    cat2.clear_cache()  # and cat2.field

    b = 3
    with CaptureLog() as cl:
        nk = treecorr.NKCorrelation(min_sep=0.4, max_sep=1.0, bin_size=0.1, max_top=0,
                                    logger=cl.logger)
        i1, i2, sep = nk.sample_pairs(100, cat1, cat2, metric='Arc',
                                      min_sep=nk.left_edges[b], max_sep=nk.right_edges[b])
    print(cl.output)
    nk.process(cat1, cat2, metric='Arc')
    print('len(npairs) = ',len(nk.npairs))
    print('npairs = ',nk.npairs)
    assert "Sampled %d pairs out of a total of %d"%(100, nk.npairs[b]) in cl.output
    print('i1 = ',i1)
    print('i2 = ',i2)
    print('sep = ',sep)
    assert len(i1) == 100
    assert len(i2) == 100
    assert len(sep) == 100
    r1 = (x1**2 + y1**2 + z1**2)**0.5
    r2 = (x2**2 + y2**2 + z2**2)**0.5
    xx1 = x1/r1
    yy1 = y1/r1
    zz1 = z1/r1
    xx2 = x2/r2
    yy2 = y2/r2
    zz2 = z2/r2
    chord_sep = ((xx1[i1]-xx2[i2])**2 + (yy1[i1]-yy2[i2])**2 + (zz1[i1]-zz2[i2])**2)**0.5
    arc_sep = np.arcsin(chord_sep/2.)*2.
    print('arc_sep = ',arc_sep)
    np.testing.assert_allclose(sep, arc_sep, rtol=0.1)
    np.testing.assert_array_less(sep, nk.right_edges[b])
    np.testing.assert_array_less(nk.left_edges[b], sep)

    # Finally, check spherical coords with non-default units.
    ra1, dec1 = coord.CelestialCoord.xyz_to_radec(x1,y1,z1)
    ra2, dec2 = coord.CelestialCoord.xyz_to_radec(x2,y2,z2)
    cat1 = treecorr.Catalog(ra=ra1, dec=dec1, ra_units='rad', dec_units='rad')
    cat2 = treecorr.Catalog(ra=ra2, dec=dec2, ra_units='rad', dec_units='rad')

    nn = treecorr.NNCorrelation(min_sep=1., max_sep=60., nbins=50, sep_units='deg', metric='Arc')
    nn.process(cat1, cat2)
    print('rnom = ',nn.rnom)
    print('npairs = ',nn.npairs.astype(int))

    b = 5
    n = 50
    i1, i2, sep = nn.sample_pairs(n, cat1, cat2,
                                  min_sep=nn.left_edges[b], max_sep=nn.right_edges[b])

    print('i1 = ',i1)
    print('i2 = ',i2)
    print('sep = ',sep)
    assert nn.npairs[b] > n
    assert len(i1) == n
    assert len(i2) == n
    assert len(sep) == n

    c1 = [coord.CelestialCoord(r*coord.radians, d*coord.radians) for (r,d) in zip(ra1,dec1)]
    c2 = [coord.CelestialCoord(r*coord.radians, d*coord.radians) for (r,d) in zip(ra2,dec2)]
    actual_sep = np.array([c1[i1[k]].distanceTo(c2[i2[k]]) / coord.degrees for k in range(n)])
    print('actual_sep = ',actual_sep)
    np.testing.assert_allclose(sep, actual_sep, rtol=0.1)
    np.testing.assert_array_less(sep, nn.right_edges[b])
    np.testing.assert_array_less(nn.left_edges[b], sep)



if __name__ == '__main__':
    test_count_near()
    test_get_near()
    test_sample_pairs()
