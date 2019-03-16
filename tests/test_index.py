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

    cat = treecorr.Catalog(x=x, y=y, w=w)
    field = cat.getNField()

    t0 = time.time()
    n1 = field.count_near(0.5, 0.8, 0., 0.1)
    t1 = time.time()
    n2 = np.sum((x[w>0]-0.5)**2 + (y[w>0]-0.8)**2 < 0.1**2)
    t2 = time.time()
    n3 = field.count_near(0.5, 0.8, 0., 0.1)
    t3 = time.time()
    print('n1 = ',n1,'  time = ',t1-t0)
    print('n2 = ',n2,'  time = ',t2-t1)
    print('n3 = ',n3,'  time = ',t3-t2)
    assert n2 == n1
    assert n2 == n3
    assert t1-t0 < t2-t1
    assert t3-t2 < t2-t1

    # 3D coords

    r = np.sqrt(x*x+y*y+z*z)
    dec = np.arcsin(z/r) * coord.radians / coord.degrees
    ra = np.arctan2(y,x) * coord.radians / coord.degrees

    cat = treecorr.Catalog(ra=ra, dec=dec, r=r, ra_units='deg', dec_units='deg',
                           w=w, g1=w, g2=w, k=w)
    field = cat.getNField()

    t0 = time.time()
    n1 = field.count_near(0.5, 0.8, 0.3, 0.1)
    t1 = time.time()
    n2 = np.sum((x[w>0]-0.5)**2 + (y[w>0]-0.8)**2 + (z[w>0]-0.3)**2 < 0.1**2)
    t2 = time.time()
    n3 = field.count_near(0.5, 0.8, 0.3, 0.1)
    t3 = time.time()
    print('n1 = ',n1,'  time = ',t1-t0)
    print('n2 = ',n2,'  time = ',t2-t1)
    print('n3 = ',n3,'  time = ',t3-t2)
    assert n2 == n1
    assert n2 == n3
    assert t1-t0 < t2-t1
    assert t3-t2 < t2-t1

    # Spherical
    cat = treecorr.Catalog(ra=ra, dec=dec, ra_units='deg', dec_units='deg',
                           w=w, g1=w, g2=w, k=w)
    field = cat.getNField()

    x /= r
    y /= r
    z /= r
    t0 = time.time()
    n1 = field.count_near(0.5, 0.8, 0.3317, 0.1)
    t1 = time.time()
    n2 = np.sum((x[w>0]-0.5)**2 + (y[w>0]-0.8)**2 + (z[w>0]-0.3317)**2 < 0.1**2)
    t2 = time.time()
    n3 = field.count_near(0.5, 0.8, 0.3317, 0.1)
    t3 = time.time()
    print('n1 = ',n1,'  time = ',t1-t0)
    print('n2 = ',n2,'  time = ',t2-t1)
    print('n3 = ',n3,'  time = ',t3-t2)
    assert n1 == n2
    assert n3 == n2
    assert t1-t0 < t2-t1
    assert t3-t2 < t2-t1

    # Finally, check that GField and KField also work.
    kfield = cat.getKField()
    gfield = cat.getGField()
    t4 = time.time()
    n4 = kfield.count_near(0.5, 0.8, 0.3317, 0.1)
    t5 = time.time()
    n5 = gfield.count_near(0.5, 0.8, 0.3317, 0.1)
    t6 = time.time()
    print('n4 = ',n4,'  time = ',t5-t4)
    print('n5 = ',n5,'  time = ',t6-t5)
    assert n4 == n2
    assert n5 == n2


if __name__ == '__main__':
    test_count_near()
