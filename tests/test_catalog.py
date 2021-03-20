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

from __future__ import print_function
import numpy as np
import os
import sys
import time
import coord
import gc
import copy
import pickle
from numpy import pi
import fitsio
import treecorr

from test_helper import get_from_wiki, CaptureLog, assert_raises, do_pickle, timer, assert_warns

@timer
def test_ascii():

    nobj = 5000
    rng = np.random.RandomState(8675309)
    x = rng.random_sample(nobj)
    y = rng.random_sample(nobj)
    z = rng.random_sample(nobj)
    ra = rng.random_sample(nobj)
    dec = rng.random_sample(nobj)
    r = rng.random_sample(nobj)
    wpos = rng.random_sample(nobj)
    g1 = rng.random_sample(nobj)
    g2 = rng.random_sample(nobj)
    k = rng.random_sample(nobj)

    # Some elements have both w and wpos = 0.
    w = wpos.copy()
    use = rng.randint(30, size=nobj).astype(float)
    w[use == 0] = 0
    wpos[use == 0] = 0

    # Others just have w = 0
    use = rng.randint(30, size=nobj).astype(float)
    w[use == 0] = 0

    flags = np.zeros(nobj).astype(int)
    for flag in [ 1, 2, 4, 8, 16 ]:
        sub = rng.random_sample(nobj) < 0.1
        flags[sub] = np.bitwise_or(flags[sub], flag)

    file_name = os.path.join('data','test.dat')
    with open(file_name, 'w') as fid:
        # These are intentionally in a different order from the order we parse them.
        fid.write('# ra, dec, x, y, k, g1, g2, w, z, r, wpos, flag\n')
        for i in range(nobj):
            fid.write((('%.8f '*11)+'%d\n')%(
                ra[i],dec[i],x[i],y[i],k[i],g1[i],g2[i],w[i],z[i],r[i],wpos[i],flags[i]))

    # Check basic input
    config = {
        'x_col' : 3,
        'y_col' : 4,
        'z_col' : 9,
        'x_units' : 'rad',
        'y_units' : 'rad',
        'w_col' : 8,
        'wpos_col' : 11,
        'k_col' : 5,
        'g1_col' : 6,
        'g2_col' : 7,
        'kk_file_name' : 'kk.out',  # These make sure k and g are required.
        'gg_file_name' : 'gg.out',
        'keep_zero_weight' : True,
    }
    cat1 = treecorr.Catalog(file_name, config)
    np.testing.assert_almost_equal(cat1.x, x)
    np.testing.assert_almost_equal(cat1.y, y)
    np.testing.assert_almost_equal(cat1.z, z)
    np.testing.assert_almost_equal(cat1.w, w)
    np.testing.assert_almost_equal(cat1.g1, g1)
    np.testing.assert_almost_equal(cat1.g2, g2)
    np.testing.assert_almost_equal(cat1.k, k)
    np.testing.assert_almost_equal(cat1.wpos, wpos)

    # Check using names
    config_names = {
        'x_col' : 'x',
        'y_col' : 'y',
        'z_col' : 'z',
        'x_units' : 'rad',
        'y_units' : 'rad',
        'w_col' : 'w',
        'wpos_col' : 'wpos',
        'k_col' : 'k',
        'g1_col' : 'g1',
        'g2_col' : 'g2',
        'keep_zero_weight' : True,
    }
    cat1b = treecorr.Catalog(file_name, config_names)
    np.testing.assert_almost_equal(cat1b.x, x)
    np.testing.assert_almost_equal(cat1b.y, y)
    np.testing.assert_almost_equal(cat1b.z, z)
    np.testing.assert_almost_equal(cat1b.w, w)
    np.testing.assert_almost_equal(cat1b.g1, g1)
    np.testing.assert_almost_equal(cat1b.g2, g2)
    np.testing.assert_almost_equal(cat1b.k, k)
    np.testing.assert_almost_equal(cat1b.wpos, wpos)

    assert_raises(ValueError, treecorr.Catalog, file_name)
    assert_raises(TypeError, treecorr.Catalog, file_name, config, x=x)
    assert_raises(TypeError, treecorr.Catalog, file_name, config, y=y)
    assert_raises(TypeError, treecorr.Catalog, file_name, config, z=z)
    assert_raises(TypeError, treecorr.Catalog, file_name, config, ra=ra)
    assert_raises(TypeError, treecorr.Catalog, file_name, config, dec=dec)
    assert_raises(TypeError, treecorr.Catalog, file_name, config, r=r)
    assert_raises(TypeError, treecorr.Catalog, file_name, config, g2=g2)
    assert_raises(TypeError, treecorr.Catalog, file_name, config, k=k)
    assert_raises(TypeError, treecorr.Catalog, file_name, config, w=w)
    assert_raises(TypeError, treecorr.Catalog, file_name, config, wpos=wpos)
    assert_raises(TypeError, treecorr.Catalog, file_name, config, flag=flag)
    assert_raises(ValueError, treecorr.Catalog, file_name, config, file_type='Invalid')

    assert_raises(ValueError, treecorr.Catalog, file_name, config, x_col=-1)
    assert_raises(ValueError, treecorr.Catalog, file_name, config, x_col=100)
    assert_raises(ValueError, treecorr.Catalog, file_name, config, x_col='invalid')
    assert_raises(ValueError, treecorr.Catalog, file_name, config, y_col=-1)
    assert_raises(ValueError, treecorr.Catalog, file_name, config, y_col=100)
    assert_raises(ValueError, treecorr.Catalog, file_name, config, y_col='invalid')
    assert_raises(ValueError, treecorr.Catalog, file_name, config, z_col=-1)
    assert_raises(ValueError, treecorr.Catalog, file_name, config, z_col=100)
    assert_raises(ValueError, treecorr.Catalog, file_name, config, z_col='invalid')
    assert_raises(ValueError, treecorr.Catalog, file_name, config, w_col=-1)
    assert_raises(ValueError, treecorr.Catalog, file_name, config, w_col=100)
    assert_raises(ValueError, treecorr.Catalog, file_name, config, w_col='invalid')
    assert_raises(ValueError, treecorr.Catalog, file_name, config, wpos_col=-1)
    assert_raises(ValueError, treecorr.Catalog, file_name, config, wpos_col=100)
    assert_raises(ValueError, treecorr.Catalog, file_name, config, wpos_col='invalid')
    assert_raises(ValueError, treecorr.Catalog, file_name, config, k_col=-1)
    assert_raises(ValueError, treecorr.Catalog, file_name, config, k_col=100)
    assert_raises(ValueError, treecorr.Catalog, file_name, config, k_col='invalid')
    assert_raises(ValueError, treecorr.Catalog, file_name, config, g1_col=-1)
    assert_raises(ValueError, treecorr.Catalog, file_name, config, g1_col=100)
    assert_raises(ValueError, treecorr.Catalog, file_name, config, g1_col='invalid')
    assert_raises(ValueError, treecorr.Catalog, file_name, config, g2_col=-1)
    assert_raises(ValueError, treecorr.Catalog, file_name, config, g2_col=100)
    assert_raises(ValueError, treecorr.Catalog, file_name, config, g2_col='invalid')
    assert_raises(ValueError, treecorr.Catalog, file_name, config, flag_col=-1)
    assert_raises(ValueError, treecorr.Catalog, file_name, config, flag_col=100)
    assert_raises(ValueError, treecorr.Catalog, file_name, config, flag_col='invalid')
    assert_raises(ValueError, treecorr.Catalog, file_name, config, ra_col=4)
    assert_raises(ValueError, treecorr.Catalog, file_name, config, dec_col=4)
    assert_raises(ValueError, treecorr.Catalog, file_name, config, r_col=4)
    assert_raises(ValueError, treecorr.Catalog, file_name, config, x_col=0)
    assert_raises(ValueError, treecorr.Catalog, file_name, config, y_col=0)
    assert_raises(ValueError, treecorr.Catalog, file_name, config, x_col=0, y_col=0, z_col=0)

    # Check flags
    config['flag_col'] = 12
    print('config = ',config)
    cat2 = treecorr.Catalog(file_name, config, file_type='ASCII')
    np.testing.assert_almost_equal(cat2.w[flags==0], w[flags==0])
    np.testing.assert_almost_equal(cat2.w[flags!=0], 0.)

    # Check ok_flag
    config['ok_flag'] = 4
    cat3 = treecorr.Catalog(file_name, config)
    np.testing.assert_almost_equal(cat3.w[np.logical_or(flags==0, flags==4)],
                                   w[np.logical_or(flags==0, flags==4)])
    np.testing.assert_almost_equal(cat3.w[np.logical_and(flags!=0, flags!=4)], 0.)

    # Check ignore_flag
    del config['ok_flag']
    config['ignore_flag'] = 16
    cat4 = treecorr.Catalog(file_name, config)
    np.testing.assert_almost_equal(cat4.w[flags < 16], w[flags < 16])
    np.testing.assert_almost_equal(cat4.w[flags >= 16], 0.)

    # If weight is missing, automatically make it when there are flags
    del config['w_col']
    del config['wpos_col']
    cat4 = treecorr.Catalog(file_name, config)
    np.testing.assert_almost_equal(cat4.w[flags < 16], 1.)
    np.testing.assert_almost_equal(cat4.w[flags >= 16], 0.)
    config['w_col'] = 8  # Put them back for later.
    config['wpos_col'] = 11

    # Check different units for x,y
    config['x_units'] = 'arcsec'
    config['y_units'] = 'arcsec'
    del config['z_col']
    cat5 = treecorr.Catalog(file_name, config)
    np.testing.assert_almost_equal(cat5.x, x * (pi/180./3600.))
    np.testing.assert_almost_equal(cat5.y, y * (pi/180./3600.))

    config['x_units'] = 'arcmin'
    config['y_units'] = 'arcmin'
    cat5 = treecorr.Catalog(file_name, config)
    np.testing.assert_almost_equal(cat5.x, x * (pi/180./60.))
    np.testing.assert_almost_equal(cat5.y, y * (pi/180./60.))

    config['x_units'] = 'deg'
    config['y_units'] = 'deg'
    cat5 = treecorr.Catalog(file_name, config)
    np.testing.assert_almost_equal(cat5.x, x * (pi/180.))
    np.testing.assert_almost_equal(cat5.y, y * (pi/180.))

    del config['x_units']  # Default is radians
    del config['y_units']
    cat5 = treecorr.Catalog(file_name, config)
    np.testing.assert_almost_equal(cat5.x, x)
    np.testing.assert_almost_equal(cat5.y, y)

    # Check ra,dec
    del config['x_col']
    del config['y_col']
    config['ra_col'] = 1
    config['dec_col'] = 2
    config['r_col'] = 10
    config['ra_units'] = 'rad'
    config['dec_units'] = 'rad'
    cat6 = treecorr.Catalog(file_name, config)
    np.testing.assert_almost_equal(cat6.ra, ra)
    np.testing.assert_almost_equal(cat6.dec, dec)

    config['ra_units'] = 'deg'
    config['dec_units'] = 'deg'
    cat6 = treecorr.Catalog(file_name, config)
    np.testing.assert_almost_equal(cat6.ra, ra * (pi/180.))
    np.testing.assert_almost_equal(cat6.dec, dec * (pi/180.))

    config['ra_units'] = 'hour'
    config['dec_units'] = 'deg'
    cat6 = treecorr.Catalog(file_name, config)
    np.testing.assert_almost_equal(cat6.ra, ra * (pi/12.))
    np.testing.assert_almost_equal(cat6.dec, dec * (pi/180.))

    del config_names['x_col']
    del config_names['y_col']
    del config_names['z_col']
    del config_names['x_units']
    del config_names['y_units']
    config_names['ra_col'] = 'ra'
    config_names['dec_col'] = 'dec'
    config_names['r_col'] = 'r'
    config_names['ra_units'] = 'hour'
    config_names['dec_units'] = 'deg'
    cat6b = treecorr.Catalog(file_name, config_names)
    np.testing.assert_almost_equal(cat6b.ra, ra * (pi/12.))
    np.testing.assert_almost_equal(cat6b.dec, dec * (pi/180.))

    assert_raises(ValueError, treecorr.Catalog, file_name, config, ra_col=-1)
    assert_raises(ValueError, treecorr.Catalog, file_name, config, ra_col=100)
    assert_raises(ValueError, treecorr.Catalog, file_name, config, ra_col='invalid')
    assert_raises(ValueError, treecorr.Catalog, file_name, config, dec_col=-1)
    assert_raises(ValueError, treecorr.Catalog, file_name, config, dec_col=100)
    assert_raises(ValueError, treecorr.Catalog, file_name, config, dec_col='invalid')
    assert_raises(ValueError, treecorr.Catalog, file_name, config, r_col=-1)
    assert_raises(ValueError, treecorr.Catalog, file_name, config, r_col=100)
    assert_raises(ValueError, treecorr.Catalog, file_name, config, r_col='invalid')
    assert_raises(ValueError, treecorr.Catalog, file_name, config, x_col=4)
    assert_raises(ValueError, treecorr.Catalog, file_name, config, y_col=4)
    assert_raises(ValueError, treecorr.Catalog, file_name, config, z_col=4)

    # Check using a different delimiter, comment marker
    csv_file_name = os.path.join('data','test.csv')
    with open(csv_file_name, 'w') as fid:
        # These are intentionally in a different order from the order we parse them.
        fid.write('% This file uses commas for its delimiter')
        fid.write('% And more than one header line.')
        fid.write('% Plus some extra comment lines every so often.')
        fid.write('% And we use a weird comment marker to boot.')
        fid.write('% ra,dec,x,y,k,g1,g2,w,flag\n')
        for i in range(nobj):
            fid.write((('%.8f,'*11)+'%d\n')%(
                ra[i],dec[i],x[i],y[i],k[i],g1[i],g2[i],w[i],z[i],r[i],wpos[i],flags[i]))
            if i%100 == 0:
                fid.write('%%%% Line %d\n'%i)
    config['delimiter'] = ','
    config['comment_marker'] = '%'
    cat7 = treecorr.Catalog(csv_file_name, config)
    np.testing.assert_almost_equal(cat7.ra, ra * (pi/12.))
    np.testing.assert_almost_equal(cat7.dec, dec * (pi/180.))
    np.testing.assert_almost_equal(cat7.r, r)
    np.testing.assert_almost_equal(cat7.g1, g1)
    np.testing.assert_almost_equal(cat7.g2, g2)
    np.testing.assert_almost_equal(cat7.w[flags < 16], w[flags < 16])
    np.testing.assert_almost_equal(cat7.w[flags >= 16], 0.)

    # Check flip_g1, flip_g2
    del config['delimiter']
    del config['comment_marker']
    config['flip_g1'] = True
    cat8 = treecorr.Catalog(file_name, config)
    np.testing.assert_almost_equal(cat8.g1, -g1)
    np.testing.assert_almost_equal(cat8.g2, g2)

    config['flip_g2'] = 'true'
    cat8 = treecorr.Catalog(file_name, config)
    np.testing.assert_almost_equal(cat8.g1, -g1)
    np.testing.assert_almost_equal(cat8.g2, -g2)

    config['flip_g1'] = 'n'
    config['flip_g2'] = 'yes'
    cat8 = treecorr.Catalog(file_name, config)
    np.testing.assert_almost_equal(cat8.g1, g1)
    np.testing.assert_almost_equal(cat8.g2, -g2)

    # Check overriding values with kwargs
    cat8 = treecorr.Catalog(file_name, config, flip_g1=True, flip_g2=False)
    np.testing.assert_almost_equal(cat8.g1, -g1)
    np.testing.assert_almost_equal(cat8.g2, g2)

    # Check copy command
    cat9 = cat8.copy()
    np.testing.assert_almost_equal(cat9.ra, cat8.ra)
    np.testing.assert_almost_equal(cat9.dec, cat8.dec)
    np.testing.assert_almost_equal(cat9.r, cat8.r)
    np.testing.assert_almost_equal(cat9.g1, cat8.g1)
    np.testing.assert_almost_equal(cat9.g2, cat8.g2)
    np.testing.assert_almost_equal(cat9.w, cat8.w)

    # Swapping w and wpos leads to zeros being copied from wpos to w
    cat10 = treecorr.Catalog(file_name, config, w_col=11, wpos_col=8, flag_col=0)
    np.testing.assert_almost_equal(cat10.wpos, w)
    np.testing.assert_almost_equal(cat10.w, w)

    # And if there is wpos, but no w, copy over the zeros, but not the other values
    with CaptureLog() as cl:
        cat10 = treecorr.Catalog(file_name, config, w_col=0, wpos_col=11, flag_col=0,
                                 logger=cl.logger)
        cat10.x  # Force the read to happen.
    np.testing.assert_almost_equal(cat10.wpos, wpos)
    np.testing.assert_almost_equal(cat10.w[wpos==0], 0)
    np.testing.assert_almost_equal(cat10.w[wpos!=0], 1)
    assert 'Some wpos values are zero, setting w=0 for these points' in cl.output

    # Test using a limited set of rows
    del config['flip_g1']
    del config['flip_g2']
    config['first_row'] = 1010
    config['last_row'] = 3456
    cat11 = treecorr.Catalog(file_name, config)
    np.testing.assert_equal(len(cat11.ra), 2447)
    np.testing.assert_equal(cat11.ntot, 2447)
    np.testing.assert_equal(cat11.nobj, np.sum(cat11.w != 0))
    np.testing.assert_equal(cat11.sumw, np.sum(cat11.w))
    np.testing.assert_equal(cat11.sumw, np.sum(cat6.w[1009:3456]))
    np.testing.assert_almost_equal(cat11.g1[1111], g1[2120])
    np.testing.assert_almost_equal(cat11.g2[1111], g2[2120])
    np.testing.assert_almost_equal(cat11.k[1111], k[2120])

    config['file_name'] = file_name
    cat12 = treecorr.read_catalogs(config, key='file_name', is_rand=True)[0]
    np.testing.assert_equal(len(cat12.x), 2447)
    np.testing.assert_equal(cat12.ntot, 2447)
    np.testing.assert_equal(cat12.nobj, np.sum(cat12.w != 0))
    np.testing.assert_equal(cat12.sumw, np.sum(cat12.w))
    np.testing.assert_equal(cat12.sumw, np.sum(cat6.w[1009:3456]))
    assert cat12.g1 is None
    assert cat12.g2 is None
    assert cat12.k is None

    # Check every_nth
    config['every_nth'] = 10
    cat13 = treecorr.Catalog(file_name, config)
    np.testing.assert_equal(len(cat13.x), 245)
    np.testing.assert_equal(cat13.ntot, 245)
    np.testing.assert_equal(cat13.nobj, np.sum(cat13.w != 0))
    np.testing.assert_equal(cat13.sumw, np.sum(cat13.w))
    print('first few = ',cat13.w[:3])
    print('from cat6: ',cat6.w[1009:1039])
    print('last few = ',cat13.w[-3:])
    print('from cat6: ',cat6.w[3426:3456])
    print('cat13.w = ',cat13.w)
    print('len = ',len(cat13.w))
    print('cat6.w[1009:3456:10] = ',cat6.w[1009:3456:10])
    print('len = ',len(cat6.w[1009:3456:10]))
    print('cat6.w[3449] = ',cat6.w[3449])
    print('cat6.w[3456] = ',cat6.w[3456])
    print('cat6.w[3459] = ',cat6.w[3459])
    np.testing.assert_equal(cat13.sumw, np.sum(cat6.w[1009:3456:10]))
    np.testing.assert_almost_equal(cat13.g1[100], g1[2009])
    np.testing.assert_almost_equal(cat13.g2[100], g2[2009])
    np.testing.assert_almost_equal(cat13.k[100], k[2009])

    # Check every_nth with no first/last
    del config['first_row']
    del config['last_row']
    cat13a = treecorr.Catalog(file_name, config)
    np.testing.assert_equal(len(cat13a.x), 500)
    np.testing.assert_equal(cat13a.ntot, 500)
    np.testing.assert_equal(cat13a.nobj, np.sum(cat13a.w != 0))
    np.testing.assert_equal(cat13a.sumw, np.sum(cat13a.w))
    np.testing.assert_equal(cat13a.sumw, np.sum(cat6.w[::10]))
    np.testing.assert_almost_equal(cat13a.g1[100], g1[1000])
    np.testing.assert_almost_equal(cat13a.g2[100], g2[1000])
    np.testing.assert_almost_equal(cat13a.k[100], k[1000])

    do_pickle(cat1)
    do_pickle(cat2)
    do_pickle(cat3)
    do_pickle(cat4)
    do_pickle(cat5)
    do_pickle(cat6)
    do_pickle(cat7)
    do_pickle(cat8)
    do_pickle(cat9)
    do_pickle(cat10)

    # Check repr.  Usually too long, but cat13 is short enough to eval properly.
    from numpy import array  # noqa: F401
    original = np.get_printoptions()
    np.set_printoptions(precision=20)
    #print('cat13 = ',repr(cat13))
    cat13a = eval(repr(cat13))
    assert cat13a == cat13

    # Check unloaded catalog.  (Recapitulates do_pickle, but careful not to call == too soon.)
    cat14 = treecorr.Catalog(file_name, config)
    cat14a = pickle.loads(pickle.dumps(cat14))
    assert cat14._x is None  # Loading isn't forced by pickling
    assert cat14a._x is None
    cat14b = copy.copy(cat14)
    assert cat14._x is None  # Or copy
    assert cat14b._x is None
    cat14c = copy.deepcopy(cat14)
    assert cat14._x is None  # Or deepcopy
    assert cat14c._x is None
    print('cat14 = ',repr(cat14))
    cat14d = eval(repr(cat14))
    assert cat14._x is None  # Or repr
    assert cat14d._x is None
    # == does finish the load though, so now check that they are all equal.
    assert cat14a == cat14
    assert cat14b == cat14
    assert cat14c == cat14
    assert cat14d == cat14
    assert cat14._x is not None
    assert cat14a._x is not None
    assert cat14b._x is not None
    assert cat14c._x is not None
    assert cat14d._x is not None
    np.set_printoptions(**original)

    # Can also unload the Catalog to recover the memory
    cat14a.unload()
    assert cat14a._x is None  # Unloaded now.
    assert cat14a._y is None
    assert cat14a._z is None
    assert cat14a._ra is None
    assert cat14a._dec is None
    assert cat14a._r is None
    assert cat14a._w is None
    assert cat14a._wpos is None
    assert cat14a._g1 is None
    assert cat14a._g2 is None
    assert cat14a._k is None
    assert cat14a == cat14    # When needed, it will reload, e.g. here to check equality.

@timer
def test_fits():
    _test_aardvark('Aardvark.fit', 'FITS', 'AARDWOLF')

@timer
def test_hdf5():
    try:
        import h5py  # noqa: F401
    except ImportError:
        print('Skipping HdfReader tests, since h5py not installed.')
        return
    _test_aardvark('Aardvark.hdf5', 'HDF', '/')

@timer
def test_parquet():
    try:
        import pandas  # noqa: F401
        import pyarrow # noqa: F401
    except ImportError:
        print('Skipping ParquetReader tests, since pandas or pyarrow not installed.')
        return
    _test_aardvark('Aardvark.parquet', 'Parquet', None)

def _test_aardvark(filename, file_type, ext):
    get_from_wiki(filename)
    file_name = os.path.join('data',filename)
    config = treecorr.read_config('Aardvark.yaml')
    config['verbose'] = 1
    config['kk_file_name'] = 'kk.out'
    config['gg_file_name'] = 'gg.out'
    config['file_name'] = file_name
    config['ext'] = ext

    # Just test a few random particular values
    cat1 = treecorr.Catalog(file_name, config)
    np.testing.assert_equal(len(cat1.ra), 390935)
    np.testing.assert_equal(cat1.nobj, 390935)
    np.testing.assert_almost_equal(cat1.ra[0], 56.4195 * (pi/180.))
    np.testing.assert_almost_equal(cat1.ra[390934], 78.4782 * (pi/180.))
    np.testing.assert_almost_equal(cat1.dec[290333], 83.1579 * (pi/180.))
    np.testing.assert_almost_equal(cat1.g1[46392], 0.0005066675)
    np.testing.assert_almost_equal(cat1.g2[46392], -0.0001006742)
    np.testing.assert_almost_equal(cat1.k[46392], -0.0008628797)

    cat1b = treecorr.Catalog(file_name, config, every_nth=10)
    np.testing.assert_equal(len(cat1b.ra), 39094)
    np.testing.assert_equal(cat1b.nobj, 39094)
    np.testing.assert_almost_equal(cat1.ra[0], 56.4195 * (pi/180.))
    np.testing.assert_almost_equal(cat1.ra[39094], 60.9119 * (pi/180.))

    assert_raises(ValueError, treecorr.Catalog, file_name, config, ra_col='invalid')
    assert_raises(ValueError, treecorr.Catalog, file_name, config, dec_col='invalid')
    assert_raises(ValueError, treecorr.Catalog, file_name, config, r_col='invalid')
    assert_raises(ValueError, treecorr.Catalog, file_name, config, w_col='invalid')
    assert_raises(ValueError, treecorr.Catalog, file_name, config, wpos_col='invalid')
    assert_raises(ValueError, treecorr.Catalog, file_name, config, flag_col='invalid')
    assert_raises(ValueError, treecorr.Catalog, file_name, config, g1_col='invalid')
    assert_raises(ValueError, treecorr.Catalog, file_name, config, g2_col='invalid')
    assert_raises(ValueError, treecorr.Catalog, file_name, config, k_col='invalid')
    assert_raises(ValueError, treecorr.Catalog, file_name, config, ra_col='0')
    assert_raises(ValueError, treecorr.Catalog, file_name, config, dec_col='0')
    assert_raises(ValueError, treecorr.Catalog, file_name, config, x_col='x')
    assert_raises(ValueError, treecorr.Catalog, file_name, config, y_col='y')
    assert_raises(ValueError, treecorr.Catalog, file_name, config, z_col='z')
    assert_raises(ValueError, treecorr.Catalog, file_name, config, ra_col='0', dec_col='0')
    assert_raises(ValueError, treecorr.Catalog, file_name, config, g1_col='0')
    assert_raises(ValueError, treecorr.Catalog, file_name, config, g2_col='0')
    assert_raises(ValueError, treecorr.Catalog, file_name, config, k_col='0')
    assert_raises(TypeError, treecorr.Catalog, file_name, config, x_units='arcmin')
    assert_raises(TypeError, treecorr.Catalog, file_name, config, y_units='arcmin')
    del config['ra_units']
    assert_raises(TypeError, treecorr.Catalog, file_name, config)
    del config['dec_units']
    assert_raises(TypeError, treecorr.Catalog, file_name, config, ra_units='deg')

    # The catalog doesn't have x, y, or w, but test that functionality as well.
    del config['ra_col']
    del config['dec_col']
    config['x_col'] = 'RA'
    config['y_col'] = 'DEC'
    config['w_col'] = 'MU'
    config['flag_col'] = 'INDEX'
    config['ignore_flag'] = 64
    config['keep_zero_weight'] = True
    cat2 = treecorr.Catalog(file_name, config)
    np.testing.assert_almost_equal(cat2.x[390934], 78.4782, decimal=4)
    np.testing.assert_almost_equal(cat2.y[290333], 83.1579, decimal=4)
    np.testing.assert_almost_equal(cat2.w[46392], 0.)        # index = 1200379
    np.testing.assert_almost_equal(cat2.w[46393], 0.9995946) # index = 1200386

    assert_raises(ValueError, treecorr.Catalog, file_name, config, x_col='invalid')
    assert_raises(ValueError, treecorr.Catalog, file_name, config, y_col='invalid')
    assert_raises(ValueError, treecorr.Catalog, file_name, config, z_col='invalid')
    assert_raises(ValueError, treecorr.Catalog, file_name, config, ra_col='invalid')
    assert_raises(ValueError, treecorr.Catalog, file_name, config, dec_col='invalid')
    assert_raises(ValueError, treecorr.Catalog, file_name, config, r_col='invalid')
    assert_raises(ValueError, treecorr.Catalog, file_name, config, w_col='invalid')
    assert_raises(ValueError, treecorr.Catalog, file_name, config, wpos_col='invalid')
    assert_raises(ValueError, treecorr.Catalog, file_name, config, flag_col='invalid')
    assert_raises(ValueError, treecorr.Catalog, file_name, config, g1_col='invalid')
    assert_raises(ValueError, treecorr.Catalog, file_name, config, g2_col='invalid')
    assert_raises(ValueError, treecorr.Catalog, file_name, config, k_col='invalid')

    # Test using a limited set of rows
    # Also explicit file_type
    config['first_row'] = 101
    config['last_row'] = 50000
    config['file_type'] = file_type
    cat3 = treecorr.Catalog(file_name, config)
    np.testing.assert_equal(len(cat3.x), 49900)
    np.testing.assert_equal(cat3.ntot, 49900)
    np.testing.assert_equal(cat3.nobj, np.sum(cat3.w != 0))
    np.testing.assert_equal(cat3.sumw, np.sum(cat3.w))
    np.testing.assert_equal(cat3.sumw, np.sum(cat2.w[100:50000]))
    np.testing.assert_almost_equal(cat3.g1[46292], cat2.g1[46392])
    np.testing.assert_almost_equal(cat3.g2[46292], cat2.g2[46392])
    np.testing.assert_almost_equal(cat3.k[46292], cat2.k[46392])

    cat4 = treecorr.read_catalogs(config, key='file_name', is_rand=True)[0]
    np.testing.assert_equal(len(cat4.x), 49900)
    np.testing.assert_equal(cat4.ntot, 49900)
    np.testing.assert_equal(cat4.nobj, np.sum(cat4.w != 0))
    np.testing.assert_equal(cat4.sumw, np.sum(cat4.w))
    np.testing.assert_equal(cat4.sumw, np.sum(cat2.w[100:50000]))
    assert cat4.g1 is None
    assert cat4.g2 is None
    assert cat4.k is None

    # Check all rows with every_nth first.  This used to not work right due to a bug in fitsio.
    # cf. https://github.com/esheldon/fitsio/pull/286
    config['every_nth'] = 100
    del config['first_row']
    del config['last_row']
    cat5a = treecorr.Catalog(file_name, config)
    np.testing.assert_equal(len(cat5a.x), 3910)
    np.testing.assert_equal(cat5a.ntot, 3910)
    np.testing.assert_equal(cat5a.nobj, np.sum(cat5a.w != 0))
    np.testing.assert_equal(cat5a.sumw, np.sum(cat5a.w))
    np.testing.assert_equal(cat5a.sumw, np.sum(cat2.w[::100]))
    np.testing.assert_almost_equal(cat5a.g1[123], cat2.g1[12300])
    np.testing.assert_almost_equal(cat5a.g2[123], cat2.g2[12300])
    np.testing.assert_almost_equal(cat5a.k[123], cat2.k[12300])

    # Now with first, last, and every_nth
    config['first_row'] = 101
    config['last_row'] = 50000
    cat5 = treecorr.Catalog(file_name, config)
    np.testing.assert_equal(len(cat5.x), 499)
    np.testing.assert_equal(cat5.ntot, 499)
    np.testing.assert_equal(cat5.nobj, np.sum(cat5.w != 0))
    np.testing.assert_equal(cat5.sumw, np.sum(cat5.w))
    np.testing.assert_equal(cat5.sumw, np.sum(cat2.w[100:50000:100]))
    np.testing.assert_almost_equal(cat5.g1[123], cat2.g1[12400])
    np.testing.assert_almost_equal(cat5.g2[123], cat2.g2[12400])
    np.testing.assert_almost_equal(cat5.k[123], cat2.k[12400])

    do_pickle(cat1)
    do_pickle(cat2)
    do_pickle(cat3)
    do_pickle(cat4)

    # Check repr.  Usually too long, but cat13 is short enough to eval properly.
    from numpy import array  # noqa: F401
    original = np.get_printoptions()
    np.set_printoptions(precision=20)
    #print('cat5 = ',repr(cat5))
    cat5a = eval(repr(cat5))
    assert cat5a == cat5

    # Check unloaded catalog.  (Recapitulates do_pickle, but careful not to call == too soon.)
    cat6 = treecorr.Catalog(file_name, config)
    cat6a = pickle.loads(pickle.dumps(cat6))
    assert cat6._x is None  # Loading isn't forced by pickling
    assert cat6a._x is None
    cat6b = copy.copy(cat6)
    assert cat6._x is None  # Or copy
    assert cat6b._x is None
    cat6c = copy.deepcopy(cat6)
    assert cat6._x is None  # Or deepcopy
    assert cat6c._x is None
    print('cat6 = ',repr(cat6))
    cat6d = eval(repr(cat6))
    assert cat6._x is None  # Or repr
    assert cat6d._x is None
    # == does finish the load though, so now check that they are all equal.
    assert cat6a == cat6
    assert cat6b == cat6
    assert cat6c == cat6
    assert cat6d == cat6
    assert cat6._x is not None
    assert cat6a._x is not None
    assert cat6b._x is not None
    assert cat6c._x is not None
    assert cat6d._x is not None
    np.set_printoptions(**original)

    # Can also unload the Catalog to recover the memory
    cat6a.unload()
    assert cat6a._x is None  # Unloaded now.
    assert cat6a._y is None
    assert cat6a._z is None
    assert cat6a._ra is None
    assert cat6a._dec is None
    assert cat6a._r is None
    assert cat6a._w is None
    assert cat6a._wpos is None
    assert cat6a._g1 is None
    assert cat6a._g2 is None
    assert cat6a._k is None
    assert cat6a == cat6    # When needed, it will reload, e.g. here to check equality.

    assert_raises(ValueError, treecorr.Catalog, file_name, config, first_row=-10)
    assert_raises(ValueError, treecorr.Catalog, file_name, config, first_row=0)
    assert_raises(ValueError, treecorr.Catalog, file_name, config, first_row=60000)
    assert_raises(ValueError, treecorr.Catalog, file_name, config, first_row=50001)

    assert_raises(TypeError, treecorr.read_catalogs, config)
    assert_raises(TypeError, treecorr.read_catalogs, config, key='file_name', list_key='file_name')

    # If gg output not given, it is still invalid to only have one or the other of g1,g2.
    del config['gg_file_name']
    assert_raises(ValueError, treecorr.Catalog, file_name, config, g1_col='0')
    assert_raises(ValueError, treecorr.Catalog, file_name, config, g2_col='0')

    assert_raises(ValueError, treecorr.Catalog, file_name, config, every_nth=0)
    assert_raises(ValueError, treecorr.Catalog, file_name, config, every_nth=-10)

@timer
def test_ext():

    ngal = 200
    s = 10.
    rng = np.random.RandomState(8675309)
    x = rng.normal(0,s, (ngal,) )
    y = rng.normal(0,s, (ngal,) )
    z = rng.normal(0,s, (ngal,) )
    ra, dec, r = coord.CelestialCoord.xyz_to_radec(x,y,z,return_r=True)
    wpos = rng.random_sample(ngal)
    w = wpos * rng.binomial(1, 0.95, (ngal,))
    flag = rng.binomial(3, 0.02, (ngal,))
    k = rng.normal(0,3, (ngal,) )
    g1 = rng.normal(0,0.1, (ngal,) )
    g2 = rng.normal(0,0.1, (ngal,) )
    patch = np.arange(ngal) % 5

    data = [x,y,z,ra,dec,r,w,wpos,flag,k,g1,g2]
    names = ['x','y','z','ra','dec','r','w','wpos','flag','k','g1','g2']

    fname = os.path.join('data','test_ext.fits')
    with fitsio.FITS(fname, 'rw', clobber=True) as f:
        f.write(data,names=names,extname='1')
        f.write(data,names=names,extname='2')
        f.write(data[:3],names=names[:3],extname='xyz')
        f.write(data[3:6],names=names[3:6],extname='radec')
        f.write(data[:3]+data[6:9],names=names[:3]+names[6:9],extname='w')
        f.write(data[:3]+data[9:],names=names[:3]+names[9:],extname='kg')
        f.write([patch],names=['patch'],extname='patch')
        f.write(data,names=names[::-1],extname='reverse')

    cat1 = treecorr.Catalog(fname, allow_xyz=True,
                            x_col='x', y_col='y', z_col='z',
                            ra_col='ra', dec_col='dec', r_col='r',
                            ra_units='rad', dec_units='rad',
                            w_col='w', wpos_col='wpos', flag_col='flag',
                            k_col='k', g1_col='g1', g2_col='g2',
                            ext=1)
    cat2 = treecorr.Catalog(fname, allow_xyz=True,
                            x_col='x', y_col='y', z_col='z',
                            ra_col='ra', dec_col='dec', r_col='r',
                            ra_units='rad', dec_units='rad',
                            w_col='w', wpos_col='wpos', flag_col='flag',
                            k_col='k', g1_col='g1', g2_col='g2',
                            ext=2)
    assert cat2 == cat1

    cat3 = treecorr.Catalog(fname,
                            x_col='x', y_col='y', z_col='z',
                            ext=3)
    use = np.where(flag == 0)
    np.testing.assert_array_equal(cat3.x[use], cat1.x)
    np.testing.assert_array_equal(cat3.y[use], cat1.y)
    np.testing.assert_array_equal(cat3.z[use], cat1.z)

    cat4 = treecorr.Catalog(fname,
                            ra_col='ra', dec_col='dec', r_col='r',
                            ra_units='rad', dec_units='rad',
                            ext=4)
    print('cat4.ra = ',cat4.ra[use])
    print('cat1.ra = ',cat1.ra)
    np.testing.assert_allclose(cat4.ra[use], cat1.ra)  # roundtrip doesn't have to be exact.
    np.testing.assert_allclose(cat4.dec[use], cat1.dec)
    np.testing.assert_allclose(cat4.r[use], cat1.r)
    np.testing.assert_allclose(cat4.x[use], cat1.x)
    np.testing.assert_allclose(cat4.y[use], cat1.y)
    np.testing.assert_allclose(cat4.z[use], cat1.z)

    cat5 = treecorr.Catalog(fname,
                            x_col='x', y_col='y', z_col='z',
                            w_col='w', wpos_col='wpos', flag_col='flag',
                            ext=5)
    np.testing.assert_array_equal(cat5.x, cat1.x)
    np.testing.assert_array_equal(cat5.y, cat1.y)
    np.testing.assert_array_equal(cat5.z, cat1.z)
    np.testing.assert_array_equal(cat5.w, cat1.w)
    np.testing.assert_array_equal(cat5.wpos, cat1.wpos)

    cat6 = treecorr.Catalog(fname,
                            x_col='x', y_col='y', z_col='z',
                            k_col='k', g1_col='g1', g2_col='g2',
                            ext=6)
    np.testing.assert_array_equal(cat6.x[use], cat1.x)
    np.testing.assert_array_equal(cat6.y[use], cat1.y)
    np.testing.assert_array_equal(cat6.z[use], cat1.z)
    np.testing.assert_array_equal(cat6.k[use], cat1.k)
    np.testing.assert_array_equal(cat6.g1[use], cat1.g1)
    np.testing.assert_array_equal(cat6.g2[use], cat1.g2)

    cat7 = treecorr.Catalog(fname, allow_xyz=True,
                            x_col='x', y_col='y', z_col='z',
                            ra_col='ra', dec_col='dec', r_col='r',
                            ra_units='rad', dec_units='rad',
                            w_col='w', wpos_col='wpos', flag_col='flag',
                            k_col='k', g1_col='g1', g2_col='g2',
                            ext=8)
    assert cat7 != cat1  # This one has all the column names wrong.

    cat8 = treecorr.Catalog(fname, allow_xyz=True,
                            x_col='x', y_col='y', z_col='z',
                            ra_col='ra', dec_col='dec', r_col='r',
                            ra_units='rad', dec_units='rad',
                            w_col='w', wpos_col='wpos', flag_col='flag',
                            k_col='k', g1_col='g1', g2_col='g2',
                            ext=-1)
    assert cat8 == cat7  # -1 is allowed and means the last one.

    cat9 = treecorr.Catalog(fname, allow_xyz=True,
                            x_col='x', y_col='y', z_col='z',
                            ra_col='ra', dec_col='dec', r_col='r',
                            ra_units='rad', dec_units='rad',
                            w_col='w', wpos_col='wpos', flag_col='flag',
                            k_col='k', g1_col='g1', g2_col='g2',
                            x_ext=1, y_ext=1, z_ext=1,
                            ra_ext=2, dec_ext=1, r_ext=2,
                            w_ext=1, wpos_ext=2, flag_ext=1,
                            k_ext=1, g1_ext=1, g2_ext=2)
    assert cat9 == cat1

    cat10 = treecorr.Catalog(fname, allow_xyz=True,
                             x_col='x', y_col='y', z_col='z',
                             ra_col='ra', dec_col='dec', r_col='r',
                             ra_units='rad', dec_units='rad',
                             w_col='w', wpos_col='wpos', flag_col='flag',
                             k_col='k', g1_col='g1', g2_col='g2',
                             x_ext=3, y_ext=3, z_ext=3,
                             ra_ext=4, dec_ext=4, r_ext=4,
                             w_ext=5, wpos_ext=5, flag_ext=5,
                             k_ext=6, g1_ext=6, g2_ext=6)
    assert cat10 == cat1

    # Not all columns in given ext
    with assert_raises(ValueError):
        treecorr.Catalog(fname, allow_xyz=True,
                         x_col='x', y_col='y', z_col='z',
                         ra_col='ra', dec_col='dec', r_col='r',
                         ra_units='rad', dec_units='rad',
                         w_col='w', wpos_col='wpos', flag_col='flag',
                         k_col='k', g1_col='g1', g2_col='g2',
                         ext=3)

    # Invalid ext
    with assert_raises(ValueError):
        treecorr.Catalog(fname, allow_xyz=True,
                         x_col='x', y_col='y', z_col='z',
                         ra_col='ra', dec_col='dec', r_col='r',
                         ra_units='rad', dec_units='rad',
                         w_col='w', wpos_col='wpos', flag_col='flag',
                         k_col='k', g1_col='g1', g2_col='g2',
                         ext=9)
    with assert_raises(ValueError):
        treecorr.Catalog(fname, allow_xyz=True,
                         x_col='x', y_col='y', z_col='z',
                         ra_col='ra', dec_col='dec', r_col='r',
                         ra_units='rad', dec_units='rad',
                         w_col='w', wpos_col='wpos', flag_col='flag',
                         k_col='k', g1_col='g1', g2_col='g2',
                         ext=0)
    with assert_raises(ValueError):
        treecorr.Catalog(fname, allow_xyz=True,
                         x_col='x', y_col='y', z_col='z',
                         ra_col='ra', dec_col='dec', r_col='r',
                         ra_units='rad', dec_units='rad',
                         w_col='w', wpos_col='wpos', flag_col='flag',
                         k_col='k', g1_col='g1', g2_col='g2',
                         ext=-20)

    # Not all columns in given ext
    with assert_raises(ValueError):
        treecorr.Catalog(fname,
                         ra_col='ra', dec_col='dec',
                         ra_units='rad', dec_units='rad',
                         k_col='k', g1_col='g1', g2_col='g2',
                         ext=6)

    # Position columns required
    with assert_raises(ValueError):
        treecorr.Catalog(fname,
                         k_col='k', g1_col='g1', g2_col='g2',
                         ext=6)

    # Missing units
    with assert_raises(TypeError):
        treecorr.Catalog(fname, allow_xyz=True,
                         x_col='x', y_col='y', z_col='z',
                         ra_col='ra', dec_col='dec', r_col='r',
                         w_col='w', wpos_col='wpos', flag_col='flag',
                         k_col='k', g1_col='g1', g2_col='g2',
                         ext=1)
    with assert_raises(TypeError):
        treecorr.Catalog(fname, allow_xyz=True,
                         x_col='x', y_col='y', z_col='z',
                         ra_col='ra', dec_col='dec', r_col='r',
                         ra_units='rad',
                         w_col='w', wpos_col='wpos', flag_col='flag',
                         k_col='k', g1_col='g1', g2_col='g2',
                         ext=1)
    with assert_raises(TypeError):
        treecorr.Catalog(fname, allow_xyz=True,
                         x_col='x', y_col='y', z_col='z',
                         ra_col='ra', dec_col='dec', r_col='r',
                         dec_units='rad',
                         w_col='w', wpos_col='wpos', flag_col='flag',
                         k_col='k', g1_col='g1', g2_col='g2',
                         ext=1)

    if sys.version_info < (3,): return  # mock only available on python 3
    from unittest import mock

    # test that the case where we can't slice works
    # by pretending that we are using an old fitsio version,
    # temporarily.
    with mock.patch('fitsio.__version__', '1.0.6'):
        cat11 = treecorr.Catalog(fname, allow_xyz=True,
                                 x_col='x', y_col='y', z_col='z',
                                 ra_col='ra', dec_col='dec', r_col='r',
                                 ra_units='rad', dec_units='rad',
                                 w_col='w', wpos_col='wpos', flag_col='flag',
                                 k_col='k', g1_col='g1', g2_col='g2',
                                 x_ext=3, y_ext=3, z_ext=3,
                                 ra_ext=4, dec_ext=4, r_ext=4,
                                 w_ext=5, wpos_ext=5, flag_ext=5,
                                 k_ext=6, g1_ext=6, g2_ext=6)
        assert cat11 == cat1
        # and equiv for RA
        cat12 = treecorr.Catalog(fname,
                                ra_col='ra', dec_col='dec', r_col='r',
                                ra_units='rad', dec_units='rad',
                                ext=4, ra_ext=4, last_row=120)
        cat13 = treecorr.Catalog(fname,
                                ra_col='ra', dec_col='dec', r_col='r',
                                ra_units='rad', dec_units='rad',
                                ext=4, ra_ext=4, last_row=100)

        np.testing.assert_allclose(cat12.ra[:99], cat13.ra[:99])
        # and equiv for RA
        cat14 = treecorr.Catalog(fname,
                                x_col='ra', y_col='dec',
                                x_units='rad', y_units='rad',
                                ext=4, ra_ext=4, last_row=120)
        cat15 = treecorr.Catalog(fname,
                                x_col='ra', y_col='dec',
                                x_units='rad', y_units='rad',
                                ext=4, x_ext=4, last_row=100)

        np.testing.assert_allclose(cat14.x[:99], cat15.x[:99])

        cat16 = treecorr.Catalog(fname,
                                ra_col='ra', dec_col='dec',
                                ra_units='rad', dec_units='rad',
                                ext=4,  last_row=100,
                                patch_col='patch', patch_ext='patch',
                                patch=0)

        np.testing.assert_allclose(cat16.ra, cat4.ra[:100][::5])


@timer
def test_hdu():
    # Repeat test_ext, but with deprecated hdu names.

    ngal = 200
    s = 10.
    rng = np.random.RandomState(8675309)
    x = rng.normal(0,s, (ngal,) )
    y = rng.normal(0,s, (ngal,) )
    z = rng.normal(0,s, (ngal,) )
    ra, dec, r = coord.CelestialCoord.xyz_to_radec(x,y,z,return_r=True)
    wpos = rng.random_sample(ngal)
    w = wpos * rng.binomial(1, 0.95, (ngal,))
    flag = rng.binomial(3, 0.02, (ngal,))
    k = rng.normal(0,3, (ngal,) )
    g1 = rng.normal(0,0.1, (ngal,) )
    g2 = rng.normal(0,0.1, (ngal,) )

    data = [x,y,z,ra,dec,r,w,wpos,flag,k,g1,g2]
    names = ['x','y','z','ra','dec','r','w','wpos','flag','k','g1','g2']

    fname = os.path.join('data','test_hdu.fits')
    with fitsio.FITS(fname, 'rw', clobber=True) as f:
        f.write(data,names=names,extname='1')
        f.write(data,names=names,extname='2')
        f.write(data[:3],names=names[:3],extname='xyz')
        f.write(data[3:6],names=names[3:6],extname='radec')
        f.write(data[:3]+data[6:9],names=names[:3]+names[6:9],extname='w')
        f.write(data[:3]+data[9:],names=names[:3]+names[9:],extname='kg')
        f.write(data,names=names[::-1],extname='reverse')

    with assert_warns(FutureWarning):
        cat1 = treecorr.Catalog(fname, allow_xyz=True,
                                x_col='x', y_col='y', z_col='z',
                                ra_col='ra', dec_col='dec', r_col='r',
                                ra_units='rad', dec_units='rad',
                                w_col='w', wpos_col='wpos', flag_col='flag',
                                k_col='k', g1_col='g1', g2_col='g2',
                               hdu=1)
    with assert_warns(FutureWarning):
        cat2 = treecorr.Catalog(fname, allow_xyz=True,
                                x_col='x', y_col='y', z_col='z',
                                ra_col='ra', dec_col='dec', r_col='r',
                                ra_units='rad', dec_units='rad',
                                w_col='w', wpos_col='wpos', flag_col='flag',
                                k_col='k', g1_col='g1', g2_col='g2',
                                hdu=2)
    assert cat2 == cat1

    with assert_warns(FutureWarning):
        cat3 = treecorr.Catalog(fname,
                                x_col='x', y_col='y', z_col='z',
                                hdu=3)
    use = np.where(flag == 0)
    np.testing.assert_array_equal(cat3.x[use], cat1.x)
    np.testing.assert_array_equal(cat3.y[use], cat1.y)
    np.testing.assert_array_equal(cat3.z[use], cat1.z)

    with assert_warns(FutureWarning):
        cat4 = treecorr.Catalog(fname,
                                ra_col='ra', dec_col='dec', r_col='r',
                                ra_units='rad', dec_units='rad',
                                hdu=4)
    print('cat4.ra = ',cat4.ra[use])
    print('cat1.ra = ',cat1.ra)
    np.testing.assert_allclose(cat4.ra[use], cat1.ra)  # roundtrip doesn't have to be exact.
    np.testing.assert_allclose(cat4.dec[use], cat1.dec)
    np.testing.assert_allclose(cat4.r[use], cat1.r)
    np.testing.assert_allclose(cat4.x[use], cat1.x)
    np.testing.assert_allclose(cat4.y[use], cat1.y)
    np.testing.assert_allclose(cat4.z[use], cat1.z)

    with assert_warns(FutureWarning):
        cat5 = treecorr.Catalog(fname,
                                x_col='x', y_col='y', z_col='z',
                                w_col='w', wpos_col='wpos', flag_col='flag',
                                hdu=5)
    np.testing.assert_array_equal(cat5.x, cat1.x)
    np.testing.assert_array_equal(cat5.y, cat1.y)
    np.testing.assert_array_equal(cat5.z, cat1.z)
    np.testing.assert_array_equal(cat5.w, cat1.w)
    np.testing.assert_array_equal(cat5.wpos, cat1.wpos)

    with assert_warns(FutureWarning):
        cat6 = treecorr.Catalog(fname,
                                x_col='x', y_col='y', z_col='z',
                                k_col='k', g1_col='g1', g2_col='g2',
                                hdu=6)
    np.testing.assert_array_equal(cat6.x[use], cat1.x)
    np.testing.assert_array_equal(cat6.y[use], cat1.y)
    np.testing.assert_array_equal(cat6.z[use], cat1.z)
    np.testing.assert_array_equal(cat6.k[use], cat1.k)
    np.testing.assert_array_equal(cat6.g1[use], cat1.g1)
    np.testing.assert_array_equal(cat6.g2[use], cat1.g2)

    with assert_warns(FutureWarning):
        cat7 = treecorr.Catalog(fname, allow_xyz=True,
                                x_col='x', y_col='y', z_col='z',
                                ra_col='ra', dec_col='dec', r_col='r',
                                ra_units='rad', dec_units='rad',
                                w_col='w', wpos_col='wpos', flag_col='flag',
                                k_col='k', g1_col='g1', g2_col='g2',
                                hdu=7)
    assert cat7 != cat1  # This one has all the column names wrong.

    with assert_warns(FutureWarning):
        cat8 = treecorr.Catalog(fname, allow_xyz=True,
                                x_col='x', y_col='y', z_col='z',
                                ra_col='ra', dec_col='dec', r_col='r',
                                ra_units='rad', dec_units='rad',
                                w_col='w', wpos_col='wpos', flag_col='flag',
                                k_col='k', g1_col='g1', g2_col='g2',
                                hdu=-1)
    assert cat8 == cat7  # -1 is allowed and means the last one.

    with assert_warns(FutureWarning):
        cat9 = treecorr.Catalog(fname, allow_xyz=True,
                                x_col='x', y_col='y', z_col='z',
                                ra_col='ra', dec_col='dec', r_col='r',
                                ra_units='rad', dec_units='rad',
                                w_col='w', wpos_col='wpos', flag_col='flag',
                                k_col='k', g1_col='g1', g2_col='g2',
                                x_hdu=1, y_hdu=1, z_hdu=1,
                                ra_hdu=2, dec_hdu=1, r_hdu=2,
                                w_hdu=1, wpos_hdu=2, flag_hdu=1,
                                k_hdu=1, g1_hdu=1, g2_hdu=2)
    assert cat9 == cat1

    with assert_warns(FutureWarning):
        cat10 = treecorr.Catalog(fname, allow_xyz=True,
                                 x_col='x', y_col='y', z_col='z',
                                 ra_col='ra', dec_col='dec', r_col='r',
                                 ra_units='rad', dec_units='rad',
                                 w_col='w', wpos_col='wpos', flag_col='flag',
                                 k_col='k', g1_col='g1', g2_col='g2',
                                 x_hdu=3, y_hdu=3, z_hdu=3,
                                 ra_hdu=4, dec_hdu=4, r_hdu=4,
                                 w_hdu=5, wpos_hdu=5, flag_hdu=5,
                                 k_hdu=6, g1_hdu=6, g2_hdu=6)
    assert cat10 == cat1


@timer
def test_direct():

    nobj = 5000
    rng = np.random.RandomState(8675309)
    x = rng.random_sample(nobj)
    y = rng.random_sample(nobj)
    ra = rng.random_sample(nobj)
    dec = rng.random_sample(nobj)
    w = rng.random_sample(nobj)
    g1 = rng.random_sample(nobj)
    g2 = rng.random_sample(nobj)
    k = rng.random_sample(nobj)

    cat1 = treecorr.Catalog(x=x, y=y, w=w, g1=g1, g2=g2, k=k)
    np.testing.assert_almost_equal(cat1.x, x)
    np.testing.assert_almost_equal(cat1.y, y)
    np.testing.assert_almost_equal(cat1.w, w)
    np.testing.assert_almost_equal(cat1.g1, g1)
    np.testing.assert_almost_equal(cat1.g2, g2)
    np.testing.assert_almost_equal(cat1.k, k)

    cat2 = treecorr.Catalog(ra=ra, dec=dec, w=w, g1=g1, g2=g2, k=k,
                            ra_units='hours', dec_units='degrees')
    np.testing.assert_almost_equal(cat2.ra, ra * coord.hours / coord.radians)
    np.testing.assert_almost_equal(cat2.dec, dec * coord.degrees / coord.radians)
    np.testing.assert_almost_equal(cat2.w, w)
    np.testing.assert_almost_equal(cat2.g1, g1)
    np.testing.assert_almost_equal(cat2.g2, g2)
    np.testing.assert_almost_equal(cat2.k, k)

    do_pickle(cat1)
    do_pickle(cat2)

    # unload() is valid on direct Catalogs, but doesn't do anything.
    cat2.unload()
    assert cat2._x is not None  # Cannot unload without file

    assert_raises(TypeError, treecorr.Catalog, x=x)
    assert_raises(TypeError, treecorr.Catalog, y=y)
    assert_raises(TypeError, treecorr.Catalog, z=x)
    assert_raises(TypeError, treecorr.Catalog, r=x)
    assert_raises(TypeError, treecorr.Catalog, x=x, y=y, r=x)
    assert_raises(TypeError, treecorr.Catalog, x=x, y=y, ra=ra, dec=dec)
    assert_raises(TypeError, treecorr.Catalog, x=x, y=y, ra=ra, dec=dec,
                  ra_units='hours', dec_units='degrees')
    assert_raises(TypeError, treecorr.Catalog, x=x, y=y, ra_units='hours')
    assert_raises(TypeError, treecorr.Catalog, x=x, y=y, dec_units='degrees')
    assert_raises(TypeError, treecorr.Catalog, ra=ra, ra_units='hours')
    assert_raises(TypeError, treecorr.Catalog, dec=dec, dec_units='degrees')
    assert_raises(TypeError, treecorr.Catalog, x=x, y=y, g1=g1)
    assert_raises(TypeError, treecorr.Catalog, x=x, y=y, g2=g2)
    assert_raises(TypeError, treecorr.Catalog, ra=ra, dec=dec)
    assert_raises(TypeError, treecorr.Catalog, ra=ra, dec=dec, ra_unit='deg')
    assert_raises(TypeError, treecorr.Catalog, ra=ra, dec=dec, dec_unit='deg')
    assert_raises(TypeError, treecorr.Catalog, ra=ra, dec=dec,
                  ra_units='hours', dec_units='degrees', x_units='arcmin')
    assert_raises(TypeError, treecorr.Catalog, ra=ra, dec=dec,
                  ra_units='hours', dec_units='degrees', y_units='arcmin')
    assert_raises(TypeError, treecorr.Catalog, x=x, y=y, x_units='arcmin')
    assert_raises(TypeError, treecorr.Catalog, x=x, y=y, y_units='arcmin')
    assert_raises(TypeError, treecorr.Catalog, x=x, y=y, z=x, z_units='arcmin')

    assert_raises(ValueError, treecorr.Catalog, x=x, y=y[4:])
    assert_raises(ValueError, treecorr.Catalog, x=x, y=y, z=x[4:])
    assert_raises(ValueError, treecorr.Catalog, x=x, y=y, w=w[4:])
    assert_raises(ValueError, treecorr.Catalog, x=x, y=y, w=w, wpos=w[4:])
    assert_raises(ValueError, treecorr.Catalog, x=x, y=y, w=w, g1=g1[4:], g2=g2[4:])
    assert_raises(ValueError, treecorr.Catalog, x=x, y=y, w=w, g1=g1[4:], g2=g2)
    assert_raises(ValueError, treecorr.Catalog, x=x, y=y, w=w, g1=g1, g2=g2[4:])
    assert_raises(ValueError, treecorr.Catalog, x=x, y=y, w=w, k=k[4:])
    assert_raises(ValueError, treecorr.Catalog, ra=ra, dec=dec[4:], w=w, g1=g1, g2=g2, k=k,
                  ra_units='hours', dec_units='degrees')
    assert_raises(ValueError, treecorr.Catalog, ra=ra[4:], dec=dec, w=w, g1=g1, g2=g2, k=k,
                  ra_units='hours', dec_units='degrees')
    assert_raises(ValueError, treecorr.Catalog, ra=ra, dec=dec, r=x[4:], w=w, g1=g1, g2=g2, k=k,
                  ra_units='hours', dec_units='degrees')
    assert_raises(ValueError, treecorr.Catalog, x=[], y=[])
    assert_raises(ValueError, treecorr.Catalog, x=x, y=y, w=np.zeros_like(x))

@timer
def test_var():
    nobj = 5000
    rng = np.random.RandomState(8675309)

    # First without weights
    cats = []
    allg1 = []
    allg2 = []
    allk = []
    for i in range(10):
        x = rng.random_sample(nobj)
        y = rng.random_sample(nobj)
        g1 = rng.random_sample(nobj) - 0.5
        g2 = rng.random_sample(nobj) - 0.5
        k = rng.random_sample(nobj) - 0.5
        cat = treecorr.Catalog(x=x, y=y, g1=g1, g2=g2, k=k)
        varg = (np.sum(g1**2) + np.sum(g2**2)) / (2.*len(g1))
        vark = np.var(k, ddof=0)
        assert np.isclose(cat.vark, vark)
        assert np.isclose(cat.varg, varg)
        assert np.isclose(treecorr.calculateVarK(cat), vark)
        assert np.isclose(treecorr.calculateVarK([cat]), vark)
        assert np.isclose(treecorr.calculateVarG(cat), varg)
        assert np.isclose(treecorr.calculateVarG([cat]), varg)
        cats.append(cat)
        allg1.extend(g1)
        allg2.extend(g2)
        allk.extend(k)

    allg1 = np.array(allg1)
    allg2 = np.array(allg2)
    allk = np.array(allk)
    varg = (np.sum(allg1**2) + np.sum(allg2**2)) / (2. * len(allg1))
    vark = np.sum((allk-np.mean(allk))**2) / len(allk)
    assert np.isclose(treecorr.calculateVarG(cats), varg)
    assert np.isclose(treecorr.calculateVarK(cats), vark)

    # Now with weights
    cats = []
    allg1 = []
    allg2 = []
    allk = []
    allw = []
    for i in range(10):
        x = rng.random_sample(nobj)
        y = rng.random_sample(nobj)
        w = rng.random_sample(nobj)
        g1 = rng.random_sample(nobj)
        g2 = rng.random_sample(nobj)
        k = rng.random_sample(nobj)
        cat = treecorr.Catalog(x=x, y=y, w=w, g1=g1, g2=g2, k=k)
        varg = np.sum(w**2 * (g1**2 + g2**2)) / np.sum(w) / 2.
        meank = np.sum(w*k)/np.sum(w)
        vark = np.sum(w**2 * (k-meank)**2) / np.sum(w)
        assert np.isclose(cat.varg, varg)
        assert np.isclose(cat.vark, vark)
        assert np.isclose(treecorr.calculateVarG(cat), varg)
        assert np.isclose(treecorr.calculateVarG([cat]), varg)
        assert np.isclose(treecorr.calculateVarK(cat), vark)
        assert np.isclose(treecorr.calculateVarK([cat]), vark)
        cats.append(cat)
        allg1.extend(g1)
        allg2.extend(g2)
        allk.extend(k)
        allw.extend(w)

    allg1 = np.array(allg1)
    allg2 = np.array(allg2)
    allk = np.array(allk)
    allw = np.array(allw)
    varg = np.sum(allw**2 * (allg1**2 + allg2**2)) / np.sum(allw) / 2.
    meank = np.sum(allw*allk)/np.sum(allw)
    vark = np.sum(allw**2 * (allk-meank)**2) / np.sum(allw)
    assert np.isclose(treecorr.calculateVarG(cats), varg)
    assert np.isclose(treecorr.calculateVarK(cats), vark)

    # With no g1,g2,k, varg=vark=0
    cat = treecorr.Catalog(x=x, y=y)
    assert cat.varg == 0
    assert cat.vark == 0
    cat = treecorr.Catalog(x=x, y=y, w=w)
    assert cat.varg == 0
    assert cat.vark == 0


@timer
def test_nan():
    # Test handling of Nan values (w -> 0)

    nobj = 5000
    rng = np.random.RandomState(8675309)
    x = rng.random_sample(nobj)
    y = rng.random_sample(nobj)
    z = rng.random_sample(nobj)
    ra = rng.random_sample(nobj)
    dec = rng.random_sample(nobj)
    r = rng.random_sample(nobj)
    w = rng.random_sample(nobj)
    wpos = rng.random_sample(nobj)
    g1 = rng.random_sample(nobj)
    g2 = rng.random_sample(nobj)
    k = rng.random_sample(nobj)

    # Turn 1% of these values into NaN
    x[rng.choice(nobj, nobj//100)] = np.nan
    y[rng.choice(nobj, nobj//100)] = np.nan
    z[rng.choice(nobj, nobj//100)] = np.nan
    ra[rng.choice(nobj, nobj//100)] = np.nan
    dec[rng.choice(nobj, nobj//100)] = np.nan
    r[rng.choice(nobj, nobj//100)] = np.nan
    w[rng.choice(nobj, nobj//100)] = np.nan
    wpos[rng.choice(nobj, nobj//100)] = np.nan
    g1[rng.choice(nobj, nobj//100)] = np.nan
    g2[rng.choice(nobj, nobj//100)] = np.nan
    k[rng.choice(nobj, nobj//100)] = np.nan
    print('x is nan at ',np.where(np.isnan(x)))
    print('y is nan at ',np.where(np.isnan(y)))
    print('z is nan at ',np.where(np.isnan(z)))
    print('ra is nan at ',np.where(np.isnan(ra)))
    print('dec is nan at ',np.where(np.isnan(dec)))
    print('w is nan at ',np.where(np.isnan(w)))
    print('wpos is nan at ',np.where(np.isnan(wpos)))
    print('g1 is nan at ',np.where(np.isnan(g1)))
    print('g2 is nan at ',np.where(np.isnan(g2)))
    print('k is nan at ',np.where(np.isnan(k)))

    with CaptureLog() as cl:
        cat1 = treecorr.Catalog(x=x, y=y, z=z, w=w, k=k, logger=cl.logger, keep_zero_weight=True)
    assert "NaNs found in x column." in cl.output
    assert "NaNs found in y column." in cl.output
    assert "NaNs found in z column." in cl.output
    assert "NaNs found in k column." in cl.output
    assert "NaNs found in w column." in cl.output
    mask = np.isnan(x) | np.isnan(y) | np.isnan(z) | np.isnan(k) | np.isnan(w)
    good = ~mask
    assert cat1.ntot == nobj
    assert cat1.nobj == np.sum(good)
    np.testing.assert_almost_equal(cat1.x[good], x[good])
    np.testing.assert_almost_equal(cat1.y[good], y[good])
    np.testing.assert_almost_equal(cat1.z[good], z[good])
    np.testing.assert_almost_equal(cat1.w[good], w[good])
    np.testing.assert_almost_equal(cat1.k[good], k[good])
    np.testing.assert_almost_equal(cat1.w[mask], 0)

    with CaptureLog() as cl:
        cat2 = treecorr.Catalog(ra=ra, dec=dec, r=r, w=w, wpos=wpos, g1=g1, g2=g2,
                                ra_units='hours', dec_units='degrees', logger=cl.logger,
                                keep_zero_weight=True)
    assert "NaNs found in ra column." in cl.output
    assert "NaNs found in dec column." in cl.output
    assert "NaNs found in r column." in cl.output
    assert "NaNs found in g1 column." in cl.output
    assert "NaNs found in g2 column." in cl.output
    assert "NaNs found in w column." in cl.output
    assert "NaNs found in wpos column." in cl.output
    mask = (np.isnan(ra) | np.isnan(dec) | np.isnan(r) |
            np.isnan(g1) | np.isnan(g2) | np.isnan(wpos) | np.isnan(w))
    good = ~mask
    assert cat2.ntot == nobj
    assert cat2.nobj == np.sum(good)
    np.testing.assert_almost_equal(cat2.ra[good], ra[good] * coord.hours / coord.radians)
    np.testing.assert_almost_equal(cat2.dec[good], dec[good] * coord.degrees / coord.radians)
    np.testing.assert_almost_equal(cat2.r[good], r[good])
    np.testing.assert_almost_equal(cat2.w[good], w[good])
    np.testing.assert_almost_equal(cat2.wpos[good], wpos[good])
    np.testing.assert_almost_equal(cat2.g1[good], g1[good])
    np.testing.assert_almost_equal(cat2.g2[good], g2[good])
    np.testing.assert_almost_equal(cat2.w[mask], 0)

    # If no weight column, it is make automatically to deal with Nans.
    with CaptureLog() as cl:
        cat3 = treecorr.Catalog(x=x, y=y, g1=g1, g2=g2, logger=cl.logger, keep_zero_weight=True)
    mask = np.isnan(x) | np.isnan(y) | np.isnan(g1) | np.isnan(g2)
    good = ~mask
    assert cat3.ntot == nobj
    assert cat3.nobj == np.sum(good)
    np.testing.assert_almost_equal(cat3.x[good], x[good])
    np.testing.assert_almost_equal(cat3.y[good], y[good])
    np.testing.assert_almost_equal(cat3.w[good], 1.)
    np.testing.assert_almost_equal(cat3.g1[good], g1[good])
    np.testing.assert_almost_equal(cat3.g2[good], g2[good])
    np.testing.assert_almost_equal(cat3.w[mask], 0)

@timer
def test_nan2():
    # This test is in response to issue #90.  Indeed it is largely the script that Joe helpfully
    # posted as a proof of the problem.

    N = 10000
    np.random.seed(1234)
    ra = np.random.uniform(0, 20, N)
    dec = np.random.uniform(0, 20, N)

    g1 = np.random.uniform(-0.05, 0.05, N)
    g2 = np.random.uniform(-0.05, 0.05, N)
    k = np.random.uniform(-0.05, 0.05, N)

    config = {
        'min_sep': 0.5,
        'max_sep': 100.0,
        'nbins': 10,
        'bin_slop':0.,
        'sep_units':'arcmin',
    }

    gg1 = treecorr.GGCorrelation(config)
    cat1 = treecorr.Catalog(ra=ra, dec=dec, g1=g1, g2=g2, k=k, ra_units='deg', dec_units='deg')
    gg1.process(cat1)

    # Now add some Nan's to the end of the data
    # TreeCorr's message warns about this but says it's ignoring the rows
    ra = np.concatenate((ra, [0.0]))
    dec = np.concatenate((dec, [0.0]))
    g1 = np.concatenate((g1, [np.nan]))
    g2 = np.concatenate((g2, [np.nan]))
    k = np.concatenate((k, [np.nan]))

    gg2 = treecorr.GGCorrelation(config)
    cat2 = treecorr.Catalog(ra=ra, dec=dec, g1=g1, g2=g2, k=k, ra_units='deg', dec_units='deg',
                            keep_zero_weight=True)

    assert cat2.nobj == cat1.nobj  # same number of non-zero weight
    assert cat2.ntot != cat1.ntot  # but not the same total

    # With (default) keep_zero_weight=False, catalogs are identical
    cat2b = treecorr.Catalog(ra=ra, dec=dec, g1=g1, g2=g2, k=k, ra_units='deg', dec_units='deg',
                             keep_zero_weight=False)
    assert cat2b.nobj == cat1.nobj
    assert cat2b.ntot == cat1.ntot
    assert cat2b == cat1

    gg2.process(cat2)
    print('cat1.nobj, ntot = ',cat1.nobj,cat1.ntot)
    print('cat2.nobj, ntot = ',cat2.nobj,cat2.ntot)
    print('gg1.weight = ',gg1.weight)
    print('gg2.weight = ',gg2.weight)
    print('diff = ',gg1.weight-gg2.weight)
    print('gg1.xip = ',gg1.xip)
    print('gg2.xip = ',gg2.xip)
    print('diff = ',gg1.xip-gg2.xip)
    print('gg1.xim = ',gg1.xim)
    print('gg2.xim = ',gg2.xim)
    print('diff = ',gg1.xim-gg2.xim)
    print('gg1.npairs = ',gg1.npairs)
    print('gg2.npairs = ',gg2.npairs)

    # First make sure that this particular random seed leads to different npairs for the
    # range being measured.
    assert np.any(gg1.npairs != gg2.npairs)

    # But same weight
    np.testing.assert_allclose(gg1.weight, gg2.weight)

    # Passes - NaNs ignored in mean calculation
    # Not exactly identical, because cells are different in detail, so sums have different
    # order of operations.
    np.testing.assert_allclose(gg1.xip, gg2.xip, atol=1.e-8)
    np.testing.assert_allclose(gg1.xim, gg2.xim, atol=1.e-8)

    # Used to fail from cat2.varg being NaN
    np.testing.assert_allclose(gg1.varxip, gg2.varxip)
    np.testing.assert_allclose(gg1.varxim, gg2.varxim)

    # Check the underlying varg, vark
    np.testing.assert_allclose(cat1.varg, cat2.varg)
    np.testing.assert_allclose(cat1.vark, cat2.vark)

    # Catalog generation with > 20 nans, to test the other pathway in the warnings code.
    g1[:100] = np.nan
    with CaptureLog() as cl:
        cat2c = treecorr.Catalog(ra=ra, dec=dec, g1=g1, g2=g2, ra_units='deg', dec_units='deg',
                                 logger=cl.logger)
    assert cat2c.nobj == cat2c.ntot == 9900
    print(cl.output)
    assert "Skipping rows starting" in cl.output

@timer
def test_contiguous():
    # This unit test comes from Melanie Simet who discovered a bug in earlier
    # versions of the code that the Catalog didn't correctly handle input arrays
    # that were not contiguous in memory.  We want to make sure this kind of
    # input works correctly.  It also checks that the input dtype doesn't have
    # to be float

    source_data = np.array([
            (0.0380569697547, 0.0142782758818, 0.330845443464, -0.111049332655),
            (-0.0261291090735, 0.0863787933931, 0.122954685209, 0.40260430406),
            (-0.0261291090735, 0.0863787933931, 0.122954685209, 0.40260430406),
            (0.125086697534, 0.0283621046495, -0.208159531309, 0.142491564101),
            (0.0457709426026, -0.0299249486373, -0.0406555089425, 0.24515956887),
            (-0.00338578248926, 0.0460291122935, 0.363057738173, -0.524536297555)],
            dtype=[('ra', None), ('dec', np.float64), ('g1', np.float32),
                   ('g2', np.float128)])

    config = {'min_sep': 0.05, 'max_sep': 0.2, 'sep_units': 'degrees', 'nbins': 5 }

    cat1 = treecorr.Catalog(ra=[0], dec=[0], ra_units='deg', dec_units='deg') # dumb lens
    cat2 = treecorr.Catalog(ra=source_data['ra'], ra_units='deg',
                            dec=source_data['dec'], dec_units='deg',
                            g1=source_data['g1'],
                            g2=source_data['g2'])
    cat2_float = treecorr.Catalog(ra=source_data['ra'].astype(float), ra_units='deg',
                                  dec=source_data['dec'].astype(float), dec_units='deg',
                                  g1=source_data['g1'].astype(float),
                                  g2=source_data['g2'].astype(float))

    print("dtypes of original arrays: ", [source_data[key].dtype for key in ['ra','dec','g1','g2']])
    print("dtypes of cat2 arrays: ", [getattr(cat2,key).dtype for key in ['ra','dec','g1','g2']])
    print("is original g2 array contiguous?", source_data['g2'].flags['C_CONTIGUOUS'])
    print("is cat2.g2 array contiguous?", cat2.g2.flags['C_CONTIGUOUS'])
    assert not source_data['g2'].flags['C_CONTIGUOUS']
    assert cat2.g2.flags['C_CONTIGUOUS']

    ng = treecorr.NGCorrelation(config)
    ng.process(cat1,cat2)
    ng_float = treecorr.NGCorrelation(config)
    ng_float.process(cat1,cat2_float)
    np.testing.assert_equal(ng.xi, ng_float.xi)

    # While we're at it, check that non-1d inputs work, but emit a warning.
    if __name__ == '__main__':
        v = 1
    else:
        v = 0
    cat2_non1d = treecorr.Catalog(ra=source_data['ra'].reshape(3,2), ra_units='deg',
                                  dec=source_data['dec'].reshape(1,1,1,6), dec_units='deg',
                                  g1=source_data['g1'].reshape(6,1),
                                  g2=source_data['g2'].reshape(1,6), verbose=v)
    ng.process(cat1,cat2_non1d)
    np.testing.assert_equal(ng.xi, ng_float.xi)


@timer
def test_list():
    # Test different ways to read in a list of catalog names.
    # This is based on the bug report for Issue #10.

    nobj = 5000
    rng = np.random.RandomState(8675309)

    x_list = []
    y_list = []
    file_names = []
    ncats = 3

    for k in range(ncats):
        x = rng.random_sample(nobj)
        y = rng.random_sample(nobj)
        file_name = os.path.join('data','test_list%d.dat'%k)

        with open(file_name, 'w') as fid:
            # These are intentionally in a different order from the order we parse them.
            fid.write('# ra,dec,x,y,k,g1,g2,w,flag\n')
            for i in range(nobj):
                fid.write(('%.8f %.8f\n')%(x[i],y[i]))
        x_list.append(x)
        y_list.append(y)
        file_names.append(file_name)

    # Start with file_name being a list:
    config = {
        'x_col' : 1,
        'y_col' : 2,
        'file_name' : file_names
    }

    cats = treecorr.read_catalogs(config, key='file_name')
    np.testing.assert_equal(len(cats), ncats)
    for k in range(ncats):
        np.testing.assert_almost_equal(cats[k].x, x_list[k])
        np.testing.assert_almost_equal(cats[k].y, y_list[k])

    # Next check that the list can be just a string with spaces between names:
    config['file_name'] = " ".join(file_names)

    # Also check that it is ok to include file_list to read_catalogs.
    cats = treecorr.read_catalogs(config, 'file_name', 'file_list')
    np.testing.assert_equal(len(cats), ncats)
    for k in range(ncats):
        np.testing.assert_almost_equal(cats[k].x, x_list[k])
        np.testing.assert_almost_equal(cats[k].y, y_list[k])

    # Next check that having the names in a file_list file works:
    list_name = os.path.join('data','test_list.txt')
    with open(list_name, 'w') as fid:
        for name in file_names:
            fid.write(name + '\n')
    del config['file_name']
    config['file_list'] = list_name

    cats = treecorr.read_catalogs(config, 'file_name', 'file_list')
    np.testing.assert_equal(len(cats), ncats)
    for k in range(ncats):
        np.testing.assert_almost_equal(cats[k].x, x_list[k])
        np.testing.assert_almost_equal(cats[k].y, y_list[k])

    # Also, should be allowed to omit file_name arg:
    cats = treecorr.read_catalogs(config, list_key='file_list')
    np.testing.assert_equal(len(cats), ncats)
    for k in range(ncats):
        np.testing.assert_almost_equal(cats[k].x, x_list[k])
        np.testing.assert_almost_equal(cats[k].y, y_list[k])

@timer
def test_write():
    # Test that writing a Catalog to a file and then reading it back in works correctly
    ngal = 20000
    s = 10.
    rng = np.random.RandomState(8675309)
    x = rng.normal(222,50, (ngal,) )
    y = rng.normal(138,20, (ngal,) )
    z = rng.normal(912,130, (ngal,) )
    w = rng.normal(1.3, 0.1, (ngal,) )

    ra = rng.normal(11.34, 0.9, (ngal,) )
    dec = rng.normal(-48.12, 4.3, (ngal,) )
    r = rng.normal(1024, 230, (ngal,) )

    k = rng.normal(0,s, (ngal,) )
    g1 = rng.normal(0,s, (ngal,) )
    g2 = rng.normal(0,s, (ngal,) )

    cat1 = treecorr.Catalog(x=x, y=y, z=z)
    cat2 = treecorr.Catalog(ra=ra, dec=dec, r=r, ra_units='hour', dec_units='deg',
                            w=w, g1=g1, g2=g2, k=k)

    # Test ASCII output
    cat1.write(os.path.join('output','cat1.dat'), cat_precision=20)
    cat1_asc = treecorr.Catalog(os.path.join('output','cat1.dat'), file_type='ASCII',
                                x_col=1, y_col=2, z_col=3)
    np.testing.assert_almost_equal(cat1_asc.x, x)
    np.testing.assert_almost_equal(cat1_asc.y, y)
    np.testing.assert_almost_equal(cat1_asc.z, z)

    cat2.write(os.path.join('output','cat2.dat'), file_type='ASCII')
    cat2_asc = treecorr.Catalog(os.path.join('output','cat2.dat'), ra_col=1, dec_col=2,
                                r_col=3, w_col=4, g1_col=5, g2_col=6, k_col=7,
                                ra_units='rad', dec_units='rad')
    np.testing.assert_almost_equal(cat2_asc.ra, ra)
    np.testing.assert_almost_equal(cat2_asc.dec, dec)
    np.testing.assert_almost_equal(cat2_asc.r, r)
    np.testing.assert_almost_equal(cat2_asc.w, w)
    np.testing.assert_almost_equal(cat2_asc.g1, g1)
    np.testing.assert_almost_equal(cat2_asc.g2, g2)
    np.testing.assert_almost_equal(cat2_asc.k, k)

    cat2r_asc = treecorr.Catalog(os.path.join('output','cat2.dat'), ra_col=1, dec_col=2,
                                 r_col=3, w_col=4, g1_col=5, g2_col=6, k_col=7,
                                 ra_units='rad', dec_units='rad', is_rand=True)
    np.testing.assert_almost_equal(cat2r_asc.ra, ra)
    np.testing.assert_almost_equal(cat2r_asc.dec, dec)
    np.testing.assert_almost_equal(cat2r_asc.r, r)
    np.testing.assert_almost_equal(cat2r_asc.w, w)
    assert cat2r_asc.g1 is None
    assert cat2r_asc.g2 is None
    assert cat2r_asc.k is None

    # Test FITS output
    cat1.write(os.path.join('output','cat1.fits'), file_type='FITS')
    cat1_fits = treecorr.Catalog(os.path.join('output','cat1.fits'),
                                 x_col='x', y_col='y', z_col='z')
    np.testing.assert_almost_equal(cat1_fits.x, x)
    np.testing.assert_almost_equal(cat1_fits.y, y)
    np.testing.assert_almost_equal(cat1_fits.z, z)

    cat2.write(os.path.join('output','cat2.fits'))
    cat2_fits = treecorr.Catalog(os.path.join('output','cat2.fits'), ra_col='ra', dec_col='dec',
                                 r_col='r', w_col='w', g1_col='g1', g2_col='g2', k_col='k',
                                 ra_units='rad', dec_units='rad', file_type='FITS')
    np.testing.assert_almost_equal(cat2_fits.ra, ra)
    np.testing.assert_almost_equal(cat2_fits.dec, dec)
    np.testing.assert_almost_equal(cat2_fits.r, r)
    np.testing.assert_almost_equal(cat2_fits.w, w)
    np.testing.assert_almost_equal(cat2_fits.g1, g1)
    np.testing.assert_almost_equal(cat2_fits.g2, g2)
    np.testing.assert_almost_equal(cat2_fits.k, k)

    cat2r_fits = treecorr.Catalog(os.path.join('output','cat2.fits'), ra_col='ra', dec_col='dec',
                                  r_col='r', w_col='w', g1_col='g1', g2_col='g2', k_col='k',
                                  ra_units='rad', dec_units='rad', file_type='FITS', is_rand=True)
    np.testing.assert_almost_equal(cat2r_fits.ra, ra)
    np.testing.assert_almost_equal(cat2r_fits.dec, dec)
    np.testing.assert_almost_equal(cat2r_fits.r, r)
    np.testing.assert_almost_equal(cat2r_fits.w, w)
    assert cat2r_fits.g1 is None
    assert cat2r_fits.g2 is None
    assert cat2r_fits.k is None

    # Test HDF5 output
    try:
        import h5py  # noqa: F401
    except ImportError:
        print('Skipping hdf5 write tests, since h5py not installed.')
        return

    cat1.write(os.path.join('output','cat1.hdf5'), file_type='HDF')
    cat1_hdf5 = treecorr.Catalog(os.path.join('output','cat1.hdf5'),
                                 x_col='x', y_col='y', z_col='z')
    np.testing.assert_almost_equal(cat1_hdf5.x, x)
    np.testing.assert_almost_equal(cat1_hdf5.y, y)
    np.testing.assert_almost_equal(cat1_hdf5.z, z)

    cat2.write(os.path.join('output','cat2.hdf'))
    cat2_hdf5 = treecorr.Catalog(os.path.join('output','cat2.hdf'), ra_col='ra', dec_col='dec',
                                 r_col='r', w_col='w', g1_col='g1', g2_col='g2', k_col='k',
                                 ra_units='rad', dec_units='rad', file_type='HDF5')
    np.testing.assert_almost_equal(cat2_hdf5.ra, ra)
    np.testing.assert_almost_equal(cat2_hdf5.dec, dec)
    np.testing.assert_almost_equal(cat2_hdf5.r, r)
    np.testing.assert_almost_equal(cat2_hdf5.w, w)
    np.testing.assert_almost_equal(cat2_hdf5.g1, g1)
    np.testing.assert_almost_equal(cat2_hdf5.g2, g2)
    np.testing.assert_almost_equal(cat2_hdf5.k, k)

    cat2r_hdf5 = treecorr.Catalog(os.path.join('output','cat2.hdf'), ra_col='ra', dec_col='dec',
                                  r_col='r', w_col='w', g1_col='g1', g2_col='g2', k_col='k',
                                  ra_units='rad', dec_units='rad', file_type='HDF5', is_rand=True)
    np.testing.assert_almost_equal(cat2r_hdf5.ra, ra)
    np.testing.assert_almost_equal(cat2r_hdf5.dec, dec)
    np.testing.assert_almost_equal(cat2r_hdf5.r, r)
    np.testing.assert_almost_equal(cat2r_hdf5.w, w)
    assert cat2r_hdf5.g1 is None
    assert cat2r_hdf5.g2 is None
    assert cat2r_hdf5.k is None

@timer
def test_field():
    # Test making various kinds of fields
    # Note: This is mostly just a coverage test to make sure there aren't any errors
    # when doing this manually.  The real functionality tests of using the fields are
    # all elsewhere.

    ngal = 2000
    s = 10.
    rng = np.random.RandomState(8675309)
    x = rng.normal(222,50, (ngal,) )
    y = rng.normal(138,20, (ngal,) )
    z = rng.normal(912,130, (ngal,) )
    w = rng.normal(1.3, 0.1, (ngal,) )

    ra = rng.normal(11.34, 0.9, (ngal,) )
    dec = rng.normal(-48.12, 4.3, (ngal,) )

    k = rng.normal(0,s, (ngal,) )
    g1 = rng.normal(0,s, (ngal,) )
    g2 = rng.normal(0,s, (ngal,) )

    cat1 = treecorr.Catalog(x=x, y=y, z=z, g1=g1, g2=g2, k=k)
    cat2 = treecorr.Catalog(ra=ra, dec=dec, ra_units='hour', dec_units='deg',
                            w=w, g1=g1, g2=g2, k=k)
    cat2.logger = None
    cat3 = treecorr.Catalog(x=x, y=y, g1=g1, g2=g2, k=k, w=w)
    cat3 = cat3.copy()  # This tests that post-pickled catalog still works correctly.
    cat4 = treecorr.Catalog(x=x, y=y, w=w)
    logger = treecorr.config.setup_logger(1)

    assert cat1.field is None  # Before calling get*Field, this is None.
    assert cat2.field is None
    assert cat3.field is None

    t0 = time.time()
    nfield1 = cat1.getNField()
    nfield2 = cat2.getNField(0.01, 1)
    nfield3 = cat3.getNField(1,300, logger=logger)
    t1 = time.time()
    nfield1b = cat1.getNField()
    nfield2b = cat2.getNField(0.01, 1)
    nfield3b = cat3.getNField(1,300, logger=logger)
    t2 = time.time()
    assert cat1.nfields.count == 1
    assert cat2.nfields.count == 1
    assert cat3.nfields.count == 1
    assert cat1.nfields.last_value is nfield1
    assert cat2.nfields.last_value is nfield2
    assert cat3.nfields.last_value is nfield3
    assert cat1.field is nfield1
    assert cat2.field is nfield2
    assert cat3.field is nfield3
    # The second time, they should already be made and taken from the cache, so much faster.
    print('nfield: ',t1-t0,t2-t1)
    assert t2-t1 < t1-t0

    t0 = time.time()
    gfield1 = cat1.getGField()
    gfield2 = cat2.getGField(0.01, 1)
    gfield3 = cat3.getGField(1,300, logger=logger)
    t1 = time.time()
    gfield1b = cat1.getGField()
    gfield2b = cat2.getGField(0.01, 1)
    gfield3b = cat3.getGField(1,300, logger=logger)
    t2 = time.time()
    assert_raises(TypeError, cat4.getGField)
    assert cat1.gfields.count == 1
    assert cat2.gfields.count == 1
    assert cat3.gfields.count == 1
    assert cat1.field is gfield1
    assert cat2.field is gfield2
    assert cat3.field is gfield3
    assert gfield1b is gfield1
    assert gfield2b is gfield2
    assert gfield3b is gfield3
    print('gfield: ',t1-t0,t2-t1)
    assert t2-t1 < t1-t0

    t0 = time.time()
    kfield1 = cat1.getKField()
    kfield2 = cat2.getKField(0.01, 1)
    kfield3 = cat3.getKField(1,300, logger=logger)
    t1 = time.time()
    kfield1b = cat1.getKField()
    kfield2b = cat2.getKField(0.01, 1)
    kfield3b = cat3.getKField(1,300, logger=logger)
    t2 = time.time()
    assert_raises(TypeError, cat4.getKField)
    assert cat1.kfields.count == 1
    assert cat2.kfields.count == 1
    assert cat3.kfields.count == 1
    assert cat1.field is kfield1
    assert cat2.field is kfield2
    assert cat3.field is kfield3
    assert kfield1b is kfield1
    assert kfield2b is kfield2
    assert kfield3b is kfield3
    print('kfield: ',t1-t0,t2-t1)
    assert t2-t1 < t1-t0

    t0 = time.time()
    nsimplefield1 = cat1.getNSimpleField()
    nsimplefield2 = cat2.getNSimpleField()
    nsimplefield3 = cat3.getNSimpleField(logger=logger)
    t1 = time.time()
    nsimplefield1b = cat1.getNSimpleField()
    nsimplefield2b = cat2.getNSimpleField()
    nsimplefield3b = cat3.getNSimpleField(logger=logger)
    t2 = time.time()
    assert cat1.nsimplefields.count == 1
    assert cat2.nsimplefields.count == 1
    assert cat3.nsimplefields.count == 1
    assert cat1.field is kfield1   # SimpleFields don't supplant the field attribute
    assert cat2.field is kfield2
    assert cat3.field is kfield3
    assert nsimplefield1b is nsimplefield1
    assert nsimplefield2b is nsimplefield2
    assert nsimplefield3b is nsimplefield3
    print('nsimplefield: ',t1-t0,t2-t1)
    assert t2-t1 < t1-t0

    t0 = time.time()
    gsimplefield1 = cat1.getGSimpleField()
    gsimplefield2 = cat2.getGSimpleField()
    gsimplefield3 = cat3.getGSimpleField(logger=logger)
    t1 = time.time()
    gsimplefield1b = cat1.getGSimpleField()
    gsimplefield2b = cat2.getGSimpleField()
    gsimplefield3b = cat3.getGSimpleField(logger=logger)
    t2 = time.time()
    assert_raises(TypeError, cat4.getGSimpleField)
    assert cat1.gsimplefields.count == 1
    assert cat2.gsimplefields.count == 1
    assert cat3.gsimplefields.count == 1
    assert cat1.field is kfield1   # SimpleFields don't supplant the field attribute
    assert cat2.field is kfield2
    assert cat3.field is kfield3
    assert gsimplefield1b is gsimplefield1
    assert gsimplefield2b is gsimplefield2
    assert gsimplefield3b is gsimplefield3
    print('gsimplefield: ',t1-t0,t2-t1)
    assert t2-t1 < t1-t0

    t0 = time.time()
    ksimplefield1 = cat1.getKSimpleField()
    ksimplefield2 = cat2.getKSimpleField()
    ksimplefield3 = cat3.getKSimpleField(logger=logger)
    t1 = time.time()
    ksimplefield1b = cat1.getKSimpleField()
    ksimplefield2b = cat2.getKSimpleField()
    ksimplefield3b = cat3.getKSimpleField(logger=logger)
    t2 = time.time()
    assert_raises(TypeError, cat4.getKSimpleField)
    assert cat1.ksimplefields.count == 1
    assert cat2.ksimplefields.count == 1
    assert cat3.ksimplefields.count == 1
    assert cat1.field is kfield1   # SimpleFields don't supplant the field attribute
    assert cat2.field is kfield2
    assert cat3.field is kfield3
    assert ksimplefield1b is ksimplefield1
    assert ksimplefield2b is ksimplefield2
    assert ksimplefield3b is ksimplefield3
    print('ksimplefield: ',t1-t0,t2-t1)
    assert t2-t1 < t1-t0

    # By default, only one is saved.  Check resize_cache option.
    cat1.resize_cache(3)
    assert cat1.nfields.size == 3
    assert cat1.kfields.size == 3
    assert cat1.gfields.size == 3
    assert cat1.nsimplefields.size == 3
    assert cat1.ksimplefields.size == 3
    assert cat1.gsimplefields.size == 3
    assert cat1.nfields.count == 1
    assert cat1.kfields.count == 1
    assert cat1.gfields.count == 1
    assert cat1.nsimplefields.count == 1
    assert cat1.ksimplefields.count == 1
    assert cat1.gsimplefields.count == 1
    assert cat1.field is kfield1

    t0 = time.time()
    nfield1 = cat1.getNField()
    nfield2 = cat1.getNField(0.01, 1)
    nfield3 = cat1.getNField(1,300, logger=logger)
    t1 = time.time()
    nfield1b = cat1.getNField()
    nfield2b = cat1.getNField(0.01, 1)
    nfield3b = cat1.getNField(1,300, logger=logger)
    t2 = time.time()
    assert cat1.nfields.count == 3
    print('after resize(3) nfield: ',t1-t0,t2-t1)
    assert t2-t1 < t1-t0
    assert nfield1b is nfield1
    assert nfield2b is nfield2
    assert nfield3b is nfield3
    assert cat1.nfields.values()
    assert nfield2 in cat1.nfields.values()
    assert nfield3 in cat1.nfields.values()
    assert cat1.nfields.last_value is nfield3
    assert cat1.field is nfield3

    # clear_cache will manually remove them.
    cat1.clear_cache()
    print('values = ',cat1.nfields.values())
    print('len(cache) = ',len(cat1.nfields.cache))
    assert len(cat1.nfields.values()) == 0
    assert cat1.nfields.count == 0
    assert cat1.gfields.count == 0
    assert cat1.kfields.count == 0
    assert cat1.nsimplefields.count == 0
    assert cat1.gsimplefields.count == 0
    assert cat1.ksimplefields.count == 0
    assert cat1.field is None

    # Can also resize to 0
    cat1.resize_cache(0)
    assert cat1.nfields.count == 0
    assert cat1.nfields.size == 0
    t0 = time.time()
    nfield1 = cat1.getNField()
    nfield2 = cat1.getNField(0.01, 1)
    nfield3 = cat1.getNField(1,300, logger=logger)
    t1 = time.time()
    nfield1b = cat1.getNField()
    nfield2b = cat1.getNField(0.01, 1)
    nfield3b = cat1.getNField(1,300, logger=logger)
    t2 = time.time()
    # This time, not much time difference.
    print('after resize(0) nfield: ',t1-t0,t2-t1)
    assert cat1.nfields.count == 0
    assert nfield1b is not nfield1
    assert nfield2b is not nfield2
    assert nfield3b is not nfield3
    assert len(cat1.nfields.values()) == 0
    assert cat1.nfields.last_value is None

    # The field still holds this, since it hasn't been garbage collected.
    assert cat1.field is nfield3b
    del nfield3b  # Delete the version from this scope so it can be garbage collected.
    print('before garbage collection: cat1.field = ',cat1.field)
    gc.collect()
    print('after garbage collection: cat1.field = ',cat1.field)
    assert cat1.field is None

    # Check NotImplementedError for base classes.
    assert_raises(NotImplementedError, treecorr.Field)
    assert_raises(NotImplementedError, treecorr.SimpleField)


@timer
def test_lru():
    f = lambda x: x+1
    size = 10
    # Test correct size cache gets created
    cache = treecorr.util.LRU_Cache(f, maxsize=size)
    assert len(cache.cache) == size
    assert cache.size == size
    assert cache.count == 0
    # Insert f(0) = 1 into cache and check that we can get it back
    assert cache(0) == f(0)
    assert cache.size == size
    assert cache.count == 1
    assert cache(0) == f(0)
    assert cache.size == size
    assert cache.count == 1

    # Manually manipulate cache so we can check for hit
    cache.cache[(0,)][3] = 2
    cache.count += 1
    assert cache(0) == 2
    assert cache.count == 2

    # Insert (and check) 1 thru size into cache.  This should bump out the (0,).
    for i in range(1, size+1):
        assert cache(i) == f(i)
    assert (0,) not in cache.cache
    assert cache.size == size
    assert cache.count == size

    # Test non-destructive cache expansion
    newsize = 20
    cache.resize(newsize)
    for i in range(1, size+1):
        assert (i,) in cache.cache
        assert cache(i) == f(i)
    assert len(cache.cache) == newsize
    assert cache.size == newsize
    assert cache.count == size

    # Add new items until the (1,) gets bumped
    for i in range(size+1, newsize+2):
        assert cache(i) == f(i)
    assert (1,) not in cache.cache
    assert cache.size == newsize
    assert cache.count == newsize

    # "Resize" to same size does nothing.
    cache.resize(newsize)
    assert len(cache.cache) == newsize
    assert cache.size == newsize
    assert cache.count == newsize
    assert (1,) not in cache.cache
    for i in range(2, newsize+2):
        assert (i,) in cache.cache
    assert cache.size == newsize
    assert cache.count == newsize

    # Test mostly non-destructive cache contraction.
    # Already bumped (0,) and (1,), so (2,) should be the first to get bumped
    for i in range(newsize-1, size, -1):
        assert (newsize - (i - 1),) in cache.cache
        cache.resize(i)
        assert (newsize - (i - 1),) not in cache.cache

    # Check if is works with size=0
    cache.resize(0)
    print('cache.cache = ',cache.cache)
    print('cache.root = ',cache.root)
    assert cache.root[0] == cache.root
    assert cache.root[1] == cache.root
    assert cache.size == 0
    assert cache.count == 0
    for i in range(10):
        assert cache(i) == f(i)
    print('=> cache.cache = ',cache.cache)
    print('=> cache.root = ',cache.root)
    assert cache.root[0] == cache.root
    assert cache.root[1] == cache.root
    assert cache.size == 0
    assert cache.count == 0

    assert_raises(ValueError, cache.resize, -20)


if __name__ == '__main__':
    test_ascii()
    test_fits()
    test_hdf5()
    test_parquet()
    test_ext()
    test_hdu()
    test_direct()
    test_var()
    test_nan()
    test_nan2()
    test_contiguous()
    test_list()
    test_write()
    test_field()
    test_lru()
