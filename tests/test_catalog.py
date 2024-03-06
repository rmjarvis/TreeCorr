# Copyright (c) 2003-2024 by Mike Jarvis
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
import os
import sys
import time
import coord
import gc
import copy
import pickle
import platform
from numpy import pi
import treecorr
from unittest import mock

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
    k = rng.random_sample(nobj)
    z1 = rng.random_sample(nobj)
    z2 = rng.random_sample(nobj)
    v1 = rng.random_sample(nobj)
    v2 = rng.random_sample(nobj)
    g1 = rng.random_sample(nobj)
    g2 = rng.random_sample(nobj)
    t1 = rng.random_sample(nobj)
    t2 = rng.random_sample(nobj)
    q1 = rng.random_sample(nobj)
    q2 = rng.random_sample(nobj)

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
        fid.write('# ra, dec, x, y, k, g1, g2, w, z, v1, v2, r, wpos, flag, t1, t2, z1, z2, q1, q2\n')
        for i in range(nobj):
            fid.write((('%.8f '*13)+'%d'+(' %.8f'*6)+'\n')%(
                ra[i],dec[i],x[i],y[i],k[i],g1[i],g2[i],w[i],z[i],
                v1[i],v2[i],r[i],wpos[i],flags[i],t1[i],t2[i],z1[i],z2[i],q1[i],q2[i]))

    # Check basic input
    config = {
        'x_col' : 3,
        'y_col' : 4,
        'z_col' : 9,
        'x_units' : 'rad',
        'y_units' : 'rad',
        'w_col' : 8,
        'wpos_col' : 13,
        'k_col' : 5,
        'z1_col' : 17,
        'z2_col' : 18,
        'v1_col' : 10,
        'v2_col' : 11,
        'g1_col' : 6,
        'g2_col' : 7,
        't1_col' : 15,
        't2_col' : 16,
        'q1_col' : 19,
        'q2_col' : 20,
        'kk_file_name' : 'kk.out',  # These make sure k, g, etc. are required.
        'zz_file_name' : 'zz.out',
        'gg_file_name' : 'gg.out',
        'vv_file_name' : 'vv.out',
        'tt_file_name' : 'tt.out',
        'qq_file_name' : 'qq.out',
        'keep_zero_weight' : True,
    }
    cat1 = treecorr.Catalog(file_name, config)
    np.testing.assert_almost_equal(cat1.x, x)
    np.testing.assert_almost_equal(cat1.y, y)
    np.testing.assert_almost_equal(cat1.z, z)
    np.testing.assert_almost_equal(cat1.w, w)
    np.testing.assert_almost_equal(cat1.k, k)
    np.testing.assert_almost_equal(cat1.z1, z1)
    np.testing.assert_almost_equal(cat1.z2, z2)
    np.testing.assert_almost_equal(cat1.v1, v1)
    np.testing.assert_almost_equal(cat1.v2, v2)
    np.testing.assert_almost_equal(cat1.g1, g1)
    np.testing.assert_almost_equal(cat1.g2, g2)
    np.testing.assert_almost_equal(cat1.t1, t1)
    np.testing.assert_almost_equal(cat1.t2, t2)
    np.testing.assert_almost_equal(cat1.q1, q1)
    np.testing.assert_almost_equal(cat1.q2, q2)
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
        'z1_col' : 'z1',
        'z2_col' : 'z2',
        'v1_col' : 'v1',
        'v2_col' : 'v2',
        'g1_col' : 'g1',
        'g2_col' : 'g2',
        't1_col' : 't1',
        't2_col' : 't2',
        'q1_col' : 'q1',
        'q2_col' : 'q2',
        'keep_zero_weight' : True,
    }
    cat1b = treecorr.Catalog(file_name, config_names)
    np.testing.assert_almost_equal(cat1b.x, x)
    np.testing.assert_almost_equal(cat1b.y, y)
    np.testing.assert_almost_equal(cat1b.z, z)
    np.testing.assert_almost_equal(cat1b.w, w)
    np.testing.assert_almost_equal(cat1b.k, k)
    np.testing.assert_almost_equal(cat1b.z1, z1)
    np.testing.assert_almost_equal(cat1b.z2, z2)
    np.testing.assert_almost_equal(cat1b.v1, v1)
    np.testing.assert_almost_equal(cat1b.v2, v2)
    np.testing.assert_almost_equal(cat1b.g1, g1)
    np.testing.assert_almost_equal(cat1b.g2, g2)
    np.testing.assert_almost_equal(cat1b.t1, t1)
    np.testing.assert_almost_equal(cat1b.t2, t2)
    np.testing.assert_almost_equal(cat1b.q1, q1)
    np.testing.assert_almost_equal(cat1b.q2, q2)
    np.testing.assert_almost_equal(cat1b.wpos, wpos)

    assert_raises(ValueError, treecorr.Catalog, file_name)
    assert_raises(TypeError, treecorr.Catalog, file_name, config, x=x)
    assert_raises(TypeError, treecorr.Catalog, file_name, config, y=y)
    assert_raises(TypeError, treecorr.Catalog, file_name, config, z=z)
    assert_raises(TypeError, treecorr.Catalog, file_name, config, ra=ra)
    assert_raises(TypeError, treecorr.Catalog, file_name, config, dec=dec)
    assert_raises(TypeError, treecorr.Catalog, file_name, config, r=r)
    assert_raises(TypeError, treecorr.Catalog, file_name, config, g2=g2)
    assert_raises(TypeError, treecorr.Catalog, file_name, config, z1=z1)
    assert_raises(TypeError, treecorr.Catalog, file_name, config, v1=v1)
    assert_raises(TypeError, treecorr.Catalog, file_name, config, t1=t1)
    assert_raises(TypeError, treecorr.Catalog, file_name, config, q2=q2)
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
    assert_raises(ValueError, treecorr.Catalog, file_name, config, z1_col=-1)
    assert_raises(ValueError, treecorr.Catalog, file_name, config, z1_col=100)
    assert_raises(ValueError, treecorr.Catalog, file_name, config, z1_col='invalid')
    assert_raises(ValueError, treecorr.Catalog, file_name, config, z2_col=-1)
    assert_raises(ValueError, treecorr.Catalog, file_name, config, z2_col=100)
    assert_raises(ValueError, treecorr.Catalog, file_name, config, z2_col='invalid')
    assert_raises(ValueError, treecorr.Catalog, file_name, config, v1_col=-1)
    assert_raises(ValueError, treecorr.Catalog, file_name, config, v1_col=100)
    assert_raises(ValueError, treecorr.Catalog, file_name, config, v1_col='invalid')
    assert_raises(ValueError, treecorr.Catalog, file_name, config, v2_col=-1)
    assert_raises(ValueError, treecorr.Catalog, file_name, config, v2_col=100)
    assert_raises(ValueError, treecorr.Catalog, file_name, config, v2_col='invalid')
    assert_raises(ValueError, treecorr.Catalog, file_name, config, t1_col=-1)
    assert_raises(ValueError, treecorr.Catalog, file_name, config, t1_col=100)
    assert_raises(ValueError, treecorr.Catalog, file_name, config, t1_col='invalid')
    assert_raises(ValueError, treecorr.Catalog, file_name, config, t2_col=-1)
    assert_raises(ValueError, treecorr.Catalog, file_name, config, t2_col=100)
    assert_raises(ValueError, treecorr.Catalog, file_name, config, t2_col='invalid')
    assert_raises(ValueError, treecorr.Catalog, file_name, config, q1_col=-1)
    assert_raises(ValueError, treecorr.Catalog, file_name, config, q1_col=100)
    assert_raises(ValueError, treecorr.Catalog, file_name, config, q1_col='invalid')
    assert_raises(ValueError, treecorr.Catalog, file_name, config, q2_col=-1)
    assert_raises(ValueError, treecorr.Catalog, file_name, config, q2_col=100)
    assert_raises(ValueError, treecorr.Catalog, file_name, config, q2_col='invalid')
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
    config['flag_col'] = 14
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
    config['wpos_col'] = 13

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

    config['x_units'] = None  # Default is radians
    config['y_units'] = None  # Default is radians
    cat5 = treecorr.Catalog(file_name, config)
    np.testing.assert_almost_equal(cat5.x, x)
    np.testing.assert_almost_equal(cat5.y, y)

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
    config['r_col'] = 12
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

    # Check using eval feature to apply the units rather than ra_units/dec_units
    config_names['ra_units'] = 'rad'
    config_names['dec_units'] = 'rad'
    config_names['ra_eval'] = 'ra * np.pi/12.'
    config_names['dec_eval'] = 'dec * math.pi/180.'
    cat6c = treecorr.Catalog(file_name, config_names)
    np.testing.assert_almost_equal(cat6c.ra, ra * (pi/12.))
    np.testing.assert_almost_equal(cat6c.dec, dec * (pi/180.))

    # Can also skip ra_col, dec_col and specify them in extra_cols.
    del config_names['ra_col']
    del config_names['dec_col']
    config_names['extra_cols'] = ['ra', 'dec']
    cat6d = treecorr.Catalog(file_name, config_names)
    np.testing.assert_almost_equal(cat6c.ra, ra * (pi/12.))
    np.testing.assert_almost_equal(cat6c.dec, dec * (pi/180.))

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
        fid.write('% ra,dec,x,y,k,g1,g2,v1,v2,w,flag,t1,t2,z1,z2,q1,q2\n')
        for i in range(nobj):
            fid.write((('%.8f,'*13)+'%d'+(',%.8f'*6)+'\n')%(
                ra[i],dec[i],x[i],y[i],k[i],g1[i],g2[i],w[i],z[i],
                v1[i],v2[i],r[i],wpos[i],flags[i],t1[i],t2[i],z1[i],z2[i],q1[i],q2[i]))
            if i%100 == 0:
                fid.write('%%%% Line %d\n'%i)
    config['delimiter'] = ','
    config['comment_marker'] = '%'
    cat7 = treecorr.Catalog(csv_file_name, config)
    np.testing.assert_almost_equal(cat7.ra, ra * (pi/12.))
    np.testing.assert_almost_equal(cat7.dec, dec * (pi/180.))
    np.testing.assert_almost_equal(cat7.r, r)
    np.testing.assert_almost_equal(cat7.z1, z1)
    np.testing.assert_almost_equal(cat7.z2, z2)
    np.testing.assert_almost_equal(cat7.v1, v1)
    np.testing.assert_almost_equal(cat7.v2, v2)
    np.testing.assert_almost_equal(cat7.g1, g1)
    np.testing.assert_almost_equal(cat7.g2, g2)
    np.testing.assert_almost_equal(cat7.t1, t1)
    np.testing.assert_almost_equal(cat7.t2, t2)
    np.testing.assert_almost_equal(cat7.q1, q1)
    np.testing.assert_almost_equal(cat7.q2, q2)
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

    # Check flip_z1, flip_z2
    config['flip_z1'] = True
    cat8 = treecorr.Catalog(file_name, config)
    np.testing.assert_almost_equal(cat8.z1, -z1)
    np.testing.assert_almost_equal(cat8.z2, z2)

    config['flip_z2'] = 'true'
    cat8 = treecorr.Catalog(file_name, config)
    np.testing.assert_almost_equal(cat8.z1, -z1)
    np.testing.assert_almost_equal(cat8.z2, -z2)

    config['flip_z1'] = 'n'
    config['flip_z2'] = 'yes'
    cat8 = treecorr.Catalog(file_name, config)
    np.testing.assert_almost_equal(cat8.z1, z1)
    np.testing.assert_almost_equal(cat8.z2, -z2)

    cat8 = treecorr.Catalog(file_name, config, flip_z1=True, flip_z2=False)
    np.testing.assert_almost_equal(cat8.z1, -z1)
    np.testing.assert_almost_equal(cat8.z2, z2)

    # Check flip_v1, flip_v2
    config['flip_v1'] = True
    cat8 = treecorr.Catalog(file_name, config)
    np.testing.assert_almost_equal(cat8.v1, -v1)
    np.testing.assert_almost_equal(cat8.v2, v2)

    config['flip_v2'] = 'true'
    cat8 = treecorr.Catalog(file_name, config)
    np.testing.assert_almost_equal(cat8.v1, -v1)
    np.testing.assert_almost_equal(cat8.v2, -v2)

    config['flip_v1'] = 'n'
    config['flip_v2'] = 'yes'
    cat8 = treecorr.Catalog(file_name, config)
    np.testing.assert_almost_equal(cat8.v1, v1)
    np.testing.assert_almost_equal(cat8.v2, -v2)

    cat8 = treecorr.Catalog(file_name, config, flip_v1=True, flip_v2=False)
    np.testing.assert_almost_equal(cat8.v1, -v1)
    np.testing.assert_almost_equal(cat8.v2, v2)

    # Check flip_t1, flip_t2
    config['flip_t1'] = True
    cat8 = treecorr.Catalog(file_name, config)
    np.testing.assert_almost_equal(cat8.t1, -t1)
    np.testing.assert_almost_equal(cat8.t2, t2)

    config['flip_t2'] = 'true'
    cat8 = treecorr.Catalog(file_name, config)
    np.testing.assert_almost_equal(cat8.t1, -t1)
    np.testing.assert_almost_equal(cat8.t2, -t2)

    config['flip_t1'] = 'n'
    config['flip_t2'] = 'yes'
    cat8 = treecorr.Catalog(file_name, config)
    np.testing.assert_almost_equal(cat8.t1, t1)
    np.testing.assert_almost_equal(cat8.t2, -t2)

    cat8 = treecorr.Catalog(file_name, config, flip_t1=True, flip_t2=False)
    np.testing.assert_almost_equal(cat8.t1, -t1)
    np.testing.assert_almost_equal(cat8.t2, t2)

    # Check flip_q1, flip_q2
    config['flip_q1'] = True
    cat8 = treecorr.Catalog(file_name, config)
    np.testing.assert_almost_equal(cat8.q1, -q1)
    np.testing.assert_almost_equal(cat8.q2, q2)

    config['flip_q2'] = 'true'
    cat8 = treecorr.Catalog(file_name, config)
    np.testing.assert_almost_equal(cat8.q1, -q1)
    np.testing.assert_almost_equal(cat8.q2, -q2)

    config['flip_q1'] = 'n'
    config['flip_q2'] = 'yes'
    cat8 = treecorr.Catalog(file_name, config)
    np.testing.assert_almost_equal(cat8.q1, q1)
    np.testing.assert_almost_equal(cat8.q2, -q2)

    cat8 = treecorr.Catalog(file_name, config, flip_q1=True, flip_q2=False)
    np.testing.assert_almost_equal(cat8.q1, -q1)
    np.testing.assert_almost_equal(cat8.q2, q2)

    # Check copy command
    cat9 = cat8.copy()
    np.testing.assert_almost_equal(cat9.ra, cat8.ra)
    np.testing.assert_almost_equal(cat9.dec, cat8.dec)
    np.testing.assert_almost_equal(cat9.r, cat8.r)
    np.testing.assert_almost_equal(cat9.z1, cat8.z1)
    np.testing.assert_almost_equal(cat9.z2, cat8.z2)
    np.testing.assert_almost_equal(cat9.v1, cat8.v1)
    np.testing.assert_almost_equal(cat9.v2, cat8.v2)
    np.testing.assert_almost_equal(cat9.g1, cat8.g1)
    np.testing.assert_almost_equal(cat9.g2, cat8.g2)
    np.testing.assert_almost_equal(cat9.t1, cat8.t1)
    np.testing.assert_almost_equal(cat9.t2, cat8.t2)
    np.testing.assert_almost_equal(cat9.q1, cat8.q1)
    np.testing.assert_almost_equal(cat9.q2, cat8.q2)
    np.testing.assert_almost_equal(cat9.w, cat8.w)

    # Swapping w and wpos leads to zeros being copied from wpos to w
    cat10 = treecorr.Catalog(file_name, config, w_col=13, wpos_col=8, flag_col=0)
    np.testing.assert_almost_equal(cat10.wpos, w)
    np.testing.assert_almost_equal(cat10.w, w)

    # And if there is wpos, but no w, copy over the zeros, but not the other values
    with CaptureLog() as cl:
        cat10 = treecorr.Catalog(file_name, config, w_col=0, wpos_col=13, flag_col=0,
                                 logger=cl.logger)
        cat10.x  # Force the read to happen.
    np.testing.assert_almost_equal(cat10.wpos, wpos)
    np.testing.assert_almost_equal(cat10.w[wpos==0], 0)
    np.testing.assert_almost_equal(cat10.w[wpos!=0], 1)
    assert 'Some wpos values are zero, setting w=0 for these points' in cl.output

    # Test using a limited set of rows
    del config['flip_z1']
    del config['flip_z2']
    del config['flip_v1']
    del config['flip_v2']
    del config['flip_g1']
    del config['flip_g2']
    del config['flip_t1']
    del config['flip_t2']
    del config['flip_q1']
    del config['flip_q2']
    config['first_row'] = 1010
    config['last_row'] = 3456
    cat11 = treecorr.Catalog(file_name, config)
    np.testing.assert_equal(len(cat11.ra), 2447)
    np.testing.assert_equal(cat11.ntot, 2447)
    np.testing.assert_equal(cat11.nobj, np.sum(cat11.w != 0))
    np.testing.assert_equal(cat11.sumw, np.sum(cat11.w))
    np.testing.assert_equal(cat11.sumw, np.sum(cat6.w[1009:3456]))
    np.testing.assert_almost_equal(cat11.k[1111], k[2120])
    np.testing.assert_almost_equal(cat11.z1[1111], z1[2120])
    np.testing.assert_almost_equal(cat11.z2[1111], z2[2120])
    np.testing.assert_almost_equal(cat11.v1[1111], v1[2120])
    np.testing.assert_almost_equal(cat11.v2[1111], v2[2120])
    np.testing.assert_almost_equal(cat11.g1[1111], g1[2120])
    np.testing.assert_almost_equal(cat11.g2[1111], g2[2120])
    np.testing.assert_almost_equal(cat11.t1[1111], t1[2120])
    np.testing.assert_almost_equal(cat11.t2[1111], t2[2120])
    np.testing.assert_almost_equal(cat11.q1[1111], q1[2120])
    np.testing.assert_almost_equal(cat11.q2[1111], q2[2120])

    config['file_name'] = file_name
    cat12 = treecorr.read_catalogs(config, key='file_name', is_rand=True)[0]
    np.testing.assert_equal(len(cat12.x), 2447)
    np.testing.assert_equal(cat12.ntot, 2447)
    np.testing.assert_equal(cat12.nobj, np.sum(cat12.w != 0))
    np.testing.assert_equal(cat12.sumw, np.sum(cat12.w))
    np.testing.assert_equal(cat12.sumw, np.sum(cat6.w[1009:3456]))
    assert cat12.k is None
    assert cat12.z1 is None
    assert cat12.z2 is None
    assert cat12.v1 is None
    assert cat12.v2 is None
    assert cat12.g1 is None
    assert cat12.g2 is None
    assert cat12.t1 is None
    assert cat12.t2 is None
    assert cat12.q1 is None
    assert cat12.q2 is None

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
    np.testing.assert_almost_equal(cat13.k[100], k[2009])
    np.testing.assert_almost_equal(cat13.z1[100], z1[2009])
    np.testing.assert_almost_equal(cat13.z2[100], z2[2009])
    np.testing.assert_almost_equal(cat13.v1[100], v1[2009])
    np.testing.assert_almost_equal(cat13.v2[100], v2[2009])
    np.testing.assert_almost_equal(cat13.g1[100], g1[2009])
    np.testing.assert_almost_equal(cat13.g2[100], g2[2009])
    np.testing.assert_almost_equal(cat13.t1[100], t1[2009])
    np.testing.assert_almost_equal(cat13.t2[100], t2[2009])
    np.testing.assert_almost_equal(cat13.q1[100], q1[2009])
    np.testing.assert_almost_equal(cat13.q2[100], q2[2009])

    # Check every_nth with no first/last
    del config['first_row']
    del config['last_row']
    cat13a = treecorr.Catalog(file_name, config)
    np.testing.assert_equal(len(cat13a.x), 500)
    np.testing.assert_equal(cat13a.ntot, 500)
    np.testing.assert_equal(cat13a.nobj, np.sum(cat13a.w != 0))
    np.testing.assert_equal(cat13a.sumw, np.sum(cat13a.w))
    np.testing.assert_equal(cat13a.sumw, np.sum(cat6.w[::10]))
    np.testing.assert_almost_equal(cat13a.k[100], k[1000])
    np.testing.assert_almost_equal(cat13a.z1[100], z1[1000])
    np.testing.assert_almost_equal(cat13a.z2[100], z2[1000])
    np.testing.assert_almost_equal(cat13a.v1[100], v1[1000])
    np.testing.assert_almost_equal(cat13a.v2[100], v2[1000])
    np.testing.assert_almost_equal(cat13a.t1[100], t1[1000])
    np.testing.assert_almost_equal(cat13a.t2[100], t2[1000])
    np.testing.assert_almost_equal(cat13a.g1[100], g1[1000])
    np.testing.assert_almost_equal(cat13a.g2[100], g2[1000])
    np.testing.assert_almost_equal(cat13a.q1[100], q1[1000])
    np.testing.assert_almost_equal(cat13a.q2[100], q2[1000])

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
    assert cat14a._k is None
    assert cat14a._z1 is None
    assert cat14a._z2 is None
    assert cat14a._v1 is None
    assert cat14a._v2 is None
    assert cat14a._g1 is None
    assert cat14a._g2 is None
    assert cat14a._t1 is None
    assert cat14a._t2 is None
    assert cat14a._q1 is None
    assert cat14a._q2 is None
    assert cat14a == cat14    # When needed, it will reload, e.g. here to check equality.

@timer
def test_fits():
    try:
        import fitsio
    except ImportError:
        print('Skip test_fits, since fitsio not installed.')
        return
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
    include_v = (file_type != 'Parquet')  # Parquet cannot handle duplicated names.
    if include_v:
        config['zz_file_name'] = 'zz.out'
        config['vv_file_name'] = 'vv.out'
        config['tt_file_name'] = 'tt.out'
        config['qq_file_name'] = 'qq.out'
        config['z1_col'] = config['g1_col']
        config['z2_col'] = config['g2_col']
        config['v1_col'] = config['g1_col']
        config['v2_col'] = config['g2_col']
        config['t1_col'] = config['g1_col']
        config['t2_col'] = config['g2_col']
        config['q1_col'] = config['g1_col']
        config['q2_col'] = config['g2_col']
    config['file_name'] = file_name
    config['ext'] = ext

    # Just test a few random particular values
    cat1 = treecorr.Catalog(file_name, config)
    np.testing.assert_equal(len(cat1.ra), 390935)
    np.testing.assert_equal(cat1.nobj, 390935)
    np.testing.assert_almost_equal(cat1.ra[0], 56.4195 * (pi/180.))
    np.testing.assert_almost_equal(cat1.ra[390934], 78.4782 * (pi/180.))
    np.testing.assert_almost_equal(cat1.dec[290333], 83.1579 * (pi/180.))
    np.testing.assert_almost_equal(cat1.k[46392], -0.0008628797)
    np.testing.assert_almost_equal(cat1.g1[46392], 0.0005066675)
    np.testing.assert_almost_equal(cat1.g2[46392], -0.0001006742)
    if include_v:
        np.testing.assert_almost_equal(cat1.z1[46392], 0.0005066675)
        np.testing.assert_almost_equal(cat1.z2[46392], -0.0001006742)
        np.testing.assert_almost_equal(cat1.v1[46392], 0.0005066675)
        np.testing.assert_almost_equal(cat1.v2[46392], -0.0001006742)
        np.testing.assert_almost_equal(cat1.t1[46392], 0.0005066675)
        np.testing.assert_almost_equal(cat1.t2[46392], -0.0001006742)
        np.testing.assert_almost_equal(cat1.q1[46392], 0.0005066675)
        np.testing.assert_almost_equal(cat1.q2[46392], -0.0001006742)

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
    assert_raises(ValueError, treecorr.Catalog, file_name, config, k_col='invalid')
    assert_raises(ValueError, treecorr.Catalog, file_name, config, z1_col='invalid')
    assert_raises(ValueError, treecorr.Catalog, file_name, config, z2_col='invalid')
    assert_raises(ValueError, treecorr.Catalog, file_name, config, v1_col='invalid')
    assert_raises(ValueError, treecorr.Catalog, file_name, config, v2_col='invalid')
    assert_raises(ValueError, treecorr.Catalog, file_name, config, g1_col='invalid')
    assert_raises(ValueError, treecorr.Catalog, file_name, config, g2_col='invalid')
    assert_raises(ValueError, treecorr.Catalog, file_name, config, t1_col='invalid')
    assert_raises(ValueError, treecorr.Catalog, file_name, config, t2_col='invalid')
    assert_raises(ValueError, treecorr.Catalog, file_name, config, q1_col='invalid')
    assert_raises(ValueError, treecorr.Catalog, file_name, config, q2_col='invalid')
    assert_raises(ValueError, treecorr.Catalog, file_name, config, ra_col='0')
    assert_raises(ValueError, treecorr.Catalog, file_name, config, dec_col='0')
    assert_raises(ValueError, treecorr.Catalog, file_name, config, x_col='x')
    assert_raises(ValueError, treecorr.Catalog, file_name, config, y_col='y')
    assert_raises(ValueError, treecorr.Catalog, file_name, config, z_col='z')
    assert_raises(ValueError, treecorr.Catalog, file_name, config, ra_col='0', dec_col='0')
    assert_raises(ValueError, treecorr.Catalog, file_name, config, k_col='0')
    assert_raises(ValueError, treecorr.Catalog, file_name, config, g1_col='0')
    assert_raises(ValueError, treecorr.Catalog, file_name, config, g2_col='0')
    if include_v:
        assert_raises(ValueError, treecorr.Catalog, file_name, config, z1_col='0')
        assert_raises(ValueError, treecorr.Catalog, file_name, config, z2_col='0')
        assert_raises(ValueError, treecorr.Catalog, file_name, config, v1_col='0')
        assert_raises(ValueError, treecorr.Catalog, file_name, config, v2_col='0')
        assert_raises(ValueError, treecorr.Catalog, file_name, config, t1_col='0')
        assert_raises(ValueError, treecorr.Catalog, file_name, config, t2_col='0')
        assert_raises(ValueError, treecorr.Catalog, file_name, config, q1_col='0')
        assert_raises(ValueError, treecorr.Catalog, file_name, config, q2_col='0')
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

    # Test some eval calculations
    center_ra = np.mean(cat1.ra) * coord.radians
    center_dec = np.mean(cat1.dec) * coord.radians
    center = coord.CelestialCoord(center_ra, center_dec)
    center_repr = repr(center)
    ntot = cat1.ntot
    cat3 = treecorr.Catalog(
        file_name,
        # tangent plane projection for ra,dec -> x,y
        x_eval=f"{center_repr}.project_rad(RA*math.pi/180, DEC*math.pi/180)[0]",
        y_eval=f"{center_repr}.project_rad(RA*math.pi/180, DEC*math.pi/180)[1]",
        # distortion rather than shear
        g1_eval="GAMMA1 * 2 / (1 + GAMMA1**2 + GAMMA2**2)",
        g2_eval="GAMMA2 * 2 / (1 + GAMMA1**2 + GAMMA2**2)",
        # random spin-3 directions with mu for the amplitude.  (This one is pretty contrived.)
        t1_eval=f"KAPPA * np.random.default_rng(1234).normal(0,0.3,{ntot})",
        t2_eval=f"KAPPA * np.random.default_rng(1235).normal(0,0.3,{ntot})",
        extra_cols=['RA', 'DEC', 'GAMMA1', 'GAMMA2', 'KAPPA'])
    print('made cat3')
    print('cat3 = ',cat3)
    print('x = ',cat3.x)
    print('y = ',cat3.y)
    print('g1 = ',cat3.g1)
    print('g2 = ',cat3.g2)
    print('t1 = ',cat3.t1)
    print('t2 = ',cat3.t2)
    x1, y1 = center.project_rad(cat1.ra, cat1.dec)
    np.testing.assert_allclose(cat3.x, x1, atol=1.e-6)
    np.testing.assert_allclose(cat3.y, y1, atol=1.e-6)
    gsq = cat1.g1**2 + cat1.g2**2
    np.testing.assert_allclose(cat3.g1, cat1.g1 * 2 / (1+gsq), rtol=1.e-6)
    np.testing.assert_allclose(cat3.g2, cat1.g2 * 2 / (1+gsq), rtol=1.e-6)
    np.testing.assert_allclose(np.mean(cat3.t1), 0., atol=1.e-3)
    np.testing.assert_allclose(np.mean(cat3.t2), 0., atol=1.e-3)
    np.testing.assert_allclose(np.std(cat3.t1), np.std(cat3.t2), rtol=0.1)

    assert_raises(ValueError, treecorr.Catalog, file_name, config, x_col='invalid')
    assert_raises(ValueError, treecorr.Catalog, file_name, config, y_col='invalid')
    assert_raises(ValueError, treecorr.Catalog, file_name, config, z_col='invalid')
    assert_raises(ValueError, treecorr.Catalog, file_name, config, ra_col='invalid')
    assert_raises(ValueError, treecorr.Catalog, file_name, config, dec_col='invalid')
    assert_raises(ValueError, treecorr.Catalog, file_name, config, r_col='invalid')
    assert_raises(ValueError, treecorr.Catalog, file_name, config, w_col='invalid')
    assert_raises(ValueError, treecorr.Catalog, file_name, config, wpos_col='invalid')
    assert_raises(ValueError, treecorr.Catalog, file_name, config, flag_col='invalid')
    assert_raises(ValueError, treecorr.Catalog, file_name, config, k_col='invalid')
    assert_raises(ValueError, treecorr.Catalog, file_name, config, z1_col='invalid')
    assert_raises(ValueError, treecorr.Catalog, file_name, config, z2_col='invalid')
    assert_raises(ValueError, treecorr.Catalog, file_name, config, v1_col='invalid')
    assert_raises(ValueError, treecorr.Catalog, file_name, config, v2_col='invalid')
    assert_raises(ValueError, treecorr.Catalog, file_name, config, g1_col='invalid')
    assert_raises(ValueError, treecorr.Catalog, file_name, config, g2_col='invalid')
    assert_raises(ValueError, treecorr.Catalog, file_name, config, t1_col='invalid')
    assert_raises(ValueError, treecorr.Catalog, file_name, config, t2_col='invalid')
    assert_raises(ValueError, treecorr.Catalog, file_name, config, q1_col='invalid')
    assert_raises(ValueError, treecorr.Catalog, file_name, config, q2_col='invalid')

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
    np.testing.assert_almost_equal(cat3.k[46292], cat2.k[46392])
    np.testing.assert_almost_equal(cat3.g1[46292], cat2.g1[46392])
    np.testing.assert_almost_equal(cat3.g2[46292], cat2.g2[46392])
    if include_v:
        np.testing.assert_almost_equal(cat3.z1[46292], cat2.z1[46392])
        np.testing.assert_almost_equal(cat3.z2[46292], cat2.z2[46392])
        np.testing.assert_almost_equal(cat3.v1[46292], cat2.v1[46392])
        np.testing.assert_almost_equal(cat3.v2[46292], cat2.v2[46392])
        np.testing.assert_almost_equal(cat3.t1[46292], cat2.t1[46392])
        np.testing.assert_almost_equal(cat3.t2[46292], cat2.t2[46392])
        np.testing.assert_almost_equal(cat3.q1[46292], cat2.q1[46392])
        np.testing.assert_almost_equal(cat3.q2[46292], cat2.q2[46392])

    cat4 = treecorr.read_catalogs(config, key='file_name', is_rand=True)[0]
    np.testing.assert_equal(len(cat4.x), 49900)
    np.testing.assert_equal(cat4.ntot, 49900)
    np.testing.assert_equal(cat4.nobj, np.sum(cat4.w != 0))
    np.testing.assert_equal(cat4.sumw, np.sum(cat4.w))
    np.testing.assert_equal(cat4.sumw, np.sum(cat2.w[100:50000]))
    assert cat4.k is None
    assert cat4.z1 is None
    assert cat4.z2 is None
    assert cat4.v1 is None
    assert cat4.v2 is None
    assert cat4.g1 is None
    assert cat4.g2 is None
    assert cat4.t1 is None
    assert cat4.t2 is None
    assert cat4.q1 is None
    assert cat4.q2 is None

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
    np.testing.assert_almost_equal(cat5a.k[123], cat2.k[12300])
    np.testing.assert_almost_equal(cat5a.g1[123], cat2.g1[12300])
    np.testing.assert_almost_equal(cat5a.g2[123], cat2.g2[12300])
    if include_v:
        np.testing.assert_almost_equal(cat5a.z1[123], cat2.z1[12300])
        np.testing.assert_almost_equal(cat5a.z2[123], cat2.z2[12300])
        np.testing.assert_almost_equal(cat5a.v1[123], cat2.v1[12300])
        np.testing.assert_almost_equal(cat5a.v2[123], cat2.v2[12300])
        np.testing.assert_almost_equal(cat5a.t1[123], cat2.t1[12300])
        np.testing.assert_almost_equal(cat5a.t2[123], cat2.t2[12300])
        np.testing.assert_almost_equal(cat5a.q1[123], cat2.q1[12300])
        np.testing.assert_almost_equal(cat5a.q2[123], cat2.q2[12300])

    # Now with first, last, and every_nth
    config['first_row'] = 101
    config['last_row'] = 50000
    cat5 = treecorr.Catalog(file_name, config)
    np.testing.assert_equal(len(cat5.x), 499)
    np.testing.assert_equal(cat5.ntot, 499)
    np.testing.assert_equal(cat5.nobj, np.sum(cat5.w != 0))
    np.testing.assert_equal(cat5.sumw, np.sum(cat5.w))
    np.testing.assert_equal(cat5.sumw, np.sum(cat2.w[100:50000:100]))
    np.testing.assert_almost_equal(cat5.k[123], cat2.k[12400])
    np.testing.assert_almost_equal(cat5.g1[123], cat2.g1[12400])
    np.testing.assert_almost_equal(cat5.g2[123], cat2.g2[12400])
    if include_v:
        np.testing.assert_almost_equal(cat5.z1[123], cat2.z1[12400])
        np.testing.assert_almost_equal(cat5.z2[123], cat2.z2[12400])
        np.testing.assert_almost_equal(cat5.v1[123], cat2.v1[12400])
        np.testing.assert_almost_equal(cat5.v2[123], cat2.v2[12400])
        np.testing.assert_almost_equal(cat5.t1[123], cat2.t1[12400])
        np.testing.assert_almost_equal(cat5.t2[123], cat2.t2[12400])
        np.testing.assert_almost_equal(cat5.q1[123], cat2.q1[12400])
        np.testing.assert_almost_equal(cat5.q2[123], cat2.q2[12400])

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
    #print('cat6 = ',repr(cat6))
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
    assert cat6a._k is None
    assert cat6a._z1 is None
    assert cat6a._z2 is None
    assert cat6a._v1 is None
    assert cat6a._v2 is None
    assert cat6a._g1 is None
    assert cat6a._g2 is None
    assert cat6a._t1 is None
    assert cat6a._t2 is None
    assert cat6a._q1 is None
    assert cat6a._q2 is None
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

    if include_v:
        del config['vv_file_name']
        assert_raises(ValueError, treecorr.Catalog, file_name, config, z1_col='0')
        assert_raises(ValueError, treecorr.Catalog, file_name, config, z2_col='0')
        assert_raises(ValueError, treecorr.Catalog, file_name, config, v1_col='0')
        assert_raises(ValueError, treecorr.Catalog, file_name, config, v2_col='0')
        assert_raises(ValueError, treecorr.Catalog, file_name, config, t1_col='0')
        assert_raises(ValueError, treecorr.Catalog, file_name, config, t2_col='0')
        assert_raises(ValueError, treecorr.Catalog, file_name, config, q1_col='0')
        assert_raises(ValueError, treecorr.Catalog, file_name, config, q2_col='0')

    assert_raises(ValueError, treecorr.Catalog, file_name, config, every_nth=0)
    assert_raises(ValueError, treecorr.Catalog, file_name, config, every_nth=-10)

@timer
def test_ext():
    try:
        import fitsio
    except ImportError:
        print('Skip test_ext, since fitsio not installed.')
        return

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
    z1 = rng.normal(0,0.1, (ngal,) )
    z2 = rng.normal(0,0.1, (ngal,) )
    v1 = rng.normal(0,0.1, (ngal,) )
    v2 = rng.normal(0,0.1, (ngal,) )
    g1 = rng.normal(0,0.1, (ngal,) )
    g2 = rng.normal(0,0.1, (ngal,) )
    t1 = rng.normal(0,0.1, (ngal,) )
    t2 = rng.normal(0,0.1, (ngal,) )
    q1 = rng.normal(0,0.1, (ngal,) )
    q2 = rng.normal(0,0.1, (ngal,) )
    patch = np.arange(ngal) % 5

    data = [x,y,z,ra,dec,r,flag,w,wpos,k,z1,z2,v1,v2,g1,g2,t1,t2,q1,q2]
    names = ['x','y','z','ra','dec','r','flag','w','wpos',
             'k','z1','z2','v1','v2','g1','g2','t1','t2','q1','q2']

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
                            k_col='k', z1_col='z1', z2_col='z2',
                            v1_col='v1', v2_col='v2', g1_col='g1', g2_col='g2',
                            t1_col='t1', t2_col='t2', q1_col='q1', q2_col='q2',
                            ext=1)
    cat2 = treecorr.Catalog(fname, allow_xyz=True,
                            x_col='x', y_col='y', z_col='z',
                            ra_col='ra', dec_col='dec', r_col='r',
                            ra_units='rad', dec_units='rad',
                            w_col='w', wpos_col='wpos', flag_col='flag',
                            k_col='k', z1_col='z1', z2_col='z2',
                            v1_col='v1', v2_col='v2', g1_col='g1', g2_col='g2',
                            t1_col='t1', t2_col='t2', q1_col='q1', q2_col='q2',
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
                            k_col='k', z1_col='z1', z2_col='z2',
                            v1_col='v1', v2_col='v2', g1_col='g1', g2_col='g2',
                            t1_col='t1', t2_col='t2', q1_col='q1', q2_col='q2',
                            ext=6)
    np.testing.assert_array_equal(cat6.x[use], cat1.x)
    np.testing.assert_array_equal(cat6.y[use], cat1.y)
    np.testing.assert_array_equal(cat6.z[use], cat1.z)
    np.testing.assert_array_equal(cat6.k[use], cat1.k)
    np.testing.assert_array_equal(cat6.z1[use], cat1.z1)
    np.testing.assert_array_equal(cat6.z2[use], cat1.z2)
    np.testing.assert_array_equal(cat6.v1[use], cat1.v1)
    np.testing.assert_array_equal(cat6.v2[use], cat1.v2)
    np.testing.assert_array_equal(cat6.g1[use], cat1.g1)
    np.testing.assert_array_equal(cat6.g2[use], cat1.g2)
    np.testing.assert_array_equal(cat6.t1[use], cat1.t1)
    np.testing.assert_array_equal(cat6.t2[use], cat1.t2)
    np.testing.assert_array_equal(cat6.q1[use], cat1.q1)
    np.testing.assert_array_equal(cat6.q2[use], cat1.q2)

    cat7 = treecorr.Catalog(fname, allow_xyz=True,
                            x_col='x', y_col='y', z_col='z',
                            ra_col='ra', dec_col='dec', r_col='r',
                            ra_units='rad', dec_units='rad',
                            w_col='w', wpos_col='wpos', flag_col='flag',
                            k_col='k', z1_col='z1', z2_col='z2',
                            v1_col='v1', v2_col='v2', g1_col='g1', g2_col='g2',
                            t1_col='t1', t2_col='t2', q1_col='q1', q2_col='q2',
                            ext=8)
    assert cat7 != cat1  # This one has all the column names wrong.

    cat8 = treecorr.Catalog(fname, allow_xyz=True,
                            x_col='x', y_col='y', z_col='z',
                            ra_col='ra', dec_col='dec', r_col='r',
                            ra_units='rad', dec_units='rad',
                            w_col='w', wpos_col='wpos', flag_col='flag',
                            k_col='k', z1_col='z1', z2_col='z2',
                            v1_col='v1', v2_col='v2', g1_col='g1', g2_col='g2',
                            t1_col='t1', t2_col='t2', q1_col='q1', q2_col='q2',
                            ext=-1)
    assert cat8 == cat7  # -1 is allowed and means the last one.

    cat9 = treecorr.Catalog(fname, allow_xyz=True,
                            x_col='x', y_col='y', z_col='z',
                            ra_col='ra', dec_col='dec', r_col='r',
                            ra_units='rad', dec_units='rad',
                            w_col='w', wpos_col='wpos', flag_col='flag',
                            k_col='k', z1_col='z1', z2_col='z2',
                            v1_col='v1', v2_col='v2', g1_col='g1', g2_col='g2',
                            t1_col='t1', t2_col='t2', q1_col='q1', q2_col='q2',
                            x_ext=1, y_ext=1, z_ext=1,
                            ra_ext=2, dec_ext=1, r_ext=2,
                            w_ext=1, wpos_ext=2, flag_ext=1,
                            k_ext=1, z1_ext=1, z2_ext=2,
                            v1_ext=1, v2_ext=2, g1_ext=1, g2_ext=2,
                            t1_ext=2, t2_ext=1, q1_ext=2, q2_ext=1)
    assert cat9 == cat1

    cat10 = treecorr.Catalog(fname, allow_xyz=True,
                             x_col='x', y_col='y', z_col='z',
                             ra_col='ra', dec_col='dec', r_col='r',
                             ra_units='rad', dec_units='rad',
                             w_col='w', wpos_col='wpos', flag_col='flag',
                             k_col='k', z1_col='z1', z2_col='z2',
                             v1_col='v1', v2_col='v2', g1_col='g1', g2_col='g2',
                             t1_col='t1', t2_col='t2', q1_col='q1', q2_col='q2',
                             x_ext=3, y_ext=3, z_ext=3,
                             ra_ext=4, dec_ext=4, r_ext=4,
                             w_ext=5, wpos_ext=5, flag_ext=5,
                             k_ext=6, z1_ext=6, z2_ext=6,
                             v1_ext=6, v2_ext=6, g1_ext=6, g2_ext=6,
                             t1_ext=6, t2_ext=6, q1_ext=6, q2_ext=6)
    assert cat10 == cat1

    # Not all columns in given ext
    with assert_raises(ValueError):
        treecorr.Catalog(fname, allow_xyz=True,
                         x_col='x', y_col='y', z_col='z',
                         ra_col='ra', dec_col='dec', r_col='r',
                         ra_units='rad', dec_units='rad',
                         w_col='w', wpos_col='wpos', flag_col='flag',
                         k_col='k', z1_col='z1', z2_col='z2',
                         v1_col='v1', v2_col='v2', g1_col='g1', g2_col='g2',
                         t1_col='t1', t2_col='t2', q1_col='q1', q2_col='q2',
                         ext=3)

    # Invalid ext
    with assert_raises(ValueError):
        treecorr.Catalog(fname, allow_xyz=True,
                         x_col='x', y_col='y', z_col='z',
                         ra_col='ra', dec_col='dec', r_col='r',
                         ra_units='rad', dec_units='rad',
                         w_col='w', wpos_col='wpos', flag_col='flag',
                         k_col='k', z1_col='z1', z2_col='z2',
                         v1_col='v1', v2_col='v2', g1_col='g1', g2_col='g2',
                         t1_col='t1', t2_col='t2', q1_col='q1', q2_col='q2',
                         ext=9)
    with assert_raises(ValueError):
        treecorr.Catalog(fname, allow_xyz=True,
                         x_col='x', y_col='y', z_col='z',
                         ra_col='ra', dec_col='dec', r_col='r',
                         ra_units='rad', dec_units='rad',
                         w_col='w', wpos_col='wpos', flag_col='flag',
                         k_col='k', z1_col='z1', z2_col='z2',
                         v1_col='v1', v2_col='v2', g1_col='g1', g2_col='g2',
                         t1_col='t1', t2_col='t2', q1_col='q1', q2_col='q2',
                         ext=0)
    with assert_raises(ValueError):
        treecorr.Catalog(fname, allow_xyz=True,
                         x_col='x', y_col='y', z_col='z',
                         ra_col='ra', dec_col='dec', r_col='r',
                         ra_units='rad', dec_units='rad',
                         w_col='w', wpos_col='wpos', flag_col='flag',
                         k_col='k', z1_col='z1', z2_col='z2',
                         v1_col='v1', v2_col='v2', g1_col='g1', g2_col='g2',
                         t1_col='t1', t2_col='t2', q1_col='q1', q2_col='q2',
                         ext=-20)

    # Not all columns in given ext
    with assert_raises(ValueError):
        treecorr.Catalog(fname,
                         ra_col='ra', dec_col='dec',
                         ra_units='rad', dec_units='rad',
                         k_col='k', z1_col='z1', z2_col='z2',
                         v1_col='v1', v2_col='v2', g1_col='g1', g2_col='g2',
                         t1_col='t1', t2_col='t2', q1_col='q1', q2_col='q2',
                         ext=6)

    # Position columns required
    with assert_raises(ValueError):
        treecorr.Catalog(fname,
                         k_col='k', z1_col='z1', z2_col='z2',
                         v1_col='v1', v2_col='v2', g1_col='g1', g2_col='g2',
                         t1_col='t1', t2_col='t2', q1_col='q1', q2_col='q2',
                         ext=6)

    # Missing units
    with assert_raises(TypeError):
        treecorr.Catalog(fname, allow_xyz=True,
                         x_col='x', y_col='y', z_col='z',
                         ra_col='ra', dec_col='dec', r_col='r',
                         w_col='w', wpos_col='wpos', flag_col='flag',
                         k_col='k', z1_col='z1', z2_col='z2',
                         v1_col='v1', v2_col='v2', g1_col='g1', g2_col='g2',
                         t1_col='t1', t2_col='t2', q1_col='q1', q2_col='q2',
                         ext=1)
    with assert_raises(TypeError):
        treecorr.Catalog(fname, allow_xyz=True,
                         x_col='x', y_col='y', z_col='z',
                         ra_col='ra', dec_col='dec', r_col='r',
                         ra_units='rad',
                         w_col='w', wpos_col='wpos', flag_col='flag',
                         k_col='k', z1_col='z1', z2_col='z2',
                         v1_col='v1', v2_col='v2', g1_col='g1', g2_col='g2',
                         t1_col='t1', t2_col='t2', q1_col='q1', q2_col='q2',
                         ext=1)
    with assert_raises(TypeError):
        treecorr.Catalog(fname, allow_xyz=True,
                         x_col='x', y_col='y', z_col='z',
                         ra_col='ra', dec_col='dec', r_col='r',
                         dec_units='rad',
                         w_col='w', wpos_col='wpos', flag_col='flag',
                         k_col='k', z1_col='z1', z2_col='z2',
                         v1_col='v1', v2_col='v2', g1_col='g1', g2_col='g2',
                         t1_col='t1', t2_col='t2', q1_col='q1', q2_col='q2',
                         ext=1)


@timer
def test_direct():

    nobj = 5000
    rng = np.random.RandomState(8675309)
    x = rng.random_sample(nobj)
    y = rng.random_sample(nobj)
    ra = rng.random_sample(nobj)
    dec = rng.random_sample(nobj)
    w = rng.random_sample(nobj)
    k = rng.random_sample(nobj)
    z1 = rng.random_sample(nobj)
    z2 = rng.random_sample(nobj)
    v1 = rng.random_sample(nobj)
    v2 = rng.random_sample(nobj)
    g1 = rng.random_sample(nobj)
    g2 = rng.random_sample(nobj)
    t1 = rng.random_sample(nobj)
    t2 = rng.random_sample(nobj)
    q1 = rng.random_sample(nobj)
    q2 = rng.random_sample(nobj)

    cat1 = treecorr.Catalog(x=x, y=y, w=w, k=k, z1=z1, z2=z2,
                            v1=v1, v2=v2, g1=g1, g2=g2,
                            t1=t1, t2=t2, q1=q1, q2=q2)
    np.testing.assert_almost_equal(cat1.x, x)
    np.testing.assert_almost_equal(cat1.y, y)
    np.testing.assert_almost_equal(cat1.w, w)
    np.testing.assert_almost_equal(cat1.k, k)
    np.testing.assert_almost_equal(cat1.z1, z1)
    np.testing.assert_almost_equal(cat1.z2, z2)
    np.testing.assert_almost_equal(cat1.v1, v1)
    np.testing.assert_almost_equal(cat1.v2, v2)
    np.testing.assert_almost_equal(cat1.g1, g1)
    np.testing.assert_almost_equal(cat1.g2, g2)
    np.testing.assert_almost_equal(cat1.t1, t1)
    np.testing.assert_almost_equal(cat1.t2, t2)
    np.testing.assert_almost_equal(cat1.q1, q1)
    np.testing.assert_almost_equal(cat1.q2, q2)

    cat2 = treecorr.Catalog(ra=ra, dec=dec, w=w, k=k, z1=z1, z2=z2,
                            v1=v1, v2=v2, g1=g1, g2=g2, t1=t1, t2=t2, q1=q1, q2=q2,
                            ra_units='hours', dec_units='degrees')
    np.testing.assert_almost_equal(cat2.ra, ra * coord.hours / coord.radians)
    np.testing.assert_almost_equal(cat2.dec, dec * coord.degrees / coord.radians)
    np.testing.assert_almost_equal(cat2.w, w)
    np.testing.assert_almost_equal(cat2.k, k)
    np.testing.assert_almost_equal(cat2.z1, z1)
    np.testing.assert_almost_equal(cat2.z2, z2)
    np.testing.assert_almost_equal(cat2.v1, v1)
    np.testing.assert_almost_equal(cat2.v2, v2)
    np.testing.assert_almost_equal(cat2.g1, g1)
    np.testing.assert_almost_equal(cat2.g2, g2)
    np.testing.assert_almost_equal(cat2.t1, t1)
    np.testing.assert_almost_equal(cat2.t2, t2)
    np.testing.assert_almost_equal(cat2.q1, q1)
    np.testing.assert_almost_equal(cat2.q2, q2)

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
    assert_raises(TypeError, treecorr.Catalog, x=x, y=y, z1=z1)
    assert_raises(TypeError, treecorr.Catalog, x=x, y=y, z2=z2)
    assert_raises(TypeError, treecorr.Catalog, x=x, y=y, v1=v1)
    assert_raises(TypeError, treecorr.Catalog, x=x, y=y, v2=v2)
    assert_raises(TypeError, treecorr.Catalog, x=x, y=y, g1=g1)
    assert_raises(TypeError, treecorr.Catalog, x=x, y=y, g2=g2)
    assert_raises(TypeError, treecorr.Catalog, x=x, y=y, t1=t1)
    assert_raises(TypeError, treecorr.Catalog, x=x, y=y, t2=t2)
    assert_raises(TypeError, treecorr.Catalog, x=x, y=y, q1=q1)
    assert_raises(TypeError, treecorr.Catalog, x=x, y=y, q2=q2)
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
    assert_raises(ValueError, treecorr.Catalog, x=x, y=y, w=w, k=k[4:])
    assert_raises(ValueError, treecorr.Catalog, x=x, y=y, w=w, z1=z1[4:], z2=z2[4:])
    assert_raises(ValueError, treecorr.Catalog, x=x, y=y, w=w, z1=z1[4:], z2=z2)
    assert_raises(ValueError, treecorr.Catalog, x=x, y=y, w=w, z1=z1, z2=z2[4:])
    assert_raises(ValueError, treecorr.Catalog, x=x, y=y, w=w, v1=v1[4:], v2=v2[4:])
    assert_raises(ValueError, treecorr.Catalog, x=x, y=y, w=w, v1=v1[4:], v2=v2)
    assert_raises(ValueError, treecorr.Catalog, x=x, y=y, w=w, v1=v1, v2=v2[4:])
    assert_raises(ValueError, treecorr.Catalog, x=x, y=y, w=w, g1=g1[4:], g2=g2[4:])
    assert_raises(ValueError, treecorr.Catalog, x=x, y=y, w=w, g1=g1[4:], g2=g2)
    assert_raises(ValueError, treecorr.Catalog, x=x, y=y, w=w, g1=g1, g2=g2[4:])
    assert_raises(ValueError, treecorr.Catalog, x=x, y=y, w=w, t1=t1[4:], t2=t2[4:])
    assert_raises(ValueError, treecorr.Catalog, x=x, y=y, w=w, t1=t1[4:], t2=t2)
    assert_raises(ValueError, treecorr.Catalog, x=x, y=y, w=w, t1=t1, t2=t2[4:])
    assert_raises(ValueError, treecorr.Catalog, x=x, y=y, w=w, q1=q1[4:], q2=q2[4:])
    assert_raises(ValueError, treecorr.Catalog, x=x, y=y, w=w, q1=q1[4:], q2=q2)
    assert_raises(ValueError, treecorr.Catalog, x=x, y=y, w=w, q1=q1, q2=q2[4:])
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
    allk = []
    allz1 = []
    allz2 = []
    allv1 = []
    allv2 = []
    allg1 = []
    allg2 = []
    allt1 = []
    allt2 = []
    allq1 = []
    allq2 = []
    for i in range(10):
        x = rng.random_sample(nobj)
        y = rng.random_sample(nobj)
        k = rng.random_sample(nobj) - 0.5
        z1 = rng.random_sample(nobj) - 0.5
        z2 = rng.random_sample(nobj) - 0.5
        v1 = rng.random_sample(nobj) - 0.5
        v2 = rng.random_sample(nobj) - 0.5
        g1 = rng.random_sample(nobj) - 0.5
        g2 = rng.random_sample(nobj) - 0.5
        t1 = rng.random_sample(nobj) - 0.5
        t2 = rng.random_sample(nobj) - 0.5
        q1 = rng.random_sample(nobj) - 0.5
        q2 = rng.random_sample(nobj) - 0.5
        cat = treecorr.Catalog(x=x, y=y, g1=g1, g2=g2, k=k, z1=z1, z2=z2, v1=v1, v2=v2,
                               t1=t1, t2=t2, q1=q1, q2=q2)
        vark = np.var(k, ddof=0)
        varz = (np.var(z1, ddof=0) + np.var(z2, ddof=0))/2
        varv = (np.var(v1, ddof=0) + np.var(v2, ddof=0))/2
        varg = (np.var(g1, ddof=0) + np.var(g2, ddof=0))/2
        vart = (np.var(t1, ddof=0) + np.var(t2, ddof=0))/2
        varq = (np.var(q1, ddof=0) + np.var(q2, ddof=0))/2
        assert np.isclose(cat.vark, vark)
        assert np.isclose(cat.varg, varg)
        assert np.isclose(cat.varv, varv)
        assert np.isclose(cat.vart, vart)
        assert np.isclose(cat.varq, varq)
        assert np.isclose(treecorr.calculateVarK(cat), vark)
        assert np.isclose(treecorr.calculateVarK([cat]), vark)
        assert np.isclose(treecorr.calculateVarG(cat), varg)
        assert np.isclose(treecorr.calculateVarG([cat]), varg)
        assert np.isclose(treecorr.calculateVarV(cat), varv)
        assert np.isclose(treecorr.calculateVarV([cat]), varv)
        assert np.isclose(treecorr.calculateVarT(cat), vart)
        assert np.isclose(treecorr.calculateVarT([cat]), vart)
        assert np.isclose(treecorr.calculateVarQ(cat), varq)
        assert np.isclose(treecorr.calculateVarQ([cat]), varq)
        cats.append(cat)
        allk.extend(k)
        allz1.extend(z1)
        allz2.extend(z2)
        allv1.extend(v1)
        allv2.extend(v2)
        allg1.extend(g1)
        allg2.extend(g2)
        allt1.extend(t1)
        allt2.extend(t2)
        allq1.extend(q1)
        allq2.extend(q2)

    allk = np.array(allk)
    allz1 = np.array(allz1)
    allz2 = np.array(allz2)
    allv1 = np.array(allv1)
    allv2 = np.array(allv2)
    allg1 = np.array(allg1)
    allg2 = np.array(allg2)
    allt1 = np.array(allt1)
    allt2 = np.array(allt2)
    allq1 = np.array(allq1)
    allq2 = np.array(allq2)
    vark = np.var(allk, ddof=0)
    varz = (np.var(allz1, ddof=0) + np.var(allz2, ddof=0))/2
    varv = (np.var(allv1, ddof=0) + np.var(allv2, ddof=0))/2
    varg = (np.var(allg1, ddof=0) + np.var(allg2, ddof=0))/2
    vart = (np.var(allt1, ddof=0) + np.var(allt2, ddof=0))/2
    varq = (np.var(allq1, ddof=0) + np.var(allq2, ddof=0))/2
    assert np.isclose(treecorr.calculateVarK(cats), vark)
    assert np.isclose(treecorr.calculateVarG(cats), varg)
    assert np.isclose(treecorr.calculateVarV(cats), varv)
    assert np.isclose(treecorr.calculateVarT(cats), vart)
    assert np.isclose(treecorr.calculateVarQ(cats), varq)

    # Now with weights
    cats = []
    allk = []
    allz1 = []
    allz2 = []
    allv1 = []
    allv2 = []
    allg1 = []
    allg2 = []
    allt1 = []
    allt2 = []
    allq1 = []
    allq2 = []
    allw = []
    for i in range(10):
        x = rng.random_sample(nobj)
        y = rng.random_sample(nobj)
        w = rng.random_sample(nobj)
        k = rng.random_sample(nobj)
        z1 = rng.random_sample(nobj)
        z2 = rng.random_sample(nobj)
        v1 = rng.random_sample(nobj)
        v2 = rng.random_sample(nobj)
        g1 = rng.random_sample(nobj)
        g2 = rng.random_sample(nobj)
        t1 = rng.random_sample(nobj)
        t2 = rng.random_sample(nobj)
        q1 = rng.random_sample(nobj)
        q2 = rng.random_sample(nobj)
        cat = treecorr.Catalog(x=x, y=y, w=w, g1=g1, g2=g2, k=k, z1=z1, z2=z2,
                               v1=v1, v2=v2, t1=t1, t2=t2, q1=q1, q2=q2)
        meank = np.sum(w*k)/np.sum(w)
        vark = np.sum(w**2 * (k-meank)**2) / np.sum(w)
        meang1 = np.sum(w*g1)/np.sum(w)
        meang2 = np.sum(w*g2)/np.sum(w)
        varg = np.sum(w**2 * ((g1-meang1)**2 + (g2-meang2)**2)) / (2*np.sum(w))
        meanz1 = np.sum(w*z1)/np.sum(w)
        meanz2 = np.sum(w*z2)/np.sum(w)
        varz = np.sum(w**2 * ((z1-meanz1)**2 + (z2-meanz2)**2)) / (2*np.sum(w))
        meanv1 = np.sum(w*v1)/np.sum(w)
        meanv2 = np.sum(w*v2)/np.sum(w)
        varv = np.sum(w**2 * ((v1-meanv1)**2 + (v2-meanv2)**2)) / (2*np.sum(w))
        meant1 = np.sum(w*t1)/np.sum(w)
        meant2 = np.sum(w*t2)/np.sum(w)
        vart = np.sum(w**2 * ((t1-meant1)**2 + (t2-meant2)**2)) / (2*np.sum(w))
        meanq1 = np.sum(w*q1)/np.sum(w)
        meanq2 = np.sum(w*q2)/np.sum(w)
        varq = np.sum(w**2 * ((q1-meanq1)**2 + (q2-meanq2)**2)) / (2*np.sum(w))
        assert np.isclose(cat.vark, vark)
        assert np.isclose(cat.varg, varg)
        assert np.isclose(cat.varv, varv)
        assert np.isclose(cat.vart, vart)
        assert np.isclose(cat.varq, varq)
        assert np.isclose(treecorr.calculateVarK(cat), vark)
        assert np.isclose(treecorr.calculateVarK([cat]), vark)
        assert np.isclose(treecorr.calculateVarG(cat), varg)
        assert np.isclose(treecorr.calculateVarG([cat]), varg)
        assert np.isclose(treecorr.calculateVarV(cat), varv)
        assert np.isclose(treecorr.calculateVarV([cat]), varv)
        assert np.isclose(treecorr.calculateVarT(cat), vart)
        assert np.isclose(treecorr.calculateVarT([cat]), vart)
        assert np.isclose(treecorr.calculateVarQ(cat), varq)
        assert np.isclose(treecorr.calculateVarQ([cat]), varq)
        cats.append(cat)
        allk.extend(k)
        allz1.extend(z1)
        allz2.extend(z2)
        allv1.extend(v1)
        allv2.extend(v2)
        allg1.extend(g1)
        allg2.extend(g2)
        allt1.extend(t1)
        allt2.extend(t2)
        allq1.extend(q1)
        allq2.extend(q2)
        allw.extend(w)

    allk = np.array(allk)
    allz1 = np.array(allz1)
    allz2 = np.array(allz2)
    allv1 = np.array(allv1)
    allv2 = np.array(allv2)
    allg1 = np.array(allg1)
    allg2 = np.array(allg2)
    allt1 = np.array(allt1)
    allt2 = np.array(allt2)
    allq1 = np.array(allq1)
    allq2 = np.array(allq2)
    allw = np.array(allw)
    meank = np.sum(allw*allk)/np.sum(allw)
    vark = np.sum(allw**2 * (allk-meank)**2) / np.sum(allw)
    meang1 = np.sum(allw*allg1)/np.sum(allw)
    meang2 = np.sum(allw*allg2)/np.sum(allw)
    varg = np.sum(allw**2 * ((allg1-meang1)**2 + (allg2-meang2)**2)) / (2*np.sum(allw))
    meanz1 = np.sum(allw*allz1)/np.sum(allw)
    meanz2 = np.sum(allw*allz2)/np.sum(allw)
    varz = np.sum(allw**2 * ((allz1-meanz1)**2 + (allz2-meanz2)**2)) / (2*np.sum(allw))
    meanv1 = np.sum(allw*allv1)/np.sum(allw)
    meanv2 = np.sum(allw*allv2)/np.sum(allw)
    varv = np.sum(allw**2 * ((allv1-meanv1)**2 + (allv2-meanv2)**2)) / (2*np.sum(allw))
    meant1 = np.sum(allw*allt1)/np.sum(allw)
    meant2 = np.sum(allw*allt2)/np.sum(allw)
    vart = np.sum(allw**2 * ((allt1-meant1)**2 + (allt2-meant2)**2)) / (2*np.sum(allw))
    meanq1 = np.sum(allw*allq1)/np.sum(allw)
    meanq2 = np.sum(allw*allq2)/np.sum(allw)
    varq = np.sum(allw**2 * ((allq1-meanq1)**2 + (allq2-meanq2)**2)) / (2*np.sum(allw))
    assert np.isclose(treecorr.calculateVarK(cats), vark)
    assert np.isclose(treecorr.calculateVarG(cats), varg)
    assert np.isclose(treecorr.calculateVarV(cats), varv)
    assert np.isclose(treecorr.calculateVarT(cats), vart)
    assert np.isclose(treecorr.calculateVarQ(cats), varq)

    # With no g1,g2,k, varg=vark=0
    cat = treecorr.Catalog(x=x, y=y)
    assert cat.vark == 0
    assert cat.varg == 0
    assert cat.varv == 0
    assert cat.vart == 0
    assert cat.varq == 0
    cat = treecorr.Catalog(x=x, y=y, w=w)
    assert cat.vark == 0
    assert cat.varg == 0
    assert cat.varv == 0
    assert cat.vart == 0
    assert cat.varq == 0

    # If variances are specified on input, use them.
    cats = []
    allk = []
    allz1 = []
    allz2 = []
    allv1 = []
    allv2 = []
    allg1 = []
    allg2 = []
    allt1 = []
    allt2 = []
    allq1 = []
    allq2 = []
    for i in range(10):
        x = rng.random_sample(nobj)
        y = rng.random_sample(nobj)
        k = rng.random_sample(nobj) - 0.5
        z1 = rng.random_sample(nobj) - 0.5
        z2 = rng.random_sample(nobj) - 0.5
        v1 = rng.random_sample(nobj) - 0.5
        v2 = rng.random_sample(nobj) - 0.5
        g1 = rng.random_sample(nobj) - 0.5
        g2 = rng.random_sample(nobj) - 0.5
        t1 = rng.random_sample(nobj) - 0.5
        t2 = rng.random_sample(nobj) - 0.5
        q1 = rng.random_sample(nobj) - 0.5
        q2 = rng.random_sample(nobj) - 0.5
        vark = np.var(k, ddof=0)
        varz = (np.var(z1, ddof=0) + np.var(z2, ddof=0))/2
        varv = (np.var(v1, ddof=0) + np.var(v2, ddof=0))/2
        varg = (np.var(g1, ddof=0) + np.var(g2, ddof=0))/2
        vart = (np.var(t1, ddof=0) + np.var(t2, ddof=0))/2
        varq = (np.var(q1, ddof=0) + np.var(q2, ddof=0))/2
        cat = treecorr.Catalog(x=x, y=y, g1=g1, g2=g2, k=k, z1=z1, z2=z2,
                               v1=v1, v2=v2, t1=t1, t2=t2, q1=q1, q2=q2,
                               vark=3*vark, varz=vark, varg=5*varg, varv=7*varv,
                               vart=4*vart, varq=8*varq)
        assert np.isclose(cat.vark, 3*vark)
        assert np.isclose(cat.varz, vark)
        assert np.isclose(cat.varg, 5*varg)
        assert np.isclose(cat.varv, 7*varv)
        assert np.isclose(cat.vart, 4*vart)
        assert np.isclose(cat.varq, 8*varq)
        assert np.isclose(treecorr.calculateVarK(cat), 3*vark)
        assert np.isclose(treecorr.calculateVarK([cat]), 3*vark)
        assert np.isclose(treecorr.calculateVarZ(cat), vark)
        assert np.isclose(treecorr.calculateVarZ([cat]), vark)
        assert np.isclose(treecorr.calculateVarG(cat), 5*varg)
        assert np.isclose(treecorr.calculateVarG([cat]), 5*varg)
        assert np.isclose(treecorr.calculateVarV(cat), 7*varv)
        assert np.isclose(treecorr.calculateVarV([cat]), 7*varv)
        assert np.isclose(treecorr.calculateVarT(cat), 4*vart)
        assert np.isclose(treecorr.calculateVarT([cat]), 4*vart)
        assert np.isclose(treecorr.calculateVarQ(cat), 8*varq)
        assert np.isclose(treecorr.calculateVarQ([cat]), 8*varq)
        cats.append(cat)
        allk.extend(k)
        allz1.extend(z1)
        allz2.extend(z2)
        allv1.extend(v1)
        allv2.extend(v2)
        allg1.extend(g1)
        allg2.extend(g2)
        allt1.extend(t1)
        allt2.extend(t2)
        allq1.extend(q1)
        allq2.extend(q2)

    allk = np.array(allk)
    allg1 = np.array(allg1)
    allg2 = np.array(allg2)
    allv1 = np.array(allv1)
    allv2 = np.array(allv2)
    allt1 = np.array(allt1)
    allt2 = np.array(allt2)
    allq1 = np.array(allq1)
    allq2 = np.array(allq2)
    vark = np.var(allk, ddof=0)
    varz = (np.var(allz1, ddof=0) + np.var(allz2, ddof=0))/2
    varv = (np.var(allv1, ddof=0) + np.var(allv2, ddof=0))/2
    varg = (np.var(allg1, ddof=0) + np.var(allg2, ddof=0))/2
    vart = (np.var(allt1, ddof=0) + np.var(allt2, ddof=0))/2
    varq = (np.var(allq1, ddof=0) + np.var(allq2, ddof=0))/2
    # These aren't exactly the same because the means in each catalog are slightly different.
    # But it's pretty close.
    assert np.isclose(treecorr.calculateVarK(cats), 3*vark, rtol=1.e-3)
    assert np.isclose(treecorr.calculateVarG(cats), 5*varg, rtol=1.e-3)
    assert np.isclose(treecorr.calculateVarV(cats), 7*varv, rtol=1.e-3)
    assert np.isclose(treecorr.calculateVarT(cats), 4*vart, rtol=1.e-3)
    assert np.isclose(treecorr.calculateVarQ(cats), 8*varq, rtol=1.e-3)

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
    k = rng.random_sample(nobj)
    z1 = rng.random_sample(nobj)
    z2 = rng.random_sample(nobj)
    v1 = rng.random_sample(nobj)
    v2 = rng.random_sample(nobj)
    g1 = rng.random_sample(nobj)
    g2 = rng.random_sample(nobj)
    t1 = rng.random_sample(nobj)
    t2 = rng.random_sample(nobj)
    q1 = rng.random_sample(nobj)
    q2 = rng.random_sample(nobj)

    # Turn 1% of these values into NaN
    x[rng.choice(nobj, nobj//100)] = np.nan
    y[rng.choice(nobj, nobj//100)] = np.nan
    z[rng.choice(nobj, nobj//100)] = np.nan
    ra[rng.choice(nobj, nobj//100)] = np.nan
    dec[rng.choice(nobj, nobj//100)] = np.nan
    r[rng.choice(nobj, nobj//100)] = np.nan
    w[rng.choice(nobj, nobj//100)] = np.nan
    wpos[rng.choice(nobj, nobj//100)] = np.nan
    k[rng.choice(nobj, nobj//100)] = np.nan
    z1[rng.choice(nobj, nobj//100)] = np.nan
    z2[rng.choice(nobj, nobj//100)] = np.nan
    v1[rng.choice(nobj, nobj//100)] = np.nan
    v2[rng.choice(nobj, nobj//100)] = np.nan
    g1[rng.choice(nobj, nobj//100)] = np.nan
    g2[rng.choice(nobj, nobj//100)] = np.nan
    t1[rng.choice(nobj, nobj//100)] = np.nan
    t2[rng.choice(nobj, nobj//100)] = np.nan
    q1[rng.choice(nobj, nobj//100)] = np.nan
    q2[rng.choice(nobj, nobj//100)] = np.nan
    print('x is nan at ',np.where(np.isnan(x)))
    print('y is nan at ',np.where(np.isnan(y)))
    print('z is nan at ',np.where(np.isnan(z)))
    print('ra is nan at ',np.where(np.isnan(ra)))
    print('dec is nan at ',np.where(np.isnan(dec)))
    print('w is nan at ',np.where(np.isnan(w)))
    print('wpos is nan at ',np.where(np.isnan(wpos)))
    print('k is nan at ',np.where(np.isnan(k)))
    print('z1 is nan at ',np.where(np.isnan(z1)))
    print('z2 is nan at ',np.where(np.isnan(z2)))
    print('v1 is nan at ',np.where(np.isnan(v1)))
    print('v2 is nan at ',np.where(np.isnan(v2)))
    print('g1 is nan at ',np.where(np.isnan(g1)))
    print('g2 is nan at ',np.where(np.isnan(g2)))
    print('t1 is nan at ',np.where(np.isnan(t1)))
    print('t2 is nan at ',np.where(np.isnan(t2)))
    print('q1 is nan at ',np.where(np.isnan(q1)))
    print('q2 is nan at ',np.where(np.isnan(q2)))

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
        cat2 = treecorr.Catalog(ra=ra, dec=dec, r=r, w=w, wpos=wpos, z1=z1, z2=z2,
                                v1=v1, v2=v2, g1=g1, g2=g2, t1=t1, t2=t2, q1=q1, q2=q2,
                                ra_units='hours', dec_units='degrees', logger=cl.logger,
                                keep_zero_weight=True)
    assert "NaNs found in ra column." in cl.output
    assert "NaNs found in dec column." in cl.output
    assert "NaNs found in r column." in cl.output
    assert "NaNs found in z1 column." in cl.output
    assert "NaNs found in z2 column." in cl.output
    assert "NaNs found in v1 column." in cl.output
    assert "NaNs found in v2 column." in cl.output
    assert "NaNs found in g1 column." in cl.output
    assert "NaNs found in g2 column." in cl.output
    assert "NaNs found in t1 column." in cl.output
    assert "NaNs found in t2 column." in cl.output
    assert "NaNs found in q1 column." in cl.output
    assert "NaNs found in q2 column." in cl.output
    assert "NaNs found in w column." in cl.output
    assert "NaNs found in wpos column." in cl.output
    mask = (np.isnan(ra) | np.isnan(dec) | np.isnan(r) | np.isnan(z1) | np.isnan(z2) |
            np.isnan(v1) | np.isnan(v2) | np.isnan(g1) | np.isnan(g2) |
            np.isnan(t1) | np.isnan(t2) | np.isnan(q1) | np.isnan(q2) |
            np.isnan(wpos) | np.isnan(w))
    good = ~mask
    assert cat2.ntot == nobj
    assert cat2.nobj == np.sum(good)
    np.testing.assert_almost_equal(cat2.ra[good], ra[good] * coord.hours / coord.radians)
    np.testing.assert_almost_equal(cat2.dec[good], dec[good] * coord.degrees / coord.radians)
    np.testing.assert_almost_equal(cat2.r[good], r[good])
    np.testing.assert_almost_equal(cat2.w[good], w[good])
    np.testing.assert_almost_equal(cat2.wpos[good], wpos[good])
    np.testing.assert_almost_equal(cat2.z1[good], z1[good])
    np.testing.assert_almost_equal(cat2.z2[good], z2[good])
    np.testing.assert_almost_equal(cat2.v1[good], v1[good])
    np.testing.assert_almost_equal(cat2.v2[good], v2[good])
    np.testing.assert_almost_equal(cat2.g1[good], g1[good])
    np.testing.assert_almost_equal(cat2.g2[good], g2[good])
    np.testing.assert_almost_equal(cat2.t1[good], t1[good])
    np.testing.assert_almost_equal(cat2.t2[good], t2[good])
    np.testing.assert_almost_equal(cat2.q1[good], q1[good])
    np.testing.assert_almost_equal(cat2.q2[good], q2[good])
    np.testing.assert_almost_equal(cat2.w[mask], 0)

    # If no weight column, it is make automatically to deal with Nans.
    with CaptureLog() as cl:
        cat3 = treecorr.Catalog(x=x, y=y, g1=g1, g2=g2, z1=z1, z2=z2, v1=v1, v2=v2,
                                t1=t1, t2=t2, q1=q1, q2=q2,
                                logger=cl.logger, keep_zero_weight=True)
    mask = (np.isnan(x) | np.isnan(y) | np.isnan(g1) | np.isnan(g2) |
            np.isnan(z1) | np.isnan(z2) | np.isnan(v1) | np.isnan(v2) |
            np.isnan(t1) | np.isnan(t2) | np.isnan(q1) | np.isnan(q2))
    good = ~mask
    assert cat3.ntot == nobj
    assert cat3.nobj == np.sum(good)
    np.testing.assert_almost_equal(cat3.x[good], x[good])
    np.testing.assert_almost_equal(cat3.y[good], y[good])
    np.testing.assert_almost_equal(cat3.w[good], 1.)
    np.testing.assert_almost_equal(cat3.z1[good], z1[good])
    np.testing.assert_almost_equal(cat3.z2[good], z2[good])
    np.testing.assert_almost_equal(cat3.v1[good], v1[good])
    np.testing.assert_almost_equal(cat3.v2[good], v2[good])
    np.testing.assert_almost_equal(cat3.g1[good], g1[good])
    np.testing.assert_almost_equal(cat3.g2[good], g2[good])
    np.testing.assert_almost_equal(cat3.t1[good], t1[good])
    np.testing.assert_almost_equal(cat3.t2[good], t2[good])
    np.testing.assert_almost_equal(cat3.q1[good], q1[good])
    np.testing.assert_almost_equal(cat3.q2[good], q2[good])
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
        'angle_slop': 1.,
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

    try:
        # This doesn't exist on all systems.
        float128 = np.float128
    except AttributeError:
        float128 = float

    source_data = np.array([
            (0.0380569697547, 0.0142782758818, 0.330845443464, -0.111049332655),
            (-0.0261291090735, 0.0863787933931, 0.122954685209, 0.40260430406),
            (-0.0261291090735, 0.0863787933931, 0.122954685209, 0.40260430406),
            (0.125086697534, 0.0283621046495, -0.208159531309, 0.142491564101),
            (0.0457709426026, -0.0299249486373, -0.0406555089425, 0.24515956887),
            (-0.00338578248926, 0.0460291122935, 0.363057738173, -0.524536297555)],
            dtype=[('ra', None), ('dec', np.float64), ('g1', np.float32),
                   ('g2', float128)])

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
    cat1.write(os.path.join('output','cat1.dat'), precision=20)
    cat1_asc = treecorr.Catalog(os.path.join('output','cat1.dat'), file_type='ASCII',
                                x_col=1, y_col=2, z_col=3)
    np.testing.assert_almost_equal(cat1_asc.x, x)
    np.testing.assert_almost_equal(cat1_asc.y, y)
    np.testing.assert_almost_equal(cat1_asc.z, z)

    cat2.write(os.path.join('output','cat2.dat'), file_type='ASCII')
    cat2_asc = treecorr.Catalog(os.path.join('output','cat2.dat'), ra_col=1, dec_col=2,
                                r_col=3, w_col=4, k_col=5, g1_col=6, g2_col=7,
                                ra_units='rad', dec_units='rad')
    np.testing.assert_almost_equal(cat2_asc.ra, ra)
    np.testing.assert_almost_equal(cat2_asc.dec, dec)
    np.testing.assert_almost_equal(cat2_asc.r, r)
    np.testing.assert_almost_equal(cat2_asc.w, w)
    np.testing.assert_almost_equal(cat2_asc.g1, g1)
    np.testing.assert_almost_equal(cat2_asc.g2, g2)
    np.testing.assert_almost_equal(cat2_asc.k, k)

    cat2r_asc = treecorr.Catalog(os.path.join('output','cat2.dat'), ra_col=1, dec_col=2,
                                 r_col=3, w_col=4, k_col=5, g1_col=6, g2_col=7,
                                 ra_units='rad', dec_units='rad', is_rand=True)
    np.testing.assert_almost_equal(cat2r_asc.ra, ra)
    np.testing.assert_almost_equal(cat2r_asc.dec, dec)
    np.testing.assert_almost_equal(cat2r_asc.r, r)
    np.testing.assert_almost_equal(cat2r_asc.w, w)
    assert cat2r_asc.g1 is None
    assert cat2r_asc.g2 is None
    assert cat2r_asc.k is None

    # Test FITS output
    try:
        import fitsio
    except ImportError:
        print('Skipping fits write tests, since h5py not installed.')
        pass
    else:
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

        cat2r_fits = treecorr.Catalog(os.path.join('output','cat2.fits'),
                                      ra_col='ra', dec_col='dec',
                                      r_col='r', w_col='w', g1_col='g1', g2_col='g2', k_col='k',
                                      ra_units='rad', dec_units='rad', file_type='FITS',
                                      is_rand=True)
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
    else:
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

        cat2r_hdf5 = treecorr.Catalog(os.path.join('output','cat2.hdf'),
                                      ra_col='ra', dec_col='dec',
                                      r_col='r', w_col='w', g1_col='g1', g2_col='g2', k_col='k',
                                      ra_units='rad', dec_units='rad', file_type='HDF5',
                                      is_rand=True)
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
    z1 = rng.normal(0,s, (ngal,) )
    z2 = rng.normal(0,s, (ngal,) )
    v1 = rng.normal(0,s, (ngal,) )
    v2 = rng.normal(0,s, (ngal,) )
    g1 = rng.normal(0,s, (ngal,) )
    g2 = rng.normal(0,s, (ngal,) )
    t1 = rng.normal(0,s, (ngal,) )
    t2 = rng.normal(0,s, (ngal,) )
    q1 = rng.normal(0,s, (ngal,) )
    q2 = rng.normal(0,s, (ngal,) )

    cat1 = treecorr.Catalog(x=x, y=y, z=z, g1=g1, g2=g2, k=k, z1=z1, z2=z2,
                            v1=v1, v2=v2, t1=t1, t2=t2, q1=q1, q2=q2)
    cat2 = treecorr.Catalog(ra=ra, dec=dec, ra_units='hour', dec_units='deg',
                            w=w, g1=g1, g2=g2, k=k, z1=z1, z2=z2,
                            v1=v1, v2=v2, t1=t1, t2=t2, q1=q1, q2=q2)
    cat2.logger = None
    cat3 = treecorr.Catalog(x=x, y=y, g1=g1, g2=g2, k=k, w=w, z1=z1, z2=z2,
                            v1=v1, v2=v2, t1=t1, t2=t2, q1=q1, q2=q2)
    cat3 = cat3.copy()  # This tests that post-pickled catalog still works correctly.
    cat4 = treecorr.Catalog(x=x, y=y, w=w)
    logger = treecorr.config.setup_logger(1)

    assert cat1.field is None  # Before calling get*Field, this is None.
    assert cat2.field is None
    assert cat3.field is None

    t0 = time.time()
    nfield1 = cat1.getNField()
    nfield2 = cat2.getNField(min_size=0.01, max_size=1)
    nfield3 = cat3.getNField(min_size=1, max_size=300, logger=logger)
    t1 = time.time()
    nfield1b = cat1.getNField()
    nfield2b = cat2.getNField(min_size=0.01, max_size=1)
    nfield3b = cat3.getNField(min_size=1, max_size=300, logger=logger)
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
    if platform.python_implementation() != 'PyPy':
        assert t2-t1 <= t1-t0

    t0 = time.time()
    gfield1 = cat1.getGField()
    gfield2 = cat2.getGField(min_size=0.01, max_size=1)
    gfield3 = cat3.getGField(min_size=1, max_size=300, logger=logger)
    t1 = time.time()
    gfield1b = cat1.getGField()
    gfield2b = cat2.getGField(min_size=0.01, max_size=1)
    gfield3b = cat3.getGField(min_size=1, max_size=300, logger=logger)
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
    if platform.python_implementation() != 'PyPy':
        assert t2-t1 <= t1-t0

    t0 = time.time()
    kfield1 = cat1.getKField()
    kfield2 = cat2.getKField(min_size=0.01, max_size=1)
    kfield3 = cat3.getKField(min_size=1, max_size=300, logger=logger)
    t1 = time.time()
    kfield1b = cat1.getKField()
    kfield2b = cat2.getKField(min_size=0.01, max_size=1)
    kfield3b = cat3.getKField(min_size=1, max_size=300, logger=logger)
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
    if platform.python_implementation() != 'PyPy':
        assert t2-t1 <= t1-t0

    t0 = time.time()
    zfield1 = cat1.getZField()
    zfield2 = cat2.getZField(min_size=0.01, max_size=1)
    zfield3 = cat3.getZField(min_size=1, max_size=300, logger=logger)
    t1 = time.time()
    zfield1b = cat1.getZField()
    zfield2b = cat2.getZField(min_size=0.01, max_size=1)
    zfield3b = cat3.getZField(min_size=1, max_size=300, logger=logger)
    t2 = time.time()
    assert_raises(TypeError, cat4.getZField)
    assert cat1.zfields.count == 1
    assert cat2.zfields.count == 1
    assert cat3.zfields.count == 1
    assert cat1.field is zfield1
    assert cat2.field is zfield2
    assert cat3.field is zfield3
    assert zfield1b is zfield1
    assert zfield2b is zfield2
    assert zfield3b is zfield3
    print('zfield: ',t1-t0,t2-t1)
    if platform.python_implementation() != 'PyPy':
        assert t2-t1 <= t1-t0

    t0 = time.time()
    vfield1 = cat1.getVField()
    vfield2 = cat2.getVField(min_size=0.01, max_size=1)
    vfield3 = cat3.getVField(min_size=1, max_size=300, logger=logger)
    t1 = time.time()
    vfield1b = cat1.getVField()
    vfield2b = cat2.getVField(min_size=0.01, max_size=1)
    vfield3b = cat3.getVField(min_size=1, max_size=300, logger=logger)
    t2 = time.time()
    assert_raises(TypeError, cat4.getVField)
    assert cat1.vfields.count == 1
    assert cat2.vfields.count == 1
    assert cat3.vfields.count == 1
    assert cat1.field is vfield1
    assert cat2.field is vfield2
    assert cat3.field is vfield3
    assert vfield1b is vfield1
    assert vfield2b is vfield2
    assert vfield3b is vfield3
    print('vfield: ',t1-t0,t2-t1)
    if platform.python_implementation() != 'PyPy':
        assert t2-t1 <= t1-t0

    tfield1 = cat1.getTField()
    tfield2 = cat2.getTField(min_size=0.01, max_size=1)
    tfield3 = cat3.getTField(min_size=1, max_size=300, logger=logger)
    tfield1b = cat1.getTField()
    tfield2b = cat2.getTField(min_size=0.01, max_size=1)
    tfield3b = cat3.getTField(min_size=1, max_size=300, logger=logger)
    assert_raises(TypeError, cat4.getTField)
    assert cat1.tfields.count == 1
    assert cat2.tfields.count == 1
    assert cat3.tfields.count == 1
    assert cat1.field is tfield1
    assert cat2.field is tfield2
    assert cat3.field is tfield3
    assert tfield1b is tfield1
    assert tfield2b is tfield2
    assert tfield3b is tfield3

    qfield1 = cat1.getQField()
    qfield2 = cat2.getQField(min_size=0.01, max_size=1)
    qfield3 = cat3.getQField(min_size=1, max_size=300, logger=logger)
    qfield1b = cat1.getQField()
    qfield2b = cat2.getQField(min_size=0.01, max_size=1)
    qfield3b = cat3.getQField(min_size=1, max_size=300, logger=logger)
    assert_raises(TypeError, cat4.getQField)
    assert cat1.qfields.count == 1
    assert cat2.qfields.count == 1
    assert cat3.qfields.count == 1
    assert cat1.field is qfield1
    assert cat2.field is qfield2
    assert cat3.field is qfield3
    assert qfield1b is qfield1
    assert qfield2b is qfield2
    assert qfield3b is qfield3

    # By default, only one is saved.  Check resize_cache option.
    cat1.resize_cache(3)
    assert cat1.nfields.size == 3
    assert cat1.kfields.size == 3
    assert cat1.gfields.size == 3
    assert cat1.nfields.count == 1
    assert cat1.kfields.count == 1
    assert cat1.gfields.count == 1
    assert cat1.field is qfield1

    t0 = time.time()
    nfield1 = cat1.getNField()
    nfield2 = cat1.getNField(min_size=0.01, max_size=1)
    nfield3 = cat1.getNField(min_size=1, max_size=300, logger=logger)
    t1 = time.time()
    nfield1b = cat1.getNField()
    nfield2b = cat1.getNField(min_size=0.01, max_size=1)
    nfield3b = cat1.getNField(min_size=1, max_size=300, logger=logger)
    t2 = time.time()
    assert cat1.nfields.count == 3
    print('after resize(3) nfield: ',t1-t0,t2-t1)
    if platform.python_implementation() != 'PyPy':
        assert t2-t1 <= t1-t0
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
    assert cat1.field is None

    # Can also resize to 0
    cat1.resize_cache(0)
    assert cat1.nfields.count == 0
    assert cat1.nfields.size == 0
    t0 = time.time()
    nfield1 = cat1.getNField()
    nfield2 = cat1.getNField(min_size=0.01, max_size=1)
    nfield3 = cat1.getNField(min_size=1, max_size=300, logger=logger)
    t1 = time.time()
    nfield1b = cat1.getNField()
    nfield2b = cat1.getNField(min_size=0.01, max_size=1)
    nfield3b = cat1.getNField(min_size=1, max_size=300, logger=logger)
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

@timer
def test_combine():
    nobj = 100
    rng = np.random.RandomState(8675309)
    x = rng.random_sample(nobj)
    y = rng.random_sample(nobj)
    z = rng.random_sample(nobj)
    w = rng.random_sample(nobj)
    k = rng.random_sample(nobj)
    z1 = rng.random_sample(nobj)
    z2 = rng.random_sample(nobj)
    v1 = rng.random_sample(nobj)
    v2 = rng.random_sample(nobj)
    g1 = rng.random_sample(nobj)
    g2 = rng.random_sample(nobj)
    t1 = rng.random_sample(nobj)
    t2 = rng.random_sample(nobj)
    q1 = rng.random_sample(nobj)
    q2 = rng.random_sample(nobj)

    # This is the full catalog with all rows
    cat1 = treecorr.Catalog(x=x, y=y, z=z, w=w, g1=g1, g2=g2, k=k, z1=z1, z2=z2,
                            v1=v1, v2=v2, t1=t1, t2=t2, q1=q1, q2=q2)
    np.testing.assert_array_equal(cat1.x, x)
    np.testing.assert_array_equal(cat1.y, y)
    np.testing.assert_array_equal(cat1.z, z)
    np.testing.assert_array_equal(cat1.w, w)
    np.testing.assert_array_equal(cat1.k, k)
    np.testing.assert_array_equal(cat1.z1, z1)
    np.testing.assert_array_equal(cat1.z2, z2)
    np.testing.assert_array_equal(cat1.v1, v1)
    np.testing.assert_array_equal(cat1.v2, v2)
    np.testing.assert_array_equal(cat1.g1, g1)
    np.testing.assert_array_equal(cat1.g2, g2)
    np.testing.assert_array_equal(cat1.t1, t1)
    np.testing.assert_array_equal(cat1.t2, t2)
    np.testing.assert_array_equal(cat1.q1, q1)
    np.testing.assert_array_equal(cat1.q2, q2)

    # Now build it up slowly.
    cats = [treecorr.Catalog(x=x[i:j], y=y[i:j], z=z[i:j], w=w[i:j], k=k[i:j],
                             z1=z1[i:j], z2=z2[i:j],
                             v1=v1[i:j], v2=v2[i:j], g1=g1[i:j], g2=g2[i:j],
                             t1=t1[i:j], t2=t2[i:j], q1=q1[i:j], q2=q2[i:j])
            for (i,j) in [(0,20), (20,33), (33,82), (82,83), (83,100)]]
    cat2 = treecorr.Catalog.combine(cats)
    np.testing.assert_array_equal(cat2.x, x)
    np.testing.assert_array_equal(cat2.y, y)
    np.testing.assert_array_equal(cat2.z, z)
    np.testing.assert_array_equal(cat2.w, w)
    np.testing.assert_array_equal(cat2.k, k)
    np.testing.assert_array_equal(cat2.z1, z1)
    np.testing.assert_array_equal(cat2.z2, z2)
    np.testing.assert_array_equal(cat2.v1, v1)
    np.testing.assert_array_equal(cat2.v2, v2)
    np.testing.assert_array_equal(cat2.g1, g1)
    np.testing.assert_array_equal(cat2.g2, g2)
    np.testing.assert_array_equal(cat2.t1, t1)
    np.testing.assert_array_equal(cat2.t2, t2)
    np.testing.assert_array_equal(cat2.q1, q1)
    np.testing.assert_array_equal(cat2.q2, q2)
    assert cat2.ra is None
    assert cat2.dec is None
    assert cat2.ntot == cat1.ntot
    assert cat2.nobj == cat1.nobj
    assert cat2.nontrivial_w == cat1.nontrivial_w
    assert cat2.sumw == cat1.sumw
    assert cat2.sumw2 == cat1.sumw2
    assert cat2.coords == cat1.coords
    assert cat2 == cat1

    # Error if cat2 is missing some columns
    cat_no_t_q = treecorr.Catalog(x=x, y=y, z=z, w=w, g1=g1, g2=g2, k=k, v1=v1, v2=v2)
    assert_raises(ValueError, treecorr.Catalog.combine, [cat2, cat_no_t_q])

    # Can also use the mask to build up slowly
    cat3 = treecorr.Catalog.combine(
        [cat1]*5,
        mask_list=[slice(i,j) for i,j in [(0,20), (20,33), (33,82), (82,83), (83,100)]])
    assert cat3 == cat1
    assert cat3.ntot == cat1.ntot
    assert cat3.nobj == cat1.nobj
    assert cat3.nontrivial_w == cat1.nontrivial_w
    assert cat3.sumw == cat1.sumw
    assert cat3.sumw2 == cat1.sumw2
    assert cat3.coords == cat1.coords

    k = np.arange(100)
    cat4 = treecorr.Catalog.combine(
        [cat1]*5,
        mask_list=[(k>=i) & (k<j) for i,j in [(0,20), (20,33), (33,82), (82,83), (83,100)]])
    assert cat4 == cat1
    assert cat4.ntot == cat1.ntot
    assert cat4.nobj == cat1.nobj
    assert cat4.nontrivial_w == cat1.nontrivial_w
    assert cat4.sumw == cat1.sumw
    assert cat4.sumw2 == cat1.sumw2
    assert cat4.coords == cat1.coords

    # Check ra, dec
    cats = [treecorr.Catalog(ra=x[i:j], dec=y[i:j], r=z[i:j], w=w[i:j], k=k[i:j],
                             ra_units='deg', dec_units='deg')
            for (i,j) in [(0,20), (20,33), (33,82), (82,83), (83,100)]]
    cat5 = treecorr.Catalog.combine(cats)
    np.testing.assert_allclose(cat5.ra, x*np.pi/180)
    np.testing.assert_allclose(cat5.dec, y*np.pi/180)
    np.testing.assert_array_equal(cat5.r, z)
    np.testing.assert_array_equal(cat5.w, w)
    np.testing.assert_array_equal(cat5.k, k)
    assert cat5.z1 is None
    assert cat5.z2 is None
    assert cat5.v1 is None
    assert cat5.v2 is None
    assert cat5.g1 is None
    assert cat5.g2 is None
    assert cat5.t1 is None
    assert cat5.t2 is None
    assert cat5.q1 is None
    assert cat5.q2 is None
    assert cat5.ntot == len(x)

    # Check low_mem
    file_name = os.path.join('data', 'test_combine.dat')
    cat1.write(file_name)
    cats = [treecorr.Catalog(file_name, x_col='x', y_col='y', w_col='w', q1_col='q1', q2_col='q2')
            for i in range(5)]
    masks = [slice(i,j) for i,j in [(0,20), (20,33), (33,82), (82,83), (83,100)]]
    for c in cats:
        assert not c.loaded
    # Load just cats[2].  This should stay loaded after combine.
    cats[2].load()
    cat6 = treecorr.Catalog.combine(cats, mask_list=masks, low_mem=True)
    np.testing.assert_allclose(cat6.x, x)
    np.testing.assert_allclose(cat6.y, y)
    np.testing.assert_allclose(cat6.w, w)
    np.testing.assert_allclose(cat6.q1, q1)
    np.testing.assert_allclose(cat6.q2, q2)
    assert cat6.z is None
    assert cat6.k is None
    assert cat6.g1 is None
    assert cat6.g2 is None
    assert cat6.ntot == len(x)
    assert not cats[0].loaded
    assert not cats[1].loaded
    assert cats[2].loaded
    assert not cats[3].loaded
    assert not cats[4].loaded

    # An empty list is invalid.
    assert_raises(ValueError, treecorr.Catalog.combine, [])
    assert_raises(ValueError, treecorr.Catalog.combine, [cat1], mask_list=[])
    assert_raises(ValueError, treecorr.Catalog.combine, [cat1]*2, mask_list=[slice(None)])
    assert_raises(TypeError, treecorr.Catalog.combine, [cat1], [slice(None)])


if __name__ == '__main__':
    test_ascii()
    test_fits()
    test_hdf5()
    test_parquet()
    test_ext()
    test_direct()
    test_var()
    test_nan()
    test_nan2()
    test_contiguous()
    test_list()
    test_write()
    test_field()
    test_lru()
    test_combine()
