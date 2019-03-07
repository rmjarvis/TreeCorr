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
import os
import time
import coord
import warnings
from numpy import pi
import treecorr

from test_helper import get_from_wiki, CaptureLog

def test_ascii():

    nobj = 5000
    np.random.seed(8675309)
    x = np.random.random_sample(nobj)
    y = np.random.random_sample(nobj)
    z = np.random.random_sample(nobj)
    ra = np.random.random_sample(nobj)
    dec = np.random.random_sample(nobj)
    r = np.random.random_sample(nobj)
    wpos = np.random.random_sample(nobj)
    g1 = np.random.random_sample(nobj)
    g2 = np.random.random_sample(nobj)
    k = np.random.random_sample(nobj)

    # Some elements have both w and wpos = 0.
    w = wpos.copy()
    use = np.random.randint(30, size=nobj).astype(float)
    w[use == 0] = 0
    wpos[use == 0] = 0

    # Others just have w = 0
    use = np.random.randint(30, size=nobj).astype(float)
    w[use == 0] = 0

    flags = np.zeros(nobj).astype(int)
    for flag in [ 1, 2, 4, 8, 16 ]:
        sub = np.random.random_sample(nobj) < 0.1
        flags[sub] = np.bitwise_or(flags[sub], flag)

    file_name = os.path.join('data','test.dat')
    with open(file_name, 'w') as fid:
        # These are intentionally in a different order from the order we parse them.
        fid.write('# ra,dec,x,y,k,g1,g2,w,z,r,wpos,flag\n')
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

    # Check flags
    config['flag_col'] = 12
    cat2 = treecorr.Catalog(file_name, config)
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
    config['w_col'] = 8  # Put it back for later.
    config['wpos_col'] = 11  # Put them back for later.

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
    cat10 = treecorr.Catalog(file_name, config, w_col=0, wpos_col=11, flag_col=0)
    np.testing.assert_almost_equal(cat10.wpos, wpos)
    np.testing.assert_almost_equal(cat10.w[wpos==0], 0)
    np.testing.assert_almost_equal(cat10.w[wpos!=0], 1)


def test_fits():
    get_from_wiki('Aardvark.fit')
    file_name = os.path.join('data','Aardvark.fit')
    config = treecorr.read_config('configs/Aardvark.yaml')
    config['verbose'] = 1

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

    # The catalog doesn't have x, y, or w, but test that functionality as well.
    del config['ra_col']
    del config['dec_col']
    config['x_col'] = 'RA'
    config['y_col'] = 'DEC'
    config['w_col'] = 'MU'
    config['flag_col'] = 'INDEX'
    config['ignore_flag'] = 64
    cat2 = treecorr.Catalog(file_name, config)
    np.testing.assert_almost_equal(cat2.x[390934], 78.4782, decimal=4)
    np.testing.assert_almost_equal(cat2.y[290333], 83.1579, decimal=4)
    np.testing.assert_almost_equal(cat2.w[46392], 0.)        # index = 1200379
    np.testing.assert_almost_equal(cat2.w[46393], 0.9995946) # index = 1200386

    # Test using a limited set of rows
    config['first_row'] = 101
    config['last_row'] = 50000
    cat3 = treecorr.Catalog(file_name, config)
    np.testing.assert_equal(len(cat3.x), 49900)
    np.testing.assert_equal(cat3.ntot, 49900)
    np.testing.assert_equal(cat3.nobj, sum(cat3.w != 0))
    np.testing.assert_equal(cat3.sumw, sum(cat3.w))
    np.testing.assert_equal(cat3.sumw, sum(cat2.w[100:50000]))
    np.testing.assert_almost_equal(cat3.g1[46292], 0.0005066675)
    np.testing.assert_almost_equal(cat3.g2[46292], -0.0001006742)
    np.testing.assert_almost_equal(cat3.k[46292], -0.0008628797)

    cat4 = treecorr.read_catalogs(config, key='file_name', is_rand=True)[0]
    np.testing.assert_equal(len(cat4.x), 49900)
    np.testing.assert_equal(cat4.ntot, 49900)
    np.testing.assert_equal(cat4.nobj, sum(cat4.w != 0))
    np.testing.assert_equal(cat4.sumw, sum(cat4.w))
    np.testing.assert_equal(cat4.sumw, sum(cat2.w[100:50000]))
    assert cat4.g1 is None
    assert cat4.g2 is None
    assert cat4.k is None


def test_direct():

    nobj = 5000
    np.random.seed(8675309)
    x = np.random.random_sample(nobj)
    y = np.random.random_sample(nobj)
    ra = np.random.random_sample(nobj)
    dec = np.random.random_sample(nobj)
    w = np.random.random_sample(nobj)
    g1 = np.random.random_sample(nobj)
    g2 = np.random.random_sample(nobj)
    k = np.random.random_sample(nobj)

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
 

def test_var():
    nobj = 5000
    np.random.seed(8675309)

    # First without weights
    cats = []
    allg1 = []
    allg2 = []
    allk = []
    for i in range(10):
        x = np.random.random_sample(nobj)
        y = np.random.random_sample(nobj)
        g1 = np.random.random_sample(nobj) - 0.5
        g2 = np.random.random_sample(nobj) - 0.5
        k = np.random.random_sample(nobj) - 0.5
        cat = treecorr.Catalog(x=x, y=y, g1=g1, g2=g2, k=k)
        varg = (np.sum(g1**2) + np.sum(g2**2)) / (2.*len(g1))
        vark = np.sum(k**2) / len(k)
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
    vark = np.sum(allk**2) / len(allk)
    assert np.isclose(treecorr.calculateVarG(cats), varg)
    assert np.isclose(treecorr.calculateVarK(cats), vark)

    # Now with weights
    cats = []
    allg1 = []
    allg2 = []
    allk = []
    allw = []
    for i in range(10):
        x = np.random.random_sample(nobj)
        y = np.random.random_sample(nobj)
        w = np.random.random_sample(nobj)
        g1 = np.random.random_sample(nobj)
        g2 = np.random.random_sample(nobj)
        k = np.random.random_sample(nobj)
        cat = treecorr.Catalog(x=x, y=y, w=w, g1=g1, g2=g2, k=k)
        varg = np.sum(w**2 * (g1**2 + g2**2)) / np.sum(w) / 2.
        vark = np.sum(w**2 * k**2) / np.sum(w)
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
    vark = np.sum(allw**2 * allk**2) / np.sum(allw)
    assert np.isclose(treecorr.calculateVarG(cats), varg)
    assert np.isclose(treecorr.calculateVarK(cats), vark)


def test_nan():
    # Test handling of Nan values (w -> 0)

    nobj = 5000
    np.random.seed(8675309)
    x = np.random.random_sample(nobj)
    y = np.random.random_sample(nobj)
    z = np.random.random_sample(nobj)
    ra = np.random.random_sample(nobj)
    dec = np.random.random_sample(nobj)
    r = np.random.random_sample(nobj)
    w = np.random.random_sample(nobj)
    wpos = np.random.random_sample(nobj)
    g1 = np.random.random_sample(nobj)
    g2 = np.random.random_sample(nobj)
    k = np.random.random_sample(nobj)

    # Turn 1% of these values into NaN
    x[np.random.choice(nobj, nobj//100)] = np.nan
    y[np.random.choice(nobj, nobj//100)] = np.nan
    z[np.random.choice(nobj, nobj//100)] = np.nan
    ra[np.random.choice(nobj, nobj//100)] = np.nan
    dec[np.random.choice(nobj, nobj//100)] = np.nan
    r[np.random.choice(nobj, nobj//100)] = np.nan
    w[np.random.choice(nobj, nobj//100)] = np.nan
    wpos[np.random.choice(nobj, nobj//100)] = np.nan
    g1[np.random.choice(nobj, nobj//100)] = np.nan
    g2[np.random.choice(nobj, nobj//100)] = np.nan
    k[np.random.choice(nobj, nobj//100)] = np.nan
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
        cat1 = treecorr.Catalog(x=x, y=y, z=z, w=w, k=k, logger=cl.logger)
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
                                ra_units='hours', dec_units='degrees', logger=cl.logger)
    assert "NaNs found in ra column." in cl.output
    assert "NaNs found in dec column." in cl.output
    assert "NaNs found in r column." in cl.output
    assert "NaNs found in g1 column." in cl.output
    assert "NaNs found in g2 column." in cl.output
    assert "NaNs found in w column." in cl.output
    assert "NaNs found in wpos column." in cl.output
    mask = np.isnan(ra) | np.isnan(dec) | np.isnan(r) | np.isnan(g1) | np.isnan(g2) | np.isnan(wpos) | np.isnan(w)
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
        cat3 = treecorr.Catalog(x=x, y=y, g1=g1, g2=g2, logger=cl.logger)
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


def test_list():
    # Test different ways to read in a list of catalog names.
    # This is based on the bug report for Issue #10.

    nobj = 5000
    np.random.seed(8675309)

    x_list = []
    y_list = []
    file_names = []
    ncats = 3

    for k in range(ncats):
        x = np.random.random_sample(nobj)
        y = np.random.random_sample(nobj)
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

def test_write():
    # Test that writing a Catalog to a file and then reading it back in works correctly
    ngal = 20000
    s = 10.
    np.random.seed(8675309)
    x = np.random.normal(222,50, (ngal,) )
    y = np.random.normal(138,20, (ngal,) )
    z = np.random.normal(912,130, (ngal,) )
    w = np.random.normal(1.3, 0.1, (ngal,) )

    ra = np.random.normal(11.34, 0.9, (ngal,) )
    dec = np.random.normal(-48.12, 4.3, (ngal,) )
    r = np.random.normal(1024, 230, (ngal,) )

    k = np.random.normal(0,s, (ngal,) )
    g1 = np.random.normal(0,s, (ngal,) )
    g2 = np.random.normal(0,s, (ngal,) )

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

def test_field():
    # Test making various kinds of fields
    # Note: This is mostly just a coverage test to make sure there aren't any errors
    # when doing this manually.  The real functionality tests of using the fields are
    # all elsewhere.

    ngal = 2000
    s = 10.
    np.random.seed(8675309)
    x = np.random.normal(222,50, (ngal,) )
    y = np.random.normal(138,20, (ngal,) )
    z = np.random.normal(912,130, (ngal,) )
    w = np.random.normal(1.3, 0.1, (ngal,) )

    ra = np.random.normal(11.34, 0.9, (ngal,) )
    dec = np.random.normal(-48.12, 4.3, (ngal,) )
    r = np.random.normal(1024, 230, (ngal,) )

    k = np.random.normal(0,s, (ngal,) )
    g1 = np.random.normal(0,s, (ngal,) )
    g2 = np.random.normal(0,s, (ngal,) )

    cat1 = treecorr.Catalog(x=x, y=y, z=z, g1=g1, g2=g2, k=k)
    cat2 = treecorr.Catalog(ra=ra, dec=dec, ra_units='hour', dec_units='deg',
                            w=w, g1=g1, g2=g2, k=k)
    cat2.logger = None
    cat3 = treecorr.Catalog(x=x, y=y, g1=g1, g2=g2, k=k, w=w)
    logger = treecorr.config.setup_logger(1)

    t0 = time.time()
    nfield1 = cat1.getNField(1,1000)
    nfield2 = cat2.getNField(0.01, 1)
    nfield3 = cat3.getNField(1,300, logger=logger)
    t1 = time.time()
    nfield1b = cat1.getNField(1,1000)
    nfield2b = cat2.getNField(0.01, 1)
    nfield3b = cat3.getNField(1,300, logger=logger)
    t2 = time.time()
    # The second time, they should already be made and taken from the cache, so much faster.
    print('nfield: ',t1-t0,t2-t1)
    assert t2-t1 < t1-t0

    t0 = time.time()
    gfield1 = cat1.getGField(1,1000)
    gfield2 = cat2.getGField(0.01, 1)
    gfield3 = cat3.getGField(1,300, logger=logger)
    t1 = time.time()
    gfield1b = cat1.getGField(1,1000)
    gfield2b = cat2.getGField(0.01, 1)
    gfield3b = cat3.getGField(1,300, logger=logger)
    t2 = time.time()
    print('gfield: ',t1-t0,t2-t1)
    assert t2-t1 < t1-t0

    t0 = time.time()
    kfield1 = cat1.getKField(1,1000)
    kfield2 = cat2.getKField(0.01, 1)
    kfield3 = cat3.getKField(1,300, logger=logger)
    t1 = time.time()
    kfield1b = cat1.getKField(1,1000)
    kfield2b = cat2.getKField(0.01, 1)
    kfield3b = cat3.getKField(1,300, logger=logger)
    t2 = time.time()
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
    print('ksimplefield: ',t1-t0,t2-t1)
    assert t2-t1 < t1-t0


if __name__ == '__main__':
    test_ascii()
    test_fits()
    test_direct()
    test_var()
    test_contiguous()
    test_list()
    test_write()
    test_field()
