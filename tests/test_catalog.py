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
from numpy import pi

from test_helper import get_from_wiki

def test_ascii():

    nobj = 5000
    numpy.random.seed(8675309)
    x = numpy.random.random_sample(nobj)
    y = numpy.random.random_sample(nobj)
    z = numpy.random.random_sample(nobj)
    ra = numpy.random.random_sample(nobj)
    dec = numpy.random.random_sample(nobj)
    r = numpy.random.random_sample(nobj)
    w = numpy.random.random_sample(nobj)
    g1 = numpy.random.random_sample(nobj)
    g2 = numpy.random.random_sample(nobj)
    k = numpy.random.random_sample(nobj)

    flags = numpy.zeros(nobj).astype(int)
    for flag in [ 1, 2, 4, 8, 16 ]:
        sub = numpy.random.random_sample(nobj) < 0.1
        flags[sub] = numpy.bitwise_or(flags[sub], flag)

    file_name = os.path.join('data','test.dat')
    with open(file_name, 'w') as fid:
        # These are intentionally in a different order from the order we parse them.
        fid.write('# ra,dec,x,y,k,g1,g2,w,flag,z,r\n')
        for i in range(nobj):
            fid.write((('%.8f '*10)+'%d\n')%(
                ra[i],dec[i],x[i],y[i],k[i],g1[i],g2[i],w[i],z[i],r[i],flags[i]))

    # Check basic input
    config = {
        'x_col' : 3,
        'y_col' : 4,
        'z_col' : 9,
        'x_units' : 'rad',
        'y_units' : 'rad',
        'w_col' : 8,
        'g1_col' : 6,
        'g2_col' : 7,
        'k_col' : 5,
    }
    cat1 = treecorr.Catalog(file_name, config)
    numpy.testing.assert_almost_equal(cat1.x, x)
    numpy.testing.assert_almost_equal(cat1.y, y)
    numpy.testing.assert_almost_equal(cat1.z, z)
    numpy.testing.assert_almost_equal(cat1.w, w)
    numpy.testing.assert_almost_equal(cat1.g1, g1)
    numpy.testing.assert_almost_equal(cat1.g2, g2)
    numpy.testing.assert_almost_equal(cat1.k, k)

    # Check flags
    config['flag_col'] = 11
    cat2 = treecorr.Catalog(file_name, config)
    numpy.testing.assert_almost_equal(cat2.w[flags==0], w[flags==0])
    numpy.testing.assert_almost_equal(cat2.w[flags!=0], 0.)

    # Check ok_flag
    config['ok_flag'] = 4
    cat3 = treecorr.Catalog(file_name, config)
    numpy.testing.assert_almost_equal(cat3.w[numpy.logical_or(flags==0, flags==4)], 
                                      w[numpy.logical_or(flags==0, flags==4)])
    numpy.testing.assert_almost_equal(cat3.w[numpy.logical_and(flags!=0, flags!=4)], 0.)

    # Check ignore_flag
    del config['ok_flag']
    config['ignore_flag'] = 16
    cat4 = treecorr.Catalog(file_name, config)
    numpy.testing.assert_almost_equal(cat4.w[flags < 16], w[flags < 16])
    numpy.testing.assert_almost_equal(cat4.w[flags >= 16], 0.)

    # Check different units for x,y
    config['x_units'] = 'arcsec'
    config['y_units'] = 'arcsec'
    del config['z_col']
    cat5 = treecorr.Catalog(file_name, config)
    numpy.testing.assert_almost_equal(cat5.x, x * (pi/180./3600.))
    numpy.testing.assert_almost_equal(cat5.y, y * (pi/180./3600.))

    config['x_units'] = 'arcmin'
    config['y_units'] = 'arcmin'
    cat5 = treecorr.Catalog(file_name, config)
    numpy.testing.assert_almost_equal(cat5.x, x * (pi/180./60.))
    numpy.testing.assert_almost_equal(cat5.y, y * (pi/180./60.))

    config['x_units'] = 'deg'
    config['y_units'] = 'deg'
    cat5 = treecorr.Catalog(file_name, config)
    numpy.testing.assert_almost_equal(cat5.x, x * (pi/180.))
    numpy.testing.assert_almost_equal(cat5.y, y * (pi/180.))

    del config['x_units']  # Default is radians
    del config['y_units']
    cat5 = treecorr.Catalog(file_name, config)
    numpy.testing.assert_almost_equal(cat5.x, x)
    numpy.testing.assert_almost_equal(cat5.y, y)

    # Check ra,dec
    del config['x_col']
    del config['y_col']
    config['ra_col'] = 1
    config['dec_col'] = 2
    config['r_col'] = 10
    config['ra_units'] = 'rad'
    config['dec_units'] = 'rad'
    cat6 = treecorr.Catalog(file_name, config)
    numpy.testing.assert_almost_equal(cat6.ra, ra)
    numpy.testing.assert_almost_equal(cat6.dec, dec)

    config['ra_units'] = 'deg'
    config['dec_units'] = 'deg'
    cat6 = treecorr.Catalog(file_name, config)
    numpy.testing.assert_almost_equal(cat6.ra, ra * (pi/180.))
    numpy.testing.assert_almost_equal(cat6.dec, dec * (pi/180.))

    config['ra_units'] = 'hour'
    config['dec_units'] = 'deg'
    cat6 = treecorr.Catalog(file_name, config)
    numpy.testing.assert_almost_equal(cat6.ra, ra * (pi/12.))
    numpy.testing.assert_almost_equal(cat6.dec, dec * (pi/180.))

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
            fid.write((('%.8f,'*10)+'%d\n')%(
                ra[i],dec[i],x[i],y[i],k[i],g1[i],g2[i],w[i],z[i],r[i],flags[i]))
            if i%100 == 0:
                fid.write('%%%% Line %d\n'%i)
    config['delimiter'] = ','
    config['comment_marker'] = '%'
    cat7 = treecorr.Catalog(csv_file_name, config)
    numpy.testing.assert_almost_equal(cat7.ra, ra * (pi/12.))
    numpy.testing.assert_almost_equal(cat7.dec, dec * (pi/180.))
    numpy.testing.assert_almost_equal(cat7.r, r)
    numpy.testing.assert_almost_equal(cat7.g1, g1)
    numpy.testing.assert_almost_equal(cat7.g2, g2)
    numpy.testing.assert_almost_equal(cat7.w[flags < 16], w[flags < 16])
    numpy.testing.assert_almost_equal(cat7.w[flags >= 16], 0.)

    # Check flip_g1, flip_g2
    del config['delimiter']
    del config['comment_marker']
    config['flip_g1'] = True
    cat8 = treecorr.Catalog(file_name, config)
    numpy.testing.assert_almost_equal(cat8.g1, -g1)
    numpy.testing.assert_almost_equal(cat8.g2, g2)

    config['flip_g2'] = 'true'
    cat8 = treecorr.Catalog(file_name, config)
    numpy.testing.assert_almost_equal(cat8.g1, -g1)
    numpy.testing.assert_almost_equal(cat8.g2, -g2)

    config['flip_g1'] = 'n'
    config['flip_g2'] = 'yes'
    cat8 = treecorr.Catalog(file_name, config)
    numpy.testing.assert_almost_equal(cat8.g1, g1)
    numpy.testing.assert_almost_equal(cat8.g2, -g2)

    # Check overriding values with kwargs
    cat8 = treecorr.Catalog(file_name, config, flip_g1=True, flip_g2=False)
    numpy.testing.assert_almost_equal(cat8.g1, -g1)
    numpy.testing.assert_almost_equal(cat8.g2, g2)


 
def test_fits():
    get_from_wiki('Aardvark.fit')
    file_name = os.path.join('data','Aardvark.fit')
    config = treecorr.read_config('Aardvark.params')

    # Just test a few random particular values
    cat1 = treecorr.Catalog(file_name, config)
    numpy.testing.assert_equal(len(cat1.ra), 390935)
    numpy.testing.assert_equal(cat1.nobj, 390935)
    numpy.testing.assert_almost_equal(cat1.ra[0], 56.4195 * (pi/180.))
    numpy.testing.assert_almost_equal(cat1.ra[390934], 78.4782 * (pi/180.))
    numpy.testing.assert_almost_equal(cat1.dec[290333], 83.1579 * (pi/180.))
    numpy.testing.assert_almost_equal(cat1.g1[46392], 0.0005066675)
    numpy.testing.assert_almost_equal(cat1.g2[46392], -0.0001006742)
    numpy.testing.assert_almost_equal(cat1.k[46392], -0.0008628797)

    # The catalog doesn't have x, y, or w, but test that functionality as well.
    del config['ra_col']
    del config['dec_col']
    config['x_col'] = 'RA'
    config['y_col'] = 'DEC'
    config['w_col'] = 'MU'
    config['flag_col'] = 'INDEX'
    config['ignore_flag'] = 64
    cat2 = treecorr.Catalog(file_name, config)
    numpy.testing.assert_almost_equal(cat2.x[390934], 78.4782, decimal=4)
    numpy.testing.assert_almost_equal(cat2.y[290333], 83.1579, decimal=4)
    numpy.testing.assert_almost_equal(cat2.w[46392], 0.)        # index = 1200379
    numpy.testing.assert_almost_equal(cat2.w[46393], 0.9995946) # index = 1200386

    # Test using a limited set of rows
    config['first_row'] = 101
    config['last_row'] = 50000
    cat3 = treecorr.Catalog(file_name, config)
    numpy.testing.assert_equal(len(cat3.x), 49900)
    numpy.testing.assert_equal(cat3.ntot, 49900)
    numpy.testing.assert_equal(cat3.nobj, sum(cat3.w != 0))
    numpy.testing.assert_equal(cat3.sumw, sum(cat3.w))
    numpy.testing.assert_equal(cat3.sumw, sum(cat2.w[100:50000]))
    numpy.testing.assert_almost_equal(cat3.g1[46292], 0.0005066675)
    numpy.testing.assert_almost_equal(cat3.g2[46292], -0.0001006742)
    numpy.testing.assert_almost_equal(cat3.k[46292], -0.0008628797)


def test_direct():

    nobj = 5000
    numpy.random.seed(8675309)
    x = numpy.random.random_sample(nobj)
    y = numpy.random.random_sample(nobj)
    ra = numpy.random.random_sample(nobj)
    dec = numpy.random.random_sample(nobj)
    w = numpy.random.random_sample(nobj)
    g1 = numpy.random.random_sample(nobj)
    g2 = numpy.random.random_sample(nobj)
    k = numpy.random.random_sample(nobj)

    cat1 = treecorr.Catalog(x=x, y=y, w=w, g1=g1, g2=g2, k=k)
    numpy.testing.assert_almost_equal(cat1.x, x)
    numpy.testing.assert_almost_equal(cat1.y, y)
    numpy.testing.assert_almost_equal(cat1.w, w)
    numpy.testing.assert_almost_equal(cat1.g1, g1)
    numpy.testing.assert_almost_equal(cat1.g2, g2)
    numpy.testing.assert_almost_equal(cat1.k, k)

    cat2 = treecorr.Catalog(ra=ra, dec=dec, w=w, g1=g1, g2=g2, k=k,
                            ra_units='hours', dec_units='degrees')
    numpy.testing.assert_almost_equal(cat2.ra, ra * treecorr.hours)
    numpy.testing.assert_almost_equal(cat2.dec, dec * treecorr.degrees)
    numpy.testing.assert_almost_equal(cat2.w, w)
    numpy.testing.assert_almost_equal(cat2.g1, g1)
    numpy.testing.assert_almost_equal(cat2.g2, g2)
    numpy.testing.assert_almost_equal(cat2.k, k)

def test_contiguous():
    # This unit test comes from Melanie Simet who discovered a bug in earlier
    # versions of the code that the Catalog didn't correctly handle input arrays
    # that were not contiguous in memory.  We want to make sure this kind of
    # input works correctly.  It also checks that the input dtype doesn't have
    # to be float

    source_data = numpy.array([
            (0.0380569697547, 0.0142782758818, 0.330845443464, -0.111049332655),
            (-0.0261291090735, 0.0863787933931, 0.122954685209, 0.40260430406),
            (-0.0261291090735, 0.0863787933931, 0.122954685209, 0.40260430406),
            (0.125086697534, 0.0283621046495, -0.208159531309, 0.142491564101),
            (0.0457709426026, -0.0299249486373, -0.0406555089425, 0.24515956887),
            (-0.00338578248926, 0.0460291122935, 0.363057738173, -0.524536297555)],
            dtype=[('ra', None), ('dec', numpy.float64), ('g1', numpy.float32),
                   ('g2', numpy.float128)])

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
    numpy.testing.assert_equal(ng.xi, ng_float.xi)

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
    numpy.testing.assert_equal(ng.xi, ng_float.xi)


def test_list():
    # Test different ways to read in a list of catalog names.
    # This is based on the bug report for Issue #10.

    nobj = 5000
    numpy.random.seed(8675309)

    x_list = []
    y_list = []
    file_names = []
    ncats = 3

    for k in range(ncats):
        x = numpy.random.random_sample(nobj)
        y = numpy.random.random_sample(nobj)
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
    numpy.testing.assert_equal(len(cats), ncats)
    for k in range(ncats):
        numpy.testing.assert_almost_equal(cats[k].x, x_list[k])
        numpy.testing.assert_almost_equal(cats[k].y, y_list[k])

    # Next check that the list can be just a string with spaces between names:
    config['file_name'] = " ".join(file_names)

    # Also check that it is ok to include file_list to read_catalogs.
    cats = treecorr.read_catalogs(config, 'file_name', 'file_list')
    numpy.testing.assert_equal(len(cats), ncats)
    for k in range(ncats):
        numpy.testing.assert_almost_equal(cats[k].x, x_list[k])
        numpy.testing.assert_almost_equal(cats[k].y, y_list[k])

    # Next check that having the names in a file_list file works:
    list_name = os.path.join('data','test_list.txt')
    with open(list_name, 'w') as fid:
        for name in file_names:
            fid.write(name + '\n')
    del config['file_name']
    config['file_list'] = list_name

    cats = treecorr.read_catalogs(config, 'file_name', 'file_list')
    numpy.testing.assert_equal(len(cats), ncats)
    for k in range(ncats):
        numpy.testing.assert_almost_equal(cats[k].x, x_list[k])
        numpy.testing.assert_almost_equal(cats[k].y, y_list[k])

    # Also, should be allowed to omit file_name arg:
    cats = treecorr.read_catalogs(config, list_key='file_list')
    numpy.testing.assert_equal(len(cats), ncats)
    for k in range(ncats):
        numpy.testing.assert_almost_equal(cats[k].x, x_list[k])
        numpy.testing.assert_almost_equal(cats[k].y, y_list[k])

def test_write():
    # Test that writing a Catalog to a file and then reading it back in works correctly
    ngal = 20000
    s = 10.
    numpy.random.seed(8675309)
    x = numpy.random.normal(222,50, (ngal,) )
    y = numpy.random.normal(138,20, (ngal,) )
    z = numpy.random.normal(912,130, (ngal,) )
    w = numpy.random.normal(1.3, 0.1, (ngal,) )

    ra = numpy.random.normal(11.34, 0.9, (ngal,) )
    dec = numpy.random.normal(-48.12, 4.3, (ngal,) )
    r = numpy.random.normal(1024, 230, (ngal,) )

    k = numpy.random.normal(0,s, (ngal,) )
    g1 = numpy.random.normal(0,s, (ngal,) )
    g2 = numpy.random.normal(0,s, (ngal,) )

    cat1 = treecorr.Catalog(x=x, y=y, z=z)
    cat2 = treecorr.Catalog(ra=ra, dec=dec, r=r, ra_units='hour', dec_units='deg',
                            w=w, g1=g1, g2=g2, k=k)

    # Test ASCII output
    cat1.write(os.path.join('output','cat1.dat'))
    cat1_asc = treecorr.Catalog(os.path.join('output','cat1.dat'), file_type='ASCII',
                                x_col=1, y_col=2, z_col=3)
    numpy.testing.assert_almost_equal(cat1_asc.x, x)
    numpy.testing.assert_almost_equal(cat1_asc.y, y)
    numpy.testing.assert_almost_equal(cat1_asc.z, z)

    cat2.write(os.path.join('output','cat2.dat'), file_type='ASCII')
    cat2_asc = treecorr.Catalog(os.path.join('output','cat2.dat'), ra_col=1, dec_col=2, 
                                r_col=3, w_col=4, g1_col=5, g2_col=6, k_col=7, 
                                ra_units='rad', dec_units='rad')
    numpy.testing.assert_almost_equal(cat2_asc.ra, ra)
    numpy.testing.assert_almost_equal(cat2_asc.dec, dec)
    numpy.testing.assert_almost_equal(cat2_asc.r, r)
    numpy.testing.assert_almost_equal(cat2_asc.w, w)
    numpy.testing.assert_almost_equal(cat2_asc.g1, g1)
    numpy.testing.assert_almost_equal(cat2_asc.g2, g2)
    numpy.testing.assert_almost_equal(cat2_asc.k, k)

    # Test FITS output
    cat1.write(os.path.join('output','cat1.fits'), file_type='FITS')
    cat1_fits = treecorr.Catalog(os.path.join('output','cat1.fits'),
                                 x_col='x', y_col='y', z_col='z')
    numpy.testing.assert_almost_equal(cat1_fits.x, x)
    numpy.testing.assert_almost_equal(cat1_fits.y, y)
    numpy.testing.assert_almost_equal(cat1_fits.z, z)

    cat2.write(os.path.join('output','cat2.fits'))
    cat2_fits = treecorr.Catalog(os.path.join('output','cat2.fits'), ra_col='ra', dec_col='dec', 
                                 r_col='r', w_col='w', g1_col='g1', g2_col='g2', k_col='k', 
                                 ra_units='rad', dec_units='rad', file_type='FITS')
    numpy.testing.assert_almost_equal(cat2_fits.ra, ra)
    numpy.testing.assert_almost_equal(cat2_fits.dec, dec)
    numpy.testing.assert_almost_equal(cat2_fits.r, r)
    numpy.testing.assert_almost_equal(cat2_fits.w, w)
    numpy.testing.assert_almost_equal(cat2_fits.g1, g1)
    numpy.testing.assert_almost_equal(cat2_fits.g2, g2)
    numpy.testing.assert_almost_equal(cat2_fits.k, k)


if __name__ == '__main__':
    test_ascii()
    test_fits()
    test_direct()
    test_contiguous()
    test_list()
    test_write()
