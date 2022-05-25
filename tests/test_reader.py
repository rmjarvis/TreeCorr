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

import os
import sys
import numpy as np
from unittest import mock

from treecorr.reader import FitsReader, HdfReader, PandasReader, AsciiReader, ParquetReader
from treecorr.writer import FitsWriter, HdfWriter, AsciiWriter
from test_helper import get_from_wiki, assert_raises, timer, CaptureLog
from treecorr.util import make_writer, make_reader

@timer
def test_fits_reader():
    try:
        import fitsio
    except ImportError:
        print('Skip test_fits_reader, since fitsio not installed')
        return

    get_from_wiki('Aardvark.fit')
    r = FitsReader(os.path.join('data','Aardvark.fit'))

    # Check things not allowed if not in context
    with assert_raises(RuntimeError):
        r.read(['RA'], slice(0,10,2), 1)
    with assert_raises(RuntimeError):
        r.read('RA')
    with assert_raises(RuntimeError):
        r.row_count('DEC', 1)
    with assert_raises(RuntimeError):
        r.row_count()
    with assert_raises(RuntimeError):
        r.names(1)
    with assert_raises(RuntimeError):
        r.names()
    with assert_raises(RuntimeError):
        1 in r

    with r:
        assert_raises(ValueError, r.check_valid_ext, 'invalid')
        assert_raises(ValueError, r.check_valid_ext, 0)
        r.check_valid_ext('AARDWOLF')
        r.check_valid_ext(1)

        # Default ext is 1
        assert r.default_ext == 1

        # Default ext is "in" reader
        assert 1 in r

        # Probably can slice, but depends on installed fitsio version
        assert r.can_slice == (fitsio.__version__ > '1.0.6')

        s = slice(0, 10, 2)
        for ext in [1, 'AARDWOLF']:
            data = r.read(['RA'], s, ext)
            dec = r.read('DEC', s, ext)
            assert data['RA'].size == 5
            assert dec.size == 5

            assert r.row_count('RA', ext) == 390935
            assert r.row_count('GAMMA1', ext) == 390935
            assert set(r.names(ext)) == set("INDEX RA DEC Z EPSILON GAMMA1 GAMMA2 KAPPA MU".split())
            assert set(r.names(ext)) == set(r.names())

        # Can read without slice or ext to use defaults
        assert r.row_count() == 390935
        g2 = r.read('GAMMA2')
        assert len(g2) == 390935
        d = r.read(['KAPPA', 'MU'])
        assert len(d['KAPPA']) == 390935
        assert len(d['MU']) == 390935
        kappa = d['KAPPA']
        mu = d['MU']

        # check we can also index by integer, not just number
        d = r.read(['DEC'], np.arange(10), 'AARDWOLF')
        assert d.size==10

    # Again check things not allowed if not in context
    with assert_raises(RuntimeError):
        r.read(['RA'], slice(0,10,2), 1)
    with assert_raises(RuntimeError):
        r.read('RA')
    with assert_raises(RuntimeError):
        r.row_count('DEC', 1)
    with assert_raises(RuntimeError):
        r.row_count()
    with assert_raises(RuntimeError):
        r.names(1)
    with assert_raises(RuntimeError):
        r.names()
    with assert_raises(RuntimeError):
        1 in r

    # Check writer too.
    with FitsWriter(os.path.join('output','test_fits_writer.fits')) as w:
        w.write(['KAPPA', 'MU'], [kappa, mu], params={'test': True}, ext='KM')
    with FitsReader(os.path.join('output','test_fits_writer.fits')) as r:
        params = r.read_params(ext='KM')
        data = r.read_data(ext='KM')
        with assert_raises(OSError):
            params = r.read_params(ext='KK')
    assert params['test'] is True
    assert np.array_equal(data['KAPPA'], kappa)
    assert np.array_equal(data['MU'], mu)

    # Use make_writer, make_reader
    with make_writer(os.path.join('output','test_fits_writer.fits')) as w:
        w.write(['KAPPA', 'MU'], [kappa, mu], params={'test': True})
    with make_reader(os.path.join('output','test_fits_writer.fits')) as r:
        params = r.read_params()
        data = r.read_data()
    assert params['test'] is True
    assert np.array_equal(data['KAPPA'], kappa)
    assert np.array_equal(data['MU'], mu)

    # No params
    with make_writer(os.path.join('output','test_fits_writer.fits')) as w:
        w.write(['KAPPA', 'MU'], [kappa, mu])
    with make_reader(os.path.join('output','test_fits_writer.fits')) as r:
        params = r.read_params()
        data = r.read_data()
    assert 'test' not in params   # The test key isn't in params
    assert params['naxis1'] == 16       # But there are all the regular fits header items.
    assert params['naxis2'] == 390935
    assert np.array_equal(data['KAPPA'], kappa)
    assert np.array_equal(data['MU'], mu)

    with assert_raises(ValueError):
        make_writer(os.path.join('output','test_fits_writer.fits'), file_type='invalid')
    with assert_raises(ValueError):
        make_reader(os.path.join('output','test_fits_writer.fits'), file_type='invalid')

    # Not allowed to write when not in with context
    w = FitsWriter(os.path.join('output','test_fits_writer.fits'))
    with assert_raises(RuntimeError):
        w.write(['KAPPA', 'MU'], [kappa, mu], params={'test': True}, ext='KM')

    # Regardless of the system's fitsio version, check the two cases in code.
    with mock.patch('fitsio.__version__', '1.0.6'):
        with FitsReader(os.path.join('data','Aardvark.fit')) as r:
            assert not r.can_slice
    with mock.patch('fitsio.__version__', '1.1.0'):
        with FitsReader(os.path.join('data','Aardvark.fit')) as r:
            assert r.can_slice
    with mock.patch.dict(sys.modules, {'fitsio':None}):
        with CaptureLog() as cl:
            with assert_raises(ImportError):
                FitsReader(os.path.join('data','Aardvark.fit'), logger=cl.logger)
        assert 'Cannot read' in cl.output
        with CaptureLog() as cl:
            with assert_raises(ImportError):
                FitsWriter(os.path.join('output','test_fits_writer.fits'), logger=cl.logger)
        assert 'Cannot write to' in cl.output

@timer
def test_hdf_reader():
    try:
        import h5py  # noqa: F401
    except ImportError:
        print('Skipping HdfReader tests, since h5py not installed.')
        return

    get_from_wiki('Aardvark.hdf5')
    r = HdfReader(os.path.join('data','Aardvark.hdf5'))

    # Check things not allowed if not in context
    with assert_raises(RuntimeError):
        r.read(['RA'], slice(0,10,2), '/')
    with assert_raises(RuntimeError):
        r.read('RA')
    with assert_raises(RuntimeError):
        r.row_count('DEC', '/')
    with assert_raises(RuntimeError):
        r.row_count('DEC')
    with assert_raises(RuntimeError):
        r.names('/')
    with assert_raises(RuntimeError):
        r.names()
    with assert_raises(RuntimeError):
        '/' in r

    with r:

        # '/' is the only extension in this file.
        # TODO: Add an hdf5 example with other valid choices for ext
        assert_raises(ValueError, r.check_valid_ext, 'invalid')
        r.check_valid_ext('/')

        # Default ext is '/'
        assert r.default_ext == '/'

        # Default ext is "in" reader
        assert '/' in r

        # Can always slice
        assert r.can_slice

        s = slice(0, 10, 2)
        data = r.read(['RA'], s)
        dec = r.read('DEC', s)
        assert data['RA'].size == 5
        assert dec.size == 5

        assert r.row_count('RA') == 390935
        assert r.row_count('RA','/') == 390935
        assert r.row_count('GAMMA1') == 390935
        # Unlike the other readers, this needs a column name.
        with assert_raises(TypeError):
            r.row_count()
        assert set(r.names()) == set("INDEX RA DEC Z EPSILON GAMMA1 GAMMA2 KAPPA MU".split())
        assert set(r.names('/')) == set(r.names())

        # Can read without slice or ext to use defaults
        g2 = r.read('GAMMA2')
        assert len(g2) == 390935
        d = r.read(['KAPPA', 'MU'])
        assert len(d['KAPPA']) == 390935
        assert len(d['MU']) == 390935
        kappa = d['KAPPA']
        mu = d['MU']

    # Again check things not allowed if not in context
    with assert_raises(RuntimeError):
        r.read(['RA'], slice(0,10,2), '/')
    with assert_raises(RuntimeError):
        r.read('RA')
    with assert_raises(RuntimeError):
        r.row_count('DEC', '/')
    with assert_raises(RuntimeError):
        r.row_count('DEC')
    with assert_raises(RuntimeError):
        r.names('/')
    with assert_raises(RuntimeError):
        r.names()
    with assert_raises(RuntimeError):
        '/' in r

    # Check writer too.
    with HdfWriter(os.path.join('output','test_hdf_writer.hdf')) as w:
        w.write(['KAPPA', 'MU'], [kappa, mu], params={'test': True}, ext='KM')
    with HdfReader(os.path.join('output','test_hdf_writer.hdf')) as r:
        params = r.read_params(ext='KM')
        data = r.read_data(ext='KM')
        with assert_raises(OSError):
            params = r.read_params(ext='KK')
    assert params['test']
    assert np.array_equal(data['KAPPA'], kappa)
    assert np.array_equal(data['MU'], mu)

    # Not allowed to write when not in with context
    w = HdfWriter(os.path.join('output','test_hdf_writer.hdf'))
    with assert_raises(RuntimeError):
        w.write(['KAPPA', 'MU'], [kappa, mu], params={'test': True}, ext='KM')


@timer
def test_parquet_reader():
    try:
        import pandas  # noqa: F401
        import pyarrow # noqa: F401
    except ImportError:
        print('Skipping ParquetReader tests, since pandas or pyarrow not installed.')
        return

    get_from_wiki('Aardvark.parquet')
    r = ParquetReader(os.path.join('data','Aardvark.parquet'))

    # Check things not allowed if not in context
    with assert_raises(RuntimeError):
        r.read(['RA'], slice(0,10,2), None)
    with assert_raises(RuntimeError):
        r.read('RA')
    with assert_raises(RuntimeError):
        r.row_count('DEC', None)
    with assert_raises(RuntimeError):
        r.row_count('DEC')
    with assert_raises(RuntimeError):
        r.row_count()
    with assert_raises(RuntimeError):
        r.names(None)
    with assert_raises(RuntimeError):
        r.names()

    with r:

        # None is the only extension in this file.
        assert_raises(ValueError, r.check_valid_ext, 'invalid')
        r.check_valid_ext(None)

        # Default ext is None
        assert r.default_ext == None

        # Default ext is "in" reader
        assert None in r

        # Can always slice
        assert r.can_slice

        s = slice(0, 10, 2)
        data = r.read(['RA'], s)
        dec = r.read('DEC', s)
        assert data['RA'].size == 5
        assert dec.size == 5

        assert r.row_count('RA') == 390935
        assert r.row_count('RA',None) == 390935
        assert r.row_count('GAMMA1') == 390935
        assert r.row_count() == 390935
        print('names = ',set(r.names()))
        print('names = ',set("INDEX RA DEC Z GAMMA1 GAMMA2 KAPPA MU".split()))
        assert set(r.names()) == set("INDEX RA DEC Z GAMMA1 GAMMA2 KAPPA MU".split())
        assert set(r.names(None)) == set(r.names())

    # Again check things not allowed if not in context
    with assert_raises(RuntimeError):
        r.read(['RA'], slice(0,10,2), None)
    with assert_raises(RuntimeError):
        r.read('RA')
    with assert_raises(RuntimeError):
        r.row_count('DEC', None)
    with assert_raises(RuntimeError):
        r.row_count('DEC')
    with assert_raises(RuntimeError):
        r.row_count()
    with assert_raises(RuntimeError):
        r.names(None)
    with assert_raises(RuntimeError):
        r.names()

def _test_ascii_reader(r, has_names=True):
    # Same tests for AsciiReader and PandasReader

    # Check things not allowed if not in context
    with assert_raises(RuntimeError):
        r.read([1,3,9], None)
    with assert_raises(RuntimeError):
        r.read([1,3,9])
    with assert_raises(RuntimeError):
        r.read('ra')
    with assert_raises(RuntimeError):
        r.row_count(1, None)
    with assert_raises(RuntimeError):
        r.row_count()
    with assert_raises(RuntimeError):
        r.names(None)
    with assert_raises(RuntimeError):
        r.names()

    with r:
        # None is only value ext.
        assert_raises(ValueError, r.check_valid_ext, 'invalid')
        assert_raises(ValueError, r.check_valid_ext, '0')
        assert_raises(ValueError, r.check_valid_ext, 1)
        r.check_valid_ext(None)
        assert r.default_ext is None

        # Default ext is "in" reader
        assert None in r

        # Can always slice
        assert r.can_slice

        # cols are: ra, dec, x, y, k, g1, g2, w, z, r, wpos, flag
        s = slice(0, 10, 2)
        data = r.read([1,3,9], s)
        dec = r.read(2, s)
        assert sorted(data.keys()) == [1,3,9]
        assert data[1].size == 5
        assert data[3].size == 5
        assert data[9].size == 5
        assert dec.size == 5
        # Check a few random values
        assert data[1][0] == 0.34044927  # ra, row 1
        assert data[3][4] == 0.01816738  # x, row 9
        assert data[9][3] == 0.79008204  # z, row 7

        assert r.row_count(1, None) == 20
        assert r.row_count() == 20
        assert r.ncols == 12
        for i in range(12):
            assert str(i+1) in r.names()

        all_data = r.read(range(1,r.ncols+1))
        assert len(all_data) == 12
        assert len(all_data[1]) == 20
        assert r.row_count() == 20

        # Check reading specific rows
        s2 = np.array([0,6,8])
        data2 = r.read([1,3,9], s2)
        dec2 = r.read(2, s2)
        assert sorted(data2.keys()) == [1,3,9]
        assert data2[1].size == 3
        assert data2[3].size == 3
        assert data2[9].size == 3
        assert dec2.size == 3
        # Check the same values in this selection
        assert data2[1][0] == 0.34044927  # ra, row 1
        assert data2[3][2] == 0.01816738  # x, row 9
        assert data2[9][1] == 0.79008204  # z, row 7

        if not has_names:
            return
        # Repeat with column names
        data = r.read(['ra','x','z'], s)
        dec = r.read('dec', s)
        assert sorted(data.keys()) == ['ra','x','z']
        assert data['ra'].size == 5
        assert data['x'].size == 5
        assert data['z'].size == 5
        assert dec.size == 5
        # Check the same random values
        assert data['ra'][0] == 0.34044927
        assert data['x'][4] == 0.01816738
        assert data['z'][3] == 0.79008204

        assert r.row_count('ra', None) == 20
        assert r.row_count() == 20
        assert r.ncols == 12
        names = ['ra', 'dec', 'x', 'y', 'k', 'g1', 'g2', 'w', 'z', 'r', 'wpos', 'flag']
        for name in names:
            assert name in r.names()

        all_data = r.read(names)
        assert len(all_data) == 12
        assert len(all_data['ra']) == 20
        assert r.row_count() == 20

        g1 = all_data['g1']
        g2 = all_data['g2']

        # Check reading specific rows
        data2 = r.read(['ra','x','z'], s2)
        dec2 = r.read('dec', s2)
        assert sorted(data2.keys()) == ['ra','x','z']
        assert data2['ra'].size == 3
        assert data2['x'].size == 3
        assert data2['z'].size == 3
        assert dec2.size == 3
        assert data2['ra'][0] == 0.34044927
        assert data2['x'][2] == 0.01816738
        assert data2['z'][1] == 0.79008204

    # Again check things not allowed if not in context
    with assert_raises(RuntimeError):
        r.read([1,3,9], None)
    with assert_raises(RuntimeError):
        r.read([1,3,9])
    with assert_raises(RuntimeError):
        r.read('ra')
    r.nrows = None
    with assert_raises(RuntimeError):
        r.row_count(1, None)
    with assert_raises(RuntimeError):
        r.row_count()
    with assert_raises(RuntimeError):
        r.names(None)
    with assert_raises(RuntimeError):
        r.names()

    # Check writer too.
    with AsciiWriter(os.path.join('output','test_ascii_writer.dat'), precision=16) as w:
        w.write(['g1', 'g2'], [g1, g2], params={'test': True}, ext='g1g2')
    with AsciiReader(os.path.join('output','test_ascii_writer.dat')) as r:
        params = r.read_params(ext='g1g2')
        data = r.read_data(ext='g1g2')
    assert params['test']
    assert np.array_equal(data['g1'], g1)
    assert np.array_equal(data['g2'], g2)
    with AsciiReader(os.path.join('output','test_ascii_writer.dat')) as r:
        with assert_raises(OSError):
            params = r.read_params(ext='gg')

    # Test no ext name
    with AsciiWriter(os.path.join('output','test_ascii_writer.dat'), precision=16) as w:
        w.write(['g1', 'g2'], [g1, g2], params={'test': True})
    with AsciiReader(os.path.join('output','test_ascii_writer.dat')) as r:
        params = r.read_params()
        data = r.read_data()
    assert params['test']
    assert np.array_equal(data['g1'], g1)
    assert np.array_equal(data['g2'], g2)

    # And no params
    with AsciiWriter(os.path.join('output','test_ascii_writer.dat'), precision=16) as w:
        w.write(['g1', 'g2'], [g1, g2], ext='g1g2')
    with AsciiReader(os.path.join('output','test_ascii_writer.dat')) as r:
        params = r.read_params(ext='g1g2')
        data = r.read_data(ext='g1g2')
    assert params == {}
    assert np.array_equal(data['g1'], g1)
    assert np.array_equal(data['g2'], g2)

    # Neither
    with AsciiWriter(os.path.join('output','test_ascii_writer.dat'), precision=16) as w:
        w.write(['g1', 'g2'], [g1, g2])
    with AsciiReader(os.path.join('output','test_ascii_writer.dat')) as r:
        params = r.read_params()
        data = r.read_data()
    assert params == {}
    assert np.array_equal(data['g1'], g1)
    assert np.array_equal(data['g2'], g2)

    # Not allowed to write when not in with context
    w = AsciiWriter(os.path.join('output','test_ascii_writer.dat'), precision=16)
    with assert_raises(RuntimeError):
        w.write(['g1', 'g2'], [g1, g2], params={'test': True}, ext='g1g2')


@timer
def test_ascii_reader():
    # These all have the same data, but different comment lines
    _test_ascii_reader(AsciiReader(os.path.join('data','test1.dat')))
    _test_ascii_reader(AsciiReader(os.path.join('data','test2.dat')),False)
    _test_ascii_reader(AsciiReader(os.path.join('data','test3.dat')))

@timer
def test_pandas_reader():
    try:
        import pandas  # noqa: F401
    except ImportError:
        print('Skipping PandasReader tests, since pandas not installed.')
        return

    _test_ascii_reader(PandasReader(os.path.join('data','test1.dat')))
    _test_ascii_reader(PandasReader(os.path.join('data','test2.dat')),False)
    _test_ascii_reader(PandasReader(os.path.join('data','test3.dat')))

if __name__ == '__main__':
    test_fits_reader()
    test_hdf_reader()
    test_parquet_reader()
    test_ascii_reader()
    test_pandas_reader()
