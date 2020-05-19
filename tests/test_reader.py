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
import treecorr

from treecorr.reader import FitsReader, HdfReader
from test_helper import get_from_wiki, assert_raises, assert_warns, timer

def _test_reader(file_name, reader_class, ext, def_ext, bad_ext='invalid'):
    get_from_wiki(file_name)
    file_name = os.path.join('data',file_name)
    reader = reader_class(file_name)

    assert_raises(ValueError, reader.check_valid_ext, bad_ext)
    assert_raises(ValueError, reader.check_valid_ext, 'invalid2')
    reader.check_valid_ext(ext)

    assert reader.choose_extension({}, 'ext', 0) == def_ext


    s = slice(0, 10, 2)
    data = reader.read(ext, ['RA'], s)
    dec = reader.read(ext, 'DEC', s)
    assert data['RA'].size == 5
    assert dec.size == 5
    assert reader.row_count(ext, 'RA') == 390935
    assert reader.row_count(ext, 'GAMMA1') == 390935

    assert set(reader.names(ext)) == set("INDEX RA DEC Z EPSILON GAMMA1 GAMMA2 KAPPA MU".split())
    return reader


@timer
def test_fits_reader():
    r = _test_reader('Aardvark.fit', FitsReader, 'AARDWOLF', 1)
    r = _test_reader('Aardvark.fit', FitsReader, 1, 1, 0)
    # check we can also index by integer, not just number
    d = r.read('AARDWOLF', ['DEC'], np.arange(10))
    assert r.choose_extension({}, 'g1_ext', 0) == 1
    assert r.choose_extension({'g1_ext':'gg1'}, 'g1_ext', 0, 'ext') == 'gg1'
    assert r.choose_extension({'g1_ext':'gg1'}, 'g2_ext', 0, 0) == 0
    assert d.size==10


@timer
def test_hdf_reader():
    r = _test_reader('Aardvark.hdf5', HdfReader, '/', '/')
    r = _test_reader('Aardvark.hdf5', HdfReader, '', '/')
    assert r.choose_extension({'g1_ext': 'g1'}, 'g1_ext', 0) == 'g1'
    assert r.choose_extension({}, 'g1_ext', 0, 'ext') == 'ext'


@timer
def test_context():
    get_from_wiki('Aardvark.fit')
    with FitsReader(os.path.join('data','Aardvark.fit')) as infile:
        pass

@timer
def test_can_slice():
    get_from_wiki('Aardvark.hdf5')
    with HdfReader(os.path.join('data','Aardvark.hdf5')) as infile:
        assert infile.can_slice

    if sys.version_info < (3,): return  # mock only available on python 3
    from unittest import mock

    # Regardless of the system's fitsio version, check the two cases in code.
    get_from_wiki('Aardvark.fit')
    with mock.patch('fitsio.__version__', '1.0.6'):
        with FitsReader(os.path.join('data','Aardvark.fit')) as infile:
            assert not infile.can_slice
    with mock.patch('fitsio.__version__', '1.1.0'):
        with FitsReader(os.path.join('data','Aardvark.fit')) as infile:
            assert infile.can_slice


if __name__ == '__main__':
    test_fits_reader()
    test_hdf_reader()
    test_context()
    test_can_slice()
