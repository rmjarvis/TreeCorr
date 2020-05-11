from treecorr.catalog_formats import FitsReader, HdfReader
from test_helper import get_from_wiki, assert_raises, assert_warns
import os
import numpy as np

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

def test_context():
    get_from_wiki('Aardvark.fit')
    with FitsReader(os.path.join('data','Aardvark.fit')) as infile:
        pass


def test_fits_reader():
    r = _test_reader('Aardvark.fit', FitsReader, 1, 1, 0)
    # check we can also index by integer, not just number
    d = r.read('AARDWOLF', ['DEC'], np.arange(10))
    assert r.choose_extension({}, 'g1_ext', 0) == 1
    assert r.choose_extension({'g1_ext':'gg1'}, 'g1_ext', 0, 'ext') == 'gg1'
    assert r.choose_extension({'g1_ext':'gg1'}, 'g2_ext', 0, 0) == 0
    assert d.size==10

def test_hdf_reader():
    r = _test_reader('Aardvark.hdf5', HdfReader, '/', '/')
    r = _test_reader('Aardvark.hdf5', HdfReader, '', '/')
    assert r.choose_extension({'g1_ext': 'g1'}, 'g1_ext', 0) == 'g1'
    assert r.choose_extension({}, 'g1_ext', 0, 'ext') == 'ext'

def test_hdu_warning():
    num = 0
    with assert_warns(FutureWarning):
        ext = FitsReader.choose_extension({'hdu': 1}, 'ext', num)
        assert ext == 1
    with assert_warns(FutureWarning):
        ext = FitsReader.choose_extension({'x_hdu': 'hdu_name'}, 'x_ext', num)
        assert ext == 'hdu_name'

    with assert_warns(FutureWarning):
        ext = HdfReader.choose_extension({'hdu': 'potato'}, 'ext', num)
        assert ext == 'potato'
        ext = HdfReader.choose_extension({'ra_hdu': 'group_name'}, 'ra_ext', num)
        assert ext == 'group_name'



if __name__ == '__main__':
    test_hdu_warning()
    test_fits_reader()
    test_hdf_reader()
    test_context()
