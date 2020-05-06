from treecorr.catalog_formats import FitsReader, HdfReader
from test_helper import get_from_wiki, assert_raises
import os
import numpy as np

def _test_reader(file_name, reader_class, ext, bad_ext='invalid'):
    get_from_wiki(file_name)
    file_name = os.path.join('data',file_name)
    reader = reader_class(file_name)

    assert_raises(ValueError, reader.check_valid_ext, bad_ext)
    assert_raises(ValueError, reader.check_valid_ext, 'invalid2')
    reader.check_valid_ext(ext)


    s = slice(0, 10, 2)
    data = reader.read(ext, ['RA'], s)
    assert data['RA'].size == 5
    assert reader.row_count(ext, 'RA') == 390935
    assert reader.row_count(ext, 'GAMMA1') == 390935

    assert set(reader.names(ext)) == set("INDEX RA DEC Z EPSILON GAMMA1 GAMMA2 KAPPA MU".split())
    return reader

def test_fits_reader():
    r = _test_reader('Aardvark.fit', FitsReader, 1, 0)
    # check we can also index by integer, not just number
    d = r.read('AARDWOLF', ['DEC'], np.arange(10))
    assert d.size==10

def test_hdf_reader():
    _test_reader('Aardvark.hdf5', HdfReader, '/')


if __name__ == '__main__':

    test_fits_reader()
    test_hdf_reader()