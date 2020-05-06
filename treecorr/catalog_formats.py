import numpy as np

class FitsReader:
    def __init__(self, file_name):
        import fitsio
        self.file = fitsio.FITS(file_name, 'r')

    def read(self, hdu, cols, s):
        return self.file[hdu][cols][s]

    def row_count(self, hdu, col=None):
        return self.file[hdu].get_nrows()

    def names(self, hdu):
        return self.file[hdu].get_colnames()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, exc_traceback):
        self.file.close()

class HdfReader:
    def __init__(self, file_name):
        import h5py
        self.file = h5py.File(file_name, 'r')

    def read(self, group, cols, s):
        if np.isscalar(cols):
            return self.file[group][cols][s]
        else:
            return {col: self.file[group][col][s] for col in cols}

    def row_count(self, group, col):
        return self.file[group][col].size

    def names(self, group):
        return list(self.file[group].keys())

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, exc_traceback):
        self.file.close()