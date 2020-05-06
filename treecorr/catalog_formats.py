import numpy as np

class FitsReader:
    subgroup_type = int
    subgroup_default = 1
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
    subgroup_type = str
    subgroup_default = '/'
    def __init__(self, file_name):
        import h5py
        self.file = h5py.File(file_name, 'r')

    def _group(self, group):
        # get a group from a name, using
        # the root if the group is empty
        if group == '':
            g = self.file['/']
        else:
            g = self.file[group]
        return g


    def read(self, group, cols, s):
        g = self._group(group)

        if np.isscalar(cols):
            return g[cols][s]
        else:
            return {col: g[col][s] for col in cols}

    def row_count(self, group, col):
        return self._group(group)[col].size

    def names(self, group):
        return list(self._group(group).keys())

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, exc_traceback):
        self.file.close()