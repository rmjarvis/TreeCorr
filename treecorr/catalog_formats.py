import numpy as np

class FitsReader:
    subgroup_type = int
    subgroup_default = 1
    def __init__(self, file_name):
        import fitsio
        # packaging is used by pip so is installed basically everywhere
        from packaging.version import parse
        self.file = fitsio.FITS(file_name, 'r')
        self.file_name = file_name
        self.can_slice = parse(fitsio.__version__) > parse('1.0.6')

    def read(self, hdu, cols, s):
        return self.file[hdu][cols][s]

    def row_count(self, hdu, col=None):
        return self.file[hdu].get_nrows()

    def names(self, hdu):
        return self.file[hdu].get_colnames()

    def __contains__(self, hdu):
        return hdu in self.file

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, exc_traceback):
        self.file.close()

    def check_valid_ext(self, hdu):
        import fitsio
        if not isinstance(self.file[hdu], fitsio.hdu.TableHDU):
            raise ValueError("Invalid hdu={} for file {} (Not a TableHDU)".format(
                             hdu,self.file_name))


class HdfReader:
    can_slice = True
    subgroup_type = str
    subgroup_default = '/'
    def __init__(self, file_name):
        import h5py
        self.file = h5py.File(file_name, 'r')

    def __contains__(self, ext):
        return ext in self.file.keys()

    def _group(self, group):
        # get a group from a name, using
        # the root if the group is empty
        if group == '':
            group = '/'
        return self.file[group]

    def check_valid_ext(self, hdu):
        return True

    def read(self, group, cols, s):
        g = self._group(group)
        print(group, cols, s)
        if np.isscalar(cols):
            data = g[cols][s]
        else:
            data = {col: g[col][s] for col in cols}
        print("Done")
        return data

    def row_count(self, group, col):
        return self._group(group)[col].size

    def names(self, group):
        return list(self._group(group).keys())

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, exc_traceback):
        self.file.close()