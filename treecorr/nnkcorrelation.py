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

"""
.. module:: nnkcorrelation
"""

import numpy as np

from . import _treecorr
from .catalog import calculateVarK
from .corr3base import Corr3
from .util import lazy_property
from .config import make_minimal_config


class KNNCorrelation(Corr3):
    r"""This class handles the calculation and storage of a 3-point scalar-count-count correlation
    function, where as usual K represents any spin-0 scalar field.

    With this class, point 1 of the triangle (i.e. the vertex opposite d1) is the one with the
    scalar value.  Use `NKNCorrelation` and `NNKCorrelation` for classes with the scalar in the
    other two positions.

    See the doc string of `Corr3` for a description of how the triangles are binned along
    with the attributes related to the different binning options.

    In addition to the attributes common to all `Corr3` subclasses, objects of this class
    hold the following attributes:

    Attributes:
        zeta:       The correlation function, :math:`\zeta`.
        varzeta:    The variance estimate, only including the shot noise propagated into the
                    final correlation.

    The typical usage pattern is as follows:

        >>> knn = treecorr.KNNCorrelation(config)
        >>> knn.process(cat1, cat2)        # Compute cross-correlation of two fields.
        >>> knn.process(cat1, cat2, cat3)  # Compute cross-correlation of three fields.
        >>> krr.process(cat1, rand)        # Compute cross-correlation with randoms.
        >>> kdr.process(cat1, cat2, rand)  # Compute cross-correlation with randoms and data
        >>> knn.write(file_name)           # Write out to a file.
        >>> knn.calculateZeta(krr=krr, kdr=kdr) # Calculate zeta using randoms
        >>> zeta = knn.zeta                # Access correlation function
        >>> zetar = knn.zetar              # Access real and imaginary parts separately
        >>> zetai = knn.zetai

    Parameters:
        config (dict):  A configuration dict that can be used to pass in kwargs if desired.
                        This dict is allowed to have addition entries besides those listed
                        in `Corr3`, which are ignored here. (default: None)
        logger:         If desired, a logger object for logging. (default: None, in which case
                        one will be built according to the config dict's verbose level.)

    Keyword Arguments:
        **kwargs:       See the documentation for `Corr3` for the list of allowed keyword
                        arguments, which may be passed either directly or in the config dict.
    """
    _cls = 'KNNCorrelation'
    _letter1 = 'K'
    _letter2 = 'N'
    _letter3 = 'N'
    _letters = 'KNN'
    _builder = _treecorr.KNNCorr
    _calculateVar1 = staticmethod(calculateVarK)
    _calculateVar2 = lambda *args, **kwargs: None
    _calculateVar3 = lambda *args, **kwargs: None
    _sig1 = 'sig_k'
    _sig2 = None
    _sig3 = None
    _default_angle_slop = 1

    def __init__(self, config=None, *, logger=None, **kwargs):
        super().__init__(config, logger=logger, **kwargs)

        self._krr = None
        self._kdr = None
        self._krd = None
        shape = self.data_shape
        self._z[0] = np.zeros(shape, dtype=float)
        if self.bin_type == 'LogMultipole':
            self._z[1] = np.zeros(shape, dtype=float)
        self._zeta = None
        self._varzeta = None
        self.logger.debug('Finished building KNNCorr')

    @property
    def zeta(self):
        if self._zeta is None:
            if self._z[1].size:
                return self._z[0] + 1j * self._z[1]
            else:
                return self._z[0]
        else:
            return self._zeta

    def copy(self):
        ret = super().copy()
        # True is possible during read before we finish reading in these attributes.
        if self._krr is not None and self._krr is not True:
            ret._krr = self._krr.copy()
        if self._kdr is not None and self._kdr is not True:
            ret._kdr = self._kdr.copy()
        if self._krd is not None and self._krd is not True:
            ret._krd = self._krd.copy()
        return ret

    def finalize(self, vark):
        """Finalize the calculation of the correlation function.

        Parameters:
            vark (float):   The variance of the scalar field.
        """
        self._finalize()
        self._var_num = vark
        if self.bin_type in ['LogSAS', 'LogMultipole']:
            self._var_num *= 2

    @property
    def varzeta(self):
        if self._varzeta is None:
            self._varzeta = self._calculate_varzeta()
        return self._varzeta

    def _clear(self):
        super()._clear()
        self._krr = None
        self._krd = None
        self._kdr = None
        self._zeta = None
        self._varzeta = None

    def write(self, file_name, *, file_type=None, precision=None, write_patch_results=False,
              write_cov=False):
        super().write(file_name, file_type=file_type, precision=precision,
                      write_patch_results=write_patch_results, write_cov=write_cov)

    write.__doc__ = Corr3.write.__doc__.format(
        r"""
        zeta            The estimator of :math:`\zeta` (For LogMultipole, this is split
                        into real and imaginary parts, zetar and zetai.)
        sigma_zeta      The sqrt of the variance estimate of :math:`\zeta`.
        """)

    @property
    def _write_class_col_names(self):
        if self.bin_type == 'LogMultipole':
            return ['zetar', 'zetai', 'sigma_zeta']
        else:
            return ['zeta', 'sigma_zeta']

    @property
    def _write_class_data(self):
        if self.bin_type == 'LogMultipole':
            return [self._z[0], self._z[1], np.sqrt(self.varzeta) ]
        else:
            return [self.zeta, np.sqrt(self.varzeta)]

    def _read_from_data(self, data, params):
        super()._read_from_data(data, params)
        s = self.data_shape
        if self.bin_type == 'LogMultipole':
            self._z[0] = data['zetar'].reshape(s)
            self._z[1] = data['zetai'].reshape(s)
        else:
            self._z[0] = data['zeta'].reshape(s)
        self._varzeta = data['sigma_zeta'].reshape(s)**2

class NKNCorrelation(Corr3):
    r"""This class handles the calculation and storage of a 3-point count-scalar-count correlation
    function, where as usual K represents any spin-0 scalar field.

    With this class, point 2 of the triangle (i.e. the vertex opposite d2) is the one with the
    scalar value.  Use `KNNCorrelation` and `NNKCorrelation` for classes with the scalar in the
    other two positions.

    See the doc string of `Corr3` for a description of how the triangles are binned along
    with the attributes related to the different binning options.

    In addition to the attributes common to all `Corr3` subclasses, objects of this class
    hold the following attributes:

    Attributes:
        zeta:       The correlation function, :math:`\zeta`.
        varzeta:    The variance estimate, only including the shot noise propagated into the
                    final correlation.

    The typical usage pattern is as follows:

        >>> nkn = treecorr.KNNCorrelation(config)
        >>> nkn.process(cat1, cat2, cat1)  # Compute cross-correlation of two fields.
        >>> nkn.process(cat1, cat2, cat3)  # Compute cross-correlation of three fields.
        >>> rkr.process(rand, cat2, rand)  # Compute cross-correlation with randoms.
        >>> dkr.process(cat1, cat2, rand)  # Compute cross-correlation with randoms and data
        >>> nkn.write(file_name)           # Write out to a file.
        >>> nkn.calculateZeta(rkr=rkr, dkr=dkr) # Calculate zeta using randoms
        >>> zeta = nkn.zeta                # Access correlation function
        >>> zetar = nkn.zetar              # Access real and imaginary parts separately
        >>> zetai = nkn.zetai

    Parameters:
        config (dict):  A configuration dict that can be used to pass in kwargs if desired.
                        This dict is allowed to have addition entries besides those listed
                        in `Corr3`, which are ignored here. (default: None)
        logger:         If desired, a logger object for logging. (default: None, in which case
                        one will be built according to the config dict's verbose level.)

    Keyword Arguments:
        **kwargs:       See the documentation for `Corr3` for the list of allowed keyword
                        arguments, which may be passed either directly or in the config dict.
    """
    _cls = 'NKNCorrelation'
    _letter1 = 'N'
    _letter2 = 'K'
    _letter3 = 'N'
    _letters = 'NKN'
    _builder = _treecorr.NKNCorr
    _calculateVar1 = lambda *args, **kwargs: None
    _calculateVar2 = staticmethod(calculateVarK)
    _calculateVar3 = lambda *args, **kwargs: None
    _sig1 = None
    _sig2 = 'sig_k'
    _sig3 = None
    _default_angle_slop = 1

    def __init__(self, config=None, *, logger=None, **kwargs):
        super().__init__(config, logger=logger, **kwargs)

        self._rkr = None
        self._dkr = None
        self._rkd = None
        shape = self.data_shape
        self._z[0] = np.zeros(shape, dtype=float)
        if self.bin_type == 'LogMultipole':
            self._z[1] = np.zeros(shape, dtype=float)
        self._zeta = None
        self._varzeta = None
        self.logger.debug('Finished building NKNCorr')

    @property
    def zeta(self):
        if self._zeta is None:
            if self._z[1].size:
                return self._z[0] + 1j * self._z[1]
            else:
                return self._z[0]
        else:
            return self._zeta

    def copy(self):
        ret = super().copy()
        # True is possible during read before we finish reading in these attributes.
        if self._rkr is not None and self._rkr is not True:
            ret._rkr = self._rkr.copy()
        if self._dkr is not None and self._dkr is not True:
            ret._dkr = self._dkr.copy()
        if self._rkd is not None and self._rkd is not True:
            ret._rkd = self._rkd.copy()
        return ret

    def finalize(self, vark):
        """Finalize the calculation of the correlation function.

        Parameters:
            vark (float):   The variance of the scalar field.
        """
        self._finalize()
        self._var_num = vark
        if self.bin_type in ['LogSAS', 'LogMultipole']:
            self._var_num *= 2

    @property
    def varzeta(self):
        if self._varzeta is None:
            self._varzeta = self._calculate_varzeta()
        return self._varzeta

    def _clear(self):
        super()._clear()
        self._rkr = None
        self._rkd = None
        self._dkr = None
        self._zeta = None
        self._varzeta = None

    def write(self, file_name, *, file_type=None, precision=None, write_patch_results=False,
              write_cov=False):
        super().write(file_name, file_type=file_type, precision=precision,
                      write_patch_results=write_patch_results, write_cov=write_cov)

    write.__doc__ = Corr3.write.__doc__.format(
        r"""
        zeta            The estimator of :math:`\zeta` (For LogMultipole, this is split
                        into real and imaginary parts, zetar and zetai.)
        sigma_zeta      The sqrt of the variance estimate of :math:`\zeta`.
        """)

    @property
    def _write_class_col_names(self):
        if self.bin_type == 'LogMultipole':
            return ['zetar', 'zetai', 'sigma_zeta']
        else:
            return ['zeta', 'sigma_zeta']

    @property
    def _write_class_data(self):
        if self.bin_type == 'LogMultipole':
            return [self._z[0], self._z[1], np.sqrt(self.varzeta) ]
        else:
            return [self.zeta, np.sqrt(self.varzeta)]

    def _read_from_data(self, data, params):
        super()._read_from_data(data, params)
        s = self.data_shape
        if self.bin_type == 'LogMultipole':
            self._z[0] = data['zetar'].reshape(s)
            self._z[1] = data['zetai'].reshape(s)
        else:
            self._z[0] = data['zeta'].reshape(s)
        self._varzeta = data['sigma_zeta'].reshape(s)**2

class NNKCorrelation(Corr3):
    r"""This class handles the calculation and storage of a 3-point count-count-scalar correlation
    function, where as usual K represents any spin-0 scalar field.

    With this class, point 3 of the triangle (i.e. the vertex opposite d3) is the one with the
    scalar value.  Use `KNNCorrelation` and `NKNCorrelation` for classes with the scalar in the
    other two positions.

    See the doc string of `Corr3` for a description of how the triangles are binned along
    with the attributes related to the different binning options.

    In addition to the attributes common to all `Corr3` subclasses, objects of this class
    hold the following attributes:

    Attributes:
        zeta:       The correlation function, :math:`\zeta`.
        varzeta:    The variance estimate, only including the shot noise propagated into the
                    final correlation.

    The typical usage pattern is as follows:

        >>> nnk = treecorr.NNKCorrelation(config)
        >>> nnk.process(cat1, cat2)        # Compute cross-correlation of two fields.
        >>> nnk.process(cat1, cat2, cat3)  # Compute cross-correlation of three fields.
        >>> rrk.process(rand, cat2)        # Compute cross-correlation with randoms.
        >>> drk.process(cat1, rand, cat2)  # Compute cross-correlation with randoms and data
        >>> nnk.write(file_name)           # Write out to a file.
        >>> nnk.calculateZeta(rrk=rrk, drk=drk) # Calculate zeta using randoms
        >>> zeta = nnk.zeta                # Access correlation function
        >>> zetar = nnk.zetar              # Access real and imaginary parts separately
        >>> zetai = nnk.zetai

    Parameters:
        config (dict):  A configuration dict that can be used to pass in kwargs if desired.
                        This dict is allowed to have addition entries besides those listed
                        in `Corr3`, which are ignored here. (default: None)
        logger:         If desired, a logger object for logging. (default: None, in which case
                        one will be built according to the config dict's verbose level.)

    Keyword Arguments:
        **kwargs:       See the documentation for `Corr3` for the list of allowed keyword
                        arguments, which may be passed either directly or in the config dict.
    """
    _cls = 'NNKCorrelation'
    _letter1 = 'N'
    _letter2 = 'N'
    _letter3 = 'K'
    _letters = 'NNK'
    _builder = _treecorr.NNKCorr
    _calculateVar1 = lambda *args, **kwargs: None
    _calculateVar2 = lambda *args, **kwargs: None
    _calculateVar3 = staticmethod(calculateVarK)
    _sig1 = None
    _sig2 = None
    _sig3 = 'sig_k'
    _default_angle_slop = 1

    def __init__(self, config=None, *, logger=None, **kwargs):
        super().__init__(config, logger=logger, **kwargs)

        self._rrk = None
        self._drk = None
        self._rdk = None
        shape = self.data_shape
        self._z[0] = np.zeros(shape, dtype=float)
        if self.bin_type == 'LogMultipole':
            self._z[1] = np.zeros(shape, dtype=float)
        self._zeta = None
        self._varzeta = None
        self.logger.debug('Finished building NNKCorr')

    @property
    def zeta(self):
        if self._zeta is None:
            if self._z[1].size:
                return self._z[0] + 1j * self._z[1]
            else:
                return self._z[0]
        else:
            return self._zeta

    def copy(self):
        ret = super().copy()
        # True is possible during read before we finish reading in these attributes.
        if self._rrk is not None and self._rrk is not True:
            ret._rrk = self._rrk.copy()
        if self._drk is not None and self._drk is not True:
            ret._drk = self._drk.copy()
        if self._rdk is not None and self._rdk is not True:
            ret._rdk = self._rdk.copy()
        return ret

    def finalize(self, vark):
        """Finalize the calculation of the correlation function.

        Parameters:
            vark (float):   The variance of the scalar field.
        """
        self._finalize()
        self._var_num = vark
        if self.bin_type in ['LogSAS', 'LogMultipole']:
            self._var_num *= 2

    @property
    def varzeta(self):
        if self._varzeta is None:
            self._varzeta = self._calculate_varzeta()
        return self._varzeta

    def _clear(self):
        super()._clear()
        self._rrk = None
        self._rdk = None
        self._drk = None
        self._zeta = None
        self._varzeta = None

    def write(self, file_name, *, file_type=None, precision=None, write_patch_results=False,
              write_cov=False):
        super().write(file_name, file_type=file_type, precision=precision,
                      write_patch_results=write_patch_results, write_cov=write_cov)

    write.__doc__ = Corr3.write.__doc__.format(
        r"""
        zeta            The estimator of :math:`\zeta` (For LogMultipole, this is split
                        into real and imaginary parts, zetar and zetai.)
        sigma_zeta      The sqrt of the variance estimate of :math:`\zeta`.
        """)

    @property
    def _write_class_col_names(self):
        if self.bin_type == 'LogMultipole':
            return ['zetar', 'zetai', 'sigma_zeta']
        else:
            return ['zeta', 'sigma_zeta']

    @property
    def _write_class_data(self):
        if self.bin_type == 'LogMultipole':
            return [self._z[0], self._z[1], np.sqrt(self.varzeta) ]
        else:
            return [self.zeta, np.sqrt(self.varzeta)]

    def _read_from_data(self, data, params):
        super()._read_from_data(data, params)
        s = self.data_shape
        if self.bin_type == 'LogMultipole':
            self._z[0] = data['zetar'].reshape(s)
            self._z[1] = data['zetai'].reshape(s)
        else:
            self._z[0] = data['zeta'].reshape(s)
        self._varzeta = data['sigma_zeta'].reshape(s)**2
