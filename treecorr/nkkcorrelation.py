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


class NKKCorrelation(Corr3):
    r"""This class handles the calculation and storage of a 3-point scalar-count-count correlation
    function, where as usual K represents any spin-0 scalar field.

    With this class, point 1 of the triangle (i.e. the vertex opposite d1) is the one with the
    scalar value.  Use `KNKCorrelation` and `KKNCorrelation` for classes with the scalar in the
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

        >>> nkk = treecorr.NKKCorrelation(config)
        >>> nkk.process(cat1, cat2)        # Compute cross-correlation of two fields.
        >>> nkk.process(cat1, cat2, cat3)  # Compute cross-correlation of three fields.
        >>> nkk.write(file_name)           # Write out to a file.
        >>> zeta = nkk.zeta                # Access correlation function
        >>> zetar = nkk.zetar              # Access real and imaginary parts separately
        >>> zetai = nkk.zetai

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
    _cls = 'NKKCorrelation'
    _letter1 = 'N'
    _letter2 = 'K'
    _letter3 = 'K'
    _letters = 'NKK'
    _builder = _treecorr.NKKCorr
    _calculateVar1 = lambda *args, **kwargs: None
    _calculateVar2 = staticmethod(calculateVarK)
    _calculateVar3 = staticmethod(calculateVarK)
    _sig1 = None
    _sig2 = 'sig_k'
    _sig3 = 'sig_k'
    _default_angle_slop = 1

    def __init__(self, config=None, *, logger=None, **kwargs):
        super().__init__(config, logger=logger, **kwargs)

        self._rkk = None
        shape = self.data_shape
        self._z[0] = np.zeros(shape, dtype=float)
        if self.bin_type == 'LogMultipole':
            self._z[1] = np.zeros(shape, dtype=float)
        self._zeta = None
        self._varzeta = None
        self.logger.debug('Finished building NKKCorr')

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
        if self._rkk is not None and self._rkk is not True:
            ret._rkk = self._rkk.copy()
        return ret

    def finalize(self, vark1, vark2):
        """Finalize the calculation of the correlation function.

        Parameters:
            vark1 (float):  The variance of the first scalar field.
            vark2 (float):  The variance of the second scalar field.
        """
        self._finalize()
        self._var_num = vark1 * vark2
        if self.bin_type in ['LogSAS', 'LogMultipole']:
            self._var_num *= 2

    @property
    def varzeta(self):
        if self._varzeta is None:
            self._varzeta = self._calculate_varzeta()
        return self._varzeta

    def _clear(self):
        super()._clear()
        self._rkk = None
        self._zeta = None
        self._varzeta = None

    def write(self, file_name, *, file_type=None, precision=None, write_patch_results=False,
              write_cov=False):
        super().write(file_name, file_type=file_type, precision=precision,
                      write_patch_results=write_patch_results, write_cov=write_cov)

    write.__doc__ = Corr3.write.__doc__.format(
        r"""
        zeta            The estimator of :math:`\zeta` (For LogMultipole, this is split
                        into real and imaginary parts, zeta_re and zeta_im.)
        sigma_zeta      The sqrt of the variance estimate of :math:`\zeta`.
        """)

    @property
    def _write_class_col_names(self):
        if self.bin_type == 'LogMultipole':
            return ['zeta_re', 'zeta_im', 'sigma_zeta']
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
            self._z[0] = data['zeta_re'].reshape(s)
            self._z[1] = data['zeta_im'].reshape(s)
        else:
            self._z[0] = data['zeta'].reshape(s)
        self._varzeta = data['sigma_zeta'].reshape(s)**2

class KNKCorrelation(Corr3):
    r"""This class handles the calculation and storage of a 3-point count-scalar-count correlation
    function, where as usual K represents any spin-0 scalar field.

    With this class, point 2 of the triangle (i.e. the vertex opposite d2) is the one with the
    scalar value.  Use `NKKCorrelation` and `KKNCorrelation` for classes with the scalar in the
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

        >>> knk = treecorr.NKKCorrelation(config)
        >>> knk.process(cat1, cat2, cat1)  # Compute cross-correlation of two fields.
        >>> knk.process(cat1, cat2, cat3)  # Compute cross-correlation of three fields.
        >>> knk.write(file_name)           # Write out to a file.
        >>> zeta = knk.zeta                # Access correlation function
        >>> zetar = knk.zetar              # Access real and imaginary parts separately
        >>> zetai = knk.zetai

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
    _cls = 'KNKCorrelation'
    _letter1 = 'K'
    _letter2 = 'N'
    _letter3 = 'K'
    _letters = 'KNK'
    _builder = _treecorr.KNKCorr
    _calculateVar1 = staticmethod(calculateVarK)
    _calculateVar2 = lambda *args, **kwargs: None
    _calculateVar3 = staticmethod(calculateVarK)
    _sig1 = 'sig_k'
    _sig2 = None
    _sig3 = 'sig_k'
    _default_angle_slop = 1

    def __init__(self, config=None, *, logger=None, **kwargs):
        super().__init__(config, logger=logger, **kwargs)

        self._krk = None
        shape = self.data_shape
        self._z[0] = np.zeros(shape, dtype=float)
        if self.bin_type == 'LogMultipole':
            self._z[1] = np.zeros(shape, dtype=float)
        self._zeta = None
        self._varzeta = None
        self.logger.debug('Finished building KNKCorr')

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
        if self._krk is not None and self._krk is not True:
            ret._krk = self._krk.copy()
        return ret

    def finalize(self, vark1, vark2):
        """Finalize the calculation of the correlation function.

        Parameters:
            vark1 (float):  The variance of the first scalar field.
            vark2 (float):  The variance of the second scalar field.
        """
        self._finalize()
        self._var_num = vark1 * vark2
        if self.bin_type in ['LogSAS', 'LogMultipole']:
            self._var_num *= 2

    @property
    def varzeta(self):
        if self._varzeta is None:
            self._varzeta = self._calculate_varzeta()
        return self._varzeta

    def _clear(self):
        super()._clear()
        self._krk = None
        self._zeta = None
        self._varzeta = None

    def write(self, file_name, *, file_type=None, precision=None, write_patch_results=False,
              write_cov=False):
        super().write(file_name, file_type=file_type, precision=precision,
                      write_patch_results=write_patch_results, write_cov=write_cov)

    write.__doc__ = Corr3.write.__doc__.format(
        r"""
        zeta            The estimator of :math:`\zeta` (For LogMultipole, this is split
                        into real and imaginary parts, zeta_re and zeta_im.)
        sigma_zeta      The sqrt of the variance estimate of :math:`\zeta`.
        """)

    @property
    def _write_class_col_names(self):
        if self.bin_type == 'LogMultipole':
            return ['zeta_re', 'zeta_im', 'sigma_zeta']
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
            self._z[0] = data['zeta_re'].reshape(s)
            self._z[1] = data['zeta_im'].reshape(s)
        else:
            self._z[0] = data['zeta'].reshape(s)
        self._varzeta = data['sigma_zeta'].reshape(s)**2

class KKNCorrelation(Corr3):
    r"""This class handles the calculation and storage of a 3-point count-count-scalar correlation
    function, where as usual K represents any spin-0 scalar field.

    With this class, point 3 of the triangle (i.e. the vertex opposite d3) is the one with the
    scalar value.  Use `NKKCorrelation` and `KNKCorrelation` for classes with the scalar in the
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

        >>> kkn = treecorr.KKNCorrelation(config)
        >>> kkn.process(cat1, cat2)        # Compute cross-correlation of two fields.
        >>> kkn.process(cat1, cat2, cat3)  # Compute cross-correlation of three fields.
        >>> kkn.write(file_name)           # Write out to a file.
        >>> zeta = kkn.zeta                # Access correlation function
        >>> zetar = kkn.zetar              # Access real and imaginary parts separately
        >>> zetai = kkn.zetai

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
    _cls = 'KKNCorrelation'
    _letter1 = 'K'
    _letter2 = 'K'
    _letter3 = 'N'
    _letters = 'KKN'
    _builder = _treecorr.KKNCorr
    _calculateVar1 = staticmethod(calculateVarK)
    _calculateVar2 = staticmethod(calculateVarK)
    _calculateVar3 = lambda *args, **kwargs: None
    _sig1 = 'sig_k'
    _sig2 = 'sig_k'
    _sig3 = None
    _default_angle_slop = 1

    def __init__(self, config=None, *, logger=None, **kwargs):
        super().__init__(config, logger=logger, **kwargs)

        self._kkr = None
        shape = self.data_shape
        self._z[0] = np.zeros(shape, dtype=float)
        if self.bin_type == 'LogMultipole':
            self._z[1] = np.zeros(shape, dtype=float)
        self._zeta = None
        self._varzeta = None
        self.logger.debug('Finished building KKNCorr')

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
        if self._kkr is not None and self._kkr is not True:
            ret._kkr = self._kkr.copy()
        return ret

    def finalize(self, vark1, vark2):
        """Finalize the calculation of the correlation function.

        Parameters:
            vark1 (float):  The variance of the first scalar field.
            vark2 (float):  The variance of the second scalar field.
        """
        self._finalize()
        self._var_num = vark1 * vark2
        if self.bin_type in ['LogSAS', 'LogMultipole']:
            self._var_num *= 2

    @property
    def varzeta(self):
        if self._varzeta is None:
            self._varzeta = self._calculate_varzeta()
        return self._varzeta

    def _clear(self):
        super()._clear()
        self._kkr = None
        self._zeta = None
        self._varzeta = None

    def write(self, file_name, *, file_type=None, precision=None, write_patch_results=False,
              write_cov=False):
        super().write(file_name, file_type=file_type, precision=precision,
                      write_patch_results=write_patch_results, write_cov=write_cov)

    write.__doc__ = Corr3.write.__doc__.format(
        r"""
        zeta            The estimator of :math:`\zeta` (For LogMultipole, this is split
                        into real and imaginary parts, zeta_re and zeta_im.)
        sigma_zeta      The sqrt of the variance estimate of :math:`\zeta`.
        """)

    @property
    def _write_class_col_names(self):
        if self.bin_type == 'LogMultipole':
            return ['zeta_re', 'zeta_im', 'sigma_zeta']
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
            self._z[0] = data['zeta_re'].reshape(s)
            self._z[1] = data['zeta_im'].reshape(s)
        else:
            self._z[0] = data['zeta'].reshape(s)
        self._varzeta = data['sigma_zeta'].reshape(s)**2
