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
.. module:: kkgcorrelation
"""

import numpy as np

from . import _treecorr
from .catalog import calculateVarK, calculateVarG
from .corr3base import Corr3


class KKGCorrelation(Corr3):
    r"""This class handles the calculation and storage of a 3-point scalar-scalar-shear correlation
    function.

    See the docstring of `Corr3` for a description of how the triangles
    are binned.

    With this class, point 3 of the triangle (i.e. the vertex opposite d3) is the one with the
    shear value.  Use `KGKCorrelation` and `GKKCorrelation` for classes with the shear in the
    other two positions.

    See the docstring of `Corr3` for a description of how the triangles are binned along
    with the attributes related to the different binning options.

    In addition to the attributes common to all `Corr3` subclasses, objects of this class
    hold the following attributes:

    Attributes:
        zeta:       The correlation function, :math:`\zeta`.
        varzeta:    The variance estimate of :math:`\zeta`, computed according to ``var_method``
                    (default: ``'shot'``).

    The typical usage pattern is as follows::

        >>> kkg = treecorr.KKGCorrelation(config)
        >>> kkg.process(cat1, cat2)         # Compute the cross-correlation of two fields.
        >>> # kkg.process(cat1, cat2, cat3) # ... or of three fields.
        >>> kkg.write(file_name)            # Write out to a file.
        >>> zeta = kkg.zeta                 # Access the correlation function.
        >>> zetar = kkg.zetar               # Or access real and imaginary parts separately.
        >>> zetai = kkg.zetai

    See also: `KGKCorrelation`, `GKKCorrelation`, `KGGCorrelation`, `KKKCorrelation`,
    `KGCorrelation`.

    Parameters:
        config (dict):  A configuration dict that can be used to pass in kwargs if desired.
                        This dict is allowed to have additional entries besides those listed
                        in `Corr3`, which are ignored here. (default: None)
        logger (:class:`logging.Logger`):
                        If desired, a ``Logger`` object for logging. (default: None, in which case
                        one will be built according to the config dict's verbose level.)

    Keyword Arguments:
        **kwargs:       See the documentation for `Corr3` for the list of allowed keyword
                        arguments, which may be passed either directly or in the config dict.
    """
    _cls = 'KKGCorrelation'
    _letter1 = 'K'
    _letter2 = 'K'
    _letter3 = 'G'
    _letters = 'KKG'
    _builder = _treecorr.KKGCorr
    _calculateVar1 = staticmethod(calculateVarK)
    _calculateVar2 = staticmethod(calculateVarK)
    _calculateVar3 = staticmethod(calculateVarG)
    _sig1 = 'sig_k'
    _sig2 = 'sig_k'
    _sig3 = 'sig_sn (per component)'

    def __init__(self, config=None, *, logger=None, **kwargs):
        super().__init__(config, logger=logger, **kwargs)

        shape = self.data_shape
        self._z[0:2] = [np.zeros(shape, dtype=float) for _ in range(2)]
        self.logger.debug('Finished building KKGCorr')

    @property
    def zeta(self):
        return self._z[0] + 1j * self._z[1]

    @property
    def zetar(self):
        return self._z[0]

    @property
    def zetai(self):
        return self._z[1]

    def finalize(self, vark1, vark2, varg):
        """Finalize the calculation of the correlation function.

        Parameters:
            vark1 (float):  The variance of the first scalar field.
            vark2 (float):  The variance of the second scalar field.
            varg (float):   The variance per component of the shear field.
        """
        self._finalize()
        self._var_num = vark1 * vark2 * varg
        if self.bin_type in ['LogSAS', 'LogMultipole']:
            self._var_num *= 2

    @property
    def varzeta(self):
        if self._varzeta is None:
            self._calculate_varzeta(1)
        return self._varzeta[0]

    def write(self, file_name, *, file_type=None, precision=None, write_patch_results=False,
              write_cov=False):
        super().write(file_name, file_type=file_type, precision=precision,
                      write_patch_results=write_patch_results, write_cov=write_cov)

    write.__doc__ = Corr3.write.__doc__.format(
        r"""
        zetar           The real part of the estimator of :math:`\zeta`
        zetai           The imaginary part of the estimator of :math:`\zeta`
        sigma_zeta      The sqrt of the variance estimate of :math:`\zeta`
        """)

    @property
    def _write_class_col_names(self):
        return ['zetar', 'zetai', 'sigma_zeta']

    @property
    def _write_class_data(self):
        return [ self.zetar, self.zetai, np.sqrt(self.varzeta) ]

    def _read_from_data(self, data, params):
        super()._read_from_data(data, params)
        s = self.data_shape
        self._z[0] = data['zetar'].reshape(s)
        self._z[1] = data['zetai'].reshape(s)
        self._varzeta = [data['sigma_zeta'].reshape(s)**2]

class KGKCorrelation(Corr3):
    r"""This class handles the calculation and storage of a 3-point scalar-shear-scalar correlation
    function.

    See the docstring of `Corr3` for a description of how the triangles
    are binned.

    With this class, point 2 of the triangle (i.e. the vertex opposite d2) is the one with the
    shear value.  Use `KKGCorrelation` and `GKKCorrelation` for classes with the shear in the
    other two positions.

    See the docstring of `Corr3` for a description of how the triangles are binned along
    with the attributes related to the different binning options.

    In addition to the attributes common to all `Corr3` subclasses, objects of this class
    hold the following attributes:

    Attributes:
        zeta:       The correlation function, :math:`\zeta`.
        varzeta:    The variance estimate of :math:`\zeta`, computed according to ``var_method``
                    (default: ``'shot'``).

    The typical usage pattern is as follows::

        >>> kgk = treecorr.KGKCorrelation(config)
        >>> kgk.process(cat1, cat2, cat1)   # Compute the cross-correlation of two fields.
        >>> # kgk.process(cat1, cat2, cat3) # ... or of three fields.
        >>> kgk.write(file_name)            # Write out to a file.
        >>> zeta = kgk.zeta                 # Access the correlation function.
        >>> zetar = kgk.zetar               # Or access real and imaginary parts separately.
        >>> zetai = kgk.zetai

    See also: `KKGCorrelation`, `GKKCorrelation`, `KGGCorrelation`, `KKKCorrelation`,
    `KGCorrelation`.

    Parameters:
        config (dict):  A configuration dict that can be used to pass in kwargs if desired.
                        This dict is allowed to have additional entries besides those listed
                        in `Corr3`, which are ignored here. (default: None)
        logger (:class:`logging.Logger`):
                        If desired, a ``Logger`` object for logging. (default: None, in which case
                        one will be built according to the config dict's verbose level.)

    Keyword Arguments:
        **kwargs:       See the documentation for `Corr3` for the list of allowed keyword
                        arguments, which may be passed either directly or in the config dict.
    """
    _cls = 'KGKCorrelation'
    _letter1 = 'K'
    _letter2 = 'G'
    _letter3 = 'K'
    _letters = 'KGK'
    _builder = _treecorr.KGKCorr
    _calculateVar1 = staticmethod(calculateVarK)
    _calculateVar2 = staticmethod(calculateVarG)
    _calculateVar3 = staticmethod(calculateVarK)
    _sig1 = 'sig_k'
    _sig2 = 'sig_sn (per component)'
    _sig3 = 'sig_k'

    def __init__(self, config=None, *, logger=None, **kwargs):
        super().__init__(config, logger=logger, **kwargs)

        shape = self.data_shape
        self._z[0:2] = [np.zeros(shape, dtype=float) for _ in range(2)]
        self.logger.debug('Finished building KGKCorr')

    @property
    def zeta(self):
        return self._z[0] + 1j * self._z[1]

    @property
    def zetar(self):
        return self._z[0]

    @property
    def zetai(self):
        return self._z[1]

    def finalize(self, vark1, varg, vark2):
        """Finalize the calculation of the correlation function.

        Parameters:
            vark1 (float):  The variance of the first scalar field.
            varg (float):   The variance per component of the shear field.
            vark2 (float):  The variance of the second scalar field.
        """
        self._finalize()
        self._var_num = vark1 * vark2 * varg
        if self.bin_type in ['LogSAS', 'LogMultipole']:
            self._var_num *= 2

    @property
    def varzeta(self):
        if self._varzeta is None:
            self._calculate_varzeta(1)
        return self._varzeta[0]

    def write(self, file_name, *, file_type=None, precision=None, write_patch_results=False,
              write_cov=False):
        super().write(file_name, file_type=file_type, precision=precision,
                      write_patch_results=write_patch_results, write_cov=write_cov)

    write.__doc__ = Corr3.write.__doc__.format(
        r"""
        zetar           The real part of the estimator of :math:`\zeta`
        zetai           The imaginary part of the estimator of :math:`\zeta`
        sigma_zeta      The sqrt of the variance estimate of :math:`\zeta`
        """)

    @property
    def _write_class_col_names(self):
        return ['zetar', 'zetai', 'sigma_zeta']

    @property
    def _write_class_data(self):
        return [ self.zetar, self.zetai, np.sqrt(self.varzeta) ]

    def _read_from_data(self, data, params):
        super()._read_from_data(data, params)
        s = self.data_shape
        self._z[0] = data['zetar'].reshape(s)
        self._z[1] = data['zetai'].reshape(s)
        self._varzeta = [data['sigma_zeta'].reshape(s)**2]

class GKKCorrelation(Corr3):
    r"""This class handles the calculation and storage of a 3-point shear-scalar-scalar correlation
    function.

    See the docstring of `Corr3` for a description of how the triangles
    are binned.

    With this class, point 1 of the triangle (i.e. the vertex opposite d1) is the one with the
    shear value.  Use `KGKCorrelation` and `KKGCorrelation` for classes with the shear in the
    other two positions.

    See the docstring of `Corr3` for a description of how the triangles are binned along
    with the attributes related to the different binning options.

    In addition to the attributes common to all `Corr3` subclasses, objects of this class
    hold the following attributes:

    Attributes:
        zeta:       The correlation function, :math:`\zeta`.
        varzeta:    The variance estimate of :math:`\zeta`, computed according to ``var_method``
                    (default: ``'shot'``).

    The typical usage pattern is as follows::

        >>> gkk = treecorr.GKKCorrelation(config)
        >>> gkk.process(cat1, cat2)         # Compute the cross-correlation of two fields.
        >>> # gkk.process(cat1, cat2, cat3) # ... or of three fields.
        >>> gkk.write(file_name)            # Write out to a file.
        >>> zeta = gkk.zeta                 # Access the correlation function.
        >>> zetar = gkk.zetar               # Or access real and imaginary parts separately.
        >>> zetai = gkk.zetai

    See also: `KKGCorrelation`, `KGKCorrelation`, `KGGCorrelation`, `KKKCorrelation`,
    `KGCorrelation`.

    Parameters:
        config (dict):  A configuration dict that can be used to pass in kwargs if desired.
                        This dict is allowed to have additional entries besides those listed
                        in `Corr3`, which are ignored here. (default: None)
        logger (:class:`logging.Logger`):
                        If desired, a ``Logger`` object for logging. (default: None, in which case
                        one will be built according to the config dict's verbose level.)

    Keyword Arguments:
        **kwargs:       See the documentation for `Corr3` for the list of allowed keyword
                        arguments, which may be passed either directly or in the config dict.
    """
    _cls = 'GKKCorrelation'
    _letter1 = 'G'
    _letter2 = 'K'
    _letter3 = 'K'
    _letters = 'GKK'
    _builder = _treecorr.GKKCorr
    _calculateVar1 = staticmethod(calculateVarG)
    _calculateVar2 = staticmethod(calculateVarK)
    _calculateVar3 = staticmethod(calculateVarK)
    _sig1 = 'sig_sn (per component)'
    _sig2 = 'sig_k'
    _sig3 = 'sig_k'

    def __init__(self, config=None, *, logger=None, **kwargs):
        super().__init__(config, logger=logger, **kwargs)

        shape = self.data_shape
        self._z[0:2] = [np.zeros(shape, dtype=float) for _ in range(2)]
        self.logger.debug('Finished building GKKCorr')

    @property
    def zeta(self):
        return self._z[0] + 1j * self._z[1]

    @property
    def zetar(self):
        return self._z[0]

    @property
    def zetai(self):
        return self._z[1]

    def finalize(self, varg, vark1, vark2):
        """Finalize the calculation of the correlation function.

        Parameters:
            varg (float):   The variance per component of the shear field.
            vark1 (float):  The variance of the first scalar field.
            vark2 (float):  The variance of the second scalar field.
        """
        self._finalize()
        self._var_num = vark1 * vark2 * varg
        if self.bin_type in ['LogSAS', 'LogMultipole']:
            self._var_num *= 2

    @property
    def varzeta(self):
        if self._varzeta is None:
            self._calculate_varzeta(1)
        return self._varzeta[0]

    def write(self, file_name, *, file_type=None, precision=None, write_patch_results=False,
              write_cov=False):
        super().write(file_name, file_type=file_type, precision=precision,
                      write_patch_results=write_patch_results, write_cov=write_cov)

    write.__doc__ = Corr3.write.__doc__.format(
        r"""
        zetar           The real part of the estimator of :math:`\zeta`
        zetai           The imaginary part of the estimator of :math:`\zeta`
        sigma_zeta      The sqrt of the variance estimate of :math:`\zeta`
        """)

    @property
    def _write_class_col_names(self):
        return ['zetar', 'zetai', 'sigma_zeta']

    @property
    def _write_class_data(self):
        return [ self.zetar, self.zetai, np.sqrt(self.varzeta) ]

    def _read_from_data(self, data, params):
        super()._read_from_data(data, params)
        s = self.data_shape
        self._z[0] = data['zetar'].reshape(s)
        self._z[1] = data['zetai'].reshape(s)
        self._varzeta = [data['sigma_zeta'].reshape(s)**2]
