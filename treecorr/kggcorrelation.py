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
.. module:: nnncorrelation
"""

import numpy as np

from . import _treecorr
from .catalog import calculateVarK, calculateVarG
from .corr3base import Corr3
from .config import make_minimal_config


class KGGCorrelation(Corr3):
    r"""This class handles the calculation and storage of a 3-point scalar-shear-shear correlation
    function.

    With this class, point 1 of the triangle (i.e. the vertex opposite d1) is the one with the
    scalar value.  Use `GKGCorrelation` and `GGKCorrelation` for classes with the scalar in the
    other two positions.

    For the shear projection, we follow the lead of the 3-point shear-shear-shear correlation
    functions (see `GGGCorrelation` for details), which involves projecting the shear values
    at each vertex relative to the direction to the triangle's centroid.  Furthermore, the
    GGG correlations have 4 relevant complex values for each triangle:

    .. math::

        \Gamma_0 &= \langle \gamma(\mathbf{x1}) \gamma(\mathbf{x2}) \gamma(\mathbf{x3}) \rangle \\
        \Gamma_1 &= \langle \gamma(\mathbf{x1})^* \gamma(\mathbf{x2}) \gamma(\mathbf{x3}) \rangle \\
        \Gamma_2 &= \langle \gamma(\mathbf{x1}) \gamma(\mathbf{x2})^* \gamma(\mathbf{x3}) \rangle \\
        \Gamma_3 &= \langle \gamma(\mathbf{x1}) \gamma(\mathbf{x2}) \gamma(\mathbf{x3}^*) \rangle \\

    With a scalar value at vertex 1, :math:`\Gamma_0 = \Gamma_1` and :math:`\Gamma_2 = \Gamma_3^*`.
    So there are only two independent values.  However, you may access these values using whichever
    names you find most convenient: ``gam0``, ``gam1``, ``gam2`` and ``gam3`` are all valid
    attributes, which return the corresponding value.

    See the doc string of `Corr3` for a description of how the triangles are binned along
    with the attributes related to the different binning options.

    In addition to the attributes common to all `Corr3` subclasses, objects of this class
    hold the following attributes:

    Attributes:
        gam0:       The 0th "natural" correlation function, :math:`\Gamma_0`.
        gam1:       The 1st "natural" correlation function, :math:`\Gamma_1`.
        gam2:       The 2nd "natural" correlation function, :math:`\Gamma_2`.
        gam3:       The 3rd "natural" correlation function, :math:`\Gamma_3`.
        vargam0:    The variance estimate of :math:`\Gamma_0`, only including the shot noise.
        vargam1:    The variance estimate of :math:`\Gamma_1`, only including the shot noise.
        vargam2:    The variance estimate of :math:`\Gamma_2`, only including the shot noise.
        vargam3:    The variance estimate of :math:`\Gamma_3`, only including the shot noise.

    The typical usage pattern is as follows::

        >>> kgg = treecorr.KGGCorrelation(config)
        >>> kgg.process(cat)              # For auto-correlation.
        >>> kgg.process(cat1,cat2,cat3)   # For cross-correlation.
        >>> kgg.write(file_name)          # Write out to a file.
        >>> gam0 = kgg.gam0, etc.         # To access gamma values directly.
        >>> gam0r = kgg.gam0r             # You can also access real and imag parts separately.
        >>> gam0i = kgg.gam0i

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
    _cls = 'KGGCorrelation'
    _letter1 = 'K'
    _letter2 = 'G'
    _letter3 = 'G'
    _letters = 'KGG'
    _builder = _treecorr.KGGCorr
    _calculateVar1 = staticmethod(calculateVarK)
    _calculateVar2 = staticmethod(calculateVarG)
    _calculateVar3 = staticmethod(calculateVarG)
    _sig1 = 'sig_k'
    _sig2 = 'sig_sn (per component)'
    _sig3 = 'sig_sn (per component)'

    def __init__(self, config=None, *, logger=None, **kwargs):
        super().__init__(config, logger=logger, **kwargs)

        shape = self.data_shape
        # z0,1 holds gamma_0
        # z2,3 holds gamma_2
        self._z[0:4] = [np.zeros(shape, dtype=float) for _ in range(4)]
        self._vargam0 = None
        self._vargam2 = None
        self.logger.debug('Finished building KGGCorr')

    @property
    def gam0(self):
        return self._z[0] + 1j * self._z[1]

    @property
    def gam1(self):
        return self._z[0] + 1j * self._z[1]

    @property
    def gam2(self):
        return self._z[2] + 1j * self._z[3]

    @property
    def gam3(self):
        return self._z[2] - 1j * self._z[3]

    @property
    def gam0r(self):
        return self._z[0]

    @property
    def gam0i(self):
        return self._z[1]

    @property
    def gam1r(self):
        return self._z[0]

    @property
    def gam1i(self):
        return self._z[1]

    @property
    def gam2r(self):
        return self._z[2]

    @property
    def gam2i(self):
        return self._z[3]

    @property
    def gam3r(self):
        return self._z[2]

    @property
    def gam3i(self):
        return -self._z[3]

    def finalize(self, vark, varg1, varg2):
        """Finalize the calculation of the correlation function.

        Parameters:
            vark (float):   The variance of the scalar field.
            varg1 (float):  The variance per component of the first shear field.
            varg2 (float):  The variance per component of the second shear field.
        """
        self._finalize()
        mask1 = self.weightr != 0
        mask2 = self.weightr == 0
        self._var_num = 2 * vark * varg1 * varg2

        if self.bin_type in ['LogSAS', 'LogMultipole']:
            self._var_num *= 2

    @property
    def vargam0(self):
        if self._vargam0 is None:
            self._vargam0 = self._calculate_varzeta(0, self._nbins)
        return self._vargam0

    @property
    def vargam1(self):
        return self.vargam0

    @property
    def vargam2(self):
        if self._vargam2 is None:
            self._vargam2 = self._calculate_varzeta(self._nbins, 2*self._nbins)
        return self._vargam2

    @property
    def vargam3(self):
        return self.vargam2

    def _clear(self):
        super()._clear()
        self._vargam0 = None
        self._vargam2 = None

    def getStat(self):
        """The standard statistic for the current correlation object as a 1-d array.

        In this case, the concatenation of gam0.ravel() and gam2.ravel().
        """
        return np.concatenate([self.gam0.ravel(), self.gam2.ravel()])

    def getWeight(self):
        """The weight array for the current correlation object as a 1-d array.

        In this case, 2 copies of self.weight.ravel().
        """
        return np.concatenate([np.abs(self.weight.ravel())] * 2)

    def write(self, file_name, *, file_type=None, precision=None, write_patch_results=False,
              write_cov=False):
        super().write(file_name, file_type=file_type, precision=precision,
                      write_patch_results=write_patch_results, write_cov=write_cov)

    write.__doc__ = Corr3.write.__doc__.format(
        r"""
        gam0r           The real part of the estimator of :math:`\Gamma_0`
        gam0i           The imag part of the estimator of :math:`\Gamma_0`
        gam2r           The real part of the estimator of :math:`\Gamma_2`
        gam2i           The imag part of the estimator of :math:`\Gamma_2`
        sigma_gam0      The sqrt of the variance estimate of :math:`\Gamma_0`
        sigma_gam2      The sqrt of the variance estimate of :math:`\Gamma_2`
        """)

    @property
    def _write_class_col_names(self):
        return ['gam0r', 'gam0i', 'gam2r', 'gam2i', 'sigma_gam0', 'sigma_gam2']

    @property
    def _write_class_data(self):
        return [self.gam0r, self.gam0i, self.gam2r, self.gam2i,
                np.sqrt(self.vargam0), np.sqrt(self.vargam2)]

    def _read_from_data(self, data, params):
        super()._read_from_data(data, params)
        s = self.data_shape
        self._z[0] = data['gam0r'].reshape(s)
        self._z[1] = data['gam0i'].reshape(s)
        self._z[2] = data['gam2r'].reshape(s)
        self._z[3] = data['gam2i'].reshape(s)
        self._vargam0 = data['sigma_gam0'].reshape(s)**2
        self._vargam2 = data['sigma_gam2'].reshape(s)**2

class GKGCorrelation(Corr3):
    r"""This class handles the calculation and storage of a 3-point shear-scalar-shear correlation
    function.

    With this class, point 2 of the triangle (i.e. the vertex opposite d2) is the one with the
    shear value.  Use `KGGCorrelation` and `GGKCorrelation` for classes with the shear in the
    other two positions.

    For the shear projection, we follow the lead of the 3-point shear-shear-shear correlation
    functions (see `GGGCorrelation` for details), which involves projecting the shear values
    at each vertex relative to the direction to the triangle's centroid.  Furthermore, the
    GGG correlations have 4 relevant complex values for each triangle:

    .. math::

        \Gamma_0 &= \langle \gamma(\mathbf{x1}) \gamma(\mathbf{x2}) \gamma(\mathbf{x3}) \rangle \\
        \Gamma_1 &= \langle \gamma(\mathbf{x1})^* \gamma(\mathbf{x2}) \gamma(\mathbf{x3}) \rangle \\
        \Gamma_2 &= \langle \gamma(\mathbf{x1}) \gamma(\mathbf{x2})^* \gamma(\mathbf{x3}) \rangle \\
        \Gamma_3 &= \langle \gamma(\mathbf{x1}) \gamma(\mathbf{x2}) \gamma(\mathbf{x3}^*) \rangle \\

    With a scalar value at vertex 2, :math:`\Gamma_0 = \Gamma_2` and :math:`\Gamma_1 = \Gamma_3^*`.
    So there are only two independent values.  However, you may access these values using whichever
    names you find most convenient: ``gam0``, ``gam1``, ``gam2`` and ``gam3`` are all valid
    attributes, which return the corresponding value.

    See the doc string of `Corr3` for a description of how the triangles are binned along
    with the attributes related to the different binning options.

    In addition to the attributes common to all `Corr3` subclasses, objects of this class
    hold the following attributes:

    Attributes:
        gam0:       The 0th "natural" correlation function, :math:`\Gamma_0`.
        gam1:       The 1st "natural" correlation function, :math:`\Gamma_1`.
        gam2:       The 2nd "natural" correlation function, :math:`\Gamma_2`.
        gam3:       The 3rd "natural" correlation function, :math:`\Gamma_3`.
        vargam0:    The variance estimate of :math:`\Gamma_0`, only including the shot noise.
        vargam1:    The variance estimate of :math:`\Gamma_1`, only including the shot noise.
        vargam2:    The variance estimate of :math:`\Gamma_2`, only including the shot noise.
        vargam3:    The variance estimate of :math:`\Gamma_3`, only including the shot noise.

    The typical usage pattern is as follows::

        >>> gkg = treecorr.GKGCorrelation(config)
        >>> gkg.process(cat)              # For auto-correlation.
        >>> gkg.process(cat1,cat2,cat3)   # For cross-correlation.
        >>> gkg.write(file_name)          # Write out to a file.
        >>> gam0 = gkg.gam0, etc.         # To access gamma values directly.
        >>> gam0r = gkg.gam0r             # You can also access real and imag parts separately.
        >>> gam0i = gkg.gam0i

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
    _cls = 'GKGCorrelation'
    _letter1 = 'G'
    _letter2 = 'K'
    _letter3 = 'G'
    _letters = 'GKG'
    _builder = _treecorr.GKGCorr
    _calculateVar1 = staticmethod(calculateVarG)
    _calculateVar2 = staticmethod(calculateVarK)
    _calculateVar3 = staticmethod(calculateVarG)
    _sig1 = 'sig_sn (per component)'
    _sig2 = 'sig_k'
    _sig3 = 'sig_sn (per component)'

    def __init__(self, config=None, *, logger=None, **kwargs):
        super().__init__(config, logger=logger, **kwargs)

        shape = self.data_shape
        self._z[0:4] = [np.zeros(shape, dtype=float) for _ in range(4)]
        self._vargam0 = None
        self._vargam1 = None
        self.logger.debug('Finished building GKGCorr')

    @property
    def gam0(self):
        return self._z[0] + 1j * self._z[1]

    @property
    def gam1(self):
        return self._z[2] + 1j * self._z[3]

    @property
    def gam2(self):
        return self._z[0] + 1j * self._z[1]

    @property
    def gam3(self):
        return self._z[2] - 1j * self._z[3]

    @property
    def gam0r(self):
        return self._z[0]

    @property
    def gam0i(self):
        return self._z[1]

    @property
    def gam1r(self):
        return self._z[2]

    @property
    def gam1i(self):
        return self._z[3]

    @property
    def gam2r(self):
        return self._z[0]

    @property
    def gam2i(self):
        return self._z[1]

    @property
    def gam3r(self):
        return self._z[2]

    @property
    def gam3i(self):
        return -self._z[3]

    def finalize(self, varg1, vark, varg2):
        """Finalize the calculation of the correlation function.

        Parameters:
            varg1 (float):  The variance per component of the first shear field.
            vark (float):   The variance of the scalar field.
            varg2 (float):  The variance per component of the second shear field.
        """
        self._finalize()
        mask1 = self.weightr != 0
        mask2 = self.weightr == 0
        self._var_num = 2 * varg1 * vark * varg2

        if self.bin_type in ['LogSAS', 'LogMultipole']:
            self._var_num *= 2

    @property
    def vargam0(self):
        if self._vargam0 is None:
            self._vargam0 = self._calculate_varzeta(0, self._nbins)
        return self._vargam0

    @property
    def vargam1(self):
        if self._vargam1 is None:
            self._vargam1 = self._calculate_varzeta(self._nbins, 2*self._nbins)
        return self._vargam1

    @property
    def vargam2(self):
        return self.vargam0

    @property
    def vargam3(self):
        return self.vargam1

    def _clear(self):
        super()._clear()
        self._vargam0 = None
        self._vargam1 = None

    def getStat(self):
        """The standard statistic for the current correlation object as a 1-d array.

        In this case, the concatenation of gam0.ravel() and gam1.ravel().
        """
        return np.concatenate([self.gam0.ravel(), self.gam1.ravel()])

    def getWeight(self):
        """The weight array for the current correlation object as a 1-d array.

        In this case, 2 copies of self.weight.ravel().
        """
        return np.concatenate([np.abs(self.weight.ravel())] * 2)

    def write(self, file_name, *, file_type=None, precision=None, write_patch_results=False,
              write_cov=False):
        super().write(file_name, file_type=file_type, precision=precision,
                      write_patch_results=write_patch_results, write_cov=write_cov)

    write.__doc__ = Corr3.write.__doc__.format(
        r"""
        gam0r           The real part of the estimator of :math:`\Gamma_0`
        gam0i           The imag part of the estimator of :math:`\Gamma_0`
        gam1r           The real part of the estimator of :math:`\Gamma_1`
        gam1i           The imag part of the estimator of :math:`\Gamma_1`
        sigma_gam0      The sqrt of the variance estimate of :math:`\Gamma_0`
        sigma_gam1      The sqrt of the variance estimate of :math:`\Gamma_1`
        """)

    @property
    def _write_class_col_names(self):
        return ['gam0r', 'gam0i', 'gam1r', 'gam1i', 'sigma_gam0', 'sigma_gam1']

    @property
    def _write_class_data(self):
        return [self.gam0r, self.gam0i, self.gam1r, self.gam1i,
                np.sqrt(self.vargam0), np.sqrt(self.vargam1)]

    def _read_from_data(self, data, params):
        super()._read_from_data(data, params)
        s = self.data_shape
        self._z[0] = data['gam0r'].reshape(s)
        self._z[1] = data['gam0i'].reshape(s)
        self._z[2] = data['gam1r'].reshape(s)
        self._z[3] = data['gam1i'].reshape(s)
        self._vargam0 = data['sigma_gam0'].reshape(s)**2
        self._vargam1 = data['sigma_gam1'].reshape(s)**2

class GGKCorrelation(Corr3):
    r"""This class handles the calculation and storage of a 3-point shear-shear-scalar correlation
    function.

    With this class, point 1 of the triangle (i.e. the vertex opposite d1) is the one with the
    shear value.  Use `GKGCorrelation` and `KGGCorrelation` for classes with the shear in the
    other two positions.

    For the shear projection, we follow the lead of the 3-point shear-shear-shear correlation
    functions (see `GGGCorrelation` for details), which involves projecting the shear values
    at each vertex relative to the direction to the triangle's centroid.  Furthermore, the
    GGG correlations have 4 relevant complex values for each triangle:

    .. math::

        \Gamma_0 &= \langle \gamma(\mathbf{x1}) \gamma(\mathbf{x2}) \gamma(\mathbf{x3}) \rangle \\
        \Gamma_1 &= \langle \gamma(\mathbf{x1})^* \gamma(\mathbf{x2}) \gamma(\mathbf{x3}) \rangle \\
        \Gamma_2 &= \langle \gamma(\mathbf{x1}) \gamma(\mathbf{x2})^* \gamma(\mathbf{x3}) \rangle \\
        \Gamma_3 &= \langle \gamma(\mathbf{x1}) \gamma(\mathbf{x2}) \gamma(\mathbf{x3}^*) \rangle \\

    With a scalar value at vertex 3, :math:`\Gamma_0 = \Gamma_3` and :math:`\Gamma_1 = \Gamma_2^*`.
    So there are only two independent values.  However, you may access these values using whichever
    names you find most convenient: ``gam0``, ``gam1``, ``gam2`` and ``gam3`` are all valid
    attributes, which return the corresponding value.

    See the doc string of `Corr3` for a description of how the triangles are binned along
    with the attributes related to the different binning options.

    In addition to the attributes common to all `Corr3` subclasses, objects of this class
    hold the following attributes:

    Attributes:
        gam0:       The 0th "natural" correlation function, :math:`\Gamma_0`.
        gam1:       The 1st "natural" correlation function, :math:`\Gamma_1`.
        gam2:       The 2nd "natural" correlation function, :math:`\Gamma_2`.
        gam3:       The 3rd "natural" correlation function, :math:`\Gamma_3`.
        vargam0:    The variance estimate of :math:`\Gamma_0`, only including the shot noise.
        vargam1:    The variance estimate of :math:`\Gamma_1`, only including the shot noise.
        vargam2:    The variance estimate of :math:`\Gamma_2`, only including the shot noise.
        vargam3:    The variance estimate of :math:`\Gamma_3`, only including the shot noise.

    The typical usage pattern is as follows::

        >>> ggk = treecorr.GGKCorrelation(config)
        >>> ggk.process(cat)              # For auto-correlation.
        >>> ggk.process(cat1,cat2,cat3)   # For cross-correlation.
        >>> ggk.write(file_name)          # Write out to a file.
        >>> gam0 = ggk.gam0, etc.         # To access gamma values directly.
        >>> gam0r = ggk.gam0r             # You can also access real and imag parts separately.
        >>> gam0i = ggk.gam0i

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
    _cls = 'GGKCorrelation'
    _letter1 = 'G'
    _letter2 = 'G'
    _letter3 = 'K'
    _letters = 'GGK'
    _builder = _treecorr.GGKCorr
    _calculateVar1 = staticmethod(calculateVarG)
    _calculateVar2 = staticmethod(calculateVarG)
    _calculateVar3 = staticmethod(calculateVarK)
    _sig1 = 'sig_sn (per component)'
    _sig2 = 'sig_sn (per component)'
    _sig3 = 'sig_k'

    def __init__(self, config=None, *, logger=None, **kwargs):
        super().__init__(config, logger=logger, **kwargs)

        shape = self.data_shape
        self._z[0:4] = [np.zeros(shape, dtype=float) for _ in range(4)]
        self._vargam0 = None
        self._vargam1 = None
        self.logger.debug('Finished building GGKCorr')

    @property
    def gam0(self):
        return self._z[0] + 1j * self._z[1]

    @property
    def gam1(self):
        return self._z[2] + 1j * self._z[3]

    @property
    def gam2(self):
        return self._z[2] - 1j * self._z[3]

    @property
    def gam3(self):
        return self._z[0] + 1j * self._z[1]

    @property
    def gam0r(self):
        return self._z[0]

    @property
    def gam0i(self):
        return self._z[1]

    @property
    def gam1r(self):
        return self._z[2]

    @property
    def gam1i(self):
        return self._z[3]

    @property
    def gam2r(self):
        return self._z[2]

    @property
    def gam2i(self):
        return -self._z[3]

    @property
    def gam3r(self):
        return self._z[0]

    @property
    def gam3i(self):
        return self._z[1]

    def finalize(self, varg1, varg2, vark):
        """Finalize the calculation of the correlation function.

        Parameters:
            varg1 (float):  The variance per component of the first shear field.
            varg2 (float):  The variance per component of the second shear field.
            vark (float):   The variance of the scalar field.
        """
        self._finalize()
        mask1 = self.weightr != 0
        mask2 = self.weightr == 0
        self._var_num = 2 * varg1 * varg2 * vark

        if self.bin_type in ['LogSAS', 'LogMultipole']:
            self._var_num *= 2

    @property
    def vargam0(self):
        if self._vargam0 is None:
            self._vargam0 = self._calculate_varzeta(0, self._nbins)
        return self._vargam0

    @property
    def vargam1(self):
        if self._vargam1 is None:
            self._vargam1 = self._calculate_varzeta(self._nbins, 2*self._nbins)
        return self._vargam1

    @property
    def vargam2(self):
        return self.vargam1

    @property
    def vargam3(self):
        return self.vargam0

    def _clear(self):
        super()._clear()
        self._vargam0 = None
        self._vargam1 = None

    def getStat(self):
        """The standard statistic for the current correlation object as a 1-d array.

        In this case, the concatenation of gam0.ravel() and gam1.ravel().
        """
        return np.concatenate([self.gam0.ravel(), self.gam1.ravel()])

    def getWeight(self):
        """The weight array for the current correlation object as a 1-d array.

        In this case, 2 copies of self.weight.ravel().
        """
        return np.concatenate([np.abs(self.weight.ravel())] * 2)

    def write(self, file_name, *, file_type=None, precision=None, write_patch_results=False,
              write_cov=False):
        super().write(file_name, file_type=file_type, precision=precision,
                      write_patch_results=write_patch_results, write_cov=write_cov)

    write.__doc__ = Corr3.write.__doc__.format(
        r"""
        gam0r           The real part of the estimator of :math:`\Gamma_0`
        gam0i           The imag part of the estimator of :math:`\Gamma_0`
        gam1r           The real part of the estimator of :math:`\Gamma_1`
        gam1i           The imag part of the estimator of :math:`\Gamma_1`
        sigma_gam0      The sqrt of the variance estimate of :math:`\Gamma_0`
        sigma_gam1      The sqrt of the variance estimate of :math:`\Gamma_1`
        """)

    @property
    def _write_class_col_names(self):
        return ['gam0r', 'gam0i', 'gam1r', 'gam1i', 'sigma_gam0', 'sigma_gam1']

    @property
    def _write_class_data(self):
        return [self.gam0r, self.gam0i, self.gam1r, self.gam1i,
                np.sqrt(self.vargam0), np.sqrt(self.vargam1)]

    def _read_from_data(self, data, params):
        super()._read_from_data(data, params)
        s = self.data_shape
        self._z[0] = data['gam0r'].reshape(s)
        self._z[1] = data['gam0i'].reshape(s)
        self._z[2] = data['gam1r'].reshape(s)
        self._z[3] = data['gam1i'].reshape(s)
        self._vargam0 = data['sigma_gam0'].reshape(s)**2
        self._vargam1 = data['sigma_gam1'].reshape(s)**2
