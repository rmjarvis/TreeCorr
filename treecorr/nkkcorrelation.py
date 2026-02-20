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
.. module:: nkkcorrelation
"""

import numpy as np

from . import _treecorr
from .catalog import calculateVarK
from .corr3base import Corr3


class NKKCorrelation(Corr3):
    r"""This class handles the calculation and storage of a 3-point count-scalar-scalar correlation
    function.

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
        >>> rkk.process(rand, cat2)        # Compute cross-correlation with randoms.
        >>> nkk.calculateZeta(rkk=rkk)     # Calculate zeta using randoms
        >>> zeta = nkk.zeta                # Access correlation function
        >>> zetar = nkk.zetar              # Or access real and imaginary parts separately
        >>> zetai = nkk.zetai

    Parameters:
        config (dict):  A configuration dict that can be used to pass in kwargs if desired.
                        This dict is allowed to have additional entries besides those listed
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
        self._comp_varzeta = None
        self.logger.debug('Finished building NKKCorr')

    @property
    def raw_zeta(self):
        return self._z[0]

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
    def raw_varzeta(self):
        if self._varzeta is None:
            self._calculate_varzeta(1)
        return self._varzeta[0]

    @property
    def varzeta(self):
        if self._comp_varzeta is None:
            return self.raw_varzeta
        else:
            return self._comp_varzeta

    def _clear(self):
        super()._clear()
        self._rkk = None
        self._zeta = None
        self._comp_varzeta = None

    def calculateZeta(self, *, rkk=None):
        r"""Calculate the correlation function possibly given another correlation function
        that uses random points for the foreground objects.

        - If rkk is None, the simple correlation function (self.zeta) is returned.
        - If rkk is not None, then a compensated calculation is done:
          :math:`\zeta = (DKK - RKK)`, where DKK represents the correlation of the kappa
          field with the data points and RKK represents the correlation with random points.

        After calling this function, the attributes ``zeta``, ``varzeta`` and ``cov`` will
        correspond to the compensated values (if rkk is provided).  The raw, uncompensated values
        are available as ``raw_zeta`` and ``raw_varzeta``.

        Parameters:
            rkk (NKKCorrelation): The cross-correlation using random locations as the lenses (RKK),
                                  if desired.  (default: None)

        Returns:
            Tuple containing
                - zeta = array of :math:`\zeta`
                - varzeta = array of variance estimates of :math:`\zeta`
        """
        if rkk is not None:
            if self.bin_type == 'LogMultipole':
                raise TypeError("calculateZeta is not valid for LogMultipole binning")

            self._zeta = self.raw_zeta - rkk.zeta
            self._rkk = rkk

            if (rkk.npatch1 not in (1,self.npatch1) or rkk.npatch2 != self.npatch2
                    or rkk.npatch3 != self.npatch3):
                raise RuntimeError("RKK must be run with the same patches as DKK")

            if len(self.results) > 0:
                # If there are any rkk patch pairs that aren't in results (e.g. due to different
                # edge effects among the various pairs in consideration), then we need to add
                # some dummy results to make sure all the right pairs are computed when we make
                # the vectors for the covariance matrix.
                template = next(iter(self.results.values()))  # Just need something to copy.
                for ijk in rkk.results:
                    if ijk in self.results: continue
                    new_cij = template.copy()
                    new_cij._z[0][:] = 0
                    new_cij.weight[:] = 0
                    self.results[ijk] = new_cij
                    self.__dict__.pop('_ok',None)

                self._cov = self.estimate_cov(self.var_method)
                self._comp_varzeta = np.zeros(self.data_shape, dtype=float)
                self._comp_varzeta.ravel()[:] = self.cov_diag
            else:
                self._comp_varzeta = self.raw_varzeta + rkk.varzeta
        else:
            self._zeta = self.raw_zeta
            self._comp_varzeta = None

        return self._zeta, self.varzeta

    def _calculate_xi_from_pairs(self, pairs, corr_only):
        super()._calculate_xi_from_pairs(pairs, corr_only)
        if self._rkk is not None:
            # If rkk has npatch1 = 1, adjust pairs appropriately
            if self._rkk.npatch1 == 1 and not all([p[0] == 0 for p in pairs]):
                pairs = [(0,j,k,w) for i,j,k,w in pairs if i == j]
            pairs = self._rkk._keep_ok(pairs)
            self._rkk._calculate_xi_from_pairs(pairs, corr_only=True)
            self._zeta = self.raw_zeta - self._rkk.zeta

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
        self._comp_varzeta = data['sigma_zeta'].reshape(s)**2
        self._varzeta = [self._comp_varzeta]

class KNKCorrelation(Corr3):
    r"""This class handles the calculation and storage of a 3-point scalar-count-scalar correlation
    function.

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

        >>> knk = treecorr.KNKCorrelation(config)
        >>> knk.process(cat1, cat2, cat1)  # Compute cross-correlation of two fields.
        >>> knk.process(cat1, cat2, cat3)  # Compute cross-correlation of three fields.
        >>> knk.write(file_name)           # Write out to a file.
        >>> krk.process(cat1, rand, cat1)  # Compute cross-correlation with randoms.
        >>> knk.calculateZeta(krk=krk)     # Calculate zeta using randoms
        >>> zeta = knk.zeta                # Access correlation function
        >>> zetar = knk.zetar              # Or access real and imaginary parts separately
        >>> zetai = knk.zetai

    Parameters:
        config (dict):  A configuration dict that can be used to pass in kwargs if desired.
                        This dict is allowed to have additional entries besides those listed
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
        self._comp_varzeta = None
        self.logger.debug('Finished building KNKCorr')

    @property
    def raw_zeta(self):
        return self._z[0]

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
    def raw_varzeta(self):
        if self._varzeta is None:
            self._calculate_varzeta(1)
        return self._varzeta[0]

    @property
    def varzeta(self):
        if self._comp_varzeta is None:
            return self.raw_varzeta
        else:
            return self._comp_varzeta

    def _clear(self):
        super()._clear()
        self._krk = None
        self._zeta = None
        self._comp_varzeta = None

    def calculateZeta(self, *, krk=None):
        r"""Calculate the correlation function possibly given another correlation function
        that uses random points for the foreground objects.

        - If krk is None, the simple correlation function (self.zeta) is returned.
        - If krk is not None, then a compensated calculation is done:
          :math:`\zeta = (KDK - KRK)`, where KDK represents the correlation of the kappa
          field with the data points and KRK represents the correlation with random points.

        After calling this function, the attributes ``zeta``, ``varzeta`` and ``cov`` will
        correspond to the compensated values (if krk is provided).  The raw, uncompensated values
        are available as ``raw_zeta`` and ``raw_varzeta``.

        Parameters:
            krk (KNKCorrelation): The cross-correlation using random locations as the lenses (KRK),
                                  if desired.  (default: None)

        Returns:
            Tuple containing
                - zeta = array of :math:`\zeta`
                - varzeta = array of variance estimates of :math:`\zeta`
        """
        if krk is not None:
            if self.bin_type == 'LogMultipole':
                raise TypeError("calculateZeta is not valid for LogMultipole binning")

            self._zeta = self.raw_zeta - krk.zeta
            self._krk = krk

            if (krk.npatch2 not in (1,self.npatch2) or krk.npatch1 != self.npatch1
                    or krk.npatch3 != self.npatch3):
                raise RuntimeError("KRK must be run with the same patches as KDK")

            if len(self.results) > 0:
                # If there are any krk patch pairs that aren't in results (e.g. due to different
                # edge effects among the various pairs in consideration), then we need to add
                # some dummy results to make sure all the right pairs are computed when we make
                # the vectors for the covariance matrix.
                template = next(iter(self.results.values()))  # Just need something to copy.
                for ijk in krk.results:
                    if ijk in self.results: continue
                    new_cij = template.copy()
                    new_cij._z[0][:] = 0
                    new_cij.weight[:] = 0
                    self.results[ijk] = new_cij
                    self.__dict__.pop('_ok',None)

                self._cov = self.estimate_cov(self.var_method)
                self._comp_varzeta = np.zeros(self.data_shape, dtype=float)
                self._comp_varzeta.ravel()[:] = self.cov_diag
            else:
                self._comp_varzeta = self.raw_varzeta + krk.varzeta
        else:
            self._zeta = self.raw_zeta
            self._comp_varzeta = None

        return self._zeta, self.varzeta

    def _calculate_xi_from_pairs(self, pairs, corr_only):
        super()._calculate_xi_from_pairs(pairs, corr_only)
        if self._krk is not None:
            # If krk has npatch2 = 1, adjust pairs appropriately
            if self._krk.npatch2 == 1 and not all([p[1] == 0 for p in pairs]):
                pairs = [(i,0,k,w) for i,j,k,w in pairs if i == j]
            pairs = self._krk._keep_ok(pairs)
            self._krk._calculate_xi_from_pairs(pairs, corr_only=True)
            self._zeta = self.raw_zeta - self._krk.zeta

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
        self._comp_varzeta = data['sigma_zeta'].reshape(s)**2
        self._varzeta = [self._comp_varzeta]

class KKNCorrelation(Corr3):
    r"""This class handles the calculation and storage of a 3-point scalar-scalar-count correlation
    function.

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
        >>> kkr.process(cat1, rand)        # Compute cross-correlation with randoms.
        >>> kkn.calculateZeta(kkr=kkr)     # Calculate zeta using randoms
        >>> zeta = kkn.zeta                # Access correlation function
        >>> zetar = kkn.zetar              # Or access real and imaginary parts separately
        >>> zetai = kkn.zetai

    Parameters:
        config (dict):  A configuration dict that can be used to pass in kwargs if desired.
                        This dict is allowed to have additional entries besides those listed
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
        self._comp_varzeta = None
        self.logger.debug('Finished building KKNCorr')

    @property
    def raw_zeta(self):
        return self._z[0]

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
    def raw_varzeta(self):
        if self._varzeta is None:
            self._calculate_varzeta(1)
        return self._varzeta[0]

    @property
    def varzeta(self):
        if self._comp_varzeta is None:
            return self.raw_varzeta
        else:
            return self._comp_varzeta

    def _clear(self):
        super()._clear()
        self._kkr = None
        self._zeta = None
        self._comp_varzeta = None

    def calculateZeta(self, *, kkr=None):
        r"""Calculate the correlation function possibly given another correlation function
        that uses random points for the foreground objects.

        - If kkr is None, the simple correlation function (self.zeta) is returned.
        - If kkr is not None, then a compensated calculation is done:
          :math:`\zeta = (KKD - KKR)`, where KKD represents the correlation of the kappa
          field with the data points and KKR represents the correlation with random points.

        After calling this function, the attributes ``zeta``, ``varzeta`` and ``cov`` will
        correspond to the compensated values (if kkr is provided).  The raw, uncompensated values
        are available as ``raw_zeta`` and ``raw_varzeta``.

        Parameters:
            kkr (KKNCorrelation): The cross-correlation using random locations as the lenses (KKR),
                                  if desired.  (default: None)

        Returns:
            Tuple containing
                - zeta = array of :math:`\zeta`
                - varzeta = array of variance estimates of :math:`\zeta`
        """
        if kkr is not None:
            if self.bin_type == 'LogMultipole':
                raise TypeError("calculateZeta is not valid for LogMultipole binning")

            self._zeta = self.raw_zeta - kkr.zeta
            self._kkr = kkr

            if (kkr.npatch3 not in (1,self.npatch3) or kkr.npatch1 != self.npatch1
                    or kkr.npatch2 != self.npatch2):
                raise RuntimeError("KKR must be run with the same patches as KKD")

            if len(self.results) > 0:
                # If there are any kkr patch pairs that aren't in results (e.g. due to different
                # edge effects among the various pairs in consideration), then we need to add
                # some dummy results to make sure all the right pairs are computed when we make
                # the vectors for the covariance matrix.
                template = next(iter(self.results.values()))  # Just need something to copy.
                for ijk in kkr.results:
                    if ijk in self.results: continue
                    new_cij = template.copy()
                    new_cij._z[0][:] = 0
                    new_cij.weight[:] = 0
                    self.results[ijk] = new_cij
                    self.__dict__.pop('_ok',None)

                self._cov = self.estimate_cov(self.var_method)
                self._comp_varzeta = np.zeros(self.data_shape, dtype=float)
                self._comp_varzeta.ravel()[:] = self.cov_diag
            else:
                self._comp_varzeta = self.raw_varzeta + kkr.varzeta
        else:
            self._zeta = self.raw_zeta
            self._comp_varzeta = None

        return self._zeta, self.varzeta

    def _calculate_xi_from_pairs(self, pairs, corr_only):
        super()._calculate_xi_from_pairs(pairs, corr_only)
        if self._kkr is not None:
            # If kkr has npatch3 = 1, adjust pairs appropriately
            if self._kkr.npatch3 == 1 and not all([p[2] == 0 for p in pairs]):
                pairs = [(i,j,0,w) for i,j,k,w in pairs if i == k]
            pairs = self._kkr._keep_ok(pairs)
            self._kkr._calculate_xi_from_pairs(pairs, corr_only=True)
            self._zeta = self.raw_zeta - self._kkr.zeta

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
        self._comp_varzeta = data['sigma_zeta'].reshape(s)**2
        self._varzeta = [self._comp_varzeta]
