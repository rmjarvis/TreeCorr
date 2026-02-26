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


class KNNCorrelation(Corr3):
    r"""This class handles the calculation and storage of a 3-point scalar-count-count correlation
    function.

    With this class, point 1 of the triangle (i.e. the vertex opposite d1) is the one with the
    scalar value.  Use `NKNCorrelation` and `NNKCorrelation` for classes with the scalar in the
    other two positions.

    See the docstring of `Corr3` for a description of how the triangles are binned along
    with the attributes related to the different binning options.

    In addition to the attributes common to all `Corr3` subclasses, objects of this class
    hold the following attributes:

    Attributes:
        zeta:       The correlation function, :math:`\zeta`.
        varzeta:    The variance estimate of :math:`\zeta`, computed according to ``var_method``
                    (default: ``'shot'``).

    The typical usage pattern is as follows:

        >>> knn = treecorr.KNNCorrelation(config)
        >>> knn.process(cat1, cat2)          # Compute the cross-correlation of two fields.
        >>> # knn.process(cat1, cat2, cat3)  # ... or of three fields.
        >>> krr.process(cat1, rand)          # Compute the random cross-correlation.
        >>> kdr.process(cat1, cat2, rand)    # Optionally compute data-random cross-correlation.
        >>> knn.write(file_name)             # Write out to a file.
        >>> knn.calculateZeta(krr=krr, kdr=kdr)  # Calculate zeta using randoms.
        >>> zeta = knn.zeta                  # Access the correlation function.
        >>> zetar = knn.zetar                # Or access real and imaginary parts separately.
        >>> zetai = knn.zetai

    See also: `NKNCorrelation`, `NNKCorrelation`, `NKKCorrelation`, `NNNCorrelation`,
    `NKCorrelation`.

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
        self._comp_varzeta = None
        self.logger.debug('Finished building KNNCorr')

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
        if self._krr is not None and self._krr is not True:
            ret._krr = self._krr.copy()
        if self._kdr is not None and self._kdr is not True:
            ret._kdr = self._kdr.copy()
        if self._krd is not None and self._krd is not True:
            ret._krd = self._krd.copy()
        return ret

    def _zero_copy(self):
        ret = super()._zero_copy()
        ret._krr = None
        ret._kdr = None
        ret._krd = None
        ret._zeta = None
        ret._comp_varzeta = None
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
        self._krr = None
        self._krd = None
        self._kdr = None
        self._zeta = None
        self._comp_varzeta = None

    def calculateZeta(self, *, krr=None, kdr=None, krd=None):
        r"""Calculate the correlation function given another correlation function of random
        points using the same mask, and possibly cross-correlations of the data and random.

        The krr value is the `KNNCorrelation` function for random points with the scalar field.
        One can also provide a cross-correlation of the count data with randoms and the scalar.

        - If krr is None, the simple correlation function (self.zeta) is returned.
        - If only krr is given the compensated value :math:`\zeta = KDD - KRR` is returned.
        - If kdr is given and krd is None (or vice versa), then :math:`\zeta = KDD - 2KDR + KRR`
          is returned.
        - If kdr and krd are both given, then :math:`\zeta = KDD - KDR - KRD + KRR` is returned.

        Here KDD is the data KNN correlation function, which is the current object.

        After calling this method, you can use this correlation object in the
        `estimate_multi_cov` function.  Also, the calculated zeta and varzeta returned from this
        function will be available as attributes.

        .. note::

            The returned variance estimate (``varzeta``) is computed according to this object's
            ``var_method`` setting, specified when constructing the object (default: ``'shot'``).
            Internally, this method calls `Corr3.estimate_cov`; see that method for details
            about available variance and covariance estimation schemes.

        Parameters:
            krr (KNNCorrelation):   The correlation of the random points with the scalar field
                                    (KRR) (default: None)
            kdr (KNNCorrelation):   The cross-correlation of the data with both randoms and the
                                    scalar field (KDR), if desired. (default: None)
            krd (KNNCorrelation):   The cross-correlation of the randoms with both the data and the
                                    scalar field (KRD), if desired. (default: None)

        Returns:
            Tuple containing:

            - zeta = array of :math:`\zeta(r)`
            - varzeta = an estimate of the variance of :math:`\zeta(r)`
        """
        # Calculate zeta based on which randoms are provided.
        if krr is not None:
            if self.bin_type == 'LogMultipole':
                raise TypeError("calculateZeta is not valid for LogMultipole binning")

            if kdr is None and krd is None:
                self._zeta = self.raw_zeta - krr.zeta
            elif krd is not None and kdr is None:
                self._zeta = self.raw_zeta - 2.*krd.zeta + krr.zeta
            elif kdr is not None and krd is None:
                self._zeta = self.raw_zeta - 2.*kdr.zeta + krr.zeta
            else:
                self._zeta = self.raw_zeta - kdr.zeta - krd.zeta + krr.zeta

            self._krr = krr
            self._kdr = kdr
            self._krd = krd

            if (krr.npatch2 not in (1,self.npatch2) or krr.npatch3 not in (1,self.npatch3)
                    or krr.npatch1 != self.npatch1):
                raise RuntimeError("KRR must be run with the same patches as KDD")
            if krd is not None and (krd.npatch2 not in (1,self.npatch2)
                                    or krd.npatch1 != self.npatch1
                                    or krd.npatch3 != self.npatch3):
                raise RuntimeError("KRD must be run with the same patches as KDD")
            if kdr is not None and (kdr.npatch3 not in (1,self.npatch3)
                                    or kdr.npatch1 != self.npatch1
                                    or kdr.npatch2 != self.npatch2):
                raise RuntimeError("KDR must be run with the same patches as KDD")

            if len(self.results) > 0:
                added_any = False
                for results in (krr.results,
                                krd.results if krd is not None else None,
                                kdr.results if kdr is not None else None):
                    if results is None:
                        continue
                    for ijk in results:
                        if ijk in self.results: continue
                        self.results[ijk] = self._zero_copy()
                        added_any = True
                if added_any:
                    self.__dict__.pop('_ok',None)

                self._cov = self.estimate_cov(self.var_method)
                self._comp_varzeta = np.zeros(self.data_shape, dtype=float)
                self._comp_varzeta.ravel()[:] = self.cov_diag
            else:
                self._comp_varzeta = self.raw_varzeta + krr.varzeta
                if krd is not None:
                    self._comp_varzeta += krd.varzeta
                if kdr is not None:
                    self._comp_varzeta += kdr.varzeta
        else:
            if krd is not None:
                raise TypeError("krd is invalid if krr is None")
            if kdr is not None:
                raise TypeError("kdr is invalid if krr is None")
            self._zeta = self.raw_zeta
            self._comp_varzeta = None

        return self._zeta, self.varzeta

    def _calculate_xi_from_pairs(self, pairs, corr_only):
        super()._calculate_xi_from_pairs(pairs, corr_only)
        if self._krr is not None:
            # If r doesn't have patches, then convert all (i,i,i) pairs to (i,0,0).
            if self._krr.npatch2 == 1 and not all(p[1] == 0 for p in pairs):
                pairs1 = [(i,0,0,w) for i,j,k,w in pairs if i == j == k]
            else:
                pairs1 = pairs
            pairs1 = self._krr._keep_ok(pairs1)
            self._krr._calculate_xi_from_pairs(pairs1, corr_only=True)

        if self._kdr is not None:
            pairs2 = pairs
            if self._kdr.npatch3 == 1 and not all(p[2] == 0 for p in pairs):
                pairs2 = [(i,j,0,w) for i,j,k,w in pairs if j == k]
            pairs2 = self._kdr._keep_ok(pairs2)
            self._kdr._calculate_xi_from_pairs(pairs2, corr_only=True)

        if self._krd is not None:
            pairs3 = pairs
            if self._krd.npatch2 == 1 and not all(p[1] == 0 for p in pairs):
                pairs3 = [(i,0,k,w) for i,j,k,w in pairs if i == k]
            pairs3 = self._krd._keep_ok(pairs3)
            self._krd._calculate_xi_from_pairs(pairs3, corr_only=True)

        if self._krr is None:
            self._zeta = None
        elif self._kdr is None and self._krd is None:
            self._zeta = self.raw_zeta - self._krr.zeta
        elif self._krd is not None and self._kdr is None:
            self._zeta = self.raw_zeta - 2.*self._krd.zeta + self._krr.zeta
        elif self._kdr is not None and self._krd is None:
            self._zeta = self.raw_zeta - 2.*self._kdr.zeta + self._krr.zeta
        else:
            self._zeta = self.raw_zeta - self._kdr.zeta - self._krd.zeta + self._krr.zeta

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

class NKNCorrelation(Corr3):
    r"""This class handles the calculation and storage of a 3-point count-scalar-count correlation
    function.

    With this class, point 2 of the triangle (i.e. the vertex opposite d2) is the one with the
    scalar value.  Use `KNNCorrelation` and `NNKCorrelation` for classes with the scalar in the
    other two positions.

    See the docstring of `Corr3` for a description of how the triangles are binned along
    with the attributes related to the different binning options.

    In addition to the attributes common to all `Corr3` subclasses, objects of this class
    hold the following attributes:

    Attributes:
        zeta:       The correlation function, :math:`\zeta`.
        varzeta:    The variance estimate of :math:`\zeta`, computed according to ``var_method``
                    (default: ``'shot'``).

    The typical usage pattern is as follows:

        >>> nkn = treecorr.NKNCorrelation(config)
        >>> nkn.process(cat1, cat2, cat1)    # Compute the cross-correlation of two fields.
        >>> # nkn.process(cat1, cat2, cat3)  # ... or of three fields.
        >>> rkr.process(rand, cat2, rand)    # Compute the random cross-correlation.
        >>> dkr.process(cat1, cat2, rand)    # Optionally compute data-random cross-correlation.
        >>> nkn.write(file_name)             # Write out to a file.
        >>> nkn.calculateZeta(rkr=rkr, dkr=dkr)  # Calculate zeta using randoms.
        >>> zeta = nkn.zeta                  # Access the correlation function.
        >>> zetar = nkn.zetar                # Or access real and imaginary parts separately.
        >>> zetai = nkn.zetai

    See also: `KNNCorrelation`, `NNKCorrelation`, `NKKCorrelation`, `NNNCorrelation`,
    `NKCorrelation`.

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
        self._comp_varzeta = None
        self.logger.debug('Finished building NKNCorr')

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
        if self._rkr is not None and self._rkr is not True:
            ret._rkr = self._rkr.copy()
        if self._dkr is not None and self._dkr is not True:
            ret._dkr = self._dkr.copy()
        if self._rkd is not None and self._rkd is not True:
            ret._rkd = self._rkd.copy()
        return ret

    def _zero_copy(self):
        ret = super()._zero_copy()
        ret._rkr = None
        ret._dkr = None
        ret._rkd = None
        ret._zeta = None
        ret._comp_varzeta = None
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
        self._rkr = None
        self._dkr = None
        self._rkd = None
        self._zeta = None
        self._comp_varzeta = None

    def calculateZeta(self, *, rkr=None, dkr=None, rkd=None):
        r"""Calculate the correlation function given another correlation function of random
        points using the same mask, and possibly cross-correlations of the data and random.

        The rkr value is the `NKNCorrelation` function for random points with the scalar field.
        One can also provide a cross-correlation of the count data with randoms and the scalar.

        - If rkr is None, the simple correlation function (self.zeta) is returned.
        - If only rkr is given the compensated value :math:`\zeta = DKD - RKR` is returned.
        - If dkr is given and rkd is None (or vice versa), then :math:`\zeta = DKD - 2DKR + RKR`
          is returned.
        - If dkr and rkd are both given, then :math:`\zeta = DKD - DKR - RKD + RKR` is returned.

        Here DKD is the data NKN correlation function, which is the current object.

        After calling this method, you can use this correlation object in the
        `estimate_multi_cov` function.  Also, the calculated zeta and varzeta returned from this
        function will be available as attributes.

        .. note::

            The returned variance estimate (``varzeta``) is computed according to this object's
            ``var_method`` setting, specified when constructing the object (default: ``'shot'``).
            Internally, this method calls `Corr3.estimate_cov`; see that method for details
            about available variance and covariance estimation schemes.

        Parameters:
            rkr (NKNCorrelation):   The correlation of the random points with the scalar field
                                    (RKR) (default: None)
            dkr (NKNCorrelation):   The cross-correlation of the data with both randoms and the
                                    scalar field (DKR), if desired. (default: None)
            rkd (NKNCorrelation):   The cross-correlation of the randoms with both the data and the
                                    scalar field (RKD), if desired. (default: None)

        Returns:
            Tuple containing:

            - zeta = array of :math:`\zeta(r)`
            - varzeta = an estimate of the variance of :math:`\zeta(r)`
        """
        # Calculate zeta based on which randoms are provided.
        if rkr is not None:
            if self.bin_type == 'LogMultipole':
                raise TypeError("calculateZeta is not valid for LogMultipole binning")

            if dkr is None and rkd is None:
                self._zeta = self.raw_zeta - rkr.zeta
            elif rkd is not None and dkr is None:
                self._zeta = self.raw_zeta - 2.*rkd.zeta + rkr.zeta
            elif dkr is not None and rkd is None:
                self._zeta = self.raw_zeta - 2.*dkr.zeta + rkr.zeta
            else:
                self._zeta = self.raw_zeta - dkr.zeta - rkd.zeta + rkr.zeta

            self._rkr = rkr
            self._dkr = dkr
            self._rkd = rkd

            if (rkr.npatch1 not in (1,self.npatch1) or rkr.npatch3 not in (1,self.npatch3)
                    or rkr.npatch2 != self.npatch2):
                raise RuntimeError("RKR must be run with the same patches as DKD")
            if rkd is not None and (rkd.npatch1 not in (1,self.npatch1)
                                    or rkd.npatch2 != self.npatch2
                                    or rkd.npatch3 != self.npatch3):
                raise RuntimeError("RKD must be run with the same patches as DKD")
            if dkr is not None and (dkr.npatch3 not in (1,self.npatch3)
                                    or dkr.npatch1 != self.npatch1
                                    or dkr.npatch2 != self.npatch2):
                raise RuntimeError("DKR must be run with the same patches as DKD")

            if len(self.results) > 0:
                added_any = False
                for results in (rkr.results,
                                rkd.results if rkd is not None else None,
                                dkr.results if dkr is not None else None):
                    if results is None:
                        continue
                    for ijk in results:
                        if ijk in self.results: continue
                        self.results[ijk] = self._zero_copy()
                        added_any = True
                if added_any:
                    self.__dict__.pop('_ok',None)

                self._cov = self.estimate_cov(self.var_method)
                self._comp_varzeta = np.zeros(self.data_shape, dtype=float)
                self._comp_varzeta.ravel()[:] = self.cov_diag
            else:
                self._comp_varzeta = self.raw_varzeta + rkr.varzeta
                if rkd is not None:
                    self._comp_varzeta += rkd.varzeta
                if dkr is not None:
                    self._comp_varzeta += dkr.varzeta
        else:
            if rkd is not None:
                raise TypeError("rkd is invalid if rkr is None")
            if dkr is not None:
                raise TypeError("dkr is invalid if rkr is None")
            self._zeta = self.raw_zeta
            self._comp_varzeta = None

        return self._zeta, self.varzeta

    def _calculate_xi_from_pairs(self, pairs, corr_only):
        super()._calculate_xi_from_pairs(pairs, corr_only)
        if self._rkr is not None:
            # If r doesn't have patches, then convert all (i,i,i) pairs to (0,i,0).
            if self._rkr.npatch1 == 1 and not all(p[0] == 0 for p in pairs):
                pairs1 = [(0,j,0,w) for i,j,k,w in pairs if i == j == k]
            else:
                pairs1 = pairs
            pairs1 = self._rkr._keep_ok(pairs1)
            self._rkr._calculate_xi_from_pairs(pairs1, corr_only=True)

        if self._dkr is not None:
            pairs2 = pairs
            if self._dkr.npatch3 == 1 and not all(p[2] == 0 for p in pairs):
                pairs2 = [(i,j,0,w) for i,j,k,w in pairs if j == k]
            pairs2 = self._dkr._keep_ok(pairs2)
            self._dkr._calculate_xi_from_pairs(pairs2, corr_only=True)

        if self._rkd is not None:
            pairs3 = pairs
            if self._rkd.npatch1 == 1 and not all(p[0] == 0 for p in pairs):
                pairs3 = [(0,j,k,w) for i,j,k,w in pairs if i == k]
            pairs3 = self._rkd._keep_ok(pairs3)
            self._rkd._calculate_xi_from_pairs(pairs3, corr_only=True)

        if self._rkr is None:
            self._zeta = None
        elif self._dkr is None and self._rkd is None:
            self._zeta = self.raw_zeta - self._rkr.zeta
        elif self._rkd is not None and self._dkr is None:
            self._zeta = self.raw_zeta - 2.*self._rkd.zeta + self._rkr.zeta
        elif self._dkr is not None and self._rkd is None:
            self._zeta = self.raw_zeta - 2.*self._dkr.zeta + self._rkr.zeta
        else:
            self._zeta = self.raw_zeta - self._dkr.zeta - self._rkd.zeta + self._rkr.zeta

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

class NNKCorrelation(Corr3):
    r"""This class handles the calculation and storage of a 3-point count-count-scalar correlation
    function.

    With this class, point 3 of the triangle (i.e. the vertex opposite d3) is the one with the
    scalar value.  Use `KNNCorrelation` and `NKNCorrelation` for classes with the scalar in the
    other two positions.

    See the docstring of `Corr3` for a description of how the triangles are binned along
    with the attributes related to the different binning options.

    In addition to the attributes common to all `Corr3` subclasses, objects of this class
    hold the following attributes:

    Attributes:
        zeta:       The correlation function, :math:`\zeta`.
        varzeta:    The variance estimate of :math:`\zeta`, computed according to ``var_method``
                    (default: ``'shot'``).

    The typical usage pattern is as follows:

        >>> nnk = treecorr.NNKCorrelation(config)
        >>> nnk.process(cat1, cat2)          # Compute the cross-correlation of two fields.
        >>> # nnk.process(cat1, cat2, cat3)  # ... or of three fields.
        >>> rrk.process(rand, cat2)          # Compute the random cross-correlation.
        >>> drk.process(cat1, rand, cat2)    # Optionally compute data-random cross-correlation.
        >>> nnk.write(file_name)             # Write out to a file.
        >>> nnk.calculateZeta(rrk=rrk, drk=drk)  # Calculate zeta using randoms.
        >>> zeta = nnk.zeta                  # Access the correlation function.
        >>> zetar = nnk.zetar                # Or access real and imaginary parts separately.
        >>> zetai = nnk.zetai

    See also: `KNNCorrelation`, `NKNCorrelation`, `NKKCorrelation`, `NNNCorrelation`,
    `NKCorrelation`.

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
        self._comp_varzeta = None
        self.logger.debug('Finished building NNKCorr')

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
        if self._rrk is not None and self._rrk is not True:
            ret._rrk = self._rrk.copy()
        if self._drk is not None and self._drk is not True:
            ret._drk = self._drk.copy()
        if self._rdk is not None and self._rdk is not True:
            ret._rdk = self._rdk.copy()
        return ret

    def _zero_copy(self):
        ret = super()._zero_copy()
        ret._rrk = None
        ret._drk = None
        ret._rdk = None
        ret._zeta = None
        ret._comp_varzeta = None
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
        self._rrk = None
        self._rdk = None
        self._drk = None
        self._zeta = None
        self._comp_varzeta = None

    def calculateZeta(self, *, rrk=None, drk=None, rdk=None):
        r"""Calculate the correlation function given another correlation function of random
        points using the same mask, and possibly cross-correlations of the data and random.

        The rrk value is the `NNKCorrelation` function for random points with the scalar field.
        One can also provide a cross-correlation of the count data with randoms and the scalar.

        - If rrk is None, the simple correlation function (self.zeta) is returned.
        - If only rrk is given the compensated value :math:`\zeta = DDK - RRK` is returned.
        - If drk is given and rdk is None (or vice versa), then :math:`\zeta = DDK - 2DRK + RRK`
          is returned.
        - If drk and rdk are both given, then :math:`\zeta = DDK - DRK - RDK + RRK` is returned.

        Here DDK is the data NNK correlation function, which is the current object.

        After calling this method, you can use this correlation object in the
        `estimate_multi_cov` function.  Also, the calculated zeta and varzeta returned from this
        function will be available as attributes.

        .. note::

            The returned variance estimate (``varzeta``) is computed according to this object's
            ``var_method`` setting, specified when constructing the object (default: ``'shot'``).
            Internally, this method calls `Corr3.estimate_cov`; see that method for details
            about available variance and covariance estimation schemes.

        Parameters:
            rrk (NNKCorrelation):   The correlation of the random points with the scalar field
                                    (RRK) (default: None)
            drk (NNKCorrelation):   The cross-correlation of the data with both randoms and the
                                    scalar field (DRK), if desired. (default: None)
            rdk (NNKCorrelation):   The cross-correlation of the randoms with both the data and the
                                    scalar field (RDK), if desired. (default: None)

        Returns:
            Tuple containing:

            - zeta = array of :math:`\zeta(r)`
            - varzeta = an estimate of the variance of :math:`\zeta(r)`
        """
        # Calculate zeta based on which randoms are provided.
        if rrk is not None:
            if self.bin_type == 'LogMultipole':
                raise TypeError("calculateZeta is not valid for LogMultipole binning")

            if drk is None and rdk is None:
                self._zeta = self.raw_zeta - rrk.zeta
            elif rdk is not None and drk is None:
                self._zeta = self.raw_zeta - 2.*rdk.zeta + rrk.zeta
            elif drk is not None and rdk is None:
                self._zeta = self.raw_zeta - 2.*drk.zeta + rrk.zeta
            else:
                self._zeta = self.raw_zeta - drk.zeta - rdk.zeta + rrk.zeta

            self._rrk = rrk
            self._drk = drk
            self._rdk = rdk

            if (rrk.npatch1 not in (1,self.npatch1) or rrk.npatch2 not in (1,self.npatch2)
                    or rrk.npatch3 != self.npatch3):
                raise RuntimeError("RRK must be run with the same patches as DDK")
            if rdk is not None and (rdk.npatch1 not in (1,self.npatch1)
                                    or rdk.npatch2 != self.npatch2
                                    or rdk.npatch3 != self.npatch3):
                raise RuntimeError("RDK must be run with the same patches as DDK")
            if drk is not None and (drk.npatch2 not in (1,self.npatch2)
                                    or drk.npatch1 != self.npatch1
                                    or drk.npatch3 != self.npatch3):
                raise RuntimeError("DRK must be run with the same patches as DDK")

            if len(self.results) > 0:
                added_any = False
                for results in (rrk.results,
                                rdk.results if rdk is not None else None,
                                drk.results if drk is not None else None):
                    if results is None:
                        continue
                    for ijk in results:
                        if ijk in self.results: continue
                        self.results[ijk] = self._zero_copy()
                        added_any = True
                if added_any:
                    self.__dict__.pop('_ok',None)

                self._cov = self.estimate_cov(self.var_method)
                self._comp_varzeta = np.zeros(self.data_shape, dtype=float)
                self._comp_varzeta.ravel()[:] = self.cov_diag
            else:
                self._comp_varzeta = self.raw_varzeta + rrk.varzeta
                if rdk is not None:
                    self._comp_varzeta += rdk.varzeta
                if drk is not None:
                    self._comp_varzeta += drk.varzeta
        else:
            if rdk is not None:
                raise TypeError("rdk is invalid if rrk is None")
            if drk is not None:
                raise TypeError("drk is invalid if rrk is None")
            self._zeta = self.raw_zeta
            self._comp_varzeta = None

        return self._zeta, self.varzeta

    def _calculate_xi_from_pairs(self, pairs, corr_only):
        super()._calculate_xi_from_pairs(pairs, corr_only)
        if self._rrk is not None:
            # If r doesn't have patches, then convert all (i,i,i) pairs to (0,0,i).
            if self._rrk.npatch1 == 1 and not all(p[0] == 0 for p in pairs):
                pairs1 = [(0,0,k,w) for i,j,k,w in pairs if i == j == k]
            else:
                pairs1 = pairs
            pairs1 = self._rrk._keep_ok(pairs1)
            self._rrk._calculate_xi_from_pairs(pairs1, corr_only=True)

        if self._drk is not None:
            pairs2 = pairs
            if self._drk.npatch2 == 1 and not all(p[1] == 0 for p in pairs):
                pairs2 = [(i,0,k,w) for i,j,k,w in pairs if j == k]
            pairs2 = self._drk._keep_ok(pairs2)
            self._drk._calculate_xi_from_pairs(pairs2, corr_only=True)

        if self._rdk is not None:
            pairs3 = pairs
            if self._rdk.npatch1 == 1 and not all(p[0] == 0 for p in pairs):
                pairs3 = [(0,j,k,w) for i,j,k,w in pairs if i == k]
            pairs3 = self._rdk._keep_ok(pairs3)
            self._rdk._calculate_xi_from_pairs(pairs3, corr_only=True)

        if self._rrk is None:
            self._zeta = None
        elif self._drk is None and self._rdk is None:
            self._zeta = self.raw_zeta - self._rrk.zeta
        elif self._rdk is not None and self._drk is None:
            self._zeta = self.raw_zeta - 2.*self._rdk.zeta + self._rrk.zeta
        elif self._drk is not None and self._rdk is None:
            self._zeta = self.raw_zeta - 2.*self._drk.zeta + self._rrk.zeta
        else:
            self._zeta = self.raw_zeta - self._drk.zeta - self._rdk.zeta + self._rrk.zeta

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
