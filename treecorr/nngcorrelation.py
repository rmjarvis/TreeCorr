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
.. module:: nngcorrelation
"""

import numpy as np

from . import _treecorr
from .catalog import calculateVarG
from .corr3base import Corr3


class GNNCorrelation(Corr3):
    r"""This class handles the calculation and storage of a 3-point shear-count-count correlation
    function.

    With this class, point 1 of the triangle (i.e. the vertex opposite d1) is the one with the
    shear value.  Use `NGNCorrelation` and `NNGCorrelation` for classes with the shear in the
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

        >>> gnn = treecorr.GNNCorrelation(config)
        >>> gnn.process(cat1, cat2)          # Compute the cross-correlation of two fields.
        >>> # gnn.process(cat1, cat2, cat3)  # ... or of three fields.
        >>> grr.process(cat1, rand)          # Compute the random cross-correlation.
        >>> gdr.process(cat1, cat2, rand)    # Optionally compute data-random cross-correlation.
        >>> gnn.write(file_name)             # Write out to a file.
        >>> gnn.calculateZeta(grr=grr, gdr=gdr)  # Calculate zeta using randoms.
        >>> zeta = gnn.zeta                  # Access the correlation function.
        >>> zetar = gnn.zetar                # Or access real and imaginary parts separately.
        >>> zetai = gnn.zetai

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
    _cls = 'GNNCorrelation'
    _letter1 = 'G'
    _letter2 = 'N'
    _letter3 = 'N'
    _letters = 'GNN'
    _builder = _treecorr.GNNCorr
    _calculateVar1 = staticmethod(calculateVarG)
    _calculateVar2 = lambda *args, **kwargs: None
    _calculateVar3 = lambda *args, **kwargs: None
    _sig1 = 'sig_sn (per component)'
    _sig2 = None
    _sig3 = None

    def __init__(self, config=None, *, logger=None, **kwargs):
        super().__init__(config, logger=logger, **kwargs)

        self._grr = None
        self._gdr = None
        self._grd = None
        shape = self.data_shape
        self._z[0:2] = [np.zeros(shape, dtype=float) for _ in range(2)]
        self._zeta = None
        self._comp_varzeta = None
        self.logger.debug('Finished building GNNCorr')

    @property
    def raw_zeta(self):
        return self._z[0] + 1j * self._z[1]

    @property
    def zeta(self):
        if self._zeta is None:
            return self.raw_zeta
        else:
            return self._zeta

    @property
    def zetar(self):
        if self._zeta is None:
            return self._z[0]
        else:
            return np.real(self._zeta)

    @property
    def zetai(self):
        if self._zeta is None:
            return self._z[1]
        else:
            return np.imag(self._zeta)

    def copy(self):
        ret = super().copy()
        # True is possible during read before we finish reading in these attributes.
        if self._grr is not None and self._grr is not True:
            ret._grr = self._grr.copy()
        if self._gdr is not None and self._gdr is not True:
            ret._gdr = self._gdr.copy()
        if self._grd is not None and self._grd is not True:
            ret._grd = self._grd.copy()
        return ret

    def finalize(self, varg):
        """Finalize the calculation of the correlation function.

        Parameters:
            varg (float):   The variance of the shear field.
        """
        self._finalize()
        self._var_num = varg
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
        self._grr = None
        self._grd = None
        self._gdr = None
        self._zeta = None
        self._comp_varzeta = None

    def calculateZeta(self, *, grr=None, gdr=None, grd=None):
        r"""Calculate the correlation function given another correlation function of random
        points using the same mask, and possibly cross-correlations of the data and random.

        The grr value is the `GNNCorrelation` function for random points with the shear field.
        One can also provide a cross-correlation of the count data with randoms and the shear.

        - If grr is None, the simple correlation function (self.zeta) is returned.
        - If only grr is given the compensated value :math:`\zeta = GDD - GRR` is returned.
        - If gdr is given and grd is None (or vice versa), then :math:`\zeta = GDD - 2GDR + GRR`
          is returned.
        - If gdr and grd are both given, then :math:`\zeta = GDD - GDR - GRD + GRR` is returned.

        Here GDD is the data GNN correlation function, which is the current object.

        After calling this method, you can use the `Corr3.estimate_cov` method or use this
        correlation object in the `estimate_multi_cov` function.  Also, the calculated zeta and
        varzeta returned from this function will be available as attributes.

        Parameters:
            grr (GNNCorrelation):   The correlation of the random points with the shear field
                                    (GRR) (default: None)
            gdr (GNNCorrelation):   The cross-correlation of the data with both randoms and the
                                    shear field (GDR), if desired. (default: None)
            grd (GNNCorrelation):   The cross-correlation of the randoms with both the data and the
                                    shear field (GRD), if desired. (default: None)

        Returns:
            Tuple containing:

            - zeta = array of :math:`\zeta(r)`
            - varzeta = an estimate of the variance of :math:`\zeta(r)`
        """
        # Calculate zeta based on which randoms are provided.
        if grr is not None:
            if self.bin_type == 'LogMultipole':
                raise TypeError("calculateZeta is not valid for LogMultipole binning")

            if gdr is None and grd is None:
                self._zeta = self.raw_zeta - grr.zeta
            elif grd is not None and gdr is None:
                self._zeta = self.raw_zeta - 2.*grd.zeta + grr.zeta
            elif gdr is not None and grd is None:
                self._zeta = self.raw_zeta - 2.*gdr.zeta + grr.zeta
            else:
                self._zeta = self.raw_zeta - gdr.zeta - grd.zeta + grr.zeta

            self._grr = grr
            self._gdr = gdr
            self._grd = grd

            if (grr.npatch2 not in (1,self.npatch2) or grr.npatch3 not in (1,self.npatch3)
                    or grr.npatch1 != self.npatch1):
                raise RuntimeError("GRR must be run with the same patches as GDD")
            if grd is not None and (grd.npatch2 not in (1,self.npatch2)
                                    or grd.npatch1 != self.npatch1
                                    or grd.npatch3 != self.npatch3):
                raise RuntimeError("GRD must be run with the same patches as GDD")
            if gdr is not None and (gdr.npatch3 not in (1,self.npatch3)
                                    or gdr.npatch1 != self.npatch1
                                    or gdr.npatch2 != self.npatch2):
                raise RuntimeError("GDR must be run with the same patches as GDD")

            if len(self.results) > 0:
                template = next(iter(self.results.values()))  # Just need something to copy.
                all_keys = list(grr.results.keys())
                if grd is not None:
                    all_keys += list(grd.results.keys())
                if gdr is not None:
                    all_keys += list(gdr.results.keys())
                for ijk in all_keys:
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
                self._comp_varzeta = self.raw_varzeta + grr.varzeta
                if grd is not None:
                    self._comp_varzeta += grd.varzeta
                if gdr is not None:
                    self._comp_varzeta += gdr.varzeta
        else:
            if grd is not None:
                raise TypeError("grd is invalid if grr is None")
            if gdr is not None:
                raise TypeError("gdr is invalid if grr is None")
            self._zeta = self.raw_zeta
            self._comp_varzeta = None

        return self._zeta, self.varzeta

    def _calculate_xi_from_pairs(self, pairs, corr_only):
        super()._calculate_xi_from_pairs(pairs, corr_only)
        if self._grr is not None:
            # If r doesn't have patches, then convert all (i,i,i) pairs to (i,0,0).
            if self._grr.npatch2 == 1 and not all([p[1] == 0 for p in pairs]):
                pairs1 = [(i,0,0,w) for i,j,k,w in pairs if i == j == k]
            else:
                pairs1 = pairs
            pairs1 = self._grr._keep_ok(pairs1)
            self._grr._calculate_xi_from_pairs(pairs1, corr_only=True)

        if self._gdr is not None:
            pairs2 = pairs
            if self._gdr.npatch3 == 1 and not all([p[2] == 0 for p in pairs]):
                pairs2 = [(i,j,0,w) for i,j,k,w in pairs if j == k]
            pairs2 = self._gdr._keep_ok(pairs2)
            self._gdr._calculate_xi_from_pairs(pairs2, corr_only=True)

        if self._grd is not None:
            pairs3 = pairs
            if self._grd.npatch2 == 1 and not all([p[1] == 0 for p in pairs]):
                pairs3 = [(i,0,k,w) for i,j,k,w in pairs if i == k]
            pairs3 = self._grd._keep_ok(pairs3)
            self._grd._calculate_xi_from_pairs(pairs3, corr_only=True)

        if self._grr is None:
            self._zeta = None
        elif self._gdr is None and self._grd is None:
            self._zeta = self.raw_zeta - self._grr.zeta
        elif self._grd is not None and self._gdr is None:
            self._zeta = self.raw_zeta - 2.*self._grd.zeta + self._grr.zeta
        elif self._gdr is not None and self._grd is None:
            self._zeta = self.raw_zeta - 2.*self._gdr.zeta + self._grr.zeta
        else:
            self._zeta = self.raw_zeta - self._gdr.zeta - self._grd.zeta + self._grr.zeta

    def write(self, file_name, *, file_type=None, precision=None, write_patch_results=False,
              write_cov=False):
        super().write(file_name, file_type=file_type, precision=precision,
                      write_patch_results=write_patch_results, write_cov=write_cov)

    write.__doc__ = Corr3.write.__doc__.format(
        r"""
        zetar           The real part of the estimator of :math:`\zeta`
        zetai           The imaginary part of the estimator of :math:`\zeta`
        sigma_zeta      The sqrt of the variance estimate of :math:`\zeta`.
        """)

    @property
    def _write_class_col_names(self):
        return ['zetar', 'zetai', 'sigma_zeta']

    @property
    def _write_class_data(self):
        return [self._z[0], self._z[1], np.sqrt(self.varzeta) ]

    def _read_from_data(self, data, params):
        super()._read_from_data(data, params)
        s = self.data_shape
        self._z[0] = data['zetar'].reshape(s)
        self._z[1] = data['zetai'].reshape(s)
        self._comp_varzeta = data['sigma_zeta'].reshape(s)**2
        self._varzeta = [self._comp_varzeta]

class NGNCorrelation(Corr3):
    r"""This class handles the calculation and storage of a 3-point count-shear-count correlation
    function.

    With this class, point 2 of the triangle (i.e. the vertex opposite d2) is the one with the
    shear value.  Use `GNNCorrelation` and `NNGCorrelation` for classes with the shear in the
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

        >>> ngn = treecorr.NGNCorrelation(config)
        >>> ngn.process(cat1, cat2, cat1)    # Compute the cross-correlation of two fields.
        >>> # ngn.process(cat1, cat2, cat3)  # ... or of three fields.
        >>> rgr.process(rand, cat2, rand)    # Compute the random cross-correlation.
        >>> dgr.process(cat1, cat2, rand)    # Optionally compute data-random cross-correlation.
        >>> ngn.write(file_name)             # Write out to a file.
        >>> ngn.calculateZeta(rgr=rgr, dgr=dgr)  # Calculate zeta using randoms.
        >>> zeta = ngn.zeta                  # Access the correlation function.
        >>> zetar = ngn.zetar                # Or access real and imaginary parts separately.
        >>> zetai = ngn.zetai

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
    _cls = 'NGNCorrelation'
    _letter1 = 'N'
    _letter2 = 'G'
    _letter3 = 'N'
    _letters = 'NGN'
    _builder = _treecorr.NGNCorr
    _calculateVar1 = lambda *args, **kwargs: None
    _calculateVar2 = staticmethod(calculateVarG)
    _calculateVar3 = lambda *args, **kwargs: None
    _sig1 = None
    _sig2 = 'sig_sn (per component)'
    _sig3 = None

    def __init__(self, config=None, *, logger=None, **kwargs):
        super().__init__(config, logger=logger, **kwargs)

        self._rgr = None
        self._dgr = None
        self._rgd = None
        shape = self.data_shape
        self._z[0] = np.zeros(shape, dtype=float)
        self._z[1] = np.zeros(shape, dtype=float)
        self._zeta = None
        self._comp_varzeta = None
        self.logger.debug('Finished building NGNCorr')

    @property
    def raw_zeta(self):
        return self._z[0] + 1j * self._z[1]

    @property
    def zeta(self):
        if self._zeta is None:
            return self.raw_zeta
        else:
            return self._zeta

    @property
    def zetar(self):
        if self._zeta is None:
            return self._z[0]
        else:
            return np.real(self._zeta)

    @property
    def zetai(self):
        if self._zeta is None:
            return self._z[1]
        else:
            return np.imag(self._zeta)

    def copy(self):
        ret = super().copy()
        # True is possible during read before we finish reading in these attributes.
        if self._rgr is not None and self._rgr is not True:
            ret._rgr = self._rgr.copy()
        if self._dgr is not None and self._dgr is not True:
            ret._dgr = self._dgr.copy()
        if self._rgd is not None and self._rgd is not True:
            ret._rgd = self._rgd.copy()
        return ret

    def finalize(self, varg):
        """Finalize the calculation of the correlation function.

        Parameters:
            varg (float):   The variance of the shear field.
        """
        self._finalize()
        self._var_num = varg
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
        self._rgr = None
        self._rgd = None
        self._dgr = None
        self._zeta = None
        self._comp_varzeta = None

    def calculateZeta(self, *, rgr=None, dgr=None, rgd=None):
        r"""Calculate the correlation function given another correlation function of random
        points using the same mask, and possibly cross-correlations of the data and random.

        The rgr value is the `NGNCorrelation` function for random points with the shear field.
        One can also provide a cross-correlation of the count data with randoms and the shear.

        - If rgr is None, the simple correlation function (self.zeta) is returned.
        - If only rgr is given the compensated value :math:`\zeta = DGD - RGR` is returned.
        - If dgr is given and rgd is None (or vice versa), then :math:`\zeta = DGD - 2DGR + RGR`
          is returned.
        - If dgr and rgd are both given, then :math:`\zeta = DGD - DGR - RGD + RGR` is returned.

        Here DGD is the data NGN correlation function, which is the current object.

        After calling this method, you can use the `Corr3.estimate_cov` method or use this
        correlation object in the `estimate_multi_cov` function.  Also, the calculated zeta and
        varzeta returned from this function will be available as attributes.

        Parameters:
            rgr (NGNCorrelation):   The correlation of the random points with the shear field
                                    (RGR) (default: None)
            dgr (NGNCorrelation):   The cross-correlation of the data with both randoms and the
                                    shear field (DGR), if desired. (default: None)
            rgd (NGNCorrelation):   The cross-correlation of the randoms with both the data and the
                                    shear field (RGD), if desired. (default: None)

        Returns:
            Tuple containing:

            - zeta = array of :math:`\zeta(r)`
            - varzeta = an estimate of the variance of :math:`\zeta(r)`
        """
        # Calculate zeta based on which randoms are provided.
        if rgr is not None:
            if self.bin_type == 'LogMultipole':
                raise TypeError("calculateZeta is not valid for LogMultipole binning")

            if dgr is None and rgd is None:
                self._zeta = self.raw_zeta - rgr.zeta
            elif rgd is not None and dgr is None:
                self._zeta = self.raw_zeta - 2.*rgd.zeta + rgr.zeta
            elif dgr is not None and rgd is None:
                self._zeta = self.raw_zeta - 2.*dgr.zeta + rgr.zeta
            else:
                self._zeta = self.raw_zeta - dgr.zeta - rgd.zeta + rgr.zeta

            self._rgr = rgr
            self._dgr = dgr
            self._rgd = rgd

            if (rgr.npatch1 not in (1,self.npatch1) or rgr.npatch3 not in (1,self.npatch3)
                    or rgr.npatch2 != self.npatch2):
                raise RuntimeError("RGR must be run with the same patches as DGD")
            if rgd is not None and (rgd.npatch1 not in (1,self.npatch1)
                                    or rgd.npatch2 != self.npatch2
                                    or rgd.npatch3 != self.npatch3):
                raise RuntimeError("RGD must be run with the same patches as DGD")
            if dgr is not None and (dgr.npatch3 not in (1,self.npatch3)
                                    or dgr.npatch1 != self.npatch1
                                    or dgr.npatch2 != self.npatch2):
                raise RuntimeError("DGR must be run with the same patches as DGD")

            if len(self.results) > 0:
                template = next(iter(self.results.values()))  # Just need something to copy.
                all_keys = list(rgr.results.keys())
                if rgd is not None:
                    all_keys += list(rgd.results.keys())
                if dgr is not None:
                    all_keys += list(dgr.results.keys())
                for ijk in all_keys:
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
                self._comp_varzeta = self.raw_varzeta + rgr.varzeta
                if rgd is not None:
                    self._comp_varzeta += rgd.varzeta
                if dgr is not None:
                    self._comp_varzeta += dgr.varzeta
        else:
            if rgd is not None:
                raise TypeError("rgd is invalid if rgr is None")
            if dgr is not None:
                raise TypeError("dgr is invalid if rgr is None")
            self._zeta = self.raw_zeta
            self._comp_varzeta = None

        return self._zeta, self.varzeta

    def _calculate_xi_from_pairs(self, pairs, corr_only):
        super()._calculate_xi_from_pairs(pairs, corr_only)
        if self._rgr is not None:
            # If r doesn't have patches, then convert all (i,i,i) pairs to (0,i,0).
            if self._rgr.npatch1 == 1 and not all([p[0] == 0 for p in pairs]):
                pairs1 = [(0,j,0,w) for i,j,k,w in pairs if i == j == k]
            else:
                pairs1 = pairs
            pairs1 = self._rgr._keep_ok(pairs1)
            self._rgr._calculate_xi_from_pairs(pairs1, corr_only=True)

        if self._dgr is not None:
            pairs2 = pairs
            if self._dgr.npatch3 == 1 and not all([p[2] == 0 for p in pairs]):
                pairs2 = [(i,j,0,w) for i,j,k,w in pairs if j == k]
            pairs2 = self._dgr._keep_ok(pairs2)
            self._dgr._calculate_xi_from_pairs(pairs2, corr_only=True)

        if self._rgd is not None:
            pairs3 = pairs
            if self._rgd.npatch1 == 1 and not all([p[0] == 0 for p in pairs]):
                pairs3 = [(0,j,k,w) for i,j,k,w in pairs if i == k]
            pairs3 = self._rgd._keep_ok(pairs3)
            self._rgd._calculate_xi_from_pairs(pairs3, corr_only=True)

        if self._rgr is None:
            self._zeta = None
        elif self._dgr is None and self._rgd is None:
            self._zeta = self.raw_zeta - self._rgr.zeta
        elif self._rgd is not None and self._dgr is None:
            self._zeta = self.raw_zeta - 2.*self._rgd.zeta + self._rgr.zeta
        elif self._dgr is not None and self._rgd is None:
            self._zeta = self.raw_zeta - 2.*self._dgr.zeta + self._rgr.zeta
        else:
            self._zeta = self.raw_zeta - self._dgr.zeta - self._rgd.zeta + self._rgr.zeta

    def write(self, file_name, *, file_type=None, precision=None, write_patch_results=False,
              write_cov=False):
        super().write(file_name, file_type=file_type, precision=precision,
                      write_patch_results=write_patch_results, write_cov=write_cov)

    write.__doc__ = Corr3.write.__doc__.format(
        r"""
        zetar           The real part of the estimator of :math:`\zeta`
        zetai           The imaginary part of the estimator of :math:`\zeta`
        sigma_zeta      The sqrt of the variance estimate of :math:`\zeta`.
        """)

    @property
    def _write_class_col_names(self):
        return ['zetar', 'zetai', 'sigma_zeta']

    @property
    def _write_class_data(self):
        return [self._z[0], self._z[1], np.sqrt(self.varzeta) ]

    def _read_from_data(self, data, params):
        super()._read_from_data(data, params)
        s = self.data_shape
        self._z[0] = data['zetar'].reshape(s)
        self._z[1] = data['zetai'].reshape(s)
        self._comp_varzeta = data['sigma_zeta'].reshape(s)**2
        self._varzeta = [self._comp_varzeta]

class NNGCorrelation(Corr3):
    r"""This class handles the calculation and storage of a 3-point count-count-shear correlation
    function.

    With this class, point 3 of the triangle (i.e. the vertex opposite d3) is the one with the
    shear value.  Use `GNNCorrelation` and `NGNCorrelation` for classes with the shear in the
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

        >>> nng = treecorr.NNGCorrelation(config)
        >>> nng.process(cat1, cat2)          # Compute the cross-correlation of two fields.
        >>> # nng.process(cat1, cat2, cat3)  # ... or of three fields.
        >>> rrg.process(rand, cat2)          # Compute the random cross-correlation.
        >>> drg.process(cat1, rand, cat2)    # Optionally compute data-random cross-correlation.
        >>> nng.write(file_name)             # Write out to a file.
        >>> nng.calculateZeta(rrg=rrg, drg=drg)  # Calculate zeta using randoms.
        >>> zeta = nng.zeta                  # Access the correlation function.
        >>> zetar = nng.zetar                # Or access real and imaginary parts separately.
        >>> zetai = nng.zetai

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
    _cls = 'NNGCorrelation'
    _letter1 = 'N'
    _letter2 = 'N'
    _letter3 = 'G'
    _letters = 'NNG'
    _builder = _treecorr.NNGCorr
    _calculateVar1 = lambda *args, **kwargs: None
    _calculateVar2 = lambda *args, **kwargs: None
    _calculateVar3 = staticmethod(calculateVarG)
    _sig1 = None
    _sig2 = None
    _sig3 = 'sig_sn (per component)'

    def __init__(self, config=None, *, logger=None, **kwargs):
        super().__init__(config, logger=logger, **kwargs)

        self._rrg = None
        self._drg = None
        self._rdg = None
        shape = self.data_shape
        self._z[0] = np.zeros(shape, dtype=float)
        self._z[1] = np.zeros(shape, dtype=float)
        self._zeta = None
        self._comp_varzeta = None
        self.logger.debug('Finished building NNGCorr')

    @property
    def raw_zeta(self):
        return self._z[0] + 1j * self._z[1]

    @property
    def zeta(self):
        if self._zeta is None:
            return self.raw_zeta
        else:
            return self._zeta

    @property
    def zetar(self):
        if self._zeta is None:
            return self._z[0]
        else:
            return np.real(self._zeta)

    @property
    def zetai(self):
        if self._zeta is None:
            return self._z[1]
        else:
            return np.imag(self._zeta)

    def copy(self):
        ret = super().copy()
        # True is possible during read before we finish reading in these attributes.
        if self._rrg is not None and self._rrg is not True:
            ret._rrg = self._rrg.copy()
        if self._drg is not None and self._drg is not True:
            ret._drg = self._drg.copy()
        if self._rdg is not None and self._rdg is not True:
            ret._rdg = self._rdg.copy()
        return ret

    def finalize(self, varg):
        """Finalize the calculation of the correlation function.

        Parameters:
            varg (float):   The variance of the shear field.
        """
        self._finalize()
        self._var_num = varg
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
        self._rrg = None
        self._rdg = None
        self._drg = None
        self._zeta = None
        self._comp_varzeta = None

    def calculateZeta(self, *, rrg=None, drg=None, rdg=None):
        r"""Calculate the correlation function given another correlation function of random
        points using the same mask, and possibly cross-correlations of the data and random.

        The rrg value is the `NNGCorrelation` function for random points with the shear field.
        One can also provide a cross-correlation of the count data with randoms and the shear.

        - If rrg is None, the simple correlation function (self.zeta) is returned.
        - If only rrg is given the compensated value :math:`\zeta = DDG - RRG` is returned.
        - If drg is given and rdg is None (or vice versa), then :math:`\zeta = DDG - 2DRG + RRG`
          is returned.
        - If drg and rdg are both given, then :math:`\zeta = DDG - DRG - RDG + RRG` is returned.

        Here DDG is the data NNG correlation function, which is the current object.

        After calling this method, you can use the `Corr3.estimate_cov` method or use this
        correlation object in the `estimate_multi_cov` function.  Also, the calculated zeta and
        varzeta returned from this function will be available as attributes.

        Parameters:
            rrg (NNGCorrelation):   The correlation of the random points with the shear field
                                    (RRG) (default: None)
            drg (NNGCorrelation):   The cross-correlation of the data with both randoms and the
                                    shear field (DRG), if desired. (default: None)
            rdg (NNGCorrelation):   The cross-correlation of the randoms with both the data and the
                                    shear field (RDG), if desired. (default: None)

        Returns:
            Tuple containing:

            - zeta = array of :math:`\zeta(r)`
            - varzeta = an estimate of the variance of :math:`\zeta(r)`
        """
        # Calculate zeta based on which randoms are provided.
        if rrg is not None:
            if self.bin_type == 'LogMultipole':
                raise TypeError("calculateZeta is not valid for LogMultipole binning")

            if drg is None and rdg is None:
                self._zeta = self.raw_zeta - rrg.zeta
            elif rdg is not None and drg is None:
                self._zeta = self.raw_zeta - 2.*rdg.zeta + rrg.zeta
            elif drg is not None and rdg is None:
                self._zeta = self.raw_zeta - 2.*drg.zeta + rrg.zeta
            else:
                self._zeta = self.raw_zeta - drg.zeta - rdg.zeta + rrg.zeta

            self._rrg = rrg
            self._drg = drg
            self._rdg = rdg

            if (rrg.npatch1 not in (1,self.npatch1) or rrg.npatch2 not in (1,self.npatch2)
                    or rrg.npatch3 != self.npatch3):
                raise RuntimeError("RRG must be run with the same patches as DDG")
            if rdg is not None and (rdg.npatch1 not in (1,self.npatch1)
                                    or rdg.npatch2 != self.npatch2
                                    or rdg.npatch3 != self.npatch3):
                raise RuntimeError("RDG must be run with the same patches as DDG")
            if drg is not None and (drg.npatch2 not in (1,self.npatch2)
                                    or drg.npatch1 != self.npatch1
                                    or drg.npatch3 != self.npatch3):
                raise RuntimeError("DRG must be run with the same patches as DDG")

            if len(self.results) > 0:
                template = next(iter(self.results.values()))  # Just need something to copy.
                all_keys = list(rrg.results.keys())
                if rdg is not None:
                    all_keys += list(rdg.results.keys())
                if drg is not None:
                    all_keys += list(drg.results.keys())
                for ijk in all_keys:
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
                self._comp_varzeta = self.raw_varzeta + rrg.varzeta
                if rdg is not None:
                    self._comp_varzeta += rdg.varzeta
                if drg is not None:
                    self._comp_varzeta += drg.varzeta
        else:
            if rdg is not None:
                raise TypeError("rdg is invalid if rrg is None")
            if drg is not None:
                raise TypeError("drg is invalid if rrg is None")
            self._zeta = self.raw_zeta
            self._comp_varzeta = None

        return self._zeta, self.varzeta

    def _calculate_xi_from_pairs(self, pairs, corr_only):
        super()._calculate_xi_from_pairs(pairs, corr_only)
        if self._rrg is not None:
            # If r doesn't have patches, then convert all (i,i,i) pairs to (0,0,i).
            if self._rrg.npatch1 == 1 and not all([p[0] == 0 for p in pairs]):
                pairs1 = [(0,0,k,w) for i,j,k,w in pairs if i == j == k]
            else:
                pairs1 = pairs
            pairs1 = self._rrg._keep_ok(pairs1)
            self._rrg._calculate_xi_from_pairs(pairs1, corr_only=True)

        if self._drg is not None:
            pairs2 = pairs
            if self._drg.npatch2 == 1 and not all([p[1] == 0 for p in pairs]):
                pairs2 = [(i,0,k,w) for i,j,k,w in pairs if j == k]
            pairs2 = self._drg._keep_ok(pairs2)
            self._drg._calculate_xi_from_pairs(pairs2, corr_only=True)

        if self._rdg is not None:
            pairs3 = pairs
            if self._rdg.npatch1 == 1 and not all([p[0] == 0 for p in pairs]):
                pairs3 = [(0,j,k,w) for i,j,k,w in pairs if i == k]
            pairs3 = self._rdg._keep_ok(pairs3)
            self._rdg._calculate_xi_from_pairs(pairs3, corr_only=True)

        if self._rrg is None:
            self._zeta = None
        elif self._drg is None and self._rdg is None:
            self._zeta = self.raw_zeta - self._rrg.zeta
        elif self._rdg is not None and self._drg is None:
            self._zeta = self.raw_zeta - 2.*self._rdg.zeta + self._rrg.zeta
        elif self._drg is not None and self._rdg is None:
            self._zeta = self.raw_zeta - 2.*self._drg.zeta + self._rrg.zeta
        else:
            self._zeta = self.raw_zeta - self._drg.zeta - self._rdg.zeta + self._rrg.zeta

    def write(self, file_name, *, file_type=None, precision=None, write_patch_results=False,
              write_cov=False):
        super().write(file_name, file_type=file_type, precision=precision,
                      write_patch_results=write_patch_results, write_cov=write_cov)

    write.__doc__ = Corr3.write.__doc__.format(
        r"""
        zetar           The real part of the estimator of :math:`\zeta`
        zetai           The imaginary part of the estimator of :math:`\zeta`
        sigma_zeta      The sqrt of the variance estimate of :math:`\zeta`.
        """)

    @property
    def _write_class_col_names(self):
        return ['zetar', 'zetai', 'sigma_zeta']

    @property
    def _write_class_data(self):
        return [self._z[0], self._z[1], np.sqrt(self.varzeta) ]

    def _read_from_data(self, data, params):
        super()._read_from_data(data, params)
        s = self.data_shape
        self._z[0] = data['zetar'].reshape(s)
        self._z[1] = data['zetai'].reshape(s)
        self._comp_varzeta = data['sigma_zeta'].reshape(s)**2
        self._varzeta = [self._comp_varzeta]
