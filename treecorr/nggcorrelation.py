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
.. module:: nggcorrelation
"""

import numpy as np

from . import _treecorr
from .catalog import calculateVarG
from .corr3base import Corr3


class NGGCorrelation(Corr3):
    r"""This class handles the calculation and storage of a 3-point count-shear-shear correlation
    function.

    With this class, points 2 and 3 of the triangle (i.e. the vertces opposite d2,d3) are the ones
    with the shear values.  Use `GNGCorrelation` and `GGNCorrelation` for classes with the shears
    in the other positions.

    For the shear projection, we follow the lead of the 3-point shear-shear-shear correlation
    functions (see `GGGCorrelation` for details), which involves projecting the shear values
    at each vertex relative to the direction to the triangle's centroid.  Furthermore, the
    GGG correlations have 4 relevant complex values for each triangle:

    .. math::

        \Gamma_0 &= \langle \gamma(\mathbf{x1}) \gamma(\mathbf{x2}) \gamma(\mathbf{x3}) \rangle \\
        \Gamma_1 &= \langle \gamma(\mathbf{x1})^* \gamma(\mathbf{x2}) \gamma(\mathbf{x3}) \rangle \\
        \Gamma_2 &= \langle \gamma(\mathbf{x1}) \gamma(\mathbf{x2})^* \gamma(\mathbf{x3}) \rangle \\
        \Gamma_3 &= \langle \gamma(\mathbf{x1}) \gamma(\mathbf{x2}) \gamma(\mathbf{x3}^*) \rangle \\

    With only a count at vertex 1, :math:`\Gamma_0 = \Gamma_1` and :math:`\Gamma_2 = \Gamma_3^*`.
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

        >>> ngg = treecorr.NGGCorrelation(config)
        >>> ngg.process(cat1, cat2)         # Compute the cross-correlation of two fields.
        >>> # ngg.process(cat1, cat2, cat3) # ... or of three fields.
        >>> ngg.write(file_name)            # Write out to a file.
        >>> rgg.process(rand, cat2)         # Compute the random cross-correlation.
        >>> ngg.calculateZeta(rgg=rgg)      # Calculate zeta using randoms.
        >>> gam0 = ngg.gam0, etc.           # Access gamma values directly.
        >>> gam0r = ngg.gam0r               # Or access real and imaginary parts separately.
        >>> gam0i = ngg.gam0i

    See also: `GNGCorrelation`, `GGNCorrelation`, `NNGCorrelation`, `GGGCorrelation`,
    `NGCorrelation`.

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
    _cls = 'NGGCorrelation'
    _letter1 = 'N'
    _letter2 = 'G'
    _letter3 = 'G'
    _letters = 'NGG'
    _builder = _treecorr.NGGCorr
    _calculateVar1 = lambda *args, **kargs: None
    _calculateVar2 = staticmethod(calculateVarG)
    _calculateVar3 = staticmethod(calculateVarG)
    _sig1 = None
    _sig2 = 'sig_sn (per component)'
    _sig3 = 'sig_sn (per component)'

    def __init__(self, config=None, *, logger=None, **kwargs):
        super().__init__(config, logger=logger, **kwargs)

        self._rgg = None
        shape = self.data_shape
        # z0,1 holds gamma_0
        # z2,3 holds gamma_2
        self._z[0:4] = [np.zeros(shape, dtype=float) for _ in range(4)]
        self._gam0 = None
        self._gam2 = None
        self._comp_vargam0 = None
        self._comp_vargam2 = None
        self.logger.debug('Finished building NGGCorr')

    @property
    def raw_gam0(self):
        return self._z[0] + 1j * self._z[1]

    @property
    def raw_gam2(self):
        return self._z[2] + 1j * self._z[3]

    @property
    def gam0(self):
        if self._gam0 is None:
            return self._z[0] + 1j * self._z[1]
        else:
            return self._gam0

    @property
    def gam1(self):
        return self.gam0

    @property
    def gam2(self):
        if self._gam2 is None:
            return self._z[2] + 1j * self._z[3]
        else:
            return self._gam2

    @property
    def gam3(self):
        return np.conjugate(self.gam2)

    @property
    def gam0r(self):
        if self._gam0 is None:
            return self._z[0]
        else:
            return np.real(self._gam0)

    @property
    def gam0i(self):
        if self._gam0 is None:
            return self._z[1]
        else:
            return np.imag(self._gam0)

    @property
    def gam1r(self):
        return self.gam0r

    @property
    def gam1i(self):
        return self.gam0i

    @property
    def gam2r(self):
        if self._gam2 is None:
            return self._z[2]
        else:
            return np.real(self._gam2)

    @property
    def gam2i(self):
        if self._gam2 is None:
            return self._z[3]
        else:
            return np.imag(self._gam2)

    @property
    def gam3r(self):
        return self.gam2r

    @property
    def gam3i(self):
        return -self.gam2i

    def copy(self):
        ret = super().copy()
        # True is possible during read before we finish reading in these attributes.
        if self._rgg is not None and self._rgg is not True:
            ret._rgg = self._rgg.copy()
        return ret

    def finalize(self, varg1, varg2):
        """Finalize the calculation of the correlation function.

        Parameters:
            varg1 (float):  The variance per component of the first shear field.
            varg2 (float):  The variance per component of the second shear field.
        """
        self._finalize()
        self._var_num = 2 * varg1 * varg2
        if self.bin_type in ['LogSAS', 'LogMultipole']:
            self._var_num *= 2

    @property
    def raw_vargam0(self):
        if self._varzeta is None:
            self._calculate_varzeta(2)
        return self._varzeta[0]

    @property
    def vargam0(self):
        if self._comp_vargam0 is None:
            return self.raw_vargam0
        else:
            return self._comp_vargam0

    @property
    def vargam1(self):
        return self.vargam0

    @property
    def raw_vargam2(self):
        if self._varzeta is None:
            self._calculate_varzeta(2)
        return self._varzeta[1]

    @property
    def vargam2(self):
        if self._comp_vargam2 is None:
            return self.raw_vargam2
        else:
            return self._comp_vargam2

    @property
    def vargam3(self):
        return self.vargam2

    def _clear(self):
        super()._clear()
        self._rgg = None
        self._gam0 = None
        self._gam2 = None
        self._comp_vargam0 = None
        self._comp_vargam2 = None

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

    def calculateGam(self, *, rgg=None):
        r"""Calculate the correlation function possibly given another correlation function
        that uses random points for the foreground objects.

        - If rgg is None, the simple correlation functions (self.gam0, self.gam2) are returned.
        - If rgg is not None, then a compensated calculation is done:
          :math:`\Gamma_i = (DGG - RGG)`, where DGG represents the correlation of the gamma
          field with the data points and RGG represents the correlation with random points.

        After calling this function, the attributes ``gam0``, ``gam2``, ``vargam0``,
        ``vargam2``, and ``cov`` will correspond to the compensated values (if rgg is provided).
        The raw, uncompensated values are available as ``raw_gam0``, ``raw_gam2``,
        ``raw_vargam0``, and ``raw_vargam2``.

        Parameters:
            rgg (NGGCorrelation): The cross-correlation using random locations as the lenses (RGG),
                                  if desired.  (default: None)

        Returns:
            Tuple containing
                - gam0 = array of :math:`\Gamma_0`
                - gam2 = array of :math:`\Gamma_2`
                - vargam0 = array of variance estimates of :math:`\Gamma_0`
                - vargam2 = array of variance estimates of :math:`\Gamma_2`
        """
        if rgg is not None:
            if self.bin_type == 'LogMultipole':
                raise TypeError("calculateGam is not valid for LogMultipole binning")

            self._gam0 = self.raw_gam0 - rgg.gam0
            self._gam2 = self.raw_gam2 - rgg.gam2
            self._rgg = rgg

            if (rgg.npatch1 not in (1,self.npatch1) or rgg.npatch2 != self.npatch2
                    or rgg.npatch3 != self.npatch3):
                raise RuntimeError("RGG must be run with the same patches as DGG")

            if len(self.results) > 0:
                # If there are any rgg patch pairs that aren't in results (e.g. due to different
                # edge effects among the various pairs in consideration), then we need to add
                # some dummy results to make sure all the right pairs are computed when we make
                # the vectors for the covariance matrix.
                template = next(iter(self.results.values()))  # Just need something to copy.
                for ijk in rgg.results:
                    if ijk in self.results: continue
                    new_cij = template.copy()
                    for i in range(4):
                        new_cij._z[i][:] = 0
                    new_cij.weight[:] = 0
                    self.results[ijk] = new_cij
                    self.__dict__.pop('_ok',None)

                self._cov = self.estimate_cov(self.var_method)
                self._comp_vargam0 = np.zeros(self.data_shape, dtype=float)
                self._comp_vargam2 = np.zeros(self.data_shape, dtype=float)
                n1 = self._comp_vargam0.ravel().size
                self._comp_vargam0.ravel()[:] = self.cov_diag[0:n1]
                self._comp_vargam2.ravel()[:] = self.cov_diag[n1:]
            else:
                self._comp_vargam0 = self.raw_vargam0 + rgg.vargam0
                self._comp_vargam2 = self.raw_vargam2 + rgg.vargam2
        else:
            self._gam0 = self.raw_gam0
            self._gam2 = self.raw_gam2
            self._comp_vargam0 = None
            self._comp_vargam2 = None

        return self._gam0, self._gam2, self.vargam0, self.vargam2

    def _calculate_xi_from_pairs(self, pairs, corr_only):
        super()._calculate_xi_from_pairs(pairs, corr_only)
        if self._rgg is not None:
            # If rgg has npatch1 = 1, adjust pairs appropriately
            if self._rgg.npatch1 == 1 and not all([p[0] == 0 for p in pairs]):
                pairs = [(0,j,k,w) for i,j,k,w in pairs if i == j]
            pairs = self._rgg._keep_ok(pairs)
            self._rgg._calculate_xi_from_pairs(pairs, corr_only=True)
            self._gam0 = self.raw_gam0 - self._rgg.gam0
            self._gam2 = self.raw_gam2 - self._rgg.gam2

    def write(self, file_name, *, file_type=None, precision=None, write_patch_results=False,
              write_cov=False):
        super().write(file_name, file_type=file_type, precision=precision,
                      write_patch_results=write_patch_results, write_cov=write_cov)

    write.__doc__ = Corr3.write.__doc__.format(
        r"""
        gam0r           The real part of the estimator of :math:`\Gamma_0`
        gam0i           The imaginary part of the estimator of :math:`\Gamma_0`
        gam2r           The real part of the estimator of :math:`\Gamma_2`
        gam2i           The imaginary part of the estimator of :math:`\Gamma_2`
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
        self._comp_vargam0 = data['sigma_gam0'].reshape(s)**2
        self._comp_vargam2 = data['sigma_gam2'].reshape(s)**2
        self._varzeta = [self._comp_vargam0, self._comp_vargam2]

class GNGCorrelation(Corr3):
    r"""This class handles the calculation and storage of a 3-point shear-count-shear correlation
    function.

    With this class, points 1 and 3 of the triangle (i.e. the vertces opposite d2,d3) are the ones
    with the shear values.  Use `NGGCorrelation` and `GGNCorrelation` for classes with the shears
    in the other positions.

    For the shear projection, we follow the lead of the 3-point shear-shear-shear correlation
    functions (see `GGGCorrelation` for details), which involves projecting the shear values
    at each vertex relative to the direction to the triangle's centroid.  Furthermore, the
    GGG correlations have 4 relevant complex values for each triangle:

    .. math::

        \Gamma_0 &= \langle \gamma(\mathbf{x1}) \gamma(\mathbf{x2}) \gamma(\mathbf{x3}) \rangle \\
        \Gamma_1 &= \langle \gamma(\mathbf{x1})^* \gamma(\mathbf{x2}) \gamma(\mathbf{x3}) \rangle \\
        \Gamma_2 &= \langle \gamma(\mathbf{x1}) \gamma(\mathbf{x2})^* \gamma(\mathbf{x3}) \rangle \\
        \Gamma_3 &= \langle \gamma(\mathbf{x1}) \gamma(\mathbf{x2}) \gamma(\mathbf{x3}^*) \rangle \\

    With only a count at vertex 2, :math:`\Gamma_0 = \Gamma_2` and :math:`\Gamma_1 = \Gamma_3^*`.
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

        >>> gng = treecorr.GNGCorrelation(config)
        >>> gng.process(cat1, cat2, cat1)   # Compute the cross-correlation of two fields.
        >>> # gng.process(cat1, cat2, cat3) # ... or of three fields.
        >>> gng.write(file_name)            # Write out to a file.
        >>> grg.process(cat1, rand, cat1)   # Compute the random cross-correlation.
        >>> gng.calculateZeta(grg=grg)      # Calculate zeta using randoms.
        >>> gam0 = gng.gam0, etc.           # Access gamma values directly.
        >>> gam0r = gng.gam0r               # Or access real and imaginary parts separately.
        >>> gam0i = gng.gam0i

    See also: `NGGCorrelation`, `GGNCorrelation`, `NNGCorrelation`, `GGGCorrelation`,
    `NGCorrelation`.

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
    _cls = 'GNGCorrelation'
    _letter1 = 'G'
    _letter2 = 'N'
    _letter3 = 'G'
    _letters = 'GNG'
    _builder = _treecorr.GNGCorr
    _calculateVar1 = staticmethod(calculateVarG)
    _calculateVar2 = lambda *args, **kargs: None
    _calculateVar3 = staticmethod(calculateVarG)
    _sig1 = 'sig_sn (per component)'
    _sig2 = None
    _sig3 = 'sig_sn (per component)'

    def __init__(self, config=None, *, logger=None, **kwargs):
        super().__init__(config, logger=logger, **kwargs)

        self._grg = None
        shape = self.data_shape
        self._z[0:4] = [np.zeros(shape, dtype=float) for _ in range(4)]
        self._gam0 = None
        self._gam1 = None
        self._comp_vargam0 = None
        self._comp_vargam1 = None
        self.logger.debug('Finished building GNGCorr')

    @property
    def raw_gam0(self):
        return self._z[0] + 1j * self._z[1]

    @property
    def raw_gam1(self):
        return self._z[2] + 1j * self._z[3]

    @property
    def gam0(self):
        if self._gam0 is None:
            return self._z[0] + 1j * self._z[1]
        else:
            return self._gam0

    @property
    def gam1(self):
        if self._gam1 is None:
            return self._z[2] + 1j * self._z[3]
        else:
            return self._gam1

    @property
    def gam2(self):
        return self.gam0

    @property
    def gam3(self):
        return np.conjugate(self.gam1)

    @property
    def gam0r(self):
        if self._gam0 is None:
            return self._z[0]
        else:
            return np.real(self._gam0)

    @property
    def gam0i(self):
        if self._gam0 is None:
            return self._z[1]
        else:
            return np.imag(self._gam0)

    @property
    def gam1r(self):
        if self._gam1 is None:
            return self._z[2]
        else:
            return np.real(self._gam1)

    @property
    def gam1i(self):
        if self._gam1 is None:
            return self._z[3]
        else:
            return np.imag(self._gam1)

    @property
    def gam2r(self):
        return self.gam0r

    @property
    def gam2i(self):
        return self.gam0i

    @property
    def gam3r(self):
        return self.gam1r

    @property
    def gam3i(self):
        return -self.gam1i

    def copy(self):
        ret = super().copy()
        # True is possible during read before we finish reading in these attributes.
        if self._grg is not None and self._grg is not True:
            ret._grg = self._grg.copy()
        return ret

    def finalize(self, varg1, varg2):
        """Finalize the calculation of the correlation function.

        Parameters:
            varg1 (float):  The variance per component of the first shear field.
            varg2 (float):  The variance per component of the second shear field.
        """
        self._finalize()
        self._var_num = 2 * varg1 * varg2
        if self.bin_type in ['LogSAS', 'LogMultipole']:
            self._var_num *= 2

    @property
    def raw_vargam0(self):
        if self._varzeta is None:
            self._calculate_varzeta(2)
        return self._varzeta[0]

    @property
    def vargam0(self):
        if self._comp_vargam0 is None:
            return self.raw_vargam0
        else:
            return self._comp_vargam0

    @property
    def raw_vargam1(self):
        if self._varzeta is None:
            self._calculate_varzeta(2)
        return self._varzeta[1]

    @property
    def vargam1(self):
        if self._comp_vargam1 is None:
            return self.raw_vargam1
        else:
            return self._comp_vargam1

    @property
    def vargam2(self):
        return self.vargam0

    @property
    def vargam3(self):
        return self.vargam1

    def _clear(self):
        super()._clear()
        self._grg = None
        self._gam0 = None
        self._gam1 = None
        self._comp_vargam0 = None
        self._comp_vargam1 = None

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

    def calculateGam(self, *, grg=None):
        r"""Calculate the correlation function possibly given another correlation function
        that uses random points for the foreground objects.

        - If grg is None, the simple correlation functions (self.gam0, self.gam1) are returned.
        - If grg is not None, then a compensated calculation is done:
          :math:`\Gamma_i = (GDG - GRG)`, where GDG represents the correlation of the gamma
          field with the data points and GRG represents the correlation with random points.

        After calling this function, the attributes ``gam0``, ``gam1``, ``vargam0``,
        ``vargam1``, and ``cov`` will correspond to the compensated values (if grg is provided).
        The raw, uncompensated values are available as ``raw_gam0``, ``raw_gam1``,
        ``raw_vargam0``, and ``raw_vargam1``.

        Parameters:
            grg (GNGCorrelation): The cross-correlation using random locations as the lenses (GRG),
                                  if desired.  (default: None)

        Returns:
            Tuple containing
                - gam0 = array of :math:`\Gamma_0`
                - gam1 = array of :math:`\Gamma_1`
                - vargam0 = array of variance estimates of :math:`\Gamma_0`
                - vargam1 = array of variance estimates of :math:`\Gamma_1`
        """
        if grg is not None:
            if self.bin_type == 'LogMultipole':
                raise TypeError("calculateGam is not valid for LogMultipole binning")

            self._gam0 = self.raw_gam0 - grg.gam0
            self._gam1 = self.raw_gam1 - grg.gam1
            self._grg = grg

            if (grg.npatch2 not in (1,self.npatch2) or grg.npatch1 != self.npatch1
                    or grg.npatch3 != self.npatch3):
                raise RuntimeError("GRG must be run with the same patches as GDG")

            if len(self.results) > 0:
                # If there are any grg patch pairs that aren't in results (e.g. due to different
                # edge effects among the various pairs in consideration), then we need to add
                # some dummy results to make sure all the right pairs are computed when we make
                # the vectors for the covariance matrix.
                template = next(iter(self.results.values()))  # Just need something to copy.
                for ijk in grg.results:
                    if ijk in self.results: continue
                    new_cij = template.copy()
                    for i in range(4):
                        new_cij._z[i][:] = 0
                    new_cij.weight[:] = 0
                    self.results[ijk] = new_cij
                    self.__dict__.pop('_ok',None)

                self._cov = self.estimate_cov(self.var_method)
                self._comp_vargam0 = np.zeros(self.data_shape, dtype=float)
                self._comp_vargam1 = np.zeros(self.data_shape, dtype=float)
                n1 = self._comp_vargam0.ravel().size
                self._comp_vargam0.ravel()[:] = self.cov_diag[0:n1]
                self._comp_vargam1.ravel()[:] = self.cov_diag[n1:]
            else:
                self._comp_vargam0 = self.raw_vargam0 + grg.vargam0
                self._comp_vargam1 = self.raw_vargam1 + grg.vargam1
        else:
            self._gam0 = self.raw_gam0
            self._gam1 = self.raw_gam1
            self._comp_vargam0 = None
            self._comp_vargam1 = None

        return self._gam0, self._gam1, self.vargam0, self.vargam1

    def _calculate_xi_from_pairs(self, pairs, corr_only):
        super()._calculate_xi_from_pairs(pairs, corr_only)
        if self._grg is not None:
            # If grg has npatch2 = 1, adjust pairs appropriately
            if self._grg.npatch2 == 1 and not all([p[1] == 0 for p in pairs]):
                pairs = [(i,0,k,w) for i,j,k,w in pairs if i == j]
            pairs = self._grg._keep_ok(pairs)
            self._grg._calculate_xi_from_pairs(pairs, corr_only=True)
            self._gam0 = self.raw_gam0 - self._grg.gam0
            self._gam1 = self.raw_gam1 - self._grg.gam1

    def write(self, file_name, *, file_type=None, precision=None, write_patch_results=False,
              write_cov=False):
        super().write(file_name, file_type=file_type, precision=precision,
                      write_patch_results=write_patch_results, write_cov=write_cov)

    write.__doc__ = Corr3.write.__doc__.format(
        r"""
        gam0r           The real part of the estimator of :math:`\Gamma_0`
        gam0i           The imaginary part of the estimator of :math:`\Gamma_0`
        gam1r           The real part of the estimator of :math:`\Gamma_1`
        gam1i           The imaginary part of the estimator of :math:`\Gamma_1`
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
        self._comp_vargam0 = data['sigma_gam0'].reshape(s)**2
        self._comp_vargam1 = data['sigma_gam1'].reshape(s)**2
        self._varzeta = [self._comp_vargam0, self._comp_vargam1]

class GGNCorrelation(Corr3):
    r"""This class handles the calculation and storage of a 3-point shear-shear-count correlation
    function.

    With this class, points 1 and 2 of the triangle (i.e. the vertces opposite d2,d3) are the ones
    with the shear values.  Use `NGGCorrelation` and `GNGCorrelation` for classes with the shears
    in the other positions.

    For the shear projection, we follow the lead of the 3-point shear-shear-shear correlation
    functions (see `GGGCorrelation` for details), which involves projecting the shear values
    at each vertex relative to the direction to the triangle's centroid.  Furthermore, the
    GGG correlations have 4 relevant complex values for each triangle:

    .. math::

        \Gamma_0 &= \langle \gamma(\mathbf{x1}) \gamma(\mathbf{x2}) \gamma(\mathbf{x3}) \rangle \\
        \Gamma_1 &= \langle \gamma(\mathbf{x1})^* \gamma(\mathbf{x2}) \gamma(\mathbf{x3}) \rangle \\
        \Gamma_2 &= \langle \gamma(\mathbf{x1}) \gamma(\mathbf{x2})^* \gamma(\mathbf{x3}) \rangle \\
        \Gamma_3 &= \langle \gamma(\mathbf{x1}) \gamma(\mathbf{x2}) \gamma(\mathbf{x3}^*) \rangle \\

    With only a count at vertex 3, :math:`\Gamma_0 = \Gamma_3` and :math:`\Gamma_1 = \Gamma_2^*`.
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

        >>> ggn = treecorr.GGNCorrelation(config)
        >>> ggn.process(cat1, cat2)         # Compute the cross-correlation of two fields.
        >>> # ggn.process(cat1, cat2, cat3) # ... or of three fields.
        >>> ggn.write(file_name)            # Write out to a file.
        >>> ggr.process(cat1, rand)         # Compute the random cross-correlation.
        >>> ggn.calculateZeta(ggr=ggr)      # Calculate zeta using randoms.
        >>> gam0 = ggn.gam0, etc.           # Access gamma values directly.
        >>> gam0r = ggn.gam0r               # Or access real and imaginary parts separately.
        >>> gam0i = ggn.gam0i

    See also: `NGGCorrelation`, `GNGCorrelation`, `NNGCorrelation`, `GGGCorrelation`,
    `NGCorrelation`.

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
    _cls = 'GGNCorrelation'
    _letter1 = 'G'
    _letter2 = 'G'
    _letter3 = 'N'
    _letters = 'GGN'
    _builder = _treecorr.GGNCorr
    _calculateVar1 = staticmethod(calculateVarG)
    _calculateVar2 = staticmethod(calculateVarG)
    _calculateVar3 = lambda *args, **kargs: None
    _sig1 = 'sig_sn (per component)'
    _sig2 = 'sig_sn (per component)'
    _sig3 = None

    def __init__(self, config=None, *, logger=None, **kwargs):
        super().__init__(config, logger=logger, **kwargs)

        shape = self.data_shape
        self._z[0:4] = [np.zeros(shape, dtype=float) for _ in range(4)]
        self._ggr = None
        self._gam0 = None
        self._gam1 = None
        self._comp_vargam0 = None
        self._comp_vargam1 = None
        self.logger.debug('Finished building GGNCorr')

    @property
    def raw_gam0(self):
        return self._z[0] + 1j * self._z[1]

    @property
    def raw_gam1(self):
        return self._z[2] + 1j * self._z[3]

    @property
    def gam0(self):
        if self._gam0 is None:
            return self._z[0] + 1j * self._z[1]
        else:
            return self._gam0

    @property
    def gam1(self):
        if self._gam1 is None:
            return self._z[2] + 1j * self._z[3]
        else:
            return self._gam1

    @property
    def gam2(self):
        return np.conjugate(self.gam1)

    @property
    def gam3(self):
        return self.gam0

    @property
    def gam0r(self):
        if self._gam0 is None:
            return self._z[0]
        else:
            return np.real(self._gam0)

    @property
    def gam0i(self):
        if self._gam0 is None:
            return self._z[1]
        else:
            return np.imag(self._gam0)

    @property
    def gam1r(self):
        if self._gam1 is None:
            return self._z[2]
        else:
            return np.real(self._gam1)

    @property
    def gam1i(self):
        if self._gam1 is None:
            return self._z[3]
        else:
            return np.imag(self._gam1)

    @property
    def gam2r(self):
        return self.gam1r

    @property
    def gam2i(self):
        return -self.gam1i

    @property
    def gam3r(self):
        return self.gam0r

    @property
    def gam3i(self):
        return self.gam0i

    def copy(self):
        ret = super().copy()
        # True is possible during read before we finish reading in these attributes.
        if self._ggr is not None and self._ggr is not True:
            ret._ggr = self._ggr.copy()
        return ret

    def finalize(self, varg1, varg2):
        """Finalize the calculation of the correlation function.

        Parameters:
            varg1 (float):  The variance per component of the first shear field.
            varg2 (float):  The variance per component of the second shear field.
        """
        self._finalize()
        self._var_num = 2 * varg1 * varg2
        if self.bin_type in ['LogSAS', 'LogMultipole']:
            self._var_num *= 2

    @property
    def raw_vargam0(self):
        if self._varzeta is None:
            self._calculate_varzeta(2)
        return self._varzeta[0]

    @property
    def vargam0(self):
        if self._comp_vargam0 is None:
            return self.raw_vargam0
        else:
            return self._comp_vargam0

    @property
    def raw_vargam1(self):
        if self._varzeta is None:
            self._calculate_varzeta(2)
        return self._varzeta[1]

    @property
    def vargam1(self):
        if self._comp_vargam1 is None:
            return self.raw_vargam1
        else:
            return self._comp_vargam1

    @property
    def vargam2(self):
        return self.vargam1

    @property
    def vargam3(self):
        return self.vargam0

    def _clear(self):
        super()._clear()
        self._ggr = None
        self._gam0 = None
        self._gam1 = None
        self._comp_vargam0 = None
        self._comp_vargam1 = None

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

    def calculateGam(self, *, ggr=None):
        r"""Calculate the correlation function possibly given another correlation function
        that uses random points for the foreground objects.

        - If ggr is None, the simple correlation functions (self.gam0, self.gam1) are returned.
        - If ggr is not None, then a compensated calculation is done:
          :math:`\Gamma_i = (GGD - GGR)`, where GGD represents the correlation of the gamma
          field with the data points and GGR represents the correlation with random points.

        After calling this function, the attributes ``gam0``, ``gam1``, ``vargam0``,
        ``vargam1``, and ``cov`` will correspond to the compensated values (if ggr is provided).
        The raw, uncompensated values are available as ``raw_gam0``, ``raw_gam1``,
        ``raw_vargam0``, and ``raw_vargam1``.

        Parameters:
            ggr (GGNCorrelation): The cross-correlation using random locations as the lenses (GGR),
                                  if desired.  (default: None)

        Returns:
            Tuple containing
                - gam0 = array of :math:`\Gamma_0`
                - gam1 = array of :math:`\Gamma_1`
                - vargam0 = array of variance estimates of :math:`\Gamma_0`
                - vargam1 = array of variance estimates of :math:`\Gamma_1`
        """
        if ggr is not None:
            if self.bin_type == 'LogMultipole':
                raise TypeError("calculateGam is not valid for LogMultipole binning")

            self._gam0 = self.raw_gam0 - ggr.gam0
            self._gam1 = self.raw_gam1 - ggr.gam1
            self._ggr = ggr

            if (ggr.npatch3 not in (1,self.npatch3) or ggr.npatch1 != self.npatch1
                    or ggr.npatch2 != self.npatch2):
                raise RuntimeError("GGR must be run with the same patches as GGD")

            if len(self.results) > 0:
                # If there are any ggr patch pairs that aren't in results (e.g. due to different
                # edge effects among the various pairs in consideration), then we need to add
                # some dummy results to make sure all the right pairs are computed when we make
                # the vectors for the covariance matrix.
                template = next(iter(self.results.values()))  # Just need something to copy.
                for ijk in ggr.results:
                    if ijk in self.results: continue
                    new_cij = template.copy()
                    for i in range(4):
                        new_cij._z[i][:] = 0
                    new_cij.weight[:] = 0
                    self.results[ijk] = new_cij
                    self.__dict__.pop('_ok',None)

                self._cov = self.estimate_cov(self.var_method)
                self._comp_vargam0 = np.zeros(self.data_shape, dtype=float)
                self._comp_vargam1 = np.zeros(self.data_shape, dtype=float)
                n1 = self._comp_vargam0.ravel().size
                self._comp_vargam0.ravel()[:] = self.cov_diag[0:n1]
                self._comp_vargam1.ravel()[:] = self.cov_diag[n1:]
            else:
                self._comp_vargam0 = self.raw_vargam0 + ggr.vargam0
                self._comp_vargam1 = self.raw_vargam1 + ggr.vargam1
        else:
            self._gam0 = self.raw_gam0
            self._gam1 = self.raw_gam1
            self._comp_vargam0 = None
            self._comp_vargam1 = None

        return self._gam0, self._gam1, self.vargam0, self.vargam1

    def _calculate_xi_from_pairs(self, pairs, corr_only):
        super()._calculate_xi_from_pairs(pairs, corr_only)
        if self._ggr is not None:
            # If ggr has npatch3 = 1, adjust pairs appropriately
            if self._ggr.npatch3 == 1 and not all([p[2] == 0 for p in pairs]):
                pairs = [(i,j,0,w) for i,j,k,w in pairs if i == k]
            pairs = self._ggr._keep_ok(pairs)
            self._ggr._calculate_xi_from_pairs(pairs, corr_only=True)
            self._gam0 = self.raw_gam0 - self._ggr.gam0
            self._gam1 = self.raw_gam1 - self._ggr.gam1

    def write(self, file_name, *, file_type=None, precision=None, write_patch_results=False,
              write_cov=False):
        super().write(file_name, file_type=file_type, precision=precision,
                      write_patch_results=write_patch_results, write_cov=write_cov)

    write.__doc__ = Corr3.write.__doc__.format(
        r"""
        gam0r           The real part of the estimator of :math:`\Gamma_0`
        gam0i           The imaginary part of the estimator of :math:`\Gamma_0`
        gam1r           The real part of the estimator of :math:`\Gamma_1`
        gam1i           The imaginary part of the estimator of :math:`\Gamma_1`
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
        self._comp_vargam0 = data['sigma_gam0'].reshape(s)**2
        self._comp_vargam1 = data['sigma_gam1'].reshape(s)**2
        self._varzeta = [self._comp_vargam0, self._comp_vargam1]
