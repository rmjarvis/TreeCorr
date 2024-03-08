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
.. module:: ngcorrelation
"""

import numpy as np

from . import _treecorr
from .catalog import calculateVarZ
from .corr2base import Corr2
from .util import make_writer
from .config import make_minimal_config


class BaseNZCorrelation(Corr2):
    """This class is a base class for all the N?Correlation classes, where ? is one of the
    complex fields of varying spin.

    A lot of the implementation is shared among those types, so whenever possible the shared
    implementation is done in this class.
    """
    _sig1 = None
    _sig2 = 'sig_sn (per component)'

    def __init__(self, config=None, *, logger=None, **kwargs):
        super().__init__(config, logger=logger, **kwargs)

        self._xi1 = np.zeros_like(self.rnom, dtype=float)
        self._xi2 = np.zeros_like(self.rnom, dtype=float)
        self._xi3 = self._xi4 = np.array([])
        self.xi = self.raw_xi
        self.xi_im = self.raw_xi_im
        self._rz = None
        self._raw_varxi = None
        self._varxi = None
        self.logger.debug('Finished building %s', self._cls)

    @property
    def raw_xi(self):
        return self._xi1

    @property
    def raw_xi_im(self):
        return self._xi2

    def copy(self):
        """Make a copy"""
        ret = super().copy()
        if self.xi is self.raw_xi:
            ret.xi = ret.raw_xi
            ret.xi_im = ret.raw_xi_im
        if self._rz is not None:
            ret._rz = self._rz.copy()
        return ret

    def finalize(self, varz):
        self._finalize()
        self._var_num = varz

        self.xi = self.raw_xi
        self.xi_im = self.raw_xi_im

    @property
    def raw_varxi(self):
        if self._raw_varxi is None:
            self._raw_varxi = np.zeros_like(self.rnom, dtype=float)
            if self._var_num != 0:
                self._raw_varxi.ravel()[:] = self.cov_diag
        return self._raw_varxi

    @property
    def varxi(self):
        if self._varxi is None:
            self._varxi = self.raw_varxi
        return self._varxi

    def _clear(self):
        """Clear the data vectors
        """
        super()._clear()
        self.xi = self.raw_xi
        self.xi_im = self.raw_xi_im
        self._rz = None
        self._raw_varxi = None
        self._varxi = None

    def _sum(self, others):
        # Equivalent to the operation of:
        #     self._clear()
        #     for other in others:
        #         self += other
        # but no sanity checks and use numpy.sum for faster calculation.
        np.sum([c._xi1 for c in others], axis=0, out=self._xi1)
        np.sum([c._xi2 for c in others], axis=0, out=self._xi2)
        np.sum([c.meanr for c in others], axis=0, out=self.meanr)
        np.sum([c.meanlogr for c in others], axis=0, out=self.meanlogr)
        np.sum([c.weight for c in others], axis=0, out=self.weight)
        np.sum([c.npairs for c in others], axis=0, out=self.npairs)
        self.xi = self.raw_xi
        self.xi_im = self.raw_xi_im
        self._raw_varxi = None
        self._varxi = None
        self._cov = None

    def calculateXi(self, rz=None):
        if rz is not None:
            self.xi = self.raw_xi - rz.xi
            self.xi_im = self.raw_xi_im - rz.xi_im
            self._rz = rz

            if rz.npatch1 not in (1,self.npatch1) or rz.npatch2 != self.npatch2:
                raise RuntimeError(f"R{self._letter2} must be run with the same patches as D{self._letter2}")

            if len(self.results) > 0:
                # If there are any rz patch pairs that aren't in results (e.g. due to different
                # edge effects among the various pairs in consideration), then we need to add
                # some dummy results to make sure all the right pairs are computed when we make
                # the vectors for the covariance matrix.
                template = next(iter(self.results.values()))  # Just need something to copy.
                for ij in rz.results:
                    if ij in self.results: continue
                    new_cij = template.copy()
                    new_cij.xi.ravel()[:] = 0
                    new_cij.weight.ravel()[:] = 0
                    self.results[ij] = new_cij

                self._cov = self.estimate_cov(self.var_method)
                self._varxi = np.zeros_like(self.rnom, dtype=float)
                self._varxi.ravel()[:] = self.cov_diag
            else:
                self._varxi = self.raw_varxi + rz.varxi
        else:
            self.xi = self.raw_xi
            self.xi_im = self.raw_xi_im
            self._varxi = self.raw_varxi

        return self.xi, self.xi_im, self.varxi

    def _calculate_xi_from_pairs(self, pairs):
        self._sum([self.results[ij] for ij in pairs])
        self._finalize()
        if self._rz is not None:
            # If rz has npatch1 = 1, adjust pairs appropriately
            if self._rz.npatch1 == 1 and not all([p[0] == 0 for p in pairs]):
                pairs = [(0,ij[1]) for ij in pairs if ij[0] == ij[1]]
            # Make sure all ij are in the rz results (some might be missing, which is ok)
            pairs = [ij for ij in pairs if self._rz._ok[ij[0],ij[1]]]
            self._rz._calculate_xi_from_pairs(pairs)
            self.xi -= self._rz.xi

    def write(self, file_name, rz=None, file_type=None, precision=None,
              write_patch_results=False, write_cov=False):
        self.logger.info(f'Writing {self._letters} correlations to %s',file_name)
        BaseNZCorrelation.calculateXi(self, rz)
        precision = self.config.get('precision', 4) if precision is None else precision
        with make_writer(file_name, precision, file_type, self.logger) as writer:
            self._write(writer, None, write_patch_results, write_cov=write_cov)

    @property
    def _write_col_names(self):
        return ['r_nom','meanr','meanlogr',self._zreal,self._zimag,'sigma','weight','npairs']

    @property
    def _write_data(self):
        data = [ self.rnom, self.meanr, self.meanlogr,
                 self.xi, self.xi_im, np.sqrt(self.varxi), self.weight, self.npairs ]
        data = [ col.flatten() for col in data ]
        return data

    def _read_from_data(self, data, params):
        super()._read_from_data(data, params)
        s = self.logr.shape
        self._xi1 = data[self._zreal].reshape(s)
        self._xi2 = data[self._zimag].reshape(s)
        self._varxi = data['sigma'].reshape(s)**2
        self.xi = self.raw_xi
        self.xi_im = self.raw_xi_im
        self._raw_varxi = self._varxi

class NZCorrelation(BaseNZCorrelation):
    r"""This class handles the calculation and storage of a 2-point count-complex correlation
    function, where the complex field is taken to have spin-0 rotational properties.  If the
    spin-0 field is real, you should instead use `NKCorrelation` as it will be faster.
    This class is intended for correlations of a scalar field with a complex values that
    don't change with orientation.

    Ojects of this class holds the following attributes:

    Attributes:
        nbins:     The number of bins in logr
        bin_size:  The size of the bins in logr
        min_sep:   The minimum separation being considered
        max_sep:   The maximum separation being considered

    In addition, the following attributes are numpy arrays of length (nbins):

    Attributes:
        logr:       The nominal center of the bin in log(r) (the natural logarithm of r).
        rnom:       The nominal center of the bin converted to regular distance.
                    i.e. r = exp(logr).
        meanr:      The (weighted) mean value of r for the pairs in each bin.
                    If there are no pairs in a bin, then exp(logr) will be used instead.
        meanlogr:   The (weighted) mean value of log(r) for the pairs in each bin.
                    If there are no pairs in a bin, then logr will be used instead.
        xi:         The correlation function, :math:`\xi(r) = \langle z\rangle`.
        xi_im:      The imaginary part of :math:`\xi(r)`.
        varxi:      An estimate of the variance of :math:`\xi`
        weight:     The total weight in each bin.
        npairs:     The number of pairs going into each bin (including pairs where one or
                    both objects have w=0).
        cov:        An estimate of the full covariance matrix.
        raw_xi:     The raw value of xi, uncorrected by an RZ calculation. cf. `calculateXi`
        raw_xi_im:  The raw value of xi_im, uncorrected by an RZ calculation. cf. `calculateXi`
        raw_varxi:  The raw value of varxi, uncorrected by an RZ calculation. cf. `calculateXi`

    .. note::

        The default method for estimating the variance and covariance attributes (``varxi``,
        and ``cov``) is 'shot', which only includes the shape noise propagated into
        the final correlation.  This does not include sample variance, so it is always an
        underestimate of the actual variance.  To get better estimates, you need to set
        ``var_method`` to something else and use patches in the input catalog(s).
        cf. `Covariance Estimates`.

    If ``sep_units`` are given (either in the config dict or as a named kwarg) then the distances
    will all be in these units.

    .. note::

        If you separate out the steps of the `Corr2.process` command and use `Corr2.process_cross`,
        then the units will not be applied to ``meanr`` or ``meanlogr`` until the `finalize`
        function is called.

    The typical usage pattern is as follows:

        >>> nz = treecorr.NZCorrelation(config)
        >>> nz.process(cat1,cat2)   # Compute the cross-correlation.
        >>> nz.write(file_name)     # Write out to a file.
        >>> xi = nz.xi              # Or access the correlation function directly.

    Parameters:
        config (dict):  A configuration dict that can be used to pass in kwargs if desired.
                        This dict is allowed to have addition entries besides those listed
                        in `Corr2`, which are ignored here. (default: None)
        logger:         If desired, a logger object for logging. (default: None, in which case
                        one will be built according to the config dict's verbose level.)

    Keyword Arguments:
        **kwargs:       See the documentation for `Corr2` for the list of allowed keyword
                        arguments, which may be passed either directly or in the config dict.
    """
    _cls = 'NZCorrelation'
    _letter1 = 'N'
    _letter2 = 'Z'
    _letters = 'NZ'
    _builder = _treecorr.NZCorr
    _calculateVar1 = lambda *args, **kwargs: None
    _calculateVar2 = staticmethod(calculateVarZ)
    _zreal = 'z_real'
    _zimag = 'z_imag'

    def __init__(self, config=None, *, logger=None, **kwargs):
        """Initialize `NZCorrelation`.  See class doc for details.
        """
        super().__init__(config, logger=logger, **kwargs)

    def finalize(self, varz):
        """Finalize the calculation of the correlation function.

        The `Corr2.process_cross` command accumulates values in each bin, so it can be called
        multiple times if appropriate.  Afterwards, this command finishes the calculation
        by dividing each column by the total weight.

        Parameters:
            varz (float):   The variance per component of the complex field.
        """
        super().finalize(varz)

    def calculateXi(self, *, rz=None):
        r"""Calculate the correlation function possibly given another correlation function
        that uses random points for the foreground objects.

        - If rz is None, the simple correlation function :math:`\langle z\rangle` is
          returned.
        - If rz is not None, then a compensated calculation is done:
          :math:`\langle z\rangle = (DZ - RZ)`, where DZ represents the mean field value
          around the data points and RZ represents the mean value around random points.

        After calling this function, the attributes ``xi``, ``xi_im``, ``varxi``, and ``cov`` will
        correspond to the compensated values (if rz is provided).  The raw, uncompensated values
        are available as ``rawxi``, ``raw_xi_im``, and ``raw_varxi``.

        Parameters:
            rz (NZCorrelation): The cross-correlation using random locations as the lenses
                                (RZ), if desired.  (default: None)

        Returns:
            Tuple containing

                - xi = array of the real part of :math:`\xi(R)`
                - xi_im = array of the imaginary part of :math:`\xi(R)`
                - varxi = array of the variance estimates of the above values
        """
        return super().calculateXi(rz=rz)

    def write(self, file_name, *, rz=None, file_type=None, precision=None,
              write_patch_results=False, write_cov=False):
        r"""Write the correlation function to the file, file_name.

        - If rz is None, the simple correlation function :math:`\langle z\rangle` is used.
        - If rz is not None, then a compensated calculation is done:
          :math:`\langle z\rangle = (DZ - RZ)`, where DZ represents the mean field value
          around the data points and RZ represents the mean value around random points.

        The output file will include the following columns:

        ==========      =============================================================
        Column          Description
        ==========      =============================================================
        r_nom           The nominal center of the bin in r
        meanr           The mean value :math:`\langle r \rangle` of pairs that fell
                        into each bin
        meanlogr        The mean value :math:`\langle \log(r) \rangle` of pairs that
                        fell into each bin
        z_real          The mean real component, :math:`\langle real(z) \rangle(r)`
        z_imag          The mean imaginary component, :math:`\langle imag(z) \rangle(r)`.
        sigma           The sqrt of the variance estimate of either of these
        weight          The total weight contributing to each bin
        npairs          The total number of pairs in each bin
        ==========      =============================================================

        If ``sep_units`` was given at construction, then the distances will all be in these units.
        Otherwise, they will be in either the same units as x,y,z (for flat or 3d coordinates) or
        radians (for spherical coordinates).

        Parameters:
            file_name (str):    The name of the file to write to.
            rz (NZCorrelation): The cross-correlation using random locations as the lenses
                                (RZ), if desired.  (default: None)
            file_type (str):    The type of file to write ('ASCII' or 'FITS').  (default: determine
                                the type automatically from the extension of file_name.)
            precision (int):    For ASCII output catalogs, the desired precision. (default: 4;
                                this value can also be given in the constructor in the config dict.)
            write_patch_results (bool): Whether to write the patch-based results as well.
                                        (default: False)
            write_cov (bool):   Whether to write the covariance matrix as well. (default: False)
        """
        super().write(file_name, rz, file_type, precision, write_patch_results, write_cov)
