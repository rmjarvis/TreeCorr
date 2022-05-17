# Copyright (c) 2003-2019 by Mike Jarvis
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

from . import _lib, _ffi
from .catalog import calculateVarG
from .binnedcorr2 import BinnedCorr2
from .util import double_ptr as dp
from .util import make_writer, make_reader
from .util import depr_pos_kwargs


class NGCorrelation(BinnedCorr2):
    r"""This class handles the calculation and storage of a 2-point count-shear correlation
    function.  This is the tangential shear profile around lenses, commonly referred to as
    galaxy-galaxy lensing.

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
        xi:         The correlation function, :math:`\xi(r) = \langle \gamma_T\rangle`.
        xi_im:      The imaginary part of :math:`\xi(r)`.
        varxi:      An estimate of the variance of :math:`\xi`
        weight:     The total weight in each bin.
        npairs:     The number of pairs going into each bin (including pairs where one or
                    both objects have w=0).
        cov:        An estimate of the full covariance matrix.
        raw_xi:     The raw value of xi, uncorrected by an RG calculation. cf. `calculateXi`
        raw_xi_im:  The raw value of xi_im, uncorrected by an RG calculation. cf. `calculateXi`
        raw_varxi:  The raw value of varxi, uncorrected by an RG calculation. cf. `calculateXi`

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

        If you separate out the steps of the `process` command and use `process_cross`,
        then the units will not be applied to ``meanr`` or ``meanlogr`` until the `finalize`
        function is called.

    The typical usage pattern is as follows:

        >>> ng = treecorr.NGCorrelation(config)
        >>> ng.process(cat1,cat2)   # Compute the cross-correlation.
        >>> ng.write(file_name)     # Write out to a file.
        >>> xi = gg.xi              # Or access the correlation function directly.

    Parameters:
        config (dict):  A configuration dict that can be used to pass in kwargs if desired.
                        This dict is allowed to have addition entries besides those listed
                        in `BinnedCorr2`, which are ignored here. (default: None)
        logger:         If desired, a logger object for logging. (default: None, in which case
                        one will be built according to the config dict's verbose level.)

    Keyword Arguments:
        **kwargs:       See the documentation for `BinnedCorr2` for the list of allowed keyword
                        arguments, which may be passed either directly or in the config dict.
    """
    @depr_pos_kwargs
    def __init__(self, config=None, *, logger=None, **kwargs):
        """Initialize `NGCorrelation`.  See class doc for details.
        """
        BinnedCorr2.__init__(self, config, logger=logger, **kwargs)

        self._ro._d1 = 1  # NData
        self._ro._d2 = 3  # GData
        self.xi = np.zeros_like(self.rnom, dtype=float)
        self.xi_im = np.zeros_like(self.rnom, dtype=float)
        self.varxi = np.zeros_like(self.rnom, dtype=float)
        self.meanr = np.zeros_like(self.rnom, dtype=float)
        self.meanlogr = np.zeros_like(self.rnom, dtype=float)
        self.weight = np.zeros_like(self.rnom, dtype=float)
        self.npairs = np.zeros_like(self.rnom, dtype=float)
        self.raw_xi = self.xi
        self.raw_xi_im = self.xi_im
        self.raw_varxi = self.varxi
        self._rg = None
        self.logger.debug('Finished building NGCorr')

    @property
    def corr(self):
        if self._corr is None:
            self._corr = _lib.BuildCorr2(
                    self._d1, self._d2, self._bintype,
                    self._min_sep,self._max_sep,self._nbins,self._bin_size,self.b,
                    self.min_rpar, self.max_rpar, self.xperiod, self.yperiod, self.zperiod,
                    dp(self.raw_xi),dp(self.raw_xi_im), dp(None), dp(None),
                    dp(self.meanr),dp(self.meanlogr),dp(self.weight),dp(self.npairs))
        return self._corr

    def __del__(self):
        # Using memory allocated from the C layer means we have to explicitly deallocate it
        # rather than being able to rely on the Python memory manager.
        if self._corr is not None:
            if not _ffi._lock.locked(): # pragma: no branch
                _lib.DestroyCorr2(self.corr, self._d1, self._d2, self._bintype)

    def __eq__(self, other):
        """Return whether two `NGCorrelation` instances are equal"""
        return (isinstance(other, NGCorrelation) and
                self.nbins == other.nbins and
                self.bin_size == other.bin_size and
                self.min_sep == other.min_sep and
                self.max_sep == other.max_sep and
                self.sep_units == other.sep_units and
                self.coords == other.coords and
                self.bin_type == other.bin_type and
                self.bin_slop == other.bin_slop and
                self.min_rpar == other.min_rpar and
                self.max_rpar == other.max_rpar and
                self.xperiod == other.xperiod and
                self.yperiod == other.yperiod and
                self.zperiod == other.zperiod and
                np.array_equal(self.meanr, other.meanr) and
                np.array_equal(self.meanlogr, other.meanlogr) and
                np.array_equal(self.xi, other.xi) and
                np.array_equal(self.xi_im, other.xi_im) and
                np.array_equal(self.varxi, other.varxi) and
                np.array_equal(self.weight, other.weight) and
                np.array_equal(self.npairs, other.npairs))

    def copy(self):
        """Make a copy"""
        ret = NGCorrelation.__new__(NGCorrelation)
        for key, item in self.__dict__.items():
            if isinstance(item, np.ndarray):
                # Only items that might change need to by deep copied.
                ret.__dict__[key] = item.copy()
            else:
                # For everything else, shallow copy is fine.
                # In particular don't deep copy config or logger
                # Most of the rest are scalars, which copy fine this way.
                # And the read-only things are all in _ro.
                # The results dict is trickier.  We rely on it being copied in places, but we
                # never add more to it after the copy, so shallow copy is fine.
                ret.__dict__[key] = item
        ret._corr = None # We'll want to make a new one of these if we need it.
        if self.xi is self.raw_xi:
            ret.raw_xi = ret.xi
            ret.raw_xi_im = ret.xi_im
            ret.raw_varxi = ret.varxi
        else:
            ret.raw_xi = self.raw_xi.copy()
            ret.raw_xi_im = self.raw_xi_im.copy()
            ret.raw_varxi = self.raw_varxi.copy()
        if self._rg is not None:
            ret._rg = self._rg.copy()
        return ret

    def __repr__(self):
        return 'NGCorrelation(config=%r)'%self.config

    @depr_pos_kwargs
    def process_cross(self, cat1, cat2, *, metric=None, num_threads=None):
        """Process a single pair of catalogs, accumulating the cross-correlation.

        This accumulates the weighted sums into the bins, but does not finalize
        the calculation by dividing by the total weight at the end.  After
        calling this function as often as desired, the `finalize` command will
        finish the calculation.

        Parameters:
            cat1 (Catalog):     The first catalog to process
            cat2 (Catalog):     The second catalog to process
            metric (str):       Which metric to use.  See `Metrics` for details.
                                (default: 'Euclidean'; this value can also be given in the
                                constructor in the config dict.)
            num_threads (int):  How many OpenMP threads to use during the calculation.
                                (default: use the number of cpu cores; this value can also be given
                                in the constructor in the config dict.)
        """
        if cat1.name == '' and cat2.name == '':
            self.logger.info('Starting process NG cross-correlations')
        else:
            self.logger.info('Starting process NG cross-correlations for cats %s, %s.',
                             cat1.name, cat2.name)

        self._set_metric(metric, cat1.coords, cat2.coords)
        self._set_num_threads(num_threads)
        min_size, max_size = self._get_minmax_size()

        f1 = cat1.getNField(min_size=min_size, max_size=max_size,
                            split_method=self.split_method,
                            brute=self.brute is True or self.brute == 1,
                            min_top=self.min_top, max_top=self.max_top,
                            coords=self.coords)
        f2 = cat2.getGField(min_size=min_size, max_size=max_size,
                            split_method=self.split_method,
                            brute=self.brute is True or self.brute == 2,
                            min_top=self.min_top, max_top=self.max_top,
                            coords=self.coords)

        self.logger.info('Starting %d jobs.',f1.nTopLevelNodes)
        _lib.ProcessCross2(self.corr, f1.data, f2.data, self.output_dots,
                           f1._d, f2._d, self._coords, self._bintype, self._metric)

    @depr_pos_kwargs
    def process_pairwise(self, cat1, cat2, *, metric=None, num_threads=None):
        """Process a single pair of catalogs, accumulating the cross-correlation, only using
        the corresponding pairs of objects in each catalog.

        This accumulates the weighted sums into the bins, but does not finalize
        the calculation by dividing by the total weight at the end.  After
        calling this function as often as desired, the `finalize` command will
        finish the calculation.

        .. warning::

            .. deprecated:: 4.1

                This function is deprecated and slated to be removed.
                If you have a need for it, please open an issue to describe your use case.

        Parameters:
            cat1 (Catalog):     The first catalog to process
            cat2 (Catalog):     The second catalog to process
            metric (str):       Which metric to use.  See `Metrics` for details.
                                (default: 'Euclidean'; this value can also be given in the
                                constructor in the config dict.)
            num_threads (int):  How many OpenMP threads to use during the calculation.
                                (default: use the number of cpu cores; this value can also be given
                                in the constructor in the config dict.)
        """
        import warnings
        warnings.warn("The process_pairwise function is slated to be removed in a future version. "+
                      "If you are actually using this function usefully, please "+
                      "open an issue to describe your use case.", FutureWarning)
        if cat1.name == '' and cat2.name == '':
            self.logger.info('Starting process NG pairwise-correlations')
        else:
            self.logger.info('Starting process NG pairwise-correlations for cats %s, %s.',
                             cat1.name, cat2.name)

        self._set_metric(metric, cat1.coords, cat2.coords)
        self._set_num_threads(num_threads)

        f1 = cat1.getNSimpleField()
        f2 = cat2.getGSimpleField()

        _lib.ProcessPair(self.corr, f1.data, f2.data, self.output_dots,
                         f1._d, f2._d, self._coords, self._bintype, self._metric)

    def _finalize(self):
        mask1 = self.weight != 0
        mask2 = self.weight == 0

        self.raw_xi[mask1] /= self.weight[mask1]
        self.raw_xi_im[mask1] /= self.weight[mask1]
        self.meanr[mask1] /= self.weight[mask1]
        self.meanlogr[mask1] /= self.weight[mask1]

        # Update the units of meanr, meanlogr
        self._apply_units(mask1)

        # Use meanr, meanlogr when available, but set to nominal when no pairs in bin.
        self.meanr[mask2] = self.rnom[mask2]
        self.meanlogr[mask2] = self.logr[mask2]

    def finalize(self, varg):
        """Finalize the calculation of the correlation function.

        The `process_cross` command accumulates values in each bin, so it can be called
        multiple times if appropriate.  Afterwards, this command finishes the calculation
        by dividing each column by the total weight.

        Parameters:
            varg (float):   The shear variance per component for the second field.
        """
        self._finalize()
        self._var_num = varg
        self.cov = self.estimate_cov(self.var_method)
        self.raw_varxi.ravel()[:] = self.cov.diagonal()

        self.xi = self.raw_xi
        self.xi_im = self.raw_xi_im
        self.varxi = self.raw_varxi

    def _clear(self):
        """Clear the data vectors
        """
        self.raw_xi.ravel()[:] = 0
        self.raw_xi_im.ravel()[:] = 0
        self.raw_varxi.ravel()[:] = 0
        self.meanr.ravel()[:] = 0
        self.meanlogr.ravel()[:] = 0
        self.weight.ravel()[:] = 0
        self.npairs.ravel()[:] = 0
        self.xi = self.raw_xi
        self.xi_im = self.raw_xi_im
        self.varxi = self.raw_varxi

    def __iadd__(self, other):
        """Add a second `NGCorrelation`'s data to this one.

        .. note::

            For this to make sense, both `NGCorrelation` objects should not have had `finalize`
            called yet.  Then, after adding them together, you should call `finalize` on the sum.
        """
        if not isinstance(other, NGCorrelation):
            raise TypeError("Can only add another NGCorrelation object")
        if not (self._nbins == other._nbins and
                self.min_sep == other.min_sep and
                self.max_sep == other.max_sep):
            raise ValueError("NGCorrelation to be added is not compatible with this one.")

        self._set_metric(other.metric, other.coords, other.coords)
        self.raw_xi.ravel()[:] += other.raw_xi.ravel()[:]
        self.raw_xi_im.ravel()[:] += other.raw_xi_im.ravel()[:]
        self.meanr.ravel()[:] += other.meanr.ravel()[:]
        self.meanlogr.ravel()[:] += other.meanlogr.ravel()[:]
        self.weight.ravel()[:] += other.weight.ravel()[:]
        self.npairs.ravel()[:] += other.npairs.ravel()[:]
        return self

    def _sum(self, others):
        # Equivalent to the operation of:
        #     self._clear()
        #     for other in others:
        #         self += other
        # but no sanity checks and use numpy.sum for faster calculation.
        np.sum([c.raw_xi for c in others], axis=0, out=self.raw_xi)
        np.sum([c.raw_xi_im for c in others], axis=0, out=self.raw_xi_im)
        np.sum([c.meanr for c in others], axis=0, out=self.meanr)
        np.sum([c.meanlogr for c in others], axis=0, out=self.meanlogr)
        np.sum([c.weight for c in others], axis=0, out=self.weight)
        np.sum([c.npairs for c in others], axis=0, out=self.npairs)
        self.xi = self.raw_xi
        self.xi_im = self.raw_xi_im
        self.varxi = self.raw_varxi

    @depr_pos_kwargs
    def process(self, cat1, cat2, *, metric=None, num_threads=None, comm=None, low_mem=False,
                initialize=True, finalize=True):
        """Compute the correlation function.

        Both arguments may be lists, in which case all items in the list are used
        for that element of the correlation.

        Parameters:
            cat1 (Catalog):     A catalog or list of catalogs for the N field.
            cat2 (Catalog):     A catalog or list of catalogs for the G field.
            metric (str):       Which metric to use.  See `Metrics` for details.
                                (default: 'Euclidean'; this value can also be given in the
                                constructor in the config dict.)
            num_threads (int):  How many OpenMP threads to use during the calculation.
                                (default: use the number of cpu cores; this value can also be given
                                in the constructor in the config dict.)
            comm (mpi4py.Comm): If running MPI, an mpi4py Comm object to communicate between
                                processes.  If used, the rank=0 process will have the final
                                computation. This only works if using patches. (default: None)
            low_mem (bool):     Whether to sacrifice a little speed to try to reduce memory usage.
                                This only works if using patches. (default: False)
            initialize (bool):  Whether to begin the calculation with a call to
                                `BinnedCorr2.clear`.  (default: True)
            finalize (bool):    Whether to complete the calculation with a call to `finalize`.
                                (default: True)
        """
        import math
        if initialize:
            self.clear()
            self._rg = None

        if not isinstance(cat1,list):
            cat1 = cat1.get_patches(low_mem=low_mem)
        if not isinstance(cat2,list):
            cat2 = cat2.get_patches(low_mem=low_mem)

        self._process_all_cross(cat1, cat2, metric, num_threads, comm, low_mem)

        if finalize:
            varg = calculateVarG(cat2, low_mem=low_mem)
            self.logger.info("varg = %f: sig_sn (per component) = %f",varg,math.sqrt(varg))
            self.finalize(varg)

    @depr_pos_kwargs
    def calculateXi(self, *, rg=None):
        r"""Calculate the correlation function possibly given another correlation function
        that uses random points for the foreground objects.

        - If rg is None, the simple correlation function :math:`\langle \gamma_T\rangle` is
          returned.
        - If rg is not None, then a compensated calculation is done:
          :math:`\langle \gamma_T\rangle = (DG - RG)`, where DG represents the mean shear
          around the lenses and RG represents the mean shear around random points.

        After calling this function, the attributes ``xi``, ``xi_im``, ``varxi``, and ``cov`` will
        correspond to the compensated values (if rg is provided).  The raw, uncompensated values
        are available as ``rawxi``, ``raw_xi_im``, and ``raw_varxi``.

        Parameters:
            rg (NGCorrelation): The cross-correlation using random locations as the lenses
                                (RG), if desired.  (default: None)

        Returns:
            Tuple containing

                - xi = array of the real part of :math:`\xi(R)`
                - xi_im = array of the imaginary part of :math:`\xi(R)`
                - varxi = array of the variance estimates of the above values
        """
        if rg is not None:
            self.xi = self.raw_xi - rg.xi
            self.xi_im = self.raw_xi_im - rg.xi_im
            self._rg = rg

            if rg.npatch1 not in (1,self.npatch1) or rg.npatch2 != self.npatch2:
                raise RuntimeError("RG must be run with the same patches as DG")

            if len(self.results) > 0:
                # If there are any rg patch pairs that aren't in results (e.g. due to different
                # edge effects among the various pairs in consideration), then we need to add
                # some dummy results to make sure all the right pairs are computed when we make
                # the vectors for the covariance matrix.
                template = next(iter(self.results.values()))  # Just need something to copy.
                for ij in rg.results:
                    if ij in self.results: continue
                    new_cij = template.copy()
                    new_cij.xi.ravel()[:] = 0
                    new_cij.weight.ravel()[:] = 0
                    self.results[ij] = new_cij

                self.cov = self.estimate_cov(self.var_method)
                self.varxi.ravel()[:] = self.cov.diagonal()
            else:
                self.varxi = self.raw_varxi + rg.varxi
        else:
            self.xi = self.raw_xi
            self.xi_im = self.raw_xi_im
            self.varxi = self.raw_varxi

        return self.xi, self.xi_im, self.varxi

    def _calculate_xi_from_pairs(self, pairs):
        self._sum([self.results[ij] for ij in pairs])
        self._finalize()
        if self._rg is not None:
            # If rg has npatch1 = 1, adjust pairs appropriately
            if self._rg.npatch1 == 1:
                pairs = [(0,ij[1]) for ij in pairs if ij[0] == ij[1]]
            # Make sure all ij are in the rg results (some might be missing, which is ok)
            pairs = [ij for ij in pairs if self._rg._ok[ij[0],ij[1]]]
            self._rg._calculate_xi_from_pairs(pairs)
            self.xi -= self._rg.xi

    @depr_pos_kwargs
    def write(self, file_name, *, rg=None, file_type=None, precision=None,
              write_patch_results=False):
        r"""Write the correlation function to the file, file_name.

        - If rg is None, the simple correlation function :math:`\langle \gamma_T\rangle` is used.
        - If rg is not None, then a compensated calculation is done:
          :math:`\langle \gamma_T\rangle = (DG - RG)`, where DG represents the mean shear
          around the lenses and RG represents the mean shear around random points.

        The output file will include the following columns:

        ==========      =============================================================
        Column          Description
        ==========      =============================================================
        r_nom           The nominal center of the bin in r
        meanr           The mean value :math:`\langle r \rangle` of pairs that fell
                        into each bin
        meanlogr        The mean value :math:`\langle \log(r) \rangle` of pairs that
                        fell into each bin
        gamT            The real part of the mean tangential shear,
                        :math:`\langle \gamma_T \rangle(r)`
        gamX            The imag part of the mean tangential shear,
                        :math:`\langle \gamma_\times \rangle(r)`
        sigma           The sqrt of the variance estimate of either of these
        weight          The total weight contributing to each bin
        npairs          The total number of pairs in each bin
        ==========      =============================================================

        If ``sep_units`` was given at construction, then the distances will all be in these units.
        Otherwise, they will be in either the same units as x,y,z (for flat or 3d coordinates) or
        radians (for spherical coordinates).

        Parameters:
            file_name (str):    The name of the file to write to.
            rg (NGCorrelation): The cross-correlation using random locations as the lenses
                                (RG), if desired.  (default: None)
            file_type (str):    The type of file to write ('ASCII' or 'FITS').  (default: determine
                                the type automatically from the extension of file_name.)
            precision (int):    For ASCII output catalogs, the desired precision. (default: 4;
                                this value can also be given in the constructor in the config dict.)
            write_patch_results (bool): Whether to write the patch-based results as well.
                                        (default: False)
        """
        self.logger.info('Writing NG correlations to %s',file_name)
        self.calculateXi(rg=rg)
        precision = self.config.get('precision', 4) if precision is None else precision
        name = 'main' if write_patch_results else None
        with make_writer(file_name, precision, file_type, self.logger) as writer:
            self._write(writer, name, write_patch_results)

    @property
    def _write_col_names(self):
        return ['r_nom','meanr','meanlogr','gamT','gamX','sigma','weight','npairs']

    @property
    def _write_data(self):
        data = [ self.rnom, self.meanr, self.meanlogr,
                 self.xi, self.xi_im, np.sqrt(self.varxi), self.weight, self.npairs ]
        data = [ col.flatten() for col in data ]
        return data

    @property
    def _write_params(self):
        return { 'coords' : self.coords, 'metric' : self.metric,
                 'sep_units' : self.sep_units, 'bin_type' : self.bin_type }

    @depr_pos_kwargs
    def read(self, file_name, *, file_type=None):
        """Read in values from a file.

        This should be a file that was written by TreeCorr, preferably a FITS file, so there
        is no loss of information.

        .. warning::

            The `NGCorrelation` object should be constructed with the same configuration
            parameters as the one being read.  e.g. the same min_sep, max_sep, etc.  This is not
            checked by the read function.

        Parameters:
            file_name (str):    The name of the file to read in.
            file_type (str):    The type of file ('ASCII' or 'FITS').  (default: determine the type
                                automatically from the extension of file_name.)
        """
        self.logger.info('Reading NG correlations from %s',file_name)
        with make_reader(file_name, file_type, self.logger) as reader:
            self._read(reader)

    def _read_from_data(self, data, params):
        s = self.logr.shape
        if 'R_nom' in data.dtype.names:  # pragma: no cover
            self._ro.rnom = data['R_nom'].reshape(s)
            self.meanr = data['meanR'].reshape(s)
            self.meanlogr = data['meanlogR'].reshape(s)
        else:
            self._ro.rnom = data['r_nom'].reshape(s)
            self.meanr = data['meanr'].reshape(s)
            self.meanlogr = data['meanlogr'].reshape(s)
        self.xi = data['gamT'].reshape(s)
        self.xi_im = data['gamX'].reshape(s)
        self.varxi = data['sigma'].reshape(s)**2
        self.weight = data['weight'].reshape(s)
        self.npairs = data['npairs'].reshape(s)
        self.coords = params['coords'].strip()
        self.metric = params['metric'].strip()
        self._ro.sep_units = params['sep_units'].strip()
        self._ro.bin_type = params['bin_type'].strip()
        self.raw_xi = self.xi
        self.raw_xi_im = self.xi_im
        self.raw_varxi = self.varxi
        self.npatch1 = params.get('npatch1', 1)
        self.npatch2 = params.get('npatch2', 1)

    @depr_pos_kwargs
    def calculateNMap(self, *, R=None, rg=None, m2_uform=None):
        r"""Calculate the aperture mass statistics from the correlation function.

        .. math::

            \langle N M_{ap} \rangle(R) &= \int_{0}^{rmax} \frac{r dr}{R^2}
            T_\times\left(\frac{r}{R}\right) \Re\xi(r) \\
            \langle N M_{\times} \rangle(R) &= \int_{0}^{rmax} \frac{r dr}{R^2}
            T_\times\left(\frac{r}{R}\right) \Im\xi(r)

        The ``m2_uform`` parameter sets which definition of the aperture mass to use.
        The default is to use 'Crittenden'.

        If ``m2_uform`` is 'Crittenden':

        .. math::

            U(r) &= \frac{1}{2\pi} (1-r^2) \exp(-r^2/2) \\
            T_\times(s) &= \frac{s^2}{128} (12-s^2) \exp(-s^2/4)

        cf. Crittenden, et al (2002): ApJ, 568, 20

        If ``m2_uform`` is 'Schneider':

        .. math::

            U(r) &= \frac{9}{\pi} (1-r^2) (1/3-r^2) \\
            T_\times(s) &= \frac{18}{\pi} s^2 \arccos(s/2) \\
            &\qquad - \frac{3}{40\pi} s^3 \sqrt{4-s^2} (196 - 74s^2 + 14s^4 - s^6)

        cf. Schneider, et al (2002): A&A, 389, 729

        In neither case is this formula in the above papers, but the derivation is similar
        to the derivations of :math:`T_+` and :math:`T_-` in Schneider et al. (2002).

        Parameters:
            R (array):          The R values at which to calculate the aperture mass statistics.
                                (default: None, which means use self.rnom)
            rg (NGCorrelation): The cross-correlation using random locations as the lenses
                                (RG), if desired.  (default: None)
            m2_uform (str):     Which form to use for the aperture mass, as described above.
                                (default: 'Crittenden'; this value can also be given in the
                                constructor in the config dict.)

        Returns:
            Tuple containing

                - nmap = array of :math:`\langle N M_{ap} \rangle(R)`
                - nmx = array of :math:`\langle N M_{\times} \rangle(R)`
                - varnmap = array of variance estimates of the above values
        """
        if m2_uform is None:
            m2_uform = self.config.get('m2_uform','Crittenden')
        if m2_uform not in ['Crittenden', 'Schneider']:
            raise ValueError("Invalid m2_uform")
        if R is None:
            R = self.rnom

        # Make s a matrix, so we can eventually do the integral by doing a matrix product.
        s = np.outer(1./R, self.meanr)
        ssq = s*s
        if m2_uform == 'Crittenden':
            exp_factor = np.exp(-ssq/4.)
            Tx = ssq * (12. - ssq) / 128. * exp_factor
        else:
            Tx = np.zeros_like(s)
            sa = s[s<2.]
            ssqa = ssq[s<2.]
            Tx[s<2.] = 196. + ssqa*(-74. + ssqa*(14. - ssqa))
            Tx[s<2.] *= -3./(40.*np.pi) * sa * ssqa * np.sqrt(4.-sa**2)
            Tx[s<2.] += 18./np.pi * ssqa * np.arccos(sa/2.)
        Tx *= ssq

        xi, xi_im, varxi = self.calculateXi(rg=rg)

        # Now do the integral by taking the matrix products.
        # Note that dlogr = bin_size
        Txxi = Tx.dot(xi)
        Txxi_im = Tx.dot(xi_im)
        nmap = Txxi * self.bin_size
        nmx = Txxi_im * self.bin_size

        # The variance of each of these is
        # Var(<NMap>(R)) = int_r=0..2R [s^4 dlogr^2 Tx(s)^2 Var(xi)]
        varnmap = (Tx**2).dot(varxi) * self.bin_size**2

        return nmap, nmx, varnmap

    @depr_pos_kwargs
    def writeNMap(self, file_name, *, R=None, rg=None, m2_uform=None, file_type=None,
                  precision=None):
        r"""Write the cross correlation of the foreground galaxy counts with the aperture mass
        based on the correlation function to the file, file_name.

        If rg is provided, the compensated calculation will be used for :math:`\xi`.

        See `calculateNMap` for an explanation of the ``m2_uform`` parameter.

        The output file will include the following columns:

        ==========      =========================================================
        Column          Description
        ==========      =========================================================
        R               The radius of the aperture.
        NMap            An estimate of :math:`\langle N_{ap} M_{ap} \rangle(R)`
        NMx             An estimate of :math:`\langle N_{ap} M_\times \rangle(R)`
        sig_nmap        The sqrt of the variance estimate of either of these
        ==========      =========================================================


        Parameters:
            file_name (str):    The name of the file to write to.
            R (array):          The R values at which to calculate the aperture mass statistics.
                                (default: None, which means use self.rnom)
            rg (NGCorrelation): The cross-correlation using random locations as the lenses
                                (RG), if desired.  (default: None)
            m2_uform (str):     Which form to use for the aperture mass.  (default: 'Crittenden';
                                this value can also be given in the constructor in the config dict.)
            file_type (str):    The type of file to write ('ASCII' or 'FITS').  (default: determine
                                the type automatically from the extension of file_name.)
            precision (int):    For ASCII output catalogs, the desired precision. (default: 4;
                                this value can also be given in the constructor in the config dict.)
        """
        self.logger.info('Writing NMap from NG correlations to %s',file_name)
        if R is None:
            R = self.rnom

        nmap, nmx, varnmap = self.calculateNMap(R=R, rg=rg, m2_uform=m2_uform)
        if precision is None:
            precision = self.config.get('precision', 4)

        col_names = ['R','NMap','NMx','sig_nmap']
        columns = [ R, nmap, nmx, np.sqrt(varnmap) ]
        writer = make_writer(file_name, precision, file_type, logger=self.logger)
        with writer:
            writer.write(col_names, columns)

    @depr_pos_kwargs
    def writeNorm(self, file_name, *, gg, dd, rr, R=None, dr=None, rg=None,
                  m2_uform=None, file_type=None, precision=None):
        r"""Write the normalized aperture mass cross-correlation to the file, file_name.

        The combination :math:`\langle N M_{ap}\rangle^2 / \langle M_{ap}^2\rangle
        \langle N_{ap}^2\rangle` is related to :math:`r`, the galaxy-mass correlation
        coefficient.  Similarly, :math:`\langle N_{ap}^2\rangle / \langle M_{ap}^2\rangle`
        is related to :math:`b`, the galaxy bias parameter.  cf. Hoekstra et al, 2002:
        http://adsabs.harvard.edu/abs/2002ApJ...577..604H

        This function computes these combinations and outputs them to a file.

        - if rg is provided, the compensated calculation will be used for
          :math:`\langle N_{ap} M_{ap} \rangle`.
        - if dr is provided, the compensated calculation will be used for
          :math:`\langle N_{ap}^2 \rangle`.

        See `calculateNMap` for an explanation of the ``m2_uform`` parameter.

        The output file will include the following columns:

        ==========      =====================================================================
        Column          Description
        ==========      =====================================================================
        R               The radius of the aperture
        NMap            An estimate of :math:`\langle N_{ap} M_{ap} \rangle(R)`
        NMx             An estimate of :math:`\langle N_{ap} M_\times \rangle(R)`
        sig_nmap        The sqrt of the variance estimate of either of these
        Napsq           An estimate of :math:`\langle N_{ap}^2 \rangle(R)`
        sig_napsq       The sqrt of the variance estimate of :math:`\langle N_{ap}^2 \rangle`
        Mapsq           An estimate of :math:`\langle M_{ap}^2 \rangle(R)`
        sig_mapsq       The sqrt of the variance estimate of :math:`\langle M_{ap}^2 \rangle`
        NMap_norm       The ratio :math:`\langle N_{ap} M_{ap} \rangle^2 /`
                        :math:`\langle N_{ap}^2 \rangle \langle M_{ap}^2 \rangle`
        sig_norm        The sqrt of the variance estimate of this ratio
        Nsq_Mapsq       The ratio :math:`\langle N_{ap}^2 \rangle / \langle M_{ap}^2 \rangle`
        sig_nn_mm       The sqrt of the variance estimate of this ratio
        ==========      =====================================================================

        Parameters:
            file_name (str):    The name of the file to write to.
            gg (GGCorrelation): The auto-correlation of the shear field
            dd (NNCorrelation): The auto-correlation of the lens counts (DD)
            rr (NNCorrelation): The auto-correlation of the random field (RR)
            R (array):          The R values at which to calculate the aperture mass statistics.
                                (default: None, which means use self.rnom)
            dr (NNCorrelation): The cross-correlation of the data with randoms (DR), if
                                desired, in which case the Landy-Szalay estimator will be
                                calculated.  (default: None)
            rd (NNCorrelation): The cross-correlation of the randoms with data (RD), if
                                desired. (default: None, which means use rd=dr)
            rg (NGCorrelation): The cross-correlation using random locations as the lenses
                                (RG), if desired.  (default: None)
            m2_uform (str):     Which form to use for the aperture mass.  (default: 'Crittenden';
                                this value can also be given in the constructor in the config dict.)
            file_type (str):    The type of file to write ('ASCII' or 'FITS').  (default: determine
                                the type automatically from the extension of file_name.)
            precision (int):    For ASCII output catalogs, the desired precision. (default: 4;
                                this value can also be given in the constructor in the config dict.)
        """
        self.logger.info('Writing Norm from NG correlations to %s',file_name)
        if R is None:
            R = self.rnom

        nmap, nmx, varnmap = self.calculateNMap(R=R, rg=rg, m2_uform=m2_uform)
        mapsq, mapsq_im, mxsq, mxsq_im, varmapsq = gg.calculateMapSq(R=R, m2_uform=m2_uform)
        nsq, varnsq = dd.calculateNapSq(R=R, rr=rr, dr=dr, m2_uform=m2_uform)

        nmnorm = nmap**2 / (nsq * mapsq)
        varnmnorm = nmnorm**2 * (4. * varnmap / nmap**2 + varnsq / nsq**2 + varmapsq / mapsq**2)
        nnnorm = nsq / mapsq
        varnnnorm = nnnorm**2 * (varnsq / nsq**2 + varmapsq / mapsq**2)
        if precision is None:
            precision = self.config.get('precision', 4)

        col_names = [ 'R',
                      'NMap','NMx','sig_nmap',
                      'Napsq','sig_napsq','Mapsq','sig_mapsq',
                      'NMap_norm','sig_norm','Nsq_Mapsq','sig_nn_mm' ]
        columns = [ R,
                    nmap, nmx, np.sqrt(varnmap),
                    nsq, np.sqrt(varnsq), mapsq, np.sqrt(varmapsq),
                    nmnorm, np.sqrt(varnmnorm), nnnorm, np.sqrt(varnnnorm) ]
        writer = make_writer(file_name, precision, file_type, logger=self.logger)
        with writer:
            writer.write(col_names, columns)
