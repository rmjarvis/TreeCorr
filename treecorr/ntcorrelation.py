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

from . import _treecorr
from .catalog import calculateVarT
from .corr2base import Corr2
from .util import make_writer, make_reader


class NTCorrelation(Corr2):
    r"""This class handles the calculation and storage of a 2-point count-spin-3 correlation
    function.

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
        xi:         The correlation function, :math:`\xi(r) = \langle t_R\rangle`.
        xi_im:      The imaginary part of :math:`\xi(r)`.
        varxi:      An estimate of the variance of :math:`\xi`
        weight:     The total weight in each bin.
        npairs:     The number of pairs going into each bin (including pairs where one or
                    both objects have w=0).
        cov:        An estimate of the full covariance matrix.
        raw_xi:     The raw value of xi, uncorrected by an RT calculation. cf. `calculateXi`
        raw_xi_im:  The raw value of xi_im, uncorrected by an RT calculation. cf. `calculateXi`
        raw_varxi:  The raw value of varxi, uncorrected by an RT calculation. cf. `calculateXi`

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

        >>> nt = treecorr.NTCorrelation(config)
        >>> nt.process(cat1,cat2)   # Compute the cross-correlation.
        >>> nt.write(file_name)     # Write out to a file.
        >>> xi = nt.xi              # Or access the correlation function directly.

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
    def __init__(self, config=None, *, logger=None, **kwargs):
        """Initialize `NTCorrelation`.  See class doc for details.
        """
        Corr2.__init__(self, config, logger=logger, **kwargs)

        self.xi = np.zeros_like(self.rnom, dtype=float)
        self.xi_im = np.zeros_like(self.rnom, dtype=float)
        self.meanr = np.zeros_like(self.rnom, dtype=float)
        self.meanlogr = np.zeros_like(self.rnom, dtype=float)
        self.weight = np.zeros_like(self.rnom, dtype=float)
        self.npairs = np.zeros_like(self.rnom, dtype=float)
        self.raw_xi = self.xi
        self.raw_xi_im = self.xi_im
        self._rt = None
        self._raw_varxi = None
        self._varxi = None
        self._cov = None
        self._var_num = 0
        self._processed_cats = []
        self.logger.debug('Finished building NTCorr')

    @property
    def corr(self):
        if self._corr is None:
            x = np.array([])
            self._corr = _treecorr.NTCorr(self._bintype, self._min_sep, self._max_sep, self._nbins,
                                          self._bin_size, self.b, self.min_rpar, self.max_rpar,
                                          self.xperiod, self.yperiod, self.zperiod,
                                          self.raw_xi, self.raw_xi_im, x, x,
                                          self.meanr, self.meanlogr, self.weight, self.npairs)
        return self._corr

    def __eq__(self, other):
        """Return whether two `NTCorrelation` instances are equal"""
        return (isinstance(other, NTCorrelation) and
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
        ret = NTCorrelation.__new__(NTCorrelation)
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
        else:
            ret.raw_xi = self.raw_xi.copy()
            ret.raw_xi_im = self.raw_xi_im.copy()
        if self._rt is not None:
            ret._rt = self._rt.copy()
        return ret

    def __repr__(self):
        return 'NTCorrelation(config=%r)'%self.config

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
            self.logger.info('Starting process NT cross-correlations')
        else:
            self.logger.info('Starting process NT cross-correlations for cats %s, %s.',
                             cat1.name, cat2.name)

        self._set_metric(metric, cat1.coords, cat2.coords)
        self._set_num_threads(num_threads)
        min_size, max_size = self._get_minmax_size()

        f1 = cat1.getNField(min_size=min_size, max_size=max_size,
                            split_method=self.split_method,
                            brute=self.brute is True or self.brute == 1,
                            min_top=self.min_top, max_top=self.max_top,
                            coords=self.coords)
        f2 = cat2.getTField(min_size=min_size, max_size=max_size,
                            split_method=self.split_method,
                            brute=self.brute is True or self.brute == 2,
                            min_top=self.min_top, max_top=self.max_top,
                            coords=self.coords)

        self.logger.info('Starting %d jobs.',f1.nTopLevelNodes)
        self.corr.processCross(f1.data, f2.data, self.output_dots,
                               self._bintype, self._metric)

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

    def finalize(self, vart):
        """Finalize the calculation of the correlation function.

        The `process_cross` command accumulates values in each bin, so it can be called
        multiple times if appropriate.  Afterwards, this command finishes the calculation
        by dividing each column by the total weight.

        Parameters:
            vart (float):   The variance per component of the spin-3 field.
        """
        self._finalize()
        self._var_num = vart

        self.xi = self.raw_xi
        self.xi_im = self.raw_xi_im

    @property
    def raw_varxi(self):
        if self._raw_varxi is None:
            self._raw_varxi = np.zeros_like(self.rnom, dtype=float)
            if self._var_num != 0:
                self._raw_varxi.ravel()[:] = self.cov.diagonal()
        return self._raw_varxi

    @property
    def varxi(self):
        if self._varxi is None:
            self._varxi = self.raw_varxi
        return self._varxi

    def _clear(self):
        """Clear the data vectors
        """
        self.raw_xi.ravel()[:] = 0
        self.raw_xi_im.ravel()[:] = 0
        self.meanr.ravel()[:] = 0
        self.meanlogr.ravel()[:] = 0
        self.weight.ravel()[:] = 0
        self.npairs.ravel()[:] = 0
        self.xi = self.raw_xi
        self.xi_im = self.raw_xi_im
        self._raw_varxi = None
        self._varxi = None
        self._cov = None

    def __iadd__(self, other):
        """Add a second `NTCorrelation`'s data to this one.

        .. note::

            For this to make sense, both `NTCorrelation` objects should not have had `finalize`
            called yet.  Then, after adding them together, you should call `finalize` on the sum.
        """
        if not isinstance(other, NTCorrelation):
            raise TypeError("Can only add another NTCorrelation object")
        if not (self._nbins == other._nbins and
                self.min_sep == other.min_sep and
                self.max_sep == other.max_sep):
            raise ValueError("NTCorrelation to be added is not compatible with this one.")

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
        self._raw_varxi = None
        self._varxi = None
        self._cov = None

    def process(self, cat1, cat2, *, metric=None, num_threads=None, comm=None, low_mem=False,
                initialize=True, finalize=True):
        """Compute the correlation function.

        Both arguments may be lists, in which case all items in the list are used
        for that element of the correlation.

        Parameters:
            cat1 (Catalog):     A catalog or list of catalogs for the N field.
            cat2 (Catalog):     A catalog or list of catalogs for the T field.
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
                                `Corr2.clear`.  (default: True)
            finalize (bool):    Whether to complete the calculation with a call to `finalize`.
                                (default: True)
        """
        import math
        if initialize:
            self.clear()
            self._rt = None
            self._processed_cats.clear()

        if not isinstance(cat1,list):
            cat1 = cat1.get_patches(low_mem=low_mem)
        if not isinstance(cat2,list):
            cat2 = cat2.get_patches(low_mem=low_mem)

        self._process_all_cross(cat1, cat2, metric, num_threads, comm, low_mem)

        self._processed_cats.extend(cat2)
        if finalize:
            vart = calculateVarT(self._processed_cats, low_mem=low_mem)
            self.logger.info("vart = %f: sig_sn (per component) = %f",vart,math.sqrt(vart))
            self.finalize(vart)
            self._processed_cats.clear()

    def calculateXi(self, *, rt=None):
        r"""Calculate the correlation function possibly given another correlation function
        that uses random points for the foreground objects.

        - If rt is None, the simple correlation function :math:`\langle t_R\rangle` is
          returned.
        - If rt is not None, then a compensated calculation is done:
          :math:`\langle t_R\rangle = (DT - RT)`, where DT represents the mean radial spin-3
          field around the data points and RT represents the mean radial spin-3 field around
          random points.

        After calling this function, the attributes ``xi``, ``xi_im``, ``varxi``, and ``cov`` will
        correspond to the compensated values (if rt is provided).  The raw, uncompensated values
        are available as ``rawxi``, ``raw_xi_im``, and ``raw_varxi``.

        Parameters:
            rt (NTCorrelation): The cross-correlation using random locations as the lenses
                                (RT), if desired.  (default: None)

        Returns:
            Tuple containing

                - xi = array of the real part of :math:`\xi(R)`
                - xi_im = array of the imaginary part of :math:`\xi(R)`
                - varxi = array of the variance estimates of the above values
        """
        if rt is not None:
            self.xi = self.raw_xi - rt.xi
            self.xi_im = self.raw_xi_im - rt.xi_im
            self._rt = rt

            if rt.npatch1 not in (1,self.npatch1) or rt.npatch2 != self.npatch2:
                raise RuntimeError("RT must be run with the same patches as DT")

            if len(self.results) > 0:
                # If there are any rt patch pairs that aren't in results (e.g. due to different
                # edge effects among the various pairs in consideration), then we need to add
                # some dummy results to make sure all the right pairs are computed when we make
                # the vectors for the covariance matrix.
                template = next(iter(self.results.values()))  # Just need something to copy.
                for ij in rt.results:
                    if ij in self.results: continue
                    new_cij = template.copy()
                    new_cij.xi.ravel()[:] = 0
                    new_cij.weight.ravel()[:] = 0
                    self.results[ij] = new_cij

                self._cov = self.estimate_cov(self.var_method)
                self._varxi = np.zeros_like(self.rnom, dtype=float)
                self._varxi.ravel()[:] = self.cov.diagonal()
            else:
                self._varxi = self.raw_varxi + rt.varxi
        else:
            self.xi = self.raw_xi
            self.xi_im = self.raw_xi_im
            self._varxi = self.raw_varxi

        return self.xi, self.xi_im, self.varxi

    def _calculate_xi_from_pairs(self, pairs):
        self._sum([self.results[ij] for ij in pairs])
        self._finalize()
        if self._rt is not None:
            # If rt has npatch1 = 1, adjust pairs appropriately
            if self._rt.npatch1 == 1 and not all([p[0] == 0 for p in pairs]):
                pairs = [(0,ij[1]) for ij in pairs if ij[0] == ij[1]]
            # Make sure all ij are in the rt results (some might be missing, which is ok)
            pairs = [ij for ij in pairs if self._rt._ok[ij[0],ij[1]]]
            self._rt._calculate_xi_from_pairs(pairs)
            self.xi -= self._rt.xi

    def write(self, file_name, *, rt=None, file_type=None, precision=None,
              write_patch_results=False):
        r"""Write the correlation function to the file, file_name.

        - If rt is None, the simple correlation function :math:`\langle t_R\rangle` is used.
        - If rt is not None, then a compensated calculation is done:
          :math:`\langle t_R\rangle = (DT - RT)`, where DT represents the mean spin-3 field
          around the data points and RT represents the mean spin-3 field around random points.

        The output file will include the following columns:

        ==========      =============================================================
        Column          Description
        ==========      =============================================================
        r_nom           The nominal center of the bin in r
        meanr           The mean value :math:`\langle r \rangle` of pairs that fell
                        into each bin
        meanlogr        The mean value :math:`\langle \log(r) \rangle` of pairs that
                        fell into each bin
        tR              The mean real part of the spin-3 field relative to the
                        center points.
        tR_im           The mean imaginary part of the spin-3 field relative to the
                        center points.
        sigma           The sqrt of the variance estimate of either of these
        weight          The total weight contributing to each bin
        npairs          The total number of pairs in each bin
        ==========      =============================================================

        If ``sep_units`` was given at construction, then the distances will all be in these units.
        Otherwise, they will be in either the same units as x,y,z (for flat or 3d coordinates) or
        radians (for spherical coordinates).

        Parameters:
            file_name (str):    The name of the file to write to.
            rt (NTCorrelation): The cross-correlation using random locations as the lenses
                                (RT), if desired.  (default: None)
            file_type (str):    The type of file to write ('ASCII' or 'FITS').  (default: determine
                                the type automatically from the extension of file_name.)
            precision (int):    For ASCII output catalogs, the desired precision. (default: 4;
                                this value can also be given in the constructor in the config dict.)
            write_patch_results (bool): Whether to write the patch-based results as well.
                                        (default: False)
        """
        self.logger.info('Writing NT correlations to %s',file_name)
        self.calculateXi(rt=rt)
        precision = self.config.get('precision', 4) if precision is None else precision
        name = 'main' if write_patch_results else None
        with make_writer(file_name, precision, file_type, self.logger) as writer:
            self._write(writer, name, write_patch_results)

    @property
    def _write_col_names(self):
        return ['r_nom','meanr','meanlogr','tR','tR_im','sigma','weight','npairs']

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

    def read(self, file_name, *, file_type=None):
        """Read in values from a file.

        This should be a file that was written by TreeCorr, preferably a FITS file, so there
        is no loss of information.

        .. warning::

            The `NTCorrelation` object should be constructed with the same configuration
            parameters as the one being read.  e.g. the same min_sep, max_sep, etc.  This is not
            checked by the read function.

        Parameters:
            file_name (str):    The name of the file to read in.
            file_type (str):    The type of file ('ASCII' or 'FITS').  (default: determine the type
                                automatically from the extension of file_name.)
        """
        self.logger.info('Reading NT correlations from %s',file_name)
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
        self.xi = data['tR'].reshape(s)
        self.xi_im = data['tR_im'].reshape(s)
        self._varxi = data['sigma'].reshape(s)**2
        self.weight = data['weight'].reshape(s)
        self.npairs = data['npairs'].reshape(s)
        self.coords = params['coords'].strip()
        self.metric = params['metric'].strip()
        self._ro.sep_units = params['sep_units'].strip()
        self._ro.bin_type = params['bin_type'].strip()
        self.raw_xi = self.xi
        self.raw_xi_im = self.xi_im
        self._raw_varxi = self._varxi
        self.npatch1 = params.get('npatch1', 1)
        self.npatch2 = params.get('npatch2', 1)
