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
.. module:: nnncorrelation
"""

import numpy as np

from . import _treecorr
from .catalog import calculateVarG
from .corr3base import Corr3
from .util import make_writer, make_reader


class GGGCorrelation(Corr3):
    r"""This class handles the calculation and storage of a 3-point shear-shear-shear correlation
    function.

    We use the "natural components" of the shear 3-point function described by Schneider &
    Lombardi (2003) [Astron.Astrophys. 397 (2003) 809-818].  In this paradigm, the shears
    are projected relative to some point defined by the geometry of the triangle.  They
    give several reasonable choices for this point.  We choose the triangle's centroid as the
    "most natural" point, as many simple shear fields have purely real :math:`\Gamma_0` using
    this definition.  It is also a fairly simple point to calculate in the code compared to
    some of the other options they offer, so projections relative to it are fairly efficient.

    There are 4 complex-valued 3-point shear corrletion functions defined for triples of shear
    values projected relative to the line joining the location of the shear to the cenroid of
    the triangle:

    .. math::

        \Gamma_0 &= \langle \gamma(\mathbf{x1}) \gamma(\mathbf{x2}) \gamma(\mathbf{x3}) \rangle \\
        \Gamma_1 &= \langle \gamma(\mathbf{x1})^* \gamma(\mathbf{x2}) \gamma(\mathbf{x3}) \rangle \\
        \Gamma_2 &= \langle \gamma(\mathbf{x1}) \gamma(\mathbf{x2})^* \gamma(\mathbf{x3}) \rangle \\
        \Gamma_3 &= \langle \gamma(\mathbf{x1}) \gamma(\mathbf{x2}) \gamma(\mathbf{x3})^* \rangle \\

    where :math:`\mathbf{x1}, \mathbf{x2}, \mathbf{x3}` are the corners of the triange opposite
    sides d1, d2, d3 respectively, where d1 > d2 > d3, and :math:`{}^*` indicates complex
    conjugation.

    See the doc string of `Corr3` for a description of how the triangles
    are binned.

    Ojects of this class holds the following attributes:

    Attributes:
        nbins:      The number of bins in logr where r = d2.
        bin_size:   The size of the bins in logr.
        min_sep:    The minimum separation being considered.
        max_sep:    The maximum separation being considered.
        logr1d:     The nominal centers of the nbins bins in log(r).

    If the bin_type is LogRUV, then it will have these attributes:

    Attributes:
        nubins:     The number of bins in u where u = d3/d2.
        ubin_size:  The size of the bins in u.
        min_u:      The minimum u being considered.
        max_u:      The maximum u being considered.
        nvbins:     The number of bins in v where v = +-(d1-d2)/d3.
        vbin_size:  The size of the bins in v.
        min_v:      The minimum v being considered.
        max_v:      The maximum v being considered.
        u1d:        The nominal centers of the nubins bins in u.
        v1d:        The nominal centers of the nvbins bins in v.

    If the bin_type is LogSAS, then it will have these attributes:

    Attributes:
        nphi_bins:  The number of bins in phi.
        phi_bin_size: The size of the bins in phi.
        min_phi:    The minimum phi being considered.
        max_phi:    The maximum phi being considered.
        phi1d:      The nominal centers of the nphi_bins bins in phi.

    If the bin_type is LogMultipole, then it will have these attributes:

    Attributes:
        max_n:      The maximum multipole index n being stored.
        n1d:        The multipole index n in the 2*max_n+1 bins of the third bin direction.

    In addition, the following attributes are numpy arrays whose shape is:

        * (nbins, nubins, nvbins) if bin_type is LogRUV
        * (nbins, nbins, nphi_bins) if bin_type is LogSAS
        * (nbins, nbins, 2*max_n+1) if bin_type is LogMultipole

    If bin_type is LogRUV:

    Attributes:
        logr:       The nominal center of each bin in log(r).
        rnom:       The nominal center of each bin converted to regular distance.
                    i.e. r = exp(logr).
        u:          The nominal center of each bin in u.
        v:          The nominal center of each bin in v.
        meanu:      The (weighted) mean value of u for the triangles in each bin.
        meanv:      The (weighted) mean value of v for the triangles in each bin.

    If bin_type is LogSAS:

    Attributes:
        logd2:      The nominal center of each bin in log(d2).
        d2nom:      The nominal center of each bin converted to regular d2 distance.
                    i.e. d2 = exp(logd2).
        logd3:      The nominal center of each bin in log(d3).
        d3nom:      The nominal center of each bin converted to regular d3 distance.
                    i.e. d3 = exp(logd3).
        phi:        The nominal center of each angular bin.
        meanphi:    The (weighted) mean value of phi for the triangles in each bin.

    If bin_type is LogMultipole:

    Attributes:
        logd2:      The nominal center of each bin in log(d2).
        d2nom:      The nominal center of each bin converted to regular d2 distance.
                    i.e. d2 = exp(logd2).
        logd3:      The nominal center of each bin in log(d3).
        d3nom:      The nominal center of each bin converted to regular d3 distance.
                    i.e. d3 = exp(logd3).
        n:          The multipole index n for each bin.

    For any bin_type:

    Attributes:
        gam0:       The 0th "natural" correlation function, :math:`\Gamma_0`.
        gam1:       The 1st "natural" correlation function, :math:`\Gamma_1`.
        gam2:       The 2nd "natural" correlation function, :math:`\Gamma_2`.
        gam3:       The 3rd "natural" correlation function, :math:`\Gamma_3`.
        vargam0:    The variance of :math:`\Gamma_0`, only including the shot noise
                    propagated into the final correlation.  This (and the related values for
                    1,2,3) does not include sample variance, so it is always an underestimate
                    of the actual variance.
        vargam1:    The variance of :math:`\Gamma_1`.
        vargam2:    The variance of :math:`\Gamma_2`.
        vargam3:    The variance of :math:`\Gamma_3`.
        meand1:     The (weighted) mean value of d1 for the triangles in each bin.
        meanlogd1:  The (weighted) mean value of log(d1) for the triangles in each bin.
        meand2:     The (weighted) mean value of d2 for the triangles in each bin.
        meanlogd2:  The (weighted) mean value of log(d2) for the triangles in each bin.
        meand3:     The (weighted) mean value of d3 for the triangles in each bin.
        meanlogd3:  The (weighted) mean value of log(d3) for the triangles in each bin.
        weight:     The total weight in each bin.
        ntri:       The number of triangles going into each bin (including those where one or
                    more objects have w=0).

    If ``sep_units`` are given (either in the config dict or as a named kwarg) then the distances
    will all be in these units.

    .. note::

        If you separate out the steps of the `process` command and use `process_auto` and/or
        `process_cross`, then the units will not be applied to ``meanr`` or ``meanlogr`` until
        the `finalize` function is called.

    The typical usage pattern is as follows::

        >>> ggg = treecorr.GGGCorrelation(config)
        >>> ggg.process(cat)              # For auto-correlation.
        >>> ggg.process(cat1,cat2,cat3)   # For cross-correlation.
        >>> ggg.write(file_name)          # Write out to a file.
        >>> gam0 = ggg.gam0, etc.         # To access gamma values directly.
        >>> gam0r = ggg.gam0r             # You can also access real and imag parts separately.
        >>> gam0i = ggg.gam0i

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
    def __init__(self, config=None, *, logger=None, **kwargs):
        """Initialize `GGGCorrelation`.  See class doc for details.
        """
        Corr3.__init__(self, config, logger=logger, **kwargs)

        shape = self.data_shape
        self.gam0r = np.zeros(shape, dtype=float)
        self.gam1r = np.zeros(shape, dtype=float)
        self.gam2r = np.zeros(shape, dtype=float)
        self.gam3r = np.zeros(shape, dtype=float)
        self.gam0i = np.zeros(shape, dtype=float)
        self.gam1i = np.zeros(shape, dtype=float)
        self.gam2i = np.zeros(shape, dtype=float)
        self.gam3i = np.zeros(shape, dtype=float)
        self.weightr = np.zeros(shape, dtype=float)
        if self.bin_type == 'LogMultipole':
            self.weighti = np.zeros(shape, dtype=float)
        else:
            self.weighti = np.array([])
        self.ntri = np.zeros(shape, dtype=float)
        self._vargam0 = None
        self._vargam1 = None
        self._vargam2 = None
        self._vargam3 = None
        self._cov = None
        self._var_num = 0
        self.logger.debug('Finished building GGGCorr')

    @property
    def weight(self):
        if self.weighti.size:
            return self.weightr + 1j * self.weighti
        else:
            return self.weightr

    @property
    def gam0(self):
        return self.gam0r + 1j * self.gam0i

    @property
    def gam1(self):
        return self.gam1r + 1j * self.gam1i

    @property
    def gam2(self):
        return self.gam2r + 1j * self.gam2i

    @property
    def gam3(self):
        return self.gam3r + 1j * self.gam3i

    @property
    def corr(self):
        if self._corr is None:
            self._corr = _treecorr.GGGCorr(
                    self._bintype,
                    self._min_sep, self._max_sep, self.nbins, self._bin_size, self.b,
                    self.angle_slop,
                    self._ro.min_u,self._ro.max_u,self._ro.nubins,self._ro.ubin_size,self.bu,
                    self._ro.min_v,self._ro.max_v,self._ro.nvbins,self._ro.vbin_size,self.bv,
                    self.xperiod, self.yperiod, self.zperiod,
                    self.gam0r, self.gam0i, self.gam1r, self.gam1i,
                    self.gam2r, self.gam2i, self.gam3r, self.gam3i,
                    self.meand1, self.meanlogd1, self.meand2, self.meanlogd2,
                    self.meand3, self.meanlogd3, self.meanu, self.meanv,
                    self.weightr, self.weighti, self.ntri)
        return self._corr

    def __eq__(self, other):
        """Return whether two `GGGCorrelation` instances are equal"""
        return (isinstance(other, GGGCorrelation) and
                self._equal_binning(other) and
                self._equal_bin_data(other) and
                np.array_equal(self.gam0r, other.gam0r) and
                np.array_equal(self.gam0i, other.gam0i) and
                np.array_equal(self.gam1r, other.gam1r) and
                np.array_equal(self.gam1i, other.gam1i) and
                np.array_equal(self.gam2r, other.gam2r) and
                np.array_equal(self.gam2i, other.gam2i) and
                np.array_equal(self.gam3r, other.gam3r) and
                np.array_equal(self.gam3i, other.gam3i) and
                np.array_equal(self.vargam0, other.vargam0) and
                np.array_equal(self.vargam1, other.vargam1) and
                np.array_equal(self.vargam2, other.vargam2) and
                np.array_equal(self.vargam3, other.vargam3) and
                np.array_equal(self.weight, other.weight) and
                np.array_equal(self.ntri, other.ntri))

    def copy(self):
        """Make a copy"""
        ret = GGGCorrelation.__new__(GGGCorrelation)
        for key, item in self.__dict__.items():
            if isinstance(item, np.ndarray):
                # Only items that might change need to by deep copied.
                ret.__dict__[key] = item.copy()
            else:
                # For everything else, shallow copy is fine.
                # In particular don't deep copy config or logger
                # Most of the rest are scalars, which copy fine this way.
                ret.__dict__[key] = item
        ret._corr = None # We'll want to make a new one of these if we need it.
        return ret

    def __repr__(self):
        return 'GGGCorrelation(config=%r)'%self.config

    def process_auto(self, cat, *, metric=None, num_threads=None):
        """Process a single catalog, accumulating the auto-correlation.

        This accumulates the auto-correlation for the given catalog.  After
        calling this function as often as desired, the `finalize` command will
        finish the calculation of meand1, meanlogd1, etc.

        Parameters:
            cat (Catalog):      The catalog to process
            metric (str):       Which metric to use.  See `Metrics` for details.
                                (default: 'Euclidean'; this value can also be given in the
                                constructor in the config dict.)
            num_threads (int):  How many OpenMP threads to use during the calculation.
                                (default: use the number of cpu cores; this value can also be given
                                in the constructor in the config dict.)
        """
        if cat.name == '':
            self.logger.info('Starting process GGG auto-correlations')
        else:
            self.logger.info('Starting process GGG auto-correlations for cat %s.', cat.name)

        self._set_metric(metric, cat.coords)
        self._set_num_threads(num_threads)
        min_size, max_size = self._get_minmax_size()

        field = cat.getGField(min_size=min_size, max_size=max_size,
                              split_method=self.split_method, brute=bool(self.brute),
                              min_top=self.min_top, max_top=self.max_top,
                              coords=self.coords)

        self.logger.info('Starting %d jobs.',field.nTopLevelNodes)
        self.corr.processAuto(field.data, self.output_dots, self._metric)

    def process_cross12(self, cat1, cat2, *, metric=None, ordered=True, num_threads=None):
        """Process two catalogs, accumulating the 3pt cross-correlation, where one of the
        points in each triangle come from the first catalog, and two come from the second.

        This accumulates the cross-correlation for the given catalogs as part of a larger
        auto- or cross-correlation calculation.  E.g. when splitting up a large catalog into
        patches, this is appropriate to use for the cross correlation between different patches
        as part of the complete auto-correlation of the full catalog.

        Parameters:
            cat1 (Catalog):     The first catalog to process. (1 point in each triangle will come
                                from this catalog.)
            cat2 (Catalog):     The second catalog to process. (2 points in each triangle will come
                                from this catalog.)
            metric (str):       Which metric to use.  See `Metrics` for details.
                                (default: 'Euclidean'; this value can also be given in the
                                constructor in the config dict.)
            ordered (bool):     Whether to fix the order of the triangle vertices to match the
                                catalogs. (default: True)
            num_threads (int):  How many OpenMP threads to use during the calculation.
                                (default: use the number of cpu cores; this value can also be given
                                in the constructor in the config dict.)
        """
        if cat1.name == '' and cat2.name == '':
            self.logger.info('Starting process GGG (1-2) cross-correlations')
        else:
            self.logger.info('Starting process GGG (1-2) cross-correlations for cats %s, %s.',
                             cat1.name, cat2.name)

        self._set_metric(metric, cat1.coords, cat2.coords)
        self._set_num_threads(num_threads)
        min_size, max_size = self._get_minmax_size()

        f1 = cat1.getGField(min_size=min_size, max_size=max_size,
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
        # Note: all 3 correlation objects are the same.  Thus, all triangles will be placed
        # into self.corr, whichever way the three catalogs are permuted for each triangle.
        self.corr.processCross12(f1.data, f2.data, (1 if ordered else 0),
                                 self.output_dots, self._metric)

    def process_cross(self, cat1, cat2, cat3, *, metric=None, ordered=True, num_threads=None):
        """Process a set of three catalogs, accumulating the 3pt cross-correlation.

        This accumulates the cross-correlation for the given catalogs as part of a larger
        auto- or cross-correlation calculation.  E.g. when splitting up a large catalog into
        patches, this is appropriate to use for the cross correlation between different patches
        as part of the complete auto-correlation of the full catalog.

        Parameters:
            cat1 (Catalog):     The first catalog to process
            cat2 (Catalog):     The second catalog to process
            cat3 (Catalog):     The third catalog to process
            metric (str):       Which metric to use.  See `Metrics` for details.
                                (default: 'Euclidean'; this value can also be given in the
                                constructor in the config dict.)
            ordered (bool):     Whether to fix the order of the triangle vertices to match the
                                catalogs. (default: True)
            num_threads (int):  How many OpenMP threads to use during the calculation.
                                (default: use the number of cpu cores; this value can also be given
                                in the constructor in the config dict.)
        """
        if cat1.name == '' and cat2.name == '' and cat3.name == '':
            self.logger.info('Starting process GGG cross-correlations')
        else:
            self.logger.info('Starting process GGG cross-correlations for cats %s, %s, %s.',
                             cat1.name, cat2.name, cat3.name)

        self._set_metric(metric, cat1.coords, cat2.coords, cat3.coords)
        self._set_num_threads(num_threads)
        min_size, max_size = self._get_minmax_size()

        f1 = cat1.getGField(min_size=min_size, max_size=max_size,
                            split_method=self.split_method,
                            brute=self.brute is True or self.brute == 1,
                            min_top=self.min_top, max_top=self.max_top,
                            coords=self.coords)
        f2 = cat2.getGField(min_size=min_size, max_size=max_size,
                            split_method=self.split_method,
                            brute=self.brute is True or self.brute == 2,
                            min_top=self.min_top, max_top=self.max_top,
                            coords=self.coords)
        f3 = cat3.getGField(min_size=min_size, max_size=max_size,
                            split_method=self.split_method,
                            brute=self.brute is True or self.brute == 3,
                            min_top=self.min_top, max_top=self.max_top,
                            coords=self.coords)

        self.logger.info('Starting %d jobs.',f1.nTopLevelNodes)
        # Note: all 6 correlation objects are the same.  Thus, all triangles will be placed
        # into self.corr, whichever way the three catalogs are permuted for each triangle.
        self.corr.processCross(f1.data, f2.data, f3.data,
                               (3 if ordered is True else 1 if ordered == 1 else 0),
                               self.output_dots, self._metric)

    def _finalize(self):
        mask1 = self.weightr != 0
        mask2 = self.weightr == 0

        self.meand2[mask1] /= self.weightr[mask1]
        self.meanlogd2[mask1] /= self.weightr[mask1]
        self.meand3[mask1] /= self.weightr[mask1]
        self.meanlogd3[mask1] /= self.weightr[mask1]
        if self.bin_type != 'LogMultipole':
            self.gam0r[mask1] /= self.weightr[mask1]
            self.gam0i[mask1] /= self.weightr[mask1]
            self.gam1r[mask1] /= self.weightr[mask1]
            self.gam1i[mask1] /= self.weightr[mask1]
            self.gam2r[mask1] /= self.weightr[mask1]
            self.gam2i[mask1] /= self.weightr[mask1]
            self.gam3r[mask1] /= self.weightr[mask1]
            self.gam3i[mask1] /= self.weightr[mask1]
            self.meand1[mask1] /= self.weightr[mask1]
            self.meanlogd1[mask1] /= self.weightr[mask1]
            self.meanu[mask1] /= self.weightr[mask1]
        if self.bin_type == 'LogRUV':
            self.meanv[mask1] /= self.weightr[mask1]

        # Update the units
        self._apply_units(mask1)

        # Set to nominal when no triangles in bin.
        if self.bin_type == 'LogRUV':
            self.meand2[mask2] = self.rnom[mask2]
            self.meanlogd2[mask2] = self.logr[mask2]
            self.meanu[mask2] = self.u[mask2]
            self.meanv[mask2] = self.v[mask2]
            self.meand3[mask2] = self.u[mask2] * self.meand2[mask2]
            self.meanlogd3[mask2] = np.log(self.meand3[mask2])
            self.meand1[mask2] = np.abs(self.v[mask2]) * self.meand3[mask2] + self.meand2[mask2]
            self.meanlogd1[mask2] = np.log(self.meand1[mask2])
        else:
            self.meand2[mask2] = self.d2nom[mask2]
            self.meanlogd2[mask2] = self.logd2[mask2]
            self.meand3[mask2] = self.d3nom[mask2]
            self.meanlogd3[mask2] = self.logd3[mask2]
            if self.bin_type == 'LogSAS':
                self.meanu[mask2] = self.phi[mask2]
                self.meand1[mask2] = np.sqrt(self.d2nom[mask2]**2 + self.d3nom[mask2]**2
                                             - 2*self.d2nom[mask2]*self.d3nom[mask2]*
                                             np.cos(self.phi[mask2]))
                self.meanlogd1[mask2] = np.log(self.meand1[mask2])
            else:
                self.meanu[mask2] = 0
                self.meand1[mask2] = 0
                self.meanlogd1[mask2] = 0

        if self.bin_type == 'LogMultipole':
            # Multipole only sets the meand values at [i,j,max_n].
            # (This is also where the complex weight is just a scalar = sum(www),
            # so the above normalizations are correct.)
            # Broadcast those to the rest of the values in the third dimension.
            self.ntri[:,:,:] = self.ntri[:,:,self.max_n][:,:,np.newaxis]
            self.meand2[:,:,:] = self.meand2[:,:,self.max_n][:,:,np.newaxis]
            self.meanlogd2[:,:,:] = self.meanlogd2[:,:,self.max_n][:,:,np.newaxis]
            self.meand3[:,:,:] = self.meand3[:,:,self.max_n][:,:,np.newaxis]
            self.meanlogd3[:,:,:] = self.meanlogd3[:,:,self.max_n][:,:,np.newaxis]

    def finalize(self, varg1, varg2, varg3):
        """Finalize the calculation of the correlation function.

        The `process_auto`, `process_cross12` and `process_cross` commands accumulate values in
        each bin, so they can be called multiple times if appropriate.  Afterwards, this command
        finishes the calculation by dividing by the total weight.

        Parameters:
            varg1 (float):  The variance per component of the first shear field.
            varg2 (float):  The variance per component of the second shear field.
            varg3 (float):  The variance per component of the third shear field.
        """
        self._finalize()
        mask1 = self.weightr != 0
        mask2 = self.weightr == 0
        self._var_num = 4 * varg1 * varg2 * varg3

        # I don't really understand why the variance is coming out 2x larger than the normal
        # formula for LogSAS.  But with just Gaussian noise, I need to multiply the numerator
        # by two to get the variance estimates to come out right.
        if self.bin_type in ['LogSAS', 'LogMultipole']:
            self._var_num *= 2

    @property
    def vargam0(self):
        if self._vargam0 is None:
            self._vargam0 = np.zeros(self.data_shape)
            if self._var_num != 0:
                self._vargam0.ravel()[:] = self.cov_diag[0:self._nbins].real
        return self._vargam0

    @property
    def vargam1(self):
        if self._vargam1 is None:
            self._vargam1 = np.zeros(self.data_shape)
            if self._var_num != 0:
                self._vargam1.ravel()[:] = self.cov_diag[self._nbins:2*self._nbins].real
        return self._vargam1

    @property
    def vargam2(self):
        if self._vargam2 is None:
            self._vargam2 = np.zeros(self.data_shape)
            if self._var_num != 0:
                self._vargam2.ravel()[:] = self.cov_diag[2*self._nbins:3*self._nbins].real
        return self._vargam2

    @property
    def vargam3(self):
        if self._vargam3 is None:
            self._vargam3 = np.zeros(self.data_shape)
            if self._var_num != 0:
                self._vargam3.ravel()[:] = self.cov_diag[3*self._nbins:4*self._nbins].real
        return self._vargam3

    def _clear(self):
        """Clear the data vectors
        """
        self.gam0r[:,:,:] = 0.
        self.gam0i[:,:,:] = 0.
        self.gam1r[:,:,:] = 0.
        self.gam1i[:,:,:] = 0.
        self.gam2r[:,:,:] = 0.
        self.gam2i[:,:,:] = 0.
        self.gam3r[:,:,:] = 0.
        self.gam3i[:,:,:] = 0.
        self.meand1[:,:,:] = 0.
        self.meanlogd1[:,:,:] = 0.
        self.meand2[:,:,:] = 0.
        self.meanlogd2[:,:,:] = 0.
        self.meand3[:,:,:] = 0.
        self.meanlogd3[:,:,:] = 0.
        self.meanu[:,:,:] = 0.
        if self.bin_type == 'LogRUV':
            self.meanv[:,:,:] = 0.
        self.weightr[:,:,:] = 0.
        if self.bin_type == 'LogMultipole':
            self.weighti[:,:,:] = 0.
        self.ntri[:,:,:] = 0.
        self._vargam0 = None
        self._vargam1 = None
        self._vargam2 = None
        self._vargam3 = None
        self._cov = None

    def __iadd__(self, other):
        """Add a second `GGGCorrelation`'s data to this one.

        .. note::

            For this to make sense, both `GGGCorrelation` objects should not have had `finalize`
            called yet.  Then, after adding them together, you should call `finalize` on the sum.
        """
        if not isinstance(other, GGGCorrelation):
            raise TypeError("Can only add another GGGCorrelation object")
        if not self._equal_binning(other, brief=True):
            raise ValueError("GGGCorrelation to be added is not compatible with this one.")

        if not other.nonzero: return self
        self._set_metric(other.metric, other.coords, other.coords, other.coords)
        self.gam0r[:] += other.gam0r[:]
        self.gam0i[:] += other.gam0i[:]
        self.gam1r[:] += other.gam1r[:]
        self.gam1i[:] += other.gam1i[:]
        self.gam2r[:] += other.gam2r[:]
        self.gam2i[:] += other.gam2i[:]
        self.gam3r[:] += other.gam3r[:]
        self.gam3i[:] += other.gam3i[:]
        self.meand1[:] += other.meand1[:]
        self.meanlogd1[:] += other.meanlogd1[:]
        self.meand2[:] += other.meand2[:]
        self.meanlogd2[:] += other.meanlogd2[:]
        self.meand3[:] += other.meand3[:]
        self.meanlogd3[:] += other.meanlogd3[:]
        self.meanu[:] += other.meanu[:]
        if self.bin_type == 'LogRUV':
            self.meanv[:] += other.meanv[:]
        self.weightr[:] += other.weightr[:]
        if self.bin_type == 'LogMultipole':
            self.weighti[:] += other.weighti[:]
        self.ntri[:] += other.ntri[:]
        return self

    def _sum(self, others):
        # Equivalent to the operation of:
        #     self._clear()
        #     for other in others:
        #         self += other
        # but no sanity checks and use numpy.sum for faster calculation.
        np.sum([c.gam0r for c in others], axis=0, out=self.gam0r)
        np.sum([c.gam0i for c in others], axis=0, out=self.gam0i)
        np.sum([c.gam1r for c in others], axis=0, out=self.gam1r)
        np.sum([c.gam1i for c in others], axis=0, out=self.gam1i)
        np.sum([c.gam2r for c in others], axis=0, out=self.gam2r)
        np.sum([c.gam2i for c in others], axis=0, out=self.gam2i)
        np.sum([c.gam3r for c in others], axis=0, out=self.gam3r)
        np.sum([c.gam3i for c in others], axis=0, out=self.gam3i)
        np.sum([c.meand1 for c in others], axis=0, out=self.meand1)
        np.sum([c.meanlogd1 for c in others], axis=0, out=self.meanlogd1)
        np.sum([c.meand2 for c in others], axis=0, out=self.meand2)
        np.sum([c.meanlogd2 for c in others], axis=0, out=self.meanlogd2)
        np.sum([c.meand3 for c in others], axis=0, out=self.meand3)
        np.sum([c.meanlogd3 for c in others], axis=0, out=self.meanlogd3)
        np.sum([c.meanu for c in others], axis=0, out=self.meanu)
        if self.bin_type == 'LogRUV':
            np.sum([c.meanv for c in others], axis=0, out=self.meanv)
        np.sum([c.weightr for c in others], axis=0, out=self.weightr)
        if self.bin_type == 'LogMultipole':
            np.sum([c.weighti for c in others], axis=0, out=self.weighti)
        np.sum([c.ntri for c in others], axis=0, out=self.ntri)

    def process(self, cat1, cat2=None, cat3=None, *, metric=None, ordered=True, num_threads=None,
                comm=None, low_mem=False, initialize=True, finalize=True,
                patch_method=None, algo=None, max_n=None):
        """Compute the 3pt correlation function.

        - If only 1 argument is given, then compute an auto-correlation function.
        - If 2 arguments are given, then compute a cross-correlation function with the
          first catalog taking one corner of the triangles, and the second taking two corners.
        - If 3 arguments are given, then compute a three-way cross-correlation function.

        For cross correlations, the default behavior is to use cat1 for the first vertex (P1),
        cat2 for the second vertex (P2), and cat3 for the third vertex (P3).  If only two
        catalogs are given, vertices P2 and P3 both come from cat2.  The sides d1, d2, d3,
        used to define the binning, are taken to be opposte P1, P2, P3 respectively.

        However, if you want to accumulate triangles where objects from each catalog can take
        any position in the triangles, you can set ``ordered=False``.  In this case,
        triangles will be formed where P1, P2 and P3 can come any input catalog, so long as there
        is one from cat1, one from cat2, and one from cat3 (or two from cat2 if cat3 is None).

        All arguments may be lists, in which case all items in the list are used
        for that element of the correlation.

        Parameters:
            cat1 (Catalog):     A catalog or list of catalogs for the first G field.
            cat2 (Catalog):     A catalog or list of catalogs for the second G field.
                                (default: None)
            cat3 (Catalog):     A catalog or list of catalogs for the third G field.
                                (default: None)
            metric (str):       Which metric to use.  See `Metrics` for details.
                                (default: 'Euclidean'; this value can also be given in the
                                constructor in the config dict.)
            ordered (bool):     Whether to fix the order of the triangle vertices to match the
                                catalogs. (see above; default: True)
            num_threads (int):  How many OpenMP threads to use during the calculation.
                                (default: use the number of cpu cores; this value can also be given
                                in the constructor in the config dict.)
            comm (mpi4py.Comm): If running MPI, an mpi4py Comm object to communicate between
                                processes.  If used, the rank=0 process will have the final
                                computation. This only works if using patches. (default: None)
            low_mem (bool):     Whether to sacrifice a little speed to try to reduce memory usage.
                                This only works if using patches. (default: False)
            initialize (bool):  Whether to begin the calculation with a call to
                                `Corr3.clear`.  (default: True)
            finalize (bool):    Whether to complete the calculation with a call to `finalize`.
                                (default: True)
            patch_method (str): Which patch method to use. (default is to use 'local' if
                                bin_type=LogMultipole, and 'global' otherwise)
            algo (str):         Which accumulation algorithm to use. (options are 'triangle' or
                                'multipole'; default is 'multipole' unless bin_type is 'LogRUV',
                                which can only use 'triangle')
            max_n (int):        If using the multpole algorithm, and this is not directly using
                                bin_type='LogMultipole', then this is the value of max_n to use
                                for the multipole part of the calculation. (default is to use
                                2pi/phi_bin_size; this value can also be given in the constructor
                                in the config dict.)
        """
        import math

        if algo is None:
            multipole = (self.bin_type != 'LogRUV')
        else:
            if algo not in ['triangle', 'multipole']:
                raise ValueError("Invalid algo %s"%algo)
            multipole = (algo == 'multipole')
            if multipole and self.bin_type == 'LogRUV':
                raise ValueError("LogRUV binning cannot use algo='multipole'")

        if multipole and self.bin_type != 'LogMultipole':
            config = self.config.copy()
            config['bin_type'] = 'LogMultipole'
            if max_n is None:
                max_n = 2.*np.pi / self.phi_bin_size
                # If max_n was given in constructor, use that instead.
                max_n = config.get('max_n', max_n)
            for key in ['min_phi', 'max_phi', 'nphi_bins', 'phi_bin_size']:
                config.pop(key, None)
            corr = GGGCorrelation(config, max_n=max_n)
            corr.process(cat1, cat2, cat3,
                         metric=metric, ordered=ordered, num_threads=num_threads,
                         comm=comm, low_mem=low_mem, initialize=initialize, finalize=finalize,
                         patch_method=patch_method, algo='multipole')
            corr.toSAS(target=self)
            return

        if patch_method is None:
            local = (self.bin_type == 'LogMultipole')
        else:
            if patch_method not in ['local', 'global']:
                raise ValueError("Invalid patch_method %s"%patch_method)
            local = (patch_method == 'local')
            if not local and self.bin_type == 'LogMultipole':
                raise ValueError("LogMultipole binning cannot use patch_method='global'")

        if not isinstance(cat1,list):
            cat1 = cat1.get_patches(low_mem=low_mem)
        if cat2 is not None and not isinstance(cat2,list):
            cat2 = cat2.get_patches(low_mem=low_mem)
        if cat3 is not None and not isinstance(cat3,list):
            cat3 = cat3.get_patches(low_mem=low_mem)

        if initialize:
            self.clear()

        if cat2 is None:
            if cat3 is not None:
                raise ValueError("For two catalog case, use cat1,cat2, not cat1,cat3")
            self._process_all_auto(cat1, metric, num_threads, comm, low_mem, local)
        elif cat3 is None:
            self._process_all_cross12(cat1, cat2, metric, ordered, num_threads, comm, low_mem, local)
        else:
            self._process_all_cross(cat1, cat2, cat3, metric, ordered, num_threads, comm, low_mem, local)

        if finalize:
            if cat2 is None:
                varg1 = calculateVarG(cat1, low_mem=low_mem)
                varg2 = varg1
                varg3 = varg1
                self.logger.info("varg = %f: sig_g = %f",varg1,math.sqrt(varg1))
            elif cat3 is None:
                varg1 = calculateVarG(cat1, low_mem=low_mem)
                varg2 = calculateVarG(cat2, low_mem=low_mem)
                varg3 = varg2
                self.logger.info("varg1 = %f: sig_g = %f",varg1,math.sqrt(varg1))
                self.logger.info("varg2 = %f: sig_g = %f",varg2,math.sqrt(varg2))
            else:
                varg1 = calculateVarG(cat1, low_mem=low_mem)
                varg2 = calculateVarG(cat2, low_mem=low_mem)
                varg3 = calculateVarG(cat3, low_mem=low_mem)
                self.logger.info("varg1 = %f: sig_g = %f",varg1,math.sqrt(varg1))
                self.logger.info("varg2 = %f: sig_g = %f",varg2,math.sqrt(varg2))
                self.logger.info("varg3 = %f: sig_g = %f",varg3,math.sqrt(varg3))
            self.finalize(varg1,varg2,varg3)

    def getStat(self):
        """The standard statistic for the current correlation object as a 1-d array.

        In this case, the concatenation of gam0.ravel(), gam1.ravel(), gam2.ravel(), gam3.ravel().

        .. note::

            This is a complex array, unlike most other statistics.
            The computed covariance matrix will be complex, although since it is Hermitian the
            diagonal is real, so the resulting vargam0, etc. will all be real arrays.
        """
        return np.concatenate([self.gam0.ravel(), self.gam1.ravel(),
                               self.gam2.ravel(), self.gam3.ravel()])

    def getWeight(self):
        """The weight array for the current correlation object as a 1-d array.

        In this case, 4 copies of self.weight.ravel().
        """
        return np.concatenate([np.abs(self.weight.ravel())] * 4)

    def toSAS(self, *, target=None, **kwargs):
        """Convert a multipole-binned correlation to the corresponding SAS binning.

        This is only valid for bin_type == LogMultipole.

        Keyword Arguments:
            target:     A target GGGCorrelation object with LogSAS binning to write to.
                        If this is not given, a new object will be created based on the
                        configuration paramters of the current object. (default: None)
            **kwargs:   Any kwargs that you want to use to configure the returned object.
                        Typically, might include min_phi, max_phi, nphi_bins, phi_bin_size.
                        The default phi binning is [0,pi] with nphi_bins = self.max_n.

        Returns:
            sas:        A GGGCorrelation object with bin_type=LogSAS containing the
                        same information as this object, but with the SAS binning.
        """
        if self.bin_type != 'LogMultipole':
            raise TypeError("toSAS is invalid for bin_type = %s"%self.bin_type)

        if target is None:
            config = self.config.copy()
            config['bin_type'] = 'LogSAS'
            max_n = config.pop('max_n')
            if 'nphi_bins' not in kwargs and 'phi_bin_size' not in kwargs:
                config['nphi_bins'] = max_n
            sas = GGGCorrelation(config, **kwargs)
        else:
            sas = target
            sas.clear()
        if not np.array_equal(sas.rnom1d, self.rnom1d):
            raise ValueError("toSAS cannot change sep parameters")

        # Copy these over
        sas.meand2[:,:,:] = self.meand2[:,:,0][:,:,None]
        sas.meanlogd2[:,:,:] = self.meanlogd2[:,:,0][:,:,None]
        sas.meand3[:,:,:] = self.meand3[:,:,0][:,:,None]
        sas.meanlogd3[:,:,:] = self.meanlogd3[:,:,0][:,:,None]
        sas._var_num = self._var_num
        sas.npatch1 = self.npatch1
        sas.npatch2 = self.npatch2
        sas.npatch3 = self.npatch3
        sas.coords = self.coords
        sas.metric = self.metric

        # Use nominal for meanphi
        sas.meanu[:] = sas.phi / sas._phi_units
        # Compute d1 from actual d2,d3 and nominal phi
        sas.meand1[:] = np.sqrt(sas.meand2**2 + sas.meand3**2
                                - 2*sas.meand2 * sas.meand3 * np.cos(sas.phi))
        sas.meanlogd1[:] = np.log(sas.meand1)

        # Z(d2,d3,phi) = 1/2pi sum_n Z_n(d2,d3) exp(i n phi)
        expiphi = np.exp(1j * self.n1d[:,None] * sas.phi1d)
        sas.weightr[:] = np.real(self.weight.dot(expiphi)) / (2*np.pi) * sas.phi_bin_size
        gam0 = self.gam0.dot(expiphi) / (2*np.pi) * sas.phi_bin_size
        gam1 = self.gam1.dot(expiphi) / (2*np.pi) * sas.phi_bin_size
        gam2 = self.gam2.dot(expiphi) / (2*np.pi) * sas.phi_bin_size
        gam3 = self.gam3.dot(expiphi) / (2*np.pi) * sas.phi_bin_size

        # We leave the gam_mu unnormalized in the Multipole class, so after the FT,
        # we still need to divide by weight.
        mask = sas.weightr != 0
        gam0[mask] /= sas.weightr[mask]
        gam1[mask] /= sas.weightr[mask]
        gam2[mask] /= sas.weightr[mask]
        gam3[mask] /= sas.weightr[mask]

        # Now fix the projection.
        # The multipole algorithm uses the Porth et al x projection.
        # We need to switch that to the canoical centroid projection.

        # Define some complex "vectors" where p1 is at the origin and
        # p3 is on the x axis:
        # s = p3 - p1
        # t = p2 - p1
        # u = angle bisector of s, t
        # q1 = (s+t)/3.  (this is the centroid)
        # q2 = q1-t
        # q3 = q1-s
        s = sas.meand2
        t = sas.meand3 * np.exp(1j * sas.meanphi * sas._phi_units)
        u = (1 + t/np.abs(t))/2
        q1 = (s+t)/3.
        q2 = q1-t
        q3 = q1-s

        # Currently the projection is as follows:
        # g1 is projected along u
        # g2 is projected along t
        # g3 is projected along s
        #
        # We want to have
        # g1 projected along q1
        # g2 projected along q2
        # g3 projected along q3
        #
        # The phases to multiply by are exp(2iphi_current) * exp(-2iphi_target). I.e.
        # g1phase = (u conj(q1))**2 / |u conj(q1)|**2
        # g2phase = (t conj(q2))**2 / |t conj(q2)|**2
        # g3phase = (s conj(q3))**2 / |s conj(q3)|**2
        g1phase = (u * np.conj(q1))**2
        g2phase = (t * np.conj(q2))**2
        g3phase = (s * np.conj(q3))**2
        g1phase /= np.abs(g1phase)
        g2phase /= np.abs(g2phase)
        g3phase /= np.abs(g3phase)

        # Now just multiply each gam by the appropriate combination of phases.
        gam0phase = g1phase * g2phase * g3phase
        gam1phase = np.conj(g1phase) * g2phase * g3phase
        gam2phase = g1phase * np.conj(g2phase) * g3phase
        gam3phase = g1phase * g2phase * np.conj(g3phase)
        gam0 *= gam0phase
        gam1 *= gam1phase
        gam2 *= gam2phase
        gam3 *= gam3phase

        sas.gam0r = np.real(gam0)
        sas.gam0i = np.imag(gam0)
        sas.gam1r = np.real(gam1)
        sas.gam1i = np.imag(gam1)
        sas.gam2r = np.real(gam2)
        sas.gam2i = np.imag(gam2)
        sas.gam3r = np.real(gam3)
        sas.gam3i = np.imag(gam3)

        # For ntri, we recorded the total ntri for each pair of d2,d3.
        # Allocate those proportionally to the weights.
        # Note: Multipole counts the weight for all 0 < phi < 2pi.
        # We reduce this by the fraction of this covered by [min_phi, max_phi].
        # (Typically 1/2, since usually [0,pi].)
        phi_frac = (sas.max_phi - sas.min_phi) / (2*np.pi)
        denom = np.sum(sas.weight, axis=2)
        denom[denom==0] = 1  # Don't divide by 0
        ratio = self.ntri[:,:,0] / denom * phi_frac
        sas.ntri[:] = sas.weight * ratio[:,:,None]

        for k,v in self.results.items():
            temp = sas.copy()
            temp.weightr[:] = np.real(v.weight.dot(expiphi)) / (2*np.pi) * sas.phi_bin_size
            temp.ntri[:] = temp.weight * ratio[:,:,None]
            gam0 = v.gam0.dot(expiphi) / (2*np.pi) * sas.phi_bin_size * gam0phase
            gam1 = v.gam1.dot(expiphi) / (2*np.pi) * sas.phi_bin_size * gam1phase
            gam2 = v.gam2.dot(expiphi) / (2*np.pi) * sas.phi_bin_size * gam2phase
            gam3 = v.gam3.dot(expiphi) / (2*np.pi) * sas.phi_bin_size * gam3phase
            temp.gam0r = np.real(gam0)
            temp.gam0i = np.imag(gam0)
            temp.gam1r = np.real(gam1)
            temp.gam1i = np.imag(gam1)
            temp.gam2r = np.real(gam2)
            temp.gam2i = np.imag(gam2)
            temp.gam3r = np.real(gam3)
            temp.gam3i = np.imag(gam3)

            # Undo the normalization of the d arrays.
            temp.meand1 *= temp.weightr
            temp.meand2 *= temp.weightr
            temp.meand3 *= temp.weightr
            temp.meanlogd1 *= temp.weightr
            temp.meanlogd2 *= temp.weightr
            temp.meanlogd3 *= temp.weightr
            temp.meanu *= temp.weightr

            sas.results[k] = temp

        return sas

    def write(self, file_name, *, file_type=None, precision=None, write_patch_results=False):
        r"""Write the correlation function to the file, file_name.

        As described in the doc string for `GGGCorrelation`, we use the "natural components" of
        the shear 3-point function described by Schneider & Lombardi (2003) using the triangle
        centroid as the projection point.  There are 4 complex-valued natural components, so there
        are 8 columns in the output file.

        For bin_type = LogRUV, the output file will include the following columns:

        ==========      ================================================================
        Column          Description
        ==========      ================================================================
        r_nom           The nominal center of the bin in r = d2 where d1 > d2 > d3
        u_nom           The nominal center of the bin in u = d3/d2
        v_nom           The nominal center of the bin in v = +-(d1-d2)/d3
        meanu           The mean value :math:`\langle u\rangle` of triangles that fell
                        into each bin
        meanv           The mean value :math:`\langle v\rangle` of triangles that fell
                        into each bin
        ==========      ================================================================

        For bin_type = LogSAS, the output file will include the following columns:

        ==========      ================================================================
        Column          Description
        ==========      ================================================================
        d2_nom          The nominal center of the bin in d2
        d3_nom          The nominal center of the bin in d3
        phi_nom         The nominal center of the bin in phi, the opening angle between
                        d2 and d3 in the counter-clockwise direction
        meanphi         The mean value :math:`\langle phi\rangle` of triangles that fell
                        into each bin
        ==========      ================================================================

        For bin_type = LogMultipole, the output file will include the following columns:

        ==========      ================================================================
        Column          Description
        ==========      ================================================================
        d2_nom          The nominal center of the bin in d2
        d3_nom          The nominal center of the bin in d3
        n               The multipole index n
        ==========      ================================================================

        In addition, all bin types include the following columns:

        ==========      ================================================================
        Column          Description
        ==========      ================================================================
        meand1          The mean value :math:`\langle d1\rangle` of triangles that fell
                        into each bin
        meanlogd1       The mean value :math:`\langle \log(d1)\rangle` of triangles that
                        fell into each bin
        meand2          The mean value :math:`\langle d2\rangle` of triangles that fell
                        into each bin
        meanlogd2       The mean value :math:`\langle \log(d2)\rangle` of triangles that
                        fell into each bin
        meand3          The mean value :math:`\langle d3\rangle` of triangles that fell
                        into each bin
        meanlogd3       The mean value :math:`\langle \log(d3)\rangle` of triangles that
                        fell into each bin
        gam0r           The real part of the estimator of :math:`\Gamma_0`
        gam0i           The imag part of the estimator of :math:`\Gamma_0`
        gam1r           The real part of the estimator of :math:`\Gamma_1`
        gam1i           The imag part of the estimator of :math:`\Gamma_1`
        gam2r           The real part of the estimator of :math:`\Gamma_2`
        gam2i           The imag part of the estimator of :math:`\Gamma_2`
        gam3r           The real part of the estimator of :math:`\Gamma_3`
        gam3i           The imag part of the estimator of :math:`\Gamma_3`
        sigma_gam0      The sqrt of the variance estimate of :math:`\Gamma_0`
        sigma_gam1      The sqrt of the variance estimate of :math:`\Gamma_1`
        sigma_gam2      The sqrt of the variance estimate of :math:`\Gamma_2`
        sigma_gam3      The sqrt of the variance estimate of :math:`\Gamma_3`
        weight          The total weight of triangles contributing to each bin.
                        (For LogMultipole, this is split into real and imaginary parts,
                        weight_re and weight_im.)
        ntri            The number of triangles contributing to each bin
        ==========      ================================================================

        If ``sep_units`` was given at construction, then the distances will all be in these units.
        Otherwise, they will be in either the same units as x,y,z (for flat or 3d coordinates) or
        radians (for spherical coordinates).

        Parameters:
            file_name (str):    The name of the file to write to.
            file_type (str):    The type of file to write ('ASCII' or 'FITS').  (default: determine
                                the type automatically from the extension of file_name.)
            precision (int):    For ASCII output catalogs, the desired precision. (default: 4;
                                this value can also be given in the constructor in the config dict.)
            write_patch_results (bool): Whether to write the patch-based results as well.
                                        (default: False)
        """
        self.logger.info('Writing GGG correlations to %s',file_name)
        precision = self.config.get('precision', 4) if precision is None else precision
        name = 'main' if write_patch_results else None
        with make_writer(file_name, precision, file_type, self.logger) as writer:
            self._write(writer, name, write_patch_results)

    @property
    def _write_col_names(self):
        if self.bin_type == 'LogRUV':
            col_names = ['r_nom', 'u_nom', 'v_nom',
                         'meand1', 'meanlogd1', 'meand2', 'meanlogd2',
                         'meand3', 'meanlogd3', 'meanu', 'meanv']
        elif self.bin_type == 'LogSAS':
            col_names = ['d2_nom', 'd3_nom', 'phi_nom',
                         'meand1', 'meanlogd1', 'meand2', 'meanlogd2',
                         'meand3', 'meanlogd3', 'meanphi']
        else:
            col_names = ['d2_nom', 'd3_nom', 'n',
                         'meand1', 'meanlogd1', 'meand2', 'meanlogd2',
                         'meand3', 'meanlogd3']
        col_names += ['gam0r', 'gam0i', 'gam1r', 'gam1i',
                      'gam2r', 'gam2i', 'gam3r', 'gam3i',
                      'sigma_gam0', 'sigma_gam1', 'sigma_gam2', 'sigma_gam3']
        if self.bin_type == 'LogMultipole':
            col_names += ['weight_re', 'weight_im', 'ntri']
        else:
            col_names += ['weight', 'ntri']
        return col_names

    @property
    def _write_data(self):
        if self.bin_type == 'LogRUV':
            data = [ self.rnom, self.u, self.v,
                     self.meand1, self.meanlogd1, self.meand2, self.meanlogd2,
                     self.meand3, self.meanlogd3, self.meanu, self.meanv ]
        elif self.bin_type == 'LogSAS':
            data = [ self.d2nom, self.d3nom, self.phi,
                     self.meand1, self.meanlogd1, self.meand2, self.meanlogd2,
                     self.meand3, self.meanlogd3, self.meanphi ]
        else:
            data = [ self.d2nom, self.d3nom, self.n,
                     self.meand1, self.meanlogd1, self.meand2, self.meanlogd2,
                     self.meand3, self.meanlogd3 ]
        data += [ self.gam0r, self.gam0i, self.gam1r, self.gam1i,
                  self.gam2r, self.gam2i, self.gam3r, self.gam3i,
                  np.sqrt(self.vargam0), np.sqrt(self.vargam1),
                  np.sqrt(self.vargam2), np.sqrt(self.vargam3) ]
        if self.bin_type == 'LogMultipole':
            data += [ self.weightr, self.weighti, self.ntri ]
        else:
            data += [ self.weight, self.ntri ]
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

            The `GGGCorrelation` object should be constructed with the same configuration
            parameters as the one being read.  e.g. the same min_sep, max_sep, etc.  This is not
            checked by the read function.

        Parameters:
            file_name (str):    The name of the file to read in.
            file_type (str):    The type of file ('ASCII' or 'FITS').  (default: determine the type
                                automatically from the extension of file_name.)
        """
        self.logger.info('Reading GGG correlations from %s',file_name)
        with make_reader(file_name, file_type, self.logger) as reader:
            self._read(reader)

    def _read_from_data(self, data, params):
        s = self.data_shape
        self.meand1 = data['meand1'].reshape(s)
        self.meanlogd1 = data['meanlogd1'].reshape(s)
        self.meand2 = data['meand2'].reshape(s)
        self.meanlogd2 = data['meanlogd2'].reshape(s)
        self.meand3 = data['meand3'].reshape(s)
        self.meanlogd3 = data['meanlogd3'].reshape(s)
        if self.bin_type == 'LogRUV':
            self.meanu = data['meanu'].reshape(s)
            self.meanv = data['meanv'].reshape(s)
        elif self.bin_type == 'LogSAS':
            self.meanu = data['meanphi'].reshape(s)
        self.gam0r = data['gam0r'].reshape(s)
        self.gam0i = data['gam0i'].reshape(s)
        self.gam1r = data['gam1r'].reshape(s)
        self.gam1i = data['gam1i'].reshape(s)
        self.gam2r = data['gam2r'].reshape(s)
        self.gam2i = data['gam2i'].reshape(s)
        self.gam3r = data['gam3r'].reshape(s)
        self.gam3i = data['gam3i'].reshape(s)
        # Read old output files without error.
        self._vargam0 = data['sigma_gam0'].reshape(s)**2
        self._vargam1 = data['sigma_gam1'].reshape(s)**2
        self._vargam2 = data['sigma_gam2'].reshape(s)**2
        self._vargam3 = data['sigma_gam3'].reshape(s)**2
        if self.bin_type == 'LogMultipole':
            self.weightr = data['weight_re'].reshape(s)
            self.weighti = data['weight_im'].reshape(s)
        else:
            self.weightr = data['weight'].reshape(s)
        self.ntri = data['ntri'].reshape(s)
        self.coords = params['coords'].strip()
        self.metric = params['metric'].strip()
        self.npatch1 = params.get('npatch1', 1)
        self.npatch2 = params.get('npatch2', 1)
        self.npatch3 = params.get('npatch3', 1)

    @classmethod
    def _calculateT(cls, s, t, k1, k2, k3):
        # First calculate q values:
        q1 = (s+t)/3.
        q2 = q1-t
        q3 = q1-s

        # |qi|^2 shows up a lot, so save these.
        # The a stands for "absolute", and the ^2 part is implicit.
        a1 = np.abs(q1)**2
        a2 = np.abs(q2)**2
        a3 = np.abs(q3)**2
        a123 = a1*a2*a3

        # These combinations also appear multiple times.
        # The b doesn't stand for anything.  It's just the next letter after a.
        b1 = np.conjugate(q1)**2*q2*q3
        b2 = np.conjugate(q2)**2*q1*q3
        b3 = np.conjugate(q3)**2*q1*q2

        if k1==1 and k2==1 and k3==1:

            # Some factors we use multiple times
            expfactor = -np.exp(-(a1 + a2 + a3)/2)

            # JBJ Equation 51
            # Note that we actually accumulate the Gammas with a different choice for
            # alpha_i.  We accumulate the shears relative to the q vectors, not relative to s.
            # cf. JBJ Equation 41 and footnote 3.  The upshot is that we multiply JBJ's formulae
            # by (q1q2q3)^2 / |q1q2q3|^2 for T0 and (q1*q2q3)^2/|q1q2q3|^2 for T1.
            # Then T0 becomes
            # T0 = -(|q1 q2 q3|^2)/24 exp(-(|q1|^2+|q2|^2+|q3|^2)/2)
            T0 = expfactor * a123 / 24

            # JBJ Equation 52
            # After the phase adjustment, T1 becomes:
            # T1 = -[(|q1 q2 q3|^2)/24
            #        - (q1*^2 q2 q3)/9
            #        + (q1*^4 q2^2 q3^2 + 2 |q2 q3|^2 q1*^2 q2 q3)/(|q1 q2 q3|^2)/27
            #       ] exp(-(|q1|^2+|q2|^2+|q3|^2)/2)
            T1 = expfactor * (a123 / 24 - b1 / 9 + (b1**2 + 2*a2*a3*b1) / (a123 * 27))
            T2 = expfactor * (a123 / 24 - b2 / 9 + (b2**2 + 2*a1*a3*b2) / (a123 * 27))
            T3 = expfactor * (a123 / 24 - b3 / 9 + (b3**2 + 2*a1*a2*b3) / (a123 * 27))

        else:
            # SKL Equation 63:
            k1sq = k1*k1
            k2sq = k2*k2
            k3sq = k3*k3
            Theta2 = ((k1sq*k2sq + k1sq*k3sq + k2sq*k3sq)/3.)**0.5
            k1sq /= Theta2   # These are now what SKL calls theta_i^2 / Theta^2
            k2sq /= Theta2
            k3sq /= Theta2
            Theta4 = Theta2*Theta2
            Theta6 = Theta4*Theta2
            S = k1sq * k2sq * k3sq

            # SKL Equation 64:
            Z = ((2*k2sq + 2*k3sq - k1sq) * a1 +
                 (2*k3sq + 2*k1sq - k2sq) * a2 +
                 (2*k1sq + 2*k2sq - k3sq) * a3) / (6*Theta2)
            expfactor = -S * np.exp(-Z) / Theta4

            # SKL Equation 65:
            f1 = (k2sq+k3sq)/2 + (k2sq-k3sq)*(q2-q3)/(6*q1)
            f2 = (k3sq+k1sq)/2 + (k3sq-k1sq)*(q3-q1)/(6*q2)
            f3 = (k1sq+k2sq)/2 + (k1sq-k2sq)*(q1-q2)/(6*q3)
            f1c = np.conjugate(f1)
            f2c = np.conjugate(f2)
            f3c = np.conjugate(f3)

            # SKL Equation 69:
            g1 = k2sq*k3sq + (k3sq-k2sq)*k1sq*(q2-q3)/(3*q1)
            g2 = k3sq*k1sq + (k1sq-k3sq)*k2sq*(q3-q1)/(3*q2)
            g3 = k1sq*k2sq + (k2sq-k1sq)*k3sq*(q1-q2)/(3*q3)
            g1c = np.conjugate(g1)
            g2c = np.conjugate(g2)
            g3c = np.conjugate(g3)

            # SKL Equation 62:
            T0 = expfactor * a123 * f1c**2 * f2c**2 * f3c**2 / (24.*Theta6)

            # SKL Equation 68:
            T1 = expfactor * (
                a123 * f1**2 * f2c**2 * f3c**2 / (24*Theta6) -
                b1 * f1*f2c*f3c*g1c / (9*Theta4) +
                (b1**2 * g1c**2 + 2*k2sq*k3sq*a2*a3*b1 * f2c * f3c) / (a123 * 27*Theta2))
            T2 = expfactor * (
                a123 * f1c**2 * f2**2 * f3c**2 / (24*Theta6) -
                b2 * f1c*f2*f3c*g2c / (9*Theta4) +
                (b2**2 * g2c**2 + 2*k1sq*k3sq*a1*a3*b2 * f1c * f3c) / (a123 * 27*Theta2))
            T3 = expfactor * (
                a123 * f1c**2 * f2c**2 * f3**2 / (24*Theta6) -
                b3 * f1c*f2c*f3*g3c / (9*Theta4) +
                (b3**2 * g3c**2 + 2*k1sq*k2sq*a1*a2*b3 * f1c * f2c) / (a123 * 27*Theta2))

        return T0, T1, T2, T3

    def calculateMap3(self, *, R=None, k2=1, k3=1):
        r"""Calculate the skewness of the aperture mass from the correlation function.

        The equations for this come from Jarvis, Bernstein & Jain (2004, MNRAS, 352).
        See their section 3, especially equations 51 and 52 for the :math:`T_i` functions,
        equations 60 and 61 for the calculation of :math:`\langle \cal M^3 \rangle` and
        :math:`\langle \cal M^2 M^* \rangle`, and equations 55-58 for how to convert
        these to the return values.

        If k2 or k3 != 1, then this routine calculates the generalization of the skewness
        proposed by Schneider, Kilbinger & Lombardi (2005, A&A, 431):
        :math:`\langle M_{ap}^3(R, k_2 R, k_3 R)\rangle` and related values.

        If k2 = k3 = 1 (the default), then there are only 4 combinations of Map and Mx
        that are relevant:

        - map3 = :math:`\langle M_{ap}^3(R)\rangle`
        - map2mx = :math:`\langle M_{ap}^2(R) M_\times(R)\rangle`,
        - mapmx2 = :math:`\langle M_{ap}(R) M_\times(R)\rangle`
        - mx3 = :math:`\langle M_{\rm \times}^3(R)\rangle`

        However, if k2 or k3 != 1, then there are 8 combinations:

        - map3 = :math:`\langle M_{ap}(R) M_{ap}(k_2 R) M_{ap}(k_3 R)\rangle`
        - mapmapmx = :math:`\langle M_{ap}(R) M_{ap}(k_2 R) M_\times(k_3 R)\rangle`
        - mapmxmap = :math:`\langle M_{ap}(R) M_\times(k_2 R) M_{ap}(k_3 R)\rangle`
        - mxmapmap = :math:`\langle M_\times(R) M_{ap}(k_2 R) M_{ap}(k_3 R)\rangle`
        - mxmxmap = :math:`\langle M_\times(R) M_\times(k_2 R) M_{ap}(k_3 R)\rangle`
        - mxmapmx = :math:`\langle M_\times(R) M_{ap}(k_2 R) M_\times(k_3 R)\rangle`
        - mapmxmx = :math:`\langle M_{ap}(R) M_\times(k_2 R) M_\times(k_3 R)\rangle`
        - mx3 = :math:`\langle M_\times(R) M_\times(k_2 R) M_\times(k_3 R)\rangle`

        To accommodate this full generality, we always return all 8 values, along with the
        estimated variance (which is equal for each), even when k2 = k3 = 1.

        .. note::

            The formulae for the ``m2_uform`` = 'Schneider' definition of the aperture mass,
            described in the documentation of `calculateMapSq`, are not known, so that is not an
            option here.  The calculations here use the definition that corresponds to
            ``m2_uform`` = 'Crittenden'.

        Parameters:
            R (array):      The R values at which to calculate the aperture mass statistics.
                            (default: None, which means use self.rnom1d)
            k2 (float):     If given, the ratio R2/R1 in the SKL formulae. (default: 1)
            k3 (float):     If given, the ratio R3/R1 in the SKL formulae. (default: 1)

        Returns:
            Tuple containing:

            - map3 = array of :math:`\langle M_{ap}(R) M_{ap}(k_2 R) M_{ap}(k_3 R)\rangle`
            - mapmapmx = array of :math:`\langle M_{ap}(R) M_{ap}(k_2 R) M_\times(k_3 R)\rangle`
            - mapmxmap = array of :math:`\langle M_{ap}(R) M_\times(k_2 R) M_{ap}(k_3 R)\rangle`
            - mxmapmap = array of :math:`\langle M_\times(R) M_{ap}(k_2 R) M_{ap}(k_3 R)\rangle`
            - mxmxmap = array of :math:`\langle M_\times(R) M_\times(k_2 R) M_{ap}(k_3 R)\rangle`
            - mxmapmx = array of :math:`\langle M_\times(R) M_{ap}(k_2 R) M_\times(k_3 R)\rangle`
            - mapmxmx = array of :math:`\langle M_{ap}(R) M_\times(k_2 R) M_\times(k_3 R)\rangle`
            - mx3 = array of :math:`\langle M_\times(R) M_\times(k_2 R) M_\times(k_3 R)\rangle`
            - varmap3 = array of variance estimates of the above values
        """
        # As in the calculateMapSq function, we Make s and t matrices, so we can eventually do the
        # integral by doing a matrix product.
        if R is None:
            R = self.rnom1d
        else:
            R = np.asarray(R)

        # Pick s = d2, so dlogs is bin_size
        s = d2 = np.outer(1./R, self.meand2.ravel())

        if self.bin_type == 'LogRUV':
            # We take t = d3, but we need the x and y components. (relative to s along x axis)
            # cf. Figure 1 in JBJ.
            # d1^2 = d2^2 + d3^2 - 2 d2 d3 cos(theta1)
            # tx = d3 cos(theta1) = (d2^2 + d3^2 - d1^2)/2d2
            # Simplify this using u=d3/d2 and v=(d1-d2)/d3
            #    = (d3^2 - (d1+d2)(d1-d2)) / 2d2
            #    = d3 (d3 - (d1+d2)v) / 2d2
            #    = d3 (u - (2+uv)v)/2
            #    = d3 (u - 2v - uv^2)/2
            #    = d3 (u(1-v^2)/2 - v)
            # Note that v here is really |v|.  We'll account for the sign of v in ty.
            d3 = np.outer(1./R, self.meand3.ravel())
            d1 = np.outer(1./R, self.meand1.ravel())
            u = self.meanu.ravel()
            v = self.meanv.ravel()
            tx = d3*(0.5*u*(1-v**2) - np.abs(v))
            # This form tends to be more stable near potentially degenerate triangles
            # than tx = (d2*d2 + d3*d3 - d1*d1) / (2*d2)
            # However, add a check to make sure.
            bad = (tx <= -d3) | (tx >= d3)
            if np.any(bad):  # pragma: no cover
                self.logger.warning("Warning: Detected some invalid triangles when computing Map^3")
                self.logger.warning("Excluding these triangles from the integral.")
                self.logger.debug("N bad points = %s",np.sum(bad))
                self.logger.debug("d1[bad] = %s",d1[bad])
                self.logger.debug("d2[bad] = %s",d2[bad])
                self.logger.debug("d3[bad] = %s",d3[bad])
                self.logger.debug("tx[bad] = %s",tx[bad])
            bad = np.where(bad)
            tx[bad] = 0  # for now to avoid nans
            ty = np.sqrt(d3**2 - tx**2)
            ty[:,self.meanv.ravel() > 0] *= -1.
            t = tx + 1j * ty
        else:
            d3 = np.outer(1./R, self.meand3.ravel())
            t = d3 * np.exp(1j * self.meanphi.ravel() * self._phi_units)

        # Next we need to construct the T values.
        T0, T1, T2, T3 = self._calculateT(s,t,1.,k2,k3)

        if self.bin_type == 'LogRUV':
            # Finally, account for the Jacobian in d^2t: jac = |J(tx, ty; u, v)|,
            # since our Gammas are accumulated in s, u, v, not s, tx, ty.
            # u = d3/d2, v = (d1-d2)/d3
            # tx = d3 (u - 2v - uv^2)/2
            #    = s/2 (u^2 - 2uv - u^2v^2)
            # dtx/du = s (u - v - uv^2)
            # dtx/dv = -us (1 + uv)
            # ty = sqrt(d3^2 - tx^2) = sqrt(u^2 s^2 - tx^2)
            # dty/du = s^2 u/2ty (1-v^2) (2 + 3uv - u^2 + u^2v^2)
            # dty/dv = s^2 u^2/2ty (1 + uv) (u - uv^2 - 2uv)
            #
            # After some algebra...
            #
            # J = s^3 u^2 (1+uv) / ty
            #   = d3^2 d1 / ty
            #
            jac = np.abs(d3*d3*d1/ty)
            jac[bad] = 0.  # Exclude any bad triangles from the integral.
            d2t = jac * self.ubin_size * self.vbin_size / (2.*np.pi)
        else:
            # In SAS binning, d2t is easier.
            # We bin directly in ln(d3) and phi, so
            # tx = d3 cos(phi)
            # ty = d3 sin(phi)
            # dtx/dlnd3 = d3 dtx/dd3 = d3 cos(phi)
            # dty/dlnd3 = d3 dty/dd3 = d3 sin(phi)
            # dtx/dphi = -d3 sin(phi)
            # dty/dphi = d3 cos(phi)
            # J(tx,ty; lnd3, phi) = d3^2
            d2t = d3**2 * self.bin_size * self.phi_bin_size / (2*np.pi)

        sds = s * s * self.bin_size  # Remember bin_size is dln(s)
        # Note: these are really d2t/2piR^2 and sds/R^2, which are what actually show up
        # in JBJ equations 45 and 50.

        T0 *= sds * d2t
        T1 *= sds * d2t
        T2 *= sds * d2t
        T3 *= sds * d2t

        # Now do the integral by taking the matrix products.
        gam0 = self.gam0.ravel()
        gam1 = self.gam1.ravel()
        gam2 = self.gam2.ravel()
        gam3 = self.gam3.ravel()
        vargam0 = self.vargam0.ravel()
        vargam1 = self.vargam1.ravel()
        vargam2 = self.vargam2.ravel()
        vargam3 = self.vargam3.ravel()
        mmm = T0.dot(gam0)
        mcmm = T1.dot(gam1)
        mmcm = T2.dot(gam2)
        mmmc = T3.dot(gam3)

        # These accumulate the coefficients that are being dotted to gam0,1,2,3 respectively.
        # Below, we will take the abs^2 and dot it to gam0,1,2,3 in each case to compute the
        # total variance.
        # Note: This assumes that gam0, gam1, gam2, gam3 have no covariance.
        # This is not technically true, but I think it's approximately ok.
        var0 = T0.copy()
        var1 = T1.copy()
        var2 = T2.copy()
        var3 = T3.copy()

        if self.bin_type == 'LogRUV':
            if k2 == 1 and k3 == 1:
                mmm *= 6
                mcmm += mmcm
                mcmm += mmmc
                mcmm *= 2
                mmcm = mmmc = mcmm
                var0 *= 6
                var1 *= 6
                var2 *= 6
                var3 *= 6
            else:
                # Repeat the above for the other permutations
                for (_k1, _k2, _k3, _mcmm, _mmcm, _mmmc) in [
                        (1,k3,k2,mcmm,mmmc,mmcm),
                        (k2,1,k3,mmcm,mcmm,mmmc),
                        (k2,k3,1,mmcm,mmmc,mcmm),
                        (k3,1,k2,mmmc,mcmm,mmcm),
                        (k3,k2,1,mmmc,mmcm,mcmm) ]:
                    T0, T1, T2, T3 = self._calculateT(s,t,_k1,_k2,_k3)
                    T0 *= sds * d2t
                    T1 *= sds * d2t
                    T2 *= sds * d2t
                    T3 *= sds * d2t
                    # Relies on numpy array overloading += to actually update in place.
                    mmm += T0.dot(gam0)
                    _mcmm += T1.dot(gam1)
                    _mmcm += T2.dot(gam2)
                    _mmmc += T3.dot(gam3)
                    var0 += T0
                    var1 += T1
                    var2 += T2
                    var3 += T3
        else:
            # SAS binning counts each triangle with each vertex in the c1 position.
            # Just need to account for the cases where 1-2-3 are clockwise, rather than CCW.
            if k2 == 1 and k3 == 1:
                mmm *= 2
                mcmm *= 2
                mmcm += mmmc
                mmmc = mmcm
                var0 *= 2
                var1 *= 2
                var2 *= 2
                var3 *= 2
            else:
                # Repeat the above with 2,3 swapped.
                T0, T1, T2, T3 = self._calculateT(s,t,1,k3,k2)
                T0 *= sds * d2t
                T1 *= sds * d2t
                T2 *= sds * d2t
                T3 *= sds * d2t
                mmm += T0.dot(gam0)
                mcmm += T1.dot(gam1)
                mmmc += T2.dot(gam2)
                mmcm += T3.dot(gam3)
                var0 += T0
                var1 += T1
                var2 += T2
                var3 += T3

        map3 = 0.25 * np.real(mcmm + mmcm + mmmc + mmm)
        mapmapmx = 0.25 * np.imag(mcmm + mmcm - mmmc + mmm)
        mapmxmap = 0.25 * np.imag(mcmm - mmcm + mmmc + mmm)
        mxmapmap = 0.25 * np.imag(-mcmm + mmcm + mmmc + mmm)
        mxmxmap = 0.25 * np.real(mcmm + mmcm - mmmc - mmm)
        mxmapmx = 0.25 * np.real(mcmm - mmcm + mmmc - mmm)
        mapmxmx = 0.25 * np.real(-mcmm + mmcm + mmmc - mmm)
        mx3 = 0.25 * np.imag(mcmm + mmcm + mmmc - mmm)

        var0 /= 4
        var1 /= 4
        var2 /= 4
        var3 /= 4

        # Now finally add up the coefficient squared times each vargam element.
        var = np.abs(var0**2).dot(vargam0)
        var += np.abs(var1**2).dot(vargam1)
        var += np.abs(var2**2).dot(vargam2)
        var += np.abs(var3**2).dot(vargam3)

        return map3, mapmapmx, mapmxmap, mxmapmap, mxmxmap, mxmapmx, mapmxmx, mx3, var

    def writeMap3(self, file_name, *, R=None, file_type=None, precision=None):
        r"""Write the aperture mass skewness based on the correlation function to the
        file, file_name.

        The output file will include the following columns:

        ==========      ==========================================================
        Column          Description
        ==========      ==========================================================
        R               The aperture radius
        Map3            An estimate of :math:`\langle M_{ap}^3\rangle(R)`
                        (cf. `calculateMap3`)
        Map2Mx          An estimate of :math:`\langle M_{ap}^2 M_\times\rangle(R)`
        MapMx2          An estimate of :math:`\langle M_{ap} M_\times^2\rangle(R)`
        Mx3             An estimate of :math:`\langle M_\times^3\rangle(R)`
        sig_map         The sqrt of the variance estimate of each of these
        ==========      ==========================================================

        Parameters:
            file_name (str):    The name of the file to write to.
            R (array):          The R values at which to calculate the statistics.
                                (default: None, which means use self.rnom)
            file_type (str):    The type of file to write ('ASCII' or 'FITS').  (default: determine
                                the type automatically from the extension of file_name.)
            precision (int):    For ASCII output catalogs, the desired precision. (default: 4;
                                this value can also be given in the constructor in the config dict.)
        """
        self.logger.info('Writing Map^3 from GGG correlations to %s',file_name)

        if R is None:
            R = self.rnom1d
        stats = self.calculateMap3(R=R)
        if precision is None:
            precision = self.config.get('precision', 4)

        col_names = ['R','Map3','Map2Mx', 'MapMx2', 'Mx3','sig_map']
        columns = [ R, stats[0], stats[1], stats[4], stats[7], np.sqrt(stats[8]) ]
        with make_writer(file_name, precision, file_type, logger=self.logger) as writer:
            writer.write(col_names, columns)
