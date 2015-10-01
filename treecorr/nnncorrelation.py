# Copyright (c) 2003-2015 by Mike Jarvis
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

import treecorr
import numpy

# Start by loading up the relevant C functions using ctypes
import ctypes
import os

# The numpy version of this function tries to be more portable than the native
# ctypes.cdll.LoadLibary or cdtypes.CDLL functions.
_treecorr = numpy.ctypeslib.load_library('_treecorr',os.path.dirname(__file__))

# some useful aliases
cint = ctypes.c_int
cdouble = ctypes.c_double
cdouble_ptr = ctypes.POINTER(cdouble)
cvoid_ptr = ctypes.c_void_p

_treecorr.BuildNNNCorr.restype = cvoid_ptr
_treecorr.BuildNNNCorr.argtypes = [
    cdouble, cdouble, cint, cdouble, cdouble,
    cdouble, cdouble, cint, cdouble, cdouble,
    cdouble, cdouble, cint, cdouble, cdouble,
    cdouble_ptr, cdouble_ptr, cdouble_ptr, cdouble_ptr,
    cdouble_ptr, cdouble_ptr, cdouble_ptr, cdouble_ptr,
    cdouble_ptr, cdouble_ptr ]
_treecorr.DestroyNNNCorr.argtypes = [ cvoid_ptr ]
_treecorr.ProcessAutoNNNFlat.argtypes = [ cvoid_ptr, cvoid_ptr, cint ]
_treecorr.ProcessAutoNNN3D.argtypes = [ cvoid_ptr, cvoid_ptr, cint ]
_treecorr.ProcessAutoNNNPerp.argtypes = [ cvoid_ptr, cvoid_ptr, cint ]
_treecorr.ProcessCrossNNNFlat.argtypes = [ cvoid_ptr, cvoid_ptr, cvoid_ptr, cvoid_ptr, cint ]
_treecorr.ProcessCrossNNN3D.argtypes = [ cvoid_ptr, cvoid_ptr, cvoid_ptr, cvoid_ptr, cint ]
_treecorr.ProcessCrossNNNPerp.argtypes = [ cvoid_ptr, cvoid_ptr, cvoid_ptr, cvoid_ptr, cint ]


class NNNCorrelation(treecorr.BinnedCorr3):
    """This class handles the calculation and storage of a 2-point count-count correlation
    function.  i.e. the regular density correlation function.

    See the doc string of :class:`~treecorr.BinnedCorr3` for a description of how the triangles
    are binned.
    
    Ojects of this class holds the following attributes:

        :nbins:     The number of bins in logr where r = d2
        :bin_size:  The size of the bins in logr
        :min_sep:   The minimum separation being considered
        :max_sep:   The maximum separation being considered
        :nubins:    The number of bins in u where u = d3/d2
        :ubin_size: The size of the bins in u
        :min_u:     The minimum u being considered
        :max_u:     The maximum u being considered
        :nvbins:    The number of bins in v where v = +-(d1-d2)/d3
        :vbin_size: The size of the bins in v
        :min_v:     The minimum v being considered
        :max_v:     The maximum v being considered
        :logr1d:    The nominal centers of the nbins bins in log(r).
        :u1d:       The nominal centers of the nubins bins in u.
        :v1d:       The nominal centers of the nvbins bins in v.

    In addition, the following attributes are numpy arrays whose shape is (nbins, nubins, nvbins):

        :logr:      The nominal center of the bin in log(r).
        :u:         The nominal center of the bin in u.
        :v:         The nominal center of the bin in v.
        :meand1:    The (weighted) mean value of d1 for the triangles in each bin.
        :meanlogd1: The mean value of log(d1) for the triangles in each bin.
        :meand2:    The (weighted) mean value of d2 (aka r) for the triangles in each bin.
        :meanlogd2: The mean value of log(d2) for the triangles in each bin.
        :meand2:    The (weighted) mean value of d3 for the triangles in each bin.
        :meanlogd2: The mean value of log(d3) for the triangles in each bin.
        :meanu:     The mean value of u for the triangles in each bin.
        :meanv:     The mean value of v for the triangles in each bin.
        :weight:    The total weight in each bin.
        :ntri:      The number of triangles in each bin.
        :tot:       The total number of triangles processed, which is used to normalize
                    the randoms if they have a different number of triangles.

    If sep_units are given (either in the config dict or as a named kwarg) then logr and meanlogr
    both take r to be in these units.  i.e. exp(logr) will have R in units of sep_units.

    The usage pattern is as follows:

        >>> nnn = treecorr.NNNCorrelation(config)
        >>> nnn.process(cat)         # For auto-correlation.
        >>> nnn.process(cat1,cat2,cat3)   # For cross-correlation.
        >>> rrr.process...           # Likewise for random-random correlations
        >>> drr.process...           # If desired, also do data-random correlations
        >>> rdr.process...           # ... in all three
        >>> rrd.process...           # ... permutations
        >>> rdd.process...           # Also with two data and one random
        >>> drd.process...           # ... in all three
        >>> ddr.process...           # ... permutations
        >>> nn.write(file_name,rrr,drr,...)  # Write out to a file.
        >>> zeta,varzeta = nn.calculateZeta(rrr,drr,...)  # Or get the 3pt function directly.

    :param config:      A configuration dict that can be used to pass in kwargs if desired.
                        This dict is allowed to have addition entries in addition to those listed
                        in :class:`~treecorr.BinnedCorr3`, which are ignored here. (default: None)
    :param logger:      If desired, a logger object for logging. (default: None, in which case
                        one will be built according to the config dict's verbose level.)

    See the documentation for :class:`~treecorr.BinnedCorr3` for the list of other allowed kwargs,
    which may be passed either directly or in the config dict.
    """
    def __init__(self, config=None, logger=None, **kwargs):
        treecorr.BinnedCorr3.__init__(self, config, logger, **kwargs)

        shape = (self.nbins, self.nubins, self.nvbins)
        self.meand1 = numpy.zeros(shape, dtype=float)
        self.meanlogd1 = numpy.zeros(shape, dtype=float)
        self.meand2 = numpy.zeros(shape, dtype=float)
        self.meanlogd2 = numpy.zeros(shape, dtype=float)
        self.meand3 = numpy.zeros(shape, dtype=float)
        self.meanlogd3 = numpy.zeros(shape, dtype=float)
        self.meanu = numpy.zeros(shape, dtype=float)
        self.meanv = numpy.zeros(shape, dtype=float)
        self.weight = numpy.zeros(shape, dtype=float)
        self.ntri = numpy.zeros(shape, dtype=float)
        self.tot = 0.
        self._build_corr()
        self.logger.debug('Finished building NNNCorr')

    def _build_corr(self):
        meand1 = self.meand1.ctypes.data_as(cdouble_ptr)
        meanlogd1 = self.meanlogd1.ctypes.data_as(cdouble_ptr)
        meand2 = self.meand2.ctypes.data_as(cdouble_ptr)
        meanlogd2 = self.meanlogd2.ctypes.data_as(cdouble_ptr)
        meand3 = self.meand3.ctypes.data_as(cdouble_ptr)
        meanlogd3 = self.meanlogd3.ctypes.data_as(cdouble_ptr)
        meanu = self.meanu.ctypes.data_as(cdouble_ptr)
        meanv = self.meanv.ctypes.data_as(cdouble_ptr)
        weight = self.weight.ctypes.data_as(cdouble_ptr)
        ntri = self.ntri.ctypes.data_as(cdouble_ptr)
        self.corr = _treecorr.BuildNNNCorr(
                self.min_sep,self.max_sep,self.nbins,self.bin_size,self.b,
                self.min_u,self.max_u,self.nubins,self.ubin_size,self.bu,
                self.min_v,self.max_v,self.nvbins,self.vbin_size,self.bv,
                meand1, meanlogd1, meand2, meanlogd2, meand3, meanlogd3, meanu, meanv, 
                weight, ntri);

    def __del__(self):
        # Using memory allocated from the C layer means we have to explicitly deallocate it
        # rather than being able to rely on the Python memory manager.
        if hasattr(self,'corr'):    # In case __init__ failed to get that far
            _treecorr.DestroyNNNCorr(self.corr)

    def copy(self):
        import copy
        return copy.deepcopy(self)

    def __getstate__(self):
        d = self.__dict__.copy()
        del d['corr']
        del d['logger']  # Oh well.  This is just lost in the copy.  Can't be pickled.
        return d

    def __setstate__(self, d):
        self.__dict__ = d
        self._build_corr()
        self.logger = treecorr.config.setup_logger(
                treecorr.config.get(self.config,'verbose',int,0),
                self.config.get('log_file',None))

    def __repr__(self):
        return 'NNNCorrelation(config=%r)'%self.config

    def process_auto(self, cat, metric=None, num_threads=None):
        """Process a single catalog, accumulating the auto-correlation.

        This accumulates the auto-correlation for the given catalog.  After
        calling this function as often as desired, the finalize() command will
        finish the calculation of meand1, meanlogd1, etc.

        :param cat:         The catalog to process
        :param metric:      Which metric to use.  See :meth:`~treecorr.NNNCorrelation.process` for 
                            details.  (default: 'Euclidean'; this value can also be given in the 
                            constructor in the config dict.)
        :param num_threads: How many OpenMP threads to use during the calculation.  
                            (default: use the number of cpu cores; this value can also be given in
                            the constructor in the config dict.) Note that this won't work if the 
                            system's C compiler is clang prior to version 3.7.
        """
        if cat.name == '':
            self.logger.info('Starting process NNN auto-correlations')
        else:
            self.logger.info('Starting process NNN auto-correlations for cat %s.', cat.name)

        if metric is None:
            metric = treecorr.config.get(self.config,'metric',str,'Euclidean')
        if metric not in ['Euclidean', 'Rperp']:
            raise ValueError("Invalid metric.")

        self._set_num_threads(num_threads)

        min_size = self.min_sep * self.min_u
        max_size = 2.*self.max_sep 
        b = numpy.max( (self.b, self.bu, self.bv) )
        field = cat.getNField(min_size,max_size,b,self.split_method,metric,self.max_top)

        self.logger.info('Starting %d jobs.',field.nTopLevelNodes)
        if field.flat:
            _treecorr.ProcessAutoNNNFlat(self.corr, field.data, self.output_dots)
        elif field.perp:
            _treecorr.ProcessAutoNNNPerp(self.corr, field.data, self.output_dots)
        else:
            _treecorr.ProcessAutoNNN3D(self.corr, field.data, self.output_dots)
        self.tot += (1./6.) * cat.sumw**3

    def process_cross21(self, cat1, cat2, metric=None, num_threads=None):
        """Process two catalogs, accumulating the 3pt cross-correlation, where two of the 
        points in each triangle come from the first catalog, and one from the second.

        This accumulates the cross-correlation for the given catalogs.  After
        calling this function as often as desired, the finalize() command will
        finish the calculation of meand1, meanlogd1, etc.

        :param cat1:        The first catalog to process
        :param cat2:        The second catalog to process
        :param metric:      Which metric to use.  See :meth:`~treecorr.NNNCorrelation.process` for 
                            details.  (default: 'Euclidean'; this value can also be given in the 
                            constructor in the config dict.)
        :param num_threads: How many OpenMP threads to use during the calculation.  
                            (default: use the number of cpu cores; this value can also be given in
                            the constructor in the config dict.) Note that this won't work if the 
                            system's C compiler is clang prior to version 3.7.
        """
        raise NotImplemented("No partial cross NNN yet.")


    def process_cross(self, cat1, cat2, cat3, metric=None, num_threads=None):
        """Process a set of three catalogs, accumulating the 3pt cross-correlation.

        This accumulates the cross-correlation for the given catalogs.  After
        calling this function as often as desired, the finalize() command will
        finish the calculation of meand1, meanlogd1, etc.

        :param cat1:        The first catalog to process
        :param cat2:        The second catalog to process
        :param cat3:        The third catalog to process
        :param metric:      Which metric to use.  See :meth:`~treecorr.NNNCorrelation.process` for 
                            details.  (default: 'Euclidean'; this value can also be given in the 
                            constructor in the config dict.)
        :param num_threads: How many OpenMP threads to use during the calculation.  
                            (default: use the number of cpu cores; this value can also be given in
                            the constructor in the config dict.) Note that this won't work if the 
                            system's C compiler is clang prior to version 3.7.
        """
        if cat1.name == '' and cat2.name == '' and cat3.name == '':
            self.logger.info('Starting process NNN cross-correlations')
        else:
            self.logger.info('Starting process NNN cross-correlations for cats %s, %s, %s.',
                             cat1.name, cat2.name, cat3.name)

        if metric is None:
            metric = treecorr.config.get(self.config,'metric',str,'Euclidean')
        if metric not in ['Euclidean', 'Rperp']:
            raise ValueError("Invalid metric.")
        if cat1.coords != cat2.coords or cat1.coords != cat3.coords:
            raise AttributeError("Cannot correlate catalogs with different coordinate systems.")

        self._set_num_threads(num_threads)

        min_size = self.min_sep * self.min_u
        max_size = 2.*self.max_sep 
        b = numpy.max( (self.b, self.bu, self.bv) )
        f1 = cat1.getNField(min_size,max_size,b,self.split_method,metric,self.max_top)
        f2 = cat2.getNField(min_size,max_size,b,self.split_method,metric,self.max_top)
        f3 = cat3.getNField(min_size,max_size,b,self.split_method,metric,self.max_top)

        self.logger.info('Starting %d jobs.',f1.nTopLevelNodes)
        if f1.flat:
            _treecorr.ProcessCrossNNNFlat(self.corr, f1.data, f2.data, f3.data, self.output_dots)
        elif f1.perp:
            _treecorr.ProcessCrossNNNPerp(self.corr, f1.data, f2.data, f3.data, self.output_dots)
        else:
            _treecorr.ProcessCrossNNN3D(self.corr, f1.data, f2.data, f3.data, self.output_dots)
        self.tot += cat1.sumw * cat2.sumw * cat3.sumw / 6.0


    def finalize(self):
        """Finalize the calculation of meand1, meanlogd1, etc.

        The process_auto and process_cross commands accumulate values in each bin,
        so they can be called multiple times if appropriate.  Afterwards, this command
        finishes the calculation of meanlogr, meanu, meanv by dividing by the total weight.
        """
        mask1 = self.weight != 0
        mask2 = self.weight == 0

        self.meand1[mask1] /= self.weight[mask1]
        self.meanlogd1[mask1] /= self.weight[mask1]
        self.meand2[mask1] /= self.weight[mask1]
        self.meanlogd2[mask1] /= self.weight[mask1]
        self.meand3[mask1] /= self.weight[mask1]
        self.meanlogd3[mask1] /= self.weight[mask1]
        self.meanu[mask1] /= self.weight[mask1]
        self.meanv[mask1] /= self.weight[mask1]

        # Update the units
        self.meand1[mask1] /= self.sep_units
        self.meanlogd1[mask1] -= self.log_sep_units
        self.meand2[mask1] /= self.sep_units
        self.meanlogd2[mask1] -= self.log_sep_units
        self.meand3[mask1] /= self.sep_units
        self.meanlogd3[mask1] -= self.log_sep_units

        # Use meanlogr when available, but set to nominal when no triangles in bin.
        self.meand2[mask2] = numpy.exp(self.logr[mask2])
        self.meanlogd2[mask2] = self.logr[mask2]
        self.meanu[mask2] = self.u[mask2]
        self.meanv[mask2] = self.v[mask2]
        self.meand3[mask2] = self.u[mask2] * self.meand2[mask2]
        self.meanlogd3[mask2] = numpy.log(self.meand3[mask2])
        self.meand1[mask2] = self.v[mask2] * self.meand3[mask2] + self.meand2[mask2]
        self.meanlogd1[mask2] = numpy.log(self.meand1[mask2])


    def clear(self):
        """Clear the data vectors
        """
        self.meand1[:,:,:] = 0.
        self.meanlogd1[:,:,:] = 0.
        self.meand2[:,:,:] = 0.
        self.meanlogd2[:,:,:] = 0.
        self.meand3[:,:,:] = 0.
        self.meanlogd3[:,:,:] = 0.
        self.meanu[:,:,:] = 0.
        self.meanv[:,:,:] = 0.
        self.weight[:,:,:] = 0.
        self.ntri[:,:,:] = 0.
        self.tot = 0.

    def __iadd__(self, other):
        """Add a second NNNCorrelation's data to this one.

        Note: For this to make sense, both Correlation objects should have been using
        process_auto and/or process_cross, and they should not have had finalize called yet.
        Then, after adding them together, you should call finalize on the sum.
        """
        if not isinstance(other, NNNCorrelation):
            raise AttributeError("Can only add another NNNCorrelation object")
        if not (self.nbins == other.nbins and
                self.min_sep == other.min_sep and
                self.max_sep == other.max_sep and
                self.nubins == other.nubins and
                self.min_u == other.min_u and
                self.max_u == other.max_u and
                self.nvbins == other.nvbins and
                self.min_v == other.min_v and
                self.max_v == other.max_v):
            raise ValueError("NNNCorrelation to be added is not compatible with this one.")

        self.meand1[:] += other.meand1[:]
        self.meanlogd1[:] += other.meanlogd1[:]
        self.meand2[:] += other.meand2[:]
        self.meanlogd2[:] += other.meanlogd2[:]
        self.meand3[:] += other.meand3[:]
        self.meanlogd3[:] += other.meanlogd3[:]
        self.meanu[:] += other.meanu[:]
        self.meanv[:] += other.meanv[:]
        self.weight[:] += other.weight[:]
        self.ntri[:] += other.ntri[:]
        self.tot += other.tot
        return self


    def process(self, cat1, cat2=None, cat3=None, metric=None, num_threads=None):
        """Accumulate the number of triangles of points between cat1, cat2, and cat3.

        - If only 1 argument is given, then compute an auto-correlation function.
        - If 2 arguments are given, then compute a cross-correlation function with the 
          first catalog taking two corners of the triangles. (Not implemented yet.)
        - If 3 arguments are given, then compute a cross-correlation function.

        All arguments may be lists, in which case all items in the list are used 
        for that element of the correlation.

        Note: For a correlation of multiple catalogs, it matters which corner of the
        triangle comes from which catalog.  The final accumulation will have 
        d1 > d2 > d3 where d1 is between two points in cat2,cat3; d2 is between 
        points in cat1,cat3; and d3 is between points in cat1,cat2.  To accumulate
        all the possible triangles between three catalogs, you should call this
        multiple times with the different catalogs in different positions.

        :param cat1:        A catalog or list of catalogs for the first N field.
        :param cat2:        A catalog or list of catalogs for the second N field, if any.
                            (default: None)
        :param cat3:        A catalog or list of catalogs for the third N field, if any.
                            (default: None)
        :param metric:      Which metric to use for distance measurements.  Options are:

                            - 'Euclidean' = straight line Euclidean distance between two points.
                              For spherical coordinates (ra,dec without r), this is the chord
                              distance between points on the unit sphere.
                            - 'Rperp' = the perpendicular component of the distance. For two points
                              with distance from Earth `r1, r2`, if `d` is the normal Euclidean 
                              distance and :math:`Rparallel = |r1-r2|`, then we define
                              :math:`Rperp^2 = d^2 - Rparallel^2`.

                            (default: 'Euclidean'; this value can also be given in the constructor
                            in the config dict.)

        :param num_threads: How many OpenMP threads to use during the calculation.  
                            (default: use the number of cpu cores; this value can also be given in
                            the constructor in the config dict.) Note that this won't work if the 
                            system's C compiler is clang prior to version 3.7.
        """
        self.clear()
        if not isinstance(cat1,list): cat1 = [cat1]
        if cat2 is not None and not isinstance(cat2,list): cat2 = [cat2]
        if cat3 is not None and not isinstance(cat3,list): cat3 = [cat3]
        if len(cat1) == 0:
            raise ValueError("No catalogs provided for cat1")
        if cat2 is not None and len(cat2) == 0:
            cat2 = None
        if cat3 is not None and len(cat3) == 0:
            cat3 = None
        if cat2 is None and cat3 is not None:
            raise NotImplemented("No partial cross NNN yet.")
        if cat3 is None and cat2 is not None:
            raise NotImplemented("No partial cross NNN yet.")

        if cat2 is None and cat3 is None:
            self._process_all_auto(cat1, metric, num_threads)
        else:
            assert cat2 is not None and cat3 is not None
            self._process_all_cross(cat1,cat2,cat3, metric, num_threads)
        self.finalize()

    def calculateZeta(self, rrr, drr=None, rdr=None, rrd=None,
                      ddr=None, drd=None, rdd=None):
        """Calculate the 3pt function given another 3pt function of random
        points using the same mask, and possibly cross correlations of the data and random.

        There are two possible formulae that are currently supported.

        1. The simplest formula to use is :math:`\\zeta^\\prime = (DDD-RRR)/RRR`.
           In this case, only rrr needs to be given, the NNNCorrelation of a random field.
           However, note that in this case, the return value is not normally called :math:`\\zeta`.
           Rather, this is an estimator of 

           .. math::
               \\zeta^\\prime(d1,d2,d3) = \\zeta(d1,d2,d3) + \\xi(d1) + \\xi(d2) + \\xi(d3)

           where :math:`\\xi` is the two-point correlation function for each leg of the triangle.
           You would typically want to calculate that separately and subtract off the
           two-point contributions.

        2. For auto-correlations, a better formula is 
           :math:`\\zeta = (DDD-DDR-DRD-RDD+DRR+RDR+RRD-RRR)/RRR`.
           In this case, DDR, etc. calculate the triangles where two points come from either
           the data or the randoms.  In the case of DDR for instance, points x1 and x2 on the 
           triangle have data and x3 uses randoms, where points x1, x2, x3 are opposite sides 
           d1, d2, d3 with d1 > d2 > d3 as usual; RDR has randoms at 1,3 and data at 2, etc.

        - If only rrr is provided, the first formula will be used.
        - If all of rrr, drr, rdr, rrd, ddr, drd, rdd are provided then the second will be used.

        :param rrr:         An NNCorrelation object for the random field.
        :param drr:         DRR if desired. (default: None)
        :param rdr:         RDR if desired. (default: None)
        :param rrd:         RRD if desired. (default: None)
        :param ddr:         DDR if desired. (default: None)
        :param drd:         DRD if desired. (default: None)
        :param rdd:         RDD if desired. (default: None)

        :returns:           (zeta, varzeta) as a tuple
        """
        # Each random ntri value needs to be rescaled by the ratio of total possible tri.
        if rrr.tot == 0:
            raise RuntimeError("rrr has tot=0.")

        if (drr is not None or rdr is not None or rrd is not None or
            ddr is not None or drd is not None or rdd is not None):
            if (drr is None or rdr is None or rrd is None or 
                ddr is None or drd is None or rrd is None):
                raise AttributeError("Must provide all 6 combinations rdr, drr, etc.")

        rrrw = self.tot / rrr.tot
        if drr is None:
            zeta = (self.weight - rrr.weight * rrrw)
        else:
            if rrd.tot == 0:
                raise RuntimeError("rrd has tot=0.")
            if ddr.tot == 0:
                raise RuntimeError("ddr has tot=0.")
            if rdr.tot == 0:
                raise RuntimeError("rdr has tot=0.")
            if drr.tot == 0:
                raise RuntimeError("drr has tot=0.")
            if drd.tot == 0:
                raise RuntimeError("drd has tot=0.")
            if rdd.tot == 0:
                raise RuntimeError("rdd has tot=0.")
            rrdw = self.tot / rrd.tot
            ddrw = self.tot / ddr.tot
            rdrw = self.tot / rdr.tot
            drrw = self.tot / drr.tot
            drdw = self.tot / drd.tot
            rddw = self.tot / rdd.tot
            zeta = (self.weight + rrd.weight * rrdw + rdr.weight * rdrw + drr.weight * drrw
                    - ddr.weight * ddrw - drd.weight * drdw - rdd.weight * rddw - rrr.weight * rrrw)
        if numpy.any(rrr.weight == 0):
            self.logger.warn("Warning: Some bins for the randoms had no triangles.")
            self.logger.warn("         Probably max_sep is larger than your field.")
        mask1 = rrr.weight != 0
        mask2 = rrr.weight == 0
        zeta[mask1] /= (rrr.weight[mask1] * rrrw)
        zeta[mask2] = 0

        varzeta = numpy.zeros_like(rrr.weight)
        varzeta[mask1] = 1./ (rrr.weight[mask1] * rrrw)

        return zeta, varzeta


    def write(self, file_name, rrr=None, drr=None, rdr=None, rrd=None,
              ddr=None, drd=None, rdd=None, file_type=None, prec=None):
        """Write the correlation function to the file, file_name.

        Normally, at least rrr should be provided, but if this is None, then only the 
        basic accumulated number of triangles are output (along with the columns parametrizing
        the size and shape of the triangles).

        If at least rrr is given, then it will output an estimate of the final 3pt correlation
        function, :math:`\\zeta`. There are two possible formulae that are currently supported.

        1. The simplest formula to use is :math:`\\zeta^\\prime = (DDD-RRR)/RRR`.
           In this case, only rrr needs to be given, the NNNCorrelation of a random field.
           However, note that in this case, the return value is not what is normally called
           :math:`\\zeta`.  Rather, this is an estimator of

           .. math::
               \\zeta^\\prime(d1,d2,d3) = \\zeta(d1,d2,d3) + \\xi(d1) + \\xi(d2) + \\xi(d3)

           where :math:`\\xi` is the two-point correlation function for each leg of the triangle.
           You would typically want to calculate that separately and subtract off the
           two-point contributions.

        2. For auto-correlations, a better formula is 
           :math:`\\zeta = (DDD-DDR-DRD-RDD+DRR+RDR+RRD-RRR)/RRR`.
           In this case, DDR, etc. calculate the triangles where two points come from either
           the data or the randoms.  In the case of DDR for instance, points x1 and x2 on the 
           triangle have data and x3 uses randoms, where points x1, x2, x3 are opposite sides 
           d1, d2, d3 with d1 > d2 > d3 as usual; RDR has randoms at 1,3 and data at 2, etc.
           For this case, all the combinations rrr, drr, rdr, rrd, ddr, drd, rdd must be 
           provided.

        The output file will include the following columns:

            :R_nom:         The nominal center of the bin in R = d2 where d1 > d2 > d3.
            :u_nom:         The nominal center of the bin in u = d3/d2.
            :v_nom:         The nominal center of the bin in v = +-(d1-d2)/d3.
            :meand1:        The mean value :math:`\\langle d1\\rangle` of triangles that fell
                            into each bin.
            :meanlogd1:     The mean value :math:`\\langle logd1\\rangle` of triangles that fell
                            into each bin.
            :meand2:        The mean value :math:`\\langle d2\\rangle` of triangles that fell 
                            into each bin.
            :meanlogd2:     The mean value :math:`\\langle logd2\\rangle` of triangles that fell
                            into each bin.
            :meand3:        The mean value :math:`\\langle d3\\rangle` of triangles that fell
                            into each bin.
            :meanlogd3:     The mean value :math:`\\langle logd3\\rangle` of triangles that fell
                            into each bin.
            :meanu:         The mean value :math:`\\langle u\\rangle` of triangles that fell
                            into each bin.
            :meanv:         The mean value :math:`\\langle v\\rangle` of triangles that fell
                            into each bin.

        Then if rrr is None:

            :DDD:           The total weight of triangles in each bin.
            :ntri:          The total number of triangles in each bin.

        If rrr is given, but not the cross-correlations:

            :zeta:          The estimator :math:`\\zeta^\\prime = (DDD-RRR)/RRR`, which is really
                            :math:`\\zeta(d1,d2,d3) + \\xi(d1) + \\xi(d2) + \\xi(d3)`.  
                            cf. :meth:`~treecorr.NNNCorrelation.calculateZeta`
            :sigma_zeta:    The sqrt of the variance estimate of :math:`\\zeta`.
            :DDD:           The total weight of data triangles (aka DDD) in each bin.
            :RRR:           The total weight of random triangles (aka RRR) in each bin.
            :ntri:          The number of triangles contributing to each bin.

        If all cross-correlations are given:

            :zeta:          The estimator :math:`\\zeta = (DDD-DDR-DRD-RDD+DRR+RDR+RRD-RRR)/RRR`.
            :sigma_zeta:    The sqrt of the variance estimate of :math:`\\zeta`.
            :DDD:           The total weight of DDD triangles in each bin.
            :RRR:           The total weight of RRR triangles in each bin.
            :DRR:           The total weight of DRR triangles in each bin.
            :RDR:           The total weight of RDR triangles in each bin.
            :RRD:           The total weight of RRD triangles in each bin.
            :DDR:           The total weight of DDR triangles in each bin.
            :DRD:           The total weight of DRD triangles in each bin.
            :RDD:           The total weight of RDD triangles in each bin.
            :ntri:          The number of triangles contributing to each bin.


        :param file_name:   The name of the file to write to.
        :param rrr:         An NNNCorrelation object for the random field. (default: None)
        :param drr:         DRR if desired. (default: None)
        :param rdr:         RDR if desired. (default: None)
        :param rrd:         RRD if desired. (default: None)
        :param ddr:         DDR if desired. (default: None)
        :param drd:         DRD if desired. (default: None)
        :param rdd:         RDD if desired. (default: None)
        :param file_type:   The type of file to write ('ASCII' or 'FITS').  (default: determine
                            the type automatically from the extension of file_name.)
        :param prec:        For ASCII output catalogs, the desired precision. (default: 4;
                            this value can also be given in the constructor in the config dict.)
        """
        self.logger.info('Writing NNN correlations to %s',file_name)
        
        col_names = [ 'R_nom', 'u_nom', 'v_nom', 'meand1', 'meanlogd1', 'meand2', 'meanlogd2',
                      'meand3', 'meanlogd3', 'meanu', 'meanv' ]
        columns = [ numpy.exp(self.logr), self.u, self.v,
                    self.meand1, self.meanlogd1, self.meand2, self.meanlogd2,
                    self.meand3, self.meanlogd3, self.meanu, self.meanv ]
        if rrr is None:
            if (drr is not None or rdr is not None or rrd is not None or
                ddr is not None or drd is not None or rdd is not None):
                raise AttributeError("rrr must be provided if other combinations are not None")
            col_names += [ 'DDD', 'ntri' ]
            columns += [ self.weight, self.ntri ]
        else:
            # This will check for other invalid combinations of rrr, drr, etc.
            zeta, varzeta = self.calculateZeta(rrr,drr,rdr,rrd,ddr,drd,rdd)

            col_names += [ 'zeta','sigma_zeta','DDD','RRR' ]
            columns += [ zeta, numpy.sqrt(varzeta),
                         self.weight, rrr.weight * (self.tot/rrr.tot) ]

            if drr is not None:
                col_names += ['DRR','RDR','RRD','DDR','DRD','RDD']
                columns += [ drr.weight * (self.tot/drr.tot), rdr.weight * (self.tot/rdr.tot),
                             rrd.weight * (self.tot/rrd.tot), ddr.weight * (self.tot/ddr.tot),
                             drd.weight * (self.tot/drd.tot), rdd.weight * (self.tot/rdd.tot) ]
            col_names += [ 'ntri' ]
            columns += [ self.ntri ]

        if prec is None:
            prec = self.config.get('precision', 4)

        treecorr.util.gen_write(
            file_name, col_names, columns, prec=prec, file_type=file_type, logger=self.logger)


    def read(self, file_name, file_type=None):
        """Read in values from a file.

        This should be a file that was written by TreeCorr, preferably a FITS file, so there
        is no loss of information.

        Warning: The NNNCorrelation object should be constructed with the same configuration 
        parameters as the one being read.  e.g. the same min_sep, max_sep, etc.  This is not
        checked by the read function.

        :param file_name:   The name of the file to read in.
        :param file_type:   The type of file ('ASCII' or 'FITS').  (default: determine the type
                            automatically from the extension of file_name.)
        """
        self.logger.info('Reading NNN correlations from %s',file_name)

        data = treecorr.util.gen_read(file_name, file_type=file_type)
        s = self.logr.shape
        self.logr = numpy.log(data['R_nom']).reshape(s)
        self.u = data['u_nom'].reshape(s)
        self.v = data['v_nom'].reshape(s)
        self.meand1 = data['meand1'].reshape(s)
        self.meanlogd1 = data['meanlogd1'].reshape(s)
        self.meand2 = data['meand2'].reshape(s)
        self.meanlogd2 = data['meanlogd2'].reshape(s)
        self.meand3 = data['meand3'].reshape(s)
        self.meanlogd3 = data['meanlogd3'].reshape(s)
        self.meanu = data['meanu'].reshape(s)
        self.meanv = data['meanv'].reshape(s)
        self.weight = data['DDD'].reshape(s)
        self.ntri = data['ntri'].reshape(s)


