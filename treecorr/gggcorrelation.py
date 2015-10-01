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

_treecorr.BuildGGGCorr.restype = cvoid_ptr
_treecorr.BuildGGGCorr.argtypes = [
    cdouble, cdouble, cint, cdouble, cdouble,
    cdouble, cdouble, cint, cdouble, cdouble,
    cdouble, cdouble, cint, cdouble, cdouble,
    cdouble_ptr, cdouble_ptr, cdouble_ptr, cdouble_ptr,
    cdouble_ptr, cdouble_ptr, cdouble_ptr, cdouble_ptr,
    cdouble_ptr, cdouble_ptr, cdouble_ptr, cdouble_ptr,
    cdouble_ptr, cdouble_ptr, cdouble_ptr, cdouble_ptr,
    cdouble_ptr, cdouble_ptr ]
_treecorr.DestroyGGGCorr.argtypes = [ cvoid_ptr ]
_treecorr.ProcessAutoGGGFlat.argtypes = [ cvoid_ptr, cvoid_ptr, cint ]
_treecorr.ProcessAutoGGG3D.argtypes = [ cvoid_ptr, cvoid_ptr, cint ]
_treecorr.ProcessAutoGGGPerp.argtypes = [ cvoid_ptr, cvoid_ptr, cint ]
_treecorr.ProcessCrossGGGFlat.argtypes = [ cvoid_ptr, cvoid_ptr, cvoid_ptr, cvoid_ptr, cint ]
_treecorr.ProcessCrossGGG3D.argtypes = [ cvoid_ptr, cvoid_ptr, cvoid_ptr, cvoid_ptr, cint ]
_treecorr.ProcessCrossGGGPerp.argtypes = [ cvoid_ptr, cvoid_ptr, cvoid_ptr, cvoid_ptr, cint ]


class GGGCorrelation(treecorr.BinnedCorr3):
    """This class handles the calculation and storage of a 3-point shear-shear-shear correlation
    function.

    We use the "natural components" of the shear 3-point function described by Schneider &
    Lombardi (2003) [Astron.Astrophys. 397 (2003) 809-818].  In this paradigm, the shears
    are projected relative to some point defined by the geometry of the triangle.  They
    give several reasonable choices for this point.  We choose the triangle's centroid as the 
    "most natural" point, as many simple shear fields have purely real :math:`\\Gamma_0` using
    this definition.  It is also a fairly simple point to calculate in the code compared to 
    some of the other options they offer, so projections relative to it are fairly efficient.

    There are 4 complex-valued 3-point shear corrletion functions defined for triples of shear
    values projected relative to the line joining the location of the shear to the cenroid of
    the triangle:

    .. math::

        \\Gamma_0 &= \\langle \\gamma(\\mathbf{x1}) \\gamma(\\mathbf{x2}) \\gamma(\\mathbf{x3}) \\rangle \\\\
        \\Gamma_1 &= \\langle \\gamma(\\mathbf{x1})^* \\gamma(\\mathbf{x2}) \\gamma(\\mathbf{x3}) \\rangle \\\\
        \\Gamma_2 &= \\langle \\gamma(\\mathbf{x1}) \\gamma(\\mathbf{x2})^* \\gamma(\\mathbf{x3}) \\rangle \\\\
        \\Gamma_3 &= \\langle \\gamma(\\mathbf{x1}) \\gamma(\\mathbf{x2}) \\gamma(\\mathbf{x3})^* \\rangle \\\\

    where :math:`\\mathbf{x1}, \\mathbf{x2}, \\mathbf{x3}` are the corners of the triange opposite
    sides `d1, d2, d3` respectively, where `d1 > d2 > d3`, and `*` indicates complex conjugation.

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

        :logr:      The nominal center of each bin in log(r).
        :u:         The nominal center of each bin in u.
        :v:         The nominal center of each bin in v.
        :meand1:    The (weighted) mean value of d1 for the triangles in each bin.
        :meanlogd1: The mean value of log(d1) for the triangles in each bin.
        :meand2:    The (weighted) mean value of d2 (aka r) for the triangles in each bin.
        :meanlogd2: The mean value of log(d2) for the triangles in each bin.
        :meand2:    The (weighted) mean value of d3 for the triangles in each bin.
        :meanlogd2: The mean value of log(d3) for the triangles in each bin.
        :meanu:     The mean value of u for the triangles in each bin.
        :meanv:     The mean value of v for the triangles in each bin.
        :gam0:      The 0th "natural" correlation function, :math:`\\Gamma_0(r,u,v)`.
        :gam1:      The 1st "natural" correlation function, :math:`\\Gamma_1(r,u,v)`.
        :gam2:      The 2nd "natural" correlation function, :math:`\\Gamma_2(r,u,v)`.
        :gam3:      The 3rd "natural" correlation function, :math:`\\Gamma_3(r,u,v)`.
        :vargam:    The variance of :math:`\\Gamma` values, only including the shot noise
                    propagated into the final correlation.  This does not include sample variance,
                    so it is always an underestimate of the actual variance.
        :weight:    The total weight in each bin.
        :ntri:      The number of triangles going into each bin.

    If sep_units are given (either in the config dict or as a named kwarg) then logr and meanlogr
    both take r to be in these units.  i.e. exp(logr) will have R in units of sep_units.

    The usage pattern is as follows::

        >>> ggg = treecorr.GGGCorrelation(config)
        >>> ggg.process(cat)              # For auto-correlation.
        >>> ggg.process(cat1,cat2,cat3)   # For cross-correlation.
        >>> ggg.write(file_name)          # Write out to a file.
        >>> gam0 = ggg.gam0, etc.         # To access gamma values directly.
        >>> gam0r = ggg.gam0r             # You can also access real and imag parts separately.
        >>> gam0i = ggg.gam0i

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
        self.gam0r = numpy.zeros(shape, dtype=float)
        self.gam1r = numpy.zeros(shape, dtype=float)
        self.gam2r = numpy.zeros(shape, dtype=float)
        self.gam3r = numpy.zeros(shape, dtype=float)
        self.gam0i = numpy.zeros(shape, dtype=float)
        self.gam1i = numpy.zeros(shape, dtype=float)
        self.gam2i = numpy.zeros(shape, dtype=float)
        self.gam3i = numpy.zeros(shape, dtype=float)
        self.vargam = numpy.zeros(shape, dtype=float)
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
        self._build_corr()
        self.logger.debug('Finished building GGGCorr')

    @property
    def gam0(self): return self.gam0r + 1j * self.gam0i
    @property
    def gam1(self): return self.gam1r + 1j * self.gam1i
    @property
    def gam2(self): return self.gam2r + 1j * self.gam2i
    @property
    def gam3(self): return self.gam3r + 1j * self.gam3i
    @gam0.setter
    def gam0(self, g0): self.gam0r = g0.real; self.gam0i = g0.imag
    @gam1.setter
    def gam1(self, g1): self.gam1r = g1.real; self.gam1i = g1.imag
    @gam2.setter
    def gam2(self, g2): self.gam2r = g2.real; self.gam2i = g2.imag
    @gam3.setter
    def gam3(self, g3): self.gam3r = g3.real; self.gam3i = g3.imag

    def _build_corr(self):
        gam0r = self.gam0r.ctypes.data_as(cdouble_ptr)
        gam0i = self.gam0i.ctypes.data_as(cdouble_ptr)
        gam1r = self.gam1r.ctypes.data_as(cdouble_ptr)
        gam1i = self.gam1i.ctypes.data_as(cdouble_ptr)
        gam2r = self.gam2r.ctypes.data_as(cdouble_ptr)
        gam2i = self.gam2i.ctypes.data_as(cdouble_ptr)
        gam3r = self.gam3r.ctypes.data_as(cdouble_ptr)
        gam3i = self.gam3i.ctypes.data_as(cdouble_ptr)
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
        self.corr = _treecorr.BuildGGGCorr(
                self.min_sep,self.max_sep,self.nbins,self.bin_size,self.b,
                self.min_u,self.max_u,self.nubins,self.ubin_size,self.bu,
                self.min_v,self.max_v,self.nvbins,self.vbin_size,self.bv,
                gam0r, gam0i, gam1r, gam1i, gam2r, gam2i, gam3r, gam3i,
                meand1, meanlogd1, meand2, meanlogd2, meand3, meanlogd3, meanu, meanv, 
                weight, ntri);

    def __del__(self):
        # Using memory allocated from the C layer means we have to explicitly deallocate it
        # rather than being able to rely on the Python memory manager.
        if hasattr(self,'corr'):    # In case __init__ failed to get that far
            _treecorr.DestroyGGGCorr(self.corr)

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
        return 'GGGCorrelation(config=%r)'%self.config

    def process_auto(self, cat, metric=None, num_threads=None):
        """Process a single catalog, accumulating the auto-correlation.

        This accumulates the auto-correlation for the given catalog.  After
        calling this function as often as desired, the finalize() command will
        finish the calculation of meand1, meanlogd1, etc.

        :param cat:         The catalog to process
        :param metric:      Which metric to use.  See :meth:`~treecorr.GGGCorrelation.process` for 
                            details.  (default: 'Euclidean'; this value can also be given in the 
                            constructor in the config dict.)
        :param num_threads: How many OpenMP threads to use during the calculation.  
                            (default: use the number of cpu cores; this value can also be given in
                            the constructor in the config dict.) Note that this won't work if the 
                            system's C compiler is clang prior to version 3.7.
        """
        if cat.name == '':
            self.logger.info('Starting process GGG auto-correlations')
        else:
            self.logger.info('Starting process GGG auto-correlations for cat %s.', cat.name)

        if metric is None:
            metric = treecorr.config.get(self.config,'metric',str,'Euclidean')
        if metric not in ['Euclidean', 'Rperp']:
            raise ValueError("Invalid metric.")

        self._set_num_threads(num_threads)

        min_size = self.min_sep * self.min_u
        max_size = 2.*self.max_sep 
        b = numpy.max( (self.b, self.bu, self.bv) )
        field = cat.getGField(min_size,max_size,b,self.split_method,metric,self.max_top)

        self.logger.info('Starting %d jobs.',field.nTopLevelNodes)
        if field.flat:
            _treecorr.ProcessAutoGGGFlat(self.corr, field.data, self.output_dots)
        elif field.perp:
            _treecorr.ProcessAutoGGGPerp(self.corr, field.data, self.output_dots)
        else:
            _treecorr.ProcessAutoGGG3D(self.corr, field.data, self.output_dots)

    def process_cross21(self, cat1, cat2, metric=None, num_threads=None):
        """Process two catalogs, accumulating the 3pt cross-correlation, where two of the 
        points in each triangle come from the first catalog, and one from the second.

        This accumulates the cross-correlation for the given catalogs.  After
        calling this function as often as desired, the finalize() command will
        finish the calculation of meand1, meanlogd1, etc.

        :param cat1:        The first catalog to process
        :param cat2:        The second catalog to process
        :param metric:      Which metric to use.  See :meth:`~treecorr.GGGCorrelation.process` for 
                            details.  (default: 'Euclidean'; this value can also be given in the 
                            constructor in the config dict.)
        :param num_threads: How many OpenMP threads to use during the calculation.  
                            (default: use the number of cpu cores; this value can also be given in
                            the constructor in the config dict.) Note that this won't work if the 
                            system's C compiler is clang prior to version 3.7.
        """
        raise NotImplemented("No partial cross GGG yet.")


    def process_cross(self, cat1, cat2, cat3, metric=None, num_threads=None):
        """Process a set of three catalogs, accumulating the 3pt cross-correlation.

        This accumulates the cross-correlation for the given catalogs.  After
        calling this function as often as desired, the finalize() command will
        finish the calculation of meand1, meanlogd1, etc.

        :param cat1:        The first catalog to process
        :param cat2:        The second catalog to process
        :param cat3:        The third catalog to process
        :param metric:      Which metric to use.  See :meth:`~treecorr.GGGCorrelation.process` for 
                            details.  (default: 'Euclidean'; this value can also be given in the 
                            constructor in the config dict.)
        :param num_threads: How many OpenMP threads to use during the calculation.  
                            (default: use the number of cpu cores; this value can also be given in
                            the constructor in the config dict.) Note that this won't work if the 
                            system's C compiler is clang prior to version 3.7.
        """
        if cat1.name == '' and cat2.name == '' and cat3.name == '':
            self.logger.info('Starting process GGG cross-correlations')
        else:
            self.logger.info('Starting process GGG cross-correlations for cats %s, %s, %s.',
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
        f1 = cat1.getGField(min_size,max_size,b,self.split_method,metric,self.max_top)
        f2 = cat2.getGField(min_size,max_size,b,self.split_method,metric,self.max_top)
        f3 = cat3.getGField(min_size,max_size,b,self.split_method,metric,self.max_top)

        self.logger.info('Starting %d jobs.',f1.nTopLevelNodes)
        if f1.flat:
            _treecorr.ProcessCrossGGGFlat(self.corr, f1.data, f2.data, f3.data, self.output_dots)
        elif f1.perp:
            _treecorr.ProcessCrossGGGPerp(self.corr, f1.data, f2.data, f3.data, self.output_dots)
        else:
            _treecorr.ProcessCrossGGG3D(self.corr, f1.data, f2.data, f3.data, self.output_dots)


    def finalize(self, varg1, varg2, varg3):
        """Finalize the calculation of the correlation function.

        The process_auto and process_cross commands accumulate values in each bin,
        so they can be called multiple times if appropriate.  Afterwards, this command
        finishes the calculation by dividing by the total weight.

        :param varg1:   The shear variance for the first field.
        :param varg2:   The shear variance for the second field.
        :param varg3:   The shear variance for the third field.
        """
        mask1 = self.weight != 0
        mask2 = self.weight == 0

        self.gam0r[mask1] /= self.weight[mask1]
        self.gam0i[mask1] /= self.weight[mask1]
        self.gam1r[mask1] /= self.weight[mask1]
        self.gam1i[mask1] /= self.weight[mask1]
        self.gam2r[mask1] /= self.weight[mask1]
        self.gam2i[mask1] /= self.weight[mask1]
        self.gam3r[mask1] /= self.weight[mask1]
        self.gam3i[mask1] /= self.weight[mask1]
        self.vargam[mask1] = varg1 * varg2 * varg3 / self.weight[mask1]
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
        self.vargam[mask2] = 0.
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
        self.gam0r[:,:,:] = 0.
        self.gam0i[:,:,:] = 0.
        self.gam1r[:,:,:] = 0.
        self.gam1i[:,:,:] = 0.
        self.gam2r[:,:,:] = 0.
        self.gam2i[:,:,:] = 0.
        self.gam3r[:,:,:] = 0.
        self.gam3i[:,:,:] = 0.
        self.vargam[:,:,:] = 0.
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

    def __iadd__(self, other):
        """Add a second GGGCorrelation's data to this one.

        Note: For this to make sense, both Correlation objects should have been using
        process_auto and/or process_cross, and they should not have had finalize called yet.
        Then, after adding them together, you should call finalize on the sum.
        """
        if not isinstance(other, GGGCorrelation):
            raise AttributeError("Can only add another GGGCorrelation object")
        if not (self.nbins == other.nbins and
                self.min_sep == other.min_sep and
                self.max_sep == other.max_sep and
                self.nubins == other.nubins and
                self.min_u == other.min_u and
                self.max_u == other.max_u and
                self.nvbins == other.nvbins and
                self.min_v == other.min_v and
                self.max_v == other.max_v):
            raise ValueError("GGGCorrelation to be added is not compatible with this one.")

        self.gam0r[:] += other.gam0r[:]
        self.gam0i[:] += other.gam0i[:]
        self.gam1r[:] += other.gam1r[:]
        self.gam1i[:] += other.gam1i[:]
        self.gam2r[:] += other.gam2r[:]
        self.gam2i[:] += other.gam2i[:]
        self.gam3r[:] += other.gam3r[:]
        self.gam3i[:] += other.gam3i[:]
        self.vargam[:] += other.vargam[:]
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
        import math
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
            raise NotImplemented("No partial cross GGG yet.")
        if cat3 is None and cat2 is not None:
            raise NotImplemented("No partial cross GGG yet.")

        if cat2 is None and cat3 is None:
            varg1 = treecorr.calculateVarG(cat1)
            varg2 = varg1
            varg3 = varg1
            self.logger.info("varg = %f: sig_g = %f",varg1,math.sqrt(varg1))
            self._process_all_auto(cat1, metric, num_threads)
        else:
            assert cat2 is not None and cat3 is not None
            varg1 = treecorr.calculateVarK(cat1)
            varg2 = treecorr.calculateVarK(cat2)
            varg3 = treecorr.calculateVarK(cat3)
            self.logger.info("varg1 = %f: sig_g = %f",varg1,math.sqrt(varg1))
            self.logger.info("varg2 = %f: sig_g = %f",varg2,math.sqrt(varg2))
            self.logger.info("varg3 = %f: sig_g = %f",varg3,math.sqrt(varg3))
            self._process_all_cross(cat1,cat2,cat3, metric, num_threads)
        self.finalize(varg1,varg2,varg3)


    def write(self, file_name, file_type=None, prec=None):
        """Write the correlation function to the file, file_name.

        As described in the doc string for :class:`~treecorr.GGGCorrelation`, we use the
        "natural components" of the shear 3-point function described by Schneider & Lombardi (2003)
        using the triangle centroid as the projection point.  There are 4 complex-valued natural
        components, so there are 8 columns in the output file.

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
            :gam0r:         The real part of the estimator of :math:`\\Gamma_0(r,u,v)`.
            :gam0i:         The imag part of the estimator of :math:`\\Gamma_0(r,u,v)`.
            :gam1r:         The real part of the estimator of :math:`\\Gamma_1(r,u,v)`.
            :gam1i:         The imag part of the estimator of :math:`\\Gamma_1(r,u,v)`.
            :gam2r:         The real part of the estimator of :math:`\\Gamma_2(r,u,v)`.
            :gam2i:         The imag part of the estimator of :math:`\\Gamma_2(r,u,v)`.
            :gam3r:         The real part of the estimator of :math:`\\Gamma_3(r,u,v)`.
            :gam3i:         The imag part of the estimator of :math:`\\Gamma_3(r,u,v)`.
            :sigma_gam:     The sqrt of the variance estimate of :math:`\\Gamma` values.
            :weight:        The total weight of triangles contributing to each bin.
            :ntri:          The number of triangles contributing to each bin.

        :param file_name:   The name of the file to write to.
        :param file_type:   The type of file to write ('ASCII' or 'FITS').  (default: determine
                            the type automatically from the extension of file_name.)
        :param prec:        For ASCII output catalogs, the desired precision. (default: 4;
                            this value can also be given in the constructor in the config dict.)
        """
        self.logger.info('Writing GGG correlations to %s',file_name)
        
        col_names = [ 'R_nom', 'u_nom', 'v_nom', 'meand1', 'meanlogd1', 'meand2', 'meanlogd2',
                      'meand3', 'meanlogd3', 'meanu', 'meanv', 
                      'gam0r', 'gam0i', 'gam1r', 'gam1i', 'gam2r', 'gam2i', 'gam3r', 'gam3i',
                      'sigma_gam', 'weight', 'ntri' ]
        columns = [ numpy.exp(self.logr), self.u, self.v,
                    self.meand1, self.meanlogd1, self.meand2, self.meanlogd2,
                    self.meand3, self.meanlogd3, self.meanu, self.meanv,
                    self.gam0r, self.gam0i, self.gam1r, self.gam1i,
                    self.gam2r, self.gam2i, self.gam3r, self.gam3i,
                    numpy.sqrt(self.vargam), self.weight, self.ntri ]
        if prec is None:
            prec = self.config.get('precision', 4)

        treecorr.util.gen_write(
            file_name, col_names, columns, prec=prec, file_type=file_type, logger=self.logger)


    def read(self, file_name, file_type=None):
        """Read in values from a file.

        This should be a file that was written by TreeCorr, preferably a FITS file, so there
        is no loss of information.

        Warning: The GGGCorrelation object should be constructed with the same configuration 
        parameters as the one being read.  e.g. the same min_sep, max_sep, etc.  This is not
        checked by the read function.

        :param file_name:   The name of the file to read in.
        :param file_type:   The type of file ('ASCII' or 'FITS').  (default: determine the type
                            automatically from the extension of file_name.)
        """
        self.logger.info('Reading GGG correlations from %s',file_name)

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
        self.gam0r = data['gam0r'].reshape(s)
        self.gam0i = data['gam0i'].reshape(s)
        self.gam1r = data['gam1r'].reshape(s)
        self.gam1i = data['gam1i'].reshape(s)
        self.gam2r = data['gam2r'].reshape(s)
        self.gam2i = data['gam2i'].reshape(s)
        self.gam3r = data['gam3r'].reshape(s)
        self.gam3i = data['gam3i'].reshape(s)
        self.vargam = data['sigma_gam'].reshape(s)**2
        self.weight = data['weight'].reshape(s)
        self.ntri = data['ntri'].reshape(s)


