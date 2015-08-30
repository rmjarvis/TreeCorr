# Copyright (c) 2003-2014 by Mike Jarvis
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
    cdouble_ptr, cdouble_ptr, cdouble_ptr, cdouble_ptr ]
_treecorr.DestroyNNNCorr.argtypes = [ cvoid_ptr ]
_treecorr.ProcessAutoNNNFlat.argtypes = [ cvoid_ptr, cvoid_ptr, cint ]
_treecorr.ProcessAutoNNNSphere.argtypes = [ cvoid_ptr, cvoid_ptr, cint ]
_treecorr.ProcessAutoNNNPerp.argtypes = [ cvoid_ptr, cvoid_ptr, cint ]
_treecorr.ProcessCrossNNNFlat.argtypes = [ cvoid_ptr, cvoid_ptr, cvoid_ptr, cvoid_ptr, cint ]
_treecorr.ProcessCrossNNNSphere.argtypes = [ cvoid_ptr, cvoid_ptr, cvoid_ptr, cvoid_ptr, cint ]
_treecorr.ProcessCrossNNNPerp.argtypes = [ cvoid_ptr, cvoid_ptr, cvoid_ptr, cvoid_ptr, cint ]


class NNNCorrelation(treecorr.BinnedCorr3):
    """This class handles the calculation and storage of a 2-point count-count correlation
    function.  i.e. the regular density correlation function.

    It holds the following attributes:

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
        :ntri:      The number of triangles going into each bin.
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

    :param config:      The configuration dict which defines attributes about how to read the file.
                        Any kwargs that are not those listed here will be added to the config, 
                        so you can even omit the config dict and just enter all parameters you
                        want as kwargs.  (default: None) 
    :param logger:      If desired, a logger object for logging. (default: None, in which case
                        one will be built according to the config dict's verbose level.)

    Other parameters are allowed to be either in the config dict or as a named kwarg.
    See the documentation for BinnedCorr3 for details.
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
        ntri = self.ntri.ctypes.data_as(cdouble_ptr)
        self.corr = _treecorr.BuildNNNCorr(
                self.min_sep,self.max_sep,self.nbins,self.bin_size,self.b,
                self.min_u,self.max_u,self.nubins,self.ubin_size,self.bu,
                self.min_v,self.max_v,self.nvbins,self.vbin_size,self.bv,
                meand1, meanlogd1, meand2, meanlogd2, meand3, meanlogd3, meanu, meanv, ntri);

    def __del__(self):
        # Using memory allocated from the C layer means we have to explicitly deallocate it
        # rather than being able to rely on the Python memory manager.
        if hasattr(self,'data'):    # In case __init__ failed to get that far
            _treecorr.DestroyNNNCorr(self.corr)

    def copy(self):
        import copy
        return copy.deepcopy(self)

    def __getstate__(self):
        d = self.__dict__.copy()
        del d['corr']
        del d['logger']  # Oh well.  This is just lost in the copy.  Can't be pickled.
        return d

    def __setstate__(self):
        self.__dict__ = d
        self._build_corr()
        self.logger = treecorr.config.setup_logger(
                treecorr.config.get(self.config,'verbose',int,0),
                self.config.get('log_file',None))

    def __repr__(self):
        return 'NNNCorrelation(config=%r)'%self.config

    def process_auto(self, cat, perp=False):
        """Process a single catalog, accumulating the auto-correlation.

        This accumulates the auto-correlation for the given catalog.  After
        calling this function as often as desired, the finalize() command will
        finish the calculation of meand1, meanlogd1, etc.

        :param cat:     The catalog to process
        :param perp:    Whether to use the perpendicular distance rather than the 3d separation
                        (for catalogs with 3d positions) (default: False)
        """
        if cat.name == '':
            self.logger.info('Starting process NNN auto-correlations')
        else:
            self.logger.info('Starting process NNN auto-correlations for cat %s.', cat.name)

        self._set_num_threads()

        min_size = self.min_sep * self.min_u
        max_size = 2.*self.max_sep 
        b = numpy.max( (self.b, self.bu, self.bv) )
        field = cat.getNField(min_size,max_size,b,self.split_method,perp,self.max_top)

        if field.sphere:
            if field.perp:
                _treecorr.ProcessAutoNNNPerp(self.corr, field.data, self.output_dots)
            else:
                _treecorr.ProcessAutoNNNSphere(self.corr, field.data, self.output_dots)
        else:
            _treecorr.ProcessAutoNNNFlat(self.corr, field.data, self.output_dots)
        self.tot += (1./6.) * cat.nobj**3

    def process_cross21(self, cat1, cat2, perp=False):
        """Process two catalogs, accumulating the 3pt cross-correlation, where two of the 
        points in each triangle come from the first catalog, and one from the second.

        This accumulates the cross-correlation for the given catalogs.  After
        calling this function as often as desired, the finalize() command will
        finish the calculation of meand1, meanlogd1, etc.

        :param cat1:    The first catalog to process
        :param cat2:    The second catalog to process
        :param perp:    Whether to use the perpendicular distance rather than the 3d separation
                        (for catalogs with 3d positions) (default: False)
        """
        raise NotImplemented("No partial cross NNN yet.")


    def process_cross(self, cat1, cat2, cat3, perp=False):
        """Process a set of three catalogs, accumulating the 3pt cross-correlation.

        This accumulates the cross-correlation for the given catalogs.  After
        calling this function as often as desired, the finalize() command will
        finish the calculation of meand1, meanlogd1, etc.

        :param cat1:    The first catalog to process
        :param cat2:    The second catalog to process
        :param cat3:    The third catalog to process
        :param perp:    Whether to use the perpendicular distance rather than the 3d separation
                        (for catalogs with 3d positions) (default: False)
        """
        if cat1.name == '' and cat2.name == '' and cat3.name == '':
            self.logger.info('Starting process NNN cross-correlations')
        else:
            self.logger.info('Starting process NNN cross-correlations for cats %s, %s, %s.',
                             cat1.name, cat2.name, cat3.name)

        self._set_num_threads()

        f1 = cat1.getNField(self.min_sep,self.max_sep,self.b,self.split_method,perp,self.max_top)
        f2 = cat2.getNField(self.min_sep,self.max_sep,self.b,self.split_method,perp,self.max_top)
        f3 = cat3.getNField(self.min_sep,self.max_sep,self.b,self.split_method,perp,self.max_top)

        if f1.sphere != f2.sphere or f1.sphere != f3.sphere:
            raise AttributeError("Cannot correlate catalogs with different coordinate systems.")

        if f1.sphere:
            if f1.perp:
                _treecorr.ProcessCrossNNNPerp(self.corr, f1.data, f2.data, f3.data, self.output_dots)
            else:
                _treecorr.ProcessCrossNNNSphere(self.corr, f1.data, f2.data, f3.data, self.output_dots)
        else:
            _treecorr.ProcessCrossNNNFlat(self.corr, f1.data, f2.data, f3.data, self.output_dots)
        self.tot += cat1.nobj * cat2.nobj * cat3.nobj / 6.0


    def finalize(self):
        """Finalize the calculation of meand1, meanlogd1, etc.

        The process_auto and process_cross commands accumulate values in each bin,
        so they can be called multiple times if appropriate.  Afterwards, this command
        finishes the calculation of meanlogr, meanu, meanv by dividing by the total ntri.
        """
        mask1 = self.ntri != 0
        mask2 = self.ntri == 0

        self.meand1[mask1] /= self.ntri[mask1]
        self.meanlogd1[mask1] /= self.ntri[mask1]
        self.meand2[mask1] /= self.ntri[mask1]
        self.meanlogd2[mask1] /= self.ntri[mask1]
        self.meand3[mask1] /= self.ntri[mask1]
        self.meanlogd3[mask1] /= self.ntri[mask1]
        self.meanu[mask1] /= self.ntri[mask1]
        self.meanv[mask1] /= self.ntri[mask1]

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
        self.ntri[:] += other.ntri[:]
        self.tot += other.tot
        return self


    def process(self, cat1, cat2=None, cat3=None, perp=False):
        """Accumulate the number of triangles of points between cat1, cat2, and cat3.

        If only 1 argument is given, then compute an auto-correlation function.
        If 2 arguments are given, then compute a cross-correlation function with the 
            first catalog taking two corners of the triangles. (Not implemented yet.)
        If 3 arguments are given, then compute a cross-correlation function.

        All arguments may be lists, in which case all items in the list are used 
        for that element of the correlation.

        Note: For a correlation of multiple catalogs, it matters which corner of the
        triangle comes from which catalog.  The final accumulation will have 
        d1 > d2 > d3 where d1 is between two points in cat2,cat3; d2 is between 
        points in cat1,cat3; and d3 is between points in cat1,cat2.  To accumulate
        all the possible triangles between three catalogs, you should call this
        multiple times with the different catalogs in different positions.

        :param cat1:    A catalog or list of catalogs for the first N field.
        :param cat2:    A catalog or list of catalogs for the second N field, if any.
                        (default: None)
        :param cat3:    A catalog or list of catalogs for the third N field, if any.
                        (default: None)
        :param perp:    Whether to use the perpendicular distance rather than the 3d separation
                        (for catalogs with 3d positions) (default: False)
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
            self._process_all_auto(cat1, perp)
        else:
            assert cat2 is not None and cat3 is not None
            self._process_all_cross(cat1,cat2,cat3, perp)
        self.finalize()


    def calculateZeta(self, rrr, rrd=None, ddr=None,
                      rdr=None, drr=None, drd=None, rdd=None):
        """Calculate the 3pt function given another 3pt function of random
        points using the same mask, and possibly cross correlations of the data and random.

        There are several possible formulae that could be used here:

        1. The simplest formula to use is (ddd-rrr)/rrr.
           In this case, only rrr needs to be given, the NNNCorrelation of a random field.

        2. For auto-correlations, a better formula is (ddd-3ddr+3rrd-rrr)/rrr.
           In this case, ddr and rrd calculate the triangles where two points come from either
           the data or the randoms, respectively.

        3. Finally, if the data correlation function is correlating two or three different
           kinds of things, then there are different "Randoms" for each of them.
           In this case, you need to do a different calculation for each of the three
           data values: (ddd-ddr-drd-rdd+rrd+rdr+drr-rrr)/rrr.

        :param rrr:         An NNCorrelation object for the random field.
        :param rrd:         RRD if desired. (default: None)
        :param ddr:         DDR if desired. (default: None)
        :param rdr:         RDR if desired. (default: None)
        :param drr:         DRR if desired. (default: None)
        :param drd:         DRD if desired. (default: None)
        :param rdd:         RDD if desired. (default: None)
                            
        :returns:           (zeta, varzeta) as a tuple
        """
        # Each random ntri value needs to be rescaled by the ratio of total possible tri.
        if rrr.tot == 0:
            raise RuntimeError("rrr has tot=0.")

        if rrd is not None or ddr is not None:
            if rrd is None or ddr is None:
                raise AttributeError("Must provide both rrd and ddr")
        if rdr is not None or drr is not None or drd is not None or rdd is not None:
            if (rdr is None or drr is None or drd is None or rdd is None or
                rrd is None or ddr is None):
                raise AttributeError("Must provide all 6 combinations rdr, drr, etc.")

        rrrw = self.tot / rrr.tot
        if rrd is None:
            zeta = (self.ntri - rrr.ntri * rrrw)
        elif rdr is None:
            if rrd.tot == 0:
                raise RuntimeError("rrd has tot=0.")
            if ddr.tot == 0:
                raise RuntimeError("ddr has tot=0.")
            rrdw = self.tot / rrd.tot
            ddrw = self.tot / ddr.tot
            zeta = (self.ntri - 3.*rrd.ntri * rrdw + 3.*ddr.ntri * ddrw - rrr.ntri * rrrw)
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
            zeta = (self.ntri - rrd.ntri * rrdw - rdr.ntri * rdrw - drr.ntr * nrrw +
                    ddr.ntri * ddrw + drd.ntri * drdw + rdd.ntri * rddw - rrr.ntri * rrrw)
        if numpy.any(rrr.ntri == 0):
            self.logger.warn("Warning: Some bins for the randoms had no triangles.")
            self.logger.warn("         Probably max_sep is larger than your field.")
        mask1 = rrr.ntri != 0
        mask2 = rrr.ntri == 0
        zeta[mask1] /= (rrr.ntri[mask1] * rrrw)
        zeta[mask2] = 0

        varzeta = numpy.zeros_like(rrr.ntri)
        varzeta[mask1] = 1./ (rrr.ntri[mask1] * rrrw)

        return zeta, varzeta


    def write(self, file_name, rrr=None, drr=None, ddr=None,
                      rdr=None, rrd=None, drd=None, rdd=None, file_type=None):
        """Write the correlation function to the file, file_name.

        Normally, at least rrr should be provided, but if this is None, then only the 
        basic accumulated number of triangles are output (along with the separation columns).

        If at least rrr is given, then it will output an estimate of the final 3pt correlation
        function, zeta. There are several possible formulae that could be used here:

        1. The simplest formula to use is (ddd-rrr)/rrr.
           In this case, only rrr needs to be given, the NNNCorrelation of a random field.

        2. For auto-correlations, a better formula is (ddd-3ddr+3drr-rrr)/rrr.
           In this case, ddr and drr calculate the triangles where two points come from either
           the data or the randoms, respectively.

        3. Finally, if the data correlation function is correlating two or three different
           kinds of things, then there are different "Randoms" for each of them.
           In this case, you need to do a different calculation for each of the three
           data values: (ddd-ddr-drd-rdd+drr+rdr+rrd-rrr)/rrr.

        :param file_name:   The name of the file to write to.
        :param rrr:         An NNNCorrelation object for the random field. (default: None)
        :param drr:         RRD if desired. (default: None)
        :param ddr:         RDD if desired. (default: None)
        :param rdr:         RDR if desired. (default: None)
        :param rrd:         RRD if desired. (default: None)
        :param drd:         DRD if desired. (default: None)
        :param rdd:         RDD if desired. (default: None)
        :param file_type:   The type of file to write ('ASCII' or 'FITS').  (default: determine
                            the type automatically from the extension of file_name.)
        """
        self.logger.info('Writing NNN correlations to %s',file_name)
        
        col_names = [ 'R_nom', 'u_nom', 'v_nom', '<d1>', '<logd1>', '<d2>', '<logd2>',
                      '<d3>', '<logd3>', '<u>', '<v>' ]
        columns = [ numpy.exp(self.logr), self.u, self.v,
                    self.meand1, self.meanlogd1, self.meand2, self.meanlogd2,
                    self.meand3, self.meanlogd3, self.meanu, self.meanv ]
        if rrr is None:
            col_names += [ 'ntri' ]
            columns += [ self.ntri ]
            if drr is not None:
                raise AttributeError("rrr must be provided if drr is not None")
            if rdr is not None:
                raise AttributeError("rrr must be provided if rdd is not None")
            if rrd is not None:
                raise AttributeError("rrr must be provided if rrd is not None")
            if ddr is not None:
                raise AttributeError("rrr must be provided if ddr is not None")
            if drd is not None:
                raise AttributeError("rrr must be provided if drd is not None")
            if rdd is not None:
                raise AttributeError("rrr must be provided if rdd is not None")
        else:
            zeta, varzeta = self.calculateZeta(rrr,drr,ddr,rdr,rrd,drd,rdd)

            col_names += [ 'zeta','sigma_zeta','DDD','RRR' ]
            columns += [ zeta, numpy.sqrt(varzeta),
                         self.ntri, rrr.ntri * (self.tot/rrr.tot) ]

            if drr is not None and rdr is None:
                col_names += ['DDR','DRR']
                columns += [ ddr.ntri * (self.tot/ddr.tot), drr.ntri * (self.tot/drr.tot) ]
            elif rdr is not None:
                col_names += ['DDR','DRD','RDD','DRR','RDR','RRD']
                columns += [ ddr.ntri * (self.tot/ddr.tot), drd.ntri * (self.tot/drd.tot),
                             rdd.ntri * (self.tot/rdd.tot), drr.ntri * (self.tot/drr.tot),
                             rdr.ntri * (self.tot/rdr.tot), rrd.ntri * (self.tot/rrd.tot) ]

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
        self.meand1 = data['<d1>'].reshape(s)
        self.meanlogd1 = data['<logd1>'].reshape(s)
        self.meand2 = data['<d2>'].reshape(s)
        self.meanlogd2 = data['<logd2>'].reshape(s)
        self.meand3 = data['<d3>'].reshape(s)
        self.meanlogd3 = data['<logd3>'].reshape(s)
        self.meanu = data['<u>'].reshape(s)
        self.meanv = data['<v>'].reshape(s)
        if 'ntri' in data.dtype.names:
            self.ntri = data['ntri'].reshape(s)
        else:
            self.ntri = data['DDD'].reshape(s)


