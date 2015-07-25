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

#_treecorr.BuildNNNCorr.restype = cvoid_ptr
#_treecorr.BuildNNNCorr.argtypes = [
#    cdouble, cdouble, cint, cdouble, cdouble,
#    cdouble_ptr, cdouble_ptr ]
#_treecorr.DestroyNNNCorr.argtypes = [ cvoid_ptr ]
##_treecorr.ProcessAutoNNNSphere.argtypes = [ cvoid_ptr, cvoid_ptr, cint ]
#_treecorr.ProcessAutoNNNFlat.argtypes = [ cvoid_ptr, cvoid_ptr, cint ]
##_treecorr.ProcessCrossNNNSphere.argtypes = [ cvoid_ptr, cvoid_ptr, cvoid_ptr, cint ]
##_treecorr.ProcessCrossNNNFlat.argtypes = [ cvoid_ptr, cvoid_ptr, cvoid_ptr, cint ]


class NNNCorrelation(treecorr.BinnedCorr3):
    """This class handles the calculation and storage of a 2-point count-count correlation
    function.  i.e. the regular density correlation function.

    It holds the following attributes:

        :logr:      The nominal center of the bin in log(r).
        :u:         The nominal center of the bin in u.
        :v:         The nominal center of the bin in v.
        :meanlogr:  The mean value of log(r) for the triangle in each bin.
                    If there are no triangles in a bin, then logr will be used instead.
        :meanu:     The mean value of u for the triangle in each bin.
                    If there are no tri in a bin, then logr will be used instead.
        :ntri:      The number of tri going into each bin.
        :tot:       The total number of tri processed, which is used to normalize
                    the randoms if they have a different number of tri.

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
        self.meanlogr = numpy.zeros(shape, dtype=float)
        self.ntri = numpy.zeros(shape, dtype=float)
        self.tot = 0.

        meanlogr = self.meanlogr.ctypes.data_as(cdouble_ptr)
        ntri = self.ntri.ctypes.data_as(cdouble_ptr)

        if False:
            self.corr = _treecorr.BuildNNNCorr(
                    self.min_sep,self.max_sep,self.nbins,self.bin_size,
                    self.min_u,self.max_u,self.nubins,self.ubin_size,
                    self.min_v,self.max_v,self.nvbins,self.vbin_size,
                    self.b, meanlogr, ntri);
        self.logger.debug('Finished building NNNCorr')


    def __del__(self):
        # Using memory allocated from the C layer means we have to explicitly deallocate it
        # rather than being able to rely on the Python memory manager.
        if False:
            if hasattr(self,'data'):    # In case __init__ failed to get that far
                _treecorr.DestroyNNNCorr(self.corr)


    def process_auto(self, cat):
        """Process a single catalog, accumulating the auto-correlation.

        This accumulates the auto-correlation for the given catalog.  After
        calling this function as often as desired, the finalize() command will
        finish the calculation of meanlogr.

        :param cat:      The catalog to process
        """
        self.logger.info('Starting process NNN auto-correlations for cat %s.',cat.name)

        self._set_num_threads()

        field = cat.getNField(self.min_sep,self.max_sep,self.b,self.split_method)

        if field.sphere:
            raise NotImplemented("No spherical NNN yet.")
            #_treecorr.ProcessAutoNNNSphere(self.corr, field.data, self.output_dots)
        else:
            _treecorr.ProcessAutoNNNFlat(self.corr, field.data, self.output_dots)
        self.tot += (1./6.) * cat.nobj**3


    def process_cross21(self, cat1, cat2):
        """Process two catalogs, accumulating the 3pt cross-correlation, where two of the 
        points in each triangle come from the first catalog, and one from the second.

        This accumulates the cross-correlation for the given catalogs.  After
        calling this function as often as desired, the finalize() command will
        finish the calculation of meanlogr.

        :param cat1:     The first catalog to process
        :param cat2:     The second catalog to process
        """
        raise NotImplemented("No cross NNN yet.")
        self.logger.info('Starting process NN cross-correlations for cats %s, %s.',
                         cat1.name, cat2.name)

        self._set_num_threads()

        f1 = cat1.getNField(self.min_sep,self.max_sep,self.b,self.split_method)
        f2 = cat2.getNField(self.min_sep,self.max_sep,self.b,self.split_method)

        if f1.sphere != f2.sphere:
            raise AttributeError("Cannot correlate catalogs with different coordinate systems.")

        if f1.sphere:
            _treecorr.ProcessCrossNNSphere(self.corr, f1.data, f2.data, self.output_dots)
        else:
            _treecorr.ProcessCrossNNFlat(self.corr, f1.data, f2.data, self.output_dots)
        self.tot += 0.5 * cat1.nobj**2 * cat2.nobj


    def process_cross(self, cat1, cat2, cat3):
        """Process a set of three catalogs, accumulating the 3pt cross-correlation.

        This accumulates the cross-correlation for the given catalogs.  After
        calling this function as often as desired, the finalize() command will
        finish the calculation of meanlogr.

        :param cat1:     The first catalog to process
        :param cat2:     The second catalog to process
        :param cat3:     The third catalog to process
        """
        raise NotImplemented("No cross NNN yet.")
        self.logger.info('Starting process NN cross-correlations for cats %s, %s.',
                         cat1.name, cat2.name)

        self._set_num_threads()

        f1 = cat1.getNField(self.min_sep,self.max_sep,self.b,self.split_method)
        f2 = cat2.getNField(self.min_sep,self.max_sep,self.b,self.split_method)

        if f1.sphere != f2.sphere:
            raise AttributeError("Cannot correlate catalogs with different coordinate systems.")

        if f1.sphere:
            _treecorr.ProcessCrossNNSphere(self.corr, f1.data, f2.data, self.output_dots)
        else:
            _treecorr.ProcessCrossNNFlat(self.corr, f1.data, f2.data, self.output_dots)
        self.tot += cat1.nobj * cat2.nobj * cat3.nobj


    def finalize(self):
        """Finalize the calculation of the correlation function.

        The process_auto and process_cross commands accumulate values in each bin,
        so they can be called multiple times if appropriate.  Afterwards, this command
        finishes the calculation of meanlogr by dividing by the total ntri.
        """
        mask1 = self.ntri != 0
        mask2 = self.ntri == 0

        self.meanlogr[mask1] /= self.ntri[mask1]

        # Update the units of meanlogr
        self.meanlogr[mask1] -= self.log_sep_units

        # Use meanlogr when available, but set to nominal when no triangles in bin.
        self.meanlogr[mask2] = self.logr[mask2]


    def clear(self):
        """Clear the data vectors
        """
        self.meanlogr[:] = 0.
        self.ntri[:] = 0.
        self.tot = 0.


    def process(self, cat1, cat2=None, cat3=None):
        """Compute the correlation function.

        If only 1 argument is given, then compute an auto-correlation function.
        If 2 arguments are given, then compute a cross-correlation function with the 
            first catalog taking two corners of the triangles.
        If 3 arguments are given, then compute a cross-correlation function.

        Both arguments may be lists, in which case all items in the list are used 
        for that element of the correlation.

        :param cat1:    A catalog or list of catalogs for the first N field.
        :param cat2:    A catalog or list of catalogs for the second N field, if any.
                        (default: None)
        :param cat3:    A catalog or list of catalogs for the second N field, if any.
                        (default: None)
        """
        self.clear()
        if not isinstance(cat1,list): cat1 = [cat1]
        if cat2 is not None and not isinstance(cat2,list): cat2 = [cat2]
        if cat3 is not None and not isinstance(cat3,list): cat3 = [cat3]
        if len(cat1) == 0:
            raise ValueError("No catalogs provided for cat1")
        if cat2 is None and cat3 is not None:
            raise AttributeError("Must provide cat2 if cat3 is given.")

        if cat2 is None or len(cat2) == 0:
            self._process_all_auto(cat1)
        elif cat3 is None or len(cat3) == 0:
            self._process_all_cross21(cat1,cat2)
        else:
            self._process_all_cross(cat1,cat2,cat3)
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

        rrrw = self.tot / rr.tot
        if rrd is None:
            zeta = (self.ntri - rrr.ntri * rrrw)
        elif rdr is None:
            if rrd.tot == 0:
                raise RuntimeError("rrd has tot=0.")
            if ddr.tot == 0:
                raise RuntimeError("ddr has tot=0.")
            rrdw = self.tot / rrd.tot
            ddrw = self.tot / ddr.tot
            zeta = (self.ntri - 3.*rrd.ntri * rrdw + ddr.ntri * ddrw - rrr.ntri * rrrw)
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
        if any(rrr.ntri == 0):
            self.logger.warn("Warning: Some bins for the randoms had no tri.")
            self.logger.warn("         Probably max_sep is larger than your field.")
        mask1 = rrr.ntri != 0
        mask2 = rrr.ntri == 0
        zeta[mask1] /= (rrr.ntri[mask1] * rrrw)
        zeta[mask2] = 0

        varzeta = numpy.zeros_like(rrr.ntri)
        varzeta[mask1] = 1./ (rrr.ntri[mask1] * rrrw)

        return zeta, varzeta


    def write(self, file_name, rrr=None, rrd=None, ddr=None,
                      rdr=None, drr=None, drd=None, rdd=None, file_type=None):
        """Write the correlation function to the file, file_name.

        Normally, at least rrr should be provided, but if this is None, then only the 
        basic accumulated number of triangles are output (along with the separation columns).

        If at least rrr is given, then it will output an estimate of the final 3pt correlation
        function, zeta. There are several possible formulae that could be used here:

        1. The simplest formula to use is (ddd-rrr)/rrr.
           In this case, only rrr needs to be given, the NNNCorrelation of a random field.

        2. For auto-correlations, a better formula is (ddd-3ddr+3rrd-rrr)/rrr.
           In this case, ddr and rrd calculate the triangles where two points come from either
           the data or the randoms, respectively.

        3. Finally, if the data correlation function is correlating two or three different
           kinds of things, then there are different "Randoms" for each of them.
           In this case, you need to do a different calculation for each of the three
           data values: (ddd-ddr-drd-rdd+rrd+rdr+drr-rrr)/rrr.

        :param file_name:   The name of the file to write to.
        :param rrr:         An NNNCorrelation object for the random field. (default: None)
        :param rrd:         RRD if desired. (default: None)
        :param ddr:         DDR if desired. (default: None)
        :param rdr:         RDR if desired. (default: None)
        :param drr:         DRR if desired. (default: None)
        :param drd:         DRD if desired. (default: None)
        :param rdd:         RDD if desired. (default: None)
        :param file_type:   The type of file to write ('ASCII' or 'FITS').  (default: determine
                            the type automatically from the extension of file_name.)
        """
        self.logger.info('Writing NNN correlations to %s',file_name)
        
        col_names = [ 'R_nom', 'u_nom', 'v_nom', '<R>', '<u>', '<v>' ]
        columns = [ numpy.exp(self.logr), self.u, self.v,
                    numpy.exp(self.meanlogr), self.meanu, self.meanv ]
        if rrr is None:
            col_names += [ 'ntri' ]
            columns += [ self.ntri ]
            if rrd is not None:
                raise AttributeError("rrr must be provided if rrd is not None")
            if rdr is not None:
                raise AttributeError("rrr must be provided if rdd is not None")
            if drr is not None:
                raise AttributeError("rrr must be provided if drr is not None")
            if ddr is not None:
                raise AttributeError("rrr must be provided if ddr is not None")
            if drd is not None:
                raise AttributeError("rrr must be provided if drd is not None")
            if rdd is not None:
                raise AttributeError("rrr must be provided if rdd is not None")
        else:
            zeta, varzeta = self.calculateZeta(rrr,rrd,ddr,rdr,drr,drd,rdd)

            col_names += [ 'zeta','sigma_zeta','DDD','RRR' ]
            columns += [ zeta, numpy.sqrt(varzeta),
                         self.ntri, rrr.ntri * (self.tot/rrr.tot) ]

            if rrd is not None and rdr is None:
                col_names += ['DDR','RRD']
                columns += [ ddr.ntri * (self.tot/ddr.tot), rrd.ntri * (self.tot/rrd.tot) ]
            elif rdr is not None:
                col_names += ['DDR','DRD','RDD','RRD','RDR','DRR']
                columns += [ ddr.ntri * (self.tot/ddr.tot), drd.ntri * (self.tot/drd.tot),
                             rdd.ntri * (self.tot/rdd.tot), rrd.ntri * (self.tot/rrd.tot),
                             rdr.ntri * (self.tot/rdr.tot), drr.ntri * (self.tot/drr.tot) ]

        self.gen_write(file_name, col_names, columns, file_type=file_type)


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

        data = self.gen_read(file_name, file_type=file_type)
        self.logr = numpy.log(data['R_nom'])
        self.u = data['u_nom']
        self.v = data['v_nom']
        self.meanlogr = numpy.log(data['<R>'])
        self.meanu = data['<u>']
        self.meanv = data['<v>']
        if 'ntri' in data.dtype.names:
            self.ntri = data['ntri']
        else:
            self.ntri = data['DDD']


