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
# 3. Neither the name of the {organization} nor the names of its
#    contributors may be used to endorse or promote products derived from
#    this software without specific prior written permission.


import treecorr
import numpy

# Start by loading up the relevant C functions using ctypes
import ctypes
import os

# The numpy version of this function tries to be more portable than the native
# ctypes.cdll.LoadLibary or cdtypes.CDLL functions.
_treecorr = numpy.ctypeslib.load_library('_treecorr',os.path.dirname(__file__))


class G2Correlation(treecorr.BinnedCorr2):
    """This class handles the calculation and storage of a 2-point shear-shear correlation
    function.

    It holds the following attributes:

        logr        The nominal center of the bin in log(r).
        meanlogr    The (weighted) mean value of log(r) for the pairs in each bin.
                    If there are no pairs in a bin, then logr will be used instead.
        xip         The correlation function, xi_plus(r).
        xim         The correlation funciton, xi_minus(r).
        xip_im      The imaginary part of xi_plus(r).
        xim_im      The imaginary part of xi_plus(r).
        varxi       The variance of xip and xim, only including the shape noise propagated
                    into the final correlation.  This does not include sample variance, so
                    it is always an underestimate of the actual variance.
        weight      The total weight in each bin.
        npairs      The number of pairs going into each bin.

    The usage pattern is as follows:

        gg = treecorr.G2Correlation(config)
        gg.process(cat1)        # For auto-correlation.
        gg.process(cat1,cat2)   # For cross-correlation.
        gg.write(file_name)     # Write out to a file.
        xip = gg.xip            # Or access the correlation function directly.
    """
    def __init__(self, config=None, logger=None, **kwargs):
        treecorr.BinnedCorr2.__init__(self, config, logger, **kwargs)

        self.xip = numpy.zeros(self.nbins, dtype=float)
        self.xim = numpy.zeros(self.nbins, dtype=float)
        self.xip_im = numpy.zeros(self.nbins, dtype=float)
        self.xim_im = numpy.zeros(self.nbins, dtype=float)

        # an alias
        double_ptr = ctypes.POINTER(ctypes.c_double)

        xip = self.xip.ctypes.data_as(double_ptr)
        xipi = self.xip_im.ctypes.data_as(double_ptr)
        xim = self.xim.ctypes.data_as(double_ptr)
        ximi = self.xim_im.ctypes.data_as(double_ptr)
        meanlogr = self.meanlogr.ctypes.data_as(double_ptr)
        weight = self.weight.ctypes.data_as(double_ptr)
        npairs = self.npairs.ctypes.data_as(double_ptr)

        _treecorr.BuildGGCorr.restype = ctypes.c_void_p
        _treecorr.BuildGGCorr.argtypes = [
            ctypes.c_double, ctypes.c_double, ctypes.c_int, ctypes.c_double, ctypes.c_double,
            double_ptr, double_ptr, double_ptr, double_ptr, double_ptr, double_ptr, double_ptr ]

        self.corr = _treecorr.BuildGGCorr(self.min_sep,self.max_sep,self.nbins,self.bin_size,self.b,
                                          xip,xipi,xim,ximi,meanlogr,weight,npairs);
        self.logger.debug('Finished building GGCorr')
 
    def __del__(self):
        # Using memory allocated from the C layer means we have to explicitly deallocate it
        # rather than being able to rely on the Python memory manager.
        if hasattr(self,'data'):    # In case __init__ failed to get that far
            _treecorr.DestroyGGCorr.argtypes = [ ctypes.c_void_p ]
            _treecorr.DestroyGGCorr(self.corr)

    def process_auto(self, cat1):
        """Process a single catalog, accumulating the auto-correlation.

        This accumulates the weighted sums into the bins, but does not finalize
        the calculation by dividing by the total weight at the end.  After
        calling this function as often as desired, the finalize() command will
        finish the calculation.
        """
        self.logger.info('Starting process G2 auto-correlations for cat %s.',cat1.file_name)
        gfield = cat1.getGField(self.min_sep,self.max_sep,self.b)

        if gfield.sphere:
            _treecorr.ProcessAutoGGSphere.argtypes = [ 
                ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int  ]
            _treecorr.ProcessAutoGGSphere(self.corr, gfield.data, self.output_dots)
        else:
            _treecorr.ProcessAutoGGFlat.argtypes = [
                ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int  ]
            _treecorr.ProcessAutoGGFlat(self.corr, gfield.data, self.output_dots)

    def process_cross(self, cat1, cat2):
        """Process a single pair of catalogs, accumulating the cross-correlation.

        This accumulates the weighted sums into the bins, but does not finalize
        the calculation by dividing by the total weight at the end.  After
        calling this function as often as desired, the finalize() command will
        finish the calculation.
        """
        self.logger.info('Starting process G2 cross-correlations for cats %s, %s.',
                         cat1.file_name, cat2.file_name)
        gfield1 = cat1.getGField(self.min_sep,self.max_sep,self.b)
        gfield2 = cat2.getGField(self.min_sep,self.max_sep,self.b)

        if gfield1.sphere != gfield2.sphere:
            raise AttributeError("Cannot correlate catalogs with different coordinate systems.")

        if gfield1.sphere:
            _treecorr.ProcessCrossGGSphere.argtypes = [ 
                ctypes.c_void_p, ctypes.c_void_p, ctypes.c_voidp, ctypes.c_int ]
            _treecorr.ProcessCrossGGSphere(self.corr, gfield1.data, gfield2.data, self.output_dots)
        else:
            _treecorr.ProcessCrossGGFlat.argtypes = [
                ctypes.c_void_p, ctypes.c_void_p, ctypes.c_voidp, ctypes.c_int ]
            _treecorr.ProcessCrossGGFlat(self.corr, gfield1.data, gfield2.data, self.output_dots)


    def finalize(self, varg1, varg2):
        """Finalize the calculation of the correlation function.

        The process_auto and process_cross commands accumulate values in each bin,
        so they can be called multiple times if appropriate.  Afterwards, this command
        finishes the calculation by dividing each column by the total weight.
        """
        mask1 = self.npairs != 0
        mask2 = self.npairs == 0

        self.xip[mask1] /= self.weight[mask1]
        self.xim[mask1] /= self.weight[mask1]
        self.xip_im[mask1] /= self.weight[mask1]
        self.xim_im[mask1] /= self.weight[mask1]
        self.meanlogr[mask1] /= self.weight[mask1]
        self.varxi[mask1] = varg1 * varg2 / self.npairs[mask1]

        # Update the units of meanlogr
        self.meanlogr[mask1] -= self.log_sep_units

        # Use meanlogr when available, but set to nominal when no pairs in bin.
        self.meanlogr[mask2] = self.logr[mask2]
        self.varxi[mask2] = 0.

    def clear(self):
        """Clear the data vectors
        """
        self.xip[:] = 0
        self.xim[:] = 0
        self.xip_im[:] = 0
        self.xim_im[:] = 0
        self.meanlogr[:] = 0
        self.weight[:] = 0
        self.npairs[:] = 0

    def process(self, cat1, cat2=None):
        """Compute the correlation function.

        If only 1 argument is given, then compute an auto-correlation function.
        If 2 arguments are given, then compute a cross-correlation function.

        Both arguments may be lists, in which case all items in the list are used 
        for that element of the correlation.
        """
        import math
        self.clear()

        if not isinstance(cat1,list): cat1 = [cat1]
        if cat2 is not None and not isinstance(cat2,list): cat2 = [cat2]
        if len(cat1) == 0:
            raise AttributeError("No catalogs provided for cat1")

        if cat2 is None or len(cat2) == 0:
            varg1 = treecorr.calculateVarG(cat1)
            varg2 = varg1
            self.logger.info("varg = %f: sig_sn (per component) = %f",varg1,math.sqrt(varg1))

            if self.config.get('do_auto_corr',False) or len(cat1) == 1:
                for c1 in cat1:
                    self.process_auto(c1)

            if self.config.get('do_cross_corr',True):
                for i,c1 in enumerate(cat1):
                    for c2 in cat1[i+1:]:
                        self.process_cross(c1,c2)
        else:
            varg1 = treecorr.calculateVarG(cat1)
            varg2 = treecorr.calculateVarG(cat2)
            self.logger.info("varg1 = %f: sig_sn (per component) = %f",varg1,math.sqrt(varg1))
            self.logger.info("varg2 = %f: sig_sn (per component) = %f",varg2,math.sqrt(varg2))
            for c1 in cat1:
                for c2 in cat2:
                    self.process_cross(c1,c2)

        self.finalize(varg1,varg2)


    def write(self, file_name):
        """Write the correlation function to the file, file_name.
        """
        self.logger.info('Writing G2 correlations to %s',file_name)
        
        output = numpy.empty( (self.nbins, 9) )
        output[:,0] = numpy.exp(self.logr)
        output[:,1] = numpy.exp(self.meanlogr)
        output[:,2] = self.xip
        output[:,3] = self.xim
        output[:,4] = self.xip_im
        output[:,5] = self.xim_im
        output[:,6] = numpy.sqrt(self.varxi)
        output[:,7] = self.weight
        output[:,8] = self.npairs

        prec = self.config.get('precision',3)
        width = prec+8
        header_form = 8*("{:^%d}."%width) + "{:^%d}"%width
        header = header_form.format('R_nom','<R>','xi+','xi-','xi+_im','xi-_im',
                                    'sigma_xi','weight','npairs')
        fmt = '%%%d.%de'%(width,prec)
        numpy.savetxt(file_name, output, fmt=fmt, header=header)

    def calculateMapSq(self, m2_uform=None):
        """Calculate the aperture mass statistics from the correlation function.

        <Map^2>(R) = int_r=0..2R [1/2 s^2 dlogr (T+(s) xi+(r) + T-(s) xi-(r)) ]
        <Mx^2>(R)  = int_r=0..2R [1/2 s^2 dlogr (T+(s) xi+(r) - T-(s) xi-(r)) ]

        The m2_uform parameter sets which definition of the aperture mass to use.
        The default is to look in the config dict that was used to build the catalog,
        or use 'Crittenden' if it is not specified.

        If m2_uform == 'Crittenden':

            U(r) = 1/2Pi (1-r^2) exp(-r^2/2)
            Q(r) = 1/4Pi r^2 exp(-r^2/2)
            T+(s) = (s^4 - 16s^2 + 32)/128  exp(-s^2/4)
            T-(s) = s^4/128  exp(-s^2/4)

        If m2_uform == 'Schneider':

            U(r) = 9/Pi (1-r^2) (1/3-r^2)
            Q(r) = 6/Pi r^2 (1-r^2)
            T+(s) = 12/5Pi (2-15s^2) arccos(s/2)
                      + 1/(100Pi) s sqrt(4-s^2) (120 + 2320s^2 - 754s^4 + 132s^6 - 9s^8)
            T-(s) = 3/70Pi s^3 (4-s^2)^(7/2)

        cf Schneider, et al (2001): http://xxx.lanl.gov/abs/astro-ph/0112441

        returns a tuple (mapsq, mapsq_im, mxsq, mxsq_im, varmapsq)
        """
        if m2_uform is None:
            m2_uform = self.config.get('m2_uform','Crittenden')
        if m2_uform not in ['Crittenden', 'Schneider']:
            raise ValueError("Invalid m2_uform")

        # Make s a matrix, so we can eventually do the integral by doing a matrix product.
        r = numpy.exp(self.logr)
        meanr = numpy.exp(self.meanlogr) # Use the actual mean r for each bin
        s = numpy.outer(1./r, meanr)  
        ssq = s*s
        if m2_uform == 'Crittenden':
            exp_factor = numpy.exp(-ssq/4.)
            Tp = (32. + ssq*(-16. + ssq))/128. * exp_factor
            Tm = ssq * ssq / 128. * exp_factor
        else:
            Tp = numpy.zeros_like(s)
            Tm = numpy.zeros_like(s)
            sa = s[s<2.]
            ssqa = ssq[s<2.]
            Tp[s<2.] = 12./(5.*numpy.pi) * (2.-15.*ssqa) * numpy.arccos(sa/2.)
            Tp[s<2.] += 1./(100.*numpy.pi) * sa * numpy.sqrt(4.-ssqa) * (
                        120. + ssqa*(2320. + ssqa*(-754. + ssqa*(132. - 9.*ssqa))))
            Tm[s<2.] = 3./(70.*numpy.pi) * sa * ssqa (4.-ssqa)**3.5
        Tp *= ssq
        Tm *= ssq

        # Now do the integral by taking the matrix products.
        # Note that dlogr = bin_size
        Tpxip = Tp.dot(self.xip)
        Tmxim = Tm.dot(self.xim)
        mapsq = (Tpxip + Tmxim) * 0.5 * self.bin_size
        mxsq = (Tpxip - Tmxim) * 0.5 * self.bin_size
        Tpxip_im = Tp.dot(self.xip_im)
        Tmxim_im = Tm.dot(self.xim_im)
        mapsq_im = (Tpxip_im + Tmxim_im) * 0.5 * self.bin_size
        mxsq_im = (Tpxip_im - Tmxim_im) * 0.5 * self.bin_size

        # The variance of each of these is 
        # Var(<Map^2>(R)) = int_r=0..2R [1/4 s^4 dlogr^2 (T+(s)^2 + T-(s)^2) Var(xi)]
        varmapsq = (Tp**2 + Tm**2).dot(self.varxi) * 0.25 * self.bin_size**2

        return mapsq, mapsq_im, mxsq, mxsq_im, varmapsq


    def calculateGamSq(self, eb=False):
        """Calculate the tophat shear variance from the correlation function.

        <Gam^2>(R) = int_r=0..2R [s^2 dlogr S+(s) xi+(r)]
        <Gam^2_E>(R) = int_r=0..2R [1/2 s^2 dlogr (S+(s) xi+(r) + S-(s) xi-(r)]
        <Gam^2_B>(R) = int_r=0..2R [1/2 s^2 dlogr (S+(s) xi+(r) - S-(s) xi-(r)]

        S+(s) = 1/Pi * (4*arccos(s/2) - s sqrt(4-s^2) )
        S-(s) = (s<=2):  1/(Pi s^4) * ( s sqrt(4-s^2) (6-s^2) - 8(3-s^2) arcsin(s/2) )
                (s>=2):  4(s^2-3)/s^4

        cf Schneider, et al (2001): http://xxx.lanl.gov/abs/astro-ph/0112441

        The default behavior is not to compute the E/B versions.  They are calculated if
        eb is set to True.

        returns a tuple (gamsq, vargamsq)                            if eb == False
                or  (gamsq, vargamsq, gamsq_e, gamsq_b, vargamsq_e)  if eb == True
        """
        r = numpy.exp(self.logr)
        meanr = numpy.exp(self.meanlogr) # Use the actual mean r for each bin
        s = numpy.outer(1./r, meanr)  
        ssq = s*s
        Sp = numpy.zeros_like(s)
        sa = s[s<2]
        ssqa = ssq[s<2]
        Sp[s<2.] = 1./numpy.pi * (4.*numpy.arccos(sa/2.) - sa*numpy.sqrt(4.-ssqa))
        Sp *= ssq

        # Now do the integral by taking the matrix products.
        # Note that dlogr = bin_size
        Spxip = Sp.dot(self.xip)
        gamsq = Spxip * self.bin_size
        vargamsq = (Sp**2).dot(self.varxi) * self.bin_size**2

        # Stop here if eb == False
        if not eb: return gamsq, vargamsq

        Sm = numpy.empty_like(s)
        Sm[s<2.] = 1./(ssqa*numpy.pi) * (sa*numpy.sqrt(4.-ssqa)*(6.-ssqa)
                                              -8.*(3.-ssqa)*numpy.arcsin(sa/2.))
        Sm[s>=2.] = 4.*(ssq[s>=2]-3.)/ssq[s>=2]
        # This already includes the extra ssq factor.

        Smxim = Sm.dot(self.xim)
        gamsq_e = (Spxip + Smxim) * 0.5 * self.bin_size
        gamsq_b = (Spxip - Smxim) * 0.5 * self.bin_size
        vargamsq_e = (Sp**2 + Sm**2).dot(self.varxi) * 0.25 * self.bin_size**2

        return gamsq, vargamsq, gamsq_e, gamsq_b, vargamsq_e


    def writeMapSq(self, file_name, m2_uform=None):
        """Write the aperture mass statistics based on the correlation function to the
        file, file_name.

        See calculateMapSq for an explanation of the m2_uform parameter.
        """
        self.logger.info('Writing Map^2 from G2 correlations to %s',file_name)

        mapsq, mapsq_im, mxsq, mxsq_im, varmapsq = self.calculateMapSq(m2_uform=None)
        gamsq, vargamsq = self.calculateGamSq()
 
        output = numpy.empty( (self.nbins, 8) )
        output[:,0] = numpy.exp(self.logr)
        output[:,1] = mapsq
        output[:,2] = mxsq
        output[:,3] = mapsq_im
        output[:,4] = -mxsq_im
        output[:,5] = numpy.sqrt(varmapsq)
        output[:,6] = gamsq
        output[:,7] = numpy.sqrt(vargamsq)

        prec = self.config.get('precision',3)
        width = prec+8
        header_form = 7*("{:^%d}."%width) + "{:^%d}"%width
        header = header_form.format('R','<Map^2>','<Mx^2>','<MMx>(a)','<MMx>(b)','sig_map',
                                    '<Gam^2>','sig_gam')
        fmt = '%%%d.%de'%(width,prec)
        numpy.savetxt(file_name, output, fmt=fmt, header=header)


