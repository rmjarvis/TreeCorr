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

class N2Correlation(treecorr.BinnedCorr2):
    """This class handles the calculation and storage of a 2-point shear-shear correlation
    function.

    It holds the following attributes:

        logr        The nominal center of the bin in log(r).
        meanlogr    The (weighted) mean value of log(r) for the pairs in each bin.
                    If there are no pairs in a bin, then logr will be used instead.
        weight      The total weight in each bin.
        npairs      The number of pairs going into each bin.

    The usage pattern is as follows:

        nn = treecorr.N2Correlation(config)
        nn.process(cat1)        # For auto-correlation.
        nn.process(cat1,cat2)   # For cross-correlation.
        nn.write(file_name)     # Write out to a file.
    """
    def __init__(self, config=None, logger=None, **kwargs):
        treecorr.BinnedCorr2.__init__(self, config, logger=None, **kwargs)

        self.meanlogr = numpy.zeros(self.nbins, dtype=float)
        self.weight = numpy.zeros(self.nbins, dtype=float)
        self.npairs = numpy.zeros(self.nbins, dtype=float)

    def process_auto(self, cat1):
        """Process a single catalog, accumulating the auto-correlation.

        This accumulates the weighted sums into the bins, but does not finalize
        the calculation by dividing by the total weight at the end.  After
        calling this function as often as desired, the finalize() command will
        finish the calculation.
        """
        self.logger.info('Process N2 auto-correlations...  or not.')

    def process_cross(self, cat1, cat2):
        """Process a single pair of catalogs, accumulating the cross-correlation.

        This accumulates the weighted sums into the bins, but does not finalize
        the calculation by dividing by the total weight at the end.  After
        calling this function as often as desired, the finalize() command will
        finish the calculation.
        """
        self.logger.info('Process N2 cross-correlations...  or not.')

    def finalize(self, varg):
        """Finalize the calculation of the correlation function.

        The process_auto and process_cross commands accumulate values in each bin,
        so they can be called multiple times if appropriate.  Afterwards, this command
        finishes the calculation by dividing each column by the total weight.
        """
        mask1 = self.npairs != 0
        mask2 = self.npairs == 0

        self.meanlogr[mask1] /= self.weight[mask1]
        # Use meanlogr when available, but set to nominal when no pairs in bin.
        self.meanlogr[mask2] = self.logr[mask2]

    def clear(self):
        """Clear the data vectors
        """
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
        self.clear()
        if not isinstance(cat1,list): cat1 = [cat1]
        if cat2 is None:
            if self.config.get('do_auto_corr',False) or len(cat1) == 1:
                for c1 in cat1:
                    self.process_auto(c1)

            if self.config.get('do_cross_corr',True):
                for i,c1 in enumerate(cat1):
                    for c2 in cat1[i+1:]:
                        self.process_cross(c1,c2)
        else:
            for c1 in cat1:
                for c2 in cat2:
                    self.process_cross(c1,c2)


    def write(self, file_name, rr, nr=None, rn=None):
        self.logger.info('Writing N2 correlations to %s',file_name)
        
        output = numpy.empty( (self.nbins, 4) )
        output[:,0] = numpy.exp(self.logr)
        output[:,1] = numpy.exp(self.meanlogr)
        output[:,2] = self.weight
        output[:,3] = self.npairs

        prec = self.config.get('precision',3)
        width = prec+8
        header_form = 3*("{:^%d}."%width) + "{:^%d}"%width
        header = header_form.format('R_nom','<R>','weight','npairs')
        fmt = '%%%d.%de'%(width,prec)
        numpy.savetxt(file_name, output, fmt=fmt, header=header)

