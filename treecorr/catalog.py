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

import treecorr
import numpy

def isKColRequired(config, num):
    """A quick helper function that checks whether we need to bother reading the k column.
    """
    return ( 'k2_file_name' in config
             or (num==0 and 'kg_file_name' in config)
             or (num==1 and 'nk_file_name' in config) )

def isGColRequired(config, num):
    """A quick helper function that checks whether we need to bother reading the g1,g2 columns.
    """
    return ( 'g2_file_name' in config
             or 'm2_file_name' in config
             or 'norm_file_name' in config
             or (num==1 and 'ng_file_name' in config)
             or (num==1 and 'nm_file_name' in config)
             or (num==1 and 'kg_file_name' in config) )


class Catalog(object):
    """A classs storing the catalog information for a particular set of objects
    to be correlated in some way.

    The usual way to build this is using a config dict:
    
        >>> cat = treecorr.Catalog(file_name, config, num=0)

    This uses the information in the config dict to read the input file, which may be either
    a FITS catalog or an ASCII catalog.  Normally the distinction is made according to the
    file_name's extension, but it can also be set explicitly with config['file_type'].

    See https://github.com/rmjarvis/TreeCorr/wiki/Configuration-Parameters
    for a complete list of all of the relevant configuration parameters.  In particular,
    the first section "Parameters about the input file".

    The num parameter is either 0 or 1 (default 0), which specifies whether this corresponds
    to the first or second file in the configuration file.  e.g. if you are doing an NG
    correlation function, then the first file is not required to have g1_col, g2_col set.
    Thus, you would define these as something like `g1_col = 0 3` and `g2_col = 0 4`.
    Then when reading the first file, you would use num=0 to indicate that you want to use the
    first elements of these vectors (i.e. no g1 or g2 columns), and for the second file, you
    would use num=1 to use columsn 3 and 4.

    An alternate way to build a Catalog is to provide the data vectors by hand.  This may
    be more convenient when using this as part of a longer script where you are building the
    data vectors in python and then want to compute the correlation function.  The syntax for this
    is:

        >>> cat = treecorr.Catalog(g1=g1, g2=g2, ra=ra, dec=dec,
                                   ra_units=ra_units, dec_units=dec_units)

    Each of these data vectors should be a numpy array.  For x,y, the units fields are optional,
    in which case the units are assumed to be arcsec.  But for ra,dec they are required.
    These units parameters may alternatively be provided in the config dict as above.
    Valid columns to include are: x,y,ra,dec,g1,g2,k,w.  As usual, exactly one of (x,y) or
    (ra,dec) must be given.  The others are optional, although if g1 or g2 is given, the other
    must also be given.  Flags may be included by setting w=0 for those objects.

    A Catalog object will have available the following attributes:

        x           The x positions, if defined, as a numpy array. (None otherwise)
        y           The y positions, if defined, as a numpy array. (None otherwise)
        ra          The right ascension, if defined, as a numpy array. (None otherwise)
        dec         The declination, if defined, as a numpy array. (None otherwise)
        r           The distance, if defined, as a numpy array. (None otherwise)
        w           The weights, as a numpy array. (All 1's if no weight column provided.)
        g1          The g1 component of the shear, if defined, as a numpy array. (None otherwise)
        g2          The g2 component of the shear, if defined, as a numpy array. (None otherwise)
        k           The convergence, kappa, if defined, as a numpy array. (None otherwise)

        nobj        The number of objects with non-zero weight
        sumw        The sum of the weights
        varg        The shear variance (aka shape noise)
        vark        The kappa variance
        name        When constructed from a file, this will be the file_name.  It is only used as
                    a reference name in logging output  after construction, so if you construct it 
                    from data vectors directly, it will be ''.  You may assign to it if you want to
                    give this catalog a specific name.
    """
    def __init__(self, file_name=None, config=None, num=0, logger=None, is_rand=False,
                 x=None, y=None, ra=None, dec=None, r=None, w=None, flag=None,
                 g1=None, g2=None, k=None, **kwargs):

        self.config = treecorr.config.merge_config(config,kwargs)
        if logger is not None:
            self.logger = logger
        else:
            self.logger = treecorr.config.setup_logger(
                    treecorr.config.get(self.config,'verbose',int,0),
                    self.config.get('log_file',None))

        # Start with everything set to None.  Overwrite as appropriate.
        self.x = None
        self.y = None
        self.ra = None
        self.dec = None
        self.r = None
        self.w = None
        self.flag = None
        self.g1 = None
        self.g2 = None
        self.k = None

        # First style -- read from a file
        if file_name is not None:
            if self.config is None:
                raise AttributeError("config must be provided when file_name is provided.")
            if any([v is not None for v in [x,y,ra,dec,r,g1,g2,k,w]]):
                raise AttributeError("Vectors may not be provided when file_name is provided.")
            self.name = file_name
            self.logger.info("Reading input file %s",self.name)

            # Figure out which file type the catalog is
            file_type = treecorr.config.get_from_list(self.config,'file_type',num)
            if file_type is None:
                import os
                name, ext = os.path.splitext(file_name)
                if ext.lower().startswith('.fit'):
                    file_type = 'FITS'
                else:
                    file_type = 'ASCII'
                self.logger.info("   file_type assumed to be %s from the file name.",file_type)

            # Read the input file
            if file_type == 'FITS':
                self.read_fits(file_name,num,is_rand)
            else:
                self.read_ascii(file_name,num,is_rand)

        # Second style -- pass in the vectors directly
        else:
            if x is not None or y is not None:
                if x is None or y is None:
                    raise AttributeError("x and y must both be provided")
                if ra is not None or dec is not None:
                    raise AttributeError("ra and dec may not be provided with x,y")
                if r is not None:
                    raise AttributeError("r may not be provided with x,y")
            if ra is not None or dec is not None:
                if ra is None or dec is None:
                    raise AttributeError("ra and dec must both be provided")
            self.name = ''
            if x is not None: self.x = numpy.array(x,dtype=float)
            if y is not None: self.y = numpy.array(y,dtype=float)
            if ra is not None: self.ra = numpy.array(ra,dtype=float)
            if dec is not None: self.dec = numpy.array(dec,dtype=float)
            if r is not None: self.r = numpy.array(r,dtype=float)
            if w is not None: self.w = numpy.array(w,dtype=float)
            if flag is not None: self.flag = numpy.array(flag,dtype=int)
            if g1 is not None: self.g1 = numpy.array(g1,dtype=float)
            if g2 is not None: self.g2 = numpy.array(g2,dtype=float)
            if k is not None: self.k = numpy.array(k,dtype=float)


        # Apply units to x,y,ra,dec
        if self.x is not None:
            if 'x_units' in self.config and not 'y_units' in self.config:
                raise AttributeError("x_units specified without specifying y_units")
            if 'y_units' in self.config and not 'x_units' in self.config:
                raise AttributeError("y_units specified without specifying x_units")
            self.x_units = treecorr.config.get_from_list(self.config,'x_units',num,str,'radians')
            self.y_units = treecorr.config.get_from_list(self.config,'y_units',num,str,'radians')
            self.x *= self.x_units
            self.y *= self.y_units
        else:
            if not self.config.get('ra_units',None):
                raise ValueError("ra_units is required when using ra, dec")
            if not self.config.get('dec_units',None):
                raise ValueError("dec_units is required when using ra, dec")
            self.ra_units = treecorr.config.get_from_list(self.config,'ra_units',num)
            self.dec_units = treecorr.config.get_from_list(self.config,'dec_units',num)
            self.ra *= self.ra_units
            self.dec *= self.dec_units

        # Apply flips if requested
        flip_g1 = treecorr.config.get_from_list(self.config,'flip_g1',num,bool,False)
        flip_g2 = treecorr.config.get_from_list(self.config,'flip_g2',num,bool,False)
        if flip_g1: self.g1 = -self.g1
        if flip_g2: self.g2 = -self.g2

        # Convert the flag to a weight
        if self.flag is not None:
            if 'ignore_flag' in self.config:
                ignore_flag = treecorr.config.get_from_list(self.config,'ignore_flag',num,int)
            else:
                ok_flag = treecorr.config.get_from_list(self.config,'ok_flag',num,int,0)
                ignore_flag = ~ok_flag
            # If we don't already have a weight column, make one with all values = 1.
            if self.w is None:
                self.w = numpy.ones_like(self.flag)
            self.w[(self.flag & ignore_flag)!=0] = 0
            self.logger.debug('Applied flag: w => %s',str(self.w))

        # Update the data according to the specified first and last row
        first_row = treecorr.config.get_from_list(self.config,'first_row',num,int,1)
        if first_row < 1:
            raise ValueError("first_row should be >= 1")
        last_row = treecorr.config.get_from_list(self.config,'last_row',num,int,-1)
        if last_row > 0 and last_row < first_row:
            raise ValueError("last_row should be >= first_row")
        if last_row > 0:
            end = last_row
        else:
            if self.x is not None: 
                end = len(self.x)
            else:
                end = len(self.ra)
        if first_row > 1: 
            start = first_row-1
        else:
            start = 0
        self.logger.debug('start..end = %d..%d',start,end)
        if self.x is not None: self.x = self.x[start:end]
        if self.y is not None: self.y = self.y[start:end]
        if self.ra is not None: self.ra = self.ra[start:end]
        if self.dec is not None: self.dec = self.dec[start:end]
        if self.r is not None: self.r = self.r[start:end]
        if self.w is not None: self.w = self.w[start:end]
        if self.k is not None: self.k = self.k[start:end]
        if self.g1 is not None: self.g1 = self.g1[start:end]
        if self.g2 is not None: self.g2 = self.g2[start:end]

        # Check that all columns have the same length:
        if self.x is not None: 
            nobj = len(self.x)
            if len(self.y) != nobj: 
                raise ValueError("x and y have different numbers of elements")
        else: 
            nobj = len(self.ra)
            if len(self.dec) != nobj: 
                raise ValueError("ra and dec have different numbers of elements")
        if nobj == 0:
            raise RuntimeError("Catalog has no objects!")
        if self.r is not None and len(self.r) != nobj:
            raise ValueError("r has the wrong numbers of elements")
        if self.w is not None and len(self.w) != nobj:
            raise ValueError("w has the wrong numbers of elements")
        if self.g1 is not None and len(self.g1) != nobj:
            raise ValueError("g1 has the wrong numbers of elements")
        if self.g2 is not None and len(self.g1) != nobj:
            raise ValueError("g1 has the wrong numbers of elements")
        if self.k is not None and len(self.k) != nobj:
            raise ValueError("k has the wrong numbers of elements")

        # Check for NaN's:
        self.checkForNaN(self.x,'x')
        self.checkForNaN(self.y,'y')
        self.checkForNaN(self.ra,'ra')
        self.checkForNaN(self.dec,'dec')
        self.checkForNaN(self.r,'r')
        self.checkForNaN(self.g1,'g1')
        self.checkForNaN(self.g2,'g2')
        self.checkForNaN(self.k,'k')
        self.checkForNaN(self.w,'w')

        # Calculate some summary parameters here that will typically be needed
        if self.w is not None:
            self.nobj = numpy.sum(self.w != 0)
            self.sumw = numpy.sum(self.w)
            if self.sumw == 0.:
                raise RuntimeError("Catalog has invalid sumw == 0")
            if self.g1 is not None:
                self.varg = numpy.sum(self.w**2 * (self.g1**2 + self.g2**2))
                # The 2 is because we need the variance _per componenet_.
                self.varg /= 2.*self.sumw**2/self.nobj
            else:
                self.varg = 0.
            if self.k is not None:
                self.vark = numpy.sum(self.w**2 * self.k**2)
                self.vark /= self.sumw**2/self.nobj
            else:
                self.vark = 0.
        else:
            self.nobj = nobj # From above.
            self.sumw = nobj
            if self.g1 is not None:
                self.varg = numpy.sum(self.g1**2 + self.g2**2) / (2.*self.nobj)
            else:
                self.varg = 0.
            if self.k is not None:
                self.vark = numpy.sum(self.k**2) / self.nobj
            else:
                self.vark = 0.
            self.w = numpy.ones( (self.nobj) )

        self.logger.info("   nobj = %d",nobj)


    def checkForNaN(self, col, col_str):
        if col is not None and any(numpy.isnan(col)):
            index = numpy.where(numpy.isnan(col))[0]
            self.logger.warn("Warning: NaNs found in %s column.  Skipping rows %s.",
                             col_str,str(index.tolist()))
            if self.w is None:
                self.w = numpy.ones_like(col)
            self.w[index] = 0


    def read_ascii(self, file_name, num=0, is_rand=False):
        """Read the catalog from an ASCII file
        """
        import numpy
        comment_marker = self.config.get('comment_marker','#')
        delimiter = self.config.get('delimiter',None)
        try:
            import pandas
            # I want read_csv to ignore header lines that start with the comment marker, but
            # there is currently a bug in read_csv that messing things up when we do this.
            # cf. https://github.com/pydata/pandas/issues/4623
            # For now, my workaround in to count how many lines start with the comment marker
            # and skip them by hand.
            skip = 0
            with open(file_name, 'r') as fid:
                for line in fid:
                    if line.startswith(comment_marker): skip += 1
                    else: break
            if delimiter is None:
                data = pandas.read_csv(file_name, comment=comment_marker, delim_whitespace=True,
                                       header=None, skiprows=skip)
            else:
                data = pandas.read_csv(file_name, comment=comment_marker, delimiter=delimiter,
                                       header=None, skiprows=skip)
            data = data.dropna(axis=0).values
        except ImportError:
            self.logger.warn("Unable to import pandas..  Using numpy.genfromtxt instead.")
            self.logger.warn("Installing pandas is recommended for increased speed when "+
                             "reading ASCII catalogs.")
            data = numpy.genfromtxt(file_name, comments=comment_marker, delimiter=delimiter)

        self.logger.debug('read data from %s, num=%d',file_name,num)
        self.logger.debug('data shape = %s',str(data.shape))

        # If only one row, then the shape comes in as one-d.  Reshape it:
        if len(data.shape) == 1:
            data = data.reshape(1,-1)
        if len(data.shape) != 2:
            raise IOError('Unable to parse the input catalog as a 2-d array')
        ncols = data.shape[1]

        # Get the column numbers or names
        x_col = treecorr.config.get_from_list(self.config,'x_col',num,int,0)
        y_col = treecorr.config.get_from_list(self.config,'y_col',num,int,0)
        ra_col = treecorr.config.get_from_list(self.config,'ra_col',num,int,0)
        dec_col = treecorr.config.get_from_list(self.config,'dec_col',num,int,0)
        r_col = treecorr.config.get_from_list(self.config,'r_col',num,int,0)
        w_col = treecorr.config.get_from_list(self.config,'w_col',num,int,0)
        flag_col = treecorr.config.get_from_list(self.config,'flag_col',num,int,0)
        g1_col = treecorr.config.get_from_list(self.config,'g1_col',num,int,0)
        g2_col = treecorr.config.get_from_list(self.config,'g2_col',num,int,0)
        k_col = treecorr.config.get_from_list(self.config,'k_col',num,int,0)

        # Read x,y or ra,dec
        if x_col != 0 or y_col != 0:
            if x_col <= 0 or x_col > ncols:
                raise AttributeError("x_col missing or invalid for file %s"%file_name)
            if y_col <= 0 or y_col > ncols:
                raise AttributeError("y_col missing or invalid for file %s"%file_name)
            if ra_col != 0:
                raise AttributeError("ra_col not allowed in conjunction with x/y cols")
            if dec_col != 0:
                raise AttributeError("dec_col not allowed in conjunction with x/y cols")
            if r_col != 0:
                raise AttributeError("r_col not allowed in conjunction with x/y cols")
            # NB. astype always copies, even if the type is already correct.
            # We actually want this, since it makes the result contiguous in memory, 
            # which we will need.
            self.x = data[:,x_col-1].astype(float)
            self.logger.debug('read x = %s',str(self.x))
            self.y = data[:,y_col-1].astype(float)
            self.logger.debug('read y = %s',str(self.y))
        elif ra_col != 0 or dec_col != 0:
            if ra_col <= 0 or ra_col > ncols:
                raise AttributeError("ra_col missing or invalid for file %s"%file_name)
            if dec_col <= 0 or dec_col > ncols:
                raise AttributeError("dec_col missing or invalid for file %s"%file_name)
            self.ra = data[:,ra_col-1].astype(float)
            self.logger.debug('read ra = %s',str(self.ra))
            self.dec = data[:,dec_col-1].astype(float)
            self.logger.debug('read dec = %s',str(self.dec))
            if r_col != 0:
                self.r = data[:,r_col-1].astype(float)
                self.logger.debug('read r = %s',str(self.r))
        else:
            raise AttributeError("No valid position columns specified for file %s"%file_name)

        # Read w
        if w_col != 0:
            if w_col <= 0 or w_col > ncols:
                raise AttributeError("w_col is invalid for file %s"%file_name)
            self.w = data[:,w_col-1].astype(float)
            self.logger.debug('read w = %s',str(self.w))

        # Read flag 
        if flag_col != 0:
            if flag_col <= 0 or flag_col > ncols:
                raise AttributeError("flag_col is invalid for file %s"%file_name)
            self.flag = data[:,flag_col-1].astype(int)
            self.logger.debug('read flag = %s',str(self.flag))

        # Return here if this file is a random catalog
        if is_rand: return

        # Read g1,g2
        if (g1_col != 0 or g2_col != 0):
            if g1_col <= 0 or g1_col > ncols or g2_col <= 0 or g2_col > ncols:
                if isGColRequired(self.config,num):
                    raise AttributeError("g1_col, g2_col are invalid for file %s"%file_name)
                else:
                    self.logger.warn("Warning: skipping g1_col, g2_col for %s, num=%d",
                                     file_name,num)
                    self.logger.warn("because they are invalid, but unneeded.")
            else:
                self.g1 = data[:,g1_col-1].astype(float)
                self.logger.debug('read g1 = %s',str(self.g1))
                self.g2 = data[:,g2_col-1].astype(float)
                self.logger.debug('read g2 = %s',str(self.g2))

        # Read k
        if k_col != 0:
            if k_col <= 0 or k_col > ncols:
                if isKColRequired(self.config,num):
                    raise AttributeError("k_col is invalid for file %s"%file_name)
                else:
                    self.logger.warn("Warning: skipping k_col for %s, num=%d",file_name,num)
                    self.logger.warn("because it is invalid, but unneeded.")
            else:
                self.k = data[:,k_col-1].astype(float)
                self.logger.debug('read k = %s',str(self.k))


    def read_fits(self, file_name, num=0, is_rand=False):
        """Read the catalog from a FITS file
        """
        # Get the column names
        x_col = treecorr.config.get_from_list(self.config,'x_col',num,str,'0')
        y_col = treecorr.config.get_from_list(self.config,'y_col',num,str,'0')
        ra_col = treecorr.config.get_from_list(self.config,'ra_col',num,str,'0')
        dec_col = treecorr.config.get_from_list(self.config,'dec_col',num,str,'0')
        r_col = treecorr.config.get_from_list(self.config,'r_col',num,str,'0')
        w_col = treecorr.config.get_from_list(self.config,'w_col',num,str,'0')
        flag_col = treecorr.config.get_from_list(self.config,'flag_col',num,str,'0')
        g1_col = treecorr.config.get_from_list(self.config,'g1_col',num,str,'0')
        g2_col = treecorr.config.get_from_list(self.config,'g2_col',num,str,'0')
        k_col = treecorr.config.get_from_list(self.config,'k_col',num,str,'0')

        # Check that position cols are valid:
        if x_col != '0' or y_col != '0':
            if x_col == '0':
                raise AttributeError("x_col missing for file %s"%file_name)
            if y_col == '0':
                raise AttributeError("y_col missing for file %s"%file_name)
            if ra_col != '0':
                raise AttributeError("ra_col not allowed in conjunction with x/y cols")
            if dec_col != '0':
                raise AttributeError("dec_col not allowed in conjunction with x/y cols")
            if r_col != '0':
                raise AttributeError("r_col not allowed in conjunction with x/y cols")
        elif ra_col != '0' or dec_col != '0':
            if ra_col == '0':
                raise AttributeError("ra_col missing for file %s"%file_name)
            if dec_col == '0':
                raise AttributeError("dec_col missing for file %s"%file_name)
        else:
            raise AttributeError("No valid position columns specified for file %s"%file_name)

        # Check that g1,g2,k cols are valid
        if g1_col == '0' and isGColRequired(self.config,num):
            raise AttributeError("g1_col is missing for file %s"%file_name)
        if g2_col == '0' and isGColRequired(self.config,num):
            raise AttributeError("g2_col is missing for file %s"%file_name)
        if k_col == '0' and isKColRequired(self.config,num):
            raise AttributeError("k_col is missing for file %s"%file_name)

        if (g1_col != '0' and g2_col == '0') or (g1_col == '0' and g2_col != '0'):
            raise AttributeError("g1_col, g2_col are invalid for file %s"%file_name)

        # OK, now go ahead and read all the columns.
        try:
            self.read_fitsio(file_name, num, is_rand,
                             x_col, y_col, ra_col, dec_col, r_col, w_col, flag_col,
                             g1_col, g2_col, k_col)
        except ImportError:
            self.read_pyfits(file_name, num, is_rand,
                             x_col, y_col, ra_col, dec_col, r_col, w_col, flag_col,
                             g1_col, g2_col, k_col)


    def read_fitsio(self, file_name, num, is_rand,
                    x_col, y_col, ra_col, dec_col, r_col, w_col, flag_col,
                    g1_col, g2_col, k_col):
        import fitsio

        hdu = treecorr.config.get_from_list(self.config,'hdu',num,int,1)

        with fitsio.FITS(file_name, 'r') as fits:

            # Read x,y or ra,dec,r
            if x_col != '0':
                x_hdu = treecorr.config.get_from_list(self.config,'x_hdu',num,int,hdu)
                y_hdu = treecorr.config.get_from_list(self.config,'y_hdu',num,int,hdu)
                if x_col not in fits[x_hdu].get_colnames():
                    raise AttributeError("x_col is invalid for file %s"%file_name)
                if y_col not in fits[y_hdu].get_colnames():
                    raise AttributeError("y_col is invalid for file %s"%file_name)
                self.x = fits[x_hdu].read_column(x_col).astype(float)
                self.logger.debug('read x = %s',str(self.x))
                self.y = fits[y_hdu].read_column(y_col).astype(float)
                self.logger.debug('read y = %s',str(self.y))
            else:
                ra_hdu = treecorr.config.get_from_list(self.config,'ra_hdu',num,int,hdu)
                dec_hdu = treecorr.config.get_from_list(self.config,'dec_hdu',num,int,hdu)
                if ra_col not in fits[ra_hdu].get_colnames():
                    raise AttributeError("ra_col is invalid for file %s"%file_name)
                if dec_col not in fits[dec_hdu].get_colnames():
                    raise AttributeError("dec_col is invalid for file %s"%file_name)
                self.ra = fits[ra_hdu].read_column(ra_col).astype(float)
                self.logger.debug('read ra = %s',str(self.ra))
                self.dec = fits[dec_hdu].read_column(dec_col).astype(float)
                self.logger.debug('read dec = %s',str(self.dec))
                if r_col != '0':
                    r_hdu = treecorr.config.get_from_list(self.config,'r_hdu',num,int,hdu)
                    if r_col not in fits[r_hdu].get_colnames():
                        raise AttributeError("r_col is invalid for file %s"%file_name)
                    self.r = fits[r_hdu].read_column(r_col).astype(float)
                    self.logger.debug('read r = %s',str(self.r))

            # Read w
            if w_col != '0':
                w_hdu = treecorr.config.get_from_list(self.config,'w_hdu',num,int,hdu)
                if w_col not in fits[w_hdu].get_colnames():
                    raise AttributeError("w_col is invalid for file %s"%file_name)
                self.w = fits[w_hdu].read_column(w_col).astype(float)
                self.logger.debug('read w = %s',str(self.w))

            # Read flag
            if flag_col != '0':
                flag_hdu = treecorr.config.get_from_list(self.config,'flag_hdu',num,int,hdu)
                if flag_col not in fits[flag_hdu].get_colnames():
                    raise AttributeError("flag_col is invalid for file %s"%file_name)
                self.flag = fits[flag_hdu].read_column(flag_col).astype(int)
                self.logger.debug('read flag = %s',str(self.flag))

            # Return here if this file is a random catalog
            if is_rand: return

            # Read g1,g2
            if g1_col != '0' and g2_col != '0':
                g1_hdu = treecorr.config.get_from_list(self.config,'g1_hdu',num,int,hdu)
                g2_hdu = treecorr.config.get_from_list(self.config,'g2_hdu',num,int,hdu)
                if (g1_col not in fits[g1_hdu].get_colnames() or
                    g2_col not in fits[g2_hdu].get_colnames()):
                    if isGColRequired(self.config,num):
                        raise AttributeError("g1_col, g2_col are invalid for file %s"%file_name)
                    else:
                        self.logger.warn("Warning: skipping g1_col, g2_col for %s, num=%d",
                                        file_name,num)
                        self.logger.warn("because they are invalid, but unneeded.")
                else:
                    self.g1 = fits[g1_hdu].read_column(g1_col).astype(float)
                    self.logger.debug('read g1 = %s',str(self.g1))
                    self.g2 = fits[g2_hdu].read_column(g2_col).astype(float)
                    self.logger.debug('read g2 = %s',str(self.g2))

            # Read k
            if k_col != '0':
                k_hdu = treecorr.config.get_from_list(self.config,'k_hdu',num,int,hdu)
                if k_col not in fits[k_hdu].get_colnames():
                    if isKColRequired(self.config,num):
                        raise AttributeError("k_col is invalid for file %s"%file_name)
                    else:
                        self.logger.warn("Warning: skipping k_col for %s, num=%d",file_name,num)
                        self.logger.warn("because it is invalid, but unneeded.")
                else:
                    self.k = fits[k_hdu].read_column(k_col).astype(float)
                    self.logger.debug('read k = %s',str(self.k))

 
    def read_pyfits(self, file_name, num, is_rand,
                    x_col, y_col, ra_col, dec_col, r_col, w_col, flag_col,
                    g1_col, g2_col, k_col):
        try:
            import astropy.io.fits as pyfits
        except:
            import pyfits

        with pyfits.open(file_name, 'readonly') as hdu_list:

            hdu = treecorr.config.get_from_list(self.config,'hdu',num,int,1)

            # Read x,y or ra,dec,r
            if x_col != '0' or y_col != '0':
                x_hdu = treecorr.config.get_from_list(self.config,'x_hdu',num,int,hdu)
                y_hdu = treecorr.config.get_from_list(self.config,'y_hdu',num,int,hdu)
                if x_col not in hdu_list[x_hdu].columns.names:
                    raise AttributeError("x_col is invalid for file %s"%file_name)
                if y_col not in hdu_list[y_hdu].columns.names:
                    raise AttributeError("y_col is invalid for file %s"%file_name)
                self.x = hdu_list[x_hdu].data.field(x_col).astype(float)
                self.logger.debug('read x = %s',str(self.x))
                self.y = hdu_list[y_hdu].data.field(y_col).astype(float)
                self.logger.debug('read y = %s',str(self.y))
            elif ra_col != '0' or dec_col != '0':
                ra_hdu = treecorr.config.get_from_list(self.config,'ra_hdu',num,int,hdu)
                dec_hdu = treecorr.config.get_from_list(self.config,'dec_hdu',num,int,hdu)
                if ra_col not in hdu_list[ra_hdu].columns.names:
                    raise AttributeError("ra_col is invalid for file %s"%file_name)
                if dec_col not in hdu_list[dec_hdu].columns.names:
                    raise AttributeError("dec_col is invalid for file %s"%file_name)
                self.ra = hdu_list[ra_hdu].data.field(ra_col).astype(float)
                self.logger.debug('read ra = %s',str(self.ra))
                self.dec = hdu_list[dec_hdu].data.field(dec_col).astype(float)
                self.logger.debug('read dec = %s',str(self.dec))
                if r_col != '0':
                    r_hdu = treecorr.config.get_from_list(self.config,'r_hdu',num,int,hdu)
                    if r_col not in hdu_list[r_hdu].columns.names:
                        raise AttributeError("r_col is invalid for file %s"%file_name)
                    self.r = hdu_list[r_hdu].data.field(r_col).astype(float)
                    self.logger.debug('read r = %s',str(self.r))

            # Read w
            if w_col != '0':
                w_hdu = treecorr.config.get_from_list(self.config,'w_hdu',num,int,hdu)
                if w_col not in hdu_list[w_hdu].columns.names:
                    raise AttributeError("w_col is invalid for file %s"%file_name)
                self.w = hdu_list[w_hdu].data.field(w_col).astype(float)
                self.logger.debug('read w = %s',str(self.w))

            # Read flag
            if flag_col != '0':
                flag_hdu = treecorr.config.get_from_list(self.config,'flag_hdu',num,int,hdu)
                if flag_col not in hdu_list[flag_hdu].columns.names:
                    raise AttributeError("flag_col is invalid for file %s"%file_name)
                self.flag = hdu_list[flag_hdu].data.field(flag_col).astype(int)
                self.logger.debug('read flag = %s',str(self.flag))

            # Return here if this file is a random catalog
            if is_rand: return

            # Read g1,g2
            if g1_col != '0':
                g1_hdu = treecorr.config.get_from_list(self.config,'g1_hdu',num,int,hdu)
                g2_hdu = treecorr.config.get_from_list(self.config,'g2_hdu',num,int,hdu)
                if (g1_col not in hdu_list[g1_hdu].columns.names or
                    g2_col not in hdu_list[g2_hdu].columns.names):
                    if isGColRequired(self.config,num):
                        raise AttributeError("g1_col, g2_col are invalid for file %s"%file_name)
                    else:
                        self.logger.warn("Warning: skipping g1_col, g2_col for %s, num=%d",
                                        file_name,num)
                        self.logger.warn("because they are invalid, but unneeded.")
                else:
                    self.g1 = hdu_list[g1_hdu].data.field(g1_col).astype(float)
                    self.logger.debug('read g1 = %s',str(self.g1))
                    self.g2 = hdu_list[g2_hdu].data.field(g2_col).astype(float)
                    self.logger.debug('read g2 = %s',str(self.g2))

            # Read k
            if k_col != '0':
                k_hdu = treecorr.config.get_from_list(self.config,'k_hdu',num,int,hdu)
                if k_col not in hdu_list[k_hdu].columns.names:
                    if isKColRequired(self.config,num):
                        raise AttributeError("k_col is invalid for file %s"%file_name)
                    else:
                        self.logger.warn("Warning: skipping k_col for %s, num=%d",file_name,num)
                        self.logger.warn("because it is invalid, but unneeded.")
                else:
                    self.k = hdu_list[k_hdu].data.field(k_col).astype(float)
                    self.logger.debug('read k = %s',str(self.k))


    def getNField(self, min_sep, max_sep, b, logger=None, config=None):
        """Return an NField based on the positions in this catalog.

        The NField object is cached, so this is efficient to call multiple times.
        """
        if (not hasattr(self,'nfield') 
            or min_sep != self.nfield.min_sep
            or max_sep != self.nfield.max_sep
            or b != self.nfield.b):

            if logger is None:
                logger = self.logger
            if config is None:
                config = self.config
            self.nfield = treecorr.NField(self,min_sep,max_sep,b,logger,config)

        return self.nfield


    def getKField(self, min_sep, max_sep, b, logger=None, config=None):
        """Return a KField based on the k values in this catalog.

        The KField object is cached, so this is efficient to call multiple times.
        """
        if (not hasattr(self,'kfield') 
            or min_sep != self.kfield.min_sep
            or max_sep != self.kfield.max_sep
            or b != self.kfield.b):

            if self.k is None:
                raise AttributeError("k are not defined.")
            if logger is None:
                logger = self.logger
            if config is None:
                config = self.config
            self.kfield = treecorr.KField(self,min_sep,max_sep,b,logger,config)

        return self.kfield


    def getGField(self, min_sep, max_sep, b, logger=None, config=None):
        """Return a GField based on the g1,g2 values in this catalog.

        The GField object is cached, so this is efficient to call multiple times.
        """
        if (not hasattr(self,'gfield') 
            or min_sep != self.gfield.min_sep
            or max_sep != self.gfield.max_sep
            or b != self.gfield.b):

            if self.g1 is None or self.g2 is None:
                raise AttributeError("g1,g2 are not defined.")
            if logger is None:
                logger = self.logger
            if config is None:
                config = self.config
            self.gfield = treecorr.GField(self,min_sep,max_sep,b,logger,config)

        return self.gfield


    def getNSimpleField(self, logger=None, config=None):
        """Return an NSimpleField based on the positions in this catalog.

        The NSimpleField object is cached, so this is efficient to call multiple times.
        """
        if not hasattr(self,'nsimplefield'):
            if logger is None:
                logger = self.logger
            if config is None:
                config = self.config
            self.nsimplefield = treecorr.NSimpleField(self,logger,config)

        return self.nsimplefield


    def getKSimpleField(self, logger=None, config=None):
        """Return a KSimpleField based on the k values in this catalog.

        The KSimpleField object is cached, so this is efficient to call multiple times.
        """
        if not hasattr(self,'ksimplefield'):
            if self.k is None:
                raise AttributeError("k are not defined.")
            if logger is None:
                logger = self.logger
            if config is None:
                config = self.config
            self.ksimplefield = treecorr.KSimpleField(self,logger,config)

        return self.ksimplefield


    def getGSimpleField(self, logger=None, config=None):
        """Return a GSimpleField based on the g1,g2 values in this catalog.

        The GSimpleField object is cached, so this is efficient to call multiple times.
        """
        if not hasattr(self,'gsimplefield'):
            if self.g1 is None or self.g2 is None:
                raise AttributeError("g1,g2 are not defined.")
            if logger is None:
                logger = self.logger
            if config is None:
                config = self.config
            self.gsimplefield = treecorr.GSimpleField(self,logger,config)

        return self.gsimplefield

    def write(self, file_name):
        """Write the catalog to a file.  Currently only ASCII output is supported.

        Note that the x,y,ra,dec columns are output using the same units as were used when
        building the Catalog.  If you want to use a different unit, you can set the catalog's
        units directly before writing.  e.g.:

            >>> cat = treecorr.Catalog('cat.dat', ra=ra, dec=dec,
                                       ra_units='hours', dec_units='degrees')
            >>> cat.ra_units = treecorr.degrees
            >>> cat.write('new_cat.dat')
        """
        import numpy

        col_names = []
        columns = []
        if self.x is not None:
            col_names.append('x')
            columns.append(self.x / self.x_units)
        if self.y is not None:
            col_names.append('y')
            columns.append(self.y / self.y_units)
        if self.ra is not None:
            col_names.append('ra')
            columns.append(self.ra / self.ra_units)
        if self.dec is not None:
            col_names.append('dec')
            columns.append(self.dec / self.dec_units)
        if self.r is not None:
            col_names.append('r')
            columns.append(self.r)
        if self.w is not None:
            col_names.append('w')
            columns.append(self.w)
        if self.g1 is not None:
            col_names.append('g1')
            columns.append(self.g1)
        if self.g2 is not None:
            col_names.append('g2')
            columns.append(self.g2)
        if self.k is not None:
            col_names.append('k')
            columns.append(self.k)

        ncol = len(col_names)
        data = numpy.empty( (self.nobj, ncol) )
        for i,col in enumerate(columns):
            data[:,i] = col

        prec = treecorr.config.get(self.config,'cat_precision',int,16)
        width = prec+8
        header_form = "{0:^%d}"%(width-1)
        for i in range(1,ncol):
            header_form += ".{%d:^%d}"%(i,width)
        header = header_form.format(*col_names)
        fmt = '%%%d.%de'%(width,prec)
        try:
            numpy.savetxt(file_name, data, fmt=fmt, header=header)
        except (AttributeError, TypeError):
            # header was added with version 1.7, so do it by hand if not available.
            with open(file_name, 'w') as fid:
                fid.write('#' + header + '\n')
                numpy.savetxt(fid, data, fmt=fmt) 


def read_catalogs(config, key=None, list_key=None, num=0, logger=None, is_rand=None):
    """Read in a list of catalogs for the given key.

    key should be the file_name parameter or similar key word.
    list_key should be be corresponging file_list parameter, if appropriate.
    At least one of key or list_key must be provided.  If both are provided, then only
    one of these should be in the config dict.

    If neither key nor list_key is found in the config dict, then a null list [] is returned.

    num indicates which key to use if any of the fields like x_col, flip_g1, etc. are lists.
    The default is 0, which means to use the first item in the list if they are lists.
    """
    if key is None and list_key is None:
        raise AttributeError("Must provide either %s or %s."%(key,list_key))
    if key is not None and key in config:
        if list_key in config:
            raise AttributeError("Cannot provide both %s and %s."%(key,list_key))
        file_names = config[key]
    elif list_key is not None and list_key in config:
        list_file = config[list_key]
        with open(list_file,'r') as fin:
            file_names = [ f.strip() for f in fin ]
    else:
        # If this key was required (i.e. file_name) then let the caller check this.
        return []
    if is_rand is None:
        if key is not None:
            is_rand = 'rand' in key
        else:
            is_rand = 'rand' in list_key
    if not isinstance(file_names,list): file_names = [ file_names ]
    return [ Catalog(file_name, config, num, logger, is_rand) for file_name in file_names ]


def calculateVarG(cat_list):
    """Calculate the overall shear variance from a list of catalogs.
        
    The catalogs are assumed to be equivalent, so this is just the average shear
    variance weighted by the number of objects in each catalog.
    """
    if len(cat_list) == 1:
        return cat_list[0].varg
    else:
        varg = 0
        ntot = 0
        for cat in cat_list:
            varg += cat.varg * cat.nobj
            ntot += cat.nobj
        return varg / ntot

def calculateVarK(cat_list):
    """Calculate the overall kappa variance from a list of catalogs.
        
    The catalogs are assumed to be equivalent, so this is just the average kappa
    variance weighted by the number of objects in each catalog.
    """
    if len(cat_list) == 1:
        return cat_list[0].vark
    else:
        vark = 0
        ntot = 0
        for cat in cat_list:
            vark += cat.vark * cat.nobj
            ntot += cat.nobj
        return vark / ntot


