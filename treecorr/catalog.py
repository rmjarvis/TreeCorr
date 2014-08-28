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
.. module:: catalog
"""

import treecorr
import numpy

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

    You may also specify any any configuration parameters as kwargs to the Catalog constructor
    if you prefer to either add parameters that are not present in the config dict or to supersede
    values in the config dict.  You can even provide all parameters as kwargs and omit the config
    dict entirely:

        >>> cat1 = treecorr.Catalog(file_name, config, flip_g1=True)
        >>> cat2 = treecorr.Catalog(file_name, ra_col=1, dec_col=2, ra_units='deg', dec_units='deg')

    An alternate way to build a Catalog is to provide the data vectors by hand.  This may
    be more convenient when using this as part of a longer script where you are building the
    data vectors in python and then want to compute the correlation function.  The syntax for this
    is:

        >>> cat = treecorr.Catalog(g1=g1, g2=g2, ra=ra, dec=dec, ra_units='hour', dec_units='deg')

    Each of these data vectors should be a numpy array.  For x,y, the units fields are optional,
    in which case the units are assumed to be arcsec.  But for ra,dec they are required.
    These units parameters may alternatively be provided in the config dict as above.
    Valid columns to include are: x,y,ra,dec,g1,g2,k,w.  As usual, exactly one of (x,y) or
    (ra,dec) must be given.  The others are optional, although if g1 or g2 is given, the other
    must also be given.  Flags may be included by setting w=0 for those objects.

    A Catalog object will have available the following attributes:

        :x:      The x positions, if defined, as a numpy array. (None otherwise)
        :y:      The y positions, if defined, as a numpy array. (None otherwise)
        :ra:     The right ascension, if defined, as a numpy array. (None otherwise)
        :dec:    The declination, if defined, as a numpy array. (None otherwise)
        :r:      The distance, if defined, as a numpy array. (None otherwise)
        :w:      The weights, as a numpy array. (All 1's if no weight column provided.)
        :g1:     The g1 component of the shear, if defined, as a numpy array. (None otherwise)
        :g2:     The g2 component of the shear, if defined, as a numpy array. (None otherwise)
        :k:      The convergence, kappa, if defined, as a numpy array. (None otherwise)

        :nobj:   The number of objects with non-zero weight
        :sumw:   The sum of the weights
        :varg:   The shear variance (aka shape noise) (0 if g1,g2 are not defined)
        :vark:   The kappa variance (0 if k is not defined)

        :name:   When constructed from a file, this will be the file_name.  It is only used as
                 a reference name in logging output  after construction, so if you construct it 
                 from data vectors directly, it will be ''.  You may assign to it if you want to
                 give this catalog a specific name.

    :param file_name:   The name of the catalog file to be read in. (default: None, in which case
                        the columns need to be entered directly with `x`, `y`, etc.)
    :param config:      The configuration dict which defines attributes about how to read the file.
                        Any kwargs that are not those listed here will be added to the config, 
                        so you can even omit the config dict and just enter all parameters you
                        want as kwargs.  (default: None) 
    :param num:         Which number catalog are we reading.  e.g. for NG correlations the catalog
                        for the N has num=0, the one for G has num=1. (default: 0)
    :param logger:      If desired, a logger object for logging. (default: None, in which case
                        one will be built according to the config dict's verbose level.)
    :param is_rand:     If this is a random file, then setting is_rand to True will let them
                        skip k_col, g1_col, and g2_col if they were set for the main catalog.
                        (default: False)
    :param x:           The x column (default: None)
    :param y:           The y column (default: None)
    :param ra:          The ra column (default: None)
    :param dec:         The dec column (default: None)
    :param r:           The r column (default: None)
    :param w:           The w column (default: None)
    :param flag:        The flag column (default: None)
    :param g1:          The g1 column (default: None)
    :param g2:          The g2 column (default: None)
    :param k:           The k column (default: None)
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
            self.x = self.makeArray(x,'x')
            self.y = self.makeArray(y,'y')
            self.ra = self.makeArray(ra,'ra')
            self.dec = self.makeArray(dec,'dec')
            self.r = self.makeArray(r,'r')
            self.w = self.makeArray(w,'w')
            self.flag = self.makeArray(flag,'flag',int)
            self.g1 = self.makeArray(g1,'g1')
            self.g2 = self.makeArray(g2,'g2')
            self.k = self.makeArray(k,'k')


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
        if flip_g1: 
            self.logger.info("   Flipping sign of g1.")
            self.g1 = -self.g1
        if flip_g2:
            self.logger.info("   Flipping sign of g2.")
            self.g2 = -self.g2

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
        if self.g1 is not None: self.g1 = self.g1[start:end]
        if self.g2 is not None: self.g2 = self.g2[start:end]
        if self.k is not None: self.k = self.k[start:end]

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


    def makeArray(self, col, col_str, dtype=float):
        """Turn the input column into a numpy array if it wasn't already.
        Also make sure the input in 1-d.

        :param col:     The input column to be converted into a numpy array.
        :param col_str: The name of the column.  Used only as information in logging output.
        :param dtype:   The dtype for the returned array.  (default: float)

        :returns:       The column converted to a 1-d numpy array.
        """
        if col is not None:
            col = numpy.array(col,dtype=dtype)
            if len(col.shape) != 1:
                s = col.shape
                col = col.reshape(-1)
                self.logger.warn("Warning: Input %s column was not 1-d.",col_str)
                self.logger.warn("         Reshaping from %s to %s",s,col.shape)
        return col


    def checkForNaN(self, col, col_str):
        """Check if the column has any NaNs.  If so, set those rows to have w[k]=0.

        :param col:     The input column to check.
        :param col_str: The name of the column.  Used only as information in logging output.
        """
        if col is not None and any(numpy.isnan(col)):
            index = numpy.where(numpy.isnan(col))[0]
            self.logger.warn("Warning: NaNs found in %s column.  Skipping rows %s.",
                             col_str,str(index.tolist()))
            if self.w is None:
                self.w = numpy.ones_like(col)
            self.w[index] = 0


    def read_ascii(self, file_name, num=0, is_rand=False):
        """Read the catalog from an ASCII file

        :param file_name:   The name of the file to read in.
        :param num:         Which number catalog are we reading. (default: 0)
        :param is_rand:     Is this a random catalog? (default: False)
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

        :param file_name:   The name of the file to read in.
        :param num:         Which number catalog are we reading. (default: 0)
        :param is_rand:     Is this a random catalog? (default: False)
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
        """Read the catalog from a FITS file using the fitsio package

        This is normally not called directly.  Use :meth:`~treecorr.Catalog.read_fits` instead.

        :param file_name:   The name of the file to read in.
        :param num:         Which number catalog are we reading.
        :param is_rand:     Is this a random catalog?
        :param x_col:       The name for x_col
        :param y_col:       The name for y_col
        :param ra_col:      The name for ra_col
        :param dec_col:     The name for dec_col
        :param r_col:       The name for r_col
        :param w_col:       The name for w_col
        :param flag_col:    The name for flat_col
        :param g1_col:      The name for g1_col
        :param g2_col:      The name for g2_col
        :param k_col:       The name for k_col
        """
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
            if g1_col != '0':
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
        """Read the catalog from a FITS file using the pyfits or astropy.io package

        This is normally not called directly.  Use :meth:`~treecorr.Catalog.read_fits` instead.

        :param file_name:   The name of the file to read in.
        :param num:         Which number catalog are we reading.
        :param is_rand:     Is this a random catalog?
        :param x_col:       The name for x_col
        :param y_col:       The name for y_col
        :param ra_col:      The name for ra_col
        :param dec_col:     The name for dec_col
        :param r_col:       The name for r_col
        :param w_col:       The name for w_col
        :param flag_col:    The name for flat_col
        :param g1_col:      The name for g1_col
        :param g2_col:      The name for g2_col
        :param k_col:       The name for k_col
        """
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


    def getNField(self, min_sep, max_sep, b, split_method='mean', logger=None):
        """Return an NField based on the positions in this catalog.

        The NField object is cached, so this is efficient to call multiple times.

        :param min_sep:         The minimum separation between points that will be needed.
        :param max_sep:         The maximum separation between points that will be needed.
        :param b:               The b parameter that will be used for the correlation function.
                                This should be bin_size * bin_slop.
        :param split_method:    Which split method to use ('mean', 'median', or 'middle')
                                (default: 'mean')
        :param logger:          A logger file if desired (default: self.logger)

        :returns:               A :class:`~treecorr.NField` object
        """
        if (not hasattr(self,'nfield') 
            or min_sep != self.nfield.min_sep
            or max_sep != self.nfield.max_sep
            or b != self.nfield.b):

            if logger is None:
                logger = self.logger
            self.nfield = treecorr.NField(self,min_sep,max_sep,b,logger)

        return self.nfield


    def getKField(self, min_sep, max_sep, b, split_method='mean', logger=None):
        """Return a KField based on the k values in this catalog.

        The KField object is cached, so this is efficient to call multiple times.

        :param min_sep:         The minimum separation between points that will be needed.
        :param max_sep:         The maximum separation between points that will be needed.
        :param b:               The b parameter that will be used for the correlation function.
                                This should be bin_size * bin_slop.
        :param split_method:    Which split method to use ('mean', 'median', or 'middle')
                                (default: 'mean')
        :param logger:          A logger file if desired (default: self.logger)

        :returns:               A :class:`~treecorr.KField` object
        """
        if (not hasattr(self,'kfield') 
            or min_sep != self.kfield.min_sep
            or max_sep != self.kfield.max_sep
            or b != self.kfield.b):

            if self.k is None:
                raise AttributeError("k are not defined.")
            if logger is None:
                logger = self.logger
            self.kfield = treecorr.KField(self,min_sep,max_sep,b,logger)

        return self.kfield


    def getGField(self, min_sep, max_sep, b, split_method='mean', logger=None):
        """Return a GField based on the g1,g2 values in this catalog.

        The GField object is cached, so this is efficient to call multiple times.

        :param min_sep:         The minimum separation between points that will be needed.
        :param max_sep:         The maximum separation between points that will be needed.
        :param b:               The b parameter that will be used for the correlation function.
                                This should be bin_size * bin_slop.
        :param split_method:    Which split method to use ('mean', 'median', or 'middle')
                                (default: 'mean')
        :param logger:          A logger file if desired (default: self.logger)

        :returns:               A :class:`~treecorr.GField` object
        """
        if (not hasattr(self,'gfield') 
            or min_sep != self.gfield.min_sep
            or max_sep != self.gfield.max_sep
            or b != self.gfield.b):

            if self.g1 is None or self.g2 is None:
                raise AttributeError("g1,g2 are not defined.")
            if logger is None:
                logger = self.logger
            self.gfield = treecorr.GField(self,min_sep,max_sep,b,logger)

        return self.gfield


    def getNSimpleField(self, logger=None):
        """Return an NSimpleField based on the positions in this catalog.

        The NSimpleField object is cached, so this is efficient to call multiple times.

        :param logger:          A logger file if desired (default: self.logger)

        :returns:               A :class:`~treecorr.NSimpleField` object
        """
        if not hasattr(self,'nsimplefield'):
            if logger is None:
                logger = self.logger
            self.nsimplefield = treecorr.NSimpleField(self,logger)

        return self.nsimplefield


    def getKSimpleField(self, logger=None):
        """Return a KSimpleField based on the k values in this catalog.

        The KSimpleField object is cached, so this is efficient to call multiple times.

        :param logger:          A logger file if desired (default: self.logger)

        :returns:               A :class:`~treecorr.KSimpleField` object
        """
        if not hasattr(self,'ksimplefield'):
            if self.k is None:
                raise AttributeError("k are not defined.")
            if logger is None:
                logger = self.logger
            self.ksimplefield = treecorr.KSimpleField(self,logger)

        return self.ksimplefield


    def getGSimpleField(self, logger=None):
        """Return a GSimpleField based on the g1,g2 values in this catalog.

        The GSimpleField object is cached, so this is efficient to call multiple times.

        :param logger:          A logger file if desired (default: self.logger)

        :returns:               A :class:`~treecorr.GSimpleField` object
        """
        if not hasattr(self,'gsimplefield'):
            if self.g1 is None or self.g2 is None:
                raise AttributeError("g1,g2 are not defined.")
            if logger is None:
                logger = self.logger
            self.gsimplefield = treecorr.GSimpleField(self,logger)

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

        :param file_name:   The name of the file to write to.
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

    :param config:      The configuration dict to use for the appropriate parameters
    :param key:         Which key name to use for the file names. e.g. 'file_name' (default: None)
    :param list_key:    Which key name to use for the name of a list file. e.g. 'file_list'.
                        Either key or list_key is required.  (default: None)
    :param num:         Which number catalog does this correspond to. e.g. file_name should use
                        num=0, file_name2 should use num=1.  (default: 0)
    :param logger:      If desired, a logger object for logging. (default: None, in which case
                        one will be built according to the config dict's verbose level.)
    :param is_rand:     If this is a random file, then setting is_rand to True will let them
                        skip k_col, g1_col, and g2_col if they were set for the main catalog.
                        (default: False)

    :returns:           A list of Catalogs
    """
    if logger is None:
        logger = treecorr.config.setup_logger(
                treecorr.config.get(config,'verbose',int,0), config.get('log_file',None))

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
        if len(file_names) == 0:
            logger.warn('Warning: %s provided, but no names were read from the file %s',
                        list_key, list_file)
            return []
    else:
        # If this key was required (i.e. file_name) then let the caller check this.
        return []
    if is_rand is None:
        if key is not None:
            is_rand = 'rand' in key
        else:
            is_rand = 'rand' in list_key
    if not isinstance(file_names,list): 
        file_names = file_names.split()
        if len(file_names) == 0:
            logger.warn('Warning: %s provided, but it seems to be an empty string',key)
            return []
    return [ Catalog(file_name, config, num, logger, is_rand) for file_name in file_names ]


def calculateVarG(cat_list):
    """Calculate the overall shear variance from a list of catalogs.
        
    The catalogs are assumed to be equivalent, so this is just the average shear
    variance (per component) weighted by the number of objects in each catalog.

    :param cat_list:    A Catalog or a list of Catalogs for which to calculate the shear variance.

    :returns:           The shear variance per component, aka shape noise.
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

    :param cat_list:    A Catalog or a list of Catalogs for which to calculate the kappa variance.

    :returns:           The kappa variance
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


def isGColRequired(config, num):
    """A quick helper function that checks whether we need to bother reading the g1,g2 columns.

    It checks the config dict for the output file names gg_file_name, ng_file_name (only if
    num == 1), etc.  If the output files indicate that we don't need the g1/g2 columns, then
    we don't need to raise an error if the g1_col or g2_col is invalid.
    
    This makes it easier to specify columns. e.g. for an NG correlation function, the 
    first catalog does not need to have the gamma columns, and typically wouldn't.  So
    if you specify g1_col=5, g2_col=6, say, and the first catalog does not have these columns,
    you would normally get an error. 
    
    But instead, we check that the calculation is going to be NG from the presence of an
    ng_file_name parameter, and we let the would-be error pass.

    :param config:  The configuration file to check.
    :param num:     Which number catalog are we working on.

    :returns:       True if some output file requires this catalog to have valid g1/g2 columns,
                    False if not.

    """
    return ( 'gg_file_name' in config
             or 'm2_file_name' in config
             or 'norm_file_name' in config
             or (num==1 and 'ng_file_name' in config)
             or (num==1 and 'nm_file_name' in config)
             or (num==1 and 'kg_file_name' in config) )



def isKColRequired(config, num):
    """A quick helper function that checks whether we need to bother reading the k column.

    The logic here is the same as for :func:`~treecorr.catalog.isGColRequired`, but we check
    for output files that require the kappa column rather than gamma.

    :param config:  The configuration file to check.
    :param num:     Which number catalog are we working on.

    :returns:       True if some output file requires this catalog to have valid g1/g2 columns,
                    False if not.

    """
    return ( 'kk_file_name' in config
             or (num==0 and 'kg_file_name' in config)
             or (num==1 and 'nk_file_name' in config) )


