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

        >>> cat = treecorr.Catalog(g1=g1, g2=g2, ra=ra, dec=dec)
    
    Each of these data vectors should be a numpy array.  The units for x,y,ra,dec should be
    in radians.  Valid columns to include are: x,y,ra,dec,g1,g2,k,w.  As usual, exactly one of
    (x,y) or (ra,dec) must be given.  The others are optional, although if g1 or g2 is given,
    the other must also be given.  Flags may be included by setting w=0 for those objects.

    A Catalog object will have available the following attributes:

        x           The x positions, if defined, as a numpy array. (None otherwise)
        y           The y positions, if defined, as a numpy array. (None otherwise)
        ra          The right ascension, if defined, as a numpy array. (None otherwise)
        dec         The declination, if defined, as a numpy array. (None otherwise)
        g1          The g1 component of the shear, if defined, as a numpy array. (None otherwise)
        g2          The g2 component of the shear, if defined, as a numpy array. (None otherwise)
        k           The convergence, kappa, if defined, as a numpy array. (None otherwise)
        w           The weights, as a numpy array. (All 1's if no weight column provided.)

        nobj        The number of objects with non-zero weight
        sumw        The sum of the weights
        varg        The shear variance (aka shape noise)
        vark        The kappa variance
        file_name   This is only used as a name after construction, so if you construct it from
                    data vectors directly, it will be ''.  You may assign to it if you want to
                    give this catalog a different name.
    """
    def __init__(self, file_name=None, config=None, num=0, logger=None, is_rand=False,
                 x=None, y=None, ra=None, dec=None, g1=None, g2=None, k=None, w=None):

        if config is None: config = {}
        if logger is not None:
            self.logger = logger
        else:
            self.logger = treecorr.config.setup_logger(config.get('verbose',0),
                                                       config.get('log_file',None))

        # First style -- read from a file
        if file_name is not None:
            if config is None:
                raise AttributeError("config must be provided when file_name is provided.")
            if any([v is not None for v in [x,y,ra,dec,g1,g2,k,w]]):
                raise AttributeError("Vectors may not be provided when file_name is provided.")
            self.file_name = file_name
            self.config = config
            logger.info("Reading input file %s",self.file_name)

            # Figure out which file type the catalog is
            file_type = treecorr.config.get_from_list(self.config,'file_type',num)
            if file_type is None:
                import os
                name, ext = os.path.splitext(file_name)
                if ext.lower().startswith('.fit'):
                    file_type = 'FITS'
                else:
                    file_type = 'ASCII'
                logger.info("   file_type assumed to be %s from the file name.",file_type)

            # Read the input file
            if file_type == 'FITS':
                self.read_fits(file_name,num,is_rand)
            else:
                self.read_ascii(file_name,num,is_rand)

            # Apply units to x,y,ra,dec
            if self.x is not None:
                if 'x_units' in self.config and not 'y_units' in self.config:
                    raise AttributeError("x_units specified without specifying y_units")
                if 'y_units' in self.config and not 'x_units' in self.config:
                    raise AttributeError("y_units specified without specifying x_units")
                x_units = treecorr.config.get_from_list(self.config,'x_units',num,str,'arcsec')
                y_units = treecorr.config.get_from_list(self.config,'y_units',num,str,'arcsec')
                self.x *= treecorr.angle_units[x_units]
                self.y *= treecorr.angle_units[y_units]
            else:
                if not self.config['ra_units']:
                    raise ValueError("ra_units is required when using ra, dec")
                if not self.config['dec_units']:
                    raise ValueError("dec_units is required when using ra, dec")
                ra_units = treecorr.config.get_from_list(self.config,'ra_units',num,str,'arcsec')
                dec_units = treecorr.config.get_from_list(self.config,'dec_units',num,str,'arcsec')
                self.ra *= treecorr.angle_units[ra_units]
                self.dec *= treecorr.angle_units[dec_units]

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
                self.w[self.w.where(self.flag & ignore_flag)] = 0
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
            if self.w is not None: self.w = self.w[start:end]
            if self.k is not None: self.k = self.k[start:end]
            if self.g1 is not None: self.g1 = self.g1[start:end]
            if self.g2 is not None: self.g2 = self.g2[start:end]

            # Project if requested
            if self.config.get('project',False):
                if self.ra is None:
                    raise AttributeError("project is invalid without ra, dec")
                self.logger.warn("Warning: You probably should not use the project option.")
                self.logger.warn("It is mostly just for testing the accuracy of the sphererical")
                self.logger.warn("geometry code.  But that is working well, so you should probably")
                self.logger.warn("just let TreeCorr handle the spherical geometry for you, rather")
                self.logger.warn("than project onto a tangent plane.")

                if params['project_ra']:
                    ra_cen = param['project_ra']*treecorr.angle_units[self.config['ra_units']]
                else:
                    ra_cen = ra.mean()
                if params['project_dec']:
                    dec_cen = param['project_dec']*treecorr.angle_units[self.config['dec_units']]
                else:
                    dec_cen = dec.mean()
                
                cen = CelestialCoord(ra_cen, dec_cen)
                projection = self.config.get('projection','lambert')
                self.x, self.y = cen.project_rad(self.ra, self.dec, projection)
                self.ra = None
                self.dec = None

        # Second style -- pass in the vectors directly
        else:
            if x is not None or y is not None:
                if x is None or y is None:
                    raise AttributeError("x and y must both be provided")
                if ra is not None or dec is not None:
                    raise AttributeError("ra and dec may not be provided with x,y")
            if ra is not None or dec is not None:
                if ra is None or dec is None:
                    raise AttributeError("ra and dec must both be provided")
            self.file_name = ''
            self.config = config  # May be None.
            self.x = x
            self.y = y
            self.ra = ra
            self.dec = dec
            self.w = w
            self.g1 = g1
            self.g2 = g2
            self.k = k

        # Check that all columns have the same length:
        if self.x is not None: 
            nobj = len(self.x)
            if len(self.y) != nobj: 
                raise ValueError("x and y have different numbers of elements")
        else: 
            nobj = len(self.ra)
            if len(self.dec) != nobj: 
                raise ValueError("ra and dec have different numbers of elements")
        if self.w is not None and len(self.w) != nobj:
            raise ValueError("w has the wrong numbers of elements")
        if self.g1 is not None and len(self.g1) != nobj:
            raise ValueError("g1 has the wrong numbers of elements")
        if self.g2 is not None and len(self.g1) != nobj:
            raise ValueError("g1 has the wrong numbers of elements")
        if self.k is not None and len(self.k) != nobj:
            raise ValueError("k has the wrong numbers of elements")

        # Calculate some summary parameters here that will typically be needed
        if self.w is not None:
            self.nobj = numpy.sum(self.w != 0)
            self.sumw = numpy.sum(self.w)
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
        logger.info("   nobj = %d",nobj)

    def read_ascii(self, file_name, num, is_rand):
        """Read the catalog from an ASCII file
        """
        import numpy
        comment_marker = self.config.get('comment_marker','#')
        delimiter = self.config.get('delimiter',None)
        try:
            import pandas
            if delimiter is None:
                data = pandas.read_csv(file_name, comment=comment_marker, header=None,
                                       delim_whitespace=True)
            else:
                data = pandas.read_csv(file_name, comment=comment_marker, header=None,
                                       delimiter=delimiter)
            data = data.as_matrix()
        except ImportError:
            self.logger.warn("Unable to import pandas..  Using numpy.genfromtxt instead.")
            self.logger.warn("Installing pandas is recommended for increased speed.")
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
        w_col = treecorr.config.get_from_list(self.config,'w_col',num,int,0)
        flag_col = treecorr.config.get_from_list(self.config,'flag_col',num,int,0)
        g1_col = treecorr.config.get_from_list(self.config,'g1_col',num,int,0)
        g2_col = treecorr.config.get_from_list(self.config,'g2_col',num,int,0)
        k_col = treecorr.config.get_from_list(self.config,'k_col',num,int,0)

        # Start with everything set to None.  Overwrite as appropriate.
        self.x = None
        self.y = None
        self.ra = None
        self.dec = None
        self.w = None
        self.flag = None
        self.g1 = None
        self.g2 = None
        self.k = None

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
            self.flag = data[:,flag_col-1].astype(float)
            self.logger.debug('read flag = %s',str(self.flag))

        # Return here if this file is a random catalog
        if is_rand: return

        # Read g1,g2
        if (g1_col != 0 or g2_col != 0) and isGColRequired(self.config,num):
            if g1_col <= 0 or g1_col > ncols:
                raise AttributeError("g1_col is missing or invalid for file %s"%file_name)
            if g2_col <= 0 or g2_col > ncols:
                raise AttributeError("g2_col is missing or invalid for file %s"%file_name)
            self.g1 = data[:,g1_col-1].astype(float)
            self.logger.debug('read g1 = %s',str(self.g1))
            self.g2 = data[:,g2_col-1].astype(float)
            self.logger.debug('read g2 = %s',str(self.g2))

        # Read k
        if k_col != 0 and isKColRequired(self.config,num):
            if k_col <= 0 or k_col > ncols:
                raise AttributeError("k_col is invalid for file %s"%file_name)
            self.k = data[:,k_col-1].astype(float)
            self.logger.debug('read k = %s',str(self.k))

    def read_fits(self, file_name, num, is_rand):
        """Read the catalog from a FITS file
        """
        try:
            import fitsio
            self.read_fitsio(file_name, num, is_rand)
        except:
            self.read_pyfits(file_name, num, is_rand)

    def read_fitsio(self, file_name, num, is_rand):
        raise NotImplementedError("Using fitsio rather than pyfits not yet implemented")

    def read_pyfits(self, file_name, num, is_rand):
        try:
            import astropy.io.fits as pyfits
        except:
            import pyfits
        hdu_list = pyfits.open(file_name, 'readonly')

        # Get the column names
        x_col = treecorr.config.get_from_list(self.config,'x_col',num,str,'0')
        y_col = treecorr.config.get_from_list(self.config,'y_col',num,str,'0')
        ra_col = treecorr.config.get_from_list(self.config,'ra_col',num,str,'0')
        dec_col = treecorr.config.get_from_list(self.config,'dec_col',num,str,'0')
        w_col = treecorr.config.get_from_list(self.config,'w_col',num,str,'0')
        flag_col = treecorr.config.get_from_list(self.config,'flag_col',num,str,'0')
        g1_col = treecorr.config.get_from_list(self.config,'g1_col',num,str,'0')
        g2_col = treecorr.config.get_from_list(self.config,'g2_col',num,str,'0')
        k_col = treecorr.config.get_from_list(self.config,'k_col',num,str,'0')

        hdu = treecorr.config.get_from_list(self.config,'hdu',num,int,1)

        # Start with everything set to None.  Overwrite as appropriate.
        self.x = None
        self.y = None
        self.ra = None
        self.dec = None
        self.w = None
        self.flag = None
        self.g1 = None
        self.g2 = None
        self.k = None

        # Read x,y or ra,dec
        if x_col != '0' or y_col != '0':
            if x_col == '0':
                raise AttributeError("x_col missing for file %s"%file_name)
            if y_col == '0':
                raise AttributeError("y_col missing for file %s"%file_name)
            if ra_col != '0':
                raise AttributeError("ra_col not allowed in conjunction with x/y cols")
            if dec_col != '0':
                raise AttributeError("dec_col not allowed in conjunction with x/y cols")
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
            if ra_col == '0':
                raise AttributeError("ra_col missing for file %s"%file_name)
            if dec_col == '0':
                raise AttributeError("dec_col missing for file %s"%file_name)
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
        else:
            raise AttributeError("No valid position columns specified for file %s"%file_name)

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
            self.flag = hdu_list[flag_hdu].data.field(flag_col).astype(float)
            self.logger.debug('read flag = %s',str(self.flag))

        # Read g1,g2
        if (g1_col != '0' or g2_col != '0') and isGColRequired(self.config,num):
            if g1_col == '0':
                raise AttributeError("g1_col is missing for file %s"%file_name)
            if g2_col == '0':
                raise AttributeError("g2_col is missing for file %s"%file_name)
            g1_hdu = treecorr.config.get_from_list(self.config,'g1_hdu',num,int,hdu)
            g2_hdu = treecorr.config.get_from_list(self.config,'g2_hdu',num,int,hdu)
            if g1_col not in hdu_list[g1_hdu].columns.names:
                raise AttributeError("g1_col is invalid for file %s"%file_name)
            if g2_col not in hdu_list[g2_hdu].columns.names:
                raise AttributeError("g2_col is invalid for file %s"%file_name)
            self.g1 = hdu_list[g1_hdu].data.field(g1_col).astype(float)
            self.logger.debug('read g1 = %s',str(self.g1))
            self.g2 = hdu_list[g2_hdu].data.field(g2_col).astype(float)
            self.logger.debug('read g2 = %s',str(self.g2))

        # Read k
        if k_col != '0' and isKColRequired(self.config,num):
            k_hdu = treecorr.config.get_from_list(self.config,'k_hdu',num,int,hdu)
            if k_col not in hdu_list[k_hdu].columns.names:
                raise AttributeError("k_col is invalid for file %s"%file_name)
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


def read_catalogs(config, key, list_key, num, logger, is_rand=False):
    """Read in a list of catalogs for the given key.

    num indicates which key to use if any of the fields like x_col, flip_g1, etc. are lists.
    """
    if key in config:
        if list_key in config:
            raise AttributeError("Cannot provide both %s and %s."%(key,list_key))
        file_names = config[key]
    elif list_key in config:
        list_file = config[list_key]
        with open(list_file,'r') as fin:
            file_names = [ f.strip() for f in fin ]
    else:
        # If this key was required (i.e. file_name) then let the caller check this.
        return []
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


