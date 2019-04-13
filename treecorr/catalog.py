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
.. module:: catalog
"""

import numpy as np
import coord
import weakref
import treecorr

class Catalog(object):
    """A set of input data (positions and other quantities) to be correlated.

    A Catalog object keeps track of the relevant information for a number of objects to
    be correlated.  The objects each have some kind of position (for instance (x,y), (ra,dec),
    (x,y,z), etc.), and possibly some extra information such as weights (w), shear values (g1,g2),
    or kappa values (k).

    The simplest way to build a Catalog is to simply pass in numpy arrays for each
    piece of information you want included.  For instance::

        >>> cat = treecorr.Catalog(x=x, y=y, k=k, w=w)

    Each of these input paramters should be a numpy array, where each corresponding element
    is the value for that object.  Of course, all the arrays should be the same size.

    In some cases, there are additional required parameters.  For instance, with RA and Dec
    positions, you need to declare what units the input values are given::

        >>> cat = treecorr.Catalog(ra=ra, dec=dec, g1=g1, g2=g2,
        ...                        ra_units='hour', dec_units='deg')

    For (ra,dec) positions, these units fields are required to specify the units of the angular
    values.  For (x,y) positions, the units are optional (and usually unnecessary).

    You can also initialize a Catalog by reading in columns from a file.  For instance::

        >>> cat = treecorr.Catalog('data.fits', ra_col='ALPHA2000', dec_col='DELTA2000',
        ...                        g1_col='E1', g2_col='E2', ra_units='deg', dec_units='deg')

    This reads the given columns from the input file.  The input file may be either
    a FITS catalog or an ASCII catalog.  Normally the file type is determined according to the
    file's extension (e.g. '.fits' here), but it can also be set explicitly with **file_type**.

    Finally, you may store all the various parameters in a configuration dict
    and just pass the dict as an argument after the file name::

        >>> config = { 'ra_col' : 'ALPHA2000',
        ...            'dec_col' : 'DELTA2000',
        ...            'g1_col' : 'E1',
        ...            'g2_col' : 'E2',
        ...            'ra_units' : 'deg',
        ...            'dec_units' : 'deg' }
        >>> cat = treecorr.Catalog(file_name, config)

    This can be useful for encapsulating all the TreeCorr options in a single place in your
    code, which might be used multiple times.  Notably, this syntax ignores any dict keys
    that are not relevant to the Catalog construction, so you can use the same config dict
    for the Catalog and your correlation objects, which can be convenient.

    See also `Configuration Parameters` for complete descriptions of all of the relevant
    configuration parameters, particularly the first section `Parameters about the input file(s)`.

    You may also override any configuration parameters or add additional parameters as kwargs
    after the config dict.  For instance, to flip the sign of the g1 values after reading
    from the input file, you could write::

        >>> cat1 = treecorr.Catalog(file_name, config, flip_g1=True)

    After construction, a Catalog object will have the following attributes:

    Attributes:

        x:      The x positions, if defined, as a numpy array (converted to radians if x_units
                was given). (None otherwise)
        y:      The y positions, if defined, as a numpy array (converted to radians if y_units
                was given). (None otherwise)
        z:      The z positions, if defined, as a numpy array. (None otherwise)
        ra:     The right ascension, if defined, as a numpy array (in radians). (None otherwise)
        dec:    The declination, if defined, as a numpy array (in radians). (None otherwise)
        r:      The distance, if defined, as a numpy array. (None otherwise)
        w:      The weights, as a numpy array. (All 1's if no weight column provided.)
        wpos:   The weights for position centroiding, as a numpy array.  (All 1's if neither
                weight column provided.)
        g1:     The g1 component of the shear, if defined, as a numpy array. (None otherwise)
        g2:     The g2 component of the shear, if defined, as a numpy array. (None otherwise)
        k:      The convergence, kappa, if defined, as a numpy array. (None otherwise)

        ntot:   The total number of objects (including those with zero weight)
        nobj:   The number of objects with non-zero weight
        sumw:   The sum of the weights
        varg:   The shear variance (aka shape noise) (0 if g1,g2 are not defined)
                Note: If there are weights, this is really :math:`\\sum(w^2 |g|^2)/\\sum(w)`,
                which is more like :math:`\\langle w \\rangle \\mathrm{Var}(g)`.  In finalize, we
                divide this by the weight in each bin, so this is the right quantity to use there.
        vark:   The kappa variance (0 if k is not defined)
                Note: If there are weights, this is really :math:`\\sum(w^2 \\kappa^2)/\\sum(w)`.

        name:   When constructed from a file, this will be the file_name.  It is only used as
                a reference name in logging output  after construction, so if you construct it
                from data vectors directly, it will be ''.  You may assign to it if you want to
                give this catalog a specific name.

        coords: What kind of coordinate system is defined for this catalog?
                The possibilities for this attribute are:

                    - 'flat' = 2-dimensional flat coordinates.  Set when x,y are given.
                    - 'spherical' = spherical coordinates.  Set when ra,dec are given.
                    - '3d' = 3-dimensional coordinates.  Set when x,y,z or ra,dec,r are given.

        field:  If any of the **get?Field** methods have been called to construct a field from
                this catalog (either explicitly or implicitly via a **corr.process()** command),
                then this attribute will hold the most recent field to have been constructed.
                Note: it holds this field as a weakref, so if caching is turned off with
                ``resize_cache(0)``, and the field has been garbage collected, then this attribute
                will be None.

    Parameters:
        file_name (str):    The name of the catalog file to be read in. (default: None, in which
                            case the columns need to be entered directly with **x**, **y**, etc.)

        config (dict):      A configuration dict which defines attributes about how to read the
                            file.  Any optional kwargs may be given here in the config dict if
                            desired.  Invalid keys in the config dict are ignored. (default: None)

        num (int):          Which number catalog are we reading.  e.g. for NG correlations the
                            catalog for the N has num=0, the one for G has num=1.  This is only
                            necessary if you are using a config dict where things like **x_col**
                            have multiple values.  (default: 0)
        logger:             If desired, a Logger object for logging. (default: None, in which case
                            one will be built according to the config dict's verbose level.)
        is_rand (bool):     If this is a random file, then setting is_rand to True will let them
                            skip k_col, g1_col, and g2_col if they were set for the main catalog.
                            (default: False)

        x (array):          The x values. (default: None; When providing values directly, either
                            x,y are required or ra,dec are required.)
        y (array):          The y values. (default: None; When providing values directly, either
                            x,y are required or ra,dec are required.)
        z (array):          The z values, if doing 3d positions. (default: None)
        ra (array):         The RA values. (default: None; When providing values directly, either
                            x,y are required or ra,dec are required.)
        dec (array):        The Dec values. (default: None; When providing values directly, either
                            x,y are required or ra,dec are required.)
        r (array):          The r values (the distances of each source from Earth). Note that r is
                            invalid in conjunction with x,y. (default: None)
        w (array):          The weights to apply when computing the correlations. (default: None)
        wpos (array):       The weights to use for position centroiding. (default: None, which
                            means to use the value weights, w, to weight the positions as well)
        flag (array):       An optional array of flags, indicating objects to skip.  Rows with
                            flag != 0 (or technically flag & ~ok_flag != 0) will be given a weight
                            of 0.  (default: None)
        g1 (array):         The g1 values to use for shear correlations. (g1,g2 may represent any
                            spinor field.) (default: None)
        g2 (array):         The g2 values to use for shear correlations. (g1,g2 may represent any
                            spinor field.) (default: None)
        k (array):          The kappa values to use for scalar correlations. (This may represent
                            any scalar field.) (default: None)

    Keyword Arguments:

        file_type (str):    What kind of file is the input file. Valid options are 'ASCII' or
                            'FITS' (default: if the file_name extension starts with .fit, then use
                            'FITS', else 'ASCII')
        delimiter (str):    For ASCII files, what delimiter to use between values. (default: None,
                            which means any whitespace)
        comment_marker (str): For ASCII files, what token indicates a comment line. (default: '#')
        first_row (int):    Which row to take as the first row to be used. (default: 1)
        last_row (int):     Which row to take as the last row to be used. (default: -1, which means
                            the last row in the file)

        x_col (str or int): The column to use for the x values. This should be an integer for ASCII
                            files or a string for FITS files. (default: 0 or '0', which means not
                            to read in this column. When reading from a file, either x_col and
                            y_col are required or ra_col and dec_col are required.)
        y_col (str or int): The column to use for the y values. This should be an integer for ASCII
                            files or a string for FITS files. (default: 0 or '0', which means not
                            to read in this column. When reading from a file, either x_col and
                            y_col are required or ra_col and dec_col are required.)
        z_col (str or int): The column to use for the z values. This should be an integer for ASCII
                            files or a string for FITS files. (default: 0 or '0', which means not
                            to read in this column.)
        ra_col (str or int): The column to use for the ra values. This should be an integer for
                            ASCII files or a string for FITS files. (default: 0 or '0', which
                            means not to read in this column. When reading from a file, either
                            x_col and y_col are required or ra_col and dec_col are required.)
        dec_col (str or int): The column to use for the dec values. This should be an integer for
                            ASCII files or a string for FITS files. (default: 0 or '0', which
                            means not to read in this column. When reading from a file, either
                            x_col and y_col are required or ra_col and dec_col are required.)
        r_col (str or int): The column to use for the r values. This should be an integer for ASCII
                            files or a string for FITS files.  Note that r_col is invalid in
                            conjunction with x_col/y_col. (default: 0 or '0', which means not to
                            read in this column.)

        x_units (str):      The units to use for the x values, given as a string.  Valid options are
                            arcsec, arcmin, degrees, hours, radians.  (default: radians, although
                            with (x,y) positions, you can often just ignore the units, and the
                            output separations will be in whatever units x and y are in.)
        y_units (str):      The units to use for the y values, given as a string.  Valid options are
                            arcsec, arcmin, degrees, hours, radians.  (default: radians, although
                            with (x,y) positions, you can often just ignore the units, and the
                            output separations will be in whatever units x and y are in.)
        ra_units (str):     The units to use for the ra values, given as a string.  Valid options
                            are arcsec, arcmin, degrees, hours, radians. (required when using
                            ra_col or providing ra directly)
        dec_units (str):    The units to use for the dec values, given as a string.  Valid options
                            are arcsec, arcmin, degrees, hours, radians. (required when using
                            dec_col or providing dec directly)

        g1_col (str or int): The column to use for the g1 values. This should be an integer for
                            ASCII files or a string for FITS files. (default: 0 or '0', which means
                            not to read in this column.)
        g2_col (str or int): The column to use for the g2 values. This should be an integer for
                            ASCII files or a string for FITS files. (default: 0 or '0', which means
                            not to read in this column.)
        k_col (str or int): The column to use for the kappa values. This should be an integer for
                            ASCII files or a string for FITS files. (default: 0 or '0', which means
                            not to read in this column.)
        w_col (str or int): The column to use for the weight values. This should be an integer for
                            ASCII files or a string for FITS files. (default: 0 or '0', which means
                            not to read in this column.)
        wpos_col (str or int): The column to use for the position weight values. This should be an
                            integer for ASCII files or a string for FITS files. (default: 0 or '0',
                            which means not to read in this column.)
        flag_col (str or int): The column to use for the flag values. This should be an integer for
                            ASCII files or a string for FITS files. Any row with flag != 0 (or
                            technically flag & ~ok_flag != 0) will be given a weight of 0.
                            (default: 0 or '0', which means not to read in this column.)
        ignore_flag (int):  Which flags should be ignored. (default: all non-zero flags are ignored.
                            Equivalent to ignore_flag = ~0.)
        ok_flag (int):      Which flags should be considered ok. (default: 0.  i.e. all non-zero
                            flags are ignored.)

        flip_g1 (bool):     Whtether to flip the sign of the input g1 values. (default: False)
        flip_g2 (bool):     Whtether to flip the sign of the input g2 values. (default: False)

        hdu (int):          For FITS files, which hdu to read. (default: 1)
        x_hdu (int):        Which hdu to use for the x values. (default: hdu)
        y_hdu (int):        Which hdu to use for the y values. (default: hdu)
        z_hdu (int):        Which hdu to use for the z values. (default: hdu)
        ra_hdu (int):       Which hdu to use for the ra values. (default: hdu)
        dec_hdu (int):      Which hdu to use for the dec values. (default: hdu)
        r_hdu (int):        Which hdu to use for the r values. (default: hdu)
        g1_hdu (int):       Which hdu to use for the g1 values. (default: hdu)
        g2_hdu (int):       Which hdu to use for the g2 values. (default: hdu)
        k_hdu (int):        Which hdu to use for the k values. (default: hdu)
        w_hdu (int):        Which hdu to use for the w values. (default: hdu)
        wpos_hdu (int):     Which hdu to use for the wpos values. (default: hdu)
        flag_hdu (int):     Which hdu to use for the flag values. (default: hdu)

        verbose (int):      If no logger is provided, this will optionally specify a logging level
                            to use.

                                - 0 means no logging output
                                - 1 means to output warnings only (default)
                                - 2 means to output various progress information
                                - 3 means to output extensive debugging information

        log_file (str):     If no logger is provided, this will specify a file to write the logging
                            output.  (default: None; i.e. output to standard output)

        split_method (str): How to split the cells in the tree when building the tree structure.
                            Options are:

                                - mean: Use the arithmetic mean of the coordinate being split.
                                  (default)
                                - median: Use the median of the coordinate being split.
                                - middle: Use the middle of the range; i.e. the average of the
                                  minimum and maximum value.
                                - random: Use a random point somewhere in the middle two quartiles
                                  of the range.

        cat_precision (int): The precision to use when writing a Catalog to an ASCII file. This
                            should be an integer, which specifies how many digits to write.
                            (default: 16)
    """
    # Dict describing the valid kwarg parameters, what types they are, and a description:
    # Each value is a tuple with the following elements:
    #    type
    #    may_be_list
    #    default value
    #    list of valid values
    #    description
    _valid_params = {
        'file_type' : (str, False, None, ['ASCII', 'FITS'],
                'The file type of the input files. The default is to use the file name extension.'),
        'delimiter' : (str, True, None, None,
                'The delimeter between values in an ASCII catalog. The default is any whitespace.'),
        'comment_marker' : (str, True, '#', None,
                'The first (non-whitespace) character of comment lines in an input ASCII catalog.'),
        'first_row' : (int, True, 1, None,
                'The first row to use from the input catalog'),
        'last_row' : (int, True, -1, None,
                'The last row to use from the input catalog.  The default is to use all of them.'),
        'x_col' : (str, True, '0', None,
                'Which column to use for x. Should be an integer for ASCII catalogs.'),
        'y_col' : (str, True, '0', None,
                'Which column to use for y. Should be an integer for ASCII catalogs.'),
        'z_col' : (str, True, '0', None,
                'Which column to use for z. Should be an integer for ASCII catalogs.'),
        'ra_col' : (str, True, '0', None,
                'Which column to use for ra. Should be an integer for ASCII catalogs.'),
        'dec_col' : (str, True, '0', None,
                'Which column to use for dec. Should be an integer for ASCII catalogs.'),
        'r_col' : (str, True, '0', None,
                'Which column to use for r.  Only valid with ra,dec. ',
                'Should be an integer for ASCII catalogs.'),
        'x_units' : (str, True, None, coord.AngleUnit.valid_names,
                'The units of x values.'),
        'y_units' : (str, True, None, coord.AngleUnit.valid_names,
                'The units of y values.'),
        'ra_units' : (str, True, None, coord.AngleUnit.valid_names,
                'The units of ra values. Required when using ra_col.'),
        'dec_units' : (str, True, None, coord.AngleUnit.valid_names,
                'The units of dec values. Required when using dec_col.'),
        'g1_col' : (str, True, '0', None,
                'Which column to use for g1. Should be an integer for ASCII catalogs.'),
        'g2_col' : (str, True, '0', None,
                'Which column to use for g2. Should be an integer for ASCII catalogs.'),
        'k_col' : (str, True, '0', None,
                'Which column to use for kappa. Should be an integer for ASCII catalogs. '),
        'w_col' : (str, True, '0', None,
                'Which column to use for weight. Should be an integer for ASCII catalogs.'),
        'wpos_col' : (str, True, '0', None,
                'Which column to use for position weight. Should be an integer for ASCII catalogs.'),
        'flag_col' : (str, True, '0', None,
                'Which column to use for flag. Should be an integer for ASCII catalogs.'),
        'ignore_flag': (int, True, None, None,
                'Ignore objects with flag & ignore_flag != 0 (bitwise &)'),
        'ok_flag': (int, True, 0, None,
                'Ignore objects with flag & ~ok_flag != 0 (bitwise &, ~)'),
        'hdu': (int, True, 1, None,
                'Which HDU in a fits file to use rather than hdu=1'),
        'x_hdu': (int, True, None, None,
                'Which HDU to use for the x_col. default is the global hdu value.'),
        'y_hdu': (int, True, None, None,
                'Which HDU to use for the y_col. default is the global hdu value.'),
        'z_hdu': (int, True, None, None,
                'Which HDU to use for the z_col. default is the global hdu value.'),
        'ra_hdu': (int, True, None, None,
                'Which HDU to use for the ra_col. default is the global hdu value.'),
        'dec_hdu': (int, True, None, None,
                'Which HDU to use for the dec_col. default is the global hdu value.'),
        'r_hdu': (int, True, None, None,
                'Which HDU to use for the r_col. default is the global hdu value.'),
        'g1_hdu': (int, True, None, None,
                'Which HDU to use for the g1_col. default is the global hdu value.'),
        'g2_hdu': (int, True, None, None,
                'Which HDU to use for the g2_col. default is the global hdu value.'),
        'k_hdu': (int, True, None, None,
                'Which HDU to use for the k_col. default is the global hdu value.'),
        'w_hdu': (int, True, None, None,
                'Which HDU to use for the w_col. default is the global hdu value.'),
        'wpos_hdu': (int, True, None, None,
                'Which HDU to use for the wpos_col. default is the global hdu value.'),
        'flag_hdu': (int, True, None, None,
                'Which HDU to use for the flag_col. default is the global hdu value.'),
        'flip_g1' : (bool, True, False, None,
                'Whether to flip the sign of g1'),
        'flip_g2' : (bool, True, False, None,
                'Whether to flip the sign of g2'),
        'verbose' : (int, False, 1, [0, 1, 2, 3],
                'How verbose the code should be during processing. ',
                '0 = Errors Only, 1 = Warnings, 2 = Progress, 3 = Debugging'),
        'log_file' : (str, False, None, None,
                'If desired, an output file for the logging output.',
                'The default is to write the output to stdout.'),
        'split_method' : (str, False, 'mean', ['mean', 'median', 'middle', 'random'],
                'Which method to use for splitting cells.'),
        'cat_precision' : (int, False, 16, None,
                'The number of digits after the decimal in the output.'),
    }
    def __init__(self, file_name=None, config=None, num=0, logger=None, is_rand=False,
                 x=None, y=None, z=None, ra=None, dec=None, r=None, w=None, wpos=None, flag=None,
                 g1=None, g2=None, k=None, **kwargs):

        self.config = treecorr.config.merge_config(config,kwargs,Catalog._valid_params)
        self.orig_config = config.copy() if config is not None else {}
        if config and kwargs:
            self.orig_config.update(kwargs)

        if logger is not None:
            self.logger = logger
        else:
            self.logger = treecorr.config.setup_logger(
                    treecorr.config.get(self.config,'verbose',int,1),
                    self.config.get('log_file',None))

        # Start with everything set to None.  Overwrite as appropriate.
        self.x = None
        self.y = None
        self.z = None
        self.ra = None
        self.dec = None
        self.r = None
        self.w = None
        self.wpos = None
        self.flag = None
        self.g1 = None
        self.g2 = None
        self.k = None
        self._setup_fields()

        # First style -- read from a file
        if file_name is not None:
            if any([v is not None for v in [x,y,z,ra,dec,r,g1,g2,k,w,wpos,flag]]):
                raise TypeError("Vectors may not be provided when file_name is provided.")
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
            elif file_type == 'ASCII':
                self.read_ascii(file_name,num,is_rand)
            else:  # pragma: no cover (This is already ensured by the config processing)
                raise ValueError("Invalid file_type %s"%file_type)

        # Second style -- pass in the vectors directly
        else:
            if x is not None or y is not None:
                if x is None or y is None:
                    raise TypeError("x and y must both be provided")
                if ra is not None or dec is not None:
                    raise TypeError("ra and dec may not be provided with x,y")
                if r is not None:
                    raise TypeError("r may not be provided with x,y")
            if ra is not None or dec is not None:
                if ra is None or dec is None:
                    raise TypeError("ra and dec must both be provided")
            if g1 is not None or g2 is not None:
                if g1 is None or g2 is None:
                    raise TypeError("g1 and g2 must both be provided")
            self.name = ''
            self.x = self.makeArray(x,'x')
            self.y = self.makeArray(y,'y')
            self.z = self.makeArray(z,'z')
            self.ra = self.makeArray(ra,'ra')
            self.dec = self.makeArray(dec,'dec')
            self.r = self.makeArray(r,'r')
            self.w = self.makeArray(w,'w')
            self.wpos = self.makeArray(wpos,'wpos')
            self.flag = self.makeArray(flag,'flag',int)
            self.g1 = self.makeArray(g1,'g1')
            self.g2 = self.makeArray(g2,'g2')
            self.k = self.makeArray(k,'k')

        # Apply units to x,y,ra,dec
        if self.x is not None:
            if 'x_units' in self.config and not 'y_units' in self.config:
                raise TypeError("x_units specified without specifying y_units")
            if 'y_units' in self.config and not 'x_units' in self.config:
                raise TypeError("y_units specified without specifying x_units")
            if 'ra_units' in self.config:
                raise TypeError("ra_units is invalid without ra")
            if 'dec_units' in self.config:
                raise TypeError("dec_units is invalid without dec")
            self.x_units = treecorr.config.get_from_list(self.config,'x_units',num,str,'radians')
            self.y_units = treecorr.config.get_from_list(self.config,'y_units',num,str,'radians')
            self.x *= self.x_units
            self.y *= self.y_units
        else:
            if not self.config.get('ra_units',None):
                raise TypeError("ra_units is required when using ra, dec")
            if not self.config.get('dec_units',None):
                raise TypeError("dec_units is required when using ra, dec")
            if 'x_units' in self.config:
                raise TypeError("x_units is invalid without x")
            if 'y_units' in self.config:
                raise TypeError("y_units is invalid without y")
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
                self.w = np.ones_like(self.flag, dtype=float)
            self.w[(self.flag & ignore_flag)!=0] = 0
            self.logger.debug('Applied flag: w => %s',str(self.w))

        # Check that all columns have the same length:
        if self.x is not None:
            self.ntot = len(self.x)
            if len(self.y) != self.ntot:
                raise ValueError("x and y have different numbers of elements")
        else:
            self.ntot = len(self.ra)
            if len(self.dec) != self.ntot:
                raise ValueError("ra and dec have different numbers of elements")
        if self.ntot == 0:
            raise ValueError("Catalog has no objects!")
        if self.z is not None and len(self.z) != self.ntot:
            raise ValueError("z has the wrong numbers of elements")
        if self.r is not None and len(self.r) != self.ntot:
            raise ValueError("r has the wrong numbers of elements")
        if self.w is not None and len(self.w) != self.ntot:
            raise ValueError("w has the wrong numbers of elements")
        if self.wpos is not None and len(self.wpos) != self.ntot:
            raise ValueError("wpos has the wrong numbers of elements")
        if self.g1 is not None and len(self.g1) != self.ntot:
            raise ValueError("g1 has the wrong numbers of elements")
        if self.g2 is not None and len(self.g2) != self.ntot:
            raise ValueError("g1 has the wrong numbers of elements")
        if self.k is not None and len(self.k) != self.ntot:
            raise ValueError("k has the wrong numbers of elements")

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
            end = self.ntot
        if first_row > 1:
            start = first_row-1
        else:
            start = 0
        self.ntot = end-start
        self.logger.debug('start..end = %d..%d',start,end)
        if self.x is not None: self.x = self.x[start:end]
        if self.y is not None: self.y = self.y[start:end]
        if self.z is not None: self.z = self.z[start:end]
        if self.ra is not None: self.ra = self.ra[start:end]
        if self.dec is not None: self.dec = self.dec[start:end]
        if self.r is not None: self.r = self.r[start:end]
        if self.w is not None: self.w = self.w[start:end]
        if self.wpos is not None: self.wpos = self.wpos[start:end]
        if self.g1 is not None: self.g1 = self.g1[start:end]
        if self.g2 is not None: self.g2 = self.g2[start:end]
        if self.k is not None: self.k = self.k[start:end]

        # Check for NaN's:
        self.checkForNaN(self.x,'x')
        self.checkForNaN(self.y,'y')
        self.checkForNaN(self.z,'z')
        self.checkForNaN(self.ra,'ra')
        self.checkForNaN(self.dec,'dec')
        self.checkForNaN(self.r,'r')
        self.checkForNaN(self.g1,'g1')
        self.checkForNaN(self.g2,'g2')
        self.checkForNaN(self.k,'k')
        self.checkForNaN(self.w,'w')
        self.checkForNaN(self.wpos,'wpos')

        # Copy w to wpos if necessary (Do this after checkForNaN's, since this may set some
        # entries to have w=0.)
        if self.wpos is None:
            self.logger.debug('Using w for wpos')
        else:
            # Check that any wpos == 0 points also have w == 0
            if np.any(self.wpos == 0.):
                if self.w is None:
                    self.logger.warning('Some wpos values are zero, setting w=0 for these points.')
                    self.w = np.ones((self.ntot), dtype=float)
                else:
                    if np.any(self.w[self.wpos == 0.] != 0.):
                        self.logger.error('Some wpos values = 0 but have w!=0. This is invalid.\n'
                                          'Setting w=0 for these points.')
                self.w[self.wpos == 0.] = 0.

        # Calculate some summary parameters here that will typically be needed
        if self.w is not None:
            self.nontrivial_w = True
            self.nobj = np.sum(self.w != 0)
            self.sumw = np.sum(self.w)
            if self.sumw == 0.:
                raise ValueError("Catalog has invalid sumw == 0")
            if self.g1 is not None:
                self.varg = np.sum(self.w**2 * (self.g1**2 + self.g2**2))
                # The 2 is because we need the variance _per componenet_.
                self.varg /= 2.*self.sumw
            else:
                self.varg = 0.
            if self.k is not None:
                self.vark = np.sum(self.w**2 * self.k**2)
                self.vark /= self.sumw
            else:
                self.vark = 0.
        else:
            self.nontrivial_w = False
            self.nobj = self.ntot
            self.sumw = self.ntot
            if self.g1 is not None:
                self.varg = np.sum(self.g1**2 + self.g2**2) / (2.*self.nobj)
            else:
                self.varg = 0.
            if self.k is not None:
                self.vark = np.sum(self.k**2) / self.nobj
            else:
                self.vark = 0.
            self.w = np.ones((self.ntot), dtype=float)

        if self.ra is not None:
            # Should have already been checked above, so just use assert here.
            assert self.x is None
            assert self.y is None
            assert self.z is None
            self.x, self.y, self.z = coord.CelestialCoord.radec_to_xyz(self.ra, self.dec)
            if self.r is None:
                self.coords = 'spherical'
            else:
                self.x *= self.r
                self.y *= self.r
                self.z *= self.r
                self.coords = '3d'
            self.x_units = self.y_units = 1.
        else:
            if self.z is None:
                self.coords = 'flat'
            else:
                self.coords = '3d'

        self.logger.info("   nobj = %d",self.nobj)


    def makeArray(self, col, col_str, dtype=float):
        """Turn the input column into a numpy array if it wasn't already.
        Also make sure the input in 1-d.

        Parameters:
            col (array-like):   The input column to be converted into a numpy array.
            col_str (str):      The name of the column.  Used only as information in logging output.
            dtype (type):       The dtype for the returned array.  (default: float)

        Returns:
            The column converted to a 1-d numpy array.
        """
        if col is not None:
            col = np.array(col,dtype=dtype)
            if len(col.shape) != 1:
                s = col.shape
                col = col.reshape(-1)
                self.logger.warning("Warning: Input %s column was not 1-d.\n"%col_str +
                                    "         Reshaping from %s to %s"%(s,col.shape))
        return col


    def checkForNaN(self, col, col_str):
        """Check if the column has any NaNs.  If so, set those rows to have w[k]=0.

        Parameters:
            col (array):    The input column to check.
            col_str (str):  The name of the column.  Used only as information in logging output.
        """
        if col is not None and any(np.isnan(col)):
            index = np.where(np.isnan(col))[0]
            self.logger.warning("Warning: NaNs found in %s column.  Skipping rows %s."%(
                                col_str,str(index.tolist())))
            if self.w is None:
                self.w = np.ones_like(col, dtype=float)
            self.w[index] = 0

    def read_ascii(self, file_name, num=0, is_rand=False):
        """Read the catalog from an ASCII file

        Parameters:
            file_name (str):    The name of the file to read in.
            num (int):          Which number catalog are we reading. (default: 0)
            is_rand (bool):     Is this a random catalog? (default: False)
        """
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
                for line in fid:  # pragma: no branch
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
            self.logger.warning("Unable to import pandas..  Using np.genfromtxt instead.\n"+
                                "Installing pandas is recommended for increased speed when "+
                                "reading ASCII catalogs.")
            data = np.genfromtxt(file_name, comments=comment_marker, delimiter=delimiter)

        self.logger.debug('read data from %s, num=%d',file_name,num)
        self.logger.debug('data shape = %s',str(data.shape))

        # If only one row, and not using pands, then the shape comes in as one-d.  Reshape it:
        if len(data.shape) == 1:
            data = data.reshape(1,-1)
        if len(data.shape) != 2:  # pragma: no cover
            raise IOError('Unable to parse the input catalog as a 2-d array')
        ncols = data.shape[1]

        # Get the column numbers or names
        x_col = treecorr.config.get_from_list(self.config,'x_col',num,int,0)
        y_col = treecorr.config.get_from_list(self.config,'y_col',num,int,0)
        z_col = treecorr.config.get_from_list(self.config,'z_col',num,int,0)
        ra_col = treecorr.config.get_from_list(self.config,'ra_col',num,int,0)
        dec_col = treecorr.config.get_from_list(self.config,'dec_col',num,int,0)
        r_col = treecorr.config.get_from_list(self.config,'r_col',num,int,0)
        w_col = treecorr.config.get_from_list(self.config,'w_col',num,int,0)
        wpos_col = treecorr.config.get_from_list(self.config,'wpos_col',num,int,0)
        flag_col = treecorr.config.get_from_list(self.config,'flag_col',num,int,0)
        g1_col = treecorr.config.get_from_list(self.config,'g1_col',num,int,0)
        g2_col = treecorr.config.get_from_list(self.config,'g2_col',num,int,0)
        k_col = treecorr.config.get_from_list(self.config,'k_col',num,int,0)

        # Read x,y or ra,dec
        if x_col != 0 or y_col != 0:
            if x_col <= 0 or x_col > ncols:
                raise TypeError("x_col missing or invalid for file %s"%file_name)
            if y_col <= 0 or y_col > ncols:
                raise TypeError("y_col missing or invalid for file %s"%file_name)
            if z_col < 0 or z_col > ncols:
                raise TypeError("z_col is invalid for file %s"%file_name)
            if ra_col != 0:
                raise TypeError("ra_col not allowed in conjunction with x/y cols")
            if dec_col != 0:
                raise TypeError("dec_col not allowed in conjunction with x/y cols")
            if r_col != 0:
                raise TypeError("r_col not allowed in conjunction with x/y cols")
            # NB. astype always copies, even if the type is already correct.
            # We actually want this, since it makes the result contiguous in memory,
            # which we will need.
            self.x = data[:,x_col-1].astype(float)
            self.logger.debug('read x = %s',str(self.x))
            self.y = data[:,y_col-1].astype(float)
            self.logger.debug('read y = %s',str(self.y))
            if z_col != 0:
                self.z = data[:,z_col-1].astype(float)
                self.logger.debug('read r = %s',str(self.r))
        elif ra_col != 0 or dec_col != 0:
            if ra_col <= 0 or ra_col > ncols:
                raise TypeError("ra_col missing or invalid for file %s"%file_name)
            if dec_col <= 0 or dec_col > ncols:
                raise TypeError("dec_col missing or invalid for file %s"%file_name)
            if r_col < 0 or r_col > ncols:
                raise TypeError("r_col is invalid for file %s"%file_name)
            if z_col != 0:
                raise TypeError("z_col not allowed in conjunction with ra/dec cols")
            self.ra = data[:,ra_col-1].astype(float)
            self.logger.debug('read ra = %s',str(self.ra))
            self.dec = data[:,dec_col-1].astype(float)
            self.logger.debug('read dec = %s',str(self.dec))
            if r_col != 0:
                self.r = data[:,r_col-1].astype(float)
                self.logger.debug('read r = %s',str(self.r))
        else:
            raise TypeError("No valid position columns specified for file %s"%file_name)

        # Read w
        if w_col != 0:
            if w_col <= 0 or w_col > ncols:
                raise TypeError("w_col is invalid for file %s"%file_name)
            self.w = data[:,w_col-1].astype(float)
            self.logger.debug('read w = %s',str(self.w))

        # Read wpos
        if wpos_col != 0:
            if wpos_col <= 0 or wpos_col > ncols:
                raise TypeError("wpos_col is invalid for file %s"%file_name)
            self.wpos = data[:,wpos_col-1].astype(float)
            self.logger.debug('read wpos = %s',str(self.wpos))

        # Read flag
        if flag_col != 0:
            if flag_col <= 0 or flag_col > ncols:
                raise TypeError("flag_col is invalid for file %s"%file_name)
            self.flag = data[:,flag_col-1].astype(int)
            self.logger.debug('read flag = %s',str(self.flag))

        # Return here if this file is a random catalog
        if is_rand: return

        # Read g1,g2
        if (g1_col != 0 or g2_col != 0):
            if g1_col <= 0 or g1_col > ncols or g2_col <= 0 or g2_col > ncols:
                if isGColRequired(self.orig_config,num):
                    raise TypeError("g1_col, g2_col are invalid for file %s"%file_name)
                else:
                    self.logger.warning("Warning: skipping g1_col, g2_col for %s, num=%d "%(
                                        file_name,num) +
                                        "because they are invalid, but unneeded.")
            else:
                self.g1 = data[:,g1_col-1].astype(float)
                self.logger.debug('read g1 = %s',str(self.g1))
                self.g2 = data[:,g2_col-1].astype(float)
                self.logger.debug('read g2 = %s',str(self.g2))

        # Read k
        if k_col != 0:
            if k_col <= 0 or k_col > ncols:
                if isKColRequired(self.orig_config,num):
                    raise TypeError("k_col is invalid for file %s"%file_name)
                else:
                    self.logger.warning("Warning: skipping k_col for %s, num=%d "%(file_name,num)+
                                        "because it is invalid, but unneeded.")
            else:
                self.k = data[:,k_col-1].astype(float)
                self.logger.debug('read k = %s',str(self.k))


    def read_fits(self, file_name, num=0, is_rand=False):
        """Read the catalog from a FITS file

        Parameters:
            file_name (str):    The name of the file to read in.
            num (int):          Which number catalog are we reading. (default: 0)
            is_rand (bool):     Is this a random catalog? (default: False)
        """
        try:
            import fitsio
        except ImportError:
            self.logger.error("Unable to import fitsio.  Cannot read catalog %s"%file_name)
            raise

        # Get the column names
        x_col = treecorr.config.get_from_list(self.config,'x_col',num,str,'0')
        y_col = treecorr.config.get_from_list(self.config,'y_col',num,str,'0')
        z_col = treecorr.config.get_from_list(self.config,'z_col',num,str,'0')
        ra_col = treecorr.config.get_from_list(self.config,'ra_col',num,str,'0')
        dec_col = treecorr.config.get_from_list(self.config,'dec_col',num,str,'0')
        r_col = treecorr.config.get_from_list(self.config,'r_col',num,str,'0')
        w_col = treecorr.config.get_from_list(self.config,'w_col',num,str,'0')
        wpos_col = treecorr.config.get_from_list(self.config,'wpos_col',num,str,'0')
        flag_col = treecorr.config.get_from_list(self.config,'flag_col',num,str,'0')
        g1_col = treecorr.config.get_from_list(self.config,'g1_col',num,str,'0')
        g2_col = treecorr.config.get_from_list(self.config,'g2_col',num,str,'0')
        k_col = treecorr.config.get_from_list(self.config,'k_col',num,str,'0')

        # Check that position cols are valid:
        if x_col != '0' or y_col != '0':
            if x_col == '0':
                raise ValueError("x_col missing for file %s"%file_name)
            if y_col == '0':
                raise ValueError("y_col missing for file %s"%file_name)
            if ra_col != '0':
                raise ValueError("ra_col not allowed in conjunction with x/y cols")
            if dec_col != '0':
                raise ValueError("dec_col not allowed in conjunction with x/y cols")
            if r_col != '0':
                raise ValueError("r_col not allowed in conjunction with x/y cols")
        elif ra_col != '0' or dec_col != '0':
            if ra_col == '0':
                raise ValueError("ra_col missing for file %s"%file_name)
            if dec_col == '0':
                raise ValueError("dec_col missing for file %s"%file_name)
            if z_col != '0':
                raise ValueError("z_col not allowed in conjunction with ra/dec cols")
        else:
            raise ValueError("No valid position columns specified for file %s"%file_name)

        # Check that g1,g2,k cols are valid
        if g1_col == '0' and isGColRequired(self.orig_config,num):
            raise ValueError("g1_col is missing for file %s"%file_name)
        if g2_col == '0' and isGColRequired(self.orig_config,num):
            raise ValueError("g2_col is missing for file %s"%file_name)
        if k_col == '0' and isKColRequired(self.orig_config,num):
            raise ValueError("k_col is missing for file %s"%file_name)

        if (g1_col != '0' and g2_col == '0') or (g1_col == '0' and g2_col != '0'):
            raise ValueError("g1_col, g2_col are invalid for file %s"%file_name)

        # OK, now go ahead and read all the columns.
        hdu = treecorr.config.get_from_list(self.config,'hdu',num,int,1)

        with fitsio.FITS(file_name, 'r') as fits:

            # Read x,y or ra,dec,r
            if x_col != '0':
                x_hdu = treecorr.config.get_from_list(self.config,'x_hdu',num,int,hdu)
                y_hdu = treecorr.config.get_from_list(self.config,'y_hdu',num,int,hdu)
                if x_col not in fits[x_hdu].get_colnames():
                    raise ValueError("x_col is invalid for file %s"%file_name)
                if y_col not in fits[y_hdu].get_colnames():
                    raise ValueError("y_col is invalid for file %s"%file_name)
                self.x = fits[x_hdu].read_column(x_col).astype(float)
                self.logger.debug('read x = %s',str(self.x))
                self.y = fits[y_hdu].read_column(y_col).astype(float)
                self.logger.debug('read y = %s',str(self.y))
                if z_col != '0':
                    z_hdu = treecorr.config.get_from_list(self.config,'z_hdu',num,int,hdu)
                    if z_col not in fits[z_hdu].get_colnames():
                        raise ValueError("z_col is invalid for file %s"%file_name)
                    self.z = fits[z_hdu].read_column(z_col).astype(float)
                    self.logger.debug('read z = %s',str(self.z))
            else:
                ra_hdu = treecorr.config.get_from_list(self.config,'ra_hdu',num,int,hdu)
                dec_hdu = treecorr.config.get_from_list(self.config,'dec_hdu',num,int,hdu)
                if ra_col not in fits[ra_hdu].get_colnames():
                    raise ValueError("ra_col is invalid for file %s"%file_name)
                if dec_col not in fits[dec_hdu].get_colnames():
                    raise ValueError("dec_col is invalid for file %s"%file_name)
                self.ra = fits[ra_hdu].read_column(ra_col).astype(float)
                self.logger.debug('read ra = %s',str(self.ra))
                self.dec = fits[dec_hdu].read_column(dec_col).astype(float)
                self.logger.debug('read dec = %s',str(self.dec))
                if r_col != '0':
                    r_hdu = treecorr.config.get_from_list(self.config,'r_hdu',num,int,hdu)
                    if r_col not in fits[r_hdu].get_colnames():
                        raise ValueError("r_col is invalid for file %s"%file_name)
                    self.r = fits[r_hdu].read_column(r_col).astype(float)
                    self.logger.debug('read r = %s',str(self.r))

            # Read w
            if w_col != '0':
                w_hdu = treecorr.config.get_from_list(self.config,'w_hdu',num,int,hdu)
                if w_col not in fits[w_hdu].get_colnames():
                    raise ValueError("w_col is invalid for file %s"%file_name)
                self.w = fits[w_hdu].read_column(w_col).astype(float)
                self.logger.debug('read w = %s',str(self.w))

            # Read wpos
            if wpos_col != '0':
                wpos_hdu = treecorr.config.get_from_list(self.config,'wpos_hdu',num,int,hdu)
                if wpos_col not in fits[wpos_hdu].get_colnames():
                    raise ValueError("wpos_col is invalid for file %s"%file_name)
                self.wpos = fits[wpos_hdu].read_column(wpos_col).astype(float)
                self.logger.debug('read wpos = %s',str(self.wpos))

            # Read flag
            if flag_col != '0':
                flag_hdu = treecorr.config.get_from_list(self.config,'flag_hdu',num,int,hdu)
                if flag_col not in fits[flag_hdu].get_colnames():
                    raise ValueError("flag_col is invalid for file %s"%file_name)
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
                    if isGColRequired(self.orig_config,num):
                        raise ValueError("g1_col, g2_col are invalid for file %s"%file_name)
                    else:
                        self.logger.warning("Warning: skipping g1_col, g2_col for %s, num=%d "%(
                                            file_name,num) +
                                            "because they are invalid, but unneeded.")
                else:
                    self.g1 = fits[g1_hdu].read_column(g1_col).astype(float)
                    self.logger.debug('read g1 = %s',str(self.g1))
                    self.g2 = fits[g2_hdu].read_column(g2_col).astype(float)
                    self.logger.debug('read g2 = %s',str(self.g2))

            # Read k
            if k_col != '0':
                k_hdu = treecorr.config.get_from_list(self.config,'k_hdu',num,int,hdu)
                if k_col not in fits[k_hdu].get_colnames():
                    if isKColRequired(self.orig_config,num):
                        raise ValueError("k_col is invalid for file %s"%file_name)
                    else:
                        self.logger.warning("Warning: skipping k_col for %s, num=%d "%(
                                            file_name,num)+
                                            "because it is invalid, but unneeded.")
                else:
                    self.k = fits[k_hdu].read_column(k_col).astype(float)
                    self.logger.debug('read k = %s',str(self.k))

    def _setup_fields(self):
        self._field = lambda : None  # Acts like a dead weakref

        # Make simple functions that call NField, etc. with self as the first argument.

        def get_nfield(*args, **kwargs): return treecorr.NField(self, *args, **kwargs)
        def get_kfield(*args, **kwargs): return treecorr.KField(self, *args, **kwargs)
        def get_gfield(*args, **kwargs): return treecorr.GField(self, *args, **kwargs)
        def get_nsimplefield(*args, **kwargs): return treecorr.NSimpleField(self, *args, **kwargs)
        def get_ksimplefield(*args, **kwargs): return treecorr.KSimpleField(self, *args, **kwargs)
        def get_gsimplefield(*args, **kwargs): return treecorr.GSimpleField(self, *args, **kwargs)

        # Now wrap these in LRU_Caches with (initially) just 1 element being cached.
        self.nfields = treecorr.util.LRU_Cache(get_nfield, 1)
        self.kfields = treecorr.util.LRU_Cache(get_kfield, 1)
        self.gfields = treecorr.util.LRU_Cache(get_gfield, 1)
        self.nsimplefields = treecorr.util.LRU_Cache(get_nsimplefield, 1)
        self.ksimplefields = treecorr.util.LRU_Cache(get_ksimplefield, 1)
        self.gsimplefields = treecorr.util.LRU_Cache(get_gsimplefield, 1)

    def resize_cache(self, maxsize):
        """Resize all field caches.

        The various kinds of fields built from this catalog are cached.  This may or may not
        be an optimization for your use case.  Normally only a single field is built for a
        given catalog, and it is usually efficient to cache it, so it can be reused multiple
        times.  E.g. for the usual Landy-Szalay NN calculation:

            >>> dd.process(data_cat)
            >>> rr.process(rand_cat)
            >>> dr.process(data_cat, rand_cat)

        the third line will be able to reuse the same fields built for the data and randoms
        in the first two lines.

        However, if you are making many different fields from the same catalog -- for instance
        because you keep changing the min_sep and max_sep for different calls -- then saving
        them all will tend to blow up the memory.

        Therefore, the default number of fields (of each type) to cache is 1.  This lets the
        first use case be efficient, but not use too much memory for the latter case.

        If you prefer a different behavior, this method lets you change the number of fields to
        cache.  The cache is an LRU (Least Recently Used) cache, which means only the n most
        recently used fields are saved.  I.e. when it is full, the least recently used field
        is removed from the cache.

        If you call this with maxsize=0, then caching will be turned off.  A new field will be
        built each time you call a process function with this catalog.

        If you call this with maxsize>1, then mutiple fields will be saved according to whatever
        number you set.  This will use more memory, but may be an optimization for you depending
        on what your are doing.

        Finally, if you want to set different sizes for the different kinds of fields, then
        you can call resize separately for the different caches:

            >>> cat.nfields.resize(maxsize)
            >>> cat.kfields.resize(maxsize)
            >>> cat.gfields.resize(maxsize)
            >>> cat.nsimplefields.resize(maxsize)
            >>> cat.ksimplefields.resize(maxsize)
            >>> cat.gsimplefields.resize(maxsize)

        Parameters:
            maxsize (float):    The new maximum number of fields of each type to cache.
        """
        self.nfields.resize(maxsize)
        self.kfields.resize(maxsize)
        self.gfields.resize(maxsize)
        self.nsimplefields.resize(maxsize)
        self.ksimplefields.resize(maxsize)
        self.gsimplefields.resize(maxsize)


    def clear_cache(self):
        """Clear all field caches.

        The various kinds of fields built from this catalog are cached.  This may or may not
        be an optimization for your use case.  Normally only a single field is built for a
        given catalog, and it is usually efficient to cache it, so it can be reused multiple
        times.  E.g. for the usual Landy-Szalay NN calculation:

            >>> dd.process(data_cat)
            >>> rr.process(rand_cat)
            >>> dr.process(data_cat, rand_cat)

        the third line will be able to reuse the same fields built for the data and randoms
        in the first two lines.

        However, this also means that the memory used for the field will persist as long as
        the catalog object does.  If you need to recover this memory and don't want to delete
        the catalog yet, this method lets you clear the cache.

        There are separate caches for each kind of field.  If you want to clear just one or
        some of them, you can call clear separately for the different caches:

            >>> cat.nfields.clear()
            >>> cat.kfields.clear()
            >>> cat.gfields.clear()
            >>> cat.nsimplefields.clear()
            >>> cat.ksimplefields.clear()
            >>> cat.gsimplefields.clear()
        """
        self.nfields.clear()
        self.kfields.clear()
        self.gfields.clear()
        self.nsimplefields.clear()
        self.ksimplefields.clear()
        self.gsimplefields.clear()
        self._field = lambda : None  # Acts like a dead weakref

    @property
    def field(self):
        # The default is to return None here.
        # This might also return None if weakref has expired.
        # But if the weakref is alive, this returns the field we want.
        return self._field()

    def getNField(self, min_size=0, max_size=None, split_method=None, brute=False,
                  min_top=3, max_top=10, coords=None, logger=None):
        """Return an `NField` based on the positions in this catalog.

        The `NField` object is cached, so this is efficient to call multiple times.
        cf. `resize_cache` and `clear_cache`

        Parameters:
            min_size (float):   The minimum radius cell required (usually min_sep). (default: 0)
            max_size (float):   The maximum radius cell required (usually max_sep). (default: None)
            split_method (str): Which split method to use ('mean', 'median', 'middle', or 'random')
                                (default: 'mean'; this value can also be given in the Catalog
                                constructor in the config dict.)
            brute (bool):       Whether to force traversal to the leaves. (default: False)
            min_top (int):      The minimum number of top layers to use when setting up the
                                field. (default: 3)
            max_top (int):      The maximum number of top layers to use when setting up the
                                field. (default: 10)
            coords (str):       The kind of coordinate system to use. (default: self.coords)
            logger:             A Logger object if desired (default: self.logger)

        Returns:
            An `NField` object
        """
        if split_method is None:
            split_method = treecorr.config.get(self.config,'split_method',str,'mean')
        if logger is None:
            logger = self.logger
        field = self.nfields(min_size, max_size, split_method, brute, min_top, max_top, coords,
                             logger=logger)
        self._field = weakref.ref(field)
        return field


    def getKField(self, min_size=0, max_size=None, split_method=None, brute=False,
                  min_top=3, max_top=10, coords=None, logger=None):
        """Return a `KField` based on the k values in this catalog.

        The `KField` object is cached, so this is efficient to call multiple times.
        cf. `resize_cache` and `clear_cache`

        Parameters:
            min_size (float):   The minimum radius cell required (usually min_sep). (default: 0)
            max_size (float):   The maximum radius cell required (usually max_sep). (default: None)
            split_method (str): Which split method to use ('mean', 'median', 'middle', or 'random')
                                (default: 'mean'; this value can also be given in the Catalog
                                constructor in the config dict.)
            brute (bool):       Whether to force traversal to the leaves. (default: False)
            min_top (int):      The minimum number of top layers to use when setting up the
                                field. (default: 3)
            max_top (int):      The maximum number of top layers to use when setting up the
                                field. (default: 10)
            coords (str):       The kind of coordinate system to use. (default self.coords)
            logger:             A Logger object if desired (default: self.logger)

        Returns:
            A `KField` object
        """
        if split_method is None:
            split_method = treecorr.config.get(self.config,'split_method',str,'mean')
        if self.k is None:
            raise TypeError("k is not defined.")
        if logger is None:
            logger = self.logger
        field = self.kfields(min_size, max_size, split_method, brute, min_top, max_top, coords,
                             logger=logger)
        self._field = weakref.ref(field)
        return field


    def getGField(self, min_size=0, max_size=None, split_method=None, brute=False,
                  min_top=3, max_top=10, coords=None, logger=None):
        """Return a `GField` based on the g1,g2 values in this catalog.

        The `GField` object is cached, so this is efficient to call multiple times.
        cf. `resize_cache` and `clear_cache`.

        Parameters:
            min_size (float):   The minimum radius cell required (usually min_sep). (default: 0)
            max_size (float):   The maximum radius cell required (usually max_sep). (default: None)
            split_method (str): Which split method to use ('mean', 'median', 'middle', or 'random')
                                (default: 'mean'; this value can also be given in the Catalog
                                constructor in the config dict.)
            brute (bool):       Whether to force traversal to the leaves. (default: False)
            min_top (int):      The minimum number of top layers to use when setting up the
                                field. (default: 3)
            max_top (int):      The maximum number of top layers to use when setting up the
                                field. (default: 10)
            coords (str):       The kind of coordinate system to use. (default self.coords)
            logger:             A Logger object if desired (default: self.logger)

        Returns:
            A `GField` object
        """
        if split_method is None:
            split_method = treecorr.config.get(self.config,'split_method',str,'mean')
        if self.g1 is None or self.g2 is None:
            raise TypeError("g1,g2 are not defined.")
        if logger is None:
            logger = self.logger
        field = self.gfields(min_size, max_size, split_method, brute, min_top, max_top, coords,
                             logger=logger)
        self._field = weakref.ref(field)
        return field


    def getNSimpleField(self, logger=None):
        """Return an `NSimpleField` based on the positions in this catalog.

        The `NSimpleField` object is cached, so this is efficient to call multiple times.
        cf. `resize_cache` and `clear_cache`

        Parameters:
            logger:     A Logger object if desired (default: self.logger)

        Returns:
            An `NSimpleField` object
        """
        if logger is None:
            logger = self.logger
        return self.nsimplefields(logger=logger)


    def getKSimpleField(self, logger=None):
        """Return a `KSimpleField` based on the k values in this catalog.

        The `KSimpleField` object is cached, so this is efficient to call multiple times.
        cf. `resize_cache` and `clear_cache`

        Parameters:
            logger:     A Logger object if desired (default: self.logger)

        Returns:
            A `KSimpleField` object
        """
        if self.k is None:
            raise TypeError("k is not defined.")
        if logger is None:
            logger = self.logger
        return self.ksimplefields(logger=logger)


    def getGSimpleField(self, logger=None):
        """Return a `GSimpleField` based on the g1,g2 values in this catalog.

        The `GSimpleField` object is cached, so this is efficient to call multiple times.
        cf. `resize_cache` and `clear_cache`

        Parameters:
            logger:             A Logger object if desired (default: self.logger)

        Returns:
            A `GSimpleField` object
        """
        if self.g1 is None or self.g2 is None:
            raise TypeError("g1,g2 are not defined.")
        if logger is None:
            logger = self.logger
        return self.gsimplefields(logger=logger)

    def write(self, file_name, file_type=None, cat_precision=None):
        """Write the catalog to a file.

        Note that the position columns are output using the same units as were used when
        building the Catalog.  If you want to use a different unit, you can set the catalog's
        units directly before writing.  e.g.:

            >>> cat = treecorr.Catalog('cat.dat', ra=ra, dec=dec,
                                       ra_units='hours', dec_units='degrees')
            >>> cat.ra_units = coord.degrees
            >>> cat.write('new_cat.dat')

        The output file will include some of the following columns (those for which the
        corresponding attribute is not None):

        ========      =======================================================
        Column        Description
        ========      =======================================================
        ra            self.ra if not None
        dec           self.dec if not None
        r             self.r if not None
        x             self.x if not None
        y             self.y if not None
        z             self.z if not None
        w             self.w if not None and self.nontrivial_w
        wpos          self.wpos if not None
        g1            self.g1 if not None
        g2            self.g2 if not None
        k             self.k if not None
        meanR         The mean value <R> of pairs that fell into each bin.
        meanlogR      The mean value <logR> of pairs that fell into each bin.
        ========      =======================================================

        Parameters:
            file_name (str):    The name of the file to write to.
            file_type (str):    The type of file to write ('ASCII' or 'FITS').  (default:
                                determine the type automatically from the extension of file_name.)
            cat_precision (int): For ASCII output catalogs, the desired precision. (default: 16;
                                this value can also be given in the Catalog constructor in the
                                config dict.)
        """
        self.logger.info('Writing catalog to %s',file_name)

        col_names = []
        columns = []
        if self.ra is not None:
            col_names.append('ra')
            columns.append(self.ra / self.ra_units)
            col_names.append('dec')
            columns.append(self.dec / self.dec_units)
            if self.r is not None:
                col_names.append('r')
                columns.append(self.r)
        else:
            col_names.append('x')
            columns.append(self.x / self.x_units)
            col_names.append('y')
            columns.append(self.y / self.y_units)
            if self.z is not None:
                col_names.append('z')
                columns.append(self.z)
        if self.nontrivial_w:
            col_names.append('w')
            columns.append(self.w)
        if self.wpos is not None:
            col_names.append('wpos')
            columns.append(self.wpos)
        if self.g1 is not None:
            col_names.append('g1')
            columns.append(self.g1)
        if self.g2 is not None:
            col_names.append('g2')
            columns.append(self.g2)
        if self.k is not None:
            col_names.append('k')
            columns.append(self.k)

        if cat_precision is None:
            cat_precision = treecorr.config.get(self.config,'cat_precision',int,16)

        treecorr.util.gen_write(file_name, col_names, columns, precision=cat_precision,
                                file_type=file_type, logger=self.logger)

    def copy(self):
        """Make a copy"""
        import copy
        return copy.deepcopy(self)

    def __getstate__(self):
        d = self.__dict__.copy()
        print('d = ',d)
        del d['logger']  # Oh well.  This is just lost in the copy.  Can't be pickled.
        del d['_field']
        del d['nfields']
        del d['kfields']
        del d['gfields']
        del d['nsimplefields']
        del d['ksimplefields']
        del d['gsimplefields']
        return d

    def __setstate__(self, d):
        self.__dict__ = d
        self.logger = treecorr.config.setup_logger(
                treecorr.config.get(self.config,'verbose',int,1),
                self.config.get('log_file',None))
        self._setup_fields()

    def __repr__(self):
        s = 'Catalog('
        if self.x is not None: s += 'x='+repr(self.x)+','
        if self.y is not None: s += 'y='+repr(self.y)+','
        if self.z is not None: s += 'z='+repr(self.z)+','
        if self.ra is not None: s += 'ra='+repr(self.ra)+','
        if self.dec is not None: s += 'dec='+repr(self.dec)+','
        if self.r is not None: s += 'r='+repr(self.r)+','
        if self.w is not None: s += 'w='+repr(self.w)+','
        if self.wpos is not None: s += 'wpos='+repr(self.wpos)+','
        if self.g1 is not None: s += 'g1='+repr(self.g1)+','
        if self.g2 is not None: s += 'g2='+repr(self.g2)+','
        if self.k is not None: s += 'k='+repr(self.k)+','
        # remove the last ','
        s = s[:-1] + ')'
        return s

    def __eq__(self, other):
        return (isinstance(other, Catalog) and
                self.nobj == other.nobj and
                self.ntot == other.ntot and
                self.sumw == other.sumw and
                self.varg == other.varg and
                self.vark == other.vark and
                self.coords == other.coords and
                np.array_equal(self.x, other.x) and
                np.array_equal(self.y, other.y) and
                np.array_equal(self.z, other.z) and
                np.array_equal(self.ra, other.ra) and
                np.array_equal(self.dec, other.dec) and
                np.array_equal(self.r, other.r) and
                np.array_equal(self.w, other.w) and
                np.array_equal(self.wpos, other.wpos) and
                np.array_equal(self.g1, other.g1) and
                np.array_equal(self.g2, other.g2) and
                np.array_equal(self.k, other.k))


def read_catalogs(config, key=None, list_key=None, num=0, logger=None, is_rand=None):
    """Read in a list of catalogs for the given key.

    key should be the file_name parameter or similar key word.
    list_key should be be corresponging file_list parameter, if appropriate.
    At least one of key or list_key must be provided.  If both are provided, then only
    one of these should be in the config dict.

    num indicates which key to use if any of the fields like x_col, flip_g1, etc. are lists.
    The default is 0, which means to use the first item in the list if they are lists.

    Parameters:
        config (dict):  The configuration dict to use for the appropriate parameters
        key (str):      Which key name to use for the file names. e.g. 'file_name' (default: None)
        list_key (str): Which key name to use for the name of a list file. e.g. 'file_list'.
                        Either key or list_key is required.  (default: None)
        num (int):      Which number catalog does this correspond to. e.g. file_name should use
                        num=0, file_name2 should use num=1.  (default: 0)
        logger:         If desired, a Logger object for logging. (default: None, in which case
                        one will be built according to the config dict's verbose level.)
        is_rand (bool): If this is a random file, then setting is_rand to True will let them
                        skip k_col, g1_col, and g2_col if they were set for the main catalog.
                        (default: False)

    Returns:
        A list of Catalogs or None if no catalogs are specified.
    """
    if logger is None:
        logger = treecorr.config.setup_logger(
                treecorr.config.get(config,'verbose',int,1), config.get('log_file',None))

    if key is None and list_key is None:
        raise TypeError("Must provide either key or list_key")
    if key is not None and key in config:
        if list_key is not None and list_key in config:
            raise TypeError("Cannot provide both key and list_key")
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
    if not isinstance(file_names,list):
        file_names = file_names.split()
    return [ Catalog(file_name, config, num, logger, is_rand) for file_name in file_names ]


def calculateVarG(cat_list):
    """Calculate the overall shear variance from a list of catalogs.

    The catalogs are assumed to be equivalent, so this is just the average shear
    variance (per component) weighted by the number of objects in each catalog.

    Parameters:
        cat_list:    A Catalog or a list of Catalogs for which to calculate the shear variance.

    Returns:
        The shear variance per component, aka shape noise.
    """
    if isinstance(cat_list,Catalog):
        return cat_list.varg
    elif len(cat_list) == 1:
        return cat_list[0].varg
    else:
        varg = 0
        sumw = 0
        for cat in cat_list:
            varg += cat.varg * cat.sumw
            sumw += cat.sumw
        return varg / sumw

def calculateVarK(cat_list):
    """Calculate the overall kappa variance from a list of catalogs.

    The catalogs are assumed to be equivalent, so this is just the average kappa
    variance weighted by the number of objects in each catalog.

    Parameters:
        cat_list:    A Catalog or a list of Catalogs for which to calculate the kappa variance.

    Returns:
        The kappa variance
    """
    if isinstance(cat_list,Catalog):
        return cat_list.vark
    elif len(cat_list) == 1:
        return cat_list[0].vark
    else:
        vark = 0
        sumw = 0
        for cat in cat_list:
            vark += cat.vark * cat.sumw
            sumw += cat.sumw
        return vark / sumw


def isGColRequired(config, num):
    """A quick helper function that checks whether we need to bother reading the g1,g2 columns.

    It checks the config dict for the output file names gg_file_name, ng_file_name (only if
    num == 1), etc.  If the output files indicate that we don't need the g1/g2 columns, then
    we don't need to raise an error if the g1_col or g2_col is invalid.

    This makes it easier to specify columns. e.g. for an NG correlation function, the
    first catalog does not need to have the g1,g2 columns, and typically wouldn't.  So
    if you specify g1_col=5, g2_col=6, say, and the first catalog does not have these columns,
    you would normally get an error.

    But instead, we check that the calculation is going to be NG from the presence of an
    ng_file_name parameter, and we let the would-be error pass.

    Parameters:
        config (dict):  The configuration file to check.
        num (int):      Which number catalog are we working on.

    Returns:
        True if some output file requires this catalog to have valid g1/g2 columns,
        False if not.

    """
    return config and ( 'gg_file_name' in config
                        or 'm2_file_name' in config
                        or 'norm_file_name' in config
                        or (num==1 and 'ng_file_name' in config)
                        or (num==1 and 'nm_file_name' in config)
                        or (num==1 and 'kg_file_name' in config) )



def isKColRequired(config, num):
    """A quick helper function that checks whether we need to bother reading the k column.

    The logic here is the same as for `isGColRequired`, but we check for output files that require
    the k column rather than g1,g2.

    Parameters:
        config (dict):  The configuration file to check.
        num (int):      Which number catalog are we working on.

    Returns:
        True if some output file requires this catalog to have valid g1/g2 columns,
        False if not.

    """
    return config and ( 'kk_file_name' in config
                        or (num==0 and 'kg_file_name' in config)
                        or (num==1 and 'nk_file_name' in config) )
