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
import copy
import os
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
    file's extension (e.g. '.fits' here), but it can also be set explicitly with ``file_type``.

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
        patch:  The patch number of each object, if patches are being used. (None otherwise)
                If the entire catalog is a single patch, then ``patch`` may be an int.
        ntot:   The total number of objects (including those with zero weight if keep_zero_weight
                is set to True)
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

        field:  If any of the ``get?Field`` methods have been called to construct a field from
                this catalog (either explicitly or implicitly via a `corr.process()
                <NNCorrelation.process>` command, then this attribute will hold the most recent
                field to have been constructed.
                Note: it holds this field as a weakref, so if caching is turned off with
                ``resize_cache(0)``, and the field has been garbage collected, then this attribute
                will be None.

    Parameters:
        file_name (str):    The name of the catalog file to be read in. (default: None, in which
                            case the columns need to be entered directly with ``x``, ``y``, etc.)

        config (dict):      A configuration dict which defines attributes about how to read the
                            file.  Any optional kwargs may be given here in the config dict if
                            desired.  Invalid keys in the config dict are ignored. (default: None)

        num (int):          Which number catalog are we reading.  e.g. for NG correlations the
                            catalog for the N has num=0, the one for G has num=1.  This is only
                            necessary if you are using a config dict where things like ``x_col``
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
        patch (array or int): Optionally, patch numbers to use for each object. (default: None)
                            Note: This may also be an int if the entire catalog represents a
                            single patch.  If ``patch_centers`` is given this will select those
                            items from the full input that correspond to the given patch number.
        patch_centers (array or str): Alternative to setting patch by hand or using kmeans, you
                            may instead give patch_centers either as a file name or an array
                            from which the patches will be determined. (default: None)

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
        every_nth (int):    Only use every nth row of the input catalog. (default: 1)

        npatch (int):       How many patches to split the catalog into (using kmeans) for the
                            purpose of jackknife variance or other options that involve running via
                            patches. (default: 1)
        kmeans_init (str):  If using kmeans to make patches, which init method to use.
                            cf. `Field.run_kmeans` (default: 'tree')
        kmeans_alt (str):   If using kmeans to make patches, whether to use the alternate kmeans
                            algorithm. cf. `Field.run_kmeans` (default: False)

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
        patch_col (str or int): The column to use for the patch numbers. This should be an integer
                            for ASCII files or a string for FITS files. (default: 0 or '0', which
                            means not to read in this column.)
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
        allow_xyz (bool):   Whether to allow x,y,z values in conjunction with ra,dec.  Normally,
                            it is an error to have both kinds of positions, but if you know that
                            the x,y,z, values are consistent with the given ra,dec values, it
                            can save time to input them, rather than calculate them using trig
                            functions. (default: False)

        flip_g1 (bool):     Whtether to flip the sign of the input g1 values. (default: False)
        flip_g2 (bool):     Whtether to flip the sign of the input g2 values. (default: False)
        keep_zero_weight (bool): Whether to keep objects with wpos=0 in the catalog (including
                            any objects that indirectly get wpos=0 due to NaN or flags), so they
                            would be included in ntot and also in npairs calculations that use
                            this Catalog, although of course not contribute to the accumulated
                            weight of pairs. (default: False)
        save_patch_dir (str): If desired, when building patches from this Catalog, save them
                            as FITS files in the given directory for more efficient loading when
                            doing cross-patch correlations with the ``low_mem`` option.

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
        patch_hdu (int):    Which hdu to use for the patch numbers. (default: hdu)
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

        num_threads (int):  How many OpenMP threads to use during the catalog load steps.
                            (default: use the number of cpu cores)
    """
    # Dict describing the valid kwarg parameters, what types they are, and a description:
    # Each value is a tuple with the following elements:
    #    type
    #    may_be_list
    #    default value
    #    list of valid values
    #    description
    _valid_params = {
        'file_type' : (str, True, None, ['ASCII', 'FITS'],
                'The file type of the input files. The default is to use the file name extension.'),
        'delimiter' : (str, True, None, None,
                'The delimeter between values in an ASCII catalog. The default is any whitespace.'),
        'comment_marker' : (str, True, '#', None,
                'The first (non-whitespace) character of comment lines in an input ASCII catalog.'),
        'first_row' : (int, True, 1, None,
                'The first row to use from the input catalog'),
        'last_row' : (int, True, -1, None,
                'The last row to use from the input catalog. The default is to use all of them.'),
        'every_nth' : (int, True, 1, None,
                'Only use every nth row of the input catalog. The default is to use all of them.'),
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
        'patch_col' : (str, True, '0', None,
                'Which column to use for patch numbers. Should be an integer for ASCII catalogs. '),
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
        'allow_xyz': (bool, True, False, None,
                'Whether to allow x,y,z inputs in conjunction with ra,dec'),
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
        'patch_hdu': (int, True, None, None,
                'Which HDU to use for the patch_col. default is the global hdu value.'),
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

        'keep_zero_weight' : (bool, False, False, None,
                'Whether to keep objects with zero weight in the catalog'),
        'npatch' : (int, False, 1, None,
                'Number of patches to split the catalog into'),
        'kmeans_init' : (str, False, 'tree', ['tree','random','kmeans++'],
                'Which initialization method to use for kmeans when making patches'),
        'kmeans_alt' : (bool, False, False, None,
                'Whether to use the alternate kmeans algorithm when making patches'),
        'patch_centers' : (str, False, None, None,
                'File with patch centers to use to determine patches'),
        'save_patch_dir' : (str, False, None, None,
                'If desired, save the patches as FITS files in this directory.'),
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
                 g1=None, g2=None, k=None, patch=None, patch_centers=None, **kwargs):

        self.config = treecorr.config.merge_config(config,kwargs,Catalog._valid_params)
        self.orig_config = config.copy() if config is not None else {}
        if config and kwargs:
            self.orig_config.update(kwargs)
        self._num = num
        self._is_rand = is_rand

        if logger is not None:
            self.logger = logger
        else:
            self.logger = treecorr.config.setup_logger(
                    treecorr.config.get(self.config,'verbose',int,1),
                    self.config.get('log_file',None))

        # Start with everything set to None.  Overwrite as appropriate.
        self._x = None
        self._y = None
        self._z = None
        self._ra = None
        self._dec = None
        self._r = None
        self._w = None
        self._wpos = None
        self._flag = None
        self._g1 = None
        self._g2 = None
        self._k = None
        self._patch = None
        self._field = lambda : None

        self._nontrivial_w = None
        self._single_patch = None
        self._nobj = None
        self._sumw = None
        self._varg = None
        self._vark = None
        self._npatch = 1
        self._patches = None
        self._centers = None

        first_row = treecorr.config.get_from_list(self.config,'first_row',num,int,1)
        if first_row < 1:
            raise ValueError("first_row should be >= 1")
        last_row = treecorr.config.get_from_list(self.config,'last_row',num,int,-1)
        if last_row > 0 and last_row < first_row:
            raise ValueError("last_row should be >= first_row")
        if last_row > 0:
            self.end = last_row
        else:
            self.end = None
        if first_row > 1:
            self.start = first_row-1
        else:
            self.start = 0
        self.every_nth = treecorr.config.get_from_list(self.config,'every_nth',num,int,1)
        if self.every_nth < 1:
            raise ValueError("every_nth should be >= 1")

        if 'npatch' in self.config and self.config['npatch'] != 1:
            self.npatch = treecorr.config.get(self.config,'npatch',int)
            if patch is not None or self.config.get('patch_col',0) not in (0,'0'):
                raise ValueError("Cannot provide both patch and npatch")
            if self.npatch < 1:
                raise ValueError("npatch must be >= 1")
        else:
            self.npatch = 1

        try:
            self._single_patch = int(patch)
        except TypeError:
            pass
        else:
            patch = None
            self._npatch = 1

        if patch_centers is None and 'patch_centers' in self.config:
            # file name version may be in a config dict, rather than kwarg.
            patch_centers = treecorr.config.get(self.config,'patch_centers',str)

        if patch_centers is not None:
            if patch is not None or self.config.get('patch_col',0) not in (0,'0'):
                raise ValueError("Cannot provide both patch and patch_centers")
            if 'npatch' in self.config and self.config['npatch'] != 1:
                raise ValueError("Cannot provide both npatch and patch_centers")
            if isinstance(patch_centers, np.ndarray):
                self._centers = patch_centers
            else:
                self._centers = self.read_patch_centers(patch_centers)

        self.save_patch_dir = self.config.get('save_patch_dir',None)
        allow_xyz = self.config.get('allow_xyz', False)

        # First style -- read from a file
        if file_name is not None:
            if any([v is not None for v in [x,y,z,ra,dec,r,g1,g2,k,patch,w,wpos,flag]]):
                raise TypeError("Vectors may not be provided when file_name is provided.")
            self.file_name = file_name
            self.name = file_name
            if self._single_patch is not None:
                self.name += " patch " + str(self._single_patch)

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
            if file_type == 'FITS':
                self._check_fits(file_name, num, is_rand)
            else:
                self._check_ascii(file_name, num, is_rand)

            self.file_type = file_type

        # Second style -- pass in the vectors directly
        else:
            self.file_type = None
            if x is not None or y is not None:
                if x is None or y is None:
                    raise TypeError("x and y must both be provided")
                if (ra is not None or dec is not None) and not allow_xyz:
                    raise TypeError("ra and dec may not be provided with x,y")
                if r is not None:
                    raise TypeError("r may not be provided with x,y")
            if ra is not None or dec is not None:
                if ra is None or dec is None:
                    raise TypeError("ra and dec must both be provided")
            if g1 is not None or g2 is not None:
                if g1 is None or g2 is None:
                    raise TypeError("g1 and g2 must both be provided")
            self.file_name = None
            self.name = ''
            if self._single_patch is not None:
                self.name = "patch " + str(self._single_patch)
            self._x = self.makeArray(x,'x')
            self._y = self.makeArray(y,'y')
            self._z = self.makeArray(z,'z')
            self._ra = self.makeArray(ra,'ra')
            self._dec = self.makeArray(dec,'dec')
            self._r = self.makeArray(r,'r')
            self._w = self.makeArray(w,'w')
            self._wpos = self.makeArray(wpos,'wpos')
            self._flag = self.makeArray(flag,'flag',int)
            self._g1 = self.makeArray(g1,'g1')
            self._g2 = self.makeArray(g2,'g2')
            self._k = self.makeArray(k,'k')
            self._patch = self.makeArray(patch,'patch',int)
            if self._patch is not None:
                self._set_npatch()

            # Apply units as appropriate
            self._apply_units()

            # Check that all columns have the same length.  (This is impossible in file input)
            if self._x is not None:
                ntot = len(self._x)
                if len(self._y) != ntot:
                    raise ValueError("x and y have different numbers of elements")
            else:
                ntot = len(self._ra)
                if len(self._dec) != ntot:
                    raise ValueError("ra and dec have different numbers of elements")
            if self._z is not None and len(self._z) != ntot:
                raise ValueError("z has the wrong numbers of elements")
            if self._r is not None and len(self._r) != ntot:
                raise ValueError("r has the wrong numbers of elements")
            if self._w is not None and len(self._w) != ntot:
                raise ValueError("w has the wrong numbers of elements")
            if self._wpos is not None and len(self._wpos) != ntot:
                raise ValueError("wpos has the wrong numbers of elements")
            if self._g1 is not None and len(self._g1) != ntot:
                raise ValueError("g1 has the wrong numbers of elements")
            if self._g2 is not None and len(self._g2) != ntot:
                raise ValueError("g2 has the wrong numbers of elements")
            if self._k is not None and len(self._k) != ntot:
                raise ValueError("k has the wrong numbers of elements")
            if self._patch is not None and len(self._patch) != ntot:
                raise ValueError("patch has the wrong numbers of elements")
            if ntot == 0:
                raise ValueError("Input arrays have zero length")

        if x is not None or self.config.get('x_col','0') not in [0,'0']:
            if 'x_units' in self.config and not 'y_units' in self.config:
                raise TypeError("x_units specified without specifying y_units")
            if 'y_units' in self.config and not 'x_units' in self.config:
                raise TypeError("y_units specified without specifying x_units")
        else:
            if 'x_units' in self.config:
                raise TypeError("x_units is invalid without x")
            if 'y_units' in self.config:
                raise TypeError("y_units is invalid without y")
        if ra is not None or self.config.get('ra_col','0') not in [0,'0']:
            if not self.config.get('ra_units',None):
                raise TypeError("ra_units is required when using ra, dec")
            if not self.config.get('dec_units',None):
                raise TypeError("dec_units is required when using ra, dec")
        else:
            if 'ra_units' in self.config:
                raise TypeError("ra_units is invalid without ra")
            if 'dec_units' in self.config:
                raise TypeError("dec_units is invalid without dec")

        if file_name is None:
            # For vector input option, can finish up now.
            if self._single_patch is not None:
                self._select_patch(self._single_patch)
            self._finish_input()

    @property
    def loaded(self):
        # _x gets set regardless of whether input used x,y or ra,dec, so the state of this
        # attribute is a good sentinal for whether the file has been loaded yet.
        return self._x is not None

    @property
    def x(self):
        self.load()
        return self._x

    @property
    def y(self):
        self.load()
        return self._y

    @property
    def z(self):
        self.load()
        return self._z

    @property
    def ra(self):
        self.load()
        return self._ra

    @property
    def dec(self):
        self.load()
        return self._dec

    @property
    def r(self):
        self.load()
        return self._r

    @property
    def w(self):
        self.load()
        return self._w

    @property
    def wpos(self):
        self.load()
        return self._wpos

    @property
    def g1(self):
        self.load()
        return self._g1

    @property
    def g2(self):
        self.load()
        return self._g2

    @property
    def k(self):
        self.load()
        return self._k

    @property
    def patch(self):
        if self._single_patch is not None:
            return self._single_patch
        else:
            self.load()
            return self._patch

    @property
    def patches(self):
        return self.get_patches()

    @property
    def patch_centers(self):
        return self.get_patch_centers()

    @property
    def varg(self):
        if self._varg is None:
            if self.nontrivial_w:
                if self.g1 is not None:
                    use = self.w != 0
                    self._varg = np.sum(self.w[use]**2 * (self.g1[use]**2 + self.g2[use]**2))
                    # The 2 is because we need the variance _per componenet_.
                    self._varg /= 2.*self.sumw
                else:
                    self._varg = 0.
            else:
                if self.g1 is not None:
                    self._varg = np.sum(self.g1**2 + self.g2**2) / (2.*self.nobj)
                else:
                    self._varg = 0.
        return self._varg

    @property
    def vark(self):
        if self._vark is None:
            if self.nontrivial_w:
                if self.k is not None:
                    use = self.w != 0
                    self._vark = np.sum(self.w[use]**2 * self.k[use]**2)
                    self._vark /= self.sumw
                else:
                    self._vark = 0.
            else:
                if self.k is not None:
                    self._vark = np.sum(self.k**2) / self.nobj
                else:
                    self._vark = 0.
        return self._vark

    @property
    def nontrivial_w(self):
        if self._nontrivial_w is None: self.load()
        return self._nontrivial_w

    @property
    def ntot(self):
        return len(self.x)

    @property
    def nobj(self):
        if self._nobj is None:
            if self.nontrivial_w:
                use = self._w != 0
                self._nobj = np.sum(use)
            else:
                self._nobj = self.ntot
        return self._nobj

    @property
    def sumw(self):
        if self._sumw is None: self.load()
        return self._sumw

    @property
    def coords(self):
        if self.ra is not None:
            if self.r is None:
                return 'spherical'
            else:
                return '3d'
        else:
            if self.z is None:
                return 'flat'
            else:
                return '3d'

    def _get_center_size(self):
        if not hasattr(self, '_cen_s'):
            mx = np.mean(self.x)
            my = np.mean(self.y)
            mz = 0
            dsq = (self.x - mx)**2 + (self.y - my)**2
            if self.z is not None:
                mz = np.mean(self.z)
                dsq += (self.z - mz)**2
            s = np.max(dsq)**0.5
            self._cen_s = (mx, my, mz, s)
        return self._cen_s

    def _finish_input(self):
        # Finish processing the data based on given inputs.

        # Apply flips if requested
        flip_g1 = treecorr.config.get_from_list(self.config,'flip_g1',self._num,bool,False)
        flip_g2 = treecorr.config.get_from_list(self.config,'flip_g2',self._num,bool,False)
        if flip_g1:
            self.logger.info("   Flipping sign of g1.")
            self._g1 = -self._g1
        if flip_g2:
            self.logger.info("   Flipping sign of g2.")
            self._g2 = -self._g2

        # Convert the flag to a weight
        if self._flag is not None:
            if 'ignore_flag' in self.config:
                ignore_flag = treecorr.config.get_from_list(self.config,'ignore_flag',self._num,int)
            else:
                ok_flag = treecorr.config.get_from_list(self.config,'ok_flag',self._num,int,0)
                ignore_flag = ~ok_flag
            # If we don't already have a weight column, make one with all values = 1.
            if self._w is None:
                self._w = np.ones_like(self._flag, dtype=float)
            self._w[(self._flag & ignore_flag)!=0] = 0
            if self._wpos is not None:
                self._wpos[(self._flag & ignore_flag)!=0] = 0
            self.logger.debug('Applied flag')

        # Check for NaN's:
        self.checkForNaN(self._x,'x')
        self.checkForNaN(self._y,'y')
        self.checkForNaN(self._z,'z')
        self.checkForNaN(self._ra,'ra')
        self.checkForNaN(self._dec,'dec')
        self.checkForNaN(self._r,'r')
        self.checkForNaN(self._g1,'g1')
        self.checkForNaN(self._g2,'g2')
        self.checkForNaN(self._k,'k')
        self.checkForNaN(self._w,'w')
        self.checkForNaN(self._wpos,'wpos')

        # If using ra/dec, generate x,y,z
        # Note: This also makes self.ntot work properly.
        self._generate_xyz()

        # Copy w to wpos if necessary (Do this after checkForNaN's, since this may set some
        # entries to have w=0.)
        if self._wpos is None:
            self.logger.debug('Using w for wpos')
        else:
            # Check that any wpos == 0 points also have w == 0
            if np.any(self._wpos == 0.):
                if self._w is None:
                    self.logger.warning('Some wpos values are zero, setting w=0 for these points.')
                    self._w = np.ones((self.ntot), dtype=float)
                else:
                    if np.any(self._w[self._wpos == 0.] != 0.):
                        self.logger.error('Some wpos values = 0 but have w!=0. This is invalid.\n'
                                          'Setting w=0 for these points.')
                self._w[self._wpos == 0.] = 0.

        if self._w is not None:
            self._nontrivial_w = True
            self._sumw = np.sum(self._w)
            if self._sumw == 0:
                raise ValueError("Catalog has invalid sumw == 0")
        else:
            self._nontrivial_w = False
            self._sumw = self.ntot
            # Make w all 1s to simplify the use of w later in code.
            self._w = np.ones((self.ntot), dtype=float)

        keep_zero_weight = treecorr.config.get(self.config,'keep_zero_weight',bool,False)
        if self._nontrivial_w and not keep_zero_weight:
            wpos = self._wpos if self._wpos is not None else self._w
            if np.any(wpos == 0):
                self.select(np.where(wpos != 0)[0])

        if self.npatch != 1:
            init = treecorr.config.get(self.config,'kmeans_init',str,'tree')
            alt = treecorr.config.get(self.config,'kmeans_alt',bool,False)
            max_top = int.bit_length(self.npatch)-1
            field = self.getNField(max_top=max_top)
            self.logger.info("Finding %d patches using kmeans.",self.npatch)
            self._patch, self._centers = field.run_kmeans(self.npatch, init=init, alt=alt)
            self._npatch = self.npatch
        elif self._centers is not None and self._patch is None and self._single_patch is None:
            if ((self.coords == 'flat' and self._centers.shape[1] != 2) or
                (self.coords != 'flat' and self._centers.shape[1] != 3)):
                raise ValueError("Centers array has wrong shape.")
            self._assign_patches()
            self.logger.info("Assigned patch numbers according %d centers",self._npatch)

        self.logger.info("   nobj = %d",self.nobj)

    def _assign_patches(self):
        # This is equivalent to the following:
        #   field = self.getNField()
        #   self._patch = field.kmeans_assign_patches(self._centers)
        # However, when the field is not already created, it's faster to just run through
        # all the points directly and assign which one is closest.
        from treecorr.util import double_ptr as dp
        from treecorr.util import long_ptr as lp
        self._patch = np.empty(self.ntot, dtype=int)
        self._npatch = self._centers.shape[0]
        centers = np.ascontiguousarray(self._centers)
        treecorr.set_omp_threads(self.config.get('num_threads',None))
        treecorr._lib.QuickAssign(dp(centers), self._npatch,
                                  dp(self.x), dp(self.y), dp(self.z), lp(self._patch), self.ntot)

    def _set_npatch(self):
        self._npatch = max(self._patch) + 1
        self.logger.info("Assigned patch numbers 0..%d",self._npatch-1)

    def _get_patch_index(self, single_patch):
        if self._patch is not None:
            # This is straightforward.  Just select the rows with patch == single_patch
            use = np.where(self._patch == single_patch)[0]
        elif self._centers is not None:
            self._generate_xyz()
            use = np.empty(self.ntot, dtype=int)
            from .util import double_ptr as dp
            from .util import long_ptr as lp
            npatch = self._centers.shape[0]
            centers = np.ascontiguousarray(self._centers)
            if self._z is None:
                assert centers.shape[1] == 2
            else:
                assert centers.shape[1] == 3
            treecorr.set_omp_threads(self.config.get('num_threads',None))
            treecorr._lib.SelectPatch(single_patch, dp(centers), npatch,
                                      dp(self._x), dp(self._y), dp(self._z),
                                      lp(use), self.ntot)
            use = np.where(use)[0]
        else:
            use = slice(None)  # Which ironically means use all. :)
        return use

    def _apply_units(self):
        # Apply units to x,y,ra,dec
        if self._ra is not None:
            self.ra_units = treecorr.config.get_from_list(self.config,'ra_units',self._num)
            self.dec_units = treecorr.config.get_from_list(self.config,'dec_units',self._num)
            self._ra *= self.ra_units
            self._dec *= self.dec_units
        else:
            self.x_units = treecorr.config.get_from_list(self.config,'x_units',self._num,str,'radians')
            self.y_units = treecorr.config.get_from_list(self.config,'y_units',self._num,str,'radians')
            self._x *= self.x_units
            self._y *= self.y_units

    def _generate_xyz(self):
        if self._x is None:
            assert self._y is None
            assert self._z is None
            assert self._ra is not None
            assert self._dec is not None
            ntot = len(self._ra)
            self._x = np.empty(ntot, dtype=float)
            self._y = np.empty(ntot, dtype=float)
            self._z = np.empty(ntot, dtype=float)
            from .util import double_ptr as dp
            treecorr.set_omp_threads(self.config.get('num_threads',None))
            treecorr._lib.GenerateXYZ(dp(self._x), dp(self._y), dp(self._z),
                                      dp(self._ra), dp(self._dec), dp(self._r), ntot)
            self.x_units = self.y_units = 1.

    def _select_patch(self, single_patch):
        # Trim the catalog to only include a single patch
        # Note: This is slightly inefficient in that it reads the whole catalog first
        # and then removes all but one patch.  But that's easier for now that figuring out
        # which items to remove along the way based on the patch_centers.
        indx = self._get_patch_index(single_patch)
        self._patch = None
        self.select(indx)
        self._npatch = 1

    def select(self, indx):
        """Trim the catalog to only include those objects with the give indices.

        Parameters:
            indx:       A numpy array of index values to keep.
        """
        if type(indx) == slice and indx == slice(None):
            return
        self._x = self._x[indx] if self._x is not None else None
        self._y = self._y[indx] if self._y is not None else None
        self._z = self._z[indx] if self._z is not None else None
        self._ra = self._ra[indx] if self._ra is not None else None
        self._dec = self._dec[indx] if self._dec is not None else None
        self._r = self._r[indx] if self._r is not None else None
        self._w = self._w[indx] if self._w is not None else None
        self._wpos = self._wpos[indx] if self._wpos is not None else None
        self._g1 = self._g1[indx] if self._g1 is not None else None
        self._g2 = self._g2[indx] if self._g2 is not None else None
        self._k = self._k[indx] if self._k is not None else None
        self._patch = self._patch[indx] if self._patch is not None else None

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
            col = col[self.start:self.end:self.every_nth]
        return col


    def checkForNaN(self, col, col_str):
        """Check if the column has any NaNs.  If so, set those rows to have w[k]=0.

        Parameters:
            col (array):    The input column to check.
            col_str (str):  The name of the column.  Used only as information in logging output.
        """
        if col is not None and np.any(np.isnan(col)):
            index = np.where(np.isnan(col))[0]
            self.logger.warning("Warning: NaNs found in %s column.  Skipping rows %s."%(
                                col_str,str(index.tolist())))
            if self._w is None:
                self._w = np.ones_like(col, dtype=float)
            self._w[index] = 0
            col[index] = 0  # Don't leave the nans there.

    def _check_ascii(self, file_name, num=0, is_rand=False):
        # Just check the consistency of the various column numbers so we can fail fast.

        # Just read 1 row so we know how many columns there are.
        # Also will give an IO error if the file is unreadable.
        comment_marker = self.config.get('comment_marker','#')
        delimiter = self.config.get('delimiter',None)
        data = np.genfromtxt(file_name, comments=comment_marker, delimiter=delimiter, max_rows=1)
        if len(data.shape) != 1:  # pragma: no cover
            raise IOError('Unable to parse the input catalog as a numpy array')
        ncols = data.shape[0]

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
        patch_col = treecorr.config.get_from_list(self.config,'patch_col',num,int,0)
        allow_xyz = self.config.get('allow_xyz', False)

        # Read x,y or ra,dec
        if x_col != 0 or y_col != 0:
            if x_col <= 0 or x_col > ncols:
                raise TypeError("x_col missing or invalid for file %s"%file_name)
            if y_col <= 0 or y_col > ncols:
                raise TypeError("y_col missing or invalid for file %s"%file_name)
            if z_col < 0 or z_col > ncols:
                raise TypeError("z_col is invalid for file %s"%file_name)
            if ra_col != 0 and not allow_xyz:
                raise TypeError("ra_col not allowed in conjunction with x/y cols")
            if dec_col != 0 and not allow_xyz:
                raise TypeError("dec_col not allowed in conjunction with x/y cols")
            if r_col != 0 and not allow_xyz:
                raise TypeError("r_col not allowed in conjunction with x/y cols")
        elif ra_col != 0 or dec_col != 0:
            if ra_col <= 0 or ra_col > ncols:
                raise TypeError("ra_col missing or invalid for file %s"%file_name)
            if dec_col <= 0 or dec_col > ncols:
                raise TypeError("dec_col missing or invalid for file %s"%file_name)
            if r_col < 0 or r_col > ncols:
                raise TypeError("r_col is invalid for file %s"%file_name)
            if z_col != 0 and not allow_xyz:
                raise TypeError("z_col not allowed in conjunction with ra/dec cols")
        else:
            raise TypeError("No valid position columns specified for file %s"%file_name)

        if w_col < 0 or w_col > ncols:
            raise TypeError("w_col is invalid for file %s"%file_name)
        if wpos_col < 0 or wpos_col > ncols:
            raise TypeError("wpos_col is invalid for file %s"%file_name)
        if flag_col < 0 or flag_col > ncols:
            raise TypeError("flag_col is invalid for file %s"%file_name)
        if patch_col < 0 or patch_col > ncols:
            raise ValueError("patch_col is invalid for file %s"%file_name)
        if is_rand: return

        if g1_col < 0 or g1_col > ncols or g2_col < 0 or g2_col > ncols or (g1_col!=0) != (g2_col!=0):
            if isGColRequired(self.orig_config,num):
                raise TypeError("g1_col, g2_col are invalid for file %s"%file_name)
            else:
                self.logger.warning("Warning: skipping g1_col, g2_col for %s, num=%d "%(
                                    file_name,num) +
                                    "because they are invalid, but unneeded.")

        if k_col < 0 or k_col > ncols:
            if isKColRequired(self.orig_config,num):
                raise TypeError("k_col is invalid for file %s"%file_name)
            else:
                self.logger.warning("Warning: skipping k_col for %s, num=%d "%(file_name,num)+
                                    "because it is invalid, but unneeded.")

    def read_ascii(self, file_name, num=0, is_rand=False):
        """Read the catalog from an ASCII file

        Parameters:
            file_name (str):    The name of the file to read in.
            num (int):          Which number catalog are we reading. (default: 0)
            is_rand (bool):     Is this a random catalog? (default: False)
        """
        comment_marker = self.config.get('comment_marker','#')
        delimiter = self.config.get('delimiter',None)
        # I want read_csv to ignore header lines that start with the comment marker, but
        # there is currently a bug in read_csv that messing things up when we do this.
        # cf. https://github.com/pydata/pandas/issues/4623
        # For now, my workaround in to count how many lines start with the comment marker
        # and skip them by hand.
        skiprows = 0
        with open(file_name, 'r') as fid:
            for line in fid:  # pragma: no branch
                if line.startswith(comment_marker): skiprows += 1
                else: break
        skiprows += self.start
        if self.end is None:
            nrows = None
        else:
            nrows = self.end - self.start
        if self.every_nth != 1:
            start = skiprows
            skiprows = lambda x: x < start or (x-start) % self.every_nth != 0
            nrows = (nrows-1) // self.every_nth + 1
        try:
            import pandas
            if delimiter is None:
                data = pandas.read_csv(file_name, comment=comment_marker, delim_whitespace=True,
                                       header=None, skiprows=skiprows, nrows=nrows)
            else:
                data = pandas.read_csv(file_name, comment=comment_marker, delimiter=delimiter,
                                       header=None, skiprows=skiprows, nrows=nrows)
            data = data.dropna(axis=0).values
        except ImportError:
            self.logger.warning("Unable to import pandas..  Using np.genfromtxt instead.\n"+
                                "Installing pandas is recommended for increased speed when "+
                                "reading ASCII catalogs.")
            if self.every_nth == 1:
                data = np.genfromtxt(file_name, comments=comment_marker, delimiter=delimiter,
                                     skip_header=skiprows, max_rows=nrows)
            else:
                # Numpy can't handle skiprows being a function.  Have to do this manually.
                data = np.genfromtxt(file_name, comments=comment_marker, delimiter=delimiter,
                                     skip_header=start, max_rows=self.end - self.start)
                data = data[::self.every_nth]

        # If only one row, and not using pands, then the shape comes in as one-d.  Reshape it:
        if len(data.shape) == 1:
            data = data.reshape(1,-1)
        ncols = data.shape[1]

        self.logger.debug('read data from %s, num=%d',file_name,num)
        self.logger.debug('data shape = %s',str(data.shape))

        # Get the column numbers
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
        patch_col = treecorr.config.get_from_list(self.config,'patch_col',num,int,0)

        # Read x,y or ra,dec
        if x_col != 0:
            # NB. astype always copies, even if the type is already correct.
            # We actually want this, since it makes the result contiguous in memory,
            # which we will need.
            self._x = data[:,x_col-1].astype(float)
            self.logger.debug('read x')
            self._y = data[:,y_col-1].astype(float)
            self.logger.debug('read y')
            if z_col != 0:
                self._z = data[:,z_col-1].astype(float)
                self.logger.debug('read r')
        if ra_col != 0:
            self._ra = data[:,ra_col-1].astype(float)
            self.logger.debug('read ra')
            self._dec = data[:,dec_col-1].astype(float)
            self.logger.debug('read dec')
            if r_col != 0:
                self._r = data[:,r_col-1].astype(float)
                self.logger.debug('read r')
        self._apply_units()

        # Read w
        if w_col != 0:
            self._w = data[:,w_col-1].astype(float)
            self.logger.debug('read w')

        # Read wpos
        if wpos_col != 0:
            self._wpos = data[:,wpos_col-1].astype(float)
            self.logger.debug('read wpos')

        # Read flag
        if flag_col != 0:
            self._flag = data[:,flag_col-1].astype(int)
            self.logger.debug('read flag')

        # Read patch
        if patch_col != 0:
            self._patch = data[:,patch_col-1].astype(int)
            self.logger.debug('read patch')
            self._set_npatch()

        # Skip g1,g2,k if this file is a random catalog
        if not is_rand:
            # Read g1,g2
            if g1_col >= 0 and g1_col <= ncols:
                self._g1 = data[:,g1_col-1].astype(float)
                self.logger.debug('read g1')
                self._g2 = data[:,g2_col-1].astype(float)
                self.logger.debug('read g2')

            # Read k
            if k_col >= 0 and k_col <= ncols:
                self._k = data[:,k_col-1].astype(float)
                self.logger.debug('read k')

        if self._single_patch is not None:
            self._select_patch(self._single_patch)

    def _check_fits(self, file_name, num=0, is_rand=False):
        # Just check the consistency of the various column numbers so we can fail fast.
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
        patch_col = treecorr.config.get_from_list(self.config,'patch_col',num,str,'0')
        allow_xyz = self.config.get('allow_xyz', False)

        if x_col != '0' or y_col != '0':
            if x_col == '0':
                raise ValueError("x_col missing for file %s"%file_name)
            if y_col == '0':
                raise ValueError("y_col missing for file %s"%file_name)
            if ra_col != '0' and not allow_xyz:
                raise ValueError("ra_col not allowed in conjunction with x/y cols")
            if dec_col != '0' and not allow_xyz:
                raise ValueError("dec_col not allowed in conjunction with x/y cols")
            if r_col != '0' and not allow_xyz:
                raise ValueError("r_col not allowed in conjunction with x/y cols")
        elif ra_col != '0' or dec_col != '0':
            if ra_col == '0':
                raise ValueError("ra_col missing for file %s"%file_name)
            if dec_col == '0':
                raise ValueError("dec_col missing for file %s"%file_name)
            if z_col != '0' and not allow_xyz:
                raise ValueError("z_col not allowed in conjunction with ra/dec cols")
        else:
            raise ValueError("No valid position columns specified for file %s"%file_name)

        if g1_col == '0' and isGColRequired(self.orig_config,num):
            raise ValueError("g1_col is missing for file %s"%file_name)
        if g2_col == '0' and isGColRequired(self.orig_config,num):
            raise ValueError("g2_col is missing for file %s"%file_name)
        if k_col == '0' and isKColRequired(self.orig_config,num):
            raise ValueError("k_col is missing for file %s"%file_name)

        if (g1_col != '0' and g2_col == '0') or (g1_col == '0' and g2_col != '0'):
            raise ValueError("g1_col, g2_col are invalid for file %s"%file_name)

        hdu = treecorr.config.get_from_list(self.config,'hdu',num,int,1)

        with fitsio.FITS(file_name, 'r') as fits:

            # Technically, this doesn't catch all possible errors.  If someone specifies
            # an invalid flag_hdu or something, then they'll get the fitsio error message.
            # But this should probably catch the majorit of error cases.
            if hdu not in fits:
                raise ValueError("Invalid hdu=%d for file %s"%(hdu,file_name))
            if not isinstance(fits[hdu], fitsio.hdu.TableHDU):
                raise ValueError("Invalid hdu=%d for file %s (Not a TableHDU)"%(hdu,file_name))

            if x_col != '0':
                x_hdu = treecorr.config.get_from_list(self.config,'x_hdu',num,int,hdu)
                y_hdu = treecorr.config.get_from_list(self.config,'y_hdu',num,int,hdu)
                if x_col not in fits[x_hdu].get_colnames():
                    raise ValueError("x_col is invalid for file %s"%file_name)
                if y_col not in fits[y_hdu].get_colnames():
                    raise ValueError("y_col is invalid for file %s"%file_name)
                if z_col != '0':
                    z_hdu = treecorr.config.get_from_list(self.config,'z_hdu',num,int,hdu)
                    if z_col not in fits[z_hdu].get_colnames():
                        raise ValueError("z_col is invalid for file %s"%file_name)
            else:
                ra_hdu = treecorr.config.get_from_list(self.config,'ra_hdu',num,int,hdu)
                dec_hdu = treecorr.config.get_from_list(self.config,'dec_hdu',num,int,hdu)
                if ra_col not in fits[ra_hdu].get_colnames():
                    raise ValueError("ra_col is invalid for file %s"%file_name)
                if dec_col not in fits[dec_hdu].get_colnames():
                    raise ValueError("dec_col is invalid for file %s"%file_name)
                if r_col != '0':
                    r_hdu = treecorr.config.get_from_list(self.config,'r_hdu',num,int,hdu)
                    if r_col not in fits[r_hdu].get_colnames():
                        raise ValueError("r_col is invalid for file %s"%file_name)

            if w_col != '0':
                w_hdu = treecorr.config.get_from_list(self.config,'w_hdu',num,int,hdu)
                if w_col not in fits[w_hdu].get_colnames():
                    raise ValueError("w_col is invalid for file %s"%file_name)

            if wpos_col != '0':
                wpos_hdu = treecorr.config.get_from_list(self.config,'wpos_hdu',num,int,hdu)
                if wpos_col not in fits[wpos_hdu].get_colnames():
                    raise ValueError("wpos_col is invalid for file %s"%file_name)

            if flag_col != '0':
                flag_hdu = treecorr.config.get_from_list(self.config,'flag_hdu',num,int,hdu)
                if flag_col not in fits[flag_hdu].get_colnames():
                    raise ValueError("flag_col is invalid for file %s"%file_name)

            if patch_col != '0':
                patch_hdu = treecorr.config.get_from_list(self.config,'patch_hdu',num,int,hdu)
                if patch_col not in fits[patch_hdu].get_colnames():
                    raise ValueError("patch_col is invalid for file %s"%file_name)

            if is_rand: return

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

            if k_col != '0':
                k_hdu = treecorr.config.get_from_list(self.config,'k_hdu',num,int,hdu)
                if k_col not in fits[k_hdu].get_colnames():
                    if isKColRequired(self.orig_config,num):
                        raise ValueError("k_col is invalid for file %s"%file_name)
                    else:
                        self.logger.warning("Warning: skipping k_col for %s, num=%d "%(
                                            file_name,num)+
                                            "because it is invalid, but unneeded.")

    def read_fits(self, file_name, num=0, is_rand=False):
        """Read the catalog from a FITS file

        Parameters:
            file_name (str):    The name of the file to read in.
            num (int):          Which number catalog are we reading. (default: 0)
            is_rand (bool):     Is this a random catalog? (default: False)
        """
        # Helper functions for things we might do in one of two places.
        def set_pos(data, x_col, y_col, z_col, ra_col, dec_col, r_col):
            if x_col != '0' and x_col in data:
                self._x = data[x_col].astype(float)
                self.logger.debug('read x')
                self._y = data[y_col].astype(float)
                self.logger.debug('read y')
                if z_col != '0':
                    self._z = data[z_col].astype(float)
                    self.logger.debug('read z')
            if ra_col != '0' and ra_col in data:
                self._ra = data[ra_col].astype(float)
                self.logger.debug('read ra')
                self._dec = data[dec_col].astype(float)
                self.logger.debug('read dec')
                if r_col != '0':
                    self._r = data[r_col].astype(float)
                    self.logger.debug('read r')
            self._apply_units()

        def set_patch(data, patch_col):
            if patch_col != '0' and patch_col in data:
                self._patch = data[patch_col].astype(int)
                self.logger.debug('read patch')
                self._set_npatch()

        import fitsio

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
        patch_col = treecorr.config.get_from_list(self.config,'patch_col',num,str,'0')

        hdu = treecorr.config.get_from_list(self.config,'hdu',num,int,1)

        with fitsio.FITS(file_name, 'r') as fits:

            # Figure out what slice to use.  If all rows, then None is faster,
            # otherwise give the range explicitly.
            # Note: this is a workaround for a bug in fitsio <= 1.0.6.
            # cf. https://github.com/esheldon/fitsio/pull/286
            # We should be able to always use s = slice(self.start, self.end, self.every_nth)
            if self.start == 0 and self.end is None and self.every_nth == 1:
                s = slice(None)
            else:
                if x_col != '0':
                    x_hdu = treecorr.config.get_from_list(self.config,'x_hdu',num,int,hdu)
                else:
                    x_hdu = treecorr.config.get_from_list(self.config,'ra_hdu',num,int,hdu)
                end = self.end if self.end is not None else fits[x_hdu].get_nrows()
                s = np.arange(self.start, end, self.every_nth)

            all_cols = [x_col, y_col, z_col,
                        ra_col, dec_col, r_col,
                        patch_col,
                        w_col, wpos_col, flag_col,
                        g1_col, g2_col, k_col]

            # It's faster to read in all the columsn in one read, rather than individually.
            # Typically (very close to always!), all the columns are in the same hdu.
            # Thus, the following would normally work fine.
            #     use_cols = [c for c in all_cols if c != '0']
            #     data = fits[hdu][use_cols][:]
            # However, we allow the option to have different columns read from different hdus.
            # So this is slightly more complicated.
            x_hdu = treecorr.config.get_from_list(self.config,'x_hdu',num,int,hdu)
            y_hdu = treecorr.config.get_from_list(self.config,'y_hdu',num,int,hdu)
            z_hdu = treecorr.config.get_from_list(self.config,'z_hdu',num,int,hdu)
            ra_hdu = treecorr.config.get_from_list(self.config,'ra_hdu',num,int,hdu)
            dec_hdu = treecorr.config.get_from_list(self.config,'dec_hdu',num,int,hdu)
            r_hdu = treecorr.config.get_from_list(self.config,'r_hdu',num,int,hdu)
            patch_hdu = treecorr.config.get_from_list(self.config,'patch_hdu',num,int,hdu)
            w_hdu = treecorr.config.get_from_list(self.config,'w_hdu',num,int,hdu)
            wpos_hdu = treecorr.config.get_from_list(self.config,'wpos_hdu',num,int,hdu)
            flag_hdu = treecorr.config.get_from_list(self.config,'flag_hdu',num,int,hdu)
            g1_hdu = treecorr.config.get_from_list(self.config,'g1_hdu',num,int,hdu)
            g2_hdu = treecorr.config.get_from_list(self.config,'g2_hdu',num,int,hdu)
            k_hdu = treecorr.config.get_from_list(self.config,'k_hdu',num,int,hdu)
            all_hdus = [x_hdu, y_hdu, z_hdu,
                        ra_hdu, dec_hdu, r_hdu,
                        patch_hdu,
                        w_hdu, wpos_hdu, flag_hdu,
                        g1_hdu, g2_hdu, k_hdu]
            col_by_hdu = dict(zip(all_cols,all_hdus))
            all_hdus = set(all_hdus + [hdu])
            all_cols = [c for c in all_cols if c != '0']

            data = {}
            # Also, if we are only reading in one patch, we should adjust s before doing this.
            if self._single_patch is not None:
                if patch_col != '0':
                    data[patch_col] = fits[patch_hdu][patch_col][s]
                    all_cols.remove(patch_col)
                    set_patch(data, patch_col)
                elif self._centers is not None:
                    pos_cols = [x_col, y_col, z_col, ra_col, dec_col, r_col]
                    pos_cols = [c for c in pos_cols if c != '0']
                    for c in pos_cols:
                        all_cols.remove(c)
                    for h in all_hdus:
                        use_cols1 = [c for c in pos_cols if col_by_hdu[c] == h]
                        data1 = fits[h][use_cols1][s]
                        for c in use_cols1:
                            data[c] = data1[c]
                    set_pos(data, x_col, y_col, z_col, ra_col, dec_col, r_col)
                use = self._get_patch_index(self._single_patch)
                self.select(use)
                if isinstance(s,np.ndarray):
                    s = s[use]
                else:
                    s = use
                self._patch = None
                data = {}  # Start fresh, since the ones we used so far are done.

                # We might actually be done now, in which case, just return.
                # (Else the fits read below won't actually work.)
                if len(all_cols) == 0:
                    return

            # Now read the rest using the updated s
            for h in all_hdus:
                use_cols1 = [c for c in all_cols if col_by_hdu[c] == h and
                                                    c in fits[h].get_colnames()]
                if len(use_cols1) == 0:
                    continue
                data1 = fits[h][use_cols1][s]
                for c in use_cols1:
                    data[c] = data1[c]

            # Set position values
            set_pos(data, x_col, y_col, z_col, ra_col, dec_col, r_col)

            # Set patch
            set_patch(data, patch_col)

            # Set w
            if w_col != '0':
                self._w = data[w_col].astype(float)
                self.logger.debug('read w')

            # Set wpos
            if wpos_col != '0':
                self._wpos = data[wpos_col].astype(float)
                self.logger.debug('read wpos')

            # Set flag
            if flag_col != '0':
                self._flag = data[flag_col].astype(int)
                self.logger.debug('read flag')

            # Skip g1,g2,k if this file is a random catalog
            if not is_rand:
                # Set g1,g2
                if g1_col in fits[g1_hdu].get_colnames():
                    self._g1 = data[g1_col].astype(float)
                    self.logger.debug('read g1')
                    self._g2 = data[g2_col].astype(float)
                    self.logger.debug('read g2')

                # Set k
                if k_col in fits[k_hdu].get_colnames():
                    self._k = data[k_col].astype(float)
                    self.logger.debug('read k')

    @property
    def nfields(self):
        if not hasattr(self, '_nfields'):
            # Make simple functions that call NField, etc. with self as the first argument.
            def get_nfield(*args, **kwargs): return treecorr.NField(self, *args, **kwargs)
            # Now wrap these in LRU_Caches with (initially) just 1 element being cached.
            self._nfields = treecorr.util.LRU_Cache(get_nfield, 1)
        return self._nfields

    @property
    def kfields(self):
        if not hasattr(self, '_kfields'):
            def get_kfield(*args, **kwargs): return treecorr.KField(self, *args, **kwargs)
            self._kfields = treecorr.util.LRU_Cache(get_kfield, 1)
        return self._kfields

    @property
    def gfields(self):
        if not hasattr(self, '_gfields'):
            def get_gfield(*args, **kwargs): return treecorr.GField(self, *args, **kwargs)
            self._gfields = treecorr.util.LRU_Cache(get_gfield, 1)
        return self._gfields

    @property
    def nsimplefields(self):
        if not hasattr(self, '_nsimplefields'):
            def get_nsimplefield(*args,**kwargs): return treecorr.NSimpleField(self,*args,**kwargs)
            self._nsimplefields = treecorr.util.LRU_Cache(get_nsimplefield, 1)
        return self._nsimplefields

    @property
    def ksimplefields(self):
        if not hasattr(self, '_ksimplefields'):
            def get_ksimplefield(*args,**kwargs): return treecorr.KSimpleField(self,*args,**kwargs)
            self._ksimplefields = treecorr.util.LRU_Cache(get_ksimplefield, 1)
        return self._ksimplefields

    @property
    def gsimplefields(self):
        if not hasattr(self, '_gsimplefields'):
            def get_gsimplefield(*args,**kwargs): return treecorr.GSimpleField(self,*args,**kwargs)
            self._gsimplefields = treecorr.util.LRU_Cache(get_gsimplefield, 1)
        return self._gsimplefields

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
        if hasattr(self, '_nfields'): self.nfields.resize(maxsize)
        if hasattr(self, '_kfields'): self.kfields.resize(maxsize)
        if hasattr(self, '_gfields'): self.gfields.resize(maxsize)
        if hasattr(self, '_nsimplefields'): self.nsimplefields.resize(maxsize)
        if hasattr(self, '_ksimplefields'): self.ksimplefields.resize(maxsize)
        if hasattr(self, '_gsimplefields'): self.gsimplefields.resize(maxsize)

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
        if hasattr(self, '_nfields'): self.nfields.clear()
        if hasattr(self, '_kfields'): self.kfields.clear()
        if hasattr(self, '_gfields'): self.gfields.clear()
        if hasattr(self, '_nsimplefields'): self.nsimplefields.clear()
        if hasattr(self, '_ksimplefields'): self.ksimplefields.clear()
        if hasattr(self, '_gsimplefields'): self.gsimplefields.clear()
        self._field = lambda : None  # Acts like a dead weakref

    @property
    def field(self):
        # The default is to return None here.
        # This might also return None if weakref has expired.
        # But if the weakref is alive, this returns the field we want.
        return self._field()

    def getNField(self, min_size=0, max_size=None, split_method=None, brute=False,
                  min_top=None, max_top=10, coords=None, logger=None):
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
                                field. (default: :math:`\\max(3, \\log_2(N_{\\rm cpu}))`)
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
                  min_top=None, max_top=10, coords=None, logger=None):
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
                                field. (default: :math:`\\max(3, \\log_2(N_{\\rm cpu}))`)
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
                  min_top=None, max_top=10, coords=None, logger=None):
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
                                field. (default: :math:`\\max(3, \\log_2(N_{\\rm cpu}))`)
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

    def _weighted_mean(self, x, idx=None):
        # Find the weighted mean of some column.
        # If weights are set, then return sum(w * x) / sum(w)
        # Else, just sum(x) / N
        if self.nontrivial_w:
            if idx is None:
                return np.sum(x * self.w) / self.sumw
            else:
                return np.sum(x[idx] * self.w[idx]) / np.sum(self.w[idx])
        else:
            return np.mean(x[idx])

    def get_patch_centers(self):
        """Return an array of patch centers corresponding to the patches in this catalog.

        If the patches were set either using K-Means or by giving the centers, then this
        will just return that same center array.  Otherwise, it will be calculated from the
        positions of the objects with each patch number.

        This function is automatically called when accessing the property
        ``patch_centers``.  So you should not normally need to call it directly.

        Returns:
            An array of center coordinates used to make the patches.
            Shape is (npatch, 2) for flat geometries or (npatch, 3) for 3d or
            spherical geometries.  In the latter case, the centers represent
            (x,y,z) coordinates on the unit sphere.
        """
        # Early exit
        if self._centers is not None:
            return self._centers

        self.load()
        if self._patch is None:
            if self.coords == 'flat':
                self._centers = np.array([[self._weighted_mean(self.x),
                                            self._weighted_mean(self.y)]])
            else:
                self._centers = np.array([[self._weighted_mean(self.x),
                                            self._weighted_mean(self.y),
                                            self._weighted_mean(self.z)]])
        else:
            npatch = self._npatch
            self._centers = np.empty((npatch,2 if self.z is None else 3))
            for p in range(npatch):
                indx = np.where(self.patch == p)[0]
                if len(indx) == 0:
                    raise RuntimeError("Cannot find center for patch %s."%p +
                                        "  No items with this patch number")
                if self.coords == 'flat':
                    self._centers[p] = [self._weighted_mean(self.x,indx),
                                        self._weighted_mean(self.y,indx)]
                else:
                    self._centers[p] = [self._weighted_mean(self.x,indx),
                                        self._weighted_mean(self.y,indx),
                                        self._weighted_mean(self.z,indx)]
        if self.coords == 'spherical':
            self._centers /= np.sqrt(np.sum(self._centers**2,axis=1))[:,np.newaxis]
        return self._centers

    def write_patch_centers(self, file_name):
        """Write the patch centers to a file.

        The output file will include the following columns:

        ========      =======================================================
        Column        Description
        ========      =======================================================
        patch         patch number (0..npatch-1)
        x             mean x values
        y             mean y values
        z             mean z values (only for spherical or 3d coordinates)
        ========      =======================================================

        It will write a FITS file if the file name ends with '.fits', otherwise an ASCII file.

        Parameters:
            file_name (str):    The name of the file to write to.
        """
        self.logger.info('Writing centers to %s',file_name)

        centers = self.patch_centers
        col_names = ['patch', 'x', 'y']
        if self.coords != 'flat':
            col_names.append('z')
        columns = [np.arange(centers.shape[0])]
        for i in range(centers.shape[1]):
            columns.append(centers[:,i])

        treecorr.util.gen_write(file_name, col_names, columns, precision=16, logger=self.logger)

    def read_patch_centers(self, file_name):
        """Read patch centers from a file.

        This function typically gets called automatically when setting patch_centers as a
        string, being the file name.  The patch centers are read from the file and returned.

        Parameters:
            file_name (str):    The name of the file to write to.

        Returns:
            The centers, as an array, which can be used to determine the patches.
        """
        self.logger.info('Reading centers from %s',file_name)

        data, params = treecorr.util.gen_read(file_name, logger=self.logger)
        if 'z' in data.dtype.names:
            return np.column_stack((data['x'],data['y'],data['z']))
        else:
            return np.column_stack((data['x'],data['y']))

    def load(self):
        """Load the data from a file, if it isn't yet loaded.

        When a Catalog is read in from a file, it tries to delay the loading of the data from
        disk until it is actually needed.  This is especially important when running over a
        set of patches, since you may not be able to fit all the patches in memory at once.

        One does not normally need to call this method explicitly.  It will run automatically
        whenever the data is needed.  However, if you want to directly control when the disk
        access happens, you can use this function.
        """
        if not self.loaded:
            self.logger.info("Reading input file %s",self.name)
            # Read the input file
            if self.file_type == 'FITS':
                self.read_fits(self.file_name,self._num,self._is_rand)
            elif self.file_type == 'ASCII':
                self.read_ascii(self.file_name,self._num,self._is_rand)
            else: # pragma: no cover
                # This is already checked, so shouldn't be possible to happen.
                raise ValueError("Invalid file_type %s"%self.file_type)
            self._finish_input()

    def unload(self):
        """Bring the Catalog back to an "unloaded" state, if possible.

        When a Catalog is read in from a file, it tries to delay the loading of the data from
        disk until it is actually needed.  After loading, this method will return the Catalog
        back to the unloaded state to recover the memory in the data arrays. If the Catalog is
        needed again during further processing, it will re-load the data from disk at that time.

        This will also call `clear_cache` to recover any memory from fields that have been
        constructed as well.

        If the Catalog was not read in from a file, then this function will only do the
        `clear_cache` step.
        """
        if self.file_type is not None:
            self._x = None
            self._y = None
            self._z = None
            self._ra = None
            self._dec = None
            self._r = None
            self._w = None
            self._wpos = None
            self._g1 = None
            self._g2 = None
            self._k = None
            self._patch = None
            if self._patches is not None:
                for p in self._patches:
                    p.unload()
        self.clear_cache()

    def get_patches(self, low_mem=False):
        """Return a list of Catalog instances each representing a single patch from this Catalog

        After calling this function once, the patches may be repeatedly accessed by the
        ``patches`` attribute, without triggering a rebuild of the patches.  Furthermore,
        if ``patches`` is accessed before calling this function, it will be called automatically
        (with the default low_mem parameter).

        Parameters:
            low_mem (bool):     Whether to try to leave the returned patch catalogs in an
                                "unloaded" state, wherein they will not load the data from a
                                file until they are used.  This only works if the current catalog
                                was loaded from a file or the patches were saved (using
                                ``save_patch_dir``). (default: False)
        """
        # Early exit
        if self._patches is not None:
            return self._patches

        if low_mem and self.file_name is not None:
            # This is a litle tricky, since we don't want to trigger a load if the catalog
            # isn't loaded yet.  So try to get the patches from centers or single_patch first.
            if self._centers is not None:
                patch_set = range(len(self._centers))
            elif self._single_patch is not None:
                patch_set = [self._single_patch]
            elif self._single_patch is not None or self.patch is None:
                # This triggers a load of the current catalog, but no choice here.
                patch_set = [None]
            else:
                patch_set = sorted(set(self.patch))
            centers = self._centers if self._patch is None else None
            self._patches = [Catalog(config=self.config, file_name=self.file_name,
                                     patch=i, npatch=1, patch_centers=centers)
                             for i in patch_set]
        elif self._single_patch is not None or self.patch is None:
            self._patches = [self]
        else:
            patch_set = sorted(set(self.patch))
            if len(patch_set) != self._npatch:
                self.logger.error("WARNING: Some patch numbers do not contain any objects!")
                missing = set(range(self._npatch)) - set(patch_set)
                self.logger.warning("The following patch numbers have no objects: %s",missing)
                self.logger.warning("This may be a problem depending on your use case.")
            self._patches = []
            for i in patch_set:
                indx = np.where(self.patch == i)[0]
                x=self.x[indx] if self.x is not None else None
                y=self.y[indx] if self.y is not None else None
                z=self.z[indx] if self.z is not None else None
                ra=self.ra[indx] if self.ra is not None else None
                dec=self.dec[indx] if self.dec is not None else None
                r=self.r[indx] if self.r is not None else None
                w=self.w[indx] if self.nontrivial_w else None
                wpos=self.wpos[indx] if self.wpos is not None else None
                g1=self.g1[indx] if self.g1 is not None else None
                g2=self.g2[indx] if self.g2 is not None else None
                k=self.k[indx] if self.k is not None else None
                check_wpos = self._wpos if self._wpos is not None else self._w
                kwargs = dict(keep_zero_weight=np.any(check_wpos==0))
                if self.ra is not None:
                    kwargs['ra_units'] = 'rad'
                    kwargs['dec_units'] = 'rad'
                    kwargs['allow_xyz'] = True
                p = Catalog(x=x, y=y, z=z, ra=ra, dec=dec, r=r, w=w, wpos=wpos,
                            g1=g1, g2=g2, k=k, patch=i, **kwargs)
                self._patches.append(p)

        # Write the patches to files if requested.
        if self.save_patch_dir is not None:
            file_names = []
            for i, p in enumerate(self._patches):
                if self.file_name is not None:
                    file_name = os.path.splitext(os.path.basename(self.file_name))[0]
                    file_name += '_%00d.fits'%i
                else:
                    file_name = 'patch%00d.fits'%i
                file_name = os.path.join(self.save_patch_dir, file_name)
                self.logger.info('Writing patch %d to %s',i,file_name)
                col_names = p.write(file_name)
                file_names.append(file_name)
                if low_mem:
                    p.unload()
            if low_mem:
                # If low_mem, replace _patches with a version the reads from these files.
                # This will typically be a lot faster for when the load does happen.
                kwargs = {c + '_col' : c for c in col_names if c != 'patch'}
                if 'ra' in col_names:
                    kwargs['ra_units'] = 'rad'
                    kwargs['dec_units'] = 'rad'
                self._patches = [Catalog(file_name=file_names[i], patch=i, **kwargs)
                                 for i in range(len(file_names))]

        return self._patches

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
        patch         self.patch if not None
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
        Returns:
            The column names that were written to the file as a list.
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
        if self._patch is not None:
            col_names.append('patch')
            columns.append(self.patch)

        if cat_precision is None:
            cat_precision = treecorr.config.get(self.config,'cat_precision',int,16)

        treecorr.util.gen_write(file_name, col_names, columns, precision=cat_precision,
                                file_type=file_type, logger=self.logger)
        return col_names

    def copy(self):
        """Make a copy"""
        import copy
        return copy.deepcopy(self)

    def __getstate__(self):
        d = self.__dict__.copy()
        d.pop('logger',None)  # Oh well.  This is just lost in the copy.  Can't be pickled.
        d.pop('_field',None)
        d.pop('_nfields',None)
        d.pop('_kfields',None)
        d.pop('_gfields',None)
        d.pop('_nsimplefields',None)
        d.pop('_ksimplefields',None)
        d.pop('_gsimplefields',None)
        return d

    def __setstate__(self, d):
        self.__dict__ = d
        self.logger = treecorr.config.setup_logger(
                treecorr.config.get(self.config,'verbose',int,1),
                self.config.get('log_file',None))
        self._field = lambda : None

    def __repr__(self):
        s = 'treecorr.Catalog('
        if self.loaded:
            if self.x is not None and self.ra is None: s += 'x='+repr(self.x)+','
            if self.y is not None and self.ra is None: s += 'y='+repr(self.y)+','
            if self.z is not None and self.ra is None: s += 'z='+repr(self.z)+','
            if self.ra is not None: s += 'ra='+repr(self.ra)+",ra_units='rad',"
            if self.dec is not None: s += 'dec='+repr(self.dec)+",dec_units='rad',"
            if self.r is not None: s += 'r='+repr(self.r)+','
            if self.nontrivial_w: s += 'w='+repr(self.w)+','
            if self.wpos is not None: s += 'wpos='+repr(self.wpos)+','
            if self.g1 is not None: s += 'g1='+repr(self.g1)+','
            if self.g2 is not None: s += 'g2='+repr(self.g2)+','
            if self.k is not None: s += 'k='+repr(self.k)+','
            if self.patch is not None: s += 'patch='+repr(self.patch)+','
            wpos = self._wpos if self._wpos is not None else self._w
            if np.any(wpos == 0): s += 'keep_zero_weight=True,'
            # remove the last ','
            s = s[:-1] + ')'
        else:
            # Catalog isn't loaded yet. Use file_name info here instead.
            s += 'file_name='+repr(self.file_name)+','
            s += 'config ='+repr(self.config)
            s += ')'
        return s

    def __eq__(self, other):
        return (isinstance(other, Catalog) and
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
                np.array_equal(self.k, other.k) and
                np.array_equal(self.patch, other.patch))


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
