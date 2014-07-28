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

# Dict describing the valid parameters, what types they are, and a description:
# Each value is a tuple with the following elements:
#    type
#    may_be_list
#    default value
#    list of valid values
#    description
corr2_valid_params = {

    # Parameters about the input catlogs

    'file_name' : (str , True, None, None,
            'The file(s) with the galaxy data.'),
    'do_auto_corr' : (bool, False, False,  None,
            'Whether to do auto-correlations within a list of files.'),
    'do_cross_corr' : (bool, False, True,  None,
            'Whether to do cross-correlations within a list of files.'),
    'file_name2' : (str, True, None,  None,
            'The file(s) to use for the second field for a cross-correlation.'),
    'rand_file_name' : (str, True, None,  None,
            'For NN correlations, a list of random files.'),
    'rand_file_name2' : (str, True, None, None,
            'The randoms for the second field for a cross-correlation.'),
    'file_list' : (str, False, None, None,
            'A text file with file names in lieu of file_name.'),
    'file_list2' : (str, False, None, None,
            'A text file with file names in lieu of file_name2.'),
    'rand_file_list' : (str, False, None, None,
            'A text file with file names in lieu of rand_file_name.'),
    'rand_file_list2' : (str, False, None, None,
            'A text file with file names in lieu of rand_file_name2.'),
    'file_type' : (str, False, None, ['ASCII', 'FITS'],
            'The file type of the input files. The default is to use the file name extension.'),
    'delimiter' : (str, False, '\0', None,
            'The delimeter between input valus in an ASCII catalog.'),
    'comment_marker' : (str, False, '#', None,
            'The first (non-whitespace) character of comment lines in an input ASCII catalog.'),
    'first_row' : (int, False, 1, None,
            'The first row to use from the input catalog'),
    'last_row' : (int, False, -1, None,
            'The last row to use from the input catalog.  The default is to use all of them.'),
    'x_col' : (str, False, '0', None,
            'Which column to use for x. Should be an integer for ASCII catalogs.'),
    'y_col' : (str, False, '0', None,
            'Which column to use for y. Should be an integer for ASCII catalogs.'),
    'ra_col' : (str, False, '0', None,
            'Which column to use for ra. Should be an integer for ASCII catalogs.'),
    'dec_col' : (str, False, '0', None,
            'Which column to use for dec. Should be an integer for ASCII catalogs.'),
    'x_units' : (str, True, 'arcsec', treecorr.angle_units.keys(),
            'The units of x values.'),
    'y_units' : (str, True, 'arcsec', treecorr.angle_units.keys(),
            'The units of y values.'),
    'ra_units' : (str, True, None, treecorr.angle_units.keys(),
            'The units of ra values. Required when using ra_col.'),
    'dec_units' : (str, True, None, treecorr.angle_units.keys(),
            'The units of dec values. Required when using dec_col.'),
    'g1_col' : (str, False, '0', None,
            'Which column to use for g1. Should be an integer for ASCII catalogs.'),
    'g2_col' : (str, False, '0', None,
            'Which column to use for g2. Should be an integer for ASCII catalogs.'),
    'k_col' : (str, False, '0', None,
            'Which column to use for kappa. Should be an integer for ASCII catalogs. '),
    'w_col' : (str, False, '0', None,
            'Which column to use for weight. Should be an integer for ASCII catalogs.'),
    'flag_col' : (str, False, '0', None,
            'Which column to use for flag. Should be an integer for ASCII catalogs.'),
    'ignore_flag': (int, False, None, None,
            'Ignore objects with flag & ignore_flag != 0 (bitwise &)'),
    'ok_flag': (int, False, 0, None,
            'Ignore objects with flag & ~ok_flag != 0 (bitwise &, ~)'),
    'hdu': (int, False, 1, None,
            'Which HDU in a fits file to use rather than hdu=1'),
    'x_hdu': (int, False, None, None,
            'Which HDU to use for the x_col. default is the global hdu value.'),
    'y_hdu': (int, False, None, None,
            'Which HDU to use for the y_col. default is the global hdu value.'),
    'ra_hdu': (int, False, None, None,
            'Which HDU to use for the ra_col. default is the global hdu value.'),
    'dec_hdu': (int, False, None, None,
            'Which HDU to use for the dec_col. default is the global hdu value.'),
    'g1_hdu': (int, False, None, None,
            'Which HDU to use for the g1_col. default is the global hdu value.'),
    'g2_hdu': (int, False, None, None,
            'Which HDU to use for the g2_col. default is the global hdu value.'),
    'k_hdu': (int, False, None, None,
            'Which HDU to use for the k_col. default is the global hdu value.'),
    'w_hdu': (int, False, None, None,
            'Which HDU to use for the w_col. default is the global hdu value.'),
    'flag_hdu': (int, False, None, None,
            'Which HDU to use for the flag_col. default is the global hdu value.'),
    'flip_g1' : (bool, False, False, None,
            'Whether to flip the sign of g1'),
    'flip_g2' : (bool, False, False, None,
            'Whether to flip the sign of g2'),
    'pairwise' : (bool, False, False, None,
            'Whether to do a pair-wise cross-correlation '),
    'project' : (bool, False, False, None,
            'Whether to do a tangent plane projection'),
    'project_ra' : (float, False, None, None,
            'The ra of the tangent point for projection.'),
    'project_dec' : (float, False, None, None,
            'The dec of the tangent point for projection.'),
    'projection' : (str, False, False, 'lambert',
            'Which kind of tangent plane projection to do.'),

    # Parameters about the binned correlation function to be calculated

    'min_sep' : (float, False, None, None,
            'The minimum separation to include in the output.'),
    'max_sep' : (float, False, None, None,
            'The maximum separation to include in the output.'),
    'nbins' : (int, False, None, None,
            'The number of output bins to use.'),
    'bin_size' : (float, False, None, None,
            'The size of the output bins in log(sep).'),
    'sep_units' : (str, False, 'arcsec', treecorr.angle_units.keys(),
            'The units to use for min_sep and max_sep.'),
    'bin_slop' : (float, False, 1, None,
            'The fraction of a bin width by which it is ok to let the pairs miss the correct bin.'),
    'smooth_scale' : (float, False, 0, None,
            'An optional smoothing scale to smooth the output values.'),

    # Parameters about the output file(s)

    'n2_file_name' : (str, False, None, None,
            'The output filename for point-point correlation function.'),
    'n2_statistic' : (str, False, 'compensated', ['compensated','simple'],
            'Which statistic to use for omega as the estimator fo the NN correlation function. '),
    'ng_file_name' : (str, False, None, None,
            'The output filename for point-shear correlation function.'),
    'ng_statistic' : (str, False, None, ['compensated', 'simple'],
            'Which statistic to use for the mean shear estimator of the NG correlation function. ',
            'The default is compensated if rand_files is given, otherwise simple'),
    'g2_file_name' : (str, False, None, None,
            'The output filename for shear-shear correlation function.'),
    'nk_file_name' : (str, False, None, None,
            'The output filename for point-kappa correlation function.'),
    'nk_statistic' : (str, False, None, ['compensated', 'simple'],
            'Which statistic to use for the mean kappa estimator of the NK correlation function. ',
            'The default is compensated if rand_files is given, otherwise simple'),
    'k2_file_name' : (str, False, None, None,
            'The output filename for kappa-kappa correlation function.'),
    'kg_file_name' : (str, False, None, None,
            'The output filename for kappa-shear correlation function.'),
    'precision' : (int, False, 3, None,
            'The number of digits after the decimal in the output.'),

    # Derived output quantities

    'm2_file_name' : (str, False, None, None,
            'The output filename for the aperture mass statistics.'),
    'm2_uform' : (str, False, 'Crittenden', ['Crittenden', 'Schneider'],
            'The function form of the aperture.'),
    'nm_file_name' : (str, False, None, None,
            'The output filename for <N Map> and related values.'),
    'norm_file_name' : (str, False, None,  None,
            'The output filename for <N Map>^2/<N^2><Map^2> and related values.'),

    # Miscellaneous parameters

    'verbose' : (int, False, 1, [0, 1, 2, 3],
            'How verbose the code should be during processing. ',
            '0 = Errors, 1 = Warnings, 2 = Progress, 3 = Debugging'),
    'num_threads' : (int, False, 1, None,
            'How many threads should be used. num_threads <= 0 means auto based on num cores.'),
    'split_method' : (str, False, 'mean', ['mean', 'median', 'middle'],
            'Which method to use for splitting cells.'),

}

def corr2(config, logger=None):
    """Run the full two-point correlation function code based on the parameters in the
    given config dict.

    The function print_corr2_params() will output information about the valid parameters
    that are expected to be in the config dict.

    Optionally a logger parameter maybe given, in which case it is used for logging.
    If not given, the logging will be based on the verbose and log_file parameters.
    """
    # Check that config doesn't have any extra parameters.
    # (Such values are probably typos.)
    # Also convert the given parameters to the correct type, etc.
    config = treecorr.config.check_config(config, corr2_valid_params)

    # Setup logger based on config verbose value
    if logger is None:
        verbose = config['verbose']
        log_file = config['log_file']
        logger = treecorr.config.setup_logger(verbose, log_file)

    # Set the number of threads
    num_threads = config['num_threads']
    if num_threads < 0:
        import multiprocessing
        num_threads = multiprocessing.cpu_count()
    logger.info('Using %d threads.',num_threads)

    # Read in the input files
    #cat1, cat2, rand1, rand2 = read_catalogs(config, logger)


def print_corr2_params():
    """Print information about the valid parameters that may be given to the corr2 function.
    """
    treecorr.config.print_params(corr2_valid_params)
