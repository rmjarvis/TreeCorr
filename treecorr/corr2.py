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

# Dict describing the valid parameters, what types they are, and a description:
# Each value is a tuple with the following elements:
#    type
#    may_be_list
#    default value
#    list of valid values
#    description
corr2_valid_params = {

    # Parameters about the input catlogs

    'file_name' : (str, True, None, None,
            'The file(s) with the galaxy data.'),
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
    'ra_col' : (str, True, '0', None,
            'Which column to use for ra. Should be an integer for ASCII catalogs.'),
    'dec_col' : (str, True, '0', None,
            'Which column to use for dec. Should be an integer for ASCII catalogs.'),
    'r_col' : (str, True, '0', None,
            'Which column to use for r.  Only valid with ra,dec. ',
            'Should be an integer for ASCII catalogs.'),
    'x_units' : (str, True, None, treecorr.angle_units.keys(),
            'The units of x values.'),
    'y_units' : (str, True, None, treecorr.angle_units.keys(),
            'The units of y values.'),
    'ra_units' : (str, True, None, treecorr.angle_units.keys(),
            'The units of ra values. Required when using ra_col.'),
    'dec_units' : (str, True, None, treecorr.angle_units.keys(),
            'The units of dec values. Required when using dec_col.'),
    'g1_col' : (str, True, '0', None,
            'Which column to use for g1. Should be an integer for ASCII catalogs.'),
    'g2_col' : (str, True, '0', None,
            'Which column to use for g2. Should be an integer for ASCII catalogs.'),
    'k_col' : (str, True, '0', None,
            'Which column to use for kappa. Should be an integer for ASCII catalogs. '),
    'w_col' : (str, True, '0', None,
            'Which column to use for weight. Should be an integer for ASCII catalogs.'),
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
    'flag_hdu': (int, True, None, None,
            'Which HDU to use for the flag_col. default is the global hdu value.'),
    'flip_g1' : (bool, True, False, None,
            'Whether to flip the sign of g1'),
    'flip_g2' : (bool, True, False, None,
            'Whether to flip the sign of g2'),
    'pairwise' : (bool, True, False, None,
            'Whether to do a pair-wise cross-correlation '),

    # Parameters about the binned correlation function to be calculated

    'min_sep' : (float, False, None, None,
            'The minimum separation to include in the output.'),
    'max_sep' : (float, False, None, None,
            'The maximum separation to include in the output.'),
    'nbins' : (int, False, None, None,
            'The number of output bins to use.'),
    'bin_size' : (float, False, None, None,
            'The size of the output bins in log(sep).'),
    'sep_units' : (str, False, None, treecorr.angle_units.keys(),
            'The units to use for min_sep and max_sep.  Also the units of the output r columns'),
    'bin_slop' : (float, False, 1, None,
            'The fraction of a bin width by which it is ok to let the pairs miss the correct bin.'),

    # Parameters about the output file(s)

    'nn_file_name' : (str, False, None, None,
            'The output filename for point-point correlation function.'),
    'nn_statistic' : (str, False, 'compensated', ['compensated','simple'],
            'Which statistic to use for omega as the estimator fo the NN correlation function. '),
    'ng_file_name' : (str, False, None, None,
            'The output filename for point-shear correlation function.'),
    'ng_statistic' : (str, False, None, ['compensated', 'simple'],
            'Which statistic to use for the mean shear estimator of the NG correlation function. ',
            'The default is compensated if rand_files is given, otherwise simple'),
    'gg_file_name' : (str, False, None, None,
            'The output filename for shear-shear correlation function.'),
    'nk_file_name' : (str, False, None, None,
            'The output filename for point-kappa correlation function.'),
    'nk_statistic' : (str, False, None, ['compensated', 'simple'],
            'Which statistic to use for the mean kappa estimator of the NK correlation function. ',
            'The default is compensated if rand_files is given, otherwise simple'),
    'kk_file_name' : (str, False, None, None,
            'The output filename for kappa-kappa correlation function.'),
    'kg_file_name' : (str, False, None, None,
            'The output filename for kappa-shear correlation function.'),
    'precision' : (int, False, 4, None,
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
            '0 = Errors Only, 1 = Warnings, 2 = Progress, 3 = Debugging'),
    'num_threads' : (int, False, None, None,
            'How many threads should be used. num_threads <= 0 means auto based on num cores.'),
    'split_method' : (str, False, 'mean', ['mean', 'median', 'middle'],
            'Which method to use for splitting cells.'),
    'log_file' : (str, False, None, None,
            'If desired, an output file for the logging output.',
            'The default is to write the output to stdout.'),
    'output_dots' : (bool, False, None, None,
            'Whether to output dots to the stdout during the C++-level computation.',
            'The default is True if verbose >= 2 and there is no log_file.  Else False.'),
}

corr2_aliases = {
    'n2_file_name' : 'nn_file_name',
    'n2_statistic' : 'nn_statistic',
    'g2_file_name' : 'gg_file_name',
    'k2_file_name' : 'kk_file_name',
}

def corr2(config, logger=None):
    """Run the full two-point correlation function code based on the parameters in the
    given config dict.

    The function print_corr2_params() will output information about the valid parameters
    that are expected to be in the config dict.

    Optionally a logger parameter maybe given, in which case it is used for logging.
    If not given, the logging will be based on the verbose and log_file parameters.

    :param config:  The configuration dict which defines what to do.
    :param logger:  If desired, a logger object for logging. (default: None, in which case
                    one will be built according to the config dict's verbose level.)
    """
    # Setup logger based on config verbose value
    if logger is None:
        logger = treecorr.config.setup_logger(
                treecorr.config.get(config,'verbose',int,1),
                config.get('log_file',None))

    # Check that config doesn't have any extra parameters.
    # (Such values are probably typos.)
    # Also convert the given parameters to the correct type, etc.
    config = treecorr.config.check_config(config, corr2_valid_params, corr2_aliases, logger)

    import pprint
    logger.debug('Using configuration dict:\n%s',pprint.pformat(config))

    if ( 'output_dots' not in config 
          and config.get('log_file',None) is None 
          and config['verbose'] >= 2 ):
        config['output_dots'] = True

    # Set the number of threads
    num_threads = config.get('num_threads',0)
    logger.debug('From config dict, num_threads = %d',num_threads)
    if num_threads <= 0:
        import multiprocessing
        num_threads = multiprocessing.cpu_count()
        logger.debug('multiprocessing.cpu_count() = %d',num_threads)
    if num_threads > 1:
        logger.debug('Telling OpenMP to use %d threads',num_threads)
        num_threads = treecorr.set_omp_threads(num_threads)
        logger.debug('OpenMP reports that it will use %d threads',num_threads)
        if num_threads > 1:
            logger.info('Using %d threads.',num_threads)
        elif 'num_threads' in config:
            # Only warn if the user specifically asked for num_threads != 1.
            logger.warn('Unable to use multiple threads, since OpenMP is not enabled.')

    # Read in the input files.  Each of these is a list.
    cat1 = treecorr.read_catalogs(config, 'file_name', 'file_list', 0, logger)
    if len(cat1) == 0:
        raise AttributeError("Either file_name or file_list is required")
    cat2 = treecorr.read_catalogs(config, 'file_name2', 'rand_file_list2', 1, logger)
    rand1 = treecorr.read_catalogs(config, 'rand_file_name', 'rand_file_list', 0, logger)
    rand2 = treecorr.read_catalogs(config, 'rand_file_name2', 'rand_file_list2', 1, logger)
    if len(cat2) == 0 and len(rand2) > 0:
        raise AttributeError("rand_file_name2 is invalid without file_name2")
    logger.info("Done reading input catalogs")

    # Do GG correlation function if necessary
    if 'gg_file_name' in config or 'm2_file_name' in config:
        logger.info("Start GG calculations...")
        gg = treecorr.GGCorrelation(config,logger)
        gg.process(cat1,cat2)
        logger.info("Done GG calculations.")
        if 'gg_file_name' in config:
            gg.write(config['gg_file_name'])
        if 'm2_file_name' in config:
            gg.writeMapSq(config['m2_file_name'])

    # Do NG correlation function if necessary
    if 'ng_file_name' in config or 'nm_file_name' in config or 'norm_file_name' in config:
        if len(cat2) == 0:
            raise AttributeError("file_name2 is required for ng correlation")
        logger.info("Start NG calculations...")
        ng = treecorr.NGCorrelation(config,logger)
        ng.process(cat1,cat2)
        logger.info("Done NG calculation.")

        # The default ng_statistic is compensated _iff_ rand files are given.
        rg = None
        if len(rand1) == 0:
            if config.get('ng_statistic',None) == 'compensated':
                raise AttributeError("rand_files is required for ng_statistic = compensated")
        elif config.get('ng_statistic','compensated') == 'compensated':
            rg = treecorr.NGCorrelation(config,logger)
            rg.process(rand1,cat2)
            logger.info("Done RG calculation.")

        if 'ng_file_name' in config:
            ng.write(config['ng_file_name'], rg)
        if 'nm_file_name' in config:
            ng.writeNMap(config['nm_file_name'], rg)

        if 'norm_file_name' in config:
            gg = treecorr.GGCorrelation(config,logger)
            gg.process(cat2)
            logger.info("Done GG calculation for norm")
            dd = treecorr.NNCorrelation(config,logger)
            dd.process(cat1)
            logger.info("Done DD calculation for norm")
            rr = treecorr.NNCorrelation(config,logger)
            rr.process(rand1)
            logger.info("Done RR calculation for norm")
            dr = None
            if config['nn_statistic'] == 'compensated':
                dr = treecorr.NNCorrelation(config,logger)
                dr.process(cat1,rand1)
                logger.info("Done DR calculation for norm")
            ng.writeNorm(config['norm_file_name'],gg,dd,rr,dr,rg)

    # Do NN correlation function if necessary
    if 'nn_file_name' in config:
        if len(rand1) == 0:
            raise AttributeError("rand_file_name is required for NN correlation")
        if len(cat2) > 0 and len(rand2) == 0:
            raise AttributeError("rand_file_name2 is required for NN cross-correlation")
        logger.info("Start NN calculations...")
        dd = treecorr.NNCorrelation(config,logger)
        dd.process(cat1,cat2)
        logger.info("Done NN calculations.")

        dr = None
        rd = None
        if len(cat2) == 0:
            rr = treecorr.NNCorrelation(config,logger)
            rr.process(rand1)
            logger.info("Done RR calculations.")

            if config['nn_statistic'] == 'compensated':
                dr = treecorr.NNCorrelation(config,logger)
                dr.process(cat1,rand1)
                logger.info("Done DR calculations.")
        else:
            rr = treecorr.NNCorrelation(config,logger)
            rr.process(rand1,rand2)
            logger.info("Done RR calculations.")

            if config['nn_statistic'] == 'compensated':
                dr = treecorr.NNCorrelation(config,logger)
                dr.process(cat1,rand2)
                logger.info("Done DR calculations.")
                rd = treecorr.NNCorrelation(config,logger)
                rd.process(rand1,cat2)
                logger.info("Done RD calculations.")
        dd.write(config['nn_file_name'],rr,dr,rd)

    # Do KK correlation function if necessary
    if 'kk_file_name' in config:
        logger.info("Start KK calculations...")
        kk = treecorr.KKCorrelation(config,logger)
        kk.process(cat1,cat2)
        logger.info("Done KK calculations.")
        kk.write(config['kk_file_name'])

    # Do NG correlation function if necessary
    if 'nk_file_name' in config:
        if len(cat2) == 0:
            raise AttributeError("file_name2 is required for nk correlation")
        logger.info("Start NK calculations...")
        nk = treecorr.NKCorrelation(config,logger)
        nk.process(cat1,cat2)
        logger.info("Done NK calculation.")

        rk = None
        if len(rand1) == 0:
            if config.get('nk_statistic',None) == 'compensated':
                raise AttributeError("rand_files is required for nk_statistic = compensated")
        elif config.get('nk_statistic','compensated') == 'compensated':
            rk = treecorr.NKCorrelation(config,logger)
            rk.process(rand1,cat2)
            logger.info("Done RK calculation.")

        nk.write(config['nk_file_name'], rk)

    # Do KG correlation function if necessary
    if 'kg_file_name' in config:
        if len(cat2) == 0:
            raise AttributeError("file_name2 is required for kg correlation")
        logger.info("Start KG calculations...")
        kg = treecorr.KGCorrelation(config,logger)
        kg.process(cat1,cat2)
        logger.info("Done KG calculation.")
        kg.write(config['kg_file_name'])


def print_corr2_params():
    """Print information about the valid parameters that may be given to the corr2 function.
    """
    treecorr.config.print_params(corr2_valid_params)
