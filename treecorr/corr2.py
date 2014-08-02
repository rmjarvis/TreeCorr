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

    'file_name' : (str, True, None, None,
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
    'x_units' : (str, True, 'arcsec', treecorr.angle_units.keys(),
            'The units of x values.'),
    'y_units' : (str, True, 'arcsec', treecorr.angle_units.keys(),
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
            'The units to use for min_sep and max_sep.  Also the units of the output r columns'),
    'bin_slop' : (float, False, 1, None,
            'The fraction of a bin width by which it is ok to let the pairs miss the correct bin.'),

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

    'verbose' : (int, False, 2, [0, 1, 2, 3],
            'How verbose the code should be during processing. ',
            '0 = Errors Only, 1 = Warnings, 2 = Progress, 3 = Debugging'),
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
    import pprint
    logger.debug('Using configuration dict:\n%s',pprint.pformat(config))

    # Set the number of threads
    num_threads = config['num_threads']
    if num_threads < 0:
        import multiprocessing
        num_threads = multiprocessing.cpu_count()
    if num_threads > 1:
        logger.info('Using %d threads.',num_threads)

    # Read in the input files.  Each of these is a list.
    cat1 = treecorr.read_catalogs(config, 'file_name', 'file_list', 0, logger)
    if len(cat1) == 0:
        raise AttributeError("Either file_name or file_list is required")
    cat2 = treecorr.read_catalogs(config, 'file_name2', 'rand_file_list2', 1, logger)
    rand1 = treecorr.read_catalogs(config, 'rand_file_name', 'rand_file_list', 0, logger,
                                   is_rand=True)
    rand2 = treecorr.read_catalogs(config, 'rand_file_name2', 'rand_file_list2', 1, logger, 
                                   is_rand=True)
    if len(cat2) == 0 and len(rand2) > 0:
        raise AttributeError("rand_file_name2 is invalid without file_name2")
    logger.info("Done reading input catalogs")

    # Do g2 correlation function if necessary
    if 'g2_file_name' in config or 'm2_file_name' in config:
        logger.info("Start g2 calculations...")
        gg = treecorr.G2Correlation(config,logger)
        gg.process(cat1,cat2)
        logger.info("Done g2 calculations.")
        if 'g2_file_name' in config:
            gg.write(config['g2_file_name'])
            logger.info("Wrote file %s",config['g2_file_name'])
        if 'm2_file_name' in config:
            gg.writeMapSq(config['m2_file_name'])
            logger.info("Wrote file %s",config['m2_file_name'])

    # Do ng correlation function if necessary
    if 'ng_file_name' in config or 'nm_file_name' in config or 'norm_file_name' in config:
        if len(cat2) == 0:
            raise AttributeError("file_name2 is required for ng correlation")
        logger.info("Start ng calculations...")
        ng = treecorr.NGCorrelation(config,logger)
        ng.process(cat1,cat2)
        logger.info("Done ng calculation.")

        # The default ng_statistic is compensated _iff_ rand files are given.
        if len(rand1) == 0:
            if config.get('ng_statistic',None) == 'compensated':
                raise AttributeError("rand_files is required for ng_statistic = compensated")
        elif config.get('ng_statistic','compensated'):
            rg = treecorr.NGCorrelation(config,logger)
            rg.process(rand1,cat2)
            logger.info("Done rg calculation.")
        else:
            rg = None

        if 'ng_file_name' in config:
            ng.write(config['ng_file_name'], rg)
            logger.info("Wrote file %s",config['ng_file_name'])
        if 'nm_file_name' in config:
            ng.writeNMap(config['nm_file_name'], rg)
            logger.info("Wrote file %s",config['nm_file_name'])

        if 'norm_file_name' in config:
            gg = treecorr.G2Correlation(config,logger)
            gg.process(cat2)
            logger.info("Done gg calculation for norm")
            nn = treecorr.N2Correlation(config,logger)
            nn.process(cat1)
            logger.info("Done nn calculation for norm")
            rr = treecorr.N2Correlation(config,logger)
            rr.process(rand1)
            logger.info("Done rr calculation for norm")
            if config['n2_statistic'] == 'compensated':
                nr = treecorr.N2Correlation(config,logger)
                nr.process(cat1,rand1)
                logger.info("Done nr calculation for norm")
            else:
                nr = None
            ng.writeNorm(config['norm_file_name'],gg,nn,rr,nr,rg)

    # Do n2 correlation function if necessary
    if 'n2_file_name' in config:
        if len(rand1) == 0:
            raise AttributeError("rand_file_name is required for n2 correlation")
        if len(cat2) > 0 and len(rand2) == 0:
            raise AttributeError("rand_file_name2 is required for n2 cross-correlation")
        logger.info("Start n2 calculations...")
        nn = treecorr.N2Correlation(config,logger)
        nn.process(cat1,cat2)
        logger.info("Done n2 calculations.")

        if len(cat2) == 0:
            rr = treecorr.N2Correlation(config,logger)
            rr.process(rand1)
            logger.info("Done r2 calculations.")

            if config['n2_statistic'] == 'compensated':
                nr = treecorr.N2Correlation(config,logger)
                nr.process(cat1,rand1)
                logger.info("Done nr calculations.")
            else:
                nr = None
            rn = None
        else:
            rr = treecorr.N2Correlation(config,logger)
            rr.process(rand1,rand2)
            logger.info("Done r2 calculations.")

            if config['n2_statistic'] == 'compensated':
                nr = treecorr.N2Correlation(config,logger)
                nr.process(cat1,rand2)
                logger.info("Done nr calculations.")
                rn = treecorr.N2Correlation(config,logger)
                rn.process(rand1,cat2)
                logger.info("Done rn calculations.")
            else:
                nr = None
                rn = None
        nn.write(config['n2_file_name'],rr,nr,rn)

    # Do k2 correlation function if necessary
    if 'k2_file_name' in config:
        logger.info("Start k2 calculations...")
        kk = treecorr.K2Correlation(config,logger)
        kk.process(cat1,cat2)
        logger.info("Done k2 calculations.")
        kk.write(config['k2_file_name'])
        logger.info("Wrote file %s",config['k2_file_name'])

    # Do ng correlation function if necessary
    if 'nk_file_name' in config:
        if len(cat2) == 0:
            raise AttributeError("file_name2 is required for nk correlation")
        logger.info("Start nk calculations...")
        nk = treecorr.NKCorrelation(config,logger)
        nk.process(cat1,cat2)
        logger.info("Done nk calculation.")

        if len(rand1) == 0:
            if config.get('nk_statistic',None) == 'compensated':
                raise AttributeError("rand_files is required for nk_statistic = compensated")
        elif config.get('nk_statistic','compensated'):
            rk = treecorr.NKCorrelation(config,logger)
            rk.process(rand1,cat2)
            logger.info("Done rk calculation.")
        else:
            rk = None

        nk.write(config['nk_file_name'], rk)
        logger.info("Wrote file %s",config['nk_file_name'])

    # Do kg correlation function if necessary
    if 'kg_file_name' in config:
        if len(cat2) == 0:
            raise AttributeError("file_name2 is required for kg correlation")
        logger.info("Start kg calculations...")
        kg = treecorr.KGCorrelation(config,logger)
        kg.process(cat1,cat2)
        logger.info("Done kg calculation.")
        kg.write(config['kg_file_name'])
        logger.info("Wrote file %s",config['kg_file_name'])



def print_corr2_params():
    """Print information about the valid parameters that may be given to the corr2 function.
    """
    treecorr.config.print_params(corr2_valid_params)
