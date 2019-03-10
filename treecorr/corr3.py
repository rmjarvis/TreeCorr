# Copyright (c) 2003-2015 by Mike Jarvis
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
.. module:: corr3
"""

import treecorr

# Dict describing the valid parameters, what types they are, and a description:
# Each value is a tuple with the following elements:
#    type
#    may_be_list
#    default value
#    list of valid values
#    description
corr3_valid_params = {

    # Parameters about the input catlogs

    'file_name' : (str, True, None, None,
            'The file(s) with the galaxy data.'),
    'file_name2' : (str, True, None,  None,
            'The file(s) to use for the second field for a cross-correlation.'),
    'file_name3' : (str, True, None,  None,
            'The file(s) to use for the third field for a cross-correlation.'),
    'rand_file_name' : (str, True, None,  None,
            'For NNN correlations, a list of random files.'),
    'rand_file_name2' : (str, True, None, None,
            'The randoms for the second field for a cross-correlation.'),
    'rand_file_name3' : (str, True, None, None,
            'The randoms for the third field for a cross-correlation.'),
    'file_list' : (str, False, None, None,
            'A text file with file names in lieu of file_name.'),
    'file_list2' : (str, False, None, None,
            'A text file with file names in lieu of file_name2.'),
    'file_list3' : (str, False, None, None,
            'A text file with file names in lieu of file_name3.'),
    'rand_file_list' : (str, False, None, None,
            'A text file with file names in lieu of rand_file_name.'),
    'rand_file_list2' : (str, False, None, None,
            'A text file with file names in lieu of rand_file_name2.'),
    'rand_file_list3' : (str, False, None, None,
            'A text file with file names in lieu of rand_file_name3.'),

    # Parameters about the output file(s)

    'nnn_file_name' : (str, False, None, None,
            'The output filename for point-point correlation function.'),
    'nnn_statistic' : (str, False, 'compensated', ['compensated','simple'],
            'Which statistic to use for omega as the estimator fo the NN correlation function. '),
    'kkk_file_name' : (str, False, None, None,
            'The output filename for kappa-kappa-kappa correlation function.'),
    'ggg_file_name' : (str, False, None, None,
            'The output filename for gamma-gamma-gamma correlation function.'),
    #'ng_file_name' : (str, False, None, None,
            #'The output filename for point-shear correlation function.'),
    #'ng_statistic' : (str, False, None, ['compensated', 'simple'],
            #'Which statistic to use for the mean shear estimator of the NG correlation function. ',
            #'The default is compensated if rand_files is given, otherwise simple'),
    #'gg_file_name' : (str, False, None, None,
            #'The output filename for shear-shear correlation function.'),
    #'nk_file_name' : (str, False, None, None,
            #'The output filename for point-kappa correlation function.'),
    #'nk_statistic' : (str, False, None, ['compensated', 'simple'],
            #'Which statistic to use for the mean kappa estimator of the NK correlation function. ',
            #'The default is compensated if rand_files is given, otherwise simple'),
    #'kg_file_name' : (str, False, None, None,
            #'The output filename for kappa-shear correlation function.'),

}

# Add in the valid parameters for the relevant classes
for c in [ treecorr.Catalog, treecorr.BinnedCorr3 ]:
    corr3_valid_params.update(c._valid_params)


corr3_aliases = {
    'n3_file_name' : 'nnn_file_name',
    'n3_statistic' : 'nnn_statistic',
    'k3_file_name' : 'kkk_file_name',
    'g3_file_name' : 'ggg_file_name',
}

def corr3(config, logger=None):
    """Run the full three-point correlation function code based on the parameters in the
    given config dict.

    The function print_corr3_params() will output information about the valid parameters
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
    config = treecorr.config.check_config(config, corr3_valid_params, corr3_aliases, logger)

    import pprint
    logger.debug('Using configuration dict:\n%s',pprint.pformat(config))

    if ( 'output_dots' not in config
          and config.get('log_file',None) is None
          and config['verbose'] >= 2 ):
        config['output_dots'] = True

    # Set the number of threads
    num_threads = config.get('num_threads',None)
    logger.debug('From config dict, num_threads = %s',num_threads)
    treecorr.set_omp_threads(num_threads, logger)

    # Read in the input files.  Each of these is a list.
    cat1 = treecorr.read_catalogs(config, 'file_name', 'file_list', 0, logger)
    if cat1 is None:
        raise AttributeError("Either file_name or file_list is required")
    cat2 = treecorr.read_catalogs(config, 'file_name2', 'rand_file_list2', 1, logger)
    cat3 = treecorr.read_catalogs(config, 'file_name3', 'rand_file_list3', 1, logger)
    rand1 = treecorr.read_catalogs(config, 'rand_file_name', 'rand_file_list', 0, logger)
    rand2 = treecorr.read_catalogs(config, 'rand_file_name2', 'rand_file_list2', 1, logger)
    rand3 = treecorr.read_catalogs(config, 'rand_file_name3', 'rand_file_list3', 1, logger)
    if cat2 is None and rand2 is not None:
        raise AttributeError("rand_file_name2 is invalid without file_name2")
    if cat3 is None and rand3 is not None:
        raise AttributeError("rand_file_name3 is invalid without file_name3")
    logger.info("Done reading input catalogs")

    # Do GGG correlation function if necessary
    if 'ggg_file_name' in config:
        logger.warning("Performing GGG calculations...")
        ggg = treecorr.GGGCorrelation(config,logger)
        ggg.process(cat1,cat2,cat3)
        logger.info("Done GGG calculations.")
        ggg.write(config['ggg_file_name'])

    # Do NNN correlation function if necessary
    if 'nnn_file_name' in config:
        logger.warning("Performing DDD calculations...")
        ddd = treecorr.NNNCorrelation(config,logger)
        ddd.process(cat1,cat2,cat3)
        logger.info("Done DDD calculations.")

        drr = None
        rdr = None
        rrd = None
        ddr = None
        drd = None
        rdd = None
        if rand1 is None:
            logger.warning("No random catalogs given.  Only doing ntri calculation.")
            rrr = None
        elif cat2 is None:
            logger.warning("Performing RRR calculations...")
            rrr = treecorr.NNNCorrelation(config,logger)
            rrr.process(rand1)
            logger.info("Done RRR calculations.")

            # For the next step, just make cat2 = cat3 = cat1 and rand2 = rand3 = rand1.
            cat2 = cat3 = cat1
            rand2 = rand3 = rand1
        else:
            if rand2 is None:
                raise AttributeError("rand_file_name2 is required when file_name2 is given")
            if cat3 is not None and rand3 is None:
                raise AttributeError("rand_file_name3 is required when file_name3 is given")
            if (cat2 is None) != (cat2 is None):
                raise NotImplementedError(
                    "Cannot yet handle 3-point corrleations with only two catalogs. "+
                    "Need both cat2 and cat3.")
            logger.warning("Performing RRR calculations...")
            rrr = treecorr.NNNCorrelation(config,logger)
            rrr.process(rand1,rand2,rand3)
            logger.info("Done RRR calculations.")

        if rrr is not None and config['nnn_statistic'] == 'compensated':
            logger.warning("Performing DRR calculations...")
            drr = treecorr.NNNCorrelation(config,logger)
            drr.process(cat1,rand2,rand3)
            logger.info("Done DRR calculations.")
            logger.warning("Performing DDR calculations...")
            ddr = treecorr.NNNCorrelation(config,logger)
            ddr.process(cat1,cat2,rand3)
            logger.info("Done DDR calculations.")
            logger.warning("Performing RDR calculations...")
            rdr = treecorr.NNNCorrelation(config,logger)
            rdr.process(rand1,cat2,rand3)
            logger.info("Done RDR calculations.")
            logger.warning("Performing RRD calculations...")
            rrd = treecorr.NNNCorrelation(config,logger)
            rrd.process(rand1,rand2,cat3)
            logger.info("Done RRD calculations.")
            logger.warning("Performing DRD calculations...")
            drd = treecorr.NNNCorrelation(config,logger)
            drd.process(cat1,rand2,cat3)
            logger.info("Done DRD calculations.")
            logger.warning("Performing RDD calculations...")
            rdd = treecorr.NNNCorrelation(config,logger)
            rdd.process(rand1,cat2,cat3)
            logger.info("Done RDD calculations.")
        ddd.write(config['nnn_file_name'],rrr,drr,rdr,rrd,ddr,drd,rdd)

    # Do KKK correlation function if necessary
    if 'kkk_file_name' in config:
        logger.warning("Performing KKK calculations...")
        kkk = treecorr.KKKCorrelation(config,logger)
        kkk.process(cat1,cat2,cat3)
        logger.info("Done KKK calculations.")
        kkk.write(config['kkk_file_name'])

    # Do NNG correlation function if necessary
    if False:
    #if 'nng_file_name' in config or 'nnm_file_name' in config:
        if cat3 is None:
            raise AttributeError("file_name3 is required for nng correlation")
        logger.warning("Performing NNG calculations...")
        nng = treecorr.NNGCorrelation(config,logger)
        nng.process(cat1,cat2,cat3)
        logger.info("Done NNG calculation.")

        # The default ng_statistic is compensated _iff_ rand files are given.
        rrg = None
        if rand1 is None:
            if config.get('nng_statistic',None) == 'compensated':
                raise AttributeError("rand_files is required for nng_statistic = compensated")
        elif config.get('nng_statistic','compensated') == 'compensated':
            rrg = treecorr.NNGCorrelation(config,logger)
            rrg.process(rand1,rand1,cat2)
            logger.info("Done RRG calculation.")

        if 'nng_file_name' in config:
            nng.write(config['nng_file_name'], rrg)
        if 'nnm_file_name' in config:
            nng.writeNNMap(config['nnm_file_name'], rrg)


    # Do NNK correlation function if necessary
    if False:
    #if 'nnk_file_name' in config:
        if cat3 is None:
            raise AttributeError("file_name3 is required for nnk correlation")
        logger.warning("Performing NNK calculations...")
        nnk = treecorr.NNKCorrelation(config,logger)
        nnk.process(cat1,cat2,cat3)
        logger.info("Done NNK calculation.")

        rrk = None
        if rand1 is None:
            if config.get('nnk_statistic',None) == 'compensated':
                raise AttributeError("rand_files is required for nnk_statistic = compensated")
        elif config.get('nnk_statistic','compensated') == 'compensated':
            rrk = treecorr.NNKCorrelation(config,logger)
            rrk.process(rand1,rand1,cat2)
            logger.info("Done RRK calculation.")

        nnk.write(config['nnk_file_name'], rrk)

    # Do KKG correlation function if necessary
    if False:
    #if 'kkg_file_name' in config:
        if cat3 is None:
            raise AttributeError("file_name3 is required for kkg correlation")
        logger.warning("Performing KKG calculations...")
        kkg = treecorr.KKGCorrelation(config,logger)
        kkg.process(cat1,cat2,cat3)
        logger.info("Done KKG calculation.")
        kkg.write(config['kkg_file_name'])


def print_corr3_params():
    """Print information about the valid parameters that may be given to the corr3 function.
    """
    treecorr.config.print_params(corr3_valid_params)
