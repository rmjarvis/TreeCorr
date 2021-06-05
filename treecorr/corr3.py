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
.. module:: corr3
"""

from .catalog import Catalog, read_catalogs
from .binnedcorr3 import BinnedCorr3
from .config import setup_logger, check_config, print_params
from .util import set_omp_threads
from .nnncorrelation import NNNCorrelation
from .kkkcorrelation import KKKCorrelation
from .gggcorrelation import GGGCorrelation


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
    'rand_file_name' : (str, True, None, None,
            'For NNN correlations, a list of random files.'),
    'file_list' : (str, False, None, None,
            'A text file with file names in lieu of file_name.'),
    'rand_file_list' : (str, False, None, None,
            'A text file with file names in lieu of rand_file_name.'),

    # Parameters about the output file(s)

    'nnn_file_name' : (str, False, None, None,
            'The output filename for point-point correlation function.'),
    'nnn_statistic' : (str, False, 'compensated', ['compensated','simple'],
            'Which statistic to use for omega as the estimator fo the NN correlation function. '),
    'kkk_file_name' : (str, False, None, None,
            'The output filename for kappa-kappa-kappa correlation function.'),
    'ggg_file_name' : (str, False, None, None,
            'The output filename for gamma-gamma-gamma correlation function.'),

    # Derived output quantities

    'm3_file_name' : (str, False, None, None,
            'The output filename for the aperture mass skewness.'),
}

# Add in the valid parameters for the relevant classes
for c in [ Catalog, BinnedCorr3 ]:
    corr3_valid_params.update(c._valid_params)


corr3_aliases = {
}

def corr3(config, logger=None):
    """Run the full three-point correlation function code based on the parameters in the
    given config dict.

    The function `print_corr3_params` will output information about the valid parameters
    that are expected to be in the config dict.

    Optionally a logger parameter maybe given, in which case it is used for logging.
    If not given, the logging will be based on the verbose and log_file parameters.

    :param config:  The configuration dict which defines what to do.
    :param logger:  If desired, a logger object for logging. (default: None, in which case
                    one will be built according to the config dict's verbose level.)
    """
    # Setup logger based on config verbose value
    if logger is None:
        logger = setup_logger(config.get('verbose',1), config.get('log_file',None))

    # Check that config doesn't have any extra parameters.
    # (Such values are probably typos.)
    # Also convert the given parameters to the correct type, etc.
    config = check_config(config, corr3_valid_params, corr3_aliases, logger)

    import pprint
    logger.debug('Using configuration dict:\n%s',pprint.pformat(config))

    if ('output_dots' not in config
            and config.get('log_file',None) is None
            and config['verbose'] >= 2):
        config['output_dots'] = True

    # Set the number of threads
    num_threads = config.get('num_threads',None)
    logger.debug('From config dict, num_threads = %s',num_threads)
    set_omp_threads(num_threads, logger)

    # Read in the input files.  Each of these is a list.
    cat1 = read_catalogs(config, 'file_name', 'file_list', num=0, logger=logger)
    # TODO: when giving file_name2, file_name3, should now do the real CrossCorrelation process.
    rand1 = read_catalogs(config, 'rand_file_name', 'rand_file_list', num=0, logger=logger)
    if len(cat1) == 0:
        raise TypeError("Either file_name or file_list is required")
    if len(rand1) == 0: rand1 = None
    logger.info("Done creating input catalogs")

    # Do GGG correlation function if necessary
    if 'ggg_file_name' in config or 'm3_file_name' in config:
        logger.warning("Performing GGG calculations...")
        ggg = GGGCorrelation(config, logger=logger)
        ggg.process(cat1)
        logger.info("Done GGG calculations.")
        if 'ggg_file_name' in config:
            ggg.write(config['ggg_file_name'])
            logger.warning("Wrote GGG correlation to %s",config['ggg_file_name'])
        if 'm3_file_name' in config:
            ggg.writeMap3(config['m3_file_name'])
            logger.warning("Wrote Map3 values to %s",config['m3_file_name'])

    # Do NNN correlation function if necessary
    if 'nnn_file_name' in config:
        logger.warning("Performing DDD calculations...")
        ddd = NNNCorrelation(config, logger=logger)
        ddd.process(cat1)
        logger.info("Done DDD calculations.")

        drr = None
        rdd = None
        if rand1 is None:
            logger.warning("No random catalogs given.  Only doing ntri calculation.")
            rrr = None
        else:
            logger.warning("Performing RRR calculations...")
            rrr = NNNCorrelation(config, logger=logger)
            rrr.process(rand1)
            logger.info("Done RRR calculations.")

        if rrr is not None and config['nnn_statistic'] == 'compensated':
            logger.warning("Performing DRR calculations...")
            drr = NNNCorrelation(config, logger=logger)
            drr.process(cat1,rand1)
            logger.info("Done DRR calculations.")
            logger.warning("Performing DDR calculations...")
            rdd = NNNCorrelation(config, logger=logger)
            rdd.process(rand1,cat1)
            logger.info("Done DDR calculations.")
        ddd.write(config['nnn_file_name'], rrr=rrr, drr=drr, rdd=rdd)
        logger.warning("Wrote NNN correlation to %s",config['nnn_file_name'])

    # Do KKK correlation function if necessary
    if 'kkk_file_name' in config:
        logger.warning("Performing KKK calculations...")
        kkk = KKKCorrelation(config, logger=logger)
        kkk.process(cat1)
        logger.info("Done KKK calculations.")
        kkk.write(config['kkk_file_name'])
        logger.warning("Wrote KKK correlation to %s",config['kkk_file_name'])


def print_corr3_params():
    """Print information about the valid parameters that may be given to the `corr3` function.
    """
    print_params(corr3_valid_params)
