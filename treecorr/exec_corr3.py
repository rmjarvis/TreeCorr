# Copyright (c) 2003-2024 by Mike Jarvis
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
.. module:: exec_corr3
"""

from .catalog import Catalog, read_catalogs
from .corr3base import Corr3
from .config import setup_logger, check_config, print_params
from .util import set_omp_threads
from .nnncorrelation import NNNCorrelation
from .kkkcorrelation import KKKCorrelation
from .gggcorrelation import GGGCorrelation
from .nnkcorrelation import NNKCorrelation, NKNCorrelation, KNNCorrelation
from .nkkcorrelation import NKKCorrelation, KNKCorrelation, KKNCorrelation
from .nngcorrelation import NNGCorrelation, NGNCorrelation, GNNCorrelation
from .nggcorrelation import NGGCorrelation, GNGCorrelation, GGNCorrelation
from .kkgcorrelation import KKGCorrelation, KGKCorrelation, GKKCorrelation
from .kggcorrelation import KGGCorrelation, GKGCorrelation, GGKCorrelation


# Dict describing the valid parameters, what types they are, and a description:
# Each value is a tuple with the following elements:
#    type
#    may_be_list
#    default value
#    list of valid values
#    description
corr3_valid_params = {

    # Parameters about the input catalogs

    'file_name' : (str, True, None, None,
            'The file(s) with the galaxy data.'),
    'file_name2' : (str, True, None, None,
            'The file(s) to use for the second field for a cross-correlation.'),
    'file_name3' : (str, True, None, None,
            'The file(s) to use for the third field for a cross-correlation.'),
    'rand_file_name' : (str, True, None, None,
            'For NNN correlations, a list of random files.'),
    'file_list' : (str, False, None, None,
            'A text file with file names in lieu of file_name.'),
    'rand_file_list' : (str, False, None, None,
            'A text file with file names in lieu of rand_file_name.'),

    # Parameters about the output file(s)

    'nnn_file_name' : (str, False, None, None,
            'The output filename for count-count-count correlation function.'),
    'nnn_statistic' : (str, False, 'compensated', ['compensated','simple'],
            'Which statistic to use for omega as the estimator for the NNN correlation function. '),
    'kkk_file_name' : (str, False, None, None,
            'The output filename for scalar-scalar-scalar correlation function.'),
    'ggg_file_name' : (str, False, None, None,
            'The output filename for shear-shear-shear correlation function.'),

    'nnk_file_name' : (str, False, None, None,
            'The output filename for count-count-scalar correlation function.'),
    'nkn_file_name' : (str, False, None, None,
            'The output filename for count-scalar-count correlation function.'),
    'knn_file_name' : (str, False, None, None,
            'The output filename for scalar-count-count correlation function.'),

    'nkk_file_name' : (str, False, None, None,
            'The output filename for count-scalar-scalar correlation function.'),
    'knk_file_name' : (str, False, None, None,
            'The output filename for scalar-count-scalar correlation function.'),
    'kkn_file_name' : (str, False, None, None,
            'The output filename for scalar-scalar-count correlation function.'),

    'nng_file_name' : (str, False, None, None,
            'The output filename for count-count-shear correlation function.'),
    'ngn_file_name' : (str, False, None, None,
            'The output filename for count-shear-count correlation function.'),
    'gnn_file_name' : (str, False, None, None,
            'The output filename for shear-count-count correlation function.'),

    'ngg_file_name' : (str, False, None, None,
            'The output filename for count-shear-shear correlation function.'),
    'gng_file_name' : (str, False, None, None,
            'The output filename for shear-count-shear correlation function.'),
    'ggn_file_name' : (str, False, None, None,
            'The output filename for shear-shear-count correlation function.'),

    'kkg_file_name' : (str, False, None, None,
            'The output filename for scalar-scalar-shear correlation function.'),
    'kgk_file_name' : (str, False, None, None,
            'The output filename for scalar-shear-scalar correlation function.'),
    'gkk_file_name' : (str, False, None, None,
            'The output filename for shear-scalar-scalar correlation function.'),

    'kgg_file_name' : (str, False, None, None,
            'The output filename for scalar-shear-shear correlation function.'),
    'gkg_file_name' : (str, False, None, None,
            'The output filename for shear-scalar-shear correlation function.'),
    'ggk_file_name' : (str, False, None, None,
            'The output filename for shear-shear-scalar correlation function.'),

    # Derived output quantities

    'm3_file_name' : (str, False, None, None,
            'The output filename for the aperture mass skewness.'),
}

# Add in the valid parameters for the relevant classes
for c in [ Catalog, Corr3 ]:
    corr3_valid_params.update(c._valid_params)


corr3_aliases = {
}

def corr3(config, logger=None):
    """Run the full three-point correlation function code based on the parameters in the
    given config dict.

    The function `print_corr3_params` will output information about the valid parameters
    that are expected to be in the config dict.

    Optionally a logger parameter may be given, in which case it is used for logging.
    If not given, the logging will be based on the verbose and log_file parameters.

    :param config:  The configuration dict which defines what to do.
    :param logger:  If desired, a ``Logger`` object for logging. (default: None, in which case
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
    cat2 = read_catalogs(config, 'file_name2', 'file_list2', num=1, logger=logger)
    cat3 = read_catalogs(config, 'file_name3', 'file_list3', num=2, logger=logger)
    rand1 = read_catalogs(config, 'rand_file_name', 'rand_file_list', num=0, logger=logger)
    if len(cat1) == 0:
        raise TypeError("Either file_name or file_list is required")
    if len(cat2) == 0: cat2 = None
    if len(cat3) == 0: cat3 = None
    if len(rand1) == 0: rand1 = None
    if cat2 is None and cat3 is not None:
        raise TypeError("file_name3 is invalid without file_name2")
    logger.info("Done creating input catalogs")

    # Do GGG correlation function if necessary
    if 'ggg_file_name' in config or 'm3_file_name' in config:
        logger.warning("Performing GGG calculations...")
        ggg = GGGCorrelation(config, logger=logger)
        ggg.process(cat1, cat2, cat3)
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
        ddd.process(cat1, cat2, cat3)
        logger.info("Done DDD calculations.")

        drr = None
        rdd = None
        if rand1 is None:
            logger.warning("No random catalogs given.  Only doing ntri calculation.")
            rrr = None
        else:
            # Note: random handling here is currently only implemented for the primary
            # data catalog in the NNN workflow.
            logger.warning("Performing RRR calculations...")
            rrr = NNNCorrelation(config, logger=logger)
            rrr.process(rand1)
            logger.info("Done RRR calculations.")

        if rrr is not None and config['nnn_statistic'] == 'compensated':
            logger.warning("Performing DRR calculations...")
            drr = NNNCorrelation(config, logger=logger)
            drr.process(cat1,rand1, ordered=False)
            logger.info("Done DRR calculations.")
            logger.warning("Performing DDR calculations...")
            rdd = NNNCorrelation(config, logger=logger)
            rdd.process(rand1,cat1, ordered=False)
            logger.info("Done DDR calculations.")
        ddd.write(config['nnn_file_name'], rrr=rrr, drr=drr, rdd=rdd)
        logger.warning("Wrote NNN correlation to %s",config['nnn_file_name'])

    # Do KKK correlation function if necessary
    if 'kkk_file_name' in config:
        logger.warning("Performing KKK calculations...")
        kkk = KKKCorrelation(config, logger=logger)
        kkk.process(cat1, cat2, cat3)
        logger.info("Done KKK calculations.")
        kkk.write(config['kkk_file_name'])
        logger.warning("Wrote KKK correlation to %s",config['kkk_file_name'])

    # Do NNK correlation function if necessary
    # Note: random-catalog support is not implemented here for all cross-correlations
    # that include N.
    if 'nnk_file_name' in config:
        logger.warning("Performing NNK calculations...")
        nnk = NNKCorrelation(config, logger=logger)
        nnk.process(cat1, cat2, cat3)
        logger.info("Done NNK calculations.")
        nnk.write(config['nnk_file_name'])
        logger.warning("Wrote NNK correlation to %s",config['nnk_file_name'])

    # Do NKN correlation function if necessary
    if 'nkn_file_name' in config:
        logger.warning("Performing NKN calculations...")
        nkn = NKNCorrelation(config, logger=logger)
        nkn.process(cat1, cat2, cat3)
        logger.info("Done NKN calculations.")
        nkn.write(config['nkn_file_name'])
        logger.warning("Wrote NKN correlation to %s",config['nkn_file_name'])

    # Do KNN correlation function if necessary
    if 'knn_file_name' in config:
        logger.warning("Performing KNN calculations...")
        knn = KNNCorrelation(config, logger=logger)
        knn.process(cat1, cat2, cat3)
        logger.info("Done KNN calculations.")
        knn.write(config['knn_file_name'])
        logger.warning("Wrote KNN correlation to %s",config['knn_file_name'])

    # Do NKK correlation function if necessary
    if 'nkk_file_name' in config:
        logger.warning("Performing NKK calculations...")
        nkk = NKKCorrelation(config, logger=logger)
        nkk.process(cat1, cat2, cat3)
        logger.info("Done NKK calculations.")
        nkk.write(config['nkk_file_name'])
        logger.warning("Wrote NKK correlation to %s",config['nkk_file_name'])

    # Do KNK correlation function if necessary
    if 'knk_file_name' in config:
        logger.warning("Performing KNK calculations...")
        knk = KNKCorrelation(config, logger=logger)
        knk.process(cat1, cat2, cat3)
        logger.info("Done KNK calculations.")
        knk.write(config['knk_file_name'])
        logger.warning("Wrote KNK correlation to %s",config['knk_file_name'])

    # Do KKN correlation function if necessary
    if 'kkn_file_name' in config:
        logger.warning("Performing KKN calculations...")
        kkn = KKNCorrelation(config, logger=logger)
        kkn.process(cat1, cat2, cat3)
        logger.info("Done KKN calculations.")
        kkn.write(config['kkn_file_name'])
        logger.warning("Wrote KKN correlation to %s",config['kkn_file_name'])

    # Do NNG correlation function if necessary
    if 'nng_file_name' in config:
        logger.warning("Performing NNG calculations...")
        nng = NNGCorrelation(config, logger=logger)
        nng.process(cat1, cat2, cat3)
        logger.info("Done NNG calculations.")
        nng.write(config['nng_file_name'])
        logger.warning("Wrote NNG correlation to %s",config['nng_file_name'])

    # Do NGN correlation function if necessary
    if 'ngn_file_name' in config:
        logger.warning("Performing NGN calculations...")
        ngn = NGNCorrelation(config, logger=logger)
        ngn.process(cat1, cat2, cat3)
        logger.info("Done NGN calculations.")
        ngn.write(config['ngn_file_name'])
        logger.warning("Wrote NGN correlation to %s",config['ngn_file_name'])

    # Do GNN correlation function if necessary
    if 'gnn_file_name' in config:
        logger.warning("Performing GNN calculations...")
        gnn = GNNCorrelation(config, logger=logger)
        gnn.process(cat1, cat2, cat3)
        logger.info("Done GNN calculations.")
        gnn.write(config['gnn_file_name'])
        logger.warning("Wrote GNN correlation to %s",config['gnn_file_name'])

    # Do NGG correlation function if necessary
    if 'ngg_file_name' in config:
        logger.warning("Performing NGG calculations...")
        ngg = NGGCorrelation(config, logger=logger)
        ngg.process(cat1, cat2, cat3)
        logger.info("Done NGG calculations.")
        ngg.write(config['ngg_file_name'])
        logger.warning("Wrote NGG correlation to %s",config['ngg_file_name'])

    # Do GNG correlation function if necessary
    if 'gng_file_name' in config:
        logger.warning("Performing GNG calculations...")
        gng = GNGCorrelation(config, logger=logger)
        gng.process(cat1, cat2, cat3)
        logger.info("Done GNG calculations.")
        gng.write(config['gng_file_name'])
        logger.warning("Wrote GNG correlation to %s",config['gng_file_name'])

    # Do GGN correlation function if necessary
    if 'ggn_file_name' in config:
        logger.warning("Performing GGN calculations...")
        ggn = GGNCorrelation(config, logger=logger)
        ggn.process(cat1, cat2, cat3)
        logger.info("Done GGN calculations.")
        ggn.write(config['ggn_file_name'])
        logger.warning("Wrote GGN correlation to %s",config['ggn_file_name'])

    # Do KKG correlation function if necessary
    if 'kkg_file_name' in config:
        logger.warning("Performing KKG calculations...")
        kkg = KKGCorrelation(config, logger=logger)
        kkg.process(cat1, cat2, cat3)
        logger.info("Done KKG calculations.")
        kkg.write(config['kkg_file_name'])
        logger.warning("Wrote KKG correlation to %s",config['kkg_file_name'])

    # Do KGK correlation function if necessary
    if 'kgk_file_name' in config:
        logger.warning("Performing KGK calculations...")
        kgk = KGKCorrelation(config, logger=logger)
        kgk.process(cat1, cat2, cat3)
        logger.info("Done KGK calculations.")
        kgk.write(config['kgk_file_name'])
        logger.warning("Wrote KGK correlation to %s",config['kgk_file_name'])

    # Do GKK correlation function if necessary
    if 'gkk_file_name' in config:
        logger.warning("Performing GKK calculations...")
        gkk = GKKCorrelation(config, logger=logger)
        gkk.process(cat1, cat2, cat3)
        logger.info("Done GKK calculations.")
        gkk.write(config['gkk_file_name'])
        logger.warning("Wrote GKK correlation to %s",config['gkk_file_name'])

    # Do KGG correlation function if necessary
    if 'kgg_file_name' in config:
        logger.warning("Performing KGG calculations...")
        kgg = KGGCorrelation(config, logger=logger)
        kgg.process(cat1, cat2, cat3)
        logger.info("Done KGG calculations.")
        kgg.write(config['kgg_file_name'])
        logger.warning("Wrote KGG correlation to %s",config['kgg_file_name'])

    # Do GKG correlation function if necessary
    if 'gkg_file_name' in config:
        logger.warning("Performing GKG calculations...")
        gkg = GKGCorrelation(config, logger=logger)
        gkg.process(cat1, cat2, cat3)
        logger.info("Done GKG calculations.")
        gkg.write(config['gkg_file_name'])
        logger.warning("Wrote GKG correlation to %s",config['gkg_file_name'])

    # Do GGK correlation function if necessary
    if 'ggk_file_name' in config:
        logger.warning("Performing GGK calculations...")
        ggk = GGKCorrelation(config, logger=logger)
        ggk.process(cat1, cat2, cat3)
        logger.info("Done GGK calculations.")
        ggk.write(config['ggk_file_name'])
        logger.warning("Wrote GGK correlation to %s",config['ggk_file_name'])


def print_corr3_params():
    """Print information about the valid parameters that may be given to the `corr3` function.
    """
    print_params(corr3_valid_params)
