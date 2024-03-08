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
.. module:: corr2ex
"""

from .catalog import Catalog, read_catalogs
from .corr2base import Corr2
from .config import setup_logger, check_config, print_params, get
from .util import set_omp_threads
from .nncorrelation import NNCorrelation
from .nkcorrelation import NKCorrelation
from .kkcorrelation import KKCorrelation
from .nzcorrelation import NZCorrelation
from .kzcorrelation import KZCorrelation
from .zzcorrelation import ZZCorrelation
from .nvcorrelation import NVCorrelation
from .kvcorrelation import KVCorrelation
from .vvcorrelation import VVCorrelation
from .ngcorrelation import NGCorrelation
from .kgcorrelation import KGCorrelation
from .ggcorrelation import GGCorrelation
from .ntcorrelation import NTCorrelation
from .ktcorrelation import KTCorrelation
from .ttcorrelation import TTCorrelation
from .nqcorrelation import NQCorrelation
from .kqcorrelation import KQCorrelation
from .qqcorrelation import QQCorrelation

# Dict describing the valid parameters, what types they are, and a description:
# Each value is a tuple with the following elements:
#    type
#    may_be_list
#    default value
#    list of valid values
#    description
corr2_valid_params = {

    # Parameters about the input catalogs

    'file_name' : (str, True, None, None,
            'The file(s) with the galaxy data.'),
    'file_name2' : (str, True, None, None,
            'The file(s) to use for the second field for a cross-correlation.'),
    'rand_file_name' : (str, True, None, None,
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

    # Parameters about the output file(s)

    'nn_file_name' : (str, False, None, None,
            'The output filename for point-point correlation function.'),
    'nn_statistic' : (str, False, 'compensated', ['compensated','simple'],
            'Which statistic to use for omega as the estimator fo the NN correlation function. '),
    'nk_file_name' : (str, False, None, None,
            'The output filename for count-scalar correlation function.'),
    'nk_statistic' : (str, False, None, ['compensated', 'simple'],
            'Which statistic to use for the estimator of the NK correlation function. ',
            'The default is compensated if rand_files is given, otherwise simple'),
    'kk_file_name' : (str, False, None, None,
            'The output filename for scalar-scalar correlation function.'),

    'nz_file_name' : (str, False, None, None,
            'The output filename for point-complex correlation function.'),
    'nz_statistic' : (str, False, None, ['compensated', 'simple'],
            'Which statistic to use for the estimator of the NZ correlation function. ',
            'The default is compensated if rand_files is given, otherwise simple'),
    'kz_file_name' : (str, False, None, None,
            'The output filename for scalar-complex correlation function.'),
    'zz_file_name' : (str, False, None, None,
            'The output filename for complex-complex correlation function.'),

    'nv_file_name' : (str, False, None, None,
            'The output filename for point-vector correlation function.'),
    'nv_statistic' : (str, False, None, ['compensated', 'simple'],
            'Which statistic to use for the estimator of the NV correlation function. ',
            'The default is compensated if rand_files is given, otherwise simple'),
    'kv_file_name' : (str, False, None, None,
            'The output filename for scalar-vector correlation function.'),
    'vv_file_name' : (str, False, None, None,
            'The output filename for vector-vector correlation function.'),

    'ng_file_name' : (str, False, None, None,
            'The output filename for point-shear correlation function.'),
    'ng_statistic' : (str, False, None, ['compensated', 'simple'],
            'Which statistic to use for the estimator of the NG correlation function. ',
            'The default is compensated if rand_files is given, otherwise simple'),
    'kg_file_name' : (str, False, None, None,
            'The output filename for scalar-shear correlation function.'),
    'gg_file_name' : (str, False, None, None,
            'The output filename for shear-shear correlation function.'),

    'nt_file_name' : (str, False, None, None,
            'The output filename for point-trefoil correlation function.'),
    'nt_statistic' : (str, False, None, ['compensated', 'simple'],
            'Which statistic to use for the estimator of the NT correlation function. ',
            'The default is compensated if rand_files is given, otherwise simple'),
    'kt_file_name' : (str, False, None, None,
            'The output filename for scalar-trefoil correlation function.'),
    'tt_file_name' : (str, False, None, None,
            'The output filename for trefoil-trefoil correlation function.'),

    'nq_file_name' : (str, False, None, None,
            'The output filename for point-quatrefoil correlation function.'),
    'nq_statistic' : (str, False, None, ['compensated', 'simple'],
            'Which statistic to use for the estimator of the NQ correlation function. ',
            'The default is compensated if rand_files is given, otherwise simple'),
    'kq_file_name' : (str, False, None, None,
            'The output filename for scalar-quatrefoil correlation function.'),
    'qq_file_name' : (str, False, None, None,
            'The output filename for quatrefoil-quatrefoil correlation function.'),

    # Derived output quantities

    'm2_file_name' : (str, False, None, None,
            'The output filename for the aperture mass statistics.'),
    'nm_file_name' : (str, False, None, None,
            'The output filename for <N Map> and related values.'),
    'norm_file_name' : (str, False, None, None,
            'The output filename for <N Map>^2/<N^2><Map^2> and related values.'),
}

# Add in the valid parameters for the relevant classes
for c in [ Catalog, Corr2 ]:
    corr2_valid_params.update(c._valid_params)

corr2_aliases = {
}

def corr2(config, logger=None):
    """Run the full two-point correlation function code based on the parameters in the
    given config dict.

    The function `print_corr2_params` will output information about the valid parameters
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
    config = check_config(config, corr2_valid_params, corr2_aliases, logger)

    # Mark that we are running the corr2 function.
    config['corr2'] = True

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
    rand1 = read_catalogs(config, 'rand_file_name', 'rand_file_list', num=0, logger=logger)
    rand2 = read_catalogs(config, 'rand_file_name2', 'rand_file_list2', num=1, logger=logger)
    if len(cat1) == 0:
        raise TypeError("Either file_name or file_list is required")
    if len(cat2) == 0: cat2 = None
    if len(rand1) == 0: rand1 = None
    if len(rand2) == 0: rand2 = None
    if cat2 is None and rand2 is not None:
        raise TypeError("rand_file_name2 is invalid without file_name2")
    logger.info("Done creating input catalogs")

    # Do NN correlation function if necessary
    if 'nn_file_name' in config:
        logger.warning("Performing DD calculations...")
        dd = NNCorrelation(config, logger=logger)
        dd.process(cat1,cat2)
        logger.info("Done DD calculations.")

        dr = None
        rd = None
        if rand1 is None:
            logger.warning("No random catalogs given.  Only doing npairs calculation.")
            rr = None
        elif cat2 is None:
            logger.warning("Performing RR calculations...")
            rr = NNCorrelation(config, logger=logger)
            rr.process(rand1)
            logger.info("Done RR calculations.")

            if config['nn_statistic'] == 'compensated':
                logger.warning("Performing DR calculations...")
                dr = NNCorrelation(config, logger=logger)
                dr.process(cat1,rand1)
                logger.info("Done DR calculations.")
        else:
            if rand2 is None:
                raise TypeError("rand_file_name2 is required when file_name2 is given")
            logger.warning("Performing RR calculations...")
            rr = NNCorrelation(config, logger=logger)
            rr.process(rand1,rand2)
            logger.info("Done RR calculations.")

            if config['nn_statistic'] == 'compensated':
                logger.warning("Performing DR calculations...")
                dr = NNCorrelation(config, logger=logger)
                dr.process(cat1,rand2)
                logger.info("Done DR calculations.")
                rd = NNCorrelation(config, logger=logger)
                rd.process(rand1,cat2)
                logger.info("Done RD calculations.")
        dd.write(config['nn_file_name'], rr=rr, dr=dr, rd=rd)
        logger.warning("Wrote NN correlation to %s",config['nn_file_name'])

    # Do NK correlation function if necessary
    if 'nk_file_name' in config:
        if cat2 is None:
            raise TypeError("file_name2 is required for nk correlation")
        logger.warning("Performing NK calculations...")
        nk = NKCorrelation(config, logger=logger)
        nk.process(cat1,cat2)
        logger.info("Done NK calculation.")

        rk = None
        if rand1 is None:
            if config.get('nk_statistic',None) == 'compensated':
                raise TypeError("rand_files is required for nk_statistic = compensated")
        elif config.get('nk_statistic','compensated') == 'compensated':
            rk = NKCorrelation(config, logger=logger)
            rk.process(rand1,cat2)
            logger.info("Done RK calculation.")

        nk.write(config['nk_file_name'], rk=rk)
        logger.warning("Wrote NK correlation to %s",config['nk_file_name'])

    # Do KK correlation function if necessary
    if 'kk_file_name' in config:
        logger.warning("Performing KK calculations...")
        kk = KKCorrelation(config, logger=logger)
        kk.process(cat1,cat2)
        logger.info("Done KK calculations.")
        kk.write(config['kk_file_name'])
        logger.warning("Wrote KK correlation to %s",config['kk_file_name'])

    # Do NG correlation function if necessary
    if 'ng_file_name' in config or 'nm_file_name' in config or 'norm_file_name' in config:
        if cat2 is None:
            raise TypeError("file_name2 is required for ng correlation")
        logger.warning("Performing NG calculations...")
        ng = NGCorrelation(config, logger=logger)
        ng.process(cat1,cat2)
        logger.info("Done NG calculation.")

        # The default ng_statistic is compensated _iff_ rand files are given.
        rg = None
        if rand1 is None:
            if config.get('ng_statistic',None) == 'compensated':
                raise TypeError("rand_files is required for ng_statistic = compensated")
        elif config.get('ng_statistic','compensated') == 'compensated':
            rg = NGCorrelation(config, logger=logger)
            rg.process(rand1,cat2)
            logger.info("Done RG calculation.")

        if 'ng_file_name' in config:
            ng.write(config['ng_file_name'], rg=rg)
            logger.warning("Wrote NG correlation to %s",config['ng_file_name'])
        if 'nm_file_name' in config:
            ng.writeNMap(config['nm_file_name'], rg=rg, m2_uform=config['m2_uform'],
                         precision=config.get('precision',None))
            logger.warning("Wrote NMap values to %s",config['nm_file_name'])

        if 'norm_file_name' in config:
            gg = GGCorrelation(config, logger=logger)
            gg.process(cat2)
            logger.info("Done GG calculation for norm")
            dd = NNCorrelation(config, logger=logger)
            dd.process(cat1)
            logger.info("Done DD calculation for norm")
            rr = NNCorrelation(config, logger=logger)
            rr.process(rand1)
            logger.info("Done RR calculation for norm")
            if config['nn_statistic'] == 'compensated':
                dr = NNCorrelation(config, logger=logger)
                dr.process(cat1,rand1)
                logger.info("Done DR calculation for norm")
            else:
                dr = None
            ng.writeNorm(config['norm_file_name'],gg=gg,dd=dd,rr=rr,dr=dr,rg=rg,
                         m2_uform=config['m2_uform'], precision=config.get('precision',None))
            logger.warning("Wrote Norm values to %s",config['norm_file_name'])

    # Do KG correlation function if necessary
    if 'kg_file_name' in config:
        if cat2 is None:
            raise TypeError("file_name2 is required for kg correlation")
        logger.warning("Performing KG calculations...")
        kg = KGCorrelation(config, logger=logger)
        kg.process(cat1,cat2)
        logger.info("Done KG calculation.")
        kg.write(config['kg_file_name'])
        logger.warning("Wrote KG correlation to %s",config['kg_file_name'])

    # Do GG correlation function if necessary
    if 'gg_file_name' in config or 'm2_file_name' in config:
        logger.warning("Performing GG calculations...")
        gg = GGCorrelation(config, logger=logger)
        gg.process(cat1,cat2)
        logger.info("Done GG calculations.")
        if 'gg_file_name' in config:
            gg.write(config['gg_file_name'])
            logger.warning("Wrote GG correlation to %s",config['gg_file_name'])
        if 'm2_file_name' in config:
            gg.writeMapSq(config['m2_file_name'], m2_uform=config['m2_uform'])
            logger.warning("Wrote Mapsq values to %s",config['m2_file_name'])

    # Do NZ correlation function if necessary
    if 'nz_file_name' in config:
        if cat2 is None:
            raise TypeError("file_name2 is required for nz correlation")
        logger.warning("Performing NZ calculations...")
        nz = NZCorrelation(config, logger=logger)
        nz.process(cat1,cat2)
        logger.info("Done NZ calculation.")

        # The default nz_statistic is compensated _iff_ rand files are given.
        rz = None
        if rand1 is None:
            if config.get('nz_statistic',None) == 'compensated':
                raise TypeError("rand_files is required for nz_statistic = compensated")
        elif config.get('nz_statistic','compensated') == 'compensated':
            rz = NZCorrelation(config, logger=logger)
            rz.process(rand1,cat2)
            logger.info("Done RZ calculation.")

        nz.write(config['nz_file_name'], rz=rz)
        logger.warning("Wrote NZ correlation to %s",config['nz_file_name'])

    # Do KZ correlation function if necessary
    if 'kz_file_name' in config:
        if cat2 is None:
            raise TypeError("file_name2 is required for kz correlation")
        logger.warning("Performing KZ calculations...")
        kz = KZCorrelation(config, logger=logger)
        kz.process(cat1,cat2)
        logger.info("Done KZ calculation.")
        kz.write(config['kz_file_name'])
        logger.warning("Wrote KZ correlation to %s",config['kz_file_name'])

    # Do ZZ correlation function if necessary
    if 'zz_file_name' in config:
        logger.warning("Performing ZZ calculations...")
        zz = ZZCorrelation(config, logger=logger)
        zz.process(cat1,cat2)
        logger.info("Done ZZ calculations.")
        zz.write(config['zz_file_name'])
        logger.warning("Wrote ZZ correlation to %s",config['zz_file_name'])

    # Do NV correlation function if necessary
    if 'nv_file_name' in config:
        if cat2 is None:
            raise TypeError("file_name2 is required for nv correlation")
        logger.warning("Performing NV calculations...")
        nv = NVCorrelation(config, logger=logger)
        nv.process(cat1,cat2)
        logger.info("Done NV calculation.")

        # The default nv_statistic is compensated _iff_ rand files are given.
        rv = None
        if rand1 is None:
            if config.get('nv_statistic',None) == 'compensated':
                raise TypeError("rand_files is required for nv_statistic = compensated")
        elif config.get('nv_statistic','compensated') == 'compensated':
            rv = NVCorrelation(config, logger=logger)
            rv.process(rand1,cat2)
            logger.info("Done RV calculation.")

        nv.write(config['nv_file_name'], rv=rv)
        logger.warning("Wrote NV correlation to %s",config['nv_file_name'])

    # Do KV correlation function if necessary
    if 'kv_file_name' in config:
        if cat2 is None:
            raise TypeError("file_name2 is required for kv correlation")
        logger.warning("Performing KV calculations...")
        kv = KVCorrelation(config, logger=logger)
        kv.process(cat1,cat2)
        logger.info("Done KV calculation.")
        kv.write(config['kv_file_name'])
        logger.warning("Wrote KV correlation to %s",config['kv_file_name'])

    # Do VV correlation function if necessary
    if 'vv_file_name' in config:
        logger.warning("Performing VV calculations...")
        vv = VVCorrelation(config, logger=logger)
        vv.process(cat1,cat2)
        logger.info("Done VV calculations.")
        vv.write(config['vv_file_name'])
        logger.warning("Wrote VV correlation to %s",config['vv_file_name'])

    # Do NT correlation function if necessary
    if 'nt_file_name' in config:
        if cat2 is None:
            raise TypeError("file_name2 is required for nt correlation")
        logger.warning("Performing NT calculations...")
        nt = NTCorrelation(config, logger=logger)
        nt.process(cat1,cat2)
        logger.info("Done NT calculation.")

        # The default nt_statistic is compensated _iff_ rand files are given.
        rt = None
        if rand1 is None:
            if config.get('nt_statistic',None) == 'compensated':
                raise TypeError("rand_files is required for nt_statistic = compensated")
        elif config.get('nt_statistic','compensated') == 'compensated':
            rt = NTCorrelation(config, logger=logger)
            rt.process(rand1,cat2)
            logger.info("Done RT calculation.")

        nt.write(config['nt_file_name'], rt=rt)
        logger.warning("Wrote NT correlation to %s",config['nt_file_name'])

    # Do KT correlation function if necessary
    if 'kt_file_name' in config:
        if cat2 is None:
            raise TypeError("file_name2 is required for kt correlation")
        logger.warning("Performing KT calculations...")
        kt = KTCorrelation(config, logger=logger)
        kt.process(cat1,cat2)
        logger.info("Done KT calculation.")
        kt.write(config['kt_file_name'])
        logger.warning("Wrote KT correlation to %s",config['kt_file_name'])

    # Do TT correlation function if necessary
    if 'tt_file_name' in config:
        logger.warning("Performing TT calculations...")
        tt = TTCorrelation(config, logger=logger)
        tt.process(cat1,cat2)
        logger.info("Done TT calculations.")
        tt.write(config['tt_file_name'])
        logger.warning("Wrote TT correlation to %s",config['tt_file_name'])

    # Do NQ correlation function if necessary
    if 'nq_file_name' in config:
        if cat2 is None:
            raise TypeError("file_name2 is required for nq correlation")
        logger.warning("Performing NQ calculations...")
        nq = NQCorrelation(config, logger=logger)
        nq.process(cat1,cat2)
        logger.info("Done NQ calculation.")

        # The default nq_statistic is compensated _iff_ rand files are given.
        rq = None
        if rand1 is None:
            if config.get('nq_statistic',None) == 'compensated':
                raise TypeError("rand_files is required for nq_statistic = compensated")
        elif config.get('nq_statistic','compensated') == 'compensated':
            rq = NQCorrelation(config, logger=logger)
            rq.process(rand1,cat2)
            logger.info("Done RQ calculation.")

        nq.write(config['nq_file_name'], rq=rq)
        logger.warning("Wrote NQ correlation to %s",config['nq_file_name'])

    # Do KQ correlation function if necessary
    if 'kq_file_name' in config:
        if cat2 is None:
            raise TypeError("file_name2 is required for kq correlation")
        logger.warning("Performing KQ calculations...")
        kq = KQCorrelation(config, logger=logger)
        kq.process(cat1,cat2)
        logger.info("Done KQ calculation.")
        kq.write(config['kq_file_name'])
        logger.warning("Wrote KQ correlation to %s",config['kq_file_name'])

    # Do QQ correlation function if necessary
    if 'qq_file_name' in config:
        logger.warning("Performing QQ calculations...")
        qq = QQCorrelation(config, logger=logger)
        qq.process(cat1,cat2)
        logger.info("Done QQ calculations.")
        qq.write(config['qq_file_name'])
        logger.warning("Wrote QQ correlation to %s",config['qq_file_name'])


def print_corr2_params():
    """Print information about the valid parameters that may be given to the `corr2` function.
    """
    print_params(corr2_valid_params)
