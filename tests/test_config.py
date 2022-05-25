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

import treecorr
import os
import sys
import logging
import numpy as np
from unittest import mock

from test_helper import CaptureLog, assert_raises, timer, assert_warns

@timer
def test_parse_variables():
    """Test parse_variables functionality
    """
    config = treecorr.read_config('configs/nn.yaml')

    # parse_variables is used by corr2 executable to add or change items in config
    # with extra command line arguments
    assert 'file_name2' not in config
    treecorr.config.parse_variable(config, 'file_name2 = data/nn_data.dat')
    assert config['file_name2'] == 'data/nn_data.dat'

    treecorr.config.parse_variable(config, 'file_name2=data/nn_data2.dat')
    assert config['file_name2'] == 'data/nn_data2.dat'

    # It's also used by params parsing, so removes trailing comments
    treecorr.config.parse_variable(config, 'file_name2=data/nn_data3.dat # The second file')
    assert config['file_name2'] == 'data/nn_data3.dat'

    # Extra whitespace is ignored
    treecorr.config.parse_variable(config, 'file_name2 = \t\tdata/nn_data4.dat       ')
    assert config['file_name2'] == 'data/nn_data4.dat'

    # Invalid if no = sign.
    with assert_raises(ValueError):
        treecorr.config.parse_variable(config, 'file_name2:data/nn_data2.dat')

    # Can specify lists with [], () or {}
    treecorr.config.parse_variable(config, 'file_name2 = [f1, f2, f3]')
    assert config['file_name2'] == ['f1', 'f2', 'f3']
    treecorr.config.parse_variable(config, 'file_name2 = (   g1\t, g2\t, g3\t)')
    assert config['file_name2'] == ['g1', 'g2', 'g3']
    treecorr.config.parse_variable(config, 'file_name2 = {h1,h2,h3}')
    assert config['file_name2'] == ['h1', 'h2', 'h3']

    # In config file, can also separate by whitespace
    treecorr.config.parse_variable(config, 'file_name2 = f1   g2  h3')
    assert config['file_name2'] == ['f1', 'g2', 'h3']

    # If starts with [, needs trailing ] or error.
    with assert_raises(ValueError):
        treecorr.config.parse_variable(config, 'file_name2 = [h1, h2, h3')

@timer
def test_parse_bool():
    """Test parse_bool functionality
    """
    # Booleans have a number of possible specifications
    assert treecorr.config.parse_bool('True') is True
    assert treecorr.config.parse_bool(True) is True
    assert treecorr.config.parse_bool(1) == 1
    assert treecorr.config.parse_bool('yes') is True
    assert treecorr.config.parse_bool('T') is True
    assert treecorr.config.parse_bool('y') is True
    assert treecorr.config.parse_bool('1') == 1
    assert treecorr.config.parse_bool('10') == 10

    assert treecorr.config.parse_bool('False') is False
    assert treecorr.config.parse_bool(False) is False
    assert treecorr.config.parse_bool(0) == 0
    assert treecorr.config.parse_bool('no') is False
    assert treecorr.config.parse_bool('F') is False
    assert treecorr.config.parse_bool('n') is False
    assert treecorr.config.parse_bool('0') == 0

    with assert_raises(ValueError):
        treecorr.config.parse_bool('G')
    with assert_raises(ValueError):
        treecorr.config.parse_bool(13.8)
    with assert_raises(ValueError):
        treecorr.config.parse_bool('13.8')
    with assert_raises(ValueError):
        treecorr.config.parse_bool('Hello')

@timer
def test_parse_unit():
    """Test parse_unit functionality
    """
    assert np.isclose(treecorr.config.parse_unit('radian'), 1.)
    assert np.isclose(treecorr.config.parse_unit('deg'), np.pi / 180.)
    assert np.isclose(treecorr.config.parse_unit('degree'), np.pi / 180.)
    assert np.isclose(treecorr.config.parse_unit('degrees'), np.pi / 180.)
    assert np.isclose(treecorr.config.parse_unit('arcmin'), np.pi / 180. / 60)
    assert np.isclose(treecorr.config.parse_unit('arcminutes'), np.pi / 180. / 60)
    assert np.isclose(treecorr.config.parse_unit('arcsec'), np.pi / 180. / 60 / 60)
    assert np.isclose(treecorr.config.parse_unit('arcseconds'), np.pi / 180. / 60 / 60)

    with assert_raises(ValueError):
        treecorr.config.parse_unit('gradians')
    with assert_raises(ValueError):
        treecorr.config.parse_unit('miles')
    with assert_raises(ValueError):
        treecorr.config.parse_unit('Mpc')


@timer
def test_read():
    """Test different ways of reading a config file.
    """
    # The config files for nn_list are designed to hit all the major features here.
    # Tests that use these config files are in test_nn.py:test_list()

    config1 = treecorr.config.read_config('configs/nn_list1.yaml')
    assert config1 == {
        'file_list': 'data/nn_list_data_files.txt',
        'rand_file_list': 'data/nn_list_rand_files.txt',
        'x_col': 1,
        'y_col': 2,
        'verbose': 1,
        'min_sep': 1.,
        'max_sep': 25.,
        'bin_size': 0.10,
        'nn_file_name': 'output/nn_list1.out',
        'nn_statistic': 'simple',
    }

    config2 = treecorr.config.read_config('configs/nn_list2.json')
    assert config2 == {
        'file_list': 'data/nn_list_data_files.txt',
        'rand_file_name': 'data/nn_list_randx.dat',
        'x_col': 1,
        'y_col': 2,
        'verbose': 1,
        'min_sep': 1.,
        'max_sep': 25.,
        'bin_size': 0.10,
        'nn_file_name': 'output/nn_list2.out',
        'nn_statistic': 'simple',
    }

    config3 = treecorr.config.read_config('configs/nn_list3.params')
    assert config3 == {
        'file_name': 'data/nn_list_datax.dat',
        'rand_file_name': ['data/nn_list_rand0.dat', 'data/nn_list_rand1.dat',
                           'data/nn_list_rand2.dat'],
        'x_col': '1',
        'y_col': '2',
        'verbose': '1',
        'min_sep': '1.',
        'max_sep': '25.',
        'bin_size': '0.10',
        'nn_file_name': 'output/nn_list3.out',
        'nn_statistic': 'simple',
    }

    config4 = treecorr.config.read_config('configs/nn_list4.config', file_type='yaml')
    assert config4 == {
        'file_list': 'data/nn_list_data_files.txt',
        'rand_file_list': 'data/nn_list_rand_files.txt',
        'file_list2': 'data/nn_list_data_files.txt',
        'rand_file_name2': 'data/nn_list_randx.dat',
        'x_col': 1,
        'y_col': 2,
        'verbose': 1,
        'min_sep': 1.,
        'max_sep': 25.,
        'bin_size': 0.10,
        'nn_file_name': 'output/nn_list4.out',
        'nn_statistic': 'simple',
    }

    config5 = treecorr.config.read_config('configs/nn_list5.config', file_type='json')
    assert config5 == {
        'file_list': 'data/nn_list_data_files.txt',
        'rand_file_name': 'data/nn_list_randx.dat',
        'file_name2': 'data/nn_list_datax.dat',
        'rand_file_list2': 'data/nn_list_rand_files.txt',
        'x_col': 1,
        'y_col': 2,
        'verbose': 1,
        'min_sep': 1.,
        'max_sep': 25.,
        'bin_size': 0.10,
        'nn_file_name': 'output/nn_list5.out',
        'nn_statistic': 'simple',
    }

    config6 = treecorr.config.read_config('configs/nn_list6.config', file_type='params')
    assert config6 == {
        'file_name': ['data/nn_list_data0.dat', 'data/nn_list_data1.dat', 'data/nn_list_data2.dat'],
        'rand_file_name': ['data/nn_list_rand0.dat', 'data/nn_list_rand1.dat',
                           'data/nn_list_rand2.dat'],
        'file_list2': 'data/nn_list_data_files.txt',
        'rand_file_list2': 'data/nn_list_rand_files.txt',
        'x_col': '1',
        'y_col': '2',
        'verbose': '1',
        'min_sep': '1.',
        'max_sep': '25.',
        'bin_size': '0.10',
        'nn_file_name': 'nn_list6.out',
        'nn_statistic': 'simple',
    }

    with assert_raises(ValueError):
        treecorr.config.read_config('configs/nn_list6.config', file_type='simple')
    with assert_raises(ValueError):
        treecorr.config.read_config('configs/nn_list6.config')


@timer
def test_logger():
    """Test setting up a logger.
    """
    logger1 = treecorr.config.setup_logger(verbose=0)
    assert logger1.level == logging.CRITICAL
    assert logger1.name == 'treecorr'
    assert len(logger1.handlers) == 1
    assert isinstance(logger1.handlers[0], logging.StreamHandler)

    logger2 = treecorr.config.setup_logger(verbose=3)
    assert logger2.level == logging.DEBUG
    assert logger2.name == 'treecorr'
    assert len(logger2.handlers) == 1
    assert isinstance(logger2.handlers[0], logging.StreamHandler)

    logger3 = treecorr.config.setup_logger(verbose=2, log_file='output/test_logger.out')
    assert logger3.level == logging.INFO
    assert logger3.name == 'treecorr_output/test_logger.out'
    assert len(logger3.handlers) == 1
    assert isinstance(logger3.handlers[0], logging.FileHandler)
    assert logger3.handlers[0].baseFilename == os.path.abspath('output/test_logger.out')

    logger4 = treecorr.config.setup_logger(verbose=1, log_file='output/test_logger.out')
    assert logger4.level == logging.WARNING
    assert logger4.name == 'treecorr_output/test_logger.out'
    assert len(logger4.handlers) == 1
    assert isinstance(logger4.handlers[0], logging.FileHandler)
    assert logger4.handlers[0].baseFilename == os.path.abspath('output/test_logger.out')

    logger5 = treecorr.config.setup_logger(verbose=1, log_file='output/test_logger2.out')
    assert logger5.level == logging.WARNING
    assert logger5.name == 'treecorr_output/test_logger2.out'
    assert len(logger5.handlers) == 1
    assert isinstance(logger5.handlers[0], logging.FileHandler)
    assert logger5.handlers[0].baseFilename == os.path.abspath('output/test_logger2.out')

    logger6 = treecorr.config.setup_logger(verbose=1, log_file=None)
    assert logger6.level == logging.WARNING
    assert logger6.name == 'treecorr'
    assert len(logger6.handlers) == 1
    assert isinstance(logger6.handlers[0], logging.StreamHandler)


@timer
def test_check():
    """Test checking the validity of config values.
    """
    # First a simple case with no conflicts
    config1 = treecorr.read_config('configs/kg.yaml')
    valid_params = treecorr.corr2_valid_params
    config2 = treecorr.config.check_config(config1.copy(), valid_params)

    # Just check a few values
    assert config2['x_col'] == '1'
    assert config2['k_col'] == ['3', '0']
    assert config2['verbose'] == 1
    assert config2['kg_file_name'] == 'output/kg.out'

    config3 = treecorr.config.check_config({'g1_ext': [3, 'g1']}, valid_params)
    assert config3['g1_ext'] == ['3', 'g1']

    # Will also have other parameters filled from the valid_params dict
    for key in config2:
        assert key in valid_params
        if key in config1:
            if isinstance(config1[key], list):
                assert [str(v) for v in config2[key]] == [str(v) for v in config1[key]]
            else:
                assert config2[key] == config1[key] or str(config2[key]) == str(config1[key])
        else:
            assert config2[key] == valid_params[key][2]

    # Check list of bool
    config1['flip_g1'] = [True, 0]
    config2 = treecorr.config.check_config(config1.copy(), valid_params)
    assert config2['flip_g1'] == [True, False]

    # Longer names are allowed
    config1['x_units'] = 'arcminutes'
    config1['y_units'] = 'arcminute'
    config2 = treecorr.config.check_config(config1.copy(), valid_params)
    assert config2['x_units'] == 'arcmin'
    assert config2['y_units'] == 'arcmin'

    # Also other aliases, but you need to list them explicitly.
    config1['reverse_g1'] = True
    with assert_raises(TypeError):
        treecorr.config.check_config(config1.copy(), valid_params)
    with assert_warns(FutureWarning):
        config2 = treecorr.config.check_config(config1.copy(), valid_params,
                                               aliases={'reverse_g1' : 'flip_g1'})
    assert config2['flip_g1'] is True
    assert 'reverse_g1' not in config2
    del config1['reverse_g1']

    # Invalid values raise errors
    config1['verbose'] = -1
    with assert_raises(ValueError):
        treecorr.config.check_config(config1.copy(), valid_params)
    config1['verbose'] = 1
    config1['metric'] = 'hyperbolic'
    with assert_raises(ValueError):
        treecorr.config.check_config(config1.copy(), valid_params)
    del config1['metric']

    # With a logger, aliases write the warning to the logger.
    config1['n2_file_name'] = 'output/n2.out'
    with CaptureLog() as cl:
        config2 = treecorr.config.check_config(config1.copy(), valid_params, logger=cl.logger,
                                               aliases={'n2_file_name' : 'nn_file_name'})
    assert "The parameter n2_file_name is deprecated." in cl.output
    assert "You should use nn_file_name instead." in cl.output

    # corr2 has a list of standard aliases
    # It is currently empty, but let's mock it up to test the functionality.
    with mock.patch('treecorr.corr2_aliases', {'n2_file_name' : 'nn_file_name'}):
        with assert_warns(FutureWarning):
            config2 = treecorr.config.check_config(config1.copy(), valid_params,
                                                   aliases=treecorr.corr2_aliases)
    assert 'n2_file_name' not in config2
    assert config2['nn_file_name'] == 'output/n2.out'
    del config1['n2_file_name']


@timer
def test_print():
    """Test print_params.
    """
    # This really just checks that the functions do something.
    # It doesn't check the output for accuracy.
    treecorr.print_corr2_params()
    treecorr.print_corr3_params()


@timer
def test_get():
    """Test getting a parameter from a config dict
    """
    config1 = treecorr.read_config('configs/kg.yaml')
    assert treecorr.config.get(config1, 'x_col', int) == 1
    assert treecorr.config.get(config1, 'x_col', str) == '1'
    assert treecorr.config.get(config1, 'x_col') == '1'
    assert treecorr.config.get(config1, 'x_col', int, 2) == 1
    assert treecorr.config.get(config1, 'ra_col', int) is None
    assert treecorr.config.get(config1, 'ra_col', int, 2) == 2

    config1['flip_g1'] = True
    assert treecorr.config.get(config1, 'flip_g1', bool) is True
    assert treecorr.config.get(config1, 'flip_g1', bool, False) is True
    assert treecorr.config.get(config1, 'flip_g2', bool, False) is False
    assert treecorr.config.get(config1, 'flip_g2', bool) is None

    config1['flip_g2'] = None
    assert treecorr.config.get(config1, 'flip_g1', bool) is True
    assert treecorr.config.get(config1, 'flip_g2', bool) is None
    assert treecorr.config.get(config1, 'flip_g2', bool, True) is True
    assert treecorr.config.get(config1, 'flip_g2', bool, False) is False

    assert treecorr.config.get_from_list(config1, 'k_col', 0, int) == 3
    assert treecorr.config.get_from_list(config1, 'k_col', 0, str) == '3'
    assert treecorr.config.get_from_list(config1, 'k_col', 0) == '3'
    assert treecorr.config.get_from_list(config1, 'k_col', 0, int, 2) == 3
    assert treecorr.config.get_from_list(config1, 'k_col', 1, int) == 0
    assert treecorr.config.get_from_list(config1, 'k_col', 1, int, 2) == 0
    assert treecorr.config.get_from_list(config1, 'ra_col', 1, int, 2) == 2
    assert treecorr.config.get_from_list(config1, 'ra_col', 1, int) is None

    config1['flip_g1'] = [True, False, None]
    assert treecorr.config.get_from_list(config1, 'flip_g1', 0, bool) is True
    assert treecorr.config.get_from_list(config1, 'flip_g1', 1, bool) is False
    assert treecorr.config.get_from_list(config1, 'flip_g1', 2, bool) is None
    assert treecorr.config.get_from_list(config1, 'flip_g1', 0, bool, False) is True
    assert treecorr.config.get_from_list(config1, 'flip_g1', 1, bool, True) is False
    assert treecorr.config.get_from_list(config1, 'flip_g1', 2, bool, True) is True
    assert treecorr.config.get_from_list(config1, 'flip_g1', 2, bool, False) is False
    assert treecorr.config.get_from_list(config1, 'flip_g2', 1, bool) is None
    assert treecorr.config.get_from_list(config1, 'flip_g2', 1, bool, False) is False
    assert treecorr.config.get_from_list(config1, 'flip_g2', 2, bool, False) is False
    assert treecorr.config.get_from_list(config1, 'flip_g2', 3, bool, True) is True

    with assert_raises(IndexError):
        treecorr.config.get_from_list(config1, 'k_col', 2, int)
    with assert_raises(IndexError):
        treecorr.config.get_from_list(config1, 'flip_g1', 3, bool)
    with assert_raises(IndexError):
        treecorr.config.get_from_list(config1, 'flip_g1', 3, bool, False)


@timer
def test_merge():
    """Test merging two config dicts.
    """
    # First a simple case with no conflicts
    config1 = treecorr.read_config('Aardvark.yaml')
    kwargs = { 'cat_precision' : 10 }
    valid_params = treecorr.Catalog._valid_params
    config2 = treecorr.config.merge_config(config1, kwargs, valid_params)

    assert config2['cat_precision'] == 10
    assert config2['ra_col'] == 'RA'
    assert config2['verbose'] == 2

    # config is allowed to have invalid parameters
    assert 'gg_file_name' in config1
    assert 'gg_file_name' not in config2

    # If either is None, then return subset of other that is valid
    config2 = treecorr.config.merge_config(config1.copy(), None, valid_params)
    for key in config2:
        assert key in valid_params
        if key in config1:
            assert config2[key] == config1[key] or config2[key] in config1[key]
        else:
            assert config2[key] == valid_params[key][2]
    assert 'gg_file_name' not in config2

    config2 = treecorr.config.merge_config(None, kwargs.copy(), valid_params)
    for key in config2:
        assert key in valid_params
        if key in kwargs:
            assert config2[key] == kwargs[key] or config2[key] in kwargs[key]
        else:
            assert config2[key] == valid_params[key][2]

    # If conflicts, kwargs takes precedence ...
    kwargs['ra_col'] = 'alpha2000'
    config2 = treecorr.config.merge_config(config1, kwargs, treecorr.Catalog._valid_params)
    assert config2['ra_col'] == 'alpha2000'

    # ... even when when the value in kwargs is None.
    kwargs = {'bin_size': 0.06, 'nbins': None}
    config2 = treecorr.config.merge_config(config1, kwargs, treecorr.BinnedCorr2._valid_params)
    assert config2['bin_size'] == 0.06
    assert config2['min_sep'] == config1['min_sep']
    assert config2['max_sep'] == config1['max_sep']
    assert kwargs['nbins'] is None
    assert config2['nbins'] is None

    # If kwargs has invalid parameters, exception is raised
    kwargs = { 'cat_prec' : 10 }
    with assert_raises(TypeError):
        treecorr.config.merge_config(config1, kwargs, treecorr.Catalog._valid_params)


@timer
def test_convert():
    """Test converting a value to a given type.
    """
    assert treecorr.config.convert('10', int, 'n_bins') == 10
    assert treecorr.config.convert('yes', bool, 'flip_g1') is True
    assert treecorr.config.convert('0.9', float, 'bin_slop') == 0.9
    assert treecorr.config.convert(None, str, 'x_units') is None
    assert np.isclose(treecorr.config.convert('deg', float, 'ra_units'), np.pi / 180.)


@timer
def test_omp():
    """Test setting the number of omp threads.
    """
    import multiprocessing

    # If num_threads <= 0 or None, get num from cpu_count
    cpus = multiprocessing.cpu_count()
    assert treecorr.set_omp_threads(0) > 0
    assert treecorr.set_omp_threads(0) <= cpus
    assert treecorr.set_omp_threads(None) > 0
    assert treecorr.set_omp_threads(None) <= cpus

    # If num_threads == 1, it should always set to 1
    assert treecorr.set_omp_threads(1) == 1

    # If num_threads > 1, it could be 1 or up to the input num_threads
    assert treecorr.set_omp_threads(2) >= 1
    assert treecorr.set_omp_threads(2) <= 2
    assert treecorr.set_omp_threads(20) >= 1
    assert treecorr.set_omp_threads(20) <= 20

    # Repeat and check that appropriate messages are emitted
    with CaptureLog() as cl:
        num_threads = treecorr.set_omp_threads(0, logger=cl.logger)
    assert "multiprocessing.cpu_count() = " in cl.output
    assert "Telling OpenMP to use %s threads"%cpus in cl.output

    with CaptureLog() as cl:
        treecorr.set_omp_threads(None, logger=cl.logger)
    assert "multiprocessing.cpu_count() = " in cl.output
    assert "Telling OpenMP to use %s threads"%cpus in cl.output

    with CaptureLog() as cl:
        treecorr.set_omp_threads(1, logger=cl.logger)
    assert "multiprocessing.cpu_count() = " not in cl.output
    assert "Telling OpenMP to use 1 threads" in cl.output
    assert "Using %s threads"%num_threads not in cl.output
    assert "Unable to use multiple threads" not in cl.output

    with CaptureLog() as cl:
        treecorr.set_omp_threads(2, logger=cl.logger)
    assert "multiprocessing.cpu_count() = " not in cl.output
    assert "Telling OpenMP to use 2 threads" in cl.output

    # It's hard to tell what happens in the next step, since we can't control what
    # treecorr._lib.SetOMPThreads does.  It depends on whether OpenMP is enabled and
    # how many cores are available.  So let's mock it up.
    with mock.patch('treecorr.util._lib') as _lib:
        # First mock with OpenMP enables and able to use lots of threads
        _lib.SetOMPThreads = lambda x: x
        assert treecorr.set_omp_threads(20) == 20
        with CaptureLog() as cl:
            treecorr.set_omp_threads(20, logger=cl.logger)
        assert "OpenMP reports that it will use 20 threads" in cl.output
        assert "Using 20 threads" in cl.output

        # Next only 4 threads available
        _lib.SetOMPThreads = lambda x: 4 if x > 4 else x
        assert treecorr.set_omp_threads(20) == 4
        with CaptureLog() as cl:
            treecorr.set_omp_threads(20, logger=cl.logger)
        assert "OpenMP reports that it will use 4 threads" in cl.output
        assert "Using 4 threads" in cl.output

        assert treecorr.set_omp_threads(2) == 2
        with CaptureLog() as cl:
            treecorr.set_omp_threads(2, logger=cl.logger)
        assert "OpenMP reports that it will use 2 threads" in cl.output

        # Finally, no OpenMP
        _lib.SetOMPThreads = lambda x: 1
        assert treecorr.set_omp_threads(20) == 1
        with CaptureLog() as cl:
            treecorr.set_omp_threads(20, logger=cl.logger)
        assert "OpenMP reports that it will use 1 threads" in cl.output
        assert "Unable to use multiple threads" in cl.output

@timer
def test_util():
    # Test some error handling in utility functions that shouldn't be possible to get to
    # in normal running, so we need to call things explicitly to get the coverage.
    with assert_raises(ValueError):
        treecorr.util.parse_metric('Euclidean', 'invalid')
    with assert_raises(ValueError):
        treecorr.util.parse_metric('Invalid', 'flat')
    with assert_raises(ValueError):
        treecorr.util.coord_enum('invalid')
    with assert_raises(ValueError):
        treecorr.util.metric_enum('Invalid')

if __name__ == '__main__':
    test_parse_variables()
    test_parse_bool()
    test_parse_unit()
    test_read()
    test_logger()
    test_check()
    test_print()
    test_get()
    test_merge()
    test_convert()
    test_omp()
    test_util()
    test_gen_read_write()
    test_gen_multi_read_write()
