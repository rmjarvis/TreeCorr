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
.. module:: config
"""

from __future__ import print_function
import treecorr
import six


def parse_variable(config, v):
    """Parse a configuration variable from a string that should look like 'key = value'
    and write that value to config[key].

    :param config:  The configuration dict to wich to write the key,value pair
    :param v:       A string of the form 'key = value'
    """
    if '=' not in v:
        raise ValueError('Improper variable specificationi: %s.  Use syntax: key = value.'%v)
    key, value = v.split('=',1)
    key = key.strip()
    # Cut off any trailing comment
    if '#' in value: value = value.split('#')[0]
    value = value.strip()
    if value[0] == '{':
        values = value[1:-1].split(',')
    else:
        values = value.split() # on whitespace
        if len(values) == 1:
            config[key] = value
        else:
            config[key] = values


def parse_bool(value):
    """Parse a value as a boolean.

    Valid string values for True are: 'true', 'yes', 't', 'y'
    Valid string values for False are: 'false', 'no', 'f', 'n', 'none'
    Capitalization is ignored.

    If value is a number, it is converted to a bool in the usual way.

    :param value:   The value to parse.

    :returns:       The value converted to a bool.
    """
    if isinstance(value,str):
        if value.strip().upper() in [ 'TRUE', 'YES', 'T', 'Y' ]:
            return True
        elif value.strip().upper() in [ 'FALSE', 'NO', 'F', 'N', 'NONE' ]:
            return False
        else:
            try:
                val = bool(int(value))
                return val
            except:
                raise ValueError("Unable to parse %s as a bool."%value)
    else:
        try:
            val = bool(value)
            return val
        except:
            raise ValueError("Unable to parse %s as a bool."%value)

def parse_unit(value):
    """Parse the input value as a string that should be one of our valid angle units in
    the treecorr.angle_units dict.

    The value is allowed to merely start with one of the unit names.  So 'deg', 'degree',
    'degrees' all convert to 'deg' which is the key in the angle_units dict.
    The return value in this case would be treecorr.angle_units['deg'], which has the 
    value pi/180.

    :param value:   The unit as a string value to parse.

    :returns:       The given unit in radians.
    """
    for unit in treecorr.angle_units.keys():
        if value.startswith(unit): return treecorr.angle_units[unit]
    raise ValueError("Unable to parse %s as an angle unit"%value)


def read_config(file_name):
    """Read a configuration dict from a file.

    :param file_name:   The file name from which the configuration dict should be read.
    """
    config = dict()
    with open(file_name) as fin:
        for v in fin:
            v = v.strip()
            if len(v) == 0 or v[0] == '#':
                pass
            elif v[0] == '+':
                include_file_name = v[1:]
                read_config(include_file_name)
            else:
                parse_variable(config,v)
    return config


def setup_logger(verbose, log_file=None):
    """Parse the integer verbosity level from the command line args into a logging_level string

    Note: This will update the verbosity if a previous call to setup_logger used a different
    value for verbose.  However, it will not update the handler to use a different log_file
    or switch between using a log_file and stdout.

    :param verbose:     An integer indicating what verbosity level to use.
    :param log_file:    If given, a file name to which to write the logging output.
                        If omitted or None, then output to stdout.

    :returns:           The logging.Logger object to use.
    """
    import logging
    logging_levels = {  0: logging.CRITICAL,
                        1: logging.WARNING,
                        2: logging.INFO,
                        3: logging.DEBUG }
    logging_level = logging_levels[verbose]

    # Setup logging to go to sys.stdout or (if requested) to an output file
    logger = logging.getLogger('treecorr')
    if len(logger.handlers) == 0:  # only add handler once!
        if log_file is None:
            handle = logging.StreamHandler()
        else:
            handle = logging.FileHandler(log_file)
        formatter = logging.Formatter('%(message)s')  # Simple text output
        handle.setFormatter(formatter)
        logger.addHandler(handle)
    logger.setLevel(logging_level)
    return logger

 
def check_config(config, params, aliases=None, logger=None):
    """Check (and update) a config dict to conform to the given parameter rules.
    The params dict has an entry for each valid config parameter whose value is a tuple
    with the following items:

    - type
    - can be a list?
    - default value
    - valid values
    - description (Multiple entries here are allowed for longer strings)

    The file corr2.py has a list of parameters for the corr2 program.

    :param config:  The config dict to check.
    :param params:  A dict of valid parameters with information about each one.
    :param aliases: A dict of deprecated parameters that are still aliases for new names.
                    (default: None)
    :param logger:  If desired, a logger object for logging any warnings here. (default: None)

    :returns:       The updated config dict.  The input config may be modified by this function.
    """
    for key in config:
        # Check if this is a deprecated alias
        if aliases and key in aliases:
            if logger:
                logger.warn("The parameter %s is deprecated.  You should use %s instead.",
                            key, aliases[key])
            new_key = aliases[key]
            config[new_key] = config[key]
            del config[key]
            key = new_key

        # Check that this is a valid key
        if key not in params:
            raise AttributeError("Invalid parameter %s."%key)

        value_type, may_be_list, default_value, valid_values = params[key][:4]

        # Get the value
        if value_type is bool:
            value = parse_bool(config[key])
        else:
            value = value_type(config[key])

        # If limited allowed values, check that this is one of them.
        if valid_values is not None:
            if value_type is str:
                # Allow the string to be longer.  e.g. degrees is valid if 'deg' is in valid_values.
                matches = [ v for v in valid_values if value.startswith(v) ]
                if len(matches) != 1:
                    raise ValueError("Parameter %s has invalid value %s.  Valid values are %s."%(
                        key, config[key], str(valid_values)))
                value = matches[0]
            else:
                if value not in valid_values:
                    raise ValueError("Parameter %s has invalid value %s.  Valid values are %s."%(
                        key, config[key], str(valid_values)))

        # Write it back to the dict with the right type
        config[key] = value

    # Write the defaults for other parameters to simplify the syntax of getting the values
    for key in params:
        if key in config: continue
        value_type, may_be_list, default_value, valid_values = params[key][:4]
        if default_value is not None:
            config[key] = default_value

    return config


def print_params(params):
    """Print the information about the valid parameters, given by the given params dict.
    See check_config for the structure of the params dict.

    :param params:  A dict of valid parameters with information about each one.
    """
    max_len = max(len(key) for key in params)
    for key in params:
        value_type, may_be_list, default_value, valid_values = params[key][:4]
        description = params[key][4:]
        print(("{0:<"+str(max_len)+"} {1}").format(key,description[0]))
        for d in description[1:]:
            print("                {0}".format(d))

        # str(value_type) looks like "<type 'float'>"
        # value_type.__name__ looks like 'float'
        if may_be_list:
            print("                Type must be {0} or a list of {0}.".format(value_type.__name__))
        else:
            print("                Type must be {0}.".format(value_type.__name__))

        if valid_values is not None:
            print("                Valid values are {0!s}".format(valid_values))
        if default_value is not None:
            print("                Default value is {0!s}".format(default_value))
        print()


def convert(value, value_type, key):
    """Convert the given value to the given type.
    
    The key helps determine what kind of conversion should be performed.
    Specifically if 'unit' is in the key value, then a unit conversion is done.
    Otherwise, it just parses 

    :param value:       The input value to be converted.  Usually a string.
    :param value_type:  The type to convert to.
    :param key:         The key for this value.  Only used to see if it includes 'unit'.

    :returns:           The converted value.
    """
    if 'unit' in key:
        return parse_unit(value)
    elif value_type == bool:
        return parse_bool(value)
    else:
        return value_type(value)

def get_from_list(config, key, num, value_type=str, default=None):
    """A helper function to get a key from config that is allowed to be a list

    Some of the config values are allowed to be lists of values, in which case we take the 
    `num` item from the list.  If they are not a list, then the given value is used for 
    all values of `num`.

    :param config:      The configuration dict from which to get the key value.
    :param key:         What key to get from config.
    :param num:         Which number element to use if the item is a list.
    :param value_type:  What type should the value be converted to. (default: str)
    :param default:     What value should be used if the key is not in the config dict.
                        (default: None)

    :returns:           The specified value, converted as needed.
    """
    if key in config:
        values = config[key]
        if isinstance(values, list):
            if num > len(values):
                raise ValueError("Not enough values in list for %s"%key)
            return convert(values[num],value_type,key)
        elif isinstance(values, str) and values[0] == '[' and values[-1] == ']':
            values = eval(values)
            if num > len(values):
                raise ValueError("Not enough values in list for %s"%key)
            return convert(values[num],value_type,key)
        else:
            return convert(values,value_type,key)
    elif default is not None:
        return convert(default,value_type,key)
    else:
        return default

def get(config, key, value_type=str, default=None):
    """A helper function to get a key from config converting to a particular type

    :param config:      The configuration dict from which to get the key value.
    :param key:         Which key to get from config.
    :param value_type:  Which type should the value be converted to. (default: str)
    :param default:     What value should be used if the key is not in the config dict.
                        (default: None)

    :returns:           The specified value, converted as needed.
    """
    if key in config:
        value = config[key]
        return convert(value, value_type, key)
    elif default is not None:
        return convert(default,value_type,key)
    else:
        return default

def merge_config(config, kwargs, valid_params):
    """Merge in the values from kwargs into config.

    If either of these is None, then the other one is returned.
    If they are both dicts, then the values in kwargs take precedence over ones in config
    if there are any keys that are in both.  Also, the kwargs dict will be modified in this case.

    :param config:          The root config (will not be modified)
    :param kwargs:          A second dict with more or updated values
    :param valid_params:    A dict of valid parameters that are allowed for this usage.
                            The config dict is allowed to have extra items, but kwargs is not.

    :returns:               The merged dict, including only items that are in valid_params.
    """
    if kwargs is None:
        kwargs = {}
    if config:
        for key, value in six.iteritems(config):
            if key in valid_params and key not in kwargs:
                kwargs[key] = value
    check_config(kwargs, valid_params)
    return kwargs


import os
import numpy
import ctypes
_treecorr = numpy.ctypeslib.load_library('_treecorr',os.path.dirname(__file__))
_treecorr.SetOMPThreads.restype = ctypes.c_int
_treecorr.SetOMPThreads.argtypes = [ ctypes.c_int ]

def set_omp_threads(num_threads, logger=None):
    """Set the number of OpenMP threads to use in the C++ layer.

    :param num_threads: The target number of threads to use
    :param logger:      If desired, a logger object for logging any warnings here. (default: None)

    :returns:           The  number of threads OpenMP reports that it will use.  Typically this
                        matches the input, but OpenMP reserves the right not to comply with
                        the requested number of threads.
    """
    input_num_threads = num_threads  # Save the input value.
    if num_threads is None or num_threads <= 0:
        import multiprocessing
        num_threads = multiprocessing.cpu_count()
        if logger:
            logger.debug('multiprocessing.cpu_count() = %d',num_threads)
    if num_threads > 1:
        if logger:
            logger.debug('Telling OpenMP to use %d threads',num_threads)
        num_threads = _treecorr.SetOMPThreads(num_threads)
        if logger:
            logger.debug('OpenMP reports that it will use %d threads',num_threads)
            if num_threads > 1:
                logger.info('Using %d threads.',num_threads)
            elif input_num_threads is not None and input_num_threads != 1:
                # Only warn if the user specifically asked for num_threads != 1.
                logger.warn('Unable to use multiple threads, since OpenMP is not enabled.')
    return num_threads



