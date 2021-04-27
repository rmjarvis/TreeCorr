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
.. module:: config
"""

from __future__ import print_function
import sys
import coord
import numpy as np
import warnings
import logging
import os


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
    if '#' in value:
        value = value.split('#')[0]
    value = value.strip()
    if value[0] in ['{','[','(']:
        if value[-1] not in ['}',']',')']:
            raise ValueError('List symbol %s not properly matched'%value[0])
        values = value[1:-1].split(',')
        values = [ vv.strip() for vv in values ]
    else:
        values = value.split() # on whitespace
    if len(values) == 1:
        config[key] = values[0]
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
                bool(int(value))
            except Exception:
                raise ValueError("Unable to parse %s as a bool."%value)
            else:
                return int(value)
    elif isinstance(value,(bool, np.bool_)):
        return value
    elif isinstance(value,int):
        # Note: integers aren't converted to bool, since brute distinguishes 1 vs 2 vs True.
        return value
    else:
        raise ValueError("Unable to parse %s as a bool."%value)

def parse_unit(value):
    """Parse the input value as a string that should be one of the valid angle units in
    coord.AngleUnit.valid_names.

    The value is allowed to merely start with one of the unit names.  So 'deg', 'degree',
    'degrees' all convert to 'deg' which is the name in coord.AngleUnit.valid_names.
    The return value in this case would be coord.AngleUnit.from_name('deg').value,
    which has the value pi/180.

    :param value:   The unit as a string value to parse.

    :returns:       The given unit in radians.
    """
    for unit in coord.AngleUnit.valid_names:
        if value.startswith(unit):
            return coord.AngleUnit.from_name(value).value
    raise ValueError("Unable to parse %s as an angle unit"%value)


def read_config(file_name, file_type='auto'):
    """Read a configuration dict from a file.

    :param file_name:   The file name from which the configuration dict should be read.
    :param file_type:   The type of config file.  Options are 'auto', 'yaml', 'json', 'params'.
                        (default: 'auto', which tries to determine the type from the extension)

    :returns:           A config dict built from the configuration file.
    """
    if file_type == 'auto':
        if file_name.endswith('.yaml'):
            file_type = 'yaml'
        elif file_name.endswith('.json'):
            file_type = 'json'
        elif file_name.endswith('.params'):
            file_type = 'params'
        else:
            raise ValueError("Unable to determine the type of config file from the extension")
    if file_type == 'yaml':
        return _read_yaml_file(file_name)
    elif file_type == 'json':
        return _read_json_file(file_name)
    elif file_type == 'params':
        return _read_params_file(file_name)
    else:
        raise ValueError("Invalid file_type %s"%file_type)

def _read_yaml_file(file_name):
    import yaml
    with open(file_name) as fin:
        config = yaml.safe_load(fin.read())
    return config

def _read_json_file(file_name):
    import json
    with open(file_name) as fin:
        config = json.load(fin)
    return config

def _read_params_file(file_name):
    config = dict()
    with open(file_name) as fin:
        for v in fin:
            v = v.strip()
            if len(v) == 0 or v[0] == '#':
                pass
            elif v[0] == '+':
                include_file_name = v[1:]
                config1 = read_config(include_file_name)
                config.update(config1)
            else:
                parse_variable(config,v)
    return config


def setup_logger(verbose, log_file=None):
    """Parse the integer verbosity level from the command line args into a logging_level string

    :param verbose:     An integer indicating what verbosity level to use.
    :param log_file:    If given, a file name to which to write the logging output.
                        If omitted or None, then output to stdout.

    :returns:           The logging.Logger object to use.
    """
    logging_levels = { 0: logging.CRITICAL,
                       1: logging.WARNING,
                       2: logging.INFO,
                       3: logging.DEBUG }
    logging_level = logging_levels[int(verbose)]

    # Setup logging to go to sys.stdout or (if requested) to an output file
    if log_file is None:
        name = 'treecorr'
    else:
        name = 'treecorr_' + log_file
    logger = logging.getLogger(name)

    if len(logger.handlers) == 0:  # only add handler once!
        if log_file is None:
            handle = logging.StreamHandler(stream=sys.stdout)
        else:
            handle = logging.FileHandler(log_file)
        formatter = logging.Formatter('%(message)s')  # Simple text output
        handle.setFormatter(formatter)
        logger.addHandler(handle)
    logger.setLevel(logging_level)
    return logger


def parse(value, value_type, name):
    """Parse the input value as the given type.

    :param value:       The value to parse.
    :param value_type:  The type expected for this.
    :param name:        The name of this value. Only used for error reporting.

    :returns: value
    """
    try:
        if value_type is bool:
            return parse_bool(value)
        elif value is None:
            return None
        else:
            return value_type(value)
    except ValueError:
        raise ValueError("Could not parse {}={} as type {}".format(name, value, value_type))


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

    :returns:       The updated config dict.
    """
    config = config.copy()
    for key in list(config.keys()):
        # Check if this is a deprecated alias
        if aliases and key in aliases:
            if logger:
                logger.warning("The parameter %s is deprecated.  You should use %s instead."%(
                               key, aliases[key]))
            else:
                warnings.warn("The parameter %s is deprecated.  You should use %s instead."%(
                              key, aliases[key]), FutureWarning)
            new_key = aliases[key]
            config[new_key] = config[key]
            del config[key]
            key = new_key

        # Check that this is a valid key
        if key not in params:
            raise TypeError("Invalid parameter %s."%key)

        value_type, may_be_list, default_value, valid_values = params[key][:4]

        # Get the value
        if may_be_list and isinstance(config[key], list):
            value = [parse(v, value_type, key) for v in config[key] ]
        else:
            value = parse(config[key], value_type, key)

        # If limited allowed value, check that this is one of them.
        if valid_values is not None:
            if value_type is str:
                matches = [ v for v in valid_values if value == v ]
                if len(matches) == 0:
                    # Allow the string to be longer.
                    # e.g. degrees is valid if 'deg' is in valid_values.
                    matches = [v for v in valid_values if isinstance(v,str) and value.startswith(v)]
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
        if key in config:
            continue
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
        value_type_str = value_type.__name__
        if may_be_list:
            print("                Type must be {0} or a list of {0}.".format(value_type_str))
        else:
            print("                Type must be {0}.".format(value_type_str))

        if valid_values is not None:
            print("                Valid values are {0!s}".format(valid_values))
        if default_value is not None:
            print("                Default value is {0!s}".format(default_value))
        print()


def convert(value, value_type, key):
    """Convert the given value to the given type.

    The ``key`` helps determine what kind of conversion should be performed.
    Specifically if 'unit' is in the ``key`` value, then a unit conversion is done.
    Otherwise, it just parses the ``value`` according to the ``value_type``.

    :param value:       The input value to be converted.  Usually a string.
    :param value_type:  The type to convert to.
    :param key:         The key for this value.  Only used to see if it includes 'unit'.

    :returns:           The converted value.
    """
    if 'unit' in key:
        return parse_unit(value)
    elif value_type == bool:
        return parse_bool(value)
    elif value is None:
        return None
    else:
        return value_type(value)

def get_from_list(config, key, num, value_type=str, default=None):
    """A helper function to get a key from config that is allowed to be a list

    Some of the config values are allowed to be lists of values, in which case we take the
    ``num`` item from the list.  If they are not a list, then the given value is used for
    all values of ``num``.

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
            if num >= len(values):
                raise IndexError("num=%d is out of range of list for %s"%(num,key))
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

def merge_config(config, kwargs, valid_params, aliases=None):
    """Merge in the values from kwargs into config.

    If either of these is None, then the other one is returned.
    If they are both dicts, then the values in kwargs take precedence over ones in config
    if there are any keys that are in both.  Also, the kwargs dict will be modified in this case.

    :param config:          The root config (will not be modified)
    :param kwargs:          A second dict with more or updated values
    :param valid_params:    A dict of valid parameters that are allowed for this usage.
                            The config dict is allowed to have extra items, but kwargs is not.
    :param aliases:         An optional dict of aliases. (default: None)

    :returns:               The merged dict, including only items that are in valid_params.
    """
    if kwargs is None:
        kwargs = {}
    if config:
        for key, value in config.items():
            if key in valid_params and key not in kwargs:
                kwargs[key] = value
    return check_config(kwargs, valid_params, aliases)
