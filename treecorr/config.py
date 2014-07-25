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


def parse_variable(config, v):
    if '=' not in v:
        raise ValueError('Improper variable specification.  Use field.item=value.')
    key, value = v.split('=',1)
    key = key.strip()
    value = value.strip()
    # Cut off any trailing comment
    if '#' in value: value = value.split('#')[0]
    if value[0] == '{':
        values = value[1:-1].split(',')
    else:
        values = value.split() # on whitespace
        if len(values) == 1:
            config[key] = value
        else:
            config[key] = values

def parse_bool(value):
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
                raise AttributeError("Unable to parse %s as a bool."%value)
    else:
        try:
            val = bool(value)
            return val
        except:
            raise AttributeError("Unable to parse %s as a bool."%value)


def read_config(file_name):
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

