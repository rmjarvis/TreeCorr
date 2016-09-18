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

# The version is stored in _version.py as recommended here:
# http://stackoverflow.com/questions/458550/standard-way-to-embed-version-into-python-package
from ._version import __version__, __version_info__

# Also let treecorr.version show the version.
version = __version__

import os,cffi,glob
# Set module level attributes for the include directory and the library file name.
include_dir = os.path.join(os.path.dirname(__file__),'include')

lib_file = os.path.join(os.path.dirname(__file__),'_treecorr.so')
# Some installation (e.g. Travis with python 3.x) name this e.g. _treecorr.cpython-34m.so,
# so if the normal name doesn't exist, look for this instead.
if not os.path.exists(lib_file):
    ext = ".cpython-%d%dm.so"%sys.version_info[:2]
    alt_file = os.path.join(os.path.dirname(__file__),'_treecorr' + ext)
    if os.path.exists(alt_file):
        lib_file = alt_file

# Load the C functions with cffi
_ffi = cffi.FFI()
for file_name in glob.glob(os.path.join(include_dir,'*_C.h')):
    _ffi.cdef(open(file_name).read())
_lib = _ffi.dlopen(lib_file)

from . import util
from .celestial import CelestialCoord, angle_units, arcsec, arcmin, degrees, hours, radians
from .config import read_config, set_omp_threads
from .catalog import Catalog, read_catalogs, calculateVarG, calculateVarK
from .binnedcorr2 import BinnedCorr2
from .ggcorrelation import GGCorrelation
from .nncorrelation import NNCorrelation
from .kkcorrelation import KKCorrelation
from .ngcorrelation import NGCorrelation
from .nkcorrelation import NKCorrelation
from .kgcorrelation import KGCorrelation
from .field import NField, KField, GField, NSimpleField, KSimpleField, GSimpleField
from .binnedcorr3 import BinnedCorr3
from .nnncorrelation import NNNCorrelation
from .kkkcorrelation import KKKCorrelation
from .gggcorrelation import GGGCorrelation
from .corr2 import corr2, print_corr2_params
from .corr3 import corr3, print_corr3_params

